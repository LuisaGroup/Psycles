#include "cycles_svm_internal.h"
#include "cycles_svm_voronoi_octave_test_data.h"
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace luisa::compute;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm::detail;
namespace fixture = psycles::test_support::voronoi_octave;
using Row = std::array<float, fixture::output_count>;

auto read_oracle() {
  std::ifstream stream{PSYCLES_VORONOI_OCTAVE_ORACLE};
  std::array<Row, fixture::case_count * 2u> result;
  for (unsigned i = 0; i < result.size(); ++i) {
    unsigned ordinal;
    if (!(stream >> ordinal) || ordinal != i) { throw std::runtime_error{"invalid oracle ordinal"}; }
    for (auto &value : result[i]) {
      if (!(stream >> value) || !std::isfinite(value)) { throw std::runtime_error{"invalid oracle value"}; }
    }
  }
  std::string trailing;
  if (stream >> trailing) { throw std::runtime_error{"extra oracle data"}; }
  return result;
}
} // namespace

int main(int argc, char **argv) {
  const std::string_view backend{argc > 1 ? argv[1] : "hip"};
  const auto oracle = read_oracle();
  const auto cases = fixture::cases<abi::SVMNodeTexVoronoi>();
  std::vector<std::uint32_t> words;
  std::vector<luisa::float4> inputs;
  for (const auto &test : cases) {
    const auto encoded = std::bit_cast<std::array<std::uint32_t, 13u>>(test.node);
    words.insert(words.end(), encoded.begin(), encoded.end());
    for (unsigned part = 0; part < 2u; ++part) {
      const auto data = test.inputs.data() + part * 4u;
      inputs.emplace_back(data[0], data[1], data[2], data[3]);
    }
  }
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto device_words = device.create_buffer<std::uint32_t>(words.size());
  auto device_inputs = device.create_buffer<luisa::float4>(inputs.size());
  auto device_output = device.create_buffer<float>(fixture::case_count * fixture::output_count);
  auto device_offsets = device.create_buffer<std::uint32_t>(fixture::case_count);
  stream << device_words.copy_from(words.data()) << device_inputs.copy_from(inputs.data());
  unsigned passed = 0u;
  for (unsigned extra = 0; extra < 2u; ++extra) {
    Kernel1D kernel = [&](BufferUInt words, BufferFloat4 inputs, BufferFloat output, BufferUInt offsets) {
      const auto index = dispatch_x();
      svm::Stack stack{fixture::stack_size};
      for (unsigned slot = 0; slot < 8u; ++slot) {
        stack[slot] = inputs.read(index * 2u + slot / 4u)[slot % 4u];
      }
      UInt pc = index * 13u;
      svm::Cursor cursor{words, pc};
      svm::node_tex_voronoi(cursor, stack, extra != 0u);
      for (unsigned slot = 0; slot < fixture::output_count; ++slot) {
        output.write(index * fixture::output_count + slot, stack[fixture::output_begin + slot]);
      }
      offsets.write(index, pc - index * 13u);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::array<float, fixture::case_count * fixture::output_count> output;
    std::array<std::uint32_t, fixture::case_count> offsets;
    stream << shader(device_words, device_inputs, device_output, device_offsets).dispatch(fixture::case_count)
           << device_output.copy_to(output.data()) << device_offsets.copy_to(offsets.data()) << synchronize();
    for (unsigned i = 0; i < fixture::case_count; ++i) {
      auto pass = offsets[i] == 13u;
      const auto ordinal = extra * fixture::case_count + i;
      for (unsigned slot = 0; slot < fixture::output_count; ++slot) {
        const auto actual = output[i * fixture::output_count + slot];
        const auto expected = oracle[ordinal][slot];
        const auto matches = std::isfinite(actual) &&
            std::abs(actual - expected) <= 2e-5f * std::max(1.0f, std::abs(expected));
        if (!matches) {
          std::cerr << "case=" << ordinal << " slot=" << slot
                    << " actual=" << actual << " original=" << expected << '\n';
        }
        pass &= matches;
      }
      if (!pass) { std::cerr << backend << " Voronoi octave mismatch: " << ordinal << '\n'; }
      passed += pass;
    }
  }
  std::cout << "Original HIP Voronoi octave states: " << passed << "/" << oracle.size() << '\n';
  return passed == oracle.size() ? 0 : 1;
}
