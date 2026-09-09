#include "cycles_holdout_state_fixture.h"
#include "cycles_svm_internal.h"
#include "cycles_svm_surface_shader.h"

#include <luisa/luisa-compute.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace luisa::compute;
namespace f = psycles::test_support::holdout_state;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace native = svm::detail;
namespace abi = psycles::compiler::cycles_svm;
constexpr auto count = unsigned(f::inputs.size());
constexpr unsigned input_float_rows = 7, input_integer_rows = 3;
static_assert(abi::SD_BACKFACING == 1 && abi::SD_TRANSPARENT == (1u << 9));
static_assert(abi::SD_MIS_FRONT == (1u << 16) && abi::SD_HAS_ONLY_VOLUME == (1u << 19));
static_assert(abi::SD_OBJECT_HOLDOUT_MASK == 1 && abi::SVM_STACK_INVALID == 255);
static_assert((abi::SD_CLOSURE_FLAGS | abi::SD_TRANSPARENT | abi::SD_BACKFACING) == 0x1fff);

svm::ShaderData make_shader_data(svm::ClosurePool &pool) {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, 1.0f),
          svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.0f, 0.0f,
          0u, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f),
          identity, identity, 0u, &pool};
}

void snapshot(svm::ShaderData &sd, Float3 holdout,
              const BufferFloat4 &output, const BufferUInt4 &metadata,
              UInt index, unsigned stage) {
  const auto out = index * f::float_rows + stage * f::snapshot_float_rows;
  const auto meta = index * f::integer_rows + stage * f::snapshot_integer_rows;
  UInt4 types = make_uint4(0u);
  for (unsigned c = 0; c < f::capacity; ++c) {
    Float4 value = make_float4(0.0f);
    Float4 normal = make_float4(0.0f);
    $if(c < sd.closure->count()) {
      const auto sc = sd.closure->common(c);
      // Slots were seeded before reset; these fields are defined and must
      // remain untouched by native closure_alloc and apply_holdout.
      value = make_float4(sc.weight, sc.sample_weight);
      normal = make_float4(sc.N, 0.0f);
      types[c] = sc.type;
    };
    output.write(out + c, value);
    output.write(out + c + 4u, normal);
  }
  output.write(out + 8u, make_float4(sd.closure_transparent_extinction, 0.0f));
  output.write(out + 9u, make_float4(sd.closure_emission_background, 0.0f));
  output.write(out + 10u, make_float4(holdout, 0.0f));
  metadata.write(meta, make_uint4(sd.flag, sd.closure->count(), sd.closure->left(), sd.object_flag));
  metadata.write(meta + 1u, types);
}

bool run(char **argv) {
  std::array<luisa::float4, count * input_float_rows> input{};
  std::array<luisa::uint4, count * input_integer_rows> control{};
  std::array<unsigned, count> payload{};
  constexpr std::array<unsigned, 8> types{
      abi::CLOSURE_HOLDOUT_ID, abi::CLOSURE_BSDF_DIFFUSE_ID,
      abi::CLOSURE_BSDF_TRANSPARENT_ID, abi::CLOSURE_BSSRDF_BURLEY_ID,
      abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, abi::CLOSURE_BSDF_RAY_PORTAL_ID,
      abi::CLOSURE_NONE_ID, abi::NBUILTIN_CLOSURES};
  for (unsigned i = 0; i < count; ++i) {
    const auto &in = f::inputs[i];
    for (unsigned c = 0; c < f::capacity; ++c) {
      const auto &sc = in.closures[c];
      input[i * input_float_rows + c] = {sc.weight[0], sc.weight[1], sc.weight[2], sc.sample_weight};
      control[i * input_integer_rows + 2][c] = types.at(unsigned(sc.kind));
    }
    input[i * input_float_rows + 4] = {in.extinction[0], in.extinction[1], in.extinction[2], in.mix};
    input[i * input_float_rows + 5] = {in.emission[0], in.emission[1], in.emission[2], 0.0f};
    input[i * input_float_rows + 6] = {in.node_weight[0], in.node_weight[1], in.node_weight[2], 0.0f};
    control[i * input_integer_rows] = {in.count, in.left, in.flags, in.object_flags};
    control[i * input_integer_rows + 1] = {in.node_enabled, in.mix_offset, 0u, 0u};
    payload[i] = in.mix_offset; // Exact typed holdout payload: byte offset, 3 padding bytes.
  }

  Context context{argv[0]};
  auto device = context.create_device(argv[1]);
  auto stream = device.create_stream();
  auto inputs = device.create_buffer<luisa::float4>(input.size());
  auto controls = device.create_buffer<luisa::uint4>(control.size());
  auto words = device.create_buffer<unsigned>(payload.size());
  auto output = device.create_buffer<luisa::float4>(count * f::float_rows);
  auto metadata = device.create_buffer<luisa::uint4>(count * f::integer_rows);
  auto cursor_output = device.create_buffer<unsigned>(count);
  Kernel1D kernel = [](BufferFloat4 inputs, BufferUInt4 controls, BufferUInt words,
                       BufferFloat4 output, BufferUInt4 metadata, BufferUInt cursors) {
    const auto i = dispatch_x();
    const auto flags = controls.read(i * input_integer_rows);
    const auto operation = controls.read(i * input_integer_rows + 1u);
    const auto types = controls.read(i * input_integer_rows + 2u);
    svm::ClosurePool pool{f::capacity};
    // Seed physical storage, then start the native logical lifetime without
    // clearing it. This catches writes outside closure_alloc's native set.
    for (unsigned slot = 0; slot < f::capacity; ++slot) {
      const auto value = inputs.read(i * input_float_rows + slot);
      const auto allocated = pool.allocate(types[slot], value.xyz());
      pool.set_sample_weight(allocated.index, value.w);
      pool.set_normal(allocated.index, make_float3(f::normal_sentinel[0],
          f::normal_sentinel[1], f::normal_sentinel[2]));
    }
    pool.reset();
    UInt c = 0u;
    $while(c < flags.x) {
      const auto value = inputs.read(i * input_float_rows + c);
      const auto allocated = pool.allocate(types[c], value.xyz());
      pool.set_sample_weight(allocated.index, value.w);
      c += 1u;
    };
    pool.set_left(flags.y);
    auto sd = make_shader_data(pool);
    sd.flag = flags.z;
    sd.object_flag = flags.w;
    const auto extinction = inputs.read(i * input_float_rows + 4u);
    sd.closure_transparent_extinction = extinction.xyz();
    sd.closure_emission_background = inputs.read(i * input_float_rows + 5u).xyz();
    snapshot(sd, make_float3(0.0f), output, metadata, i, 0u);
    UInt offset = i;
    $if(operation.x != 0u) {
      native::Stack stack;
      $if(operation.y != unsigned(abi::SVM_STACK_INVALID)) { stack[operation.y] = extinction.w; };
      native::Cursor cursor{words, offset};
      native::node_closure_holdout(cursor, stack,
          inputs.read(i * input_float_rows + 6u).xyz(), sd);
    };
    snapshot(sd, make_float3(0.0f), output, metadata, i, 1u);
    const auto holdout = native::surface_shader_apply_holdout(sd);
    snapshot(sd, holdout, output, metadata, i, 2u);
    cursors.write(i, offset);
  };
  const auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, count * f::float_rows> actual{};
  std::array<luisa::uint4, count * f::integer_rows> meta{};
  std::array<unsigned, count> cursors{};
  stream << inputs.copy_from(input.data()) << controls.copy_from(control.data())
         << words.copy_from(payload.data())
         << shader(inputs, controls, words, output, metadata, cursor_output).dispatch(count)
         << output.copy_to(actual.data()) << metadata.copy_to(meta.data())
         << cursor_output.copy_to(cursors.data()) << synchronize();

  std::ifstream oracle{PSYCLES_HOLDOUT_STATE_ORACLE};
  if (!oracle) throw std::runtime_error{"missing original Cycles holdout state"};
  unsigned mismatches = 0;
  for (unsigned i = 0; i < count; ++i) {
    unsigned index;
    if (!(oracle >> index) || index != i) throw std::runtime_error{"invalid oracle index"};
    for (unsigned lane = 0; lane < 4 * f::float_rows; ++lane) {
      float expected;
      if (!(oracle >> expected)) throw std::runtime_error{"truncated float oracle"};
      const auto value = actual[i * f::float_rows + lane / 4][lane % 4];
      if (!std::isfinite(value) || std::abs(value - expected) > 2e-6f + 2e-6f * std::abs(expected)) {
        std::cerr << f::names[i] << " float " << lane << ": " << value << " != " << expected << '\n';
        ++mismatches;
      }
    }
    for (unsigned lane = 0; lane < 4 * f::integer_rows; ++lane) {
      unsigned expected;
      if (!(oracle >> expected)) throw std::runtime_error{"truncated integer oracle"};
      const auto value = meta[i * f::integer_rows + lane / 4][lane % 4];
      if (value != expected) {
        std::cerr << f::names[i] << " uint " << lane << ": " << value << " != " << expected << '\n';
        ++mismatches;
      }
    }
    // This is a typed-payload size invariant, not a fabricated expected SVM
    // stream: the original SVMNodeClosureHoldout occupies exactly one word.
    const auto expected_offset = i + f::inputs[i].node_enabled * sizeof(abi::SVMNodeClosureHoldout) / sizeof(unsigned);
    if (cursors[i] != expected_offset) {
      std::cerr << f::names[i] << " cursor " << cursors[i] << " != " << expected_offset << '\n';
      ++mismatches;
    }
  }
  std::string extra;
  if (oracle >> extra) throw std::runtime_error{"unexpected trailing oracle data"};
  std::cout << "original Cycles holdout state: " << count << " cases, " << f::snapshots
            << " snapshots, " << mismatches << " mismatches\n";
  return mismatches == 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try { return run(argv) ? 0 : 1; }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
