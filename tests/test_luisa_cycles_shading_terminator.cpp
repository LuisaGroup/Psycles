#include "cycles_shading_terminator_fixture.h"
#include "cycles_svm_bsdf.h"
#include "cycles_svm_simple_closure.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace luisa::compute;
namespace f = psycles::test_support::shading_terminator;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;
constexpr unsigned count = unsigned(f::inputs.size()), input_rows = 6u;
constexpr unsigned observed_rows = f::bump_terms;
constexpr auto closure_mask = (svm::detail::ClosureTypeMask{1u} << closure::type_diffuse) |
                              (svm::detail::ClosureTypeMask{1u} << closure::type_translucent);
static_assert(svm::shader_data_use_bump_map_correction == f::bump_map_correction);

struct Oracle {
  std::array<luisa::float4, count * f::float_rows> values{};
  std::array<luisa::uint4, count * f::integer_rows> metadata{};
};

Oracle read_oracle(const char *path) {
  std::ifstream in{path};
  std::string magic, name;
  unsigned version{}, cases{}, floats{}, integers{};
  if (!(in >> magic >> version >> cases >> floats >> integers) ||
      magic != "cycles-shading-terminator" || version != f::format_version ||
      cases != count || floats != f::float_rows || integers != f::integer_rows) {
    throw std::runtime_error{"missing or incompatible original Cycles shading-terminator fixture"};
  }
  Oracle oracle;
  for (unsigned i = 0u; i < count; ++i) {
    unsigned id{};
    if (!(in >> id >> name) || id != i || name != f::names[i]) {
      throw std::runtime_error{"invalid original fixture case identity"};
    }
    for (unsigned row = 0u; row < f::float_rows; ++row) {
      for (unsigned lane = 0u; lane < 4u; ++lane) {
        auto &value = oracle.values[i * f::float_rows + row][lane];
        if (!(in >> value) || !std::isfinite(value)) {
          throw std::runtime_error{"truncated or nonfinite original float fixture"};
        }
      }
    }
    for (unsigned row = 0u; row < f::integer_rows; ++row) {
      for (unsigned lane = 0u; lane < 4u; ++lane) {
        if (!(in >> oracle.metadata[i * f::integer_rows + row][lane])) {
          throw std::runtime_error{"truncated original integer fixture"};
        }
      }
    }
  }
  if (in >> name) throw std::runtime_error{"unexpected trailing original fixture data"};
  return oracle;
}

class ObjectGlobals final : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const BufferFloat4 &_inputs;
  bool _enabled;
public:
  ObjectGlobals(const BufferFloat4 &inputs, bool enabled) noexcept
      : _inputs{inputs}, _enabled{enabled} {}
  [[nodiscard]] Float object_shadow_terminator_shading_offset(
      Expr<unsigned> object) const noexcept override {
    return _inputs.read(object * input_rows).w;
  }
  [[nodiscard]] bool has_shadow_terminator_shading_offset() const noexcept override { return _enabled; }
};

unsigned run(Device &device, Stream &stream, const Oracle &oracle, bool enabled) {
  std::array<luisa::float4, count * input_rows> inputs{};
  std::array<luisa::uint2, count> controls{};
  std::vector<unsigned> selected;
  const auto pack = [](f::Vector v, float w = 0.0f) { return luisa::float4{v[0], v[1], v[2], w}; };
  for (unsigned i = 0u; i < count; ++i) {
    const auto &v = f::inputs[i];
    inputs[i * input_rows] = pack(v.shader_normal, v.frequency);
    inputs[i * input_rows + 1u] = pack(v.geometric_normal);
    inputs[i * input_rows + 2u] = pack(v.closure_normal);
    inputs[i * input_rows + 3u] = pack(v.incoming);
    inputs[i * input_rows + 4u] = pack(v.evaluation_direction);
    inputs[i * input_rows + 5u] = pack(v.random);
    controls[i] = {v.flags, unsigned(v.kind)};
    // The disabled function-boundary projection only admits known identity
    // cases. The production scene-wide proof is tested independently.
    if (enabled || v.frequency == 1.0f) selected.push_back(i);
  }
  auto data = device.create_buffer<luisa::float4>(inputs.size());
  auto control = device.create_buffer<luisa::uint2>(controls.size());
  auto indices = device.create_buffer<unsigned>(selected.size());
  auto values = device.create_buffer<luisa::float4>(count * observed_rows);
  auto metadata = device.create_buffer<luisa::uint4>(count * f::integer_rows);
  Kernel1D kernel = [enabled](BufferFloat4 data, BufferUInt2 control, BufferUInt indices,
                             BufferFloat4 values, BufferUInt4 metadata) {
    const auto i = indices.read(dispatch_x());
    const auto input = data.read(i * input_rows);
    const auto flags = control.read(i);
    const auto identity = make_float4x4(1.0f);
    svm::ClosurePool pool{1u};
    svm::ShaderData sd{
        make_float3(0.0f), normalize(input.xyz()),
        normalize(data.read(i * input_rows + 1u).xyz()),
        normalize(data.read(i * input_rows + 3u).xyz()),
        svm::primitive_triangle, 0u, flags.x, 0u, 0u, 0.0f, 0.0f,
        i, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f),
        identity, identity, 0u, &pool};
    const auto normal = normalize(data.read(i * input_rows + 2u).xyz());
    $if(flags.y == unsigned(f::Kind::translucent)) {
      svm::detail::translucent_setup(sd, normal, make_float3(1.0f));
    } $else {
      svm::detail::diffuse_setup(sd, normal, make_float3(1.0f));
      $if(flags.y == unsigned(f::Kind::none)) { pool.set_type(0u, closure::type_none); };
    };
    const auto snapshot = [&] { return make_uint4(sd.flag, pool.count(), pool.left(), pool.common(0u).type); };
    const auto out = i * observed_rows;
    const auto meta = i * f::integer_rows;
    metadata.write(meta, snapshot());
    values.write(out + unsigned(f::shader_normal_frequency), make_float4(sd.N, input.w));
    values.write(out + unsigned(f::geometric_normal), make_float4(sd.Ng, 0.0f));
    values.write(out + unsigned(f::closure_normal_sample_weight),
                 make_float4(pool.common(0u).N, pool.common(0u).sample_weight));
    ObjectGlobals kg{data, enabled};
    const auto sampled = svm::detail::bsdf_sample(
        kg, sd, 0u, data.read(i * input_rows + 5u).xyz(), closure_mask);
    const auto wo = normalize(data.read(i * input_rows + 4u).xyz());
    const auto evaluated = svm::detail::bsdf_eval(kg, sd, 0u, wo, closure_mask);
    values.write(out + unsigned(f::sampled_value_pdf), make_float4(sampled.value, sampled.pdf));
    values.write(out + unsigned(f::sampled_direction_eta), make_float4(sampled.wo, sampled.eta));
    values.write(out + unsigned(f::sampled_roughness), make_float4(sampled.sampled_roughness, 0.0f, 0.0f));
    values.write(out + unsigned(f::evaluated_value_pdf), make_float4(evaluated.value, evaluated.pdf));
    values.write(out + unsigned(f::evaluation_direction), make_float4(wo, 0.0f));
    metadata.write(meta + 1u, snapshot());
    metadata.write(meta + 2u, make_uint4(sampled.label, sd.object, sd.type, flags.y));
  };
  const auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, count * observed_rows> actual{};
  std::array<luisa::uint4, count * f::integer_rows> meta{};
  stream << data.copy_from(inputs.data()) << control.copy_from(controls.data())
         << indices.copy_from(selected.data()) << values.copy_from(actual.data())
         << metadata.copy_from(meta.data())
         << shader(data, control, indices, values, metadata).dispatch(unsigned(selected.size()))
         << values.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
  unsigned mismatches = 0u;
  for (const auto i : selected) {
    for (unsigned row = 0u; row < observed_rows; ++row) {
      for (unsigned lane = 0u; lane < 4u; ++lane) {
        const auto expected = oracle.values[i * f::float_rows + row][lane];
        const auto value = actual[i * observed_rows + row][lane];
        if (!std::isfinite(value) || std::abs(value - expected) > 2e-6f + 2e-6f * std::abs(expected)) {
          std::cerr << f::names[i] << " row=" << row << " lane=" << lane
                    << " actual=" << value << " original=" << expected << '\n';
          ++mismatches;
        }
      }
    }
    for (unsigned row = 0u; row < f::integer_rows; ++row) {
      for (unsigned lane = 0u; lane < 4u; ++lane) {
        if (meta[i * f::integer_rows + row][lane] != oracle.metadata[i * f::integer_rows + row][lane]) {
          std::cerr << f::names[i] << " metadata row=" << row << " lane=" << lane << '\n';
          ++mismatches;
        }
      }
    }
  }
  // The oracle's final bump_terms row is diagnostic: parsed and finite-checked
  // above, not recreated here. Its actual influence is checked through the
  // original sampled/evaluated BSDF rows, including the disabled bump case.
  std::cout << "shading terminator capability=" << enabled << " original cases="
            << selected.size() << " mismatches=" << mismatches << '\n';
  return mismatches;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) return 2; // backend, optional original fixture path
  try {
    const auto oracle = read_oracle(argc == 3 ? argv[2] : PSYCLES_SHADING_TERMINATOR_ORACLE);
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();
    const auto enabled = run(device, stream, oracle, true);
    const auto disabled = run(device, stream, oracle, false);
    return enabled + disabled == 0u ? 0 : 1;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
