#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_principled_hair_huang.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace tables = psycles::luisa_backend::cycles45_tables;

inline constexpr std::uint32_t scenario_count = 8u;
inline constexpr std::uint32_t values_per_scenario = 5u;
inline constexpr std::uint32_t meta_per_scenario = 3u;

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat &_table;

public:
  explicit TableKernelGlobals(const BufferFloat &table) noexcept
      : _table{table} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }
};

[[nodiscard]] device_svm::ShaderData make_shader_data(UInt seed) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          0u,
          0u,
          0u,
          0u,
          0u,
          0.0f,
          0.0f,
          0u,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(0.0f),
          make_float3(0.0f),
          identity,
          identity,
          seed,
          nullptr};
}

[[nodiscard]] device_svm::HuangHairClosure
make_closure(Float pixel_coverage) noexcept {
  return {
      .common = {.weight = make_float3(0.8f, 0.4f, 0.2f),
                 .type = closure::type_hair_huang,
                 .sample_weight = 0.466666698f,
                 .N = make_float3(0.268715411f, -0.329966754f, 0.904938698f)},
      .param = {.sigma = make_float3(0.12f, 0.34f, 0.56f),
                .roughness = 0.58f,
                .tilt = 0.23f,
                .eta = 1.42f,
                .aspect_ratio = 0.5f,
                .h = 0.0675399899f},
      .extra = {.R = 0.3f,
                .TT = 0.6f,
                .TRT = 0.9f,
                .Y = make_float3(0.549442232f, 0.824163318f, 0.137360558f),
                .Z = make_float3(-0.791141689f, 0.460300654f, 0.402763069f),
                .wi = make_float3(0.985347748f, 0.170216486f, 0.0107813049f),
                .radius = 0.500089765f,
                .e2 = 0.75f,
                .pixel_coverage = pixel_coverage}};
}

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<float>, Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat table, BufferFloat4 output, BufferUInt meta) noexcept {
        const UInt scenario = dispatch_id().x;
        Float3 random;
        Float pixel_coverage;
        UInt seed;
        Float R = 0.3f;
        Float TT = 0.6f;
        Float TRT = 0.9f;
        $switch(scenario) {
          $case(0u) {
            random = make_float3(0.12f, 0.27f, 0.83f);
            pixel_coverage = 0.2f;
            seed = 0x12345678u;
          };
          $case(1u) {
            random = make_float3(0.48f, 0.61f, 0.19f);
            pixel_coverage = 1.0f;
            seed = 0x10203040u;
          };
          $case(2u) {
            random = make_float3(0.86f, 0.13f, 0.54f);
            pixel_coverage = 0.2f;
            seed = 0x89abcdefu;
          };
          $case(3u) {
            random = make_float3(0.37f, 0.92f, 0.41f);
            pixel_coverage = 1.0f;
            seed = 0x31415926u;
          };
          $case(4u) {
            random = make_float3(0.0f, 0.27f, 0.83f);
            pixel_coverage = 0.2f;
            seed = 0x12345678u;
            R = 1.0f;
            TT = 0.0f;
            TRT = 0.0f;
          };
          $case(5u) {
            random = make_float3(0.0f, 0.27f, 0.83f);
            pixel_coverage = 0.2f;
            seed = 0x12345678u;
            R = 0.0f;
            TT = 1.0f;
            TRT = 0.0f;
          };
          $case(6u) {
            random = make_float3(0.0f, 0.27f, 0.83f);
            pixel_coverage = 0.2f;
            seed = 0x12345678u;
            R = 0.0f;
            TT = 0.0f;
            TRT = 1.0f;
          };
          $default {
            random = make_float3(0.999999f, 0.27f, 0.83f);
            pixel_coverage = 0.2f;
            seed = 0x12345678u;
            R = 0.0f;
            TT = 0.0f;
            TRT = 1.0f;
          };
        };

        const TableKernelGlobals kernel_globals{table};
        auto shader_data = make_shader_data(seed);
        auto closure_value = make_closure(pixel_coverage);
        closure_value.extra.R = R;
        closure_value.extra.TT = TT;
        closure_value.extra.TRT = TRT;
        const auto sample = detail::bsdf_hair_huang_sample(
            kernel_globals, closure_value, shader_data, random);
        const auto albedo = detail::bsdf_hair_huang_albedo(closure_value);
        auto eval_shader_data = make_shader_data(seed ^ 0xa5a5a5a5u);
        const auto eval_wo = normalize(make_float3(-0.41f, 0.37f, 0.83f));
        const auto evaluation = detail::bsdf_hair_huang_eval(
            kernel_globals, closure_value, eval_shader_data, eval_wo);
        detail::bsdf_hair_huang_blur(closure_value, 0.73f);

        const auto base = scenario * values_per_scenario;
        output.write(base + 0u, make_float4(sample.wo, sample.pdf));
        output.write(base + 1u,
                     make_float4(sample.value, sample.sampled_roughness.x));
        output.write(base + 2u,
                     make_float4(albedo, sample.sampled_roughness.y));
        output.write(base + 3u, make_float4(evaluation.value, evaluation.pdf));
        output.write(base + 4u, make_float4(closure_value.param.roughness, 0.0f,
                                            0.0f, 0.0f));
        meta.write(scenario * meta_per_scenario + 0u, sample.label);
        meta.write(scenario * meta_per_scenario + 1u, shader_data.lcg_state);
        meta.write(scenario * meta_per_scenario + 2u,
                   eval_shader_data.lcg_state);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 4.0e-4f) noexcept {
  return std::isfinite(actual) && std::abs(actual - expected) <=
                                      2.0e-6f + tolerance * std::abs(expected);
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 4.0e-4f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto table = device.create_buffer<float>(tables::total_size);
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * values_per_scenario);
  auto meta =
      device.create_buffer<std::uint32_t>(scenario_count * meta_per_scenario);
  std::vector<float> table_values(tables::total_size, 1.0f);
  std::array<luisa::float4, scenario_count * values_per_scenario> actual{};
  std::array<std::uint32_t, scenario_count * meta_per_scenario> actual_meta{};
  const auto shader = device.compile(
      scattering_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  stream << table.copy_from(table_values.data())
         << shader(table, output, meta).dispatch(scenario_count)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  constexpr std::array<luisa::float4, scenario_count * values_per_scenario>
      expected{{
          {-0.541559517f, 0.178876013f, -0.821411371f, 1.0f},
          {0.344760478f, 0.246497661f, 0.176241517f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.00168190396f, 0.00163386040f, 0.00162157335f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {-0.469784856f, 0.229895025f, -0.852320731f, 1.0f},
          {0.452219099f, 0.293335885f, 0.190274909f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.00134713494f, 0.00119303423f, 0.00111957872f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {-0.247639582f, -0.545992494f, -0.800354242f, 1.0f},
          {0.280607849f, 0.194291607f, 0.134526640f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.00163555471f, 0.00162164215f, 0.00161830103f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {-0.730401158f, 0.413751453f, 0.543436706f, 1.0f},
          {0.0105374288f, 0.0105374288f, 0.0105374288f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.00471405406f, 0.00268683350f, 0.00179131783f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {0.979526758f, 0.113557979f, 0.166228354f, 1.0f},
          {0.000509787060f, 0.000186767284f, 0.0000685802806f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.00539053092f, 0.00539053092f, 0.00539053092f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {-0.541559517f, 0.178876013f, -0.821411371f, 1.0f},
          {0.554845750f, 0.396704972f, 0.283637106f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.0f, 0.0f, 0.0f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {0.673414469f, -0.181047395f, 0.716751635f, 1.0f},
          {0.0167134143f, 0.00860256702f, 0.00442782929f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.0000647446504f, 0.0000167011804f, 0.00000441399288f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
          {0.979526758f, 0.113557979f, 0.166228354f, 1.0f},
          {0.0198174398f, 0.00726038264f, 0.00266598677f, 0.579999983f},
          {0.875521421f, 0.768943369f, 0.718641162f, 0.579999983f},
          {0.0000647446504f, 0.0000167011804f, 0.00000441399288f, 1.0f},
          {0.730000019f, 0.0f, 0.0f, 0.0f},
      }};
  constexpr std::array<std::uint32_t, scenario_count * meta_per_scenario>
      expected_meta{{
          closure::label_glossy | closure::label_reflect,
          1791458448u,
          660398609u,
          closure::label_glossy | closure::label_reflect,
          1384892248u,
          3033115859u,
          closure::label_glossy | closure::label_reflect,
          366272744u,
          582654702u,
          closure::label_glossy | closure::label_reflect,
          506823939u,
          2572924615u,
          closure::label_glossy | closure::label_reflect,
          1791458448u,
          3079795677u,
          closure::label_glossy | closure::label_reflect,
          1791458448u,
          2715904383u,
          closure::label_glossy | closure::label_reflect,
          1791458448u,
          660398609u,
          closure::label_glossy | closure::label_reflect,
          1791458448u,
          660398609u,
      }};

  auto valid = true;
  for (auto index = std::size_t{0u}; index < actual.size(); ++index) {
    valid &= near(actual[index], expected[index]);
  }
  valid &= actual_meta == expected_meta;
  if (!valid) {
    std::cerr << "Cycles Principled Hair Huang scattering mismatch on "
              << backend << '\n';
    for (auto index = std::size_t{0u}; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << "value[" << index << "] = {" << value.x << ", " << value.y
                << ", " << value.z << ", " << value.w << "}\n";
    }
    for (auto index = std::size_t{0u}; index < actual_meta.size(); ++index) {
      std::cerr << "meta[" << index << "] = " << actual_meta[index] << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
