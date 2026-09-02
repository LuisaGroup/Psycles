#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_principled_hair_chiang.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t scenario_count = 4u;
inline constexpr std::uint32_t values_per_scenario = 5u;

[[nodiscard]] device_svm::ShaderData make_shader_data() noexcept {
  const auto identity = make_float4x4(1.0f);
  const auto incoming = normalize(make_float3(0.32f, 0.88f, 0.35f));
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          incoming,
          0u,
          0u,
          0u,
          0u,
          0u,
          0.0f,
          0.0f,
          0u,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity,
          0u,
          nullptr};
}

[[nodiscard]] device_svm::ChiangHairClosure make_closure() noexcept {
  const auto X = make_float3(1.0f, 0.0f, 0.0f);
  const auto incoming = normalize(make_float3(0.32f, 0.88f, 0.35f));
  return {.common = {.weight = make_float3(0.8f, 0.4f, 0.2f),
                     .type = closure::type_hair_chiang,
                     .sample_weight = 0.466666698f,
                     .N = normalize(cross(X, incoming))},
          .param = {.sigma = make_float3(0.12f, 0.34f, 0.56f),
                    .v = 0.36f,
                    .s = 0.31f,
                    .alpha = -0.23f,
                    .eta = 1.42f,
                    .m0_roughness = 0.07f,
                    .h = 0.23f}};
}

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt labels) noexcept {
        const UInt scenario = dispatch_id().x;
        Float3 random;
        $switch(scenario) {
          $case(0u) { random = make_float3(0.12f, 0.27f, 0.01f); };
          $case(1u) { random = make_float3(0.48f, 0.61f, 0.30f); };
          $case(2u) { random = make_float3(0.86f, 0.13f, 0.99f); };
          $default { random = make_float3(0.37f, 0.92f, 0.9999f); };
        };

        psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
        const auto shader_data = make_shader_data();
        auto closure_value = make_closure();
        const auto sample = detail::bsdf_hair_chiang_sample(
            kernel_globals, closure_value, shader_data, random);
        const auto albedo =
            detail::bsdf_hair_chiang_albedo(closure_value, shader_data);
        const auto eval_wo = normalize(make_float3(-0.41f, 0.37f, 0.83f));
        const auto evaluation = detail::bsdf_hair_chiang_eval(
            kernel_globals, closure_value, shader_data, eval_wo);
        detail::bsdf_hair_chiang_blur(closure_value, 0.73f);

        const auto base = scenario * values_per_scenario;
        output.write(base + 0u, make_float4(sample.wo, sample.pdf));
        output.write(base + 1u,
                     make_float4(sample.value, sample.sampled_roughness.x));
        output.write(base + 2u,
                     make_float4(albedo, sample.sampled_roughness.y));
        output.write(base + 3u, make_float4(evaluation.value, evaluation.pdf));
        output.write(base + 4u,
                     make_float4(closure_value.param.v, closure_value.param.s,
                                 closure_value.param.m0_roughness, sample.eta));
        labels.write(scenario, sample.label);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 8.0e-4f) noexcept {
  return std::isfinite(actual) && std::abs(actual - expected) <=
                                      2.0e-6f + tolerance * std::abs(expected);
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * values_per_scenario);
  auto labels = device.create_buffer<std::uint32_t>(scenario_count);
  std::array<luisa::float4, scenario_count * values_per_scenario> actual{};
  std::array<std::uint32_t, scenario_count> actual_labels{};
  const auto shader = device.compile(
      scattering_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  stream << shader(output, labels).dispatch(scenario_count)
         << output.copy_to(actual.data())
         << labels.copy_to(actual_labels.data()) << synchronize();

  constexpr std::array<luisa::float4, scenario_count * values_per_scenario>
      expected{{
          {-0.359477669f, 0.711803496f, -0.603416622f, 0.0129264155f},
          {0.007940121f, 0.0070266095f, 0.00650947355f, 0.0700000003f},
          {0.163960189f, 0.0642211512f, 0.0434151776f, 0.0700000003f},
          {0.00540719647f, 0.00332128746f, 0.00235192734f, 0.00664859079f},
          {0.730000019f, 0.730000019f, 0.730000019f, 1.0f},
          {-0.448328823f, -0.874987483f, -0.182751775f, 0.504046798f},
          {0.400137663f, 0.256152838f, 0.163986102f, 0.0700000003f},
          {0.163960189f, 0.0642211512f, 0.0434151776f, 0.0700000003f},
          {0.00540719647f, 0.00332128746f, 0.00235192734f, 0.00664859079f},
          {0.730000019f, 0.730000019f, 0.730000019f, 1.0f},
          {0.648913324f, 0.338528842f, 0.681402862f, 0.0041070357f},
          {0.0043841796f, 0.00181188353f, 0.000753900327f, 0.0700000003f},
          {0.163960189f, 0.0642211512f, 0.0434151776f, 0.0700000003f},
          {0.00540719647f, 0.00332128746f, 0.00235192734f, 0.00664859079f},
          {0.730000019f, 0.730000019f, 0.730000019f, 1.0f},
          {0.465729803f, -0.801290572f, 0.375538468f, 0.0646627769f},
          {0.0513524003f, 0.0328561626f, 0.0210281424f, 0.0700000003f},
          {0.163960189f, 0.0642211512f, 0.0434151776f, 0.0700000003f},
          {0.00540719647f, 0.00332128746f, 0.00235192734f, 0.00664859079f},
          {0.730000019f, 0.730000019f, 0.730000019f, 1.0f},
      }};
  constexpr std::array<std::uint32_t, scenario_count> expected_labels{{
      closure::label_glossy | closure::label_reflect,
      closure::label_glossy | closure::label_transmit,
      closure::label_glossy | closure::label_transmit,
      closure::label_glossy | closure::label_transmit,
  }};

  auto valid = true;
  for (auto index = std::size_t{0u}; index < actual.size(); ++index) {
    valid &= near(actual[index], expected[index]);
  }
  valid &= actual_labels == expected_labels;
  if (!valid) {
    std::cerr << "Cycles Principled Hair Chiang scattering mismatch on "
              << backend << '\n';
    for (auto index = std::size_t{0u}; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << "value[" << index << "] = {" << value.x << ", " << value.y
                << ", " << value.z << ", " << value.w << "}\n";
    }
    for (auto index = std::size_t{0u}; index < actual_labels.size(); ++index) {
      std::cerr << "label[" << index << "] = " << actual_labels[index] << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
