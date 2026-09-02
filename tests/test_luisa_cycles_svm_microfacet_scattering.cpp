#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_microfacet_scattering.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
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
namespace mapping = psycles::luisa_backend::cycles_sample_mapping;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t output_count = 42u;
inline constexpr std::uint32_t label_count = 13u;

[[nodiscard]] device_svm::FresnelGeneralizedSchlick empty_fresnel() noexcept {
  return {.thin_film = {.thickness = 0.0f, .ior = 1.0f},
          .reflection_tint = make_float3(0.0f),
          .transmission_tint = make_float3(0.0f),
          .f0 = make_float3(0.0f),
          .f90 = make_float3(0.0f),
          .exponent = 0.0f};
}

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt labels) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        const auto geometric_normal = normal;
        const auto incoming =
            normalize(make_float3(0.35f, -0.15f, 0.925f));
        const auto direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
        const auto tangent = normalize(make_float3(0.9f, 0.2f, -0.1f));
        const auto basis =
            mapping::make_orthonormals_tangent(normal, tangent);

        const auto record_sample = [&](std::uint32_t value_index,
                                       std::uint32_t label_index,
                                       const detail::BsdfSample &sample) {
          output.write(value_index,
                       make_float4(sample.wo, sample.pdf));
          output.write(value_index + 1u,
                       make_float4(sample.value, 0.0f));
          output.write(value_index + 2u,
                       make_float4(sample.sampled_roughness.x,
                                   sample.sampled_roughness.y,
                                   sample.eta, 0.0f));
          labels.write(label_index, sample.label);
        };

        const device_svm::MicrofacetClosure ggx{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_ggx,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.22f,
                      .alpha_y = 0.61f,
                      .ior = 1.45f,
                      .energy_scale = 1.17f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto ggx_sample = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, ggx, geometric_normal, incoming,
            make_float3(0.37f, 0.73f, 0.42f));
        record_sample(0u, 0u, ggx_sample);
        const auto ggx_eval = detail::bsdf_microfacet_ggx_eval(
            kernel_globals, ggx, incoming, direct);
        output.write(3u, make_float4(ggx_eval.value, ggx_eval.pdf));

        const device_svm::MicrofacetClosure beckmann{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_beckmann,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.34f,
                      .alpha_y = 0.58f,
                      .ior = 1.45f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto beckmann_sample = detail::bsdf_microfacet_beckmann_sample(
            kernel_globals, beckmann, geometric_normal, incoming,
            make_float3(0.37f, 0.73f, 0.42f));
        record_sample(4u, 1u, beckmann_sample);
        const auto beckmann_eval = detail::bsdf_microfacet_beckmann_eval(
            kernel_globals, beckmann, incoming, direct);
        output.write(7u,
                     make_float4(beckmann_eval.value, beckmann_eval.pdf));

        const device_svm::MicrofacetClosure glass{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_ggx_glass,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.28f,
                      .alpha_y = 0.28f,
                      .ior = 1.45f,
                      .energy_scale = 1.09f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::dielectric),
                      .T = make_float3(0.0f)},
            .generalized_schlick = empty_fresnel()};
        const auto glass_reflection = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, glass, geometric_normal, incoming,
            make_float3(0.37f, 0.73f, 0.01f));
        record_sample(8u, 2u, glass_reflection);
        const auto glass_refraction = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, glass, geometric_normal, incoming,
            make_float3(0.37f, 0.73f, 0.90f));
        record_sample(11u, 3u, glass_refraction);
        const auto glass_eval = detail::bsdf_microfacet_ggx_eval(
            kernel_globals, glass, incoming, direct);
        output.write(14u,
                     make_float4(glass_eval.value, glass_eval.pdf));

        const device_svm::MicrofacetConductorClosure conductor{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_ggx,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.25f,
                      .alpha_y = 0.52f,
                      .ior = 1.45f,
                      .energy_scale = 1.11f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::conductor),
                      .T = basis.tangent},
            .conductor = {
                .thin_film = {.thickness = 0.0f, .ior = 1.4f},
                .ior = make_float3(0.3f, 0.8f, 1.2f),
                .extinction = make_float3(3.0f, 2.0f, 1.5f)}};
        const auto conductor_sample = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, conductor, geometric_normal, incoming,
            make_float3(0.21f, 0.83f, 0.5f));
        record_sample(15u, 4u, conductor_sample);
        const auto conductor_eval = detail::bsdf_microfacet_ggx_eval(
            kernel_globals, conductor, incoming, direct);
        output.write(18u,
                     make_float4(conductor_eval.value, conductor_eval.pdf));

        const device_svm::MicrofacetF82TintClosure f82{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_ggx,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.42f,
                      .alpha_y = 0.18f,
                      .ior = 1.45f,
                      .energy_scale = 1.07f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::f82_tint),
                      .T = basis.tangent},
            .f82_tint = {
                .thin_film = {.thickness = 0.0f, .ior = 1.3f},
                .f0 = make_float3(0.8f, 0.4f, 0.1f),
                .b = make_float3(0.12f, -0.08f, 0.04f)}};
        const auto f82_sample = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, f82, geometric_normal, incoming,
            make_float3(0.67f, 0.19f, 0.5f));
        record_sample(19u, 5u, f82_sample);
        const auto f82_eval = detail::bsdf_microfacet_ggx_eval(
            kernel_globals, f82, incoming, direct);
        output.write(22u, make_float4(f82_eval.value, f82_eval.pdf));

        const device_svm::MicrofacetClosure generalized{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_ggx_glass,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.35f,
                      .alpha_y = 0.35f,
                      .ior = 1.5f,
                      .energy_scale = 1.06f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::generalized_schlick),
                      .T = make_float3(0.0f)},
            .generalized_schlick = {
                .thin_film = {.thickness = 0.0f, .ior = 1.4f},
                .reflection_tint = make_float3(1.0f),
                .transmission_tint = make_float3(0.7f, 0.8f, 0.9f),
                .f0 = make_float3(0.03f, 0.05f, 0.08f),
                .f90 = make_float3(1.0f),
                .exponent = -1.5f}};
        const auto generalized_reflection =
            detail::bsdf_microfacet_ggx_sample(
                kernel_globals, generalized, geometric_normal, incoming,
                make_float3(0.41f, 0.62f, 0.01f));
        record_sample(23u, 6u, generalized_reflection);
        const auto generalized_refraction =
            detail::bsdf_microfacet_ggx_sample(
                kernel_globals, generalized, geometric_normal, incoming,
                make_float3(0.41f, 0.62f, 0.90f));
        record_sample(26u, 7u, generalized_refraction);

        const device_svm::MicrofacetClosure thin_singular{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_thin_glass_transmission,
                       .sample_weight = 1.0f,
                       .N = -normal},
            .param = {.alpha_x = 1.0e-6f,
                      .alpha_y = 1.0e-6f,
                      .ior = 1.0f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = make_float3(0.0f)},
            .generalized_schlick = empty_fresnel()};
        const auto thin_singular_sample =
            detail::bsdf_thin_glass_transmission_sample(
                kernel_globals, thin_singular, geometric_normal, incoming,
                make_float3(0.37f, 0.73f, 0.5f));
        output.write(29u, make_float4(thin_singular_sample.wo,
                                      thin_singular_sample.pdf));
        output.write(30u, make_float4(thin_singular_sample.value,
                                      thin_singular_sample.sampled_roughness.x));
        labels.write(8u, thin_singular_sample.label);

        const device_svm::MicrofacetClosure thin_rough{
            .common = thin_singular.common,
            .param = {.alpha_x = 0.18f,
                      .alpha_y = 0.18f,
                      .ior = 1.0f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = make_float3(0.0f)},
            .generalized_schlick = empty_fresnel()};
        const auto thin_rough_sample =
            detail::bsdf_thin_glass_transmission_sample(
                kernel_globals, thin_rough, geometric_normal, incoming,
                make_float3(0.37f, 0.73f, 0.5f));
        output.write(31u,
                     make_float4(thin_rough_sample.wo,
                                 thin_rough_sample.pdf));
        labels.write(9u, thin_rough_sample.label);

        const auto rejected = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, ggx, geometric_normal, -normal,
            make_float3(0.37f, 0.73f, 0.5f));
        labels.write(10u, rejected.label);

        const auto glass_transmission_eval =
            detail::bsdf_microfacet_ggx_eval(
                kernel_globals, glass, incoming, glass_refraction.wo);
        output.write(32u,
                     make_float4(glass_transmission_eval.value,
                                 glass_transmission_eval.pdf));
        const auto generalized_transmission_eval =
            detail::bsdf_microfacet_ggx_eval(
                kernel_globals, generalized, incoming,
                generalized_refraction.wo);
        output.write(33u,
                     make_float4(generalized_transmission_eval.value,
                                 generalized_transmission_eval.pdf));

        const device_svm::MicrofacetClosure beckmann_refraction{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_microfacet_beckmann_refraction,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.32f,
                      .alpha_y = 0.32f,
                      .ior = 1.33f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto beckmann_refraction_sample =
            detail::bsdf_microfacet_beckmann_sample(
                kernel_globals, beckmann_refraction, geometric_normal,
                incoming, make_float3(0.29f, 0.81f, 0.2f));
        record_sample(34u, 11u, beckmann_refraction_sample);
        const auto beckmann_refraction_eval =
            detail::bsdf_microfacet_beckmann_eval(
                kernel_globals, beckmann_refraction, incoming,
                beckmann_refraction_sample.wo);
        output.write(37u,
                     make_float4(beckmann_refraction_eval.value,
                                 beckmann_refraction_eval.pdf));

        const device_svm::MicrofacetClosure singular{
            .common = ggx.common,
            .param = {.alpha_x = 1.0e-6f,
                      .alpha_y = 1.0e-6f,
                      .ior = 1.45f,
                      .energy_scale = 1.17f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          device_svm::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto singular_sample = detail::bsdf_microfacet_ggx_sample(
            kernel_globals, singular, geometric_normal, incoming,
            make_float3(0.37f, 0.73f, 0.42f));
        record_sample(38u, 12u, singular_sample);
        const auto singular_eval = detail::bsdf_microfacet_ggx_eval(
            kernel_globals, singular, incoming, direct);
        output.write(41u,
                     make_float4(singular_eval.value, singular_eval.pdf));
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-4f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 2.0e-4f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output = device.create_buffer<luisa::float4>(output_count);
  auto labels = device.create_buffer<std::uint32_t>(label_count);
  const auto shader = device.compile(
      scattering_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, output_count> actual{};
  std::array<std::uint32_t, label_count> actual_labels{};
  stream << shader(output, labels).dispatch(1u)
         << output.copy_to(actual.data())
         << labels.copy_to(actual_labels.data()) << synchronize();

  constexpr std::array expected{
      luisa::float4{0.0441898704f, -0.799224615f, 0.599406004f, 0.48666808f},
      luisa::float4{0.544672608f, 0.544672608f, 0.544672608f, 0.0f},
      luisa::float4{0.22f, 0.61f, 1.0f, 0.0f},
      luisa::float4{0.14115262f, 0.14115262f, 0.14115262f, 0.12210504f},
      luisa::float4{0.354770213f, -0.490503341f, 0.795955122f, 0.340102851f},
      luisa::float4{0.340102851f, 0.340102851f, 0.340102851f, 0.0f},
      luisa::float4{0.34f, 0.58f, 1.0f, 0.0f},
      luisa::float4{0.252569586f, 0.252569586f, 0.252569586f, 0.252569586f},
      luisa::float4{-0.0972262621f, -0.592058897f, 0.800008535f, 0.0249976385f},
      luisa::float4{0.0271253791f, 0.0271253791f, 0.0271253791f, 0.0f},
      luisa::float4{0.28f, 0.28f, 1.0f, 0.0f},
      luisa::float4{-0.284247488f, 0.229530498f, -0.930870116f, 25.3747864f},
      luisa::float4{27.6519051f, 27.6519051f, 27.6519051f, 0.0f},
      luisa::float4{0.28f, 0.28f, 1.45f, 0.0f},
      luisa::float4{0.00803633686f, 0.00803633686f, 0.00803633686f,
                    0.00740495697f},
      luisa::float4{0.0538457632f, -0.923459888f, 0.379898429f, 0.322688937f},
      luisa::float4{0.291490346f, 0.183629915f, 0.107035987f, 0.0f},
      luisa::float4{0.25f, 0.52f, 1.0f, 0.0f},
      luisa::float4{0.15349865f, 0.0964855254f, 0.0558495298f, 0.157260045f},
      luisa::float4{0.5540061f, -0.0864906162f, 0.828007936f, 0.39190951f},
      luisa::float4{0.332471043f, 0.166235521f, 0.0415588804f, 0.0f},
      luisa::float4{0.42f, 0.18f, 1.0f, 0.0f},
      luisa::float4{0.146594226f, 0.0732971132f, 0.0183242783f, 0.172209054f},
      luisa::float4{-0.00424671173f, -0.515717387f, 0.856748343f, 0.0412043482f},
      luisa::float4{0.0198352132f, 0.0330228917f, 0.0528044105f, 0.0f},
      luisa::float4{0.35f, 0.35f, 1.0f, 0.0f},
      luisa::float4{-0.294224918f, 0.217379332f, -0.930686772f, 19.1016998f},
      luisa::float4{14.7117825f, 16.466795f, 17.9401398f, 0.0f},
      luisa::float4{0.35f, 0.35f, 1.5f, 0.0f},
      luisa::float4{-0.349890679f, 0.149953157f, -0.924711108f, 1.0e6f},
      luisa::float4{1.0e6f, 1.0e6f, 1.0e6f, 0.0f},
      luisa::float4{-0.500142455f, 0.105249554f, -0.859523058f, 1.69436204f},
      luisa::float4{27.6519737f, 27.6519737f, 27.6519737f, 25.3748493f},
      luisa::float4{14.7117519f, 16.4667625f, 17.9401035f, 19.1016598f},
      luisa::float4{-0.363770127f, 0.198043242f, -0.91019243f, 30.6961956f},
      luisa::float4{30.6961956f, 30.6961956f, 30.6961956f, 0.0f},
      luisa::float4{0.32f, 0.32f, 1.33f, 0.0f},
      luisa::float4{30.6961956f, 30.6961956f, 30.6961956f, 30.6961956f},
      luisa::float4{0.0181359351f, -0.402086824f, 0.915421963f, 1.0e6f},
      luisa::float4{1.17e6f, 1.17e6f, 1.17e6f, 0.0f},
      luisa::float4{1.0e-6f, 1.0e-6f, 1.0f, 0.0f},
      luisa::float4{0.0f}};
  constexpr std::array expected_labels{
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_transmit | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_transmit | closure::label_glossy,
      closure::label_transmit | closure::label_singular,
      closure::label_transmit | closure::label_glossy,
      closure::label_none,
      closure::label_transmit | closure::label_glossy,
      closure::label_reflect | closure::label_singular};

  auto valid = actual_labels == expected_labels;
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (!near(actual[index], expected[index])) {
      valid = false;
    }
  }
  if (!valid) {
    std::cerr << "Cycles Microfacet scattering mismatch on " << backend
              << '\n';
    for (std::size_t index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": (" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << ") expected ("
                << expected[index].x << ", " << expected[index].y << ", "
                << expected[index].z << ", " << expected[index].w << ")\n";
    }
    for (std::size_t index = 0u; index < actual_labels.size(); ++index) {
      std::cerr << "label " << index << ": " << actual_labels[index]
                << " expected " << expected_labels[index] << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend =
      std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
