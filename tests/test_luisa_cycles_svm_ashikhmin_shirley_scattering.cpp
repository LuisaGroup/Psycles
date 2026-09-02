#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_ashikhmin_shirley.h"

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

inline constexpr std::uint32_t output_count = 27u;
inline constexpr std::uint32_t label_count = 8u;

[[nodiscard]] device_svm::FresnelGeneralizedSchlick
empty_fresnel() noexcept {
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
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        const auto geometric_normal = normal;
        const auto incoming =
            normalize(make_float3(0.35f, -0.15f, 0.925f));
        const auto direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
        const auto basis = mapping::make_orthonormals_tangent(
            normal, normalize(make_float3(0.9f, 0.2f, -0.1f)));

        const auto record_sample = [&](std::uint32_t value_index,
                                       std::uint32_t label_index,
                                       const detail::BsdfSample &sample) {
          output.write(value_index, make_float4(sample.wo, sample.pdf));
          output.write(value_index + 1u, make_float4(sample.value, 0.0f));
          output.write(value_index + 2u,
                       make_float4(sample.sampled_roughness.x,
                                   sample.sampled_roughness.y, sample.eta,
                                   0.0f));
          labels.write(label_index, sample.label);
        };

        const device_svm::MicrofacetClosure isotropic{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_ashikhmin_shirley,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.alpha_x = 0.37f,
                      .alpha_y = 0.37f,
                      .ior = 1.0f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          closure::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto isotropic_sample =
            detail::bsdf_ashikhmin_shirley_sample(
                isotropic, geometric_normal, incoming,
                make_float2(0.37f, 0.73f));
        record_sample(0u, 0u, isotropic_sample);
        const auto isotropic_eval = detail::bsdf_ashikhmin_shirley_eval(
            isotropic, incoming, direct);
        output.write(3u,
                     make_float4(isotropic_eval.value, isotropic_eval.pdf));

        const device_svm::MicrofacetClosure anisotropic{
            .common = isotropic.common,
            .param = {.alpha_x = 0.17f,
                      .alpha_y = 0.61f,
                      .ior = 1.0f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          closure::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto anisotropic_eval = detail::bsdf_ashikhmin_shirley_eval(
            anisotropic, incoming, direct);
        output.write(4u, make_float4(anisotropic_eval.value,
                                     anisotropic_eval.pdf));

        const auto quadrant_0 = detail::bsdf_ashikhmin_shirley_sample(
            anisotropic, geometric_normal, incoming,
            make_float2(0.13f, 0.73f));
        const auto quadrant_1 = detail::bsdf_ashikhmin_shirley_sample(
            anisotropic, geometric_normal, incoming,
            make_float2(0.37f, 0.41f));
        const auto quadrant_2 = detail::bsdf_ashikhmin_shirley_sample(
            anisotropic, geometric_normal, incoming,
            make_float2(0.62f, 0.82f));
        const auto quadrant_3 = detail::bsdf_ashikhmin_shirley_sample(
            anisotropic, geometric_normal, incoming,
            make_float2(0.88f, 0.57f));
        record_sample(5u, 1u, quadrant_0);
        record_sample(8u, 2u, quadrant_1);
        record_sample(11u, 3u, quadrant_2);
        record_sample(14u, 4u, quadrant_3);

        const auto rejected_incoming =
            detail::bsdf_ashikhmin_shirley_sample(
                anisotropic, geometric_normal, -normal,
                make_float2(0.37f, 0.73f));
        record_sample(17u, 5u, rejected_incoming);

        const auto rejected_geometric =
            detail::bsdf_ashikhmin_shirley_sample(
                anisotropic, -geometric_normal, incoming,
                make_float2(0.37f, 0.73f));
        record_sample(20u, 6u, rejected_geometric);

        const device_svm::MicrofacetClosure singular{
            .common = isotropic.common,
            .param = {.alpha_x = 1.0e-4f,
                      .alpha_y = 1.0e-4f,
                      .ior = 1.0f,
                      .energy_scale = 1.0f,
                      .fresnel_type = static_cast<std::uint32_t>(
                          closure::MicrofacetFresnel::none),
                      .T = basis.tangent},
            .generalized_schlick = empty_fresnel()};
        const auto singular_sample = detail::bsdf_ashikhmin_shirley_sample(
            singular, geometric_normal, incoming,
            make_float2(0.37f, 0.73f));
        record_sample(23u, 7u, singular_sample);
        const auto singular_eval = detail::bsdf_ashikhmin_shirley_eval(
            singular, incoming, direct);
        output.write(26u,
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
      luisa::float4{-0.0526579022f, 0.00119850039f, 0.998611927f,
                    0.414520204f},
      luisa::float4{0.393805861f, 0.393805861f, 0.393805861f, 0.0f},
      luisa::float4{0.37f, 0.37f, 1.0f, 0.0f},
      luisa::float4{0.259996474f, 0.259996474f, 0.259996474f,
                    0.28130725f},
      luisa::float4{0.0740518868f, 0.0740518868f, 0.0740518868f,
                    0.0801215917f},
      luisa::float4{0.0332171917f, 0.142536074f, 0.989232063f,
                    0.535278857f},
      luisa::float4{0.49071297f, 0.49071297f, 0.49071297f, 0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{-0.366439313f, 0.392421901f, 0.843639374f,
                    0.354017824f},
      luisa::float4{0.222224072f, 0.222224072f, 0.222224072f, 0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{-0.0422297418f, -0.737060666f, 0.674506307f,
                    0.621847391f},
      luisa::float4{0.530636549f, 0.530636549f, 0.530636549f, 0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{0.231927007f, -0.858716011f, 0.456964612f,
                    0.462558091f},
      luisa::float4{0.338558048f, 0.338558048f, 0.338558048f, 0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{-0.213537887f, 0.0883828104f, 0.972928643f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.17f, 0.61f, 1.0f, 0.0f},
      luisa::float4{0.0181359351f, -0.402086824f, 0.915421963f, 1.0e6f},
      luisa::float4{1.0e6f, 1.0e6f, 1.0e6f, 0.0f},
      luisa::float4{1.0e-4f, 1.0e-4f, 1.0f, 0.0f},
      luisa::float4{0.0f}};
  constexpr std::array expected_labels{
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_none,
      closure::label_none,
      closure::label_reflect | closure::label_singular};

  auto valid = actual_labels == expected_labels;
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (!near(actual[index], expected[index])) {
      valid = false;
    }
  }
  if (!valid) {
    std::cerr << "Cycles Ashikhmin-Shirley scattering mismatch on "
              << backend << '\n';
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
