#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_sheen.h"

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
namespace compiler_svm = psycles::compiler::cycles_svm;
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace mapping = psycles::luisa_backend::cycles_sample_mapping;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t output_count = 18u;
inline constexpr std::uint32_t label_count = 5u;

static_assert(
    closure::type_sheen ==
    static_cast<std::uint32_t>(compiler_svm::CLOSURE_BSDF_SHEEN_ID));
static_assert(
    closure::type_ashikhmin_velvet ==
    static_cast<std::uint32_t>(
        compiler_svm::CLOSURE_BSDF_ASHIKHMIN_VELVET_ID));

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt labels) noexcept {
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        const auto geometric_normal = normal;
        const auto incoming = normalize(make_float3(0.35f, -0.15f, 0.925f));
        const auto direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
        const auto random = make_float2(0.820808350f, 0.676392674f);
        const auto basis =
            mapping::make_orthonormals_safe_tangent(normal, incoming);

        const device_svm::SheenClosure sheen{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_sheen,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.roughness = 0.41f,
                      .transform_a = 0.72f,
                      .transform_b = -0.18f,
                      .T = basis.tangent,
                      .B = basis.bitangent}};
        const auto sheen_sample = detail::bsdf_sheen_sample(
            sheen, geometric_normal, incoming, random);
        const auto sheen_eval =
            detail::bsdf_sheen_eval(sheen, incoming, direct);
        output.write(0u, make_float4(sheen_sample.wo, sheen_sample.pdf));
        output.write(1u, make_float4(sheen_sample.value, 0.0f));
        output.write(2u, make_float4(sheen_eval.value, sheen_eval.pdf));
        output.write(3u,
                     make_float4(sheen.param.transform_a,
                                 sheen.param.transform_b, sheen_sample.eta,
                                 sheen_sample.sampled_roughness.x));
        labels.write(0u, sheen_sample.label);

        const auto rejected_sheen = detail::bsdf_sheen_sample(
            sheen, -geometric_normal, incoming, random);
        output.write(4u,
                     make_float4(rejected_sheen.value, rejected_sheen.pdf));
        output.write(5u, make_float4(rejected_sheen.wo, 0.0f));
        labels.write(1u, rejected_sheen.label);

        const auto backside_sheen =
            detail::bsdf_sheen_eval(sheen, incoming, -normal);
        output.write(6u,
                     make_float4(backside_sheen.value, backside_sheen.pdf));
        const auto tangent_sheen =
            detail::bsdf_sheen_eval(sheen, incoming, sheen.param.T);
        output.write(7u,
                     make_float4(tangent_sheen.value, tangent_sheen.pdf));

        const device_svm::VelvetClosure velvet{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_ashikhmin_velvet,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.sigma = 0.65f,
                      .invsigma2 = 1.0f / (0.65f * 0.65f)}};
        const auto velvet_incoming =
            normalize(0.92f * basis.tangent + 0.39f * normal);
        const auto velvet_direct =
            normalize(0.80f * basis.tangent + 0.60f * normal);
        const auto velvet_sample = detail::bsdf_ashikhmin_velvet_sample(
            velvet, geometric_normal, velvet_incoming, random);
        const auto velvet_eval = detail::bsdf_ashikhmin_velvet_eval(
            velvet, velvet_incoming, velvet_direct);
        output.write(8u, make_float4(velvet_sample.wo, velvet_sample.pdf));
        output.write(9u, make_float4(velvet_sample.value, 0.0f));
        output.write(10u, make_float4(velvet_eval.value, velvet_eval.pdf));
        output.write(11u,
                     make_float4(velvet.param.sigma,
                                 velvet.param.invsigma2, velvet_sample.eta,
                                 velvet_sample.sampled_roughness.y));
        labels.write(2u, velvet_sample.label);

        const auto rejected_velvet = detail::bsdf_ashikhmin_velvet_sample(
            velvet, -geometric_normal, velvet_incoming, random);
        output.write(12u,
                     make_float4(rejected_velvet.value,
                                 rejected_velvet.pdf));
        output.write(13u, make_float4(rejected_velvet.wo, 0.0f));
        labels.write(3u, rejected_velvet.label);

        const auto backside_velvet = detail::bsdf_ashikhmin_velvet_eval(
            velvet, velvet_incoming, -normal);
        output.write(14u,
                     make_float4(backside_velvet.value,
                                 backside_velvet.pdf));
        const auto singular_velvet =
            detail::bsdf_ashikhmin_velvet_eval(velvet, normal, normal);
        output.write(15u,
                     make_float4(singular_velvet.value,
                                 singular_velvet.pdf));

        const auto rejected_incoming =
            detail::bsdf_ashikhmin_velvet_sample(
                velvet, geometric_normal, -normal, random);
        output.write(16u,
                     make_float4(rejected_incoming.wo,
                                 rejected_incoming.pdf));
        output.write(17u,
                     make_float4(rejected_incoming.value, 0.0f));
        labels.write(4u, rejected_incoming.label);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 8.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 8.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool exactly_zero(luisa::float4 value) noexcept {
  return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f &&
         value.w == 0.0f;
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
  stream << shader(output, labels).dispatch(1u) << output.copy_to(actual.data())
         << labels.copy_to(actual_labels.data()) << synchronize();

  constexpr std::array expected{
      luisa::float4{0.552439630f, 0.507043540f, 0.661602020f, 0.287902415f},
      luisa::float4{0.287902415f, 0.287902415f, 0.287902415f, 0.0f},
      luisa::float4{0.209007919f, 0.209007919f, 0.209007919f, 0.209007919f},
      luisa::float4{0.720000029f, -0.180000007f, 1.0f, 1.0f},
      luisa::float4{0.0f},
      luisa::float4{0.552439630f, 0.507043540f, 0.661602020f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.893368244f, -0.254301727f, 0.370437086f,
                    0.159154937f},
      luisa::float4{0.163679332f, 0.163679332f, 0.163679332f, 0.0f},
      luisa::float4{0.153165922f, 0.153165922f, 0.153165922f,
                    0.159154937f},
      luisa::float4{0.649999976f, 2.36686420f, 1.0f, 1.0f},
      luisa::float4{0.0f},
      luisa::float4{0.893368244f, -0.254301727f, 0.370437086f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.893368244f, -0.254301727f, 0.370437086f, 0.0f},
      luisa::float4{0.0f}};
  constexpr std::array expected_labels{
      closure::label_reflect | closure::label_diffuse,
      closure::label_reflect | closure::label_diffuse,
      closure::label_reflect | closure::label_diffuse,
      closure::label_none, closure::label_none};

  auto valid = true;
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    valid &= near(actual[index], expected[index]);
  }
  valid &= actual_labels == expected_labels;
  valid &= exactly_zero(actual[4u]) && exactly_zero(actual[6u]) &&
           exactly_zero(actual[12u]) && exactly_zero(actual[14u]) &&
           exactly_zero(actual[15u]) && exactly_zero(actual[17u]);
  valid &= actual[3u].z == 1.0f && actual[3u].w == 1.0f &&
           actual[11u].z == 1.0f && actual[11u].w == 1.0f;
  if (!valid) {
    std::cerr << "Cycles Sheen/Velvet scattering mismatch on " << backend
              << '\n';
    for (std::size_t index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": (" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << ")\n";
    }
    std::cerr << "labels:";
    for (const auto label : actual_labels) {
      std::cerr << ' ' << label;
    }
    std::cerr << '\n';
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
