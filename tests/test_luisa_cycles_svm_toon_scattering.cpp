#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_toon.h"

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
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t output_count = 16u;
inline constexpr std::uint32_t label_count = 5u;
inline constexpr float half_pi = 1.57079632679489661923f;

static_assert(
    closure::type_diffuse_toon ==
    static_cast<std::uint32_t>(compiler_svm::CLOSURE_BSDF_DIFFUSE_TOON_ID));
static_assert(
    closure::type_glossy_toon ==
    static_cast<std::uint32_t>(compiler_svm::CLOSURE_BSDF_GLOSSY_TOON_ID));
static_assert(
    closure::type_ray_portal ==
    static_cast<std::uint32_t>(compiler_svm::CLOSURE_BSDF_RAY_PORTAL_ID));

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt labels) noexcept {
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        const auto geometric_normal = normal;
        const auto incoming = normalize(make_float3(0.35f, -0.15f, 0.925f));
        const auto direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
        const auto random = make_float2(0.820808350f, 0.676392674f);

        const device_svm::ToonClosure diffuse{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_diffuse_toon,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.size = 0.37f * half_pi, .smooth = 0.21f * half_pi}};
        const auto diffuse_sample = detail::bsdf_diffuse_toon_sample(
            diffuse, geometric_normal, incoming, random);
        const auto diffuse_eval =
            detail::bsdf_diffuse_toon_eval(diffuse, incoming, direct);
        output.write(0u, make_float4(diffuse_sample.wo, diffuse_sample.pdf));
        output.write(1u, make_float4(diffuse_sample.value, 0.0f));
        output.write(2u, make_float4(diffuse_eval.value, diffuse_eval.pdf));
        output.write(3u, make_float4(diffuse.param.size, diffuse.param.smooth,
                                     diffuse_sample.eta,
                                     diffuse_sample.sampled_roughness.x));
        labels.write(0u, diffuse_sample.label);

        const device_svm::ToonClosure glossy{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_glossy_toon,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = {.size = 0.63f * half_pi, .smooth = 0.14f * half_pi}};
        const auto glossy_sample = detail::bsdf_glossy_toon_sample(
            glossy, geometric_normal, incoming, random);
        const auto glossy_eval =
            detail::bsdf_glossy_toon_eval(glossy, incoming, direct);
        output.write(4u, make_float4(glossy_sample.wo, glossy_sample.pdf));
        output.write(5u, make_float4(glossy_sample.value, 0.0f));
        output.write(6u, make_float4(glossy_eval.value, glossy_eval.pdf));
        output.write(7u, make_float4(glossy.param.size, glossy.param.smooth,
                                     glossy_sample.eta,
                                     glossy_sample.sampled_roughness.y));
        labels.write(1u, glossy_sample.label);

        const auto rejected_diffuse = detail::bsdf_diffuse_toon_sample(
            diffuse, -geometric_normal, incoming, random);
        output.write(8u,
                     make_float4(rejected_diffuse.value, rejected_diffuse.pdf));
        labels.write(2u, rejected_diffuse.label);

        const auto rejected_glossy = detail::bsdf_glossy_toon_sample(
            glossy, geometric_normal, -normal, random);
        output.write(9u, make_float4(rejected_glossy.wo, rejected_glossy.pdf));
        output.write(10u, make_float4(rejected_glossy.value, 0.0f));
        labels.write(3u, rejected_glossy.label);

        const auto rejected_glossy_eval =
            detail::bsdf_glossy_toon_eval(glossy, incoming, -normal);
        output.write(11u, make_float4(rejected_glossy_eval.value,
                                      rejected_glossy_eval.pdf));

        const device_svm::ToonClosure shoulder{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_diffuse_toon,
                       .sample_weight = 1.0f,
                       .N = make_float3(0.0f, 0.0f, 1.0f)},
            .param = {.size = 0.2f * half_pi, .smooth = 0.4f * half_pi}};
        const auto shoulder_angle =
            shoulder.param.size + shoulder.param.smooth * 0.25f;
        const auto shoulder_direction =
            make_float3(sin(shoulder_angle), 0.0f, cos(shoulder_angle));
        const auto shoulder_eval = detail::bsdf_diffuse_toon_eval(
            shoulder, incoming, shoulder_direction);
        output.write(12u, make_float4(shoulder_eval.value, shoulder_eval.pdf));

        const device_svm::ToonClosure tangent{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_diffuse_toon,
                       .sample_weight = 1.0f,
                       .N = make_float3(0.0f, 0.0f, 1.0f)},
            .param = {.size = half_pi, .smooth = half_pi}};
        const auto tangent_eval = detail::bsdf_diffuse_toon_eval(
            tangent, incoming, make_float3(1.0f, 0.0f, 0.0f));
        output.write(13u, make_float4(tangent_eval.value, tangent_eval.pdf));

        const device_svm::ToonClosure tiny{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_diffuse_toon,
                       .sample_weight = 1.0f,
                       .N = make_float3(0.0f, 0.0f, 1.0f)},
            .param = {.size = 1.0e-5f * half_pi, .smooth = 0.0f}};
        const auto tiny_sample = detail::bsdf_diffuse_toon_sample(
            tiny, tiny.common.N, incoming, random);
        output.write(14u, make_float4(tiny_sample.wo, tiny_sample.pdf));
        output.write(15u, make_float4(tiny_sample.value, tiny.param.size));
        labels.write(4u, tiny_sample.label);
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
      luisa::float4{0.682372749f, -0.296361089f, 0.668234646f, 0.411154330f},
      luisa::float4{0.411154330f, 0.411154330f, 0.411154330f, 0.0f},
      luisa::float4{0.411154330f, 0.411154330f, 0.411154330f, 0.411154330f},
      luisa::float4{0.581194639f, 0.329867214f, 1.0f, 1.0f},
      luisa::float4{0.677203298f, -0.421548009f, 0.603069663f, 0.246169761f},
      luisa::float4{0.246169761f, 0.246169761f, 0.246169761f, 0.0f},
      luisa::float4{0.246169761f, 0.246169761f, 0.246169761f, 0.246169761f},
      luisa::float4{0.989601731f, 0.219911486f, 1.0f, 1.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.289572865f, 0.289572865f, 0.289572865f, 0.386097133f},
      luisa::float4{0.0f},
      luisa::float4{9.45510965e-6f, -3.48953949e-6f, 1.0f, 1.29006131e9f},
      luisa::float4{1.29006131e9f, 1.29006131e9f, 1.29006131e9f,
                    1.57079630e-5f}};
  constexpr std::array expected_labels{
      closure::label_reflect | closure::label_diffuse,
      closure::label_reflect | closure::label_glossy, closure::label_none,
      closure::label_none, closure::label_reflect | closure::label_diffuse};

  auto valid = true;
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    valid &= near(actual[index], expected[index]);
  }
  valid &= actual_labels == expected_labels;
  valid &= exactly_zero(actual[8u]) && exactly_zero(actual[9u]) &&
           exactly_zero(actual[10u]) && exactly_zero(actual[11u]) &&
           exactly_zero(actual[13u]);
  valid &= actual[3u].z == 1.0f && actual[3u].w == 1.0f &&
           actual[7u].z == 1.0f && actual[7u].w == 1.0f;
  valid &= std::isfinite(actual[14u].w) && actual[14u].w > 1.0e9f &&
           actual[15u].x == actual[14u].w && actual[15u].y == actual[14u].w &&
           actual[15u].z == actual[14u].w;
  if (!valid) {
    std::cerr << "Cycles Toon scattering mismatch on " << backend << '\n';
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
