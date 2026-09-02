#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_simple_closure.h"

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
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t output_count = 20u;
inline constexpr std::uint32_t label_count = 6u;

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt labels) noexcept {
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        const auto geometric_normal = normal;
        const auto incoming = normalize(make_float3(0.35f, -0.15f, 0.925f));
        const auto direct = normalize(make_float3(0.4f, 0.1f, 0.91f));
        const auto random = make_float2(0.820808350f, 0.676392674f);

        const device_svm::ShaderClosureCommon diffuse{
            .weight = make_float3(1.0f),
            .type = closure::type_diffuse,
            .sample_weight = 1.0f,
            .N = normal};
        const auto diffuse_sample = detail::bsdf_diffuse_sample(
            diffuse, geometric_normal, incoming, random);
        const auto diffuse_eval =
            detail::bsdf_diffuse_eval(diffuse, incoming, direct);
        output.write(0u, make_float4(diffuse_sample.wo, diffuse_sample.pdf));
        output.write(1u, make_float4(diffuse_sample.value, 0.0f));
        output.write(2u, make_float4(diffuse_eval.value, diffuse_eval.pdf));
        labels.write(0u, diffuse_sample.label);

        const device_svm::ShaderClosureCommon translucent{
            .weight = make_float3(1.0f),
            .type = closure::type_translucent,
            .sample_weight = 1.0f,
            .N = normal};
        const auto translucent_sample = detail::bsdf_translucent_sample(
            translucent, geometric_normal, incoming, random);
        const auto translucent_eval =
            detail::bsdf_translucent_eval(translucent, incoming, -direct);
        output.write(
            3u, make_float4(translucent_sample.wo, translucent_sample.pdf));
        output.write(4u, make_float4(translucent_sample.value, 0.0f));
        output.write(5u,
                     make_float4(translucent_eval.value, translucent_eval.pdf));
        labels.write(1u, translucent_sample.label);

        const auto oren_param = detail::oren_nayar_param(
            make_float3(0.72f, 0.31f, 0.08f), dot(normal, incoming), 0.63f);
        const device_svm::OrenNayarClosure oren{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_oren_nayar,
                       .sample_weight = 1.0f,
                       .N = normal},
            .param = oren_param};
        const auto oren_sample = detail::bsdf_oren_nayar_sample(
            oren, geometric_normal, incoming, random);
        const auto oren_eval =
            detail::bsdf_oren_nayar_eval(oren, incoming, direct);
        output.write(6u, make_float4(oren_sample.wo, oren_sample.pdf));
        output.write(7u, make_float4(oren_sample.value, oren_sample.eta));
        output.write(8u, make_float4(oren_sample.sampled_roughness.x,
                                     oren_sample.sampled_roughness.y,
                                     oren.param.a, oren.param.b));
        output.write(9u, make_float4(oren_eval.value, oren_eval.pdf));
        labels.write(2u, oren_sample.label);

        const device_svm::OrenNayarClosure rough_translucent{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_rough_translucent,
                       .sample_weight = 1.0f,
                       .N = -normal},
            .param = oren_param};
        const auto rough_translucent_sample =
            detail::bsdf_rough_translucent_sample(
                rough_translucent, geometric_normal, incoming, random);
        const auto rough_translucent_eval = detail::bsdf_rough_translucent_eval(
            rough_translucent, incoming, -direct);
        output.write(10u, make_float4(rough_translucent_sample.wo,
                                      rough_translucent_sample.pdf));
        output.write(11u, make_float4(rough_translucent_sample.value,
                                      rough_translucent_sample.eta));
        output.write(12u, make_float4(rough_translucent_eval.value,
                                      rough_translucent_eval.pdf));
        labels.write(3u, rough_translucent_sample.label);

        const device_svm::ShaderClosureCommon transparent{
            .weight = make_float3(1.0f),
            .type = closure::type_transparent,
            .sample_weight = 1.0f,
            .N = normal};
        const auto transparent_sample = detail::bsdf_transparent_sample(
            transparent, geometric_normal, incoming);
        const auto transparent_eval =
            detail::bsdf_transparent_eval(transparent, incoming, direct);
        output.write(
            13u, make_float4(transparent_sample.wo, transparent_sample.pdf));
        output.write(14u, make_float4(transparent_sample.value, 0.0f));
        output.write(15u,
                     make_float4(transparent_eval.value, transparent_eval.pdf));
        labels.write(4u, transparent_sample.label);

        const auto rejected_diffuse = detail::bsdf_diffuse_sample(
            diffuse, -geometric_normal, incoming, random);
        const auto rejected_oren =
            detail::bsdf_oren_nayar_eval(oren, incoming, -direct);
        output.write(16u,
                     make_float4(rejected_diffuse.value, rejected_diffuse.pdf));
        output.write(17u, make_float4(rejected_oren.value, rejected_oren.pdf));

        const device_svm::OrenNayarClosure zero_roughness{
            .common = oren.common,
            .param = detail::oren_nayar_param(make_float3(0.72f, 0.31f, 0.08f),
                                              dot(normal, incoming), 0.0f)};
        const auto diffuse_limit =
            detail::bsdf_oren_nayar_eval(zero_roughness, incoming, direct);
        output.write(18u, make_float4(diffuse_limit.value, diffuse_limit.pdf));

        const device_svm::OrenNayarClosure tangent_boundary{
            .common = {.weight = make_float3(1.0f),
                       .type = closure::type_oren_nayar,
                       .sample_weight = 1.0f,
                       .N = make_float3(0.0f, 0.0f, 1.0f)},
            .param = oren_param};
        const auto tangent_eval = detail::bsdf_oren_nayar_eval(
            tangent_boundary, incoming, make_float3(1.0f, 0.0f, 0.0f));
        output.write(19u, make_float4(tangent_eval.value, tangent_eval.pdf));
        labels.write(5u, rejected_diffuse.label);
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
      luisa::float4{0.765341878f, -0.286503345f, 0.576339781f, 0.244151756f},
      luisa::float4{0.244151756f, 0.244151756f, 0.244151756f, 0.0f},
      luisa::float4{0.287736595f, 0.287736595f, 0.287736595f, 0.287736595f},
      luisa::float4{-0.466513693f, 0.726881742f, -0.503991902f, 0.244151756f},
      luisa::float4{0.244151756f, 0.244151756f, 0.244151756f, 0.0f},
      luisa::float4{0.287736595f, 0.287736595f, 0.287736595f, 0.287736595f},
      luisa::float4{0.765341878f, -0.286503345f, 0.576339781f, 0.244151756f},
      luisa::float4{0.240954846f, 0.222976729f, 0.219377518f, 1.0f},
      luisa::float4{1.0f, 1.0f, 0.269455016f, 0.169756666f},
      luisa::float4{0.285112411f, 0.261658818f, 0.256963402f, 0.287736595f},
      luisa::float4{-0.466513693f, 0.726881742f, -0.503991902f, 0.244151756f},
      luisa::float4{0.230091318f, 0.212113202f, 0.208513990f, 1.0f},
      luisa::float4{0.259271771f, 0.235818163f, 0.231122762f, 0.287736595f},
      luisa::float4{-0.349890679f, 0.149953157f, -0.924711108f, 1.0e6f},
      luisa::float4{1.0e6f, 1.0e6f, 1.0e6f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.287736595f, 0.287736595f, 0.287736595f, 0.287736595f},
      luisa::float4{0.0f}};
  constexpr std::array expected_labels{
      closure::label_reflect | closure::label_diffuse,
      closure::label_transmit | closure::label_diffuse,
      closure::label_reflect | closure::label_diffuse,
      closure::label_transmit | closure::label_diffuse,
      closure::label_transmit | closure::label_transparent,
      closure::label_reflect | closure::label_diffuse};

  auto valid = true;
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    valid &= near(actual[index], expected[index]);
  }
  const auto exactly_zero = [](luisa::float4 value) noexcept {
    return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f &&
           value.w == 0.0f;
  };
  valid &= exactly_zero(actual[15u]) && exactly_zero(actual[16u]) &&
           exactly_zero(actual[17u]) && exactly_zero(actual[19u]);
  valid &= actual[13u].w == 1.0e6f && actual[14u].x == 1.0e6f &&
           actual[14u].y == 1.0e6f && actual[14u].z == 1.0e6f;
  valid &= actual[7u].w == 1.0f && actual[8u].x == 1.0f &&
           actual[8u].y == 1.0f && actual[11u].w == 1.0f;
  valid &= actual_labels == expected_labels;
  if (!valid) {
    std::cerr << "Cycles simple BSDF scattering mismatch on " << backend
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
