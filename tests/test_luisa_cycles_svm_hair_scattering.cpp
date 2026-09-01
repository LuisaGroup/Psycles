#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_fast_math.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_hair.h"

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
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace hair = psycles::luisa_backend::cycles_svm::detail;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace fast_math = psycles::luisa_backend::cycles_fast_math;

inline constexpr std::uint32_t output_count = 13u;
inline constexpr std::uint32_t meta_count = 2u;

[[nodiscard]] auto scattering_kernel() {
  return Kernel1D<Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[](BufferFloat4 output,
                                            BufferUInt meta) noexcept {
    const device_svm::HairClosure reflection{
        .common = {.weight = make_float3(0.8f),
                   .type = closure::type_hair_reflection,
                   .sample_weight = 0.8f,
                   .N = make_float3(0.0f, 0.0f, 1.0f)},
        .param = {.T = make_float3(0.0f, 1.0f, 0.0f),
                  .roughness1 = 0.1f,
                  .roughness2 = 1.0f,
                  .offset = 0.0f}};
    const device_svm::HairClosure transmission{
        .common = {.weight = make_float3(0.83f, 0.17f, 0.52f),
                   .type = closure::type_hair_transmission,
                   .sample_weight = 0.5066666603088379f,
                   .N = make_float3(0.0f, 0.0f, 1.0f)},
        .param = {.T = make_float3(0.6f, 0.8f, 0.0f),
                  .roughness1 = 0.001f,
                  .roughness2 = 1.0f,
                  .offset = -0.27f}};
    const auto incoming = make_float3(0.0f, 0.0f, 1.0f);
    const auto normal = make_float3(0.0f, 0.0f, 1.0f);
    const auto random = make_float2(0.9398114085197449f, 0.12724359333515167f);

    const auto reflection_sample =
        hair::bsdf_hair_reflection_sample(reflection, normal, incoming, random);
    const auto reflection_sample_eval = hair::bsdf_hair_reflection_eval(
        reflection, incoming, reflection_sample.wo);
    output.write(0u, make_float4(reflection_sample.wo, reflection_sample.pdf));
    output.write(1u,
                 make_float4(reflection_sample.value, reflection_sample.eta));
    output.write(2u, make_float4(reflection_sample.sampled_roughness,
                                 dot(reflection.common.N, reflection_sample.wo),
                                 reflection_sample_eval.pdf));
    output.write(
        3u,
        make_float4(reflection_sample.value * reflection.common.weight, 0.0f));

    const auto transmission_sample = hair::bsdf_hair_transmission_sample(
        transmission, normal, incoming, random);
    const auto transmission_sample_eval = hair::bsdf_hair_transmission_eval(
        transmission, incoming, transmission_sample.wo);
    output.write(4u,
                 make_float4(transmission_sample.wo, transmission_sample.pdf));
    output.write(
        5u, make_float4(transmission_sample.value, transmission_sample.eta));
    output.write(6u,
                 make_float4(transmission_sample.sampled_roughness,
                             dot(transmission.common.N, transmission_sample.wo),
                             transmission_sample_eval.pdf));
    output.write(
        7u, make_float4(transmission_sample.value * transmission.common.weight,
                        0.0f));

    const auto reflection_wrong_side = hair::bsdf_hair_reflection_eval(
        reflection, incoming, make_float3(0.0f, 0.0f, -1.0f));
    const auto transmission_wrong_side = hair::bsdf_hair_transmission_eval(
        transmission, incoming, make_float3(0.0f, 0.0f, 1.0f));
    const auto reflection_grazing = hair::bsdf_hair_reflection_eval(
        reflection, incoming, reflection.param.T);
    output.write(8u, make_float4(reflection_wrong_side.pdf,
                                 transmission_wrong_side.pdf,
                                 reflection_grazing.pdf, 0.0f));

    output.write(9u, make_float4(fast_math::arc_tangent2(0.0f, -0.0f),
                                 fast_math::arc_tangent2(-0.0f, -0.0f),
                                 fast_math::arc_cosine(2.0f),
                                 fast_math::arc_cosine(-2.0f)));
    const auto reflection_direct = hair::bsdf_hair_reflection_eval(
        reflection, incoming, make_float3(0.0f, 0.0f, 1.0f));
    output.write(10u,
                 make_float4(reflection_direct.value * reflection.common.weight,
                             reflection_direct.pdf));
    const auto transmission_direct = hair::bsdf_hair_transmission_eval(
        transmission, incoming,
        make_float3(0.0f, 8.742277657347586e-08f, -1.0f));
    output.write(
        11u, make_float4(transmission_direct.value * transmission.common.weight,
                         transmission_direct.pdf));
    /* Cycles' `(f < 1) ? crushed : 1` partition intentionally maps a
     * NaN fast-acos input to zero. This locks the predicate itself rather
     * than only its ordinary finite interval. */
    output.write(12u, make_float4(fast_math::arc_cosine(as<float>(0x7fc00000u)),
                                  0.0f, 0.0f, 0.0f));
    meta.write(0u, reflection_sample.label);
    meta.write(1u, transmission_sample.label);
  }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 8.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 8.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output = device.create_buffer<luisa::float4>(output_count);
  auto meta = device.create_buffer<std::uint32_t>(meta_count);
  const auto shader = device.compile(
      scattering_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  std::array<luisa::float4, output_count> actual{};
  std::array<std::uint32_t, meta_count> actual_meta{};
  stream << shader(output, meta).dispatch(1u) << output.copy_to(actual.data())
         << meta.copy_to(actual_meta.data()) << synchronize();

  /* Event zero of each value set is a direct external Cycles 5.2.1 path
   * trace oracle. Tolerances intentionally cover ordinary backend float
   * contraction; they do not weaken any branch, label, or support check. */
  auto valid =
      near(actual[0u].xyz(),
           luisa::float3{0.7938507795333862f, 0.6015447974205017f,
                         -0.08913364261388779f}) &&
      near(actual[0u].w, 0.03163303807377815f) &&
      near(actual[1u].xyz(), luisa::float3{0.03163303807377815f}) &&
      near(actual[1u].w, 1.0f) && near(actual[2u].x, 0.1f) &&
      near(actual[2u].y, 1.0f) && actual[2u].z < 0.0f &&
      near(actual[2u].w, 0.0f) &&
      near(actual[3u].xyz(), luisa::float3{0.02530643157660961f}) &&
      actual_meta[0u] == (closure::label_reflect | closure::label_glossy);

  valid &= near(actual[4u].xyz(),
                luisa::float3{0.2497742772102356f, -0.8188082575798035f,
                              -0.5168809294700623f}) &&
           near(actual[4u].w, 1.7582085132598877f) &&
           near(actual[5u].xyz(), luisa::float3{1.7582085132598877f}) &&
           near(actual[5u].w, 1.0f) && near(actual[6u].x, 0.001f) &&
           near(actual[6u].y, 1.0f) && actual[6u].z < 0.0f &&
           std::isfinite(actual[6u].w) && actual[6u].w > 0.0f &&
           near(actual[7u].xyz(),
                luisa::float3{1.4593130350112915f, 0.2988954484462738f,
                              0.9142683744430542f}) &&
           actual_meta[1u] == (closure::label_transmit | closure::label_glossy);

  valid &= near(actual[8u].xyz(), luisa::float3{0.0f});
  valid &= near(actual[9u].x, 3.14159265358979323846f) &&
           near(actual[9u].y, -3.14159265358979323846f) &&
           near(actual[9u].z, 0.0f) &&
           near(actual[9u].w, 3.14159265358979323846f);
  valid &= near(actual[10u].xyz(), luisa::float3{0.34622371196746826f}) &&
           actual[10u].w > 0.0f &&
           near(actual[11u].xyz(),
                luisa::float3{0.0009033323149196804f, 0.00018501987506169826f,
                              0.0005659431335516274f}) &&
           actual[11u].w > 0.0f;
  valid &= near(actual[12u].x, 0.0f);
  if (!valid) {
    std::cerr << "Cycles Hair scattering mismatch on " << backend << '\n';
    for (auto index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": (" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << ")\n";
    }
    std::cerr << "labels: " << actual_meta[0u] << ", " << actual_meta[1u]
              << '\n';
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
