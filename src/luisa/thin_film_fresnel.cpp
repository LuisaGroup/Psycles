#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

struct ComplexFloat {
  Float real;
  Float imaginary;
};

struct PolarizedDielectricFresnel {
  Float2 reflectance;
  Float cosine_transmitted;
  Float2 reflection_phase_cosine;
  Bool total_internal_reflection;
};

struct PolarizedConductorFresnel {
  Float reflection_s;
  Float reflection_p;
  ComplexFloat phase_s;
  ComplexFloat phase_p;
};

[[nodiscard]] ComplexFloat complex_multiply(const ComplexFloat &lhs,
                                            const ComplexFloat &rhs) noexcept {
  return {.real = lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
          .imaginary = lhs.real * rhs.imaginary + lhs.imaginary * rhs.real};
}

[[nodiscard]] ComplexFloat complex_scale(const ComplexFloat &value,
                                         Float scale) noexcept {
  return {.real = value.real * scale, .imaginary = value.imaginary * scale};
}

[[nodiscard]] Float square(Float value) noexcept { return value * value; }

[[nodiscard]] Float fresnel_f82_scalar(Float cosine, Float f0,
                                       Float b) noexcept {
  const auto s = clamp(1.0f - cosine, 0.0f, 1.0f);
  const auto s2 = s * s;
  const auto s5 = s2 * s2 * s;
  return clamp(lerp(f0, 1.0f, s5) - b * cosine * s5 * s, 0.0f, 1.0f);
}

[[nodiscard]] Float fresnel_f82_b_scalar(Float f0, Float f82) noexcept {
  constexpr float f = 6.0f / 7.0f;
  constexpr float f5 = f * f * f * f * f;
  return (7.0f / (f5 * f)) * (lerp(f0, 1.0f, f5) - f82);
}

[[nodiscard]] PolarizedDielectricFresnel
fresnel_dielectric_polarized(Float cosine_incoming, Float eta) noexcept {
  const auto eta_cosine_transmitted_squared =
      eta * eta - (1.0f - cosine_incoming * cosine_incoming);
  const auto total_internal_reflection = eta_cosine_transmitted_squared <= 0.0f;
  const auto cosine_i = abs(cosine_incoming);
  const auto cosine_t =
      -sqrt(max(eta_cosine_transmitted_squared, 0.0f)) / max(eta, 1.0e-20f);
  const auto denominator_s = cosine_i - eta * cosine_t;
  const auto denominator_p = eta * cosine_i - cosine_t;
  const auto amplitude_s = (cosine_i + eta * cosine_t) /
                           select(1.0f, denominator_s, denominator_s != 0.0f);
  const auto amplitude_p = (cosine_t + eta * cosine_i) /
                           select(1.0f, denominator_p, denominator_p != 0.0f);
  const auto regular_reflectance =
      make_float2(amplitude_s * amplitude_s, amplitude_p * amplitude_p);
  const auto regular_phase =
      make_float2(select(-1.0f, 1.0f, amplitude_s >= 0.0f),
                  select(-1.0f, 1.0f, amplitude_p >= 0.0f));
  return {
      .reflectance = select(regular_reflectance, make_float2(1.0f),
                            total_internal_reflection),
      .cosine_transmitted = select(cosine_t, 0.0f, total_internal_reflection),
      .reflection_phase_cosine =
          select(regular_phase, make_float2(1.0f), total_internal_reflection),
      .total_internal_reflection = total_internal_reflection};
}

[[nodiscard]] PolarizedConductorFresnel
fresnel_f82_conductor_polarized(Float cosine_incoming, Float ambient_ior,
                                Float conductor_n, Float conductor_k,
                                Float f82) noexcept {
  const auto eta1_squared = square(ambient_ior);
  const auto eta2_squared = square(conductor_n);
  const auto k2_squared = square(conductor_k);
  const auto two_eta2_k2 = 2.0f * conductor_n * conductor_k;
  const auto t1 = eta2_squared - k2_squared -
                  eta1_squared * (1.0f - square(cosine_incoming));
  const auto t2 = sqrt(square(t1) + square(two_eta2_k2));
  const auto u_squared = max(0.5f * (t2 + t1), 0.0f);
  const auto v_squared = max(0.5f * (t2 - t1), 0.0f);
  const auto u = sqrt(u_squared);
  const auto v = sqrt(v_squared);

  const auto relative_n = conductor_n / max(ambient_ior, 1.0e-20f);
  const auto relative_k_squared =
      square(conductor_k / max(ambient_ior, 1.0e-20f));
  const auto f0 = (square(relative_n - 1.0f) + relative_k_squared) /
                  max(square(relative_n + 1.0f) + relative_k_squared, 1.0e-20f);
  const auto reflectance =
      fresnel_f82_scalar(cosine_incoming, f0, fresnel_f82_b_scalar(f0, f82));

  const auto phase_s_real =
      -u_squared - v_squared + square(ambient_ior * cosine_incoming);
  const auto phase_s_imaginary = -2.0f * ambient_ior * cosine_incoming * v;
  const auto phase_s_magnitude =
      sqrt(square(phase_s_real) + square(phase_s_imaginary));

  const auto phase_p_real =
      square((eta2_squared + k2_squared) * cosine_incoming) -
      eta1_squared * (u_squared + v_squared);
  const auto phase_p_imaginary =
      2.0f * ambient_ior * cosine_incoming *
      (two_eta2_k2 * u - (eta2_squared - k2_squared) * v);
  const auto phase_p_magnitude =
      sqrt(square(phase_p_real) + square(phase_p_imaginary));

  return {
      .reflection_s = reflectance,
      .reflection_p = reflectance,
      .phase_s = {.real = select(
                      1.0f, phase_s_real / max(phase_s_magnitude, 1.0e-20f),
                      phase_s_magnitude != 0.0f),
                  .imaginary = select(0.0f,
                                      phase_s_imaginary /
                                          max(phase_s_magnitude, 1.0e-20f),
                                      phase_s_magnitude != 0.0f)},
      .phase_p = {
          .real = select(1.0f, phase_p_real / max(phase_p_magnitude, 1.0e-20f),
                         phase_p_magnitude != 0.0f),
          .imaginary =
              select(0.0f, phase_p_imaginary / max(phase_p_magnitude, 1.0e-20f),
                     phase_p_magnitude != 0.0f)}};
}

template <std::uint32_t Channel>
[[nodiscard]] ComplexFloat
lookup_sensitivity(const ShaderServices &services,
                   Float optical_path_difference) noexcept {
  static_assert(Channel < 3u);
  constexpr float two_pi = 6.28318530717958647692f;
  constexpr auto size = cycles45_tables::thin_film_table_size;
  const auto coordinate = two_pi * optical_path_difference / 60000.0f;
  return {.real = cycles_table_1d(
              services, coordinate,
              UInt{cycles45_tables::thin_film_offset + Channel * size}, size),
          .imaginary = cycles_table_1d(
              services, coordinate,
              UInt{cycles45_tables::thin_film_offset + (Channel + 3u) * size},
              size)};
}

template <std::uint32_t Channel>
[[nodiscard]] Float airy_summation(const ShaderServices &services,
                                   Float reflection_12, Float reflection_23,
                                   Float optical_path_difference,
                                   ComplexFloat phase) noexcept {
  const auto transmission_121 = 1.0f - reflection_12;
  const auto reflection_123 = reflection_12 * reflection_23;
  const auto amplitude_123 = sqrt(max(reflection_123, 0.0f));
  const auto series = square(transmission_121) * reflection_23 /
                      max(1.0f - reflection_123, 1.0e-20f);
  auto accumulator = phase;
  Float result = series + reflection_12;
  Float coefficient = series - transmission_121;
  // This is a fixed third-order Fourier approximation, not a transport or
  // scene-depth loop. The host loop deliberately records exactly the three
  // mathematical orders used by Cycles 5.2.
  for (auto order = 1u; order < 4u; ++order) {
    coefficient *= amplitude_123;
    const auto sensitivity = lookup_sensitivity<Channel>(
        services, static_cast<float>(order) * optical_path_difference);
    result += coefficient * 2.0f *
              (accumulator.real * sensitivity.real +
               accumulator.imaginary * sensitivity.imaginary);
    accumulator = complex_multiply(accumulator, phase);
  }
  return result;
}

[[nodiscard]] Float effective_film_ior(Float thickness, Float authored_ior,
                                       Float ambient_ior) noexcept {
  const auto t = clamp(thickness, 0.0f, 1.0f);
  const auto smooth = t * t * (3.0f - 2.0f * t);
  return select(authored_ior, lerp(ambient_ior, authored_ior, smooth),
                thickness < 1.0f);
}

template <std::uint32_t Channel, bool Conductive>
[[nodiscard]] Float
iridescence_channel(const ShaderServices &services, Float thickness,
                    Float authored_film_ior, Float ambient_ior,
                    Float substrate_n, Float substrate_k, Float f82,
                    Float cosine_incoming, Float &cosine_transmitted) noexcept {
  const auto film_ior =
      effective_film_ior(thickness, authored_film_ior, ambient_ior);
  const auto top = fresnel_dielectric_polarized(
      cosine_incoming, film_ior / max(ambient_ior, 1.0e-20f));
  cosine_transmitted = 0.0f;
  Float result = 1.0f;
  $if(!top.total_internal_reflection) {
    Float reflection_23_s;
    Float reflection_23_p;
    ComplexFloat phase_23_s;
    ComplexFloat phase_23_p;
    Bool bottom_total_internal_reflection = false;
    if constexpr (Conductive) {
      const auto bottom = fresnel_f82_conductor_polarized(
          -top.cosine_transmitted, film_ior, substrate_n, substrate_k, f82);
      reflection_23_s = bottom.reflection_s;
      reflection_23_p = bottom.reflection_p;
      phase_23_s = bottom.phase_s;
      phase_23_p = bottom.phase_p;
    } else {
      const auto bottom = fresnel_dielectric_polarized(
          -top.cosine_transmitted, substrate_n / max(film_ior, 1.0e-20f));
      reflection_23_s = bottom.reflectance.x;
      reflection_23_p = bottom.reflectance.y;
      phase_23_s = {.real = bottom.reflection_phase_cosine.x,
                    .imaginary = 0.0f};
      phase_23_p = {.real = bottom.reflection_phase_cosine.y,
                    .imaginary = 0.0f};
      bottom_total_internal_reflection = bottom.total_internal_reflection;
      cosine_transmitted = bottom.cosine_transmitted;
    }
    $if(!bottom_total_internal_reflection) {
      const auto optical_path_difference =
          -2.0f * film_ior * thickness * top.cosine_transmitted;
      const auto phase_s =
          complex_scale(phase_23_s, -top.reflection_phase_cosine.x);
      const auto reflection_s =
          airy_summation<Channel>(services, top.reflectance.x, reflection_23_s,
                                  optical_path_difference, phase_s);
      const auto phase_p =
          complex_scale(phase_23_p, -top.reflection_phase_cosine.y);
      const auto reflection_p =
          airy_summation<Channel>(services, top.reflectance.y, reflection_23_p,
                                  optical_path_difference, phase_p);
      result = clamp(0.5f * (reflection_s + reflection_p), 0.0f, 1.0f);
    };
  };
  return result;
}

template <std::uint32_t Channel>
[[nodiscard]] Float f82_channel(const ShaderServices &services, Float thickness,
                                Float film_ior, Float f0, Float b,
                                Float cosine_incoming) noexcept {
  const auto r = min(f0, 0.999f);
  const auto g = fresnel_f82_scalar(1.0f / 7.0f, f0, b);
  const auto sqrt_r = sqrt(max(r, 0.0f));
  const auto n = lerp((1.0f + sqrt_r) / max(1.0f - sqrt_r, 1.0e-20f),
                      (1.0f - r) / max(1.0f + r, 1.0e-20f), g);
  const auto k = sqrt(
      max((r * square(n + 1.0f) - square(n - 1.0f)) / max(1.0f - r, 1.0e-20f),
          0.0f));
  Float unused_cosine;
  return iridescence_channel<Channel, true>(services, thickness, film_ior, 1.0f,
                                            n, k, g, cosine_incoming,
                                            unused_cosine);
}

} // namespace

ThinFilmDielectricFresnel
thin_film_dielectric_fresnel(const ShaderServices &services, Float thickness,
                             Float film_ior, Float substrate_ior, Float3 f0,
                             Float cosine_incoming) noexcept {
  Float cosine_transmitted;
  const auto red = iridescence_channel<0u, false>(
      services, thickness, film_ior, 1.0f, substrate_ior, 0.0f, -1.0f,
      cosine_incoming, cosine_transmitted);
  Float ignored_cosine;
  const auto green = iridescence_channel<1u, false>(
      services, thickness, film_ior, 1.0f, substrate_ior, 0.0f, -1.0f,
      cosine_incoming, ignored_cosine);
  const auto blue = iridescence_channel<2u, false>(
      services, thickness, film_ior, 1.0f, substrate_ior, 0.0f, -1.0f,
      cosine_incoming, ignored_cosine);
  Float3 reflectance = make_float3(red, green, blue);
  const auto real_f0 = f0_from_ior(substrate_ior);
  $if((real_f0 > 1.0e-5f) & !all(reflectance == make_float3(1.0f))) {
    const auto interpolation =
        clamp((reflectance - make_float3(1.0f)) / (real_f0 - 1.0f),
              make_float3(0.0f), make_float3(1.0f));
    const auto factor = f0 / real_f0;
    reflectance *= lerp(make_float3(1.0f), factor, interpolation);
  };
  return {.reflectance = reflectance, .cosine_transmitted = cosine_transmitted};
}

Float3 thin_film_f82_fresnel(const ShaderServices &services, Float thickness,
                             Float film_ior, Float3 f0, Float3 b,
                             Float cosine_incoming) noexcept {
  return make_float3(f82_channel<0u>(services, thickness, film_ior, f0.x, b.x,
                                     cosine_incoming),
                     f82_channel<1u>(services, thickness, film_ior, f0.y, b.y,
                                     cosine_incoming),
                     f82_channel<2u>(services, thickness, film_ior, f0.z, b.z,
                                     cosine_incoming));
}

} // namespace psycles::luisa_backend::detail
