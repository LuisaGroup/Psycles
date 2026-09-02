/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_microfacet_fresnel.h"

#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;
namespace thin_film = ::psycles::luisa_backend::detail;

namespace {

[[nodiscard]] Float square(Expr<float> value) noexcept {
  return value * value;
}

[[nodiscard]] Float safe_divide(Expr<float> numerator,
                                Expr<float> denominator) noexcept {
  return select(0.0f, numerator / denominator, denominator != 0.0f);
}

[[nodiscard]] Float fresnel_conductor_channel(
    Expr<float> cosine_incoming, Expr<float> ior,
    Expr<float> extinction) noexcept {
  const auto ior_squared = square(ior);
  const auto extinction_squared = square(extinction);
  const auto two_ior_extinction = 2.0f * ior * extinction;
  const auto t1 = ior_squared - extinction_squared -
                  (1.0f - square(cosine_incoming));
  const auto t2 = sqrt(square(t1) + square(two_ior_extinction));
  const auto u_squared = max(0.5f * (t2 + t1), 0.0f);
  const auto v_squared = max(0.5f * (t2 - t1), 0.0f);
  const auto u = sqrt(u_squared);
  const auto v = sqrt(v_squared);

  const auto reflection_s = safe_divide(
      square(cosine_incoming - u) + v_squared,
      square(cosine_incoming + u) + v_squared);
  const auto t3 = (ior_squared - extinction_squared) * cosine_incoming;
  const auto t4 = two_ior_extinction * cosine_incoming;
  const auto reflection_p = safe_divide(
      square(t3 - u) + square(t4 - v),
      square(t3 + u) + square(t4 + v));
  return 0.5f * (reflection_s + reflection_p);
}

[[nodiscard]] Bool has_reflection(Expr<std::uint32_t> type) noexcept {
  return !closure_type::is_refraction_microfacet(type);
}

[[nodiscard]] Bool has_transmission(Expr<std::uint32_t> type) noexcept {
  const auto reflection = has_reflection(type);
  return closure_type::is_glass_microfacet(type) | !reflection;
}

void apply_lobe_mask(Expr<std::uint32_t> type,
                     MicrofacetFresnelEvaluation &result) noexcept {
  result.reflectance *= select(make_float3(0.0f), make_float3(1.0f),
                               has_reflection(type));
  result.transmittance *= select(make_float3(0.0f), make_float3(1.0f),
                                 has_transmission(type));
}

[[nodiscard]] MicrofacetFresnelEvaluation generalized_schlick_fresnel(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<float> cosine_incoming) noexcept {
  const auto &fresnel = closure.generalized_schlick;
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  $if(fresnel.thin_film.thickness >
      thin_film::thin_film_thickness_cutoff) {
    const auto film = thin_film::thin_film_dielectric_fresnel(
        kernel_globals, fresnel.thin_film.thickness,
        fresnel.thin_film.ior, closure.param.ior, fresnel.f0,
        cosine_incoming);
    result.reflectance = film.reflectance * fresnel.reflection_tint;
    result.transmittance =
        (make_float3(1.0f) - film.reflectance) *
        fresnel.transmission_tint;
    result.cosine_transmitted = film.cosine_transmitted;
  }
  $elif(fresnel.exponent < 0.0f) {
    const auto dielectric =
        fresnel_dielectric(cosine_incoming, closure.param.ior);
    const auto real_f0 = f0_from_ior(closure.param.ior);
    const auto interpolation =
        clamp((dielectric.reflectance - real_f0) / (1.0f - real_f0),
              0.0f, 1.0f);
    const auto reflected = lerp(fresnel.f0, fresnel.f90, interpolation);
    result.reflectance = reflected * fresnel.reflection_tint;
    result.transmittance =
        (make_float3(1.0f) - reflected) * fresnel.transmission_tint;
    result.cosine_transmitted = dielectric.cosine_transmitted;
  }
  $else {
    const auto cosine_transmitted_squared =
        1.0f - (1.0f - square(cosine_incoming)) /
                   square(closure.param.ior);
    $if(cosine_transmitted_squared <= 0.0f) {
      result.reflectance = fresnel.reflection_tint;
      result.transmittance = make_float3(0.0f);
    }
    $else {
      result.cosine_transmitted = sqrt(cosine_transmitted_squared);
      const auto fresnel_angle = select(
          cosine_incoming, result.cosine_transmitted,
          closure.param.ior < 1.0f);
      const auto interpolation =
          pow(1.0f - fresnel_angle, fresnel.exponent);
      const auto reflected = lerp(fresnel.f0, fresnel.f90, interpolation);
      result.reflectance = reflected * fresnel.reflection_tint;
      result.transmittance =
          (make_float3(1.0f) - reflected) *
          fresnel.transmission_tint;
    };
  };
  return result;
}

} // namespace

Float f0_from_ior(Expr<float> ior) noexcept {
  const auto ratio = (ior - 1.0f) / (ior + 1.0f);
  return ratio * ratio;
}

DielectricFresnel fresnel_dielectric(Expr<float> cosine_incoming,
                                     Expr<float> ior) noexcept {
  const auto eta_cosine_transmitted_squared =
      square(ior) - (1.0f - square(cosine_incoming));
  DielectricFresnel result{
      .reflectance = 1.0f,
      .cosine_transmitted = 0.0f};
  /* Preserve Cycles' negative TIR predicate. The regular divisions do not
   * enter the AST/control-flow region of the rejected domain. */
  $if(eta_cosine_transmitted_squared <= 0.0f) {}
  $else {
    const auto cosine_i = abs(cosine_incoming);
    const auto cosine_t =
        -sqrt(max(eta_cosine_transmitted_squared, 0.0f)) / ior;
    const auto reflection_s =
        (cosine_i + ior * cosine_t) / (cosine_i - ior * cosine_t);
    const auto reflection_p =
        (cosine_t + ior * cosine_i) / (ior * cosine_i - cosine_t);
    result.reflectance =
        0.5f * (square(reflection_s) + square(reflection_p));
    result.cosine_transmitted = cosine_t;
  };
  return result;
}

Float3 fresnel_conductor(Expr<float> cosine_incoming,
                         Expr<luisa::float3> ior,
                         Expr<luisa::float3> extinction) noexcept {
  return make_float3(
      fresnel_conductor_channel(cosine_incoming, ior.x, extinction.x),
      fresnel_conductor_channel(cosine_incoming, ior.y, extinction.y),
      fresnel_conductor_channel(cosine_incoming, ior.z, extinction.z));
}

Float3 fresnel_f82(Expr<float> cosine_incoming,
                   Expr<luisa::float3> f0,
                   Expr<luisa::float3> b) noexcept {
  const auto s = clamp(1.0f - cosine_incoming, 0.0f, 1.0f);
  const auto s2 = s * s;
  const auto s5 = s2 * s2 * s;
  return clamp(lerp(f0, make_float3(1.0f), s5) -
                   b * cosine_incoming * s5 * s,
               make_float3(0.0f), make_float3(1.0f));
}

MicrofacetFresnelEvaluation microfacet_fresnel(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<float> cosine_incoming) noexcept {
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(1.0f),
      .transmittance = make_float3(1.0f),
      .cosine_transmitted = 0.0f};
  $if(closure.param.fresnel_type ==
      static_cast<std::uint32_t>(MicrofacetFresnel::dielectric)) {
    const auto dielectric =
        fresnel_dielectric(cosine_incoming, closure.param.ior);
    result.reflectance = make_float3(dielectric.reflectance);
    result.transmittance = make_float3(1.0f - dielectric.reflectance);
    result.cosine_transmitted = dielectric.cosine_transmitted;
  }
  $elif(closure.param.fresnel_type ==
        static_cast<std::uint32_t>(MicrofacetFresnel::generalized_schlick)) {
    result = generalized_schlick_fresnel(kernel_globals, closure,
                                         cosine_incoming);
  }
  $elif(closure.param.fresnel_type ==
        static_cast<std::uint32_t>(MicrofacetFresnel::none)) {
    /* NONE is the exact pure-lobe path. */
    $if(has_transmission(closure.common.type)) {
      const auto dielectric =
          fresnel_dielectric(cosine_incoming, closure.param.ior);
      result.cosine_transmitted = dielectric.cosine_transmitted;
      $if(dielectric.reflectance == 1.0f) {
        result.transmittance = make_float3(0.0f);
      };
    };
  }
  $else {
    /* SVM retains no DIELECTRIC_TINT payload. CONDUCTOR and F82_TINT belong
     * to different typed union projections. Fail closed if a caller violates
     * that tag/payload invariant instead of interpreting foreign bytes as
     * the NONE variant. */
    result.reflectance = make_float3(0.0f);
    result.transmittance = make_float3(0.0f);
  };
  apply_lobe_mask(closure.common.type, result);
  return result;
}

MicrofacetFresnelEvaluation microfacet_fresnel(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    Expr<float> cosine_incoming) noexcept {
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  $if(closure.conductor.thin_film.thickness >
      thin_film::thin_film_thickness_cutoff) {
    result.reflectance = thin_film::thin_film_conductor_fresnel(
        kernel_globals, closure.conductor.thin_film.thickness,
        closure.conductor.thin_film.ior, closure.conductor.ior,
        closure.conductor.extinction, cosine_incoming);
  }
  $else {
    result.reflectance = fresnel_conductor(
        cosine_incoming, closure.conductor.ior,
        closure.conductor.extinction);
  };
  apply_lobe_mask(closure.common.type, result);
  return result;
}

MicrofacetFresnelEvaluation microfacet_fresnel(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    Expr<float> cosine_incoming) noexcept {
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  $if(closure.f82_tint.thin_film.thickness >
      thin_film::thin_film_thickness_cutoff) {
    result.reflectance = thin_film::thin_film_f82_fresnel(
        kernel_globals, closure.f82_tint.thin_film.thickness,
        closure.f82_tint.thin_film.ior, closure.f82_tint.f0,
        closure.f82_tint.b, cosine_incoming);
  }
  $else {
    result.reflectance = fresnel_f82(
        cosine_incoming, closure.f82_tint.f0, closure.f82_tint.b);
  };
  apply_lobe_mask(closure.common.type, result);
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
