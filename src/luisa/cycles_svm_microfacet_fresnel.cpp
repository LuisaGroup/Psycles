/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_microfacet_fresnel.h"

#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_closure.h>

#include <bit>

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
    Expr<float> cosine_incoming, bool may_transmit) noexcept {
  const auto fresnel = closure.load_generalized_schlick();
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  const auto usage = kernel_globals.closure_usage();
  const auto without_thin_film = [&] {
    const auto negative_exponent_path = [&] {
      const auto dielectric =
          fresnel_dielectric(cosine_incoming, closure.param.ior);
      const auto real_f0 = f0_from_ior(closure.param.ior);
      const auto interpolation =
          clamp((dielectric.reflectance - real_f0) / (1.0f - real_f0),
                0.0f, 1.0f);
      const auto reflected = lerp(fresnel.f0, fresnel.f90, interpolation);
      result.reflectance = reflected * fresnel.reflection_tint;
      if (may_transmit) {
        result.transmittance =
            (make_float3(1.0f) - reflected) * fresnel.transmission_tint;
        result.cosine_transmitted = dielectric.cosine_transmitted;
      }
    };
    if (!usage.has_fresnel(
            compiler::cycles_svm::ClosureUsage::fresnel_positive_exponent)) {
      negative_exponent_path();
    } else {
      $if(fresnel.exponent < 0.0f) { negative_exponent_path(); }
      $else {
        const auto cosine_transmitted_squared =
            1.0f - (1.0f - square(cosine_incoming)) /
                       square(closure.param.ior);
        $if(cosine_transmitted_squared <= 0.0f) {
          result.reflectance = fresnel.reflection_tint;
          if (may_transmit) {
            result.transmittance = make_float3(0.0f);
          }
        }
        $else {
          const auto cos_t = sqrt(cosine_transmitted_squared);
          const auto fresnel_angle = select(
              cosine_incoming, cos_t, closure.param.ior < 1.0f);
          const auto interpolation =
              pow(1.0f - fresnel_angle, fresnel.exponent);
          const auto reflected = lerp(fresnel.f0, fresnel.f90, interpolation);
          result.reflectance = reflected * fresnel.reflection_tint;
          if (may_transmit) {
            result.cosine_transmitted = cos_t;
            result.transmittance =
                (make_float3(1.0f) - reflected) *
                fresnel.transmission_tint;
          }
        };
      };
    }
  };
  if (kernel_globals.may_have_thin_film()) {
    $if(fresnel.thin_film.thickness >
        thin_film::thin_film_thickness_cutoff) {
      const auto film = thin_film::thin_film_dielectric_fresnel(
          kernel_globals, fresnel.thin_film.thickness,
          fresnel.thin_film.ior, closure.param.ior, fresnel.f0,
          cosine_incoming);
      result.reflectance = film.reflectance * fresnel.reflection_tint;
      if (may_transmit) {
        result.transmittance =
            (make_float3(1.0f) - film.reflectance) *
            fresnel.transmission_tint;
        result.cosine_transmitted = film.cosine_transmitted;
      }
    }
    $else { without_thin_film(); };
  } else {
    without_thin_film();
  }
  return result;
}

[[nodiscard]] MicrofacetFresnelEvaluation conductor_fresnel(
    const KernelGlobals &kernel_globals, const FresnelConductor &fresnel,
    Expr<float> cosine_incoming) noexcept {
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  if (!kernel_globals.may_have_thin_film()) {
    result.reflectance = fresnel_conductor(
        cosine_incoming, fresnel.ior, fresnel.extinction);
    return result;
  }
  $if(fresnel.thin_film.thickness > thin_film::thin_film_thickness_cutoff) {
    result.reflectance = thin_film::thin_film_conductor_fresnel(
        kernel_globals, fresnel.thin_film.thickness,
        fresnel.thin_film.ior, fresnel.ior, fresnel.extinction,
        cosine_incoming);
  }
  $else {
    result.reflectance = fresnel_conductor(
        cosine_incoming, fresnel.ior, fresnel.extinction);
  };
  return result;
}

[[nodiscard]] MicrofacetFresnelEvaluation f82_tint_fresnel(
    const KernelGlobals &kernel_globals, const FresnelF82Tint &fresnel,
    Expr<float> cosine_incoming) noexcept {
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  if (!kernel_globals.may_have_thin_film()) {
    result.reflectance = fresnel_f82(cosine_incoming, fresnel.f0, fresnel.b);
    return result;
  }
  $if(fresnel.thin_film.thickness > thin_film::thin_film_thickness_cutoff) {
    result.reflectance = thin_film::thin_film_f82_fresnel(
        kernel_globals, fresnel.thin_film.thickness,
        fresnel.thin_film.ior, fresnel.f0, fresnel.b, cosine_incoming);
  }
  $else {
    result.reflectance = fresnel_f82(cosine_incoming, fresnel.f0, fresnel.b);
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
    Expr<float> cosine_incoming, bool may_transmit) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  MicrofacetFresnelEvaluation result{
      .reflectance = make_float3(0.0f),
      .transmittance = make_float3(0.0f),
      .cosine_transmitted = 0.0f};
  constexpr auto model_bits =
      Usage::fresnel_dielectric | Usage::fresnel_conductor |
      Usage::fresnel_f82_tint | Usage::fresnel_generalized_schlick |
      Usage::fresnel_none;
  const auto active_models = usage.microfacet_fresnel_mask & model_bits;
  const auto eval_dielectric = [&] {
    const auto dielectric =
        fresnel_dielectric(cosine_incoming, closure.param.ior);
    result.reflectance = make_float3(dielectric.reflectance);
    if (may_transmit) {
      result.transmittance = make_float3(1.0f - dielectric.reflectance);
      result.cosine_transmitted = dielectric.cosine_transmitted;
    }
  };
  const auto eval_conductor = [&] {
    if (closure.storage != nullptr) {
      const auto typed =
          closure.storage->microfacet_conductor(closure.storage_index);
      result =
          conductor_fresnel(kernel_globals, typed.conductor, cosine_incoming);
    }
  };
  const auto eval_f82 = [&] {
    if (closure.storage != nullptr) {
      const auto typed =
          closure.storage->microfacet_f82_tint(closure.storage_index);
      result = f82_tint_fresnel(kernel_globals, typed.f82_tint, cosine_incoming);
    }
  };
  const auto eval_gs = [&] {
    result = generalized_schlick_fresnel(
        kernel_globals, closure, cosine_incoming, may_transmit);
  };
  const auto eval_none = [&] {
    result.reflectance = make_float3(1.0f);
    if (may_transmit) {
      result.transmittance = make_float3(1.0f);
      $if(has_transmission(closure.common.type)) {
        const auto dielectric =
            fresnel_dielectric(cosine_incoming, closure.param.ior);
        result.cosine_transmitted = dielectric.cosine_transmitted;
        $if(dielectric.reflectance == 1.0f) {
          result.transmittance = make_float3(0.0f);
        };
      };
    }
  };

  if (std::has_single_bit(active_models)) {
    if (active_models == Usage::fresnel_dielectric) {
      eval_dielectric();
    } else if (active_models == Usage::fresnel_conductor) {
      eval_conductor();
    } else if (active_models == Usage::fresnel_f82_tint) {
      eval_f82();
    } else if (active_models == Usage::fresnel_generalized_schlick) {
      eval_gs();
    } else if (active_models == Usage::fresnel_none) {
      eval_none();
    }
  } else {
    $switch(closure.param.fresnel_type) {
      if (usage.has_fresnel(Usage::fresnel_dielectric)) {
        $case(static_cast<std::uint32_t>(MicrofacetFresnel::dielectric)) {
          eval_dielectric();
        };
      }
      if (usage.has_fresnel(Usage::fresnel_conductor)) {
        $case(static_cast<std::uint32_t>(MicrofacetFresnel::conductor)) {
          eval_conductor();
        };
      }
      if (usage.has_fresnel(Usage::fresnel_f82_tint)) {
        $case(static_cast<std::uint32_t>(MicrofacetFresnel::f82_tint)) {
          eval_f82();
        };
      }
      if (usage.has_fresnel(Usage::fresnel_generalized_schlick)) {
        $case(static_cast<std::uint32_t>(
            MicrofacetFresnel::generalized_schlick)) {
          eval_gs();
        };
      }
      if (usage.has_fresnel(Usage::fresnel_none)) {
        $case(static_cast<std::uint32_t>(MicrofacetFresnel::none)) {
          eval_none();
        };
      }
      $default{};
    };
  }
  if (may_transmit) {
    apply_lobe_mask(closure.common.type, result);
  }
  return result;
}

MicrofacetFresnelEvaluation microfacet_fresnel(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    Expr<float> cosine_incoming, bool may_transmit) noexcept {
  auto result = conductor_fresnel(kernel_globals, closure.conductor, cosine_incoming);
  if (may_transmit) {
    apply_lobe_mask(closure.common.type, result);
  }
  return result;
}

MicrofacetFresnelEvaluation microfacet_fresnel(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    Expr<float> cosine_incoming, bool may_transmit) noexcept {
  auto result = f82_tint_fresnel(kernel_globals, closure.f82_tint, cosine_incoming);
  if (may_transmit) {
    apply_lobe_mask(closure.common.type, result);
  }
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
