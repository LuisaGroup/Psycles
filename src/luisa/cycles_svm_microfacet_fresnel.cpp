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

[[nodiscard]] Float4 square(Expr<luisa::float4> value) noexcept {
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

void apply_lobe_mask_spectral(
    Expr<std::uint32_t> type,
    MicrofacetFresnelSpectralEvaluation &result) noexcept {
  result.reflectance *=
      select(make_float4(0.0f), make_float4(1.0f), has_reflection(type));
  result.transmittance *=
      select(make_float4(0.0f), make_float4(1.0f), has_transmission(type));
}

struct DielectricFresnelSpectral {
  Float4 reflectance;
  Float4 cosine_transmitted;
};

[[nodiscard]] DielectricFresnelSpectral fresnel_dielectric_spectral(
    Expr<float> cosine_incoming, Expr<luisa::float4> eta_lambda) noexcept {
  const Float cos_i = abs(cosine_incoming);
  const Float sin2_i = 1.0f - square(cosine_incoming);
  const Float4 eta_cos_t_sq = eta_lambda * eta_lambda - make_float4(sin2_i);
  const auto valid = eta_cos_t_sq > 0.0f;
  const Float4 cos_t =
      -sqrt(max(eta_cos_t_sq, make_float4(0.0f))) /
      max(eta_lambda, make_float4(1.0e-6f));
  const Float4 r_s = (make_float4(cos_i) + eta_lambda * cos_t) /
                     (make_float4(cos_i) - eta_lambda * cos_t);
  const Float4 r_p = (cos_t + eta_lambda * make_float4(cos_i)) /
                     (eta_lambda * make_float4(cos_i) - cos_t);
  const Float4 refl = 0.5f * (r_s * r_s + r_p * r_p);
  return {.reflectance = select(make_float4(1.0f), refl, valid),
          .cosine_transmitted = select(make_float4(0.0f), cos_t, valid)};
}

[[nodiscard]] MicrofacetFresnelSpectralEvaluation
generalized_schlick_fresnel_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<float> cosine_incoming, Expr<luisa::float4> eta_lambda,
    const cycles_spectrum::SpectralContext &spectral, Expr<bool> is_dispersive,
    bool may_transmit) noexcept {
  const auto fresnel = closure.load_generalized_schlick();
  const auto &tables = *spectral.tables;
  const auto &lambdas = spectral.wavelengths.lambdas;
  MicrofacetFresnelSpectralEvaluation result{
      .reflectance = make_float4(0.0f),
      .transmittance = make_float4(0.0f),
      .cosine_transmitted = make_float4(0.0f)};
  const auto usage = kernel_globals.closure_usage();
  const auto refl_tint_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
      tables, fresnel.reflection_tint, lambdas);
  const auto trans_tint_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
      tables, fresnel.transmission_tint, lambdas);
  const auto f90_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
      tables, fresnel.f90, lambdas);
  const Float real_f0_scalar = f0_from_ior(closure.param.ior);
  const Float4 real_f0_lambda =
      square((eta_lambda - make_float4(1.0f)) /
             max(eta_lambda + make_float4(1.0f), make_float4(1.0e-6f)));
  const Float4 f0_base_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
      tables, fresnel.f0, lambdas);
  const Float4 f0_disp_spec = select(
      f0_base_spec,
      clamp(real_f0_lambda * (f0_base_spec / max(real_f0_scalar, 1.0e-6f)),
            make_float4(0.0f), make_float4(1.0f)),
      is_dispersive);

  const auto without_thin_film = [&] {
    const auto negative_exponent_path = [&] {
      const auto dielectric =
          fresnel_dielectric_spectral(cosine_incoming, eta_lambda);
      const auto interpolation = clamp(
          (dielectric.reflectance - real_f0_lambda) /
              max(make_float4(1.0f) - real_f0_lambda, make_float4(1.0e-6f)),
          make_float4(0.0f), make_float4(1.0f));
      const auto reflected = lerp(f0_disp_spec, f90_spec, interpolation);
      result.reflectance = reflected * refl_tint_spec;
      if (may_transmit) {
        result.transmittance =
            (make_float4(1.0f) - reflected) * trans_tint_spec;
        result.cosine_transmitted = dielectric.cosine_transmitted;
      }
    };
    if (!usage.has_fresnel(
            compiler::cycles_svm::ClosureUsage::fresnel_positive_exponent)) {
      negative_exponent_path();
    } else {
      $if(fresnel.exponent < 0.0f) { negative_exponent_path(); }
      $else {
        const Float sin2_i = 1.0f - square(cosine_incoming);
        const Float4 cos_t_sq =
            make_float4(1.0f) -
            make_float4(sin2_i) /
                max(eta_lambda * eta_lambda, make_float4(1.0e-6f));
        const auto valid = cos_t_sq > 0.0f;
        const Float4 cos_t = sqrt(max(cos_t_sq, make_float4(0.0f)));
        const Float4 fresnel_angle = select(make_float4(cosine_incoming), cos_t,
                                            eta_lambda < make_float4(1.0f));
        const Float4 interpolation = make_float4(
            pow(clamp(1.0f - fresnel_angle.x, 0.0f, 1.0f), fresnel.exponent),
            pow(clamp(1.0f - fresnel_angle.y, 0.0f, 1.0f), fresnel.exponent),
            pow(clamp(1.0f - fresnel_angle.z, 0.0f, 1.0f), fresnel.exponent),
            pow(clamp(1.0f - fresnel_angle.w, 0.0f, 1.0f), fresnel.exponent));
        const Float4 reflected = lerp(f0_disp_spec, f90_spec, interpolation);
        result.reflectance = select(make_float4(1.0f), reflected, valid) *
                             refl_tint_spec;
        if (may_transmit) {
          result.cosine_transmitted = select(make_float4(0.0f), cos_t, valid);
          result.transmittance =
              select(make_float4(0.0f),
                     (make_float4(1.0f) - reflected) * trans_tint_spec, valid);
        }
      };
    }
  };
  if (kernel_globals.may_have_thin_film()) {
    $if(fresnel.thin_film.thickness >
        thin_film::thin_film_thickness_cutoff) {
      const auto film_ior_lambda =
          select(make_float4(fresnel.thin_film.ior),
                 fresnel.thin_film.ior * eta_lambda /
                     max(closure.param.ior, 1.0e-6f),
                 is_dispersive & (closure.param.ior < 1.0f));
      // When backfacing on dispersive media, thin_film.ior was scaled by
      // 1 / ior_d; scale smoothly for dispersion only through substrate_ior.
      static_cast<void>(film_ior_lambda);
      const auto film = thin_film::thin_film_dielectric_fresnel_spectral(
          fresnel.thin_film.thickness, fresnel.thin_film.ior, eta_lambda,
          f0_disp_spec, cosine_incoming, lambdas);
      result.reflectance = film.reflectance * refl_tint_spec;
      if (may_transmit) {
        result.transmittance =
            (make_float4(1.0f) - film.reflectance) * trans_tint_spec;
        result.cosine_transmitted = film.cosine_transmitted;
      }
    }
    $else { without_thin_film(); };
  } else {
    without_thin_film();
  }
  return result;
}

[[nodiscard]] MicrofacetFresnelSpectralEvaluation conductor_fresnel_spectral(
    const KernelGlobals &kernel_globals, const FresnelConductor &fresnel,
    Expr<float> cosine_incoming,
    const cycles_spectrum::SpectralContext &spectral) noexcept {
  const auto &tables = *spectral.tables;
  const auto &lambdas = spectral.wavelengths.lambdas;
  MicrofacetFresnelSpectralEvaluation result{
      .reflectance = make_float4(0.0f),
      .transmittance = make_float4(0.0f),
      .cosine_transmitted = make_float4(0.0f)};
  if (kernel_globals.may_have_thin_film()) {
    $if(fresnel.thin_film.thickness > thin_film::thin_film_thickness_cutoff) {
      const Float4 sub_ior = cycles_spectrum::rgb_to_spectrum_reflectance(
          tables, fresnel.ior, lambdas);
      const Float4 sub_ext = cycles_spectrum::rgb_to_spectrum_reflectance(
          tables, fresnel.extinction, lambdas);
      result.reflectance = thin_film::thin_film_conductor_fresnel_spectral(
          fresnel.thin_film.thickness, fresnel.thin_film.ior, sub_ior, sub_ext,
          cosine_incoming, lambdas);
    }
    $else {
      const Float3 refl_rgb = fresnel_conductor(
          cosine_incoming, fresnel.ior, fresnel.extinction);
      result.reflectance = cycles_spectrum::rgb_to_spectrum_reflectance(
          tables, refl_rgb, lambdas);
    };
  } else {
    const Float3 refl_rgb = fresnel_conductor(
        cosine_incoming, fresnel.ior, fresnel.extinction);
    result.reflectance = cycles_spectrum::rgb_to_spectrum_reflectance(
        tables, refl_rgb, lambdas);
  }
  return result;
}

[[nodiscard]] MicrofacetFresnelSpectralEvaluation f82_tint_fresnel_spectral(
    const KernelGlobals &kernel_globals, const FresnelF82Tint &fresnel,
    Expr<float> cosine_incoming,
    const cycles_spectrum::SpectralContext &spectral) noexcept {
  const auto &tables = *spectral.tables;
  const auto &lambdas = spectral.wavelengths.lambdas;
  MicrofacetFresnelSpectralEvaluation result{
      .reflectance = make_float4(0.0f),
      .transmittance = make_float4(0.0f),
      .cosine_transmitted = make_float4(0.0f)};
  if (kernel_globals.may_have_thin_film()) {
    $if(fresnel.thin_film.thickness > thin_film::thin_film_thickness_cutoff) {
      const Float4 f0_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
          tables, fresnel.f0, lambdas);
      const Float4 b_spec =
          cycles_spectrum::rgb_to_spectrum_reflectance(
              tables, max(fresnel.b, make_float3(0.0f)), lambdas) -
          cycles_spectrum::rgb_to_spectrum_reflectance(
              tables, max(-fresnel.b, make_float3(0.0f)), lambdas);
      result.reflectance = thin_film::thin_film_f82_fresnel_spectral(
          fresnel.thin_film.thickness, fresnel.thin_film.ior, f0_spec, b_spec,
          cosine_incoming, lambdas);
    }
    $else {
      const Float3 refl_rgb = fresnel_f82(cosine_incoming, fresnel.f0, fresnel.b);
      result.reflectance = cycles_spectrum::rgb_to_spectrum_reflectance(
          tables, refl_rgb, lambdas);
    };
  } else {
    const Float3 refl_rgb = fresnel_f82(cosine_incoming, fresnel.f0, fresnel.b);
    result.reflectance = cycles_spectrum::rgb_to_spectrum_reflectance(
        tables, refl_rgb, lambdas);
  }
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

MicrofacetFresnelSpectralEvaluation microfacet_fresnel_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<float> cosine_incoming, Expr<luisa::float4> eta_lambda,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<bool> is_dispersive, bool may_transmit) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  MicrofacetFresnelSpectralEvaluation result{
      .reflectance = make_float4(0.0f),
      .transmittance = make_float4(0.0f),
      .cosine_transmitted = make_float4(0.0f)};
  constexpr auto model_bits =
      Usage::fresnel_dielectric | Usage::fresnel_conductor |
      Usage::fresnel_f82_tint | Usage::fresnel_generalized_schlick |
      Usage::fresnel_none;
  const auto active_models = usage.microfacet_fresnel_mask & model_bits;
  const auto eval_dielectric = [&] {
    const auto dielectric =
        fresnel_dielectric_spectral(cosine_incoming, eta_lambda);
    result.reflectance = dielectric.reflectance;
    if (may_transmit) {
      result.transmittance = make_float4(1.0f) - dielectric.reflectance;
      result.cosine_transmitted = dielectric.cosine_transmitted;
    }
  };
  const auto eval_conductor = [&] {
    if (closure.storage != nullptr) {
      const auto typed =
          closure.storage->microfacet_conductor(closure.storage_index);
      result = conductor_fresnel_spectral(kernel_globals, typed.conductor,
                                          cosine_incoming, spectral);
    }
  };
  const auto eval_f82 = [&] {
    if (closure.storage != nullptr) {
      const auto typed =
          closure.storage->microfacet_f82_tint(closure.storage_index);
      result = f82_tint_fresnel_spectral(kernel_globals, typed.f82_tint,
                                         cosine_incoming, spectral);
    }
  };
  const auto eval_gs = [&] {
    result = generalized_schlick_fresnel_spectral(
        kernel_globals, closure, cosine_incoming, eta_lambda, spectral,
        is_dispersive, may_transmit);
  };
  const auto eval_none = [&] {
    result.reflectance = make_float4(1.0f);
    if (may_transmit) {
      $if(has_transmission(closure.common.type)) {
        const auto dielectric =
            fresnel_dielectric_spectral(cosine_incoming, eta_lambda);
        result.cosine_transmitted = dielectric.cosine_transmitted;
        result.transmittance = select(make_float4(1.0f), make_float4(0.0f),
                                      dielectric.reflectance == 1.0f);
      }
      $else { result.transmittance = make_float4(1.0f); };
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
    apply_lobe_mask_spectral(closure.common.type, result);
  }
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
