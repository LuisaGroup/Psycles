/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_microfacet_scattering.h"

#include "cycles_svm_microfacet_fresnel.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;
namespace sample_mapping =
    ::psycles::luisa_backend::cycles_sample_mapping;

namespace {

enum class MicrofacetDistribution : std::uint32_t {
  beckmann,
  ggx,
};

[[nodiscard]] Float square(Expr<float> value) noexcept {
  return value * value;
}

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float safe_divide(Expr<float> numerator,
                                Expr<float> denominator) noexcept {
  return select(0.0f, numerator / denominator, denominator != 0.0f);
}

[[nodiscard]] Bool roughness_is_almost_specular(
    const MicrofacetParam &microfacet) noexcept {
  return microfacet.alpha_x * microfacet.alpha_y <=
         closure_type::microfacet_singular_alpha_product;
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] Float lambda_from_squared_alpha_tangent(
    Expr<float> squared_alpha_tangent) noexcept {
  if constexpr (Distribution == MicrofacetDistribution::ggx) {
    return 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
  } else {
    Float result;
    $if(squared_alpha_tangent < 0.39f) { result = 0.0f; }
    $else {
      const auto a = rsqrt(squared_alpha_tangent);
      result = ((0.396f * a - 1.259f) * a + 1.0f) /
               ((2.181f * a + 3.535f) * a);
    };
    return result;
  }
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] Float microfacet_lambda(Expr<float> alpha_squared,
                                      Expr<float> cosine) noexcept {
  return lambda_from_squared_alpha_tangent<Distribution>(
      alpha_squared * max(1.0f / square(cosine) - 1.0f, 0.0f));
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] Float microfacet_anisotropic_lambda(
    Expr<float> alpha_x, Expr<float> alpha_y,
    Expr<luisa::float3> direction) noexcept {
  const auto squared_alpha_tangent =
      (square(alpha_x * direction.x) +
       square(alpha_y * direction.y)) /
      square(direction.z);
  return lambda_from_squared_alpha_tangent<Distribution>(
      squared_alpha_tangent);
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] Float microfacet_distribution(Expr<float> alpha_squared,
                                            Expr<float> cosine_half) noexcept {
  const auto cosine_half_squared = min(square(cosine_half), 1.0f);
  const auto one_minus_cosine_half_squared =
      1.0f - cosine_half_squared;
  if constexpr (Distribution == MicrofacetDistribution::beckmann) {
    return 1.0f /
           (exp(one_minus_cosine_half_squared /
                (cosine_half_squared * alpha_squared)) *
            sample_mapping::pi * alpha_squared *
            square(cosine_half_squared));
  } else {
    return alpha_squared /
           (sample_mapping::pi *
            square(one_minus_cosine_half_squared +
                   alpha_squared * cosine_half_squared));
  }
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] Float microfacet_anisotropic_distribution(
    Expr<float> alpha_x, Expr<float> alpha_y,
    Expr<luisa::float3> half_vector) noexcept {
  const auto stretched =
      half_vector / make_float3(alpha_x, alpha_y, 1.0f);
  const auto cosine_half_squared = square(stretched.z);
  const auto alpha_squared = alpha_x * alpha_y;
  if constexpr (Distribution == MicrofacetDistribution::beckmann) {
    return exp(-(square(stretched.x) + square(stretched.y)) /
               cosine_half_squared) /
           (sample_mapping::pi * alpha_squared *
            square(cosine_half_squared));
  } else {
    return sample_mapping::inverse_pi /
           (alpha_squared * square(dot(stretched, stretched)));
  }
}

template<MicrofacetDistribution Distribution, typename Closure>
[[nodiscard]] BsdfEvaluation microfacet_eval(
    const KernelGlobals &kernel_globals, const Closure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto &microfacet = closure.param;
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  const auto cosine_outgoing = dot(normal, wo);
  const auto is_transmission = cosine_outgoing < 0.0f;
  const auto has_reflection =
      !closure_type::is_refraction_microfacet(closure.common.type);
  const auto has_transmission =
      closure_type::is_glass_microfacet(closure.common.type) |
      !has_reflection;
  const auto rejected =
      (cosine_incoming <= 0.0f) |
      roughness_is_almost_specular(microfacet) |
      (is_transmission & !has_transmission) |
      ((!is_transmission) & (!has_reflection));

  $if(!rejected) {
    Float3 half_vector = select(wi + wo,
                                -(microfacet.ior * wo + wi),
                                is_transmission);
    const auto inverse_half_length =
        safe_divide(1.0f, sqrt(dot(half_vector, half_vector)));
    half_vector *= inverse_half_length;
    const auto cosine_half_incoming = dot(half_vector, wi);
    const auto fresnel = microfacet_fresnel(
        kernel_globals, closure, cosine_half_incoming);
    const auto fresnel_is_zero =
        all(fresnel.reflectance == make_float3(0.0f)) &
        all(fresnel.transmittance == make_float3(0.0f));

    $if(!fresnel_is_zero) {
      const auto cosine_normal_half = dot(normal, half_vector);
      Float distribution;
      Float lambda_incoming;
      Float lambda_outgoing;
      $if((microfacet.alpha_x == microfacet.alpha_y) |
          is_transmission) {
        const auto alpha_squared =
            microfacet.alpha_x * microfacet.alpha_y;
        distribution = microfacet_distribution<Distribution>(
            alpha_squared, cosine_normal_half);
        lambda_incoming = microfacet_lambda<Distribution>(
            alpha_squared, cosine_incoming);
        lambda_outgoing = microfacet_lambda<Distribution>(
            alpha_squared, cosine_outgoing);
      }
      $else {
        const auto basis = sample_mapping::make_orthonormals_tangent(
            normal, microfacet.T);
        const auto local_half = make_float3(
            dot(basis.tangent, half_vector),
            dot(basis.bitangent, half_vector), cosine_normal_half);
        const auto local_incoming = make_float3(
            dot(basis.tangent, wi), dot(basis.bitangent, wi),
            cosine_incoming);
        const auto local_outgoing = make_float3(
            dot(basis.tangent, wo), dot(basis.bitangent, wo),
            cosine_outgoing);
        distribution = microfacet_anisotropic_distribution<Distribution>(
            microfacet.alpha_x, microfacet.alpha_y, local_half);
        lambda_incoming = microfacet_anisotropic_lambda<Distribution>(
            microfacet.alpha_x, microfacet.alpha_y, local_incoming);
        lambda_outgoing = microfacet_anisotropic_lambda<Distribution>(
            microfacet.alpha_x, microfacet.alpha_y, local_outgoing);
      };

      const auto common = distribution / cosine_incoming *
          select(0.25f,
                 square(microfacet.ior * inverse_half_length) *
                     abs(cosine_half_incoming * dot(half_vector, wo)),
                 is_transmission);
      const auto reflection_probability =
          average(fresnel.reflectance) /
          average(fresnel.reflectance + fresnel.transmittance);
      const auto lobe_probability =
          select(reflection_probability, 1.0f - reflection_probability,
                 is_transmission);
      result.pdf = common * lobe_probability /
                   (1.0f + lambda_incoming);
      result.value =
          select(fresnel.reflectance, fresnel.transmittance,
                 is_transmission) *
          common /
          (1.0f + lambda_outgoing + lambda_incoming);
    };
  };
  return result;
}

template<MicrofacetDistribution Distribution, typename Closure>
[[nodiscard]] BsdfSample microfacet_sample(
    const KernelGlobals &kernel_globals, const Closure &closure,
    Expr<luisa::float3> geometric_normal, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(0.0f),
                    .eta = 0.0f,
                    .label = closure_type::label_none};
  const auto &microfacet = closure.param;
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  $if(cosine_incoming > 0.0f) {
    const auto eta = microfacet.ior;
    const auto inverse_eta = safe_divide(1.0f, eta);
    Bool singular = roughness_is_almost_specular(microfacet);
    Float3 half_vector;
    Float3 local_half = make_float3(0.0f);
    Float3 local_incoming = make_float3(0.0f);
    $if(singular) { half_vector = normal; }
    $else {
      Float3 basis_x;
      Float3 basis_y;
      $if(microfacet.alpha_x == microfacet.alpha_y) {
        const auto basis = sample_mapping::make_orthonormals(normal);
        basis_x = basis.tangent;
        basis_y = basis.bitangent;
      }
      $else {
        const auto basis = sample_mapping::make_orthonormals_tangent(
            normal, microfacet.T);
        basis_x = basis.tangent;
        basis_y = basis.bitangent;
      };
      local_incoming = make_float3(dot(basis_x, wi), dot(basis_y, wi),
                                   cosine_incoming);
      if constexpr (Distribution == MicrofacetDistribution::ggx) {
        local_half = sample_mapping::sample_ggx_visible_normal_local(
            local_incoming, microfacet.alpha_x, microfacet.alpha_y,
            random.xy());
      } else {
        local_half = sample_mapping::sample_beckmann_visible_normal_local(
            local_incoming, microfacet.alpha_x, microfacet.alpha_y,
            random.xy());
      }
      half_vector = local_half.x * basis_x + local_half.y * basis_y +
                    local_half.z * normal;
    };
    const auto cosine_half_incoming = dot(half_vector, wi);
    const auto fresnel = microfacet_fresnel(
        kernel_globals, closure, cosine_half_incoming);
    const auto fresnel_is_zero =
        all(fresnel.reflectance == make_float3(0.0f)) &
        all(fresnel.transmittance == make_float3(0.0f));
    $if(!fresnel_is_zero) {
      const auto reflection_probability =
          average(fresnel.reflectance) /
          average(fresnel.reflectance + fresnel.transmittance);
      const auto refract = random.z >= reflection_probability;
      $if(refract) {
        result.wo =
            (inverse_eta * dot(half_vector, wi) +
             fresnel.cosine_transmitted) *
                half_vector -
            inverse_eta * wi;
      }
      $else {
        result.wo =
            2.0f * cosine_half_incoming * half_vector - wi;
      };

      const auto cosine_outgoing = dot(normal, result.wo);
      const auto cosine_geometric_outgoing =
          dot(geometric_normal, result.wo);
      const auto wrong_hemisphere =
          ((cosine_geometric_outgoing < 0.0f) != refract) |
          ((cosine_outgoing < 0.0f) != refract);
      $if(!wrong_hemisphere) {
        $if(refract) {
          result.value = fresnel.transmittance;
          result.pdf = 1.0f - reflection_probability;
          singular |= abs(eta - 1.0f) < 1.0e-4f;
        }
        $else {
          result.value = fresnel.reflectance;
          result.pdf = reflection_probability;
        };

        $if(singular) {
          result.pdf *= 1.0e6f;
          result.value *= 1.0e6f;
        }
        $else {
          Float distribution;
          Float lambda_incoming;
          Float lambda_outgoing;
          $if((microfacet.alpha_x == microfacet.alpha_y) | refract) {
            const auto alpha_squared =
                microfacet.alpha_x * microfacet.alpha_y;
            distribution = microfacet_distribution<Distribution>(
                alpha_squared, local_half.z);
            lambda_outgoing = microfacet_lambda<Distribution>(
                alpha_squared, cosine_outgoing);
            lambda_incoming = microfacet_lambda<Distribution>(
                alpha_squared, cosine_incoming);
          }
          $else {
            const auto local_outgoing =
                2.0f * cosine_half_incoming * local_half - local_incoming;
            distribution =
                microfacet_anisotropic_distribution<Distribution>(
                    microfacet.alpha_x, microfacet.alpha_y, local_half);
            lambda_outgoing = microfacet_anisotropic_lambda<Distribution>(
                microfacet.alpha_x, microfacet.alpha_y, local_outgoing);
            lambda_incoming = microfacet_anisotropic_lambda<Distribution>(
                microfacet.alpha_x, microfacet.alpha_y, local_incoming);
          };
          const auto common = distribution / cosine_incoming *
              select(0.25f,
                     abs(cosine_half_incoming *
                         fresnel.cosine_transmitted) /
                         square(fresnel.cosine_transmitted +
                                cosine_half_incoming * inverse_eta),
                     refract);
          result.pdf *= common / (1.0f + lambda_incoming);
          result.value *=
              common /
              (1.0f + lambda_incoming + lambda_outgoing);
        };

        result.sampled_roughness =
            make_float2(microfacet.alpha_x, microfacet.alpha_y);
        result.eta = select(1.0f, eta, refract);
        result.label =
            select(closure_type::label_reflect,
                   closure_type::label_transmit, refract) |
            select(closure_type::label_glossy,
                   closure_type::label_singular, singular);
      };
    };
  };
  return result;
}

template<typename Closure>
[[nodiscard]] BsdfEvaluation ggx_eval(
    const KernelGlobals &kernel_globals, const Closure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  auto result = microfacet_eval<MicrofacetDistribution::ggx>(
      kernel_globals, closure, wi, wo);
  result.value *= closure.param.energy_scale;
  return result;
}

template<typename Closure>
[[nodiscard]] BsdfSample ggx_sample(
    const KernelGlobals &kernel_globals, const Closure &closure,
    Expr<luisa::float3> geometric_normal, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  auto result = microfacet_sample<MicrofacetDistribution::ggx>(
      kernel_globals, closure, geometric_normal, wi, random);
  result.value *= closure.param.energy_scale;
  return result;
}

} // namespace

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo);
}

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo);
}

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random);
}

BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return microfacet_eval<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, wi, wo);
}

BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return microfacet_eval<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, wi, wo);
}

BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return microfacet_eval<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, wi, wo);
}

BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  return microfacet_sample<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, Ng, wi, random);
}

BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return microfacet_sample<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, Ng, wi, random);
}

BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return microfacet_sample<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, Ng, wi, random);
}

BsdfEvaluation bsdf_thin_glass_transmission_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return bsdf_microfacet_ggx_eval(
      kernel_globals, closure, reflect(wi, closure.common.N), wo);
}

BsdfSample bsdf_thin_glass_transmission_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(0.0f),
                    .eta = 0.0f,
                    .label = closure_type::label_none};
  $if(roughness_is_almost_specular(closure.param)) {
    result.value = make_float3(1.0e6f);
    result.wo = -wi;
    result.pdf = 1.0e6f;
    result.sampled_roughness = make_float2(0.0f);
    result.eta = 1.0f;
    result.label = closure_type::label_transmit |
                   closure_type::label_singular;
  }
  $else {
    result = bsdf_microfacet_ggx_sample(
        kernel_globals, closure, -Ng, reflect(wi, closure.common.N),
        random);
    $if(result.label != closure_type::label_none) {
      result.label = closure_type::label_transmit |
                     closure_type::label_glossy;
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
