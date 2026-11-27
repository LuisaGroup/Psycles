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
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    bool allow_transmission = true) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  constexpr auto trans_mask =
      Distribution == MicrofacetDistribution::ggx
          ? (Usage::bit(closure_type::type_microfacet_ggx_refraction) |
             Usage::bit(closure_type::type_microfacet_ggx_glass) |
             Usage::bit(closure_type::type_microfacet_multi_ggx_glass))
          : (Usage::bit(closure_type::type_microfacet_beckmann_refraction) |
             Usage::bit(closure_type::type_microfacet_beckmann_glass));
  const bool may_transmit =
      allow_transmission &&
      (usage.surface_closure_mask & trans_mask) != 0ull;
  const bool may_anisotropic =
      usage.has_fresnel(Usage::fresnel_anisotropic);

  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto &microfacet = closure.param;
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  const auto cosine_outgoing = dot(normal, wo);

  if (!may_transmit) {
    const auto rejected = (cosine_incoming <= 0.0f) |
                          (cosine_outgoing < 0.0f) |
                          roughness_is_almost_specular(microfacet);
    $if(!rejected) {
      Float3 half_vector = wi + wo;
      const auto inverse_half_length =
          safe_divide(1.0f, sqrt(dot(half_vector, half_vector)));
      half_vector *= inverse_half_length;
      const auto cosine_half_incoming = dot(half_vector, wi);
      const auto fresnel = microfacet_fresnel(
          kernel_globals, closure, cosine_half_incoming, false);
      $if(!all(fresnel.reflectance == make_float3(0.0f))) {
        const auto cosine_normal_half = dot(normal, half_vector);
        Float distribution;
        Float lambda_incoming;
        Float lambda_outgoing;
        const auto eval_isotropic = [&] {
          const auto alpha_squared =
              microfacet.alpha_x * microfacet.alpha_y;
          distribution = microfacet_distribution<Distribution>(
              alpha_squared, cosine_normal_half);
          lambda_incoming = microfacet_lambda<Distribution>(
              alpha_squared, cosine_incoming);
          lambda_outgoing = microfacet_lambda<Distribution>(
              alpha_squared, cosine_outgoing);
        };
        if (!may_anisotropic) {
          eval_isotropic();
        } else {
          $if(microfacet.alpha_x == microfacet.alpha_y) {
            eval_isotropic();
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
        }
        const auto common = 0.25f * distribution / cosine_incoming;
        result.pdf = common / (1.0f + lambda_incoming);
        result.value = fresnel.reflectance * common /
                       (1.0f + lambda_outgoing + lambda_incoming);
      };
    };
    return result;
  }

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
      const auto eval_isotropic = [&] {
        const auto alpha_squared =
            microfacet.alpha_x * microfacet.alpha_y;
        distribution = microfacet_distribution<Distribution>(
            alpha_squared, cosine_normal_half);
        lambda_incoming = microfacet_lambda<Distribution>(
            alpha_squared, cosine_incoming);
        lambda_outgoing = microfacet_lambda<Distribution>(
            alpha_squared, cosine_outgoing);
      };
      if (!may_anisotropic) {
        eval_isotropic();
      } else {
        $if((microfacet.alpha_x == microfacet.alpha_y) |
            is_transmission) {
          eval_isotropic();
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
      }

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
    Expr<luisa::float3> random,
    bool allow_transmission = true) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  constexpr auto trans_mask =
      Distribution == MicrofacetDistribution::ggx
          ? (Usage::bit(closure_type::type_microfacet_ggx_refraction) |
             Usage::bit(closure_type::type_microfacet_ggx_glass) |
             Usage::bit(closure_type::type_microfacet_multi_ggx_glass))
          : (Usage::bit(closure_type::type_microfacet_beckmann_refraction) |
             Usage::bit(closure_type::type_microfacet_beckmann_glass));
  const bool may_transmit =
      allow_transmission &&
      (usage.surface_closure_mask & trans_mask) != 0ull;
  const bool may_anisotropic =
      usage.has_fresnel(Usage::fresnel_anisotropic);

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
    Bool singular = roughness_is_almost_specular(microfacet);
    Float3 half_vector;
    Float3 local_half = make_float3(0.0f);
    Float3 local_incoming = make_float3(0.0f);
    $if(singular) { half_vector = normal; }
    $else {
      Float3 basis_x;
      Float3 basis_y;
      if (!may_anisotropic) {
        const auto basis = sample_mapping::make_orthonormals(normal);
        basis_x = basis.tangent;
        basis_y = basis.bitangent;
      } else {
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
      }
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
        kernel_globals, closure, cosine_half_incoming, may_transmit);

    if (!may_transmit) {
      $if(!all(fresnel.reflectance == make_float3(0.0f))) {
        result.wo = 2.0f * cosine_half_incoming * half_vector - wi;
        const auto cosine_outgoing = dot(normal, result.wo);
        const auto cosine_geometric_outgoing =
            dot(geometric_normal, result.wo);
        $if((cosine_geometric_outgoing >= 0.0f) & (cosine_outgoing >= 0.0f)) {
          result.value = fresnel.reflectance;
          result.pdf = 1.0f;
          $if(singular) {
            result.pdf = 1.0e6f;
            result.value *= 1.0e6f;
          }
          $else {
            Float distribution;
            Float lambda_incoming;
            Float lambda_outgoing;
            const auto sample_isotropic = [&] {
              const auto alpha_squared =
                  microfacet.alpha_x * microfacet.alpha_y;
              distribution = microfacet_distribution<Distribution>(
                  alpha_squared, local_half.z);
              lambda_outgoing = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_outgoing);
              lambda_incoming = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_incoming);
            };
            if (!may_anisotropic) {
              sample_isotropic();
            } else {
              $if(microfacet.alpha_x == microfacet.alpha_y) {
                sample_isotropic();
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
            }
            const auto common = 0.25f * distribution / cosine_incoming;
            result.pdf = common / (1.0f + lambda_incoming);
            result.value *= common / (1.0f + lambda_incoming + lambda_outgoing);
          };
          result.sampled_roughness =
              make_float2(microfacet.alpha_x, microfacet.alpha_y);
          result.eta = 1.0f;
          result.label = closure_type::label_reflect |
                         select(closure_type::label_glossy,
                                closure_type::label_singular, singular);
        };
      };
      return;
    }

    const auto eta = microfacet.ior;
    const auto inverse_eta = safe_divide(1.0f, eta);
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
          const auto sample_isotropic = [&] {
            const auto alpha_squared =
                microfacet.alpha_x * microfacet.alpha_y;
            distribution = microfacet_distribution<Distribution>(
                alpha_squared, local_half.z);
            lambda_outgoing = microfacet_lambda<Distribution>(
                alpha_squared, cosine_outgoing);
            lambda_incoming = microfacet_lambda<Distribution>(
                alpha_squared, cosine_incoming);
          };
          if (!may_anisotropic) {
            sample_isotropic();
          } else {
            $if((microfacet.alpha_x == microfacet.alpha_y) | refract) {
              sample_isotropic();
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
          }
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
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    bool allow_transmission = true) noexcept {
  auto result = microfacet_eval<MicrofacetDistribution::ggx>(
      kernel_globals, closure, wi, wo, allow_transmission);
  result.value *= closure.param.energy_scale;
  return result;
}

template<typename Closure>
[[nodiscard]] BsdfSample ggx_sample(
    const KernelGlobals &kernel_globals, const Closure &closure,
    Expr<luisa::float3> geometric_normal, Expr<luisa::float3> wi,
    Expr<luisa::float3> random,
    bool allow_transmission = true) noexcept {
  auto result = microfacet_sample<MicrofacetDistribution::ggx>(
      kernel_globals, closure, geometric_normal, wi, random,
      allow_transmission);
  result.value *= closure.param.energy_scale;
  return result;
}

[[nodiscard]] Float xyz3_lobe_reflection_probability(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<float> cosine_incoming, Expr<luisa::float4> eta_lambda,
    const cycles_spectrum::SpectralContext &spectral, Expr<bool> is_dispersive,
    Expr<bool> singular) noexcept {
  const auto fn = microfacet_fresnel_spectral(
      kernel_globals, closure, cosine_incoming, eta_lambda, spectral,
      is_dispersive, true);
  const auto avg3 = [&](Expr<luisa::float4> v) noexcept {
    return select((v.x + v.y + v.z) * (1.0f / 3.0f), v.x,
                  spectral.secondary_terminated);
  };
  const Float refl = avg3(fn.reflectance);
  const Float tot = avg3(fn.reflectance + fn.transmittance);
  const Bool has_refl =
      !closure_type::is_refraction_microfacet(closure.common.type);
  const Bool has_trans =
      closure_type::is_glass_microfacet(closure.common.type) | !has_refl;
  Float p = select(select(1.0f, 0.0f, has_trans), refl / tot, tot > 0.0f);
  const Bool clamp_two_lobe = (!singular) & has_refl & has_trans;
  p = select(p, clamp(p, 0.05f, 0.95f),
             ((refl > 0.0f) & ((tot - refl) > 0.0f)) | clamp_two_lobe);
  return p;
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] BsdfSpectralEvaluation xyz3_rough_dispersive_transmission_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<luisa::float4> eta_lambda, Expr<float> refl_prob,
    Expr<float> cosine_incoming, Expr<float> lambda_incoming,
    Expr<float> lambda_outgoing) noexcept {
  const auto &microfacet = closure.param;
  const Float alpha_squared = microfacet.alpha_x * microfacet.alpha_y;
  luisa::compute::ArrayFloat<4> eta_arr;
  eta_arr[0u] = eta_lambda.x;
  eta_arr[1u] = eta_lambda.y;
  eta_arr[2u] = eta_lambda.z;
  eta_arr[3u] = eta_lambda.w;
  luisa::compute::ArrayFloat<4> f_arr;
  luisa::compute::ArrayFloat<4> p_arr;
  for (std::uint32_t i = 0u; i < 4u; ++i) {
    f_arr[i] = 0.0f;
    p_arr[i] = 0.0f;
  }
  const UInt lane_limit = select(3u, 1u, spectral.secondary_terminated);
  $for(k, lane_limit) {
    const Float eta_k = eta_arr[k];
    const Float3 raw_h = eta_k * wo + wi;
    const Float len2_k = dot(raw_h, raw_h);
    $if(len2_k > 1.0e-12f) {
      const Float inv_k = 1.0f / sqrt(len2_k);
      const Float3 h_k = select(-raw_h, raw_h, eta_k < 1.0f) * inv_k;
      const Float chi = dot(h_k, wi);
      const Float cho = dot(h_k, wo);
      const Float cnh = dot(closure.common.N, h_k);
      $if((cnh > 0.0f) & (chi > 0.0f) & (cho < 0.0f)) {
        const Float d_k =
            microfacet_distribution<Distribution>(alpha_squared, cnh);
        const Float t_k = microfacet_fresnel_spectral(
                              kernel_globals, closure, chi,
                              make_float4(eta_k), spectral, Bool{true}, true)
                              .transmittance.x;
        const Float j_k = d_k / cosine_incoming *
                          square(eta_k * inv_k) * abs(chi * cho);
        f_arr[k] = t_k * j_k / (1.0f + lambda_incoming + lambda_outgoing);
        p_arr[k] = j_k * (1.0f - refl_prob) / (1.0f + lambda_incoming);
      };
    };
  };
  const Float4 value4 = select(
      make_float4(f_arr[0u], f_arr[1u], f_arr[2u], f_arr[2u]),
      make_float4(f_arr[0u]), spectral.secondary_terminated);
  const Float4 pdf4 = select(
      make_float4(p_arr[0u], p_arr[1u], p_arr[2u], p_arr[2u]),
      make_float4(p_arr[0u]), spectral.secondary_terminated);
  return BsdfSpectralEvaluation{
      .value = value4, .pdf = pdf4.x, .lane_pdf = pdf4};
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] BsdfSpectralEvaluation microfacet_eval_spectral_impl(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe, bool allow_transmission = true) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  constexpr auto trans_mask =
      Distribution == MicrofacetDistribution::ggx
          ? (Usage::bit(closure_type::type_microfacet_ggx_refraction) |
             Usage::bit(closure_type::type_microfacet_ggx_glass) |
             Usage::bit(closure_type::type_microfacet_multi_ggx_glass))
          : (Usage::bit(closure_type::type_microfacet_beckmann_refraction) |
             Usage::bit(closure_type::type_microfacet_beckmann_glass));
  const bool may_transmit =
      allow_transmission &&
      (usage.surface_closure_mask & trans_mask) != 0ull;
  const bool may_anisotropic =
      usage.has_fresnel(Usage::fresnel_anisotropic);
  const bool has_disp = kernel_globals.has_dispersive_materials();

  BsdfSpectralEvaluation result{.value = make_float4(0.0f), .pdf = 0.0f};
  const auto &microfacet = closure.param;
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  const auto cosine_outgoing = dot(normal, wo);

  if (!may_transmit) {
    const auto rejected = (cosine_incoming <= 0.0f) |
                          (cosine_outgoing < 0.0f) |
                          roughness_is_almost_specular(microfacet);
    $if(!rejected) {
      Float3 half_vector = wi + wo;
      const auto inverse_half_length =
          safe_divide(1.0f, sqrt(dot(half_vector, half_vector)));
      half_vector *= inverse_half_length;
      const auto cosine_half_incoming = dot(half_vector, wi);
      const auto fresnel = microfacet_fresnel_spectral(
          kernel_globals, closure, cosine_half_incoming,
          make_float4(microfacet.ior), spectral, false, false);
      $if(any(fresnel.reflectance != make_float4(0.0f))) {
        const auto cosine_normal_half = dot(normal, half_vector);
        Float distribution;
        Float lambda_incoming;
        Float lambda_outgoing;
        const auto eval_isotropic = [&] {
          const auto alpha_squared =
              microfacet.alpha_x * microfacet.alpha_y;
          distribution = microfacet_distribution<Distribution>(
              alpha_squared, cosine_normal_half);
          lambda_incoming = microfacet_lambda<Distribution>(
              alpha_squared, cosine_incoming);
          lambda_outgoing = microfacet_lambda<Distribution>(
              alpha_squared, cosine_outgoing);
        };
        if (!may_anisotropic) {
          eval_isotropic();
        } else {
          $if(microfacet.alpha_x == microfacet.alpha_y) {
            eval_isotropic();
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
        }
        const auto common = 0.25f * distribution / cosine_incoming;
        result.pdf = common / (1.0f + lambda_incoming);
        result.value = fresnel.reflectance * common /
                       (1.0f + lambda_outgoing + lambda_incoming);
      };
    };
    return result;
  }

  if (kernel_globals.xyz3_rough_dispersion()) {
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
      const Bool is_dispersive =
          has_disp ? (dispersion_abbe > 0.0f) : Bool{false};
      const Float4 eta_lambda =
          has_disp ? cycles_spectrum::cauchy_ior(
                         microfacet.ior, dispersion_abbe,
                         spectral.wavelengths.lambdas)
                   : make_float4(microfacet.ior);
      const Float refl_prob = xyz3_lobe_reflection_probability(
          kernel_globals, closure, cosine_incoming, eta_lambda, spectral,
          is_dispersive, Bool{false});
      $if(is_transmission & is_dispersive) {
        const auto alpha_squared = microfacet.alpha_x * microfacet.alpha_y;
        const auto lambda_incoming =
            microfacet_lambda<Distribution>(alpha_squared, cosine_incoming);
        const auto lambda_outgoing =
            microfacet_lambda<Distribution>(alpha_squared, cosine_outgoing);
        const auto trans_eval =
            xyz3_rough_dispersive_transmission_eval<Distribution>(
                kernel_globals, closure, wi, wo, spectral, eta_lambda,
                refl_prob, cosine_incoming, lambda_incoming, lambda_outgoing);
        result.value = trans_eval.value;
        result.pdf = trans_eval.pdf;
        result.lane_pdf = trans_eval.lane_pdf;
      }
      $else {
        const Float eta_hero = eta_lambda.x;
        const Float3 raw_h =
            select(wi + wo,
                   select(-(eta_hero * wo + wi), eta_hero * wo + wi,
                          eta_hero < 1.0f),
                   is_transmission);
        const Float len2_h = dot(raw_h, raw_h);
        $if(len2_h > 1.0e-12f) {
          const Float inverse_half_length = 1.0f / sqrt(len2_h);
          const Float3 half_vector = raw_h * inverse_half_length;
          const auto cosine_normal_half = dot(normal, half_vector);
          const auto cosine_half_incoming = dot(half_vector, wi);
          const auto valid_h = select(
              Bool{true},
              (cosine_normal_half > 0.0f) & (cosine_half_incoming > 0.0f) &
                  (dot(half_vector, wo) < 0.0f),
              is_transmission);
          $if(valid_h) {
            const auto fresnel = microfacet_fresnel_spectral(
                kernel_globals, closure, cosine_half_incoming, eta_lambda,
                spectral, is_dispersive, true);
            const auto active_lobe =
                select(fresnel.reflectance, fresnel.transmittance,
                       is_transmission);
            $if(any(active_lobe != make_float4(0.0f))) {
              Float distribution;
              Float lambda_incoming;
              Float lambda_outgoing;
              const auto eval_isotropic = [&] {
                const auto alpha_squared =
                    microfacet.alpha_x * microfacet.alpha_y;
                distribution = microfacet_distribution<Distribution>(
                    alpha_squared, cosine_normal_half);
                lambda_incoming = microfacet_lambda<Distribution>(
                    alpha_squared, cosine_incoming);
                lambda_outgoing = microfacet_lambda<Distribution>(
                    alpha_squared, cosine_outgoing);
              };
              if (!may_anisotropic) {
                eval_isotropic();
              } else {
                $if((microfacet.alpha_x == microfacet.alpha_y) |
                    is_transmission) {
                  eval_isotropic();
                }
                $else {
                  const auto basis =
                      sample_mapping::make_orthonormals_tangent(
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
                  distribution =
                      microfacet_anisotropic_distribution<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y, local_half);
                  lambda_incoming =
                      microfacet_anisotropic_lambda<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y,
                          local_incoming);
                  lambda_outgoing =
                      microfacet_anisotropic_lambda<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y,
                          local_outgoing);
                };
              }
              const auto common = distribution / cosine_incoming *
                  select(0.25f,
                         square(eta_hero * inverse_half_length) *
                             abs(cosine_half_incoming * dot(half_vector, wo)),
                         is_transmission);
              const auto lobe_probability =
                  select(refl_prob, 1.0f - refl_prob, is_transmission);
              result.pdf = common * lobe_probability / (1.0f + lambda_incoming);
              result.lane_pdf = make_float4(result.pdf);
              result.value = active_lobe * common /
                             (1.0f + lambda_outgoing + lambda_incoming);
            };
          };
        };
      };
    };
    return result;
  }

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
    const Bool is_dispersive = has_disp ? (dispersion_abbe > 0.0f) : Bool{false};
    const Float4 eta_lambda =
        has_disp ? cycles_spectrum::cauchy_ior(
                       microfacet.ior, dispersion_abbe,
                       spectral.wavelengths.lambdas)
                 : make_float4(microfacet.ior);
    const Float eta_hero = eta_lambda.x;
    Float3 half_vector = select(wi + wo, -(eta_hero * wo + wi), is_transmission);
    const auto inverse_half_length =
        safe_divide(1.0f, sqrt(dot(half_vector, half_vector)));
    half_vector *= inverse_half_length;
    const auto cosine_half_incoming = dot(half_vector, wi);
    const auto fresnel = microfacet_fresnel_spectral(
        kernel_globals, closure, cosine_half_incoming, eta_lambda, spectral,
        is_dispersive, true);
    const auto hero_tir =
        is_transmission & is_dispersive & (fresnel.transmittance.x <= 0.0f);
    const auto fresnel_is_zero =
        hero_tir |
        ((!any(fresnel.reflectance != make_float4(0.0f))) &
         (!any(fresnel.transmittance != make_float4(0.0f))));

    $if(!fresnel_is_zero) {
      const auto cosine_normal_half = dot(normal, half_vector);
      Float distribution;
      Float lambda_incoming;
      Float lambda_outgoing;
      const auto eval_isotropic = [&] {
        const auto alpha_squared =
            microfacet.alpha_x * microfacet.alpha_y;
        distribution = microfacet_distribution<Distribution>(
            alpha_squared, cosine_normal_half);
        lambda_incoming = microfacet_lambda<Distribution>(
            alpha_squared, cosine_incoming);
        lambda_outgoing = microfacet_lambda<Distribution>(
            alpha_squared, cosine_outgoing);
      };
      if (!may_anisotropic) {
        eval_isotropic();
      } else {
        $if((microfacet.alpha_x == microfacet.alpha_y) | is_transmission) {
          eval_isotropic();
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
      }

      const auto common = distribution / cosine_incoming *
          select(0.25f,
                 square(eta_hero * inverse_half_length) *
                     abs(cosine_half_incoming * dot(half_vector, wo)),
                 is_transmission);
      const auto avg_active = [&](Expr<luisa::float4> v) noexcept {
        return select((v.x + v.y + v.z + v.w) * 0.25f, v.x,
                      spectral.secondary_terminated);
      };
      const auto refl_mass = avg_active(fresnel.reflectance);
      const auto total_mass =
          avg_active(fresnel.reflectance + fresnel.transmittance);
      const auto reflection_probability =
          select(0.0f, refl_mass / total_mass, total_mass > 0.0f);
      const auto lobe_probability =
          select(reflection_probability, 1.0f - reflection_probability,
                 is_transmission);
      result.pdf = common * lobe_probability / (1.0f + lambda_incoming);
      const auto factor = common / (1.0f + lambda_outgoing + lambda_incoming);
      Float4 lobe_f = select(fresnel.reflectance, fresnel.transmittance,
                             is_transmission) *
                      factor;
      if (has_disp) {
        const Float hero_scale =
            select(4.0f, 1.0f, spectral.secondary_terminated);
        const Float4 disp_trans =
            make_float4(lobe_f.x * hero_scale, 0.0f, 0.0f, 0.0f);
        lobe_f = select(lobe_f, disp_trans, is_transmission & is_dispersive);
      }
      result.value = lobe_f;
    };
  };
  return result;
}

template<MicrofacetDistribution Distribution>
[[nodiscard]] BsdfSpectralSample microfacet_sample_spectral_impl(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> geometric_normal, Expr<luisa::float3> wi,
    Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe, bool allow_transmission = true) noexcept {
  using Usage = compiler::cycles_svm::ClosureUsage;
  const auto usage = kernel_globals.closure_usage();
  constexpr auto trans_mask =
      Distribution == MicrofacetDistribution::ggx
          ? (Usage::bit(closure_type::type_microfacet_ggx_refraction) |
             Usage::bit(closure_type::type_microfacet_ggx_glass) |
             Usage::bit(closure_type::type_microfacet_multi_ggx_glass))
          : (Usage::bit(closure_type::type_microfacet_beckmann_refraction) |
             Usage::bit(closure_type::type_microfacet_beckmann_glass));
  const bool may_transmit =
      allow_transmission &&
      (usage.surface_closure_mask & trans_mask) != 0ull;
  const bool may_anisotropic =
      usage.has_fresnel(Usage::fresnel_anisotropic);
  const bool has_disp = kernel_globals.has_dispersive_materials();

  BsdfSpectralSample result{.value = make_float4(0.0f),
                            .wo = make_float3(0.0f),
                            .pdf = 0.0f,
                            .sampled_roughness = make_float2(0.0f),
                            .eta = 0.0f,
                            .label = closure_type::label_none,
                            .terminated_secondary = false};
  const auto &microfacet = closure.param;
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  $if(cosine_incoming > 0.0f) {
    Bool singular = roughness_is_almost_specular(microfacet);
    Float3 half_vector;
    Float3 local_half = make_float3(0.0f);
    Float3 local_incoming = make_float3(0.0f);
    $if(singular) { half_vector = normal; }
    $else {
      Float3 basis_x;
      Float3 basis_y;
      if (!may_anisotropic) {
        const auto basis = sample_mapping::make_orthonormals(normal);
        basis_x = basis.tangent;
        basis_y = basis.bitangent;
      } else {
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
      }
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

    if (!may_transmit) {
      const auto fresnel = microfacet_fresnel_spectral(
          kernel_globals, closure, cosine_half_incoming,
          make_float4(microfacet.ior), spectral, false, false);
      $if(any(fresnel.reflectance != make_float4(0.0f))) {
        result.wo = 2.0f * cosine_half_incoming * half_vector - wi;
        const auto cosine_outgoing = dot(normal, result.wo);
        const auto cosine_geometric_outgoing =
            dot(geometric_normal, result.wo);
        $if((cosine_geometric_outgoing >= 0.0f) & (cosine_outgoing >= 0.0f)) {
          result.value = fresnel.reflectance;
          result.pdf = 1.0f;
          $if(singular) {
            result.pdf = 1.0e6f;
            result.value *= 1.0e6f;
          }
          $else {
            Float distribution;
            Float lambda_incoming;
            Float lambda_outgoing;
            const auto sample_isotropic = [&] {
              const auto alpha_squared =
                  microfacet.alpha_x * microfacet.alpha_y;
              distribution = microfacet_distribution<Distribution>(
                  alpha_squared, local_half.z);
              lambda_outgoing = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_outgoing);
              lambda_incoming = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_incoming);
            };
            if (!may_anisotropic) {
              sample_isotropic();
            } else {
              $if(microfacet.alpha_x == microfacet.alpha_y) {
                sample_isotropic();
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
            }
            const auto common = 0.25f * distribution / cosine_incoming;
            result.pdf = common / (1.0f + lambda_incoming);
            result.value *= common / (1.0f + lambda_incoming + lambda_outgoing);
          };
          result.sampled_roughness =
              make_float2(microfacet.alpha_x, microfacet.alpha_y);
          result.eta = 1.0f;
          result.label = closure_type::label_reflect |
                         select(closure_type::label_glossy,
                                closure_type::label_singular, singular);
        };
      };
      return;
    }

    if (kernel_globals.xyz3_rough_dispersion()) {
      const Bool is_dispersive =
          has_disp ? (dispersion_abbe > 0.0f) : Bool{false};
      const Float4 eta_lambda =
          has_disp ? cycles_spectrum::cauchy_ior(
                         microfacet.ior, dispersion_abbe,
                         spectral.wavelengths.lambdas)
                   : make_float4(microfacet.ior);
      const Float eta = eta_lambda.x;
      const Float inverse_eta = safe_divide(1.0f, eta);
      const auto fresnel = microfacet_fresnel_spectral(
          kernel_globals, closure, cosine_half_incoming, eta_lambda, spectral,
          is_dispersive, true);
      const auto fresnel_is_zero =
          (!any(fresnel.reflectance != make_float4(0.0f))) &
          (!any(fresnel.transmittance != make_float4(0.0f)));
      $if(!fresnel_is_zero) {
        const Float reflection_probability = xyz3_lobe_reflection_probability(
            kernel_globals, closure, cosine_incoming, eta_lambda, spectral,
            is_dispersive, singular);
        const auto refract = random.z >= reflection_probability;
        const auto hero_tir =
            refract & (fresnel.transmittance.x <= 0.0f);
        $if(!hero_tir) {
          const Float cos_t = fresnel.cosine_transmitted.x;
          $if(refract) {
            result.wo =
                (inverse_eta * cosine_half_incoming + cos_t) * half_vector -
                inverse_eta * wi;
          }
          $else {
            result.wo = 2.0f * cosine_half_incoming * half_vector - wi;
          };
          const auto cosine_outgoing = dot(normal, result.wo);
          const auto cosine_geometric_outgoing =
              dot(geometric_normal, result.wo);
          const auto wrong_hemisphere =
              ((cosine_geometric_outgoing < 0.0f) != refract) |
              ((cosine_outgoing < 0.0f) != refract);
          $if(!wrong_hemisphere) {
            singular |= refract & (abs(eta - 1.0f) < 1.0e-4f);
            $if(singular) {
              $if(refract) {
                const Bool collapse = is_dispersive;
                result.terminated_secondary = collapse;
                result.value = select(
                    fresnel.transmittance * 1.0e6f,
                    make_float4(fresnel.transmittance.x * 1.0e6f,
                                0.0f, 0.0f, 0.0f),
                    collapse);
                result.pdf = (1.0f - reflection_probability) * 1.0e6f;
              }
              $else {
                result.value = fresnel.reflectance * 1.0e6f;
                result.pdf = reflection_probability * 1.0e6f;
              };
              result.lane_pdf = make_float4(result.pdf);
            }
            $else {
              Float distribution;
              Float lambda_incoming;
              Float lambda_outgoing;
              const auto sample_isotropic = [&] {
                const auto alpha_squared =
                    microfacet.alpha_x * microfacet.alpha_y;
                distribution = microfacet_distribution<Distribution>(
                    alpha_squared, local_half.z);
                lambda_outgoing = microfacet_lambda<Distribution>(
                    alpha_squared, cosine_outgoing);
                lambda_incoming = microfacet_lambda<Distribution>(
                    alpha_squared, cosine_incoming);
              };
              if (!may_anisotropic) {
                sample_isotropic();
              } else {
                $if((microfacet.alpha_x == microfacet.alpha_y) | refract) {
                  sample_isotropic();
                }
                $else {
                  const auto local_outgoing =
                      2.0f * cosine_half_incoming * local_half - local_incoming;
                  distribution =
                      microfacet_anisotropic_distribution<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y, local_half);
                  lambda_outgoing =
                      microfacet_anisotropic_lambda<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y,
                          local_outgoing);
                  lambda_incoming =
                      microfacet_anisotropic_lambda<Distribution>(
                          microfacet.alpha_x, microfacet.alpha_y,
                          local_incoming);
                };
              }
              $if(refract & is_dispersive) {
                const auto trans_eval =
                    xyz3_rough_dispersive_transmission_eval<Distribution>(
                        kernel_globals, closure, wi, result.wo, spectral,
                        eta_lambda, reflection_probability, cosine_incoming,
                        lambda_incoming, lambda_outgoing);
                result.value = trans_eval.value;
                result.pdf = trans_eval.pdf;
                result.lane_pdf = trans_eval.lane_pdf;
              }
              $else {
                const auto lobe_prob = select(
                    reflection_probability, 1.0f - reflection_probability,
                    refract);
                const auto lobe_fresnel = select(
                    fresnel.reflectance, fresnel.transmittance, refract);
                const auto common = distribution / cosine_incoming *
                    select(0.25f,
                           abs(cosine_half_incoming * cos_t) /
                               square(cos_t + cosine_half_incoming * inverse_eta),
                           refract);
                result.pdf = common * lobe_prob / (1.0f + lambda_incoming);
                result.lane_pdf = make_float4(result.pdf);
                result.value = lobe_fresnel * common /
                               (1.0f + lambda_incoming + lambda_outgoing);
              };
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
      return;
    }

    const Bool is_dispersive = has_disp ? (dispersion_abbe > 0.0f) : Bool{false};
    const Float4 eta_lambda =
        has_disp ? cycles_spectrum::cauchy_ior(
                       microfacet.ior, dispersion_abbe,
                       spectral.wavelengths.lambdas)
                 : make_float4(microfacet.ior);
    const Float eta = eta_lambda.x;
    const Float inverse_eta = safe_divide(1.0f, eta);
    const auto fresnel = microfacet_fresnel_spectral(
        kernel_globals, closure, cosine_half_incoming, eta_lambda, spectral,
        is_dispersive, true);
    const auto fresnel_is_zero =
        (!any(fresnel.reflectance != make_float4(0.0f))) &
        (!any(fresnel.transmittance != make_float4(0.0f)));
    $if(!fresnel_is_zero) {
      const auto avg_active = [&](Expr<luisa::float4> v) noexcept {
        return select((v.x + v.y + v.z + v.w) * 0.25f, v.x,
                      spectral.secondary_terminated);
      };
      const auto refl_mass = avg_active(fresnel.reflectance);
      const auto total_mass =
          avg_active(fresnel.reflectance + fresnel.transmittance);
      const auto reflection_probability =
          select(0.0f, refl_mass / total_mass, total_mass > 0.0f);
      const auto refract = random.z >= reflection_probability;
      const auto hero_tir =
          refract & is_dispersive & (fresnel.transmittance.x <= 0.0f);
      $if(!hero_tir) {
        const Float cos_t = fresnel.cosine_transmitted.x;
        $if(refract) {
          result.wo =
              (inverse_eta * dot(half_vector, wi) + cos_t) * half_vector -
              inverse_eta * wi;
        }
        $else {
          result.wo = 2.0f * cosine_half_incoming * half_vector - wi;
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
            const auto sample_isotropic = [&] {
              const auto alpha_squared =
                  microfacet.alpha_x * microfacet.alpha_y;
              distribution = microfacet_distribution<Distribution>(
                  alpha_squared, local_half.z);
              lambda_outgoing = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_outgoing);
              lambda_incoming = microfacet_lambda<Distribution>(
                  alpha_squared, cosine_incoming);
            };
            if (!may_anisotropic) {
              sample_isotropic();
            } else {
              $if((microfacet.alpha_x == microfacet.alpha_y) | refract) {
                sample_isotropic();
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
            }
            const auto common = distribution / cosine_incoming *
                select(0.25f,
                       abs(cosine_half_incoming * cos_t) /
                           square(cos_t + cosine_half_incoming * inverse_eta),
                       refract);
            result.pdf *= common / (1.0f + lambda_incoming);
            result.value *= common / (1.0f + lambda_incoming + lambda_outgoing);
          };

          if (has_disp) {
            const Bool terminate = refract & is_dispersive;
            result.terminated_secondary = terminate;
            result.value = select(
                result.value,
                make_float4(result.value.x, 0.0f, 0.0f, 0.0f),
                terminate);
          }

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
  };
  return result;
}

} // namespace

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo, true);
}

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo, false);
}

BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo) noexcept {
  return ggx_eval(kernel_globals, closure, wi, wo, false);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi,
    Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random, true);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random, false);
}

BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float3> random) noexcept {
  return ggx_sample(kernel_globals, closure, Ng, wi, random, false);
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

BsdfSpectralEvaluation bsdf_microfacet_ggx_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  auto result = microfacet_eval_spectral_impl<MicrofacetDistribution::ggx>(
      kernel_globals, closure, wi, wo, spectral, dispersion_abbe, true);
  result.value *= closure.param.energy_scale;
  return result;
}

BsdfSpectralSample bsdf_microfacet_ggx_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi, Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  auto result = microfacet_sample_spectral_impl<MicrofacetDistribution::ggx>(
      kernel_globals, closure, Ng, wi, random, spectral, dispersion_abbe, true);
  result.value *= closure.param.energy_scale;
  return result;
}

BsdfSpectralEvaluation bsdf_microfacet_beckmann_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  return microfacet_eval_spectral_impl<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, wi, wo, spectral, dispersion_abbe, true);
}

BsdfSpectralSample bsdf_microfacet_beckmann_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi, Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  return microfacet_sample_spectral_impl<MicrofacetDistribution::beckmann>(
      kernel_globals, closure, Ng, wi, random, spectral, dispersion_abbe, true);
}

BsdfSpectralEvaluation bsdf_thin_glass_transmission_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> wi, Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  return bsdf_microfacet_ggx_eval_spectral(
      kernel_globals, closure, reflect(wi, closure.common.N), wo, spectral,
      dispersion_abbe);
}

BsdfSpectralSample bsdf_thin_glass_transmission_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    Expr<luisa::float3> Ng, Expr<luisa::float3> wi, Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    Expr<float> dispersion_abbe) noexcept {
  BsdfSpectralSample result{.value = make_float4(0.0f),
                            .wo = make_float3(0.0f),
                            .pdf = 0.0f,
                            .sampled_roughness = make_float2(0.0f),
                            .eta = 0.0f,
                            .label = closure_type::label_none,
                            .terminated_secondary = false};
  $if(roughness_is_almost_specular(closure.param)) {
    result.value = make_float4(1.0e6f);
    result.wo = -wi;
    result.pdf = 1.0e6f;
    result.sampled_roughness = make_float2(0.0f);
    result.eta = 1.0f;
    result.label =
        closure_type::label_transmit | closure_type::label_singular;
  }
  $else {
    result = bsdf_microfacet_ggx_sample_spectral(
        kernel_globals, closure, -Ng, reflect(wi, closure.common.N), random,
        spectral, dispersion_abbe);
    $if(result.label != closure_type::label_none) {
      result.label =
          closure_type::label_transmit | closure_type::label_glossy;
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
