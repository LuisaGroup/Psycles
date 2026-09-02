/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_ashikhmin_shirley.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;
namespace sample_mapping =
    ::psycles::luisa_backend::cycles_sample_mapping;

namespace {

[[nodiscard]] Float square(Expr<float> value) noexcept {
  return value * value;
}

[[nodiscard]] Float roughness_to_exponent(Expr<float> roughness) noexcept {
  return 2.0f / square(roughness) - 2.0f;
}

struct FirstQuadrantSample {
  Float azimuth;
  Float cosine;
};

[[nodiscard]] FirstQuadrantSample sample_first_quadrant(
    Expr<float> exponent_x, Expr<float> exponent_y,
    Expr<luisa::float2> random) noexcept {
  const auto azimuth = atan(
      sqrt((exponent_x + 1.0f) / (exponent_y + 1.0f)) *
      tan(0.5f * sample_mapping::pi * random.x));
  const auto cosine_azimuth = cos(azimuth);
  const auto sine_azimuth = sin(azimuth);
  const auto cosine = pow(
      random.y,
      1.0f / (exponent_x * square(cosine_azimuth) +
              exponent_y * square(sine_azimuth) + 1.0f));
  return {.azimuth = azimuth, .cosine = cosine};
}

} // namespace

BsdfEvaluation bsdf_ashikhmin_shirley_eval(
    const MicrofacetClosure &closure, Expr<luisa::float3> wi,
    Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto normal = closure.common.N;
  Float cosine_incoming = dot(normal, wi);
  Float cosine_outgoing = dot(normal, wo);
  const auto rejected =
      (luisa::compute::max(closure.param.alpha_x, closure.param.alpha_y) <=
       1.0e-4f) |
      (cosine_incoming < 0.0f) | (cosine_outgoing < 0.0f);
  $if(!rejected) {
    cosine_incoming = max(cosine_incoming, 1.0e-6f);
    cosine_outgoing = max(cosine_outgoing, 1.0e-6f);
    const auto half_vector =
        native_vector_math::normalize_unchecked(wi + wo);
    const auto cosine_half_incoming =
        max(abs(dot(half_vector, wi)), 1.0e-6f);
    const auto cosine_half_normal =
        max(dot(half_vector, normal), 1.0e-6f);
    const auto pump =
        1.0f /
        max(1.0e-6f,
            cosine_half_incoming *
                max(cosine_incoming, cosine_outgoing));
    const auto exponent_x = roughness_to_exponent(closure.param.alpha_x);
    const auto exponent_y = roughness_to_exponent(closure.param.alpha_y);

    Float lobe;
    Float normalization;
    $if(exponent_x == exponent_y) {
      lobe = pow(cosine_half_normal, exponent_x);
      normalization =
          (exponent_x + 1.0f) / (8.0f * sample_mapping::pi);
    }
    $else {
      const auto basis = sample_mapping::make_orthonormals_tangent(
          normal, closure.param.T);
      const auto cosine_half_x = dot(half_vector, basis.tangent);
      const auto cosine_half_y = dot(half_vector, basis.bitangent);
      $if(cosine_half_normal < 1.0f) {
        const auto exponent =
            (exponent_x * square(cosine_half_x) +
             exponent_y * square(cosine_half_y)) /
            (1.0f - square(cosine_half_normal));
        lobe = pow(cosine_half_normal, exponent);
      }
      $else { lobe = 1.0f; };
      normalization =
          sqrt((exponent_x + 1.0f) * (exponent_y + 1.0f)) /
          (8.0f * sample_mapping::pi);
    };
    const auto response =
        cosine_outgoing * normalization * lobe * pump;
    result.value = make_float3(response);
    result.pdf = normalization * lobe / cosine_half_incoming;
  };
  return result;
}

BsdfSample bsdf_ashikhmin_shirley_sample(
    const MicrofacetClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float2> input_random) noexcept {
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(
                        closure.param.alpha_x, closure.param.alpha_y),
                    /* Cycles bsdf_sample() assigns eta after the closure
                     * sampler returns, so rejected samples retain 1 as well. */
                    .eta = 1.0f,
                    .label = closure_type::label_none};
  const auto normal = closure.common.N;
  const auto cosine_incoming = dot(normal, wi);
  /* Preserve Cycles' negated positive-domain predicate: NaN is rejected. */
  $if(cosine_incoming > 0.0f) {
    const auto exponent_x = roughness_to_exponent(closure.param.alpha_x);
    const auto exponent_y = roughness_to_exponent(closure.param.alpha_y);
    Float3 basis_x;
    Float3 basis_y;
    $if(exponent_x == exponent_y) {
      const auto basis = sample_mapping::make_orthonormals(normal);
      basis_x = basis.tangent;
      basis_y = basis.bitangent;
    }
    $else {
      const auto basis = sample_mapping::make_orthonormals_tangent(
          normal, closure.param.T);
      basis_x = basis.tangent;
      basis_y = basis.bitangent;
    };

    Float2 random = input_random;
    Float azimuth;
    Float cosine_half;
    $if(exponent_x == exponent_y) {
      azimuth = 2.0f * sample_mapping::pi * random.x;
      cosine_half = pow(random.y, 1.0f / (exponent_x + 1.0f));
    }
    $else {
      $if(random.x < 0.25f) {
        random.x *= 4.0f;
        const auto sampled =
            sample_first_quadrant(exponent_x, exponent_y, random);
        azimuth = sampled.azimuth;
        cosine_half = sampled.cosine;
      }
      $elif(random.x < 0.5f) {
        random.x = 4.0f * (0.5f - random.x);
        const auto sampled =
            sample_first_quadrant(exponent_x, exponent_y, random);
        azimuth = sample_mapping::pi - sampled.azimuth;
        cosine_half = sampled.cosine;
      }
      $elif(random.x < 0.75f) {
        random.x = 4.0f * (random.x - 0.5f);
        const auto sampled =
            sample_first_quadrant(exponent_x, exponent_y, random);
        azimuth = sample_mapping::pi + sampled.azimuth;
        cosine_half = sampled.cosine;
      }
      $else {
        random.x = 4.0f * (1.0f - random.x);
        const auto sampled =
            sample_first_quadrant(exponent_x, exponent_y, random);
        azimuth = 2.0f * sample_mapping::pi - sampled.azimuth;
        cosine_half = sampled.cosine;
      };
    };

    const auto local_half = sample_mapping::spherical_cos_to_direction(
        cosine_half, azimuth);
    Float3 half_vector = local_half.x * basis_x +
                         local_half.y * basis_y +
                         local_half.z * normal;
    const auto cosine_half_incoming = dot(half_vector, wi);
    $if(cosine_half_incoming < 0.0f) { half_vector = -half_vector; };
    result.wo =
        -wi + (2.0f * cosine_half_incoming) * half_vector;
    $if(!(dot(Ng, result.wo) < 0.0f)) {
      result.label = closure_type::label_reflect |
                     closure_type::label_glossy;
      $if(luisa::compute::max(closure.param.alpha_x,
                             closure.param.alpha_y) <= 1.0e-4f) {
        result.pdf = 1.0e6f;
        result.value = make_float3(1.0e6f);
        result.label = closure_type::label_reflect |
                       closure_type::label_singular;
      }
      $else {
        const auto evaluated =
            bsdf_ashikhmin_shirley_eval(closure, wi, result.wo);
        result.value = evaluated.value;
        result.pdf = evaluated.pdf;
      };
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
