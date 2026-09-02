/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_toon.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

inline constexpr float half_pi = 1.57079632679489661923f;

[[nodiscard]] Float safe_acos(Expr<float> value) noexcept {
  return acos(clamp(value, -1.0f, 1.0f));
}

[[nodiscard]] Float toon_intensity(Expr<float> maximum, Expr<float> smooth,
                                   Expr<float> angle) noexcept {
  Float intensity = 0.0f;
  $if(angle < maximum) { intensity = 1.0f; }
  $elif((angle < maximum + smooth) & (smooth != 0.0f)) {
    intensity = 1.0f - (angle - maximum) / smooth;
  };
  return intensity;
}

[[nodiscard]] Float toon_sample_angle(const ToonClosure &closure) noexcept {
  return min(closure.param.size + closure.param.smooth, half_pi);
}

} // namespace

void node_toon(const KernelGlobals &kernel_globals, Cursor &cursor,
               Stack &stack, Expr<std::uint32_t> type,
               Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
               ShaderData &shader_data, const PathState &path_state) noexcept {
  /* svm_node_get advances over the complete typed record before Cycles tests
   * the Glossy-caustics branch. Preserve that cursor transition even when no
   * closure is allocated. */
  const auto size_input = cursor.word();
  const auto smooth_input = cursor.word();
  const auto normal_packed = cursor.word();

  const Bool glossy =
      type == static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID);
  const Bool diffuse_visibility =
      (path_state.visibility & path_ray_visibility_diffuse) != 0u;
  const Bool active =
      !glossy | kernel_globals.caustics_reflective() | !diffuse_visibility;

  $if(active) {
    const auto normal_offset = cursor.byte(normal_packed, 0u);
    auto normal =
        stack_load_float3_default(stack, normal_offset, shader_data.N);
    normal =
        native_vector_math::safe_normalize_nonzero_or(normal, shader_data.N);

    auto &pool = *shader_data.closure;
    const auto allocated =
        bsdf_allocate(shader_data, closure_weight * mix_weight);
    $if(allocated.valid) {
      /* bsdf_toon_setup_common performs these maps after allocation, for both
       * closure tags and in this order. */
      const auto size =
          clamp(stack_load_input_float(stack, size_input), 1.0e-5f, 1.0f) *
          half_pi;
      const auto smooth =
          clamp(stack_load_input_float(stack, smooth_input), 0.0f, 1.0f) *
          half_pi;
      pool.set_normal(allocated.index, normal);
      pool.set_toon_param(allocated.index, {.size = size, .smooth = smooth});
      pool.set_type(allocated.index, type);
      shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    };
  };
}

BsdfEvaluation bsdf_diffuse_toon_eval(const ToonClosure &closure,
                                      Expr<luisa::float3>,
                                      Expr<luisa::float3> wo) noexcept {
  Float3 value = make_float3(0.0f);
  Float pdf = 0.0f;
  const auto cosine = dot(closure.common.N, wo);
  $if(cosine >= 0.0f) {
    const auto angle = safe_acos(max(cosine, 0.0f));
    const auto sample_angle = toon_sample_angle(closure);
    $if(angle < sample_angle) {
      pdf = cycles_sample_mapping::inverse_two_pi /
            cycles_sample_mapping::one_minus_cosine_from_angle(sample_angle);
      value = make_float3(pdf * toon_intensity(closure.param.size,
                                               closure.param.smooth, angle));
    };
  };
  return {.value = value, .pdf = pdf};
}

BsdfSample bsdf_diffuse_toon_sample(const ToonClosure &closure,
                                    Expr<luisa::float3> Ng, Expr<luisa::float3>,
                                    Expr<luisa::float2> random) noexcept {
  const auto sample_angle = toon_sample_angle(closure);
  const auto cone = cycles_sample_mapping::sample_uniform_cone(
      closure.common.N,
      cycles_sample_mapping::one_minus_cosine_from_angle(sample_angle), random);
  Float3 value = make_float3(0.0f);
  Float pdf = cone.pdf;
  UInt label = cycles_closure::label_none;
  $if(dot(Ng, cone.direction) > 0.0f) {
    const auto angle = acos(cone.cosine);
    value = make_float3(
        pdf * toon_intensity(closure.param.size, closure.param.smooth, angle));
    label = cycles_closure::label_reflect | cycles_closure::label_diffuse;
  }
  $else { pdf = 0.0f; };
  return {.value = value,
          .wo = cone.direction,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          .label = label};
}

BsdfEvaluation bsdf_glossy_toon_eval(const ToonClosure &closure,
                                     Expr<luisa::float3> wi,
                                     Expr<luisa::float3> wo) noexcept {
  Float3 value = make_float3(0.0f);
  Float pdf = 0.0f;
  const auto incoming_cosine = dot(closure.common.N, wi);
  const auto outgoing_cosine = dot(closure.common.N, wo);
  $if((incoming_cosine > 0.0f) & (outgoing_cosine > 0.0f)) {
    const auto reflection = (2.0f * incoming_cosine) * closure.common.N - wi;
    const auto angle = safe_acos(max(dot(reflection, wo), 0.0f));
    const auto sample_angle = toon_sample_angle(closure);
    $if(angle < sample_angle) {
      pdf = cycles_sample_mapping::inverse_two_pi /
            cycles_sample_mapping::one_minus_cosine_from_angle(sample_angle);
      value = make_float3(pdf * toon_intensity(closure.param.size,
                                               closure.param.smooth, angle));
    };
  };
  return {.value = value, .pdf = pdf};
}

BsdfSample bsdf_glossy_toon_sample(const ToonClosure &closure,
                                   Expr<luisa::float3> Ng,
                                   Expr<luisa::float3> wi,
                                   Expr<luisa::float2> random) noexcept {
  Float3 value = make_float3(0.0f);
  Float3 wo = make_float3(0.0f);
  Float pdf = 0.0f;
  UInt label = cycles_closure::label_none;
  const auto incoming_cosine = dot(closure.common.N, wi);
  $if(incoming_cosine > 0.0f) {
    const auto reflection = (2.0f * incoming_cosine) * closure.common.N - wi;
    const auto sample_angle = toon_sample_angle(closure);
    const auto cone = cycles_sample_mapping::sample_uniform_cone(
        reflection,
        cycles_sample_mapping::one_minus_cosine_from_angle(sample_angle),
        random);
    wo = cone.direction;
    $if((dot(Ng, wo) > 0.0f) & (dot(closure.common.N, wo) > 0.0f)) {
      pdf = cone.pdf;
      value = make_float3(pdf * toon_intensity(closure.param.size,
                                               closure.param.smooth,
                                               acos(cone.cosine)));
      label = cycles_closure::label_glossy | cycles_closure::label_reflect;
    };
  };
  return {.value = value,
          .wo = wo,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          .label = label};
}

} // namespace psycles::luisa_backend::cycles_svm::detail
