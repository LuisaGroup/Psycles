/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_hair.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_fast_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace fast_math = luisa_backend::cycles_fast_math;

void node_hair(Cursor &cursor, Stack &stack, Expr<std::uint32_t> type,
               Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
               ShaderData &shader_data) noexcept {
  /* svm_node_get consumes the complete typed payload before bsdf_alloc. This
   * is observable when the weight is cut off or the closure pool is full. */
  const auto roughness1_input = cursor.word();
  const auto roughness2_input = cursor.word();
  const auto offset_input = cursor.word();
  const auto tangent_packed = cursor.word();
  if (shader_data.closure == nullptr) {
    return;
  }
  const auto tangent_offset = cursor.byte(tangent_packed, 0u);

  const auto allocated =
      bsdf_allocate(shader_data, closure_weight * mix_weight);
  $if(allocated.valid) {
    auto &pool = *shader_data.closure;
    const auto normal =
        maybe_ensure_valid_specular_reflection(shader_data, shader_data.N);
    const auto roughness1 = stack_load_input_float(stack, roughness1_input);
    const auto roughness2 = stack_load_input_float(stack, roughness2_input);
    Float offset = -stack_load_input_float(stack, offset_input);
    Float3 tangent;

    $if(tangent_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      tangent = normalize_cycles(stack_load_float3(stack, tangent_offset));
    }
    $elif((shader_data.type & primitive_curve) == 0u) {
      tangent = normalize_cycles(shader_data.dPdv);
      offset = 0.0f;
    }
    $else { tangent = normalize_cycles(shader_data.dPdu); };

    pool.set_type(allocated.index, type);
    pool.set_normal(allocated.index, normal);
    pool.set_hair_param(allocated.index,
                        {.T = tangent,
                         .roughness1 = clamp(roughness1, 0.001f, 1.0f),
                         .roughness2 = clamp(roughness2, 0.001f, 1.0f),
                         .offset = offset});
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    $if(type == static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_TRANSMISSION_ID)) {
      shader_data.flag |= shader_data_bsdf_has_transmission;
    };
  };
}

BsdfEvaluation bsdf_hair_reflection_eval(const HairClosure &closure,
                                         Expr<luisa::float3> wi,
                                         Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  $if(dot(closure.common.N, wo) < 0.0f) {
    result.value = make_float3(0.0f);
    result.pdf = 0.0f;
  }
  $else {
    const auto offset = closure.param.offset;
    const auto tangent = closure.param.T;
    const auto roughness1 = closure.param.roughness1;
    const auto roughness2 = closure.param.roughness2;

    const auto incoming_z = dot(tangent, wi);
    const auto local_y = normalize_cycles(wi - tangent * incoming_z);
    const auto theta_r = fast_math::half_pi - fast_math::arc_cosine(incoming_z);

    const auto outgoing_z = dot(tangent, wo);
    const auto outgoing_y = normalize_cycles(wo - tangent * outgoing_z);
    const auto theta_i = fast_math::half_pi - fast_math::arc_cosine(outgoing_z);
    const auto cosine_phi_i = dot(outgoing_y, local_y);

    $if((fast_math::half_pi - abs(theta_i) < 0.001f) | (cosine_phi_i < 0.0f)) {
      result.value = make_float3(0.0f);
      result.pdf = 0.0f;
    }
    $else {
      const auto roughness1_inverse = 1.0f / roughness1;
      const auto roughness2_inverse = 1.0f / roughness2;
      Float phi_i = fast_math::arc_cosine(cosine_phi_i) * roughness2_inverse;
      phi_i = select(fast_math::pi, phi_i, abs(phi_i) < fast_math::pi);
      const auto cosine_theta_i = fast_math::cosine(theta_i);
      const auto a = fast_math::arc_tangent2(
          ((fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
          1.0f);
      const auto b = fast_math::arc_tangent2(
          ((-fast_math::half_pi + theta_r) * 0.5f - offset) *
              roughness1_inverse,
          1.0f);
      const auto theta_half = (theta_i + theta_r) * 0.5f;
      const auto longitudinal = theta_half - offset;
      const auto phi_pdf =
          fast_math::cosine(phi_i * 0.5f) * 0.25f * roughness2_inverse;
      const auto theta_pdf =
          roughness1 /
          (2.0f * (longitudinal * longitudinal + roughness1 * roughness1) *
           (a - b) * cosine_theta_i);
      result.pdf = phi_pdf * theta_pdf;
      result.value = make_float3(result.pdf);
    };
  };
  return result;
}

BsdfEvaluation bsdf_hair_transmission_eval(const HairClosure &closure,
                                           Expr<luisa::float3> wi,
                                           Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  $if(dot(closure.common.N, wo) >= 0.0f) {
    result.value = make_float3(0.0f);
    result.pdf = 0.0f;
  }
  $else {
    const auto offset = closure.param.offset;
    const auto tangent = closure.param.T;
    const auto roughness1 = closure.param.roughness1;
    const auto roughness2 = closure.param.roughness2;

    const auto incoming_z = dot(tangent, wi);
    const auto local_y = normalize_cycles(wi - tangent * incoming_z);
    const auto theta_r = fast_math::half_pi - fast_math::arc_cosine(incoming_z);

    const auto outgoing_z = dot(tangent, wo);
    const auto outgoing_y = normalize_cycles(wo - tangent * outgoing_z);
    const auto theta_i = fast_math::half_pi - fast_math::arc_cosine(outgoing_z);
    const auto phi_i = fast_math::arc_cosine(dot(outgoing_y, local_y));

    $if(fast_math::half_pi - abs(theta_i) < 0.001f) {
      result.value = make_float3(0.0f);
      result.pdf = 0.0f;
    }
    $else {
      const auto cosine_theta_i = fast_math::cosine(theta_i);
      const auto roughness1_inverse = 1.0f / roughness1;
      const auto a = fast_math::arc_tangent2(
          ((fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
          1.0f);
      const auto b = fast_math::arc_tangent2(
          ((-fast_math::half_pi + theta_r) * 0.5f - offset) *
              roughness1_inverse,
          1.0f);
      const auto c =
          2.0f * fast_math::arc_tangent2(fast_math::half_pi / roughness2, 1.0f);
      const auto theta_half = (theta_i + theta_r) * 0.5f;
      const auto longitudinal = theta_half - offset;
      const auto azimuth = abs(phi_i);
      const auto azimuth_offset = fast_math::pi - azimuth;
      const auto theta_pdf =
          roughness1 /
          (2.0f * (longitudinal * longitudinal + roughness1 * roughness1) *
           (a - b) * cosine_theta_i);
      const auto phi_pdf =
          roughness2 /
          (c * (azimuth_offset * azimuth_offset + roughness2 * roughness2));
      result.pdf = phi_pdf * theta_pdf;
      result.value = make_float3(result.pdf);
    };
  };
  return result;
}

BsdfSample bsdf_hair_reflection_sample(const HairClosure &closure,
                                       Expr<luisa::float3>,
                                       Expr<luisa::float3> wi,
                                       Expr<luisa::float2> random) noexcept {
  const auto offset = closure.param.offset;
  const auto tangent = closure.param.T;
  const auto roughness1 = closure.param.roughness1;
  const auto roughness2 = closure.param.roughness2;
  const auto incoming_z = dot(tangent, wi);
  const auto local_y = normalize_cycles(wi - tangent * incoming_z);
  const auto local_x = cross(local_y, tangent);
  const auto theta_r = fast_math::half_pi - fast_math::arc_cosine(incoming_z);
  const auto roughness1_inverse = 1.0f / roughness1;
  const auto a = fast_math::arc_tangent2(
      ((fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
      1.0f);
  const auto b = fast_math::arc_tangent2(
      ((-fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
      1.0f);
  const auto longitudinal = roughness1 * tan(random.x * (a - b) + b);
  const auto theta_half = longitudinal + offset;
  const auto theta_i = 2.0f * theta_half - theta_r;
  const auto theta_trigonometry = fast_math::sine_cosine(theta_i);
  const auto phi =
      2.0f * fast_math::safe_arc_sine(1.0f - 2.0f * random.y) * roughness2;
  const auto phi_pdf = fast_math::cosine(phi * 0.5f) * 0.25f / roughness2;
  const auto theta_pdf =
      roughness1 /
      (2.0f * (longitudinal * longitudinal + roughness1 * roughness1) *
       (a - b) * theta_trigonometry.cosine);
  const auto phi_trigonometry = fast_math::sine_cosine(phi);
  const auto wo =
      (phi_trigonometry.cosine * theta_trigonometry.cosine) * local_y -
      (phi_trigonometry.sine * theta_trigonometry.cosine) * local_x +
      theta_trigonometry.sine * tangent;
  Float pdf = abs(phi_pdf * theta_pdf);
  $if(fast_math::half_pi - abs(theta_i) < 0.001f) { pdf = 0.0f; };
  return {.value = make_float3(pdf),
          .wo = wo,
          .pdf = pdf,
          .sampled_roughness = make_float2(roughness1, roughness2),
          .eta = 1.0f,
          .label =
              cycles_closure::label_reflect | cycles_closure::label_glossy};
}

BsdfSample bsdf_hair_transmission_sample(const HairClosure &closure,
                                         Expr<luisa::float3>,
                                         Expr<luisa::float3> wi,
                                         Expr<luisa::float2> random) noexcept {
  const auto offset = closure.param.offset;
  const auto tangent = closure.param.T;
  const auto roughness1 = closure.param.roughness1;
  const auto roughness2 = closure.param.roughness2;
  const auto incoming_z = dot(tangent, wi);
  const auto local_y = normalize_cycles(wi - tangent * incoming_z);
  const auto local_x = cross(local_y, tangent);
  const auto theta_r = fast_math::half_pi - fast_math::arc_cosine(incoming_z);
  const auto roughness1_inverse = 1.0f / roughness1;
  const auto a = fast_math::arc_tangent2(
      ((fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
      1.0f);
  const auto b = fast_math::arc_tangent2(
      ((-fast_math::half_pi + theta_r) * 0.5f - offset) * roughness1_inverse,
      1.0f);
  const auto c =
      2.0f * fast_math::arc_tangent2(fast_math::half_pi / roughness2, 1.0f);
  const auto longitudinal = roughness1 * tan(random.x * (a - b) + b);
  const auto theta_half = longitudinal + offset;
  const auto theta_i = 2.0f * theta_half - theta_r;
  const auto theta_trigonometry = fast_math::sine_cosine(theta_i);
  const auto azimuth_offset = roughness2 * tan(c * (random.y - 0.5f));
  const auto phi = azimuth_offset + fast_math::pi;
  const auto theta_pdf =
      roughness1 /
      (2.0f * (longitudinal * longitudinal + roughness1 * roughness1) *
       (a - b) * theta_trigonometry.cosine);
  const auto phi_pdf =
      roughness2 /
      (c * (azimuth_offset * azimuth_offset + roughness2 * roughness2));
  const auto phi_trigonometry = fast_math::sine_cosine(phi);
  const auto wo =
      (phi_trigonometry.cosine * theta_trigonometry.cosine) * local_y -
      (phi_trigonometry.sine * theta_trigonometry.cosine) * local_x +
      theta_trigonometry.sine * tangent;
  Float pdf = abs(phi_pdf * theta_pdf);
  $if(fast_math::half_pi - abs(theta_i) < 0.001f) { pdf = 0.0f; };
  return {.value = make_float3(pdf),
          .wo = wo,
          .pdf = pdf,
          .sampled_roughness = make_float2(roughness1, roughness2),
          .eta = 1.0f,
          .label =
              cycles_closure::label_transmit | cycles_closure::label_glossy};
}

} // namespace psycles::luisa_backend::cycles_svm::detail
