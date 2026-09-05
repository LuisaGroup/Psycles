/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_sheen.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace table_detail = ::psycles::luisa_backend::detail;

namespace {

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

struct SheenSetupResult {
  SheenParam param;
  Float albedo;
  Bool invalid;
};

[[nodiscard]] SheenSetupResult
sheen_setup_result(const KernelGlobals &kernel_globals,
                   const ShaderData &shader_data,
                   Expr<luisa::float3> normal,
                   Expr<float> roughness) noexcept {
  const auto canonical_roughness = clamp(roughness, 1.0e-3f, 1.0f);
  const auto basis = cycles_sample_mapping::make_orthonormals_safe_tangent(
      normal, shader_data.wi);
  const auto cosine = dot(normal, shader_data.wi);
  const auto transform_a = table_detail::cycles_table_2d(
      kernel_globals, cosine, canonical_roughness,
      UInt{cycles45_tables::sheen_ltc_offset}, 32u, 32u);
  const auto transform_b = table_detail::cycles_table_2d(
      kernel_globals, cosine, canonical_roughness,
      UInt{cycles45_tables::sheen_ltc_offset + 32u * 32u}, 32u, 32u);
  const auto albedo = table_detail::cycles_table_2d(
      kernel_globals, cosine, canonical_roughness,
      UInt{cycles45_tables::sheen_ltc_offset + 2u * 32u * 32u}, 32u, 32u);
  return {
      .param = {.roughness = canonical_roughness,
                .transform_a = transform_a,
                .transform_b = transform_b,
                .T = basis.tangent,
                .B = basis.bitangent},
      .albedo = albedo,
      /* Preserve Cycles' comparison form. With a malformed NaN table entry,
       * neither less-than predicate fires; changing this to >= would define a
       * different state transition. Scene tables themselves are finite. */
      .invalid = (abs(transform_a) < 1.0e-5f) | (albedo < 1.0e-5f)};
}

} // namespace

void node_sheen(const KernelGlobals &kernel_globals, Cursor &cursor,
                Stack &stack, Expr<std::uint32_t> type,
                Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
                ShaderData &shader_data) noexcept {
  const auto roughness_input = cursor.word();
  const auto normal_packed = cursor.word();
  if (shader_data.closure == nullptr) {
    return;
  }
  const auto normal_offset = cursor.byte(normal_packed, 0u);
  auto normal =
      stack_load_float3_default(stack, normal_offset, shader_data.N);
  normal = native_vector_math::safe_normalize_nonzero_or(normal, shader_data.N);

  auto &pool = *shader_data.closure;
  const auto allocated =
      bsdf_allocate(shader_data, closure_weight * mix_weight);
  $if(allocated.valid) {
    pool.set_normal(allocated.index, normal);
    const auto roughness =
        clamp(stack_load_input_float(stack, roughness_input), 0.0f, 1.0f);
    $if(type == static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID)) {
      const auto setup =
          sheen_setup_result(kernel_globals, shader_data, normal, roughness);
      pool.set_type(allocated.index,
                    static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID));
      pool.set_sheen_param(allocated.index, setup.param);
      $if(setup.invalid) {
        pool.set_type(allocated.index,
                      static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        pool.set_sample_weight(allocated.index, 0.0f);
      }
      $else {
        const auto common = pool.common(allocated.index);
        pool.set_weight(allocated.index, common.weight * setup.albedo);
        pool.set_sample_weight(allocated.index,
                               common.sample_weight * setup.albedo);
        shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
      };
    }
    $else {
      const auto canonical_sigma = max(roughness, 0.01f);
      pool.set_velvet_param(
          allocated.index,
          {.sigma = roughness,
           .invsigma2 = 1.0f / (canonical_sigma * canonical_sigma)});
      pool.set_type(
          allocated.index,
          static_cast<std::uint32_t>(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID));
      shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    };
  };
}

Float3 principled_sheen_setup(const KernelGlobals &kernel_globals,
                              ShaderData &shader_data,
                              const PathState &path_state,
                              Expr<luisa::float3> input_weight,
                              Expr<luisa::float3> normal,
                              Expr<float> roughness) noexcept {
  const Float3 weight = max(input_weight, make_float3(0.0f));
  const Float sample_weight = abs(average(weight));
  const Bool survives_cutoff =
      ((sample_weight >= CLOSURE_WEIGHT_CUTOFF) |
       ((shader_data.flag & shader_data_is_volume_shader_eval) != 0u)) &
      (sample_weight > 0.0f);
  const Bool emission_path = (path_state.flag & path_ray_emission) != 0u;

  ClosurePool::Allocation allocated{.index = 0u, .valid = false};
  Bool setup_active = false;
  $if(emission_path) { setup_active = survives_cutoff; }
  $else {
    allocated = bsdf_allocate(shader_data, weight);
    setup_active = allocated.valid;
  };

  Float3 layer_albedo = make_float3(0.0f);
  $if(setup_active) {
    const auto setup =
        sheen_setup_result(kernel_globals, shader_data, normal, roughness);

    if (shader_data.closure != nullptr) {
      auto &pool = *shader_data.closure;
      $if(!emission_path) {
        pool.set_normal(allocated.index, normal);
        pool.set_sheen_param(allocated.index, setup.param);
      };
    }

    $if(setup.invalid) {
      if (shader_data.closure != nullptr) {
        $if(!emission_path) {
          /* bsdf_alloc already wrote CLOSURE_NONE. Sheen setup clears only the
           * sample weight and leaves the consumed common slot observable. */
          shader_data.closure->set_sample_weight(allocated.index, 0.0f);
        };
      }
    }
    $else {
      layer_albedo = weight * setup.albedo;
      if (shader_data.closure != nullptr) {
        auto &pool = *shader_data.closure;
        $if(!emission_path) {
          pool.set_type(allocated.index,
                        static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID));
          pool.set_weight(allocated.index, layer_albedo);
          pool.set_sample_weight(allocated.index,
                                 sample_weight * setup.albedo);
        };
      }
      shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    };
  };
  return layer_albedo;
}

BsdfEvaluation bsdf_sheen_eval(const SheenClosure &closure,
                               Expr<luisa::float3>,
                               Expr<luisa::float3> wo) noexcept {
  const auto local_outgoing = make_float3(dot(wo, closure.param.T),
                                          dot(wo, closure.param.B),
                                          dot(wo, closure.common.N));
  const auto a = closure.param.transform_a;
  const auto b = closure.param.transform_b;
  const auto length_squared = square(a * local_outgoing.x +
                                     b * local_outgoing.z) +
                              square(a * local_outgoing.y) +
                              square(local_outgoing.z);
  const auto value = cycles_sample_mapping::inverse_pi *
                     max(local_outgoing.z, 0.0f) *
                     square(a / length_squared);
  return {.value = make_float3(value), .pdf = value};
}

BsdfSample bsdf_sheen_sample(const SheenClosure &closure,
                             Expr<luisa::float3> Ng, Expr<luisa::float3>,
                             Expr<luisa::float2> random) noexcept {
  const auto disk = cycles_sample_mapping::sample_uniform_disk(random);
  const auto disk_z = sqrt(max(1.0f - dot(disk, disk), 0.0f));
  const auto a = closure.param.transform_a;
  const auto b = closure.param.transform_b;
  const auto local_outgoing = native_vector_math::normalize_unchecked(
      make_float3(disk.x - disk_z * b, disk.y, disk_z * a));
  const auto wo = local_outgoing.x * closure.param.T +
                  local_outgoing.y * closure.param.B +
                  local_outgoing.z * closure.common.N;

  Float3 value = make_float3(0.0f);
  Float pdf = 0.0f;
  /* Keep the negated Cycles rejection predicate. A NaN geometric-normal
   * dot product does not satisfy `<= 0` and therefore falls through to the
   * arithmetic path in Cycles. */
  $if(!(dot(Ng, wo) <= 0.0f)) {
    const auto length_squared = square(a * local_outgoing.x +
                                       b * local_outgoing.z) +
                                square(a * local_outgoing.y) +
                                square(local_outgoing.z);
    pdf = cycles_sample_mapping::inverse_pi * local_outgoing.z *
          square(a / length_squared);
    value = make_float3(pdf);
  };
  return {.value = value,
          .wo = wo,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          /* Cycles returns REFLECT|DIFFUSE even when Ng rejects the sample. */
          .label = cycles_closure::label_reflect |
                   cycles_closure::label_diffuse};
}

BsdfEvaluation bsdf_ashikhmin_velvet_eval(const VelvetClosure &closure,
                                          Expr<luisa::float3> wi,
                                          Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto cos_ni = dot(closure.common.N, wi);
  const auto cos_no = dot(closure.common.N, wo);
  $if((cos_ni > 0.0f) & (cos_no > 0.0f)) {
    const auto half_vector =
        native_vector_math::normalize_unchecked(wi + wo);
    const auto cos_nh = dot(closure.common.N, half_vector);
    const auto cos_hi = abs(dot(wi, half_vector));
    $if((abs(cos_nh) < 1.0f - 1.0e-5f) & (cos_hi > 1.0e-5f)) {
      const auto cos_nh_over_hi = max(cos_nh / cos_hi, 1.0e-5f);
      const auto fac1 = 2.0f * abs(cos_nh_over_hi * cos_ni);
      const auto fac2 = 2.0f * abs(cos_nh_over_hi * cos_no);
      const auto sin_nh_squared = 1.0f - cos_nh * cos_nh;
      const auto sin_nh_fourth = sin_nh_squared * sin_nh_squared;
      const auto cotangent_squared =
          (cos_nh * cos_nh) / sin_nh_squared;
      const auto distribution =
          exp(-cotangent_squared * closure.param.invsigma2) *
          closure.param.invsigma2 * cycles_sample_mapping::inverse_pi /
          sin_nh_fourth;
      const auto masking = luisa::compute::min(
          1.0f, luisa::compute::min(fac1, fac2));
      const auto response = 0.25f * (distribution * masking) / cos_ni;
      result.value = make_float3(response);
      result.pdf = cycles_sample_mapping::inverse_two_pi;
    };
  };
  return result;
}

BsdfSample bsdf_ashikhmin_velvet_sample(
    const VelvetClosure &closure, Expr<luisa::float3> Ng,
    Expr<luisa::float3> wi, Expr<luisa::float2> random) noexcept {
  const auto hemisphere = cycles_sample_mapping::sample_uniform_hemisphere(
      closure.common.N, random);
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = hemisphere.direction,
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(1.0f),
                    .eta = 1.0f,
                    .label = cycles_closure::label_none};
  $if(dot(Ng, result.wo) > 0.0f) {
    const auto half_vector =
        native_vector_math::normalize_unchecked(wi + result.wo);
    const auto cos_ni = dot(closure.common.N, wi);
    const auto cos_no = dot(closure.common.N, result.wo);
    const auto cos_hi = abs(dot(wi, half_vector));
    const auto cos_nh = dot(closure.common.N, half_vector);
    $if((cos_ni > 1.0e-5f) &
        (abs(cos_nh) < 1.0f - 1.0e-5f) & (cos_hi > 1.0e-5f)) {
      const auto cos_nh_over_hi = max(cos_nh / cos_hi, 1.0e-5f);
      const auto fac1 = 2.0f * abs(cos_nh_over_hi * cos_ni);
      const auto fac2 = 2.0f * abs(cos_nh_over_hi * cos_no);
      const auto sin_nh_squared = 1.0f - cos_nh * cos_nh;
      const auto sin_nh_fourth = sin_nh_squared * sin_nh_squared;
      const auto cotangent_squared =
          (cos_nh * cos_nh) / sin_nh_squared;
      const auto distribution =
          exp(-cotangent_squared * closure.param.invsigma2) *
          closure.param.invsigma2 * cycles_sample_mapping::inverse_pi /
          sin_nh_fourth;
      const auto masking = luisa::compute::min(
          1.0f, luisa::compute::min(fac1, fac2));
      const auto response = 0.25f * (distribution * masking) / cos_ni;
      result.value = make_float3(response);
      result.pdf = hemisphere.pdf;
      result.label = cycles_closure::label_reflect |
                     cycles_closure::label_diffuse;
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
