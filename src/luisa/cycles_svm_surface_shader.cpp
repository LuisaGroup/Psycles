/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_surface_shader.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure = ::psycles::luisa_backend::cycles_closure;

namespace {

[[nodiscard]] SurfaceShaderBsdfEval zero_evaluation() noexcept {
  return {.diffuse = make_float3(0.0f),
          .glossy = make_float3(0.0f),
          .sum = make_float3(0.0f),
          .pdf = 0.0f,
          .average_roughness_squared = 0.0f};
}

void bsdf_eval_accum(SurfaceShaderBsdfEval &result,
                     const ShaderClosureCommon &common, Expr<luisa::float3> wo,
                     Expr<luisa::float3> value) noexcept {
  $if(closure::is_bsdf_diffuse(common.type)) { result.diffuse += value; }
  $elif(closure::is_bsdf_glossy(common.type)) { result.glossy += value; }
  $elif(closure::is_glass(common.type) & (dot(common.N, wo) > 0.0f)) {
    result.glossy += value;
  };
  result.sum += value;
}

[[nodiscard]] Bool
surface_shader_exclude(Expr<std::uint32_t> type,
                       Expr<std::uint32_t> light_shader_flags) noexcept {
  Bool result = false;
  $if((light_shader_flags & shader_exclude_any) != 0u) {
    $if(((light_shader_flags & shader_exclude_diffuse) != 0u) &
        closure::is_bsdf_diffuse(type)) {
      result = true;
    };
    $if(((light_shader_flags & shader_exclude_glossy) != 0u) &
        closure::is_bsdf_glossy(type)) {
      result = true;
    };
    $if(((light_shader_flags & shader_exclude_transmit) != 0u) &
        closure::is_bsdf_transmission(type)) {
      result = true;
    };
    constexpr auto exclude_glass =
        shader_exclude_transmit | shader_exclude_glossy;
    $if(((light_shader_flags & exclude_glass) == exclude_glass) &
        closure::is_glass(type)) {
      result = true;
    };
  };
  return result;
}

/* Cycles' one-sample model fold is an ordered reduction over the retained
 * closure prefix. For eligible closure i with selection mass s_i and
 * directional density p_i, the returned density is
 *
 *                 sum_i(s_i p_i) / sum_i(s_i).
 *
 * BSSRDFs contribute selection mass but no directional density. This same
 * reduction is used for NEE and for the complement of a sampled BSDF; the
 * latter seeds all four sums with the selected closure and skips its index.
 * That algebraic identity is the reason this helper accepts explicit seeds
 * instead of maintaining two nearly-identical implementations. */
[[nodiscard]] SurfaceShaderBsdfEval surface_shader_bsdf_eval_mis(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    Expr<luisa::float3> wo, Expr<std::uint32_t> skip_index, Expr<bool> has_skip,
    Expr<std::uint32_t> light_shader_flags, SurfaceShaderBsdfEval result,
    Expr<float> sum_pdf, Expr<float> sum_sample_weight,
    Expr<float> sum_pdf_roughness_squared,
    ClosureTypeMask closure_types) noexcept {
  const auto &pool = *shader_data.closure;
  Float accumulated_pdf = sum_pdf;
  Float accumulated_sample_weight = sum_sample_weight;
  Float accumulated_pdf_roughness_squared = sum_pdf_roughness_squared;
  UInt index = 0u;
  $while(index < pool.count()) {
    const auto common = pool.common(index);
    const auto eligible = closure::is_bsdf_or_bssrdf(common.type);
    $if(eligible & ((!has_skip) | (index != skip_index))) {
      $if(closure::is_bsdf(common.type)) {
        const auto evaluation =
            bsdf_eval(kernel_globals, shader_data, index, wo, closure_types);
        $if(evaluation.pdf != 0.0f) {
          $if(!surface_shader_exclude(common.type, light_shader_flags)) {
            bsdf_eval_accum(result, common, wo,
                            evaluation.value * common.weight);
          };
          const auto weighted_pdf = evaluation.pdf * common.sample_weight;
          accumulated_pdf += weighted_pdf;
          accumulated_pdf_roughness_squared +=
              weighted_pdf * bsdf_get_specular_roughness_squared(pool, index);
        };
      };
      accumulated_sample_weight += common.sample_weight;
    };
    index += 1u;
  };
  result.average_roughness_squared =
      select(0.0f, accumulated_pdf_roughness_squared / accumulated_pdf,
             accumulated_pdf > 0.0f);
  result.pdf = select(0.0f, accumulated_pdf / accumulated_sample_weight,
                      accumulated_sample_weight > 0.0f);
  return result;
}

} // namespace

SurfaceShaderBsdfEval
surface_shader_bsdf_eval(const KernelGlobals &kernel_globals,
                         ShaderData &shader_data, Expr<luisa::float3> wo,
                         Expr<std::uint32_t> light_shader_flags,
                         ClosureTypeMask closure_types) noexcept {
  auto result = surface_shader_bsdf_eval_mis(
      kernel_globals, shader_data, wo, 0u, false, light_shader_flags,
      zero_evaluation(), 0.0f, 0.0f, 0.0f, closure_types);
  result.pdf =
      select(0.0f, result.pdf, (light_shader_flags & shader_use_mis) != 0u);
  return result;
}

SurfaceShaderClosurePick
surface_shader_bsdf_bssrdf_pick(const ShaderData &shader_data,
                                Expr<luisa::float3> random) noexcept {
  const auto &pool = *shader_data.closure;
  UInt sampled = 0u;
  Float3 reused_random = random;

  /* This intentionally branches on the total retained closure count rather
   * than the eligible count. It is the exact Cycles fast-path condition. */
  $if(pool.count() > 1u) {
    Float sum_sample_weight = 0.0f;
    UInt index = 0u;
    $while(index < pool.count()) {
      const auto common = pool.common(index);
      $if(closure::is_bsdf_or_bssrdf(common.type)) {
        sum_sample_weight += common.sample_weight;
      };
      index += 1u;
    };

    const auto target = random.z * sum_sample_weight;
    Float partial_sum = 0.0f;
    index = 0u;
    $while(index < pool.count()) {
      const auto common = pool.common(index);
      $if(closure::is_bsdf_or_bssrdf(common.type)) {
        const auto next_sum = partial_sum + common.sample_weight;
        $if(target < next_sum) {
          sampled = index;
          reused_random = make_float3(random.xy(), (target - partial_sum) /
                                                       common.sample_weight);
          $break;
        };
        partial_sum = next_sum;
      };
      index += 1u;
    };
  };

  return {.index = sampled, .random = reused_random};
}

Float3 surface_shader_bssrdf_sample_weight(
    const ShaderData &shader_data, Expr<std::uint32_t> closure_index) noexcept {
  const auto &pool = *shader_data.closure;
  const auto selected = pool.common(closure_index);
  Float3 weight = selected.weight;
  $if(pool.count() > 1u) {
    Float sum_sample_weight = 0.0f;
    UInt index = 0u;
    $while(index < pool.count()) {
      const auto common = pool.common(index);
      $if(closure::is_bsdf_or_bssrdf(common.type)) {
        sum_sample_weight += common.sample_weight;
      };
      index += 1u;
    };
    weight *= sum_sample_weight / selected.sample_weight;
  };
  return weight;
}

SurfaceShaderBsdfSample
surface_shader_bsdf_sample_closure(const KernelGlobals &kernel_globals,
                                   ShaderData &shader_data,
                                   const SurfaceShaderClosurePick &pick,
                                   ClosureTypeMask closure_types) noexcept {
  const auto &pool = *shader_data.closure;
  const auto common = pool.common(pick.index);
  const auto sampled = bsdf_sample(kernel_globals, shader_data, pick.index,
                                   pick.random, closure_types);
  auto result =
      SurfaceShaderBsdfSample{.evaluation = zero_evaluation(),
                              .wo = sampled.wo,
                              .sampled_roughness = sampled.sampled_roughness,
                              .eta = sampled.eta,
                              .label = sampled.label};

  $if(sampled.pdf != 0.0f) {
    const auto weighted_value = sampled.value * common.weight;
    bsdf_eval_accum(result.evaluation, common, sampled.wo, weighted_value);
    const auto roughness_squared =
        bsdf_get_specular_roughness_squared(pool, pick.index);
    $if(pool.count() > 1u) {
      const auto weighted_pdf = sampled.pdf * common.sample_weight;
      result.evaluation = surface_shader_bsdf_eval_mis(
          kernel_globals, shader_data, sampled.wo, pick.index, true, 0u,
          result.evaluation, weighted_pdf, common.sample_weight,
          weighted_pdf * roughness_squared, closure_types);
    }
    $else {
      result.evaluation.pdf = sampled.pdf;
      result.evaluation.average_roughness_squared = roughness_squared;
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
