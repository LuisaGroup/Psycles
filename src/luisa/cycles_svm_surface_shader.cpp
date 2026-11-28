/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_surface_shader.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure = ::psycles::luisa_backend::cycles_closure;

Float3 surface_shader_apply_holdout(ShaderData &sd) noexcept {
  namespace abi = compiler::cycles_svm;
  Float3 weight = make_float3(0.0f);
  $if((sd.object_flag & unsigned(abi::SD_OBJECT_HOLDOUT_MASK)) != 0u) {
    $if(((sd.flag & unsigned(abi::SD_TRANSPARENT)) != 0u) &
        ((sd.flag & unsigned(abi::SD_HAS_ONLY_VOLUME)) == 0u)) {
      weight = make_float3(1.0f) - sd.closure_transparent_extinction;
      if (sd.closure != nullptr) {
        UInt index = 0u;
        $while(index < sd.closure->count()) {
          $if(sd.closure->common(index).type != unsigned(abi::CLOSURE_BSDF_TRANSPARENT_ID)) {
            sd.closure->set_type(index, unsigned(abi::NBUILTIN_CLOSURES));
          };
          index += 1u;
        };
      }
      // Keep the original integer subtraction, not a set-difference rewrite.
      // Cycles 5.2.1 SD_CLOSURE_FLAGS does not include SD_TRANSPARENT.
      sd.flag &= ~unsigned(abi::SD_CLOSURE_FLAGS - (abi::SD_TRANSPARENT | abi::SD_BSDF));
    }
    $else { weight = make_float3(1.0f); };
  }
  $else {
    if (sd.closure != nullptr) {
      UInt index = 0u;
      $while(index < sd.closure->count()) {
        const auto sc = sd.closure->common(index);
        $if(sc.type == unsigned(abi::CLOSURE_HOLDOUT_ID)) { weight += sc.weight; };
        index += 1u;
      };
    }
  };
  return weight;
}

namespace {

[[nodiscard]] SurfaceShaderBsdfEval zero_evaluation(
    bool spectral = false, bool lane_pdf = false) noexcept {
  SurfaceShaderBsdfEval out{.diffuse = make_float3(0.0f),
                            .glossy = make_float3(0.0f),
                            .sum = make_float3(0.0f),
                            .pdf = 0.0f,
                            .average_roughness_squared = 0.0f};
  if (spectral) {
    out.spectral_diffuse.emplace(make_float4(0.0f));
    out.spectral_glossy.emplace(make_float4(0.0f));
    out.spectral_sum.emplace(make_float4(0.0f));
    if (lane_pdf) {
      out.spectral_lane_pdf.emplace(make_float4(0.0f));
    }
  }
  return out;
}

template <typename Vec>
void bsdf_eval_accum(SurfaceShaderBsdfEval &result,
                     const ShaderClosureCommon &common, Expr<luisa::float3> wo,
                     const Vec &value) noexcept {
  if constexpr (std::is_same_v<Vec, Float4>) {
    $if(closure::is_bsdf_diffuse(common.type)) { *result.spectral_diffuse += value; }
    $elif(closure::is_bsdf_glossy(common.type)) { *result.spectral_glossy += value; }
    $elif(closure::is_glass(common.type) & (dot(common.N, wo) > 0.0f)) {
      *result.spectral_glossy += value;
    };
    *result.spectral_sum += value;
  } else {
    $if(closure::is_bsdf_diffuse(common.type)) { result.diffuse += value; }
    $elif(closure::is_bsdf_glossy(common.type)) { result.glossy += value; }
    $elif(closure::is_glass(common.type) & (dot(common.N, wo) > 0.0f)) {
      result.glossy += value;
    };
    result.sum += value;
  }
}

[[nodiscard]] Float4 closure_spectral_weight(
    const ClosurePool &pool, Expr<std::uint32_t> index,
    const ShaderClosureCommon &common,
    const cycles_spectrum::SpectralContext &spectral) noexcept {
  const Float4 upsampled = cycles_spectrum::rgb_to_spectrum_reflectance(
      *spectral.tables, common.weight, spectral.wavelengths.lambdas);
  if (!pool.is_spectral()) {
    return upsampled;
  }
  return select(upsampled, pool.spectral_weight(index),
                pool.has_spectral_weight(index));
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
    ClosureTypeMask closure_types,
    std::optional<Float4> initial_lane_pdf = std::nullopt) noexcept {
  const auto &pool = *shader_data.closure;
  Float accumulated_pdf = sum_pdf;
  Float accumulated_sample_weight = sum_sample_weight;
  Float accumulated_pdf_roughness_squared = sum_pdf_roughness_squared;
  const bool spectral =
      kernel_globals.is_spectral() && shader_data.spectral.has_value();
  std::optional<Float4> accumulated_lane_pdf;
  if (spectral && result.spectral_lane_pdf.has_value()) {
    accumulated_lane_pdf.emplace(initial_lane_pdf.value_or(make_float4(0.0f)));
  }
  const auto eval_slot = [&](Expr<std::uint32_t> index) noexcept {
    const auto common = pool.common(index);
    const auto eligible = closure::is_bsdf_or_bssrdf(common.type);
    $if(eligible & ((!has_skip) | (index != skip_index))) {
      $if(closure::is_bsdf(common.type)) {
        if (spectral) {
          const auto &sc = *shader_data.spectral;
          const auto evaluation = bsdf_eval_spectral(
              kernel_globals, shader_data, index, wo, sc, closure_types);
          $if(evaluation.pdf != 0.0f) {
            $if(!surface_shader_exclude(common.type, light_shader_flags)) {
              bsdf_eval_accum(
                  result, common, wo,
                  evaluation.value *
                      closure_spectral_weight(pool, index, common, sc));
            };
            const auto weighted_pdf = evaluation.pdf * common.sample_weight;
            accumulated_pdf += weighted_pdf;
            if (accumulated_lane_pdf.has_value()) {
              *accumulated_lane_pdf +=
                  evaluation.lane_pdf.value_or(make_float4(evaluation.pdf)) *
                  common.sample_weight;
            }
            accumulated_pdf_roughness_squared +=
                weighted_pdf * bsdf_get_specular_roughness_squared(pool, index);
          };
        } else {
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
        }
      };
      accumulated_sample_weight += common.sample_weight;
    };
  };
  if (pool.capacity() <= 1u) {
    $if(pool.count() != 0u) { eval_slot(0u); };
  } else {
    UInt index = 0u;
    $while(index < pool.count()) {
      eval_slot(index);
      index += 1u;
    };
  }
  result.average_roughness_squared =
      select(0.0f, accumulated_pdf_roughness_squared / accumulated_pdf,
             accumulated_pdf > 0.0f);
  result.pdf = select(0.0f, accumulated_pdf / accumulated_sample_weight,
                      accumulated_sample_weight > 0.0f);
  if (result.spectral_lane_pdf.has_value() &&
      accumulated_lane_pdf.has_value()) {
    *result.spectral_lane_pdf =
        select(make_float4(0.0f),
               *accumulated_lane_pdf / accumulated_sample_weight,
               accumulated_sample_weight > 0.0f);
  }
  return result;
}

} // namespace

void surface_shader_initialize_closures(
    ClosurePool &pool, Expr<std::uint32_t> path_visibility,
    Expr<std::uint32_t> path_flag) noexcept {
  const auto stores_closures =
      ((path_visibility & path_ray_visibility_shadow) == 0u) &
      ((path_flag & (path_ray_terminate | path_ray_emission)) == 0u);
  pool.set_left(select(
      0u, static_cast<std::uint32_t>(pool.capacity()), stores_closures));
}

void surface_shader_prepare_closures(
    ShaderData &shader_data, Expr<float> blur_roughness,
    ClosureTypeMask closure_types) noexcept {
  $if(blur_roughness > 0.0f) {
    auto &pool = *shader_data.closure;
    if (pool.capacity() <= 1u) {
      $if(pool.count() != 0u) {
        const auto t = pool.closure_type(0u);
        $if(closure::is_bsdf(t)) {
          bsdf_blur(pool, 0u, blur_roughness, closure_types);
        };
      };
    } else {
      UInt index = 0u;
      $while(index < pool.count()) {
        const auto t = pool.closure_type(index);
        $if(closure::is_bsdf(t)) {
          bsdf_blur(pool, index, blur_roughness, closure_types);
        };
        index += 1u;
      };
    }
    shader_data.flag |= select(
        0u, shader_data_bsdf_has_eval,
        blur_roughness * blur_roughness >
            closure::microfacet_singular_alpha_product);
  };
}

SurfaceShaderBsdfEval
surface_shader_bsdf_eval(const KernelGlobals &kernel_globals,
                         ShaderData &shader_data, Expr<luisa::float3> wo,
                         Expr<std::uint32_t> light_shader_flags,
                         ClosureTypeMask closure_types) noexcept {
  const bool spectral =
      kernel_globals.is_spectral() && shader_data.spectral.has_value();
  const bool lane_pdf = spectral && kernel_globals.xyz3_rough_dispersion();
  auto result = surface_shader_bsdf_eval_mis(
      kernel_globals, shader_data, wo, 0u, false, light_shader_flags,
      zero_evaluation(spectral, lane_pdf), 0.0f, 0.0f, 0.0f, closure_types);
  result.pdf =
      select(0.0f, result.pdf, (light_shader_flags & shader_use_mis) != 0u);
  if (result.spectral_lane_pdf.has_value()) {
    *result.spectral_lane_pdf =
        select(make_float4(0.0f), *result.spectral_lane_pdf,
               (light_shader_flags & shader_use_mis) != 0u);
  }
  return result;
}

SurfaceShaderClosurePick
surface_shader_bsdf_bssrdf_pick(const ShaderData &shader_data,
                                Expr<luisa::float3> random) noexcept {
  const auto &pool = *shader_data.closure;
  if (pool.capacity() <= 1u) {
    return {.index = 0u, .random = random};
  }
  UInt sampled = 0u;
  Float3 reused_random = random;

  /* This intentionally branches on the total retained closure count rather
   * than the eligible count. It is the exact Cycles fast-path condition. */
  $if(pool.count() > 1u) {
    $if(pool.count() == 2u) {
      const auto t0 = pool.closure_type(0u);
      const auto w0 = select(0.0f, pool.closure_sample_weight(0u),
                             closure::is_bsdf_or_bssrdf(t0));
      const auto t1 = pool.closure_type(1u);
      const auto w1 = select(0.0f, pool.closure_sample_weight(1u),
                             closure::is_bsdf_or_bssrdf(t1));
      const auto sum_w = w0 + w1;
      const auto target = random.z * sum_w;
      $if(target < w0) {
        sampled = 0u;
        reused_random = make_float3(random.xy(), target / w0);
      }
      $elif(target < sum_w) {
        sampled = 1u;
        reused_random = make_float3(random.xy(), (target - w0) / w1);
      };
    }
    $else {
      Float sum_sample_weight = 0.0f;
      UInt index = 0u;
      $while(index < pool.count()) {
        const auto t = pool.closure_type(index);
        $if(closure::is_bsdf_or_bssrdf(t)) {
          sum_sample_weight += pool.closure_sample_weight(index);
        };
        index += 1u;
      };

      const auto target = random.z * sum_sample_weight;
      Float partial_sum = 0.0f;
      index = 0u;
      $while(index < pool.count()) {
        const auto t = pool.closure_type(index);
        $if(closure::is_bsdf_or_bssrdf(t)) {
          const auto sw = pool.closure_sample_weight(index);
          const auto next_sum = partial_sum + sw;
          $if(target < next_sum) {
            sampled = index;
            reused_random = make_float3(random.xy(), (target - partial_sum) / sw);
            $break;
          };
          partial_sum = next_sum;
        };
        index += 1u;
      };
    };
  };

  return {.index = sampled, .random = reused_random};
}

template <typename WeightT>
[[nodiscard]] WeightT scale_bssrdf_sample_weight(
    const ClosurePool &pool, WeightT weight,
    Expr<float> selected_sample_weight) noexcept {
  if (pool.capacity() <= 1u) {
    return weight;
  }
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
    weight *= sum_sample_weight / selected_sample_weight;
  };
  return weight;
}

Float3 surface_shader_bssrdf_sample_weight(
    const ShaderData &shader_data, Expr<std::uint32_t> closure_index) noexcept {
  const auto &pool = *shader_data.closure;
  const auto selected = pool.common(closure_index);
  return scale_bssrdf_sample_weight(pool, selected.weight,
                                    selected.sample_weight);
}

Float4 surface_shader_bssrdf_sample_weight_spectral(
    const ShaderData &shader_data, Expr<std::uint32_t> closure_index,
    const cycles_spectrum::SpectralContext &spectral) noexcept {
  const auto &pool = *shader_data.closure;
  const auto selected = pool.common(closure_index);
  const auto base =
      closure_spectral_weight(pool, closure_index, selected, spectral);
  return scale_bssrdf_sample_weight(pool, base, selected.sample_weight);
}

SurfaceShaderBsdfSample
surface_shader_bsdf_sample_closure(const KernelGlobals &kernel_globals,
                                   ShaderData &shader_data,
                                   const SurfaceShaderClosurePick &pick,
                                   ClosureTypeMask closure_types) noexcept {
  const auto &pool = *shader_data.closure;
  const auto common = pool.common(pick.index);
  const bool spectral =
      kernel_globals.is_spectral() && shader_data.spectral.has_value();
  if (spectral) {
    const auto sc = *shader_data.spectral;
    const auto sampled = bsdf_sample_spectral(
        kernel_globals, shader_data, pick.index, pick.random, sc,
        closure_types);
    const bool track_lane_pdf = kernel_globals.xyz3_rough_dispersion();
    auto result = SurfaceShaderBsdfSample{
        .evaluation = zero_evaluation(true, track_lane_pdf),
        .wo = sampled.wo,
        .sampled_roughness = sampled.sampled_roughness,
        .eta = sampled.eta,
        .label = sampled.label,
        .spectral_terminated_secondary = sampled.terminated_secondary};

    $if(sampled.pdf != 0.0f) {
      const auto weighted_value =
          sampled.value * closure_spectral_weight(pool, pick.index, common, sc);
      bsdf_eval_accum(result.evaluation, common, sampled.wo, weighted_value);
      const auto roughness_squared =
          bsdf_get_specular_roughness_squared(pool, pick.index);
      if (pool.capacity() <= 1u) {
        result.evaluation.pdf = sampled.pdf;
        result.evaluation.average_roughness_squared = roughness_squared;
        if (result.evaluation.spectral_lane_pdf.has_value()) {
          *result.evaluation.spectral_lane_pdf =
              sampled.lane_pdf.value_or(make_float4(sampled.pdf));
        }
      } else {
        $if(pool.count() > 1u) {
          const auto saved_terminated = shader_data.spectral->secondary_terminated;
          shader_data.spectral->secondary_terminated =
              saved_terminated | sampled.terminated_secondary;
          const auto weighted_pdf = sampled.pdf * common.sample_weight;
          const std::optional<Float4> initial_lane_pdf =
              track_lane_pdf
                  ? std::optional<Float4>{
                        sampled.lane_pdf.value_or(make_float4(sampled.pdf)) *
                        common.sample_weight}
                  : std::nullopt;
          result.evaluation = surface_shader_bsdf_eval_mis(
              kernel_globals, shader_data, sampled.wo, pick.index, true, 0u,
              result.evaluation, weighted_pdf, common.sample_weight,
              weighted_pdf * roughness_squared, closure_types,
              initial_lane_pdf);
          shader_data.spectral->secondary_terminated = saved_terminated;
          if (kernel_globals.spectral_mode() != contract::SpectralMode::xyz3) {
            $if(sampled.terminated_secondary) {
              *result.evaluation.spectral_sum =
                  make_float4(result.evaluation.spectral_sum->x, 0.0f, 0.0f, 0.0f);
              *result.evaluation.spectral_diffuse = make_float4(
                  result.evaluation.spectral_diffuse->x, 0.0f, 0.0f, 0.0f);
              *result.evaluation.spectral_glossy = make_float4(
                  result.evaluation.spectral_glossy->x, 0.0f, 0.0f, 0.0f);
            };
          }
        }
        $else {
          result.evaluation.pdf = sampled.pdf;
          result.evaluation.average_roughness_squared = roughness_squared;
          if (result.evaluation.spectral_lane_pdf.has_value()) {
            *result.evaluation.spectral_lane_pdf =
                sampled.lane_pdf.value_or(make_float4(sampled.pdf));
          }
        };
      }
    };
    return result;
  }

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
    if (pool.capacity() <= 1u) {
      result.evaluation.pdf = sampled.pdf;
      result.evaluation.average_roughness_squared = roughness_squared;
    } else {
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
    }
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
