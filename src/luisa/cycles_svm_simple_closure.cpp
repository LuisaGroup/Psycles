/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_simple_closure.h"

#include "cycles_svm_microfacet.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float half_pi = 1.57079632679489661923f;
inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float two_pi = 6.28318530717958647692f;

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float safe_sqrt(Expr<float> value) noexcept {
  return sqrt(max(value, 0.0f));
}

[[nodiscard]] Float oren_nayar_g(Expr<float> cos_theta) noexcept {
  Float result;
  $if(cos_theta < 1.0e-6f) { result = (half_pi - 2.0f / 3.0f) - cos_theta; }
  $else {
    const auto sin_theta = safe_sqrt(1.0f - square(cos_theta));
    const auto theta = acos(clamp(cos_theta, -1.0f, 1.0f));
    result = sin_theta * (theta - 2.0f / 3.0f - sin_theta * cos_theta) +
             2.0f / 3.0f * (sin_theta / cos_theta) *
                 (1.0f - square(sin_theta) * sin_theta);
  };
  return result;
}

}// namespace

OrenNayarParam oren_nayar_param(Expr<luisa::float3> color,
                                Expr<float> normal_view,
                                Expr<float> roughness) noexcept {
  const auto sigma = clamp(roughness, 0.0f, 1.0f);
  const auto a = 1.0f / (pi + sigma * (half_pi - 2.0f / 3.0f));
  const auto b = sigma * a;
  const auto albedo = clamp(color, 0.0f, 1.0f);
  const auto energy_average = a * pi + ((two_pi - 5.6f) / 3.0f) * b;
  const auto one_minus_energy_average = 1.0f - energy_average;
  const auto multiple_scatter =
      inverse_pi * (albedo * albedo) *
      (energy_average / one_minus_energy_average) /
      (make_float3(1.0f) - albedo * one_minus_energy_average);
  const auto view_energy = a * pi + b * oren_nayar_g(max(normal_view, 0.0f));
  return {.roughness = roughness,
          .a = a,
          .b = b,
          .multiscatter_term = multiple_scatter * (1.0f - view_energy)};
}

void diffuse_setup(ShaderData &shader_data, Expr<luisa::float3> normal,
                   Expr<luisa::float3> weight) noexcept {
  auto &pool = *shader_data.closure;
  const auto allocated = bsdf_allocate(shader_data, weight);
  $if(allocated.valid) {
    pool.set_normal(allocated.index, normal);
    pool.set_type(allocated.index,
                  static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID));
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
  };
}

void oren_nayar_setup(ShaderData &shader_data, Expr<luisa::float3> normal,
                      Expr<luisa::float3> weight, Expr<float> roughness,
                      Expr<luisa::float3> color) noexcept {
  auto &pool = *shader_data.closure;
  const auto allocated = bsdf_allocate(shader_data, weight);
  $if(allocated.valid) {
    pool.set_normal(allocated.index, normal);
    pool.set_type(allocated.index,
                  static_cast<std::uint32_t>(CLOSURE_BSDF_OREN_NAYAR_ID));
    pool.set_oren_nayar_param(
        allocated.index,
        oren_nayar_param(color, dot(normal, shader_data.wi), roughness));
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
  };
}

void translucent_setup(ShaderData &shader_data, Expr<luisa::float3> normal,
                       Expr<luisa::float3> weight) noexcept {
  auto &pool = *shader_data.closure;
  const auto allocated = bsdf_allocate(shader_data, weight);
  $if(allocated.valid) {
    pool.set_normal(allocated.index, normal);
    pool.set_type(allocated.index,
                  static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSLUCENT_ID));
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval |
                        shader_data_bsdf_has_transmission;
  };
}

void transparent_setup(ShaderData &shader_data, const PathState &path_state,
                       Expr<luisa::float3> weight) noexcept {
  auto &pool = *shader_data.closure;
  const Float sample_weight = abs(average(weight));
  $if(sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
    shader_data.closure_transparent_extinction += weight;
    $if((shader_data.flag & shader_data_transparent) != 0u) {
      UInt index = 0u;
      Bool found = false;
      $while((index < pool.count()) & !found) {
        const auto closure = pool.common(index);
        $if(closure.type ==
            static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID)) {
          pool.add_weight(index, weight);
          pool.add_sample_weight(index, sample_weight);
          found = true;
        };
        index += 1u;
      };
    }
    $else {
      shader_data.flag |= shader_data_bsdf | shader_data_transparent;
      const Bool terminating = (path_state.flag & path_ray_terminate) != 0u;
      $if(terminating) { pool.set_left(1u); };
      const auto allocated = pool.allocate(
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID), weight);
      $if(allocated.valid) {
        pool.set_sample_weight(allocated.index, sample_weight);
        pool.set_normal(allocated.index, shader_data.N);
      }
      $elif(terminating) { pool.set_left(0u); };
    };
  };
}

}// namespace psycles::luisa_backend::cycles_svm::detail
