/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_simple_closure.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <limits>

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

[[nodiscard]] Float3 oren_nayar_intensity(const OrenNayarClosure &closure,
                                          Expr<luisa::float3> normal,
                                          Expr<luisa::float3> view,
                                          Expr<luisa::float3> light) noexcept {
  const auto nl = max(dot(normal, light), 0.0f);
  Float3 result = make_float3(0.0f);
  $if(closure.param.b <= 0.0f) { result = make_float3(nl * inverse_pi); }
  $else {
    const auto nv = max(dot(normal, view), 0.0f);
    Float t = dot(light, view) - nl * nv;
    $if(t > 0.0f) {
      t /= luisa::compute::max(nl, nv) + std::numeric_limits<float>::min();
    };
    const auto single_scatter = closure.param.a + closure.param.b * t;
    const auto light_energy =
        closure.param.a * pi + closure.param.b * oren_nayar_g(nl);
    const auto multi_scatter =
        closure.param.multiscatter_term * (1.0f - light_energy);
    result = nl * (make_float3(single_scatter) + multi_scatter);
  };
  return result;
}

} // namespace

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

BsdfEvaluation bsdf_diffuse_eval(const ShaderClosureCommon &closure,
                                 Expr<luisa::float3>,
                                 Expr<luisa::float3> wo) noexcept {
  const auto cosine = max(dot(closure.N, wo), 0.0f) * inverse_pi;
  return {.value = make_float3(cosine), .pdf = cosine};
}

BsdfSample bsdf_diffuse_sample(const ShaderClosureCommon &closure,
                               Expr<luisa::float3> Ng, Expr<luisa::float3>,
                               Expr<luisa::float2> random) noexcept {
  const auto hemisphere =
      cycles_sample_mapping::sample_cosine_hemisphere(closure.N, random);
  Float pdf = hemisphere.pdf;
  Float3 value = make_float3(0.0f);
  $if(dot(Ng, hemisphere.direction) > 0.0f) { value = make_float3(pdf); }
  $else { pdf = 0.0f; };
  return {.value = value,
          .wo = hemisphere.direction,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          .label =
              cycles_closure::label_reflect | cycles_closure::label_diffuse};
}

BsdfEvaluation bsdf_translucent_eval(const ShaderClosureCommon &closure,
                                     Expr<luisa::float3>,
                                     Expr<luisa::float3> wo) noexcept {
  const auto cosine = max(-dot(closure.N, wo), 0.0f) * inverse_pi;
  return {.value = make_float3(cosine), .pdf = cosine};
}

BsdfSample bsdf_translucent_sample(const ShaderClosureCommon &closure,
                                   Expr<luisa::float3> Ng, Expr<luisa::float3>,
                                   Expr<luisa::float2> random) noexcept {
  const auto hemisphere =
      cycles_sample_mapping::sample_cosine_hemisphere(-closure.N, random);
  Float pdf = hemisphere.pdf;
  Float3 value = make_float3(0.0f);
  $if(dot(Ng, hemisphere.direction) < 0.0f) { value = make_float3(pdf); }
  $else { pdf = 0.0f; };
  return {.value = value,
          .wo = hemisphere.direction,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          .label =
              cycles_closure::label_transmit | cycles_closure::label_diffuse};
}

BsdfEvaluation bsdf_oren_nayar_eval(const OrenNayarClosure &closure,
                                    Expr<luisa::float3> wi,
                                    Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto cosine = dot(closure.common.N, wo);
  $if(cosine > 0.0f) {
    result.pdf = cosine * inverse_pi;
    result.value = oren_nayar_intensity(closure, closure.common.N, wi, wo);
  };
  return result;
}

BsdfSample bsdf_oren_nayar_sample(const OrenNayarClosure &closure,
                                  Expr<luisa::float3> Ng,
                                  Expr<luisa::float3> wi,
                                  Expr<luisa::float2> random) noexcept {
  const auto hemisphere =
      cycles_sample_mapping::sample_cosine_hemisphere(closure.common.N, random);
  Float pdf = hemisphere.pdf;
  Float3 value = make_float3(0.0f);
  $if(dot(Ng, hemisphere.direction) > 0.0f) {
    value = oren_nayar_intensity(closure, closure.common.N, wi,
                                 hemisphere.direction);
  }
  $else { pdf = 0.0f; };
  return {.value = value,
          .wo = hemisphere.direction,
          .pdf = pdf,
          .sampled_roughness = make_float2(1.0f),
          .eta = 1.0f,
          .label =
              cycles_closure::label_reflect | cycles_closure::label_diffuse};
}

BsdfEvaluation bsdf_rough_translucent_eval(const OrenNayarClosure &closure,
                                           Expr<luisa::float3> wi,
                                           Expr<luisa::float3> wo) noexcept {
  return bsdf_oren_nayar_eval(closure, reflect(wi, closure.common.N), wo);
}

BsdfSample bsdf_rough_translucent_sample(const OrenNayarClosure &closure,
                                         Expr<luisa::float3> Ng,
                                         Expr<luisa::float3> wi,
                                         Expr<luisa::float2> random) noexcept {
  auto result = bsdf_oren_nayar_sample(closure, -Ng,
                                       reflect(wi, closure.common.N), random);
  result.label = cycles_closure::label_transmit | cycles_closure::label_diffuse;
  return result;
}

BsdfEvaluation bsdf_transparent_eval(const ShaderClosureCommon &,
                                     Expr<luisa::float3>,
                                     Expr<luisa::float3>) noexcept {
  return {.value = make_float3(0.0f), .pdf = 0.0f};
}

BsdfSample bsdf_transparent_sample(const ShaderClosureCommon &,
                                   Expr<luisa::float3>,
                                   Expr<luisa::float3> wi) noexcept {
  return {.value = make_float3(1.0e6f),
          .wo = -wi,
          .pdf = 1.0e6f,
          .sampled_roughness = make_float2(0.0f),
          .eta = 1.0f,
          .label = cycles_closure::label_transmit |
                   cycles_closure::label_transparent};
}

} // namespace psycles::luisa_backend::cycles_svm::detail
