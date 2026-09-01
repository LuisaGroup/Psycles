/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"
#include "cycles_svm_microfacet.h"

#include <psycles/luisa/native_vector_math.h>

#include <algorithm>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm {

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

[[nodiscard]] OrenNayarParam oren_nayar_param(Expr<luisa::float3> color,
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
  const auto allocated = detail::bsdf_allocate(shader_data, weight);
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
  const auto allocated = detail::bsdf_allocate(shader_data, weight);
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
  const auto allocated = detail::bsdf_allocate(shader_data, weight);
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

} // namespace

ClosurePool::ClosurePool(std::size_t capacity) noexcept
    : _capacity{std::min(capacity, maximum_closure_capacity)},
      _weight_and_sample{std::max(_capacity, std::size_t{1u})},
      _normal{std::max(_capacity, std::size_t{1u})},
      _type{std::max(_capacity, std::size_t{1u})},
      _oren_nayar_scalars{std::max(_capacity, std::size_t{1u})},
      _oren_nayar_multiscatter{std::max(_capacity, std::size_t{1u})},
      _microfacet_alpha_ior_energy{std::max(_capacity, std::size_t{1u})},
      _microfacet_tangent{std::max(_capacity, std::size_t{1u})},
      _microfacet_fresnel_type{std::max(_capacity, std::size_t{1u})},
      _generalized_schlick_thin_film{std::max(_capacity, std::size_t{1u})},
      _generalized_schlick_reflection_tint{
          std::max(_capacity, std::size_t{1u})},
      _generalized_schlick_transmission_tint{
          std::max(_capacity, std::size_t{1u})},
      _generalized_schlick_f0{std::max(_capacity, std::size_t{1u})},
      _generalized_schlick_f90{std::max(_capacity, std::size_t{1u})},
      _count{0u}, _left{static_cast<std::uint32_t>(_capacity)} {}

std::size_t ClosurePool::capacity() const noexcept { return _capacity; }

UInt ClosurePool::count() const noexcept { return _count; }

UInt ClosurePool::left() const noexcept { return _left; }

ClosurePool::Allocation
ClosurePool::allocate(Expr<std::uint32_t> closure_type,
                      Expr<luisa::float3> weight) noexcept {
  Allocation allocation{.index = _count, .valid = false};
  $if(_left != 0u) {
    _type.write(allocation.index, closure_type);
    _weight_and_sample.write(allocation.index, make_float4(weight, 0.0f));
    _count += 1u;
    _left -= 1u;
    allocation.valid = true;
  };
  return allocation;
}

Bool ClosurePool::allocate_extra(const Allocation &owner,
                                 Expr<std::uint32_t> slot_count) noexcept {
  Bool allocated = false;
  $if(owner.valid) {
    $if(slot_count <= _left) {
      _left -= slot_count;
      allocated = true;
    }
    $else {
      /* closure_alloc_extra is called immediately after bsdf_alloc. The
       * owner identity makes that source precondition explicit and prevents
       * an unrelated closure from being removed by a malformed caller. */
      $if((_count != 0u) & (owner.index + 1u == _count)) {
        _count -= 1u;
        _left += 1u;
      };
    };
  };
  return allocated;
}

void ClosurePool::set_type(Expr<std::uint32_t> index,
                           Expr<std::uint32_t> closure_type) noexcept {
  _type.write(index, closure_type);
}

void ClosurePool::set_weight(Expr<std::uint32_t> index,
                             Expr<luisa::float3> weight) noexcept {
  const auto stored = _weight_and_sample.read(index);
  _weight_and_sample.write(index, make_float4(weight, stored.w));
}

void ClosurePool::add_weight(Expr<std::uint32_t> index,
                             Expr<luisa::float3> weight) noexcept {
  const auto stored = _weight_and_sample.read(index);
  _weight_and_sample.write(index, make_float4(stored.xyz() + weight, stored.w));
}

void ClosurePool::set_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  const auto stored = _weight_and_sample.read(index);
  _weight_and_sample.write(index, make_float4(stored.xyz(), sample_weight));
}

void ClosurePool::add_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  const auto stored = _weight_and_sample.read(index);
  _weight_and_sample.write(index,
                           make_float4(stored.xyz(), stored.w + sample_weight));
}

void ClosurePool::set_normal(Expr<std::uint32_t> index,
                             Expr<luisa::float3> normal) noexcept {
  _normal.write(index, make_float4(normal, 0.0f));
}

void ClosurePool::set_oren_nayar_param(Expr<std::uint32_t> index,
                                       const OrenNayarParam &param) noexcept {
  _oren_nayar_scalars.write(
      index, make_float4(param.roughness, param.a, param.b, 0.0f));
  _oren_nayar_multiscatter.write(index,
                                 make_float4(param.multiscatter_term, 0.0f));
}

void ClosurePool::set_microfacet_param(
    Expr<std::uint32_t> index, const MicrofacetParam &param) noexcept {
  _microfacet_alpha_ior_energy.write(
      index, make_float4(param.alpha_x, param.alpha_y, param.ior,
                         param.energy_scale));
  _microfacet_tangent.write(index, make_float4(param.T, 0.0f));
  _microfacet_fresnel_type.write(index, param.fresnel_type);
}

void ClosurePool::set_generalized_schlick(
    Expr<std::uint32_t> index,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
  _generalized_schlick_thin_film.write(
      index, make_float4(fresnel.thin_film.thickness,
                         fresnel.thin_film.ior, fresnel.exponent, 0.0f));
  _generalized_schlick_reflection_tint.write(
      index, make_float4(fresnel.reflection_tint, 0.0f));
  _generalized_schlick_transmission_tint.write(
      index, make_float4(fresnel.transmission_tint, 0.0f));
  _generalized_schlick_f0.write(index, make_float4(fresnel.f0, 0.0f));
  _generalized_schlick_f90.write(index, make_float4(fresnel.f90, 0.0f));
}

void ClosurePool::set_left(Expr<std::uint32_t> left) noexcept { _left = left; }

ShaderClosureCommon
ClosurePool::common(Expr<std::uint32_t> index) const noexcept {
  const auto weight_and_sample = _weight_and_sample.read(index);
  return {.weight = weight_and_sample.xyz(),
          .type = _type.read(index),
          .sample_weight = weight_and_sample.w,
          .N = _normal.read(index).xyz()};
}

OrenNayarClosure
ClosurePool::oren_nayar(Expr<std::uint32_t> index) const noexcept {
  const auto scalars = _oren_nayar_scalars.read(index);
  return {.common = common(index),
          .param = {.roughness = scalars.x,
                    .a = scalars.y,
                    .b = scalars.z,
                    .multiscatter_term =
                        _oren_nayar_multiscatter.read(index).xyz()}};
}

MicrofacetClosure
ClosurePool::microfacet(Expr<std::uint32_t> index) const noexcept {
  const auto film = _generalized_schlick_thin_film.read(index);
  return {
      .common = common(index),
      .param = microfacet_param(index),
      .generalized_schlick = {
          .thin_film = {.thickness = film.x, .ior = film.y},
          .reflection_tint =
              _generalized_schlick_reflection_tint.read(index).xyz(),
          .transmission_tint =
              _generalized_schlick_transmission_tint.read(index).xyz(),
          .f0 = _generalized_schlick_f0.read(index).xyz(),
          .f90 = _generalized_schlick_f90.read(index).xyz(),
          .exponent = film.z}};
}

MicrofacetParam
ClosurePool::microfacet_param(Expr<std::uint32_t> index) const noexcept {
  const auto microfacet = _microfacet_alpha_ior_energy.read(index);
  return {.alpha_x = microfacet.x,
          .alpha_y = microfacet.y,
          .ior = microfacet.z,
          .energy_scale = microfacet.w,
          .fresnel_type = _microfacet_fresnel_type.read(index),
          .T = _microfacet_tangent.read(index).xyz()};
}

namespace detail {

void node_closure_set_weight(Cursor &cursor, Float3 &closure_weight) noexcept {
  const auto x = cursor.floating();
  const auto y = cursor.floating();
  const auto z = cursor.floating();
  closure_weight = make_float3(x, y, z);
}

void node_closure_weight(Cursor &cursor, Stack &stack,
                         Float3 &closure_weight) noexcept {
  const auto packed = cursor.word();
  closure_weight = stack_load_float3(stack, cursor.byte(packed, 0u));
}

void node_emission_weight(Cursor &cursor, Stack &stack,
                          Float3 &closure_weight) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto strength = cursor.word();
  closure_weight = stack_load_input_float3(stack, color_x, color_y, color_z) *
                   stack_load_input_float(stack, strength);
}

void node_mix_closure(Cursor &cursor, Stack &stack) noexcept {
  const auto factor = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto input_weight_offset = cursor.byte(packed_offsets, 0u);
  const auto weight1_offset = cursor.byte(packed_offsets, 1u);
  const auto weight2_offset = cursor.byte(packed_offsets, 2u);

  const Float weight = clamp(stack_load_input_float(stack, factor), 0.0f, 1.0f);
  const auto input_weight =
      stack_load_float_default(stack, input_weight_offset, 1.0f);
  $if(weight1_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight1_offset, input_weight * (1.0f - weight));
  };
  $if(weight2_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight2_offset, input_weight * weight);
  };
}

void node_closure_emission(Cursor &cursor, Stack &stack,
                           Expr<luisa::float3> closure_weight,
                           ShaderData &shader_data, Bool &supported) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Bool active = true;

  $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto mix_weight = stack_load_float(stack, mix_weight_offset);
    $if(mix_weight == 0.0f) { active = false; }
    $else { weight *= mix_weight; };
  };

  $if(active) {
    $if((shader_data.flag & shader_data_is_volume_shader_eval) != 0u) {
      /* Cycles multiplies by object_volume_density here. The object service
       * is deliberately not guessed before that exact family is copied. */
      supported = false;
    }
    $else {
      $if((shader_data.flag & shader_data_emission) != 0u) {
        shader_data.closure_emission_background += weight;
      }
      $else {
        shader_data.flag |= shader_data_emission;
        shader_data.closure_emission_background = weight;
      };
    };
  };
}

void node_closure_background(Cursor &cursor, Stack &stack,
                             Expr<luisa::float3> closure_weight,
                             ShaderData &shader_data) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Bool active = true;

  $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto mix_weight = stack_load_float(stack, mix_weight_offset);
    $if(mix_weight == 0.0f) { active = false; }
    $else { weight *= mix_weight; };
  };

  $if(active) {
    $if((shader_data.flag & shader_data_emission) != 0u) {
      shader_data.closure_emission_background += weight;
    }
    $else {
      shader_data.flag |= shader_data_emission;
      shader_data.closure_emission_background = weight;
    };
  };
}

void node_closure_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                       Stack &stack,
                       Expr<luisa::float3> closure_weight,
                       ShaderType shader_type, std::uint32_t node_feature_mask,
                       ShaderData &shader_data, const PathState &path_state,
                       Bool &supported) noexcept {
  const auto closure_type = cursor.word();
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  const auto mix_weight =
      stack_load_float_default(stack, mix_weight_offset, 1.0f);

  if (shader_type != SHADER_TYPE_SURFACE) {
    node_closure_bsdf_skip(cursor, closure_type);
    return;
  }

  if ((node_feature_mask & kernel_feature_node_bsdf) != 0u) {
    $if(mix_weight == 0.0f) { node_closure_bsdf_skip(cursor, closure_type); }
    $else {
      if (shader_data.closure == nullptr) {
        node_closure_bsdf_skip(cursor, closure_type);
        supported = false;
      } else {
        const Bool is_glass =
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID));
        const Bool is_glossy =
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_GGX_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_BECKMANN_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID));
        const Bool is_refraction =
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID));
        $if(is_glass) {
          const auto color_x = cursor.word();
          const auto color_y = cursor.word();
          const auto color_z = cursor.word();
          const auto roughness_input = cursor.word();
          const auto ior_input = cursor.word();
          const auto thin_film_thickness_input = cursor.word();
          const auto thin_film_ior_input = cursor.word();
          const auto normal_packed = cursor.word();
          const auto normal_offset = cursor.byte(normal_packed, 0u);
          auto normal =
              stack_load_float3_default(stack, normal_offset, shader_data.N);
          normal = native_vector_math::safe_normalize_nonzero_or(
              normal, shader_data.N);
          detail::glass_setup(
              kernel_globals, shader_data, path_state, closure_type, mix_weight,
              normal,
              stack_load_input_float3(stack, color_x, color_y, color_z),
              stack_load_input_float(stack, roughness_input),
              stack_load_input_float(stack, ior_input),
              stack_load_input_float(stack, thin_film_thickness_input),
              stack_load_input_float(stack, thin_film_ior_input));
        }
        $elif(is_glossy) {
          const auto color_x = cursor.word();
          const auto color_y = cursor.word();
          const auto color_z = cursor.word();
          const auto roughness_input = cursor.word();
          const auto anisotropy_input = cursor.word();
          const auto rotation_input = cursor.word();
          const auto normal_tangent_packed = cursor.word();
          const auto normal_offset = cursor.byte(normal_tangent_packed, 0u);
          const auto tangent_offset = cursor.byte(normal_tangent_packed, 1u);
          auto normal =
              stack_load_float3_default(stack, normal_offset, shader_data.N);
          normal = native_vector_math::safe_normalize_nonzero_or(
              normal, shader_data.N);
          detail::glossy_setup(
              kernel_globals, shader_data, path_state, closure_type, mix_weight,
              closure_weight, normal,
              stack_load_input_float3(stack, color_x, color_y, color_z),
              stack_load_input_float(stack, roughness_input),
              stack_load_input_float(stack, anisotropy_input),
              stack_load_input_float(stack, rotation_input),
              stack_load_float3_default(stack, tangent_offset,
                                        make_float3(0.0f)),
              tangent_offset !=
                  static_cast<std::uint32_t>(SVM_STACK_INVALID));
        }
        $elif(is_refraction) {
          const auto roughness_input = cursor.word();
          const auto ior_input = cursor.word();
          const auto normal_packed = cursor.word();
          const auto normal_offset = cursor.byte(normal_packed, 0u);
          auto normal =
              stack_load_float3_default(stack, normal_offset, shader_data.N);
          normal = native_vector_math::safe_normalize_nonzero_or(
              normal, shader_data.N);
          detail::refraction_setup(
              kernel_globals, shader_data, path_state, closure_type, mix_weight,
              closure_weight, normal,
              stack_load_input_float(stack, roughness_input),
              stack_load_input_float(stack, ior_input));
        }
        $else {
          $switch(closure_type) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) {
            const auto color_x = cursor.word();
            const auto color_y = cursor.word();
            const auto color_z = cursor.word();
            const auto roughness_input = cursor.word();
            const auto normal_packed = cursor.word();
            const auto normal_offset = cursor.byte(normal_packed, 0u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);

            const auto weight = closure_weight * mix_weight;
            const auto roughness =
                stack_load_input_float(stack, roughness_input);
            $if(roughness < 1.0e-5f) {
              diffuse_setup(shader_data, normal, weight);
            }
            $else {
              const auto color = clamp(
                  stack_load_input_float3(stack, color_x, color_y, color_z),
                  0.0f, 1.0f);
              oren_nayar_setup(shader_data, normal, weight, roughness, color);
            };
          };
          PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSLUCENT_ID) {
            const auto unused_param = cursor.word();
            const auto normal_packed = cursor.word();
            static_cast<void>(unused_param);
            const auto normal_offset = cursor.byte(normal_packed, 0u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);
            const auto weight = closure_weight * mix_weight;
            translucent_setup(
                shader_data,
                detail::maybe_ensure_valid_specular_reflection(shader_data,
                                                               normal),
                weight);
          };
          PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSPARENT_ID) {
            const auto unused_param = cursor.word();
            const auto unused_normal = cursor.word();
            static_cast<void>(unused_param);
            static_cast<void>(unused_normal);
            transparent_setup(shader_data, path_state,
                              closure_weight * mix_weight);
          };
          $default {
            node_closure_bsdf_skip(cursor, closure_type);
            supported = false;
          };
          };
        };
      }
    };
    return;
  }

  if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
    $if((mix_weight == 0.0f) |
        (closure_type !=
         static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID))) {
      node_closure_bsdf_skip(cursor, closure_type);
    }
    $else {
      /* Cycles evaluates Principled emission from this node even without
       * NODE_FEATURE_BSDF. Keep the exact cursor transition, but reject the
       * state transition until that Principled payload has been copied. */
      node_closure_bsdf_skip(cursor, closure_type);
      supported = false;
    };
    return;
  }

  node_closure_bsdf_skip(cursor, closure_type);
}

void node_closure_bsdf_skip(Cursor &cursor,
                            Expr<std::uint32_t> closure_type) noexcept {
  UInt words = static_cast<std::uint32_t>(sizeof(SVMNodeSimpleBsdfData) /
                                          sizeof(std::uint32_t));
  $switch(closure_type) {
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PRINCIPLED_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_CHIANG_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_HUANG_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PHYSICAL_CONDUCTOR) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeMetallicBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_F82_CONDUCTOR) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeMetallicBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_OREN_NAYAR_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_RAY_PORTAL_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRayPortalBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRefractionBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRefractionBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_GLOSSY_TOON_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeToonBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_TOON_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeToonBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_REFLECTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_TRANSMISSION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    $default{};
  };
  cursor.advance(words);
}

} // namespace detail

} // namespace psycles::luisa_backend::cycles_svm

#undef PSYCLES_SVM_CASE
