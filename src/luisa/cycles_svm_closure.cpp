/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"
#include "cycles_svm_bssrdf.h"
#include "cycles_svm_microfacet.h"
#include "cycles_svm_principled.h"
#include "cycles_svm_sheen.h"
#include "cycles_svm_simple_closure.h"
#include "cycles_svm_toon.h"

#include <psycles/luisa/native_vector_math.h>

#include <algorithm>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

ClosurePool::ClosurePool(std::size_t capacity) noexcept
    : _capacity{std::min(capacity, maximum_closure_capacity)},
      _weight_and_sample{std::max(_capacity, std::size_t{1u})},
      _normal{std::max(_capacity, std::size_t{1u})},
      _type{std::max(_capacity, std::size_t{1u})},
      _payload0{std::max(_capacity, std::size_t{1u})},
      _payload1{std::max(_capacity, std::size_t{1u})},
      _payload2{std::max(_capacity, std::size_t{1u})},
      _payload3{std::max(_capacity, std::size_t{1u})},
      _payload4{std::max(_capacity, std::size_t{1u})},
      _payload5{std::max(_capacity, std::size_t{1u})},
      _payload6{std::max(_capacity, std::size_t{1u})},
      _payload_tag{std::max(_capacity, std::size_t{1u})}, _count{0u},
      _left{static_cast<std::uint32_t>(_capacity)} {}

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
  _payload0.write(index, make_float4(param.roughness, param.a, param.b, 0.0f));
  _payload1.write(index, make_float4(param.multiscatter_term, 0.0f));
}

void ClosurePool::set_sheen_param(Expr<std::uint32_t> index,
                                  const SheenParam &param) noexcept {
  _payload0.write(index, make_float4(param.roughness, param.transform_a,
                                     param.transform_b, 0.0f));
  _payload1.write(index, make_float4(param.T, 0.0f));
  _payload2.write(index, make_float4(param.B, 0.0f));
}

void ClosurePool::set_velvet_param(Expr<std::uint32_t> index,
                                   const VelvetParam &param) noexcept {
  _payload0.write(index,
                  make_float4(param.sigma, param.invsigma2, 0.0f, 0.0f));
}

void ClosurePool::set_toon_param(Expr<std::uint32_t> index,
                                 const ToonParam &param) noexcept {
  _payload0.write(index, make_float4(param.size, param.smooth, 0.0f, 0.0f));
}

void ClosurePool::set_bssrdf_param(Expr<std::uint32_t> index,
                                   const BssrdfParam &param) noexcept {
  _payload0.write(index, make_float4(param.radius, param.anisotropy));
  _payload1.write(index, make_float4(param.albedo, param.ior));
  _payload2.write(index, make_float4(param.alpha, 0.0f, 0.0f, 0.0f));
}

void ClosurePool::set_microfacet_param(Expr<std::uint32_t> index,
                                       const MicrofacetParam &param) noexcept {
  _payload0.write(index, make_float4(param.alpha_x, param.alpha_y, param.ior,
                                     param.energy_scale));
  _payload1.write(index, make_float4(param.T, 0.0f));
  _payload_tag.write(index, param.fresnel_type);
}

void ClosurePool::set_generalized_schlick(
    Expr<std::uint32_t> index,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
  _payload2.write(index,
                  make_float4(fresnel.thin_film.thickness,
                              fresnel.thin_film.ior, fresnel.exponent, 0.0f));
  _payload3.write(index, make_float4(fresnel.reflection_tint, 0.0f));
  _payload4.write(index, make_float4(fresnel.transmission_tint, 0.0f));
  _payload5.write(index, make_float4(fresnel.f0, 0.0f));
  _payload6.write(index, make_float4(fresnel.f90, 0.0f));
}

void ClosurePool::set_fresnel_conductor(
    Expr<std::uint32_t> index, const FresnelConductor &fresnel) noexcept {
  _payload2.write(index, make_float4(fresnel.thin_film.thickness,
                                     fresnel.thin_film.ior, 0.0f, 0.0f));
  _payload3.write(index, make_float4(fresnel.ior, 0.0f));
  _payload4.write(index, make_float4(fresnel.extinction, 0.0f));
}

void ClosurePool::set_fresnel_f82_tint(Expr<std::uint32_t> index,
                                       const FresnelF82Tint &fresnel) noexcept {
  _payload2.write(index, make_float4(fresnel.thin_film.thickness,
                                     fresnel.thin_film.ior, 0.0f, 0.0f));
  _payload3.write(index, make_float4(fresnel.f0, 0.0f));
  _payload4.write(index, make_float4(fresnel.b, 0.0f));
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
  const auto scalars = _payload0.read(index);
  return {.common = common(index),
          .param = {.roughness = scalars.x,
                    .a = scalars.y,
                    .b = scalars.z,
                    .multiscatter_term = _payload1.read(index).xyz()}};
}

SheenClosure ClosurePool::sheen(Expr<std::uint32_t> index) const noexcept {
  const auto scalars = _payload0.read(index);
  return {.common = common(index),
          .param = {.roughness = scalars.x,
                    .transform_a = scalars.y,
                    .transform_b = scalars.z,
                    .T = _payload1.read(index).xyz(),
                    .B = _payload2.read(index).xyz()}};
}

VelvetClosure ClosurePool::velvet(Expr<std::uint32_t> index) const noexcept {
  const auto scalars = _payload0.read(index);
  return {.common = common(index),
          .param = {.sigma = scalars.x, .invsigma2 = scalars.y}};
}

ToonClosure ClosurePool::toon(Expr<std::uint32_t> index) const noexcept {
  const auto scalars = _payload0.read(index);
  return {.common = common(index),
          .param = {.size = scalars.x, .smooth = scalars.y}};
}

BssrdfClosure ClosurePool::bssrdf(Expr<std::uint32_t> index) const noexcept {
  const auto radius_anisotropy = _payload0.read(index);
  const auto albedo_ior = _payload1.read(index);
  return {.common = common(index),
          .param = {.radius = radius_anisotropy.xyz(),
                    .albedo = albedo_ior.xyz(),
                    .anisotropy = radius_anisotropy.w,
                    .ior = albedo_ior.w,
                    .alpha = _payload2.read(index).x}};
}

MicrofacetClosure
ClosurePool::microfacet(Expr<std::uint32_t> index) const noexcept {
  const auto film = _payload2.read(index);
  return {
      .common = common(index),
      .param = microfacet_param(index),
      .generalized_schlick = {.thin_film = {.thickness = film.x, .ior = film.y},
                              .reflection_tint = _payload3.read(index).xyz(),
                              .transmission_tint = _payload4.read(index).xyz(),
                              .f0 = _payload5.read(index).xyz(),
                              .f90 = _payload6.read(index).xyz(),
                              .exponent = film.z}};
}

MicrofacetConductorClosure
ClosurePool::microfacet_conductor(Expr<std::uint32_t> index) const noexcept {
  const auto film = _payload2.read(index);
  return {.common = common(index),
          .param = microfacet_param(index),
          .conductor = {.thin_film = {.thickness = film.x, .ior = film.y},
                        .ior = _payload3.read(index).xyz(),
                        .extinction = _payload4.read(index).xyz()}};
}

MicrofacetF82TintClosure
ClosurePool::microfacet_f82_tint(Expr<std::uint32_t> index) const noexcept {
  const auto film = _payload2.read(index);
  return {.common = common(index),
          .param = microfacet_param(index),
          .f82_tint = {.thin_film = {.thickness = film.x, .ior = film.y},
                       .f0 = _payload3.read(index).xyz(),
                       .b = _payload4.read(index).xyz()}};
}

MicrofacetParam
ClosurePool::microfacet_param(Expr<std::uint32_t> index) const noexcept {
  const auto microfacet = _payload0.read(index);
  return {.alpha_x = microfacet.x,
          .alpha_y = microfacet.y,
          .ior = microfacet.z,
          .energy_scale = microfacet.w,
          .fresnel_type = _payload_tag.read(index),
          .T = _payload1.read(index).xyz()};
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
        $if(closure_type ==
            static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID)) {
          node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, true,
                               shader_data, path_state, supported);
        }
        $else {
        const Bool is_bssrdf =
            (closure_type ==
             static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID)) |
            (closure_type ==
             static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID));
        const Bool is_sheen =
            (closure_type ==
             static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_ASHIKHMIN_VELVET_ID));
        const Bool is_toon =
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_DIFFUSE_TOON_ID)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_GLOSSY_TOON_ID));
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
        const Bool is_metallic =
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_PHYSICAL_CONDUCTOR)) |
            (closure_type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_F82_CONDUCTOR));
        $if(is_bssrdf) {
          node_bssrdf(cursor, stack, closure_type, closure_weight, mix_weight,
                      shader_data, path_state);
        }
        $elif(is_sheen) {
          node_sheen(kernel_globals, cursor, stack, closure_type,
                     closure_weight, mix_weight, shader_data);
        }
        $elif(is_toon) {
          node_toon(kernel_globals, cursor, stack, closure_type,
                    closure_weight, mix_weight, shader_data, path_state);
        }
        $elif(is_glass) {
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
        $elif(is_metallic) {
          const auto distribution = cursor.word();
          const auto base_ior_x = cursor.word();
          const auto base_ior_y = cursor.word();
          const auto base_ior_z = cursor.word();
          const auto edge_tint_k_x = cursor.word();
          const auto edge_tint_k_y = cursor.word();
          const auto edge_tint_k_z = cursor.word();
          const auto roughness_input = cursor.word();
          const auto anisotropy_input = cursor.word();
          const auto rotation_input = cursor.word();
          const auto thin_film_thickness_input = cursor.word();
          const auto thin_film_ior_input = cursor.word();
          const auto normal_tangent_packed = cursor.word();
          const auto normal_offset = cursor.byte(normal_tangent_packed, 0u);
          const auto tangent_offset = cursor.byte(normal_tangent_packed, 1u);
          auto normal =
              stack_load_float3_default(stack, normal_offset, shader_data.N);
          normal = native_vector_math::safe_normalize_nonzero_or(
              normal, shader_data.N);
          detail::metallic_setup(
              kernel_globals, shader_data, path_state, closure_type,
              distribution, mix_weight, normal,
              stack_load_input_float3(stack, base_ior_x, base_ior_y,
                                      base_ior_z),
              stack_load_input_float3(stack, edge_tint_k_x, edge_tint_k_y,
                                      edge_tint_k_z),
              stack_load_input_float(stack, roughness_input),
              stack_load_input_float(stack, anisotropy_input),
              stack_load_input_float(stack, rotation_input),
              stack_load_input_float(stack, thin_film_thickness_input),
              stack_load_input_float(stack, thin_film_ior_input),
              stack_load_float3_default(stack, tangent_offset,
                                        make_float3(0.0f)),
              tangent_offset !=
                  static_cast<std::uint32_t>(SVM_STACK_INVALID));
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
      if (shader_data.closure == nullptr) {
        node_closure_bsdf_skip(cursor, closure_type);
        supported = false;
      } else {
        node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, false,
                             shader_data, path_state, supported);
      }
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
