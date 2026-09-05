/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_bssrdf.h"

#include "cycles_svm_microfacet.h"
#include "cycles_svm_simple_closure.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float minimum_radius = 1.0e-8f;

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float dipole_rd(Expr<float> alpha_prime,
                              Expr<float> fourthird_a) noexcept {
  const auto s = sqrt(3.0f * (1.0f - alpha_prime));
  return 0.5f * alpha_prime * (1.0f + exp(-fourthird_a * s)) * exp(-s);
}

[[nodiscard]] Float dipole_alpha_prime(Expr<float> rd,
                                       Expr<float> fourthird_a) noexcept {
  Float result = 0.0f;
  $if(rd < 1.0e-4f) { result = 0.0f; }
  $elif(rd >= 0.995f) { result = 0.999999f; }
  $else {
    Float x0 = 0.0f;
    Float x1 = 1.0f;
    Float xmid = 0.0f;
    for (auto iteration = 0u; iteration < 12u; iteration++) {
      xmid = 0.5f * (x0 + x1);
      const auto fmid = dipole_rd(xmid, fourthird_a);
      $if(fmid < rd) { x0 = xmid; }
      $else { x1 = xmid; };
    }
    result = xmid;
  };
  return result;
}

[[nodiscard]] Float3 setup_radius(Expr<luisa::float3> input_radius,
                                  Expr<luisa::float3> albedo, Expr<float> ior,
                                  Expr<std::uint32_t> type) noexcept {
  Float3 radius = input_radius;
  const Bool compatible_scale =
      (type == static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID)) |
      (type ==
       static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID));
  $if(compatible_scale) { radius *= 0.25f * inverse_pi; }
  $elif(type ==
        static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID)) {
    const auto inverse_eta = 1.0f / ior;
    const auto diffuse_fresnel =
        inverse_eta * (-1.440f * inverse_eta + 0.710f) + 0.668f + 0.0636f * ior;
    const auto fourthird_a =
        (4.0f / 3.0f) * (1.0f + diffuse_fresnel) / (1.0f - diffuse_fresnel);
    const auto alpha_prime =
        make_float3(dipole_alpha_prime(albedo.x, fourthird_a),
                    dipole_alpha_prime(albedo.y, fourthird_a),
                    dipole_alpha_prime(albedo.z, fourthird_a));
    radius *= sqrt(3.0f * (make_float3(1.0f) - alpha_prime));
  };
  return radius;
}

} // namespace

void node_bssrdf(Cursor &cursor, Stack &stack, Expr<std::uint32_t> type,
                 Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
                 ShaderData &shader_data, const PathState &path_state) noexcept {
  const auto radius_x = cursor.word();
  const auto radius_y = cursor.word();
  const auto radius_z = cursor.word();
  const auto scale_input = cursor.word();
  const auto ior_input = cursor.word();
  const auto anisotropy_input = cursor.word();
  const auto roughness_input = cursor.word();
  const auto normal_packed = cursor.word();
  const auto normal_offset = cursor.byte(normal_packed, 0u);
  auto normal =
      stack_load_float3_default(stack, normal_offset, shader_data.N);
  normal = native_vector_math::safe_normalize_nonzero_or(normal, shader_data.N);

  const auto radius = max(
      stack_load_input_float3(stack, radius_x, radius_y, radius_z) *
          stack_load_input_float(stack, scale_input),
      make_float3(0.0f));
  bssrdf_setup(
      shader_data, path_state, type, closure_weight * mix_weight, radius,
      closure_weight,
      maybe_ensure_valid_specular_reflection(shader_data, normal),
      clamp(stack_load_input_float(stack, roughness_input), 0.0f, 1.0f),
      stack_load_input_float(stack, ior_input),
      stack_load_input_float(stack, anisotropy_input));
}

void bssrdf_setup(ShaderData &shader_data, const PathState &path_state,
                  Expr<std::uint32_t> type, Expr<luisa::float3> input_weight,
                  Expr<luisa::float3> input_radius, Expr<luisa::float3> albedo,
                  Expr<luisa::float3> normal, Expr<float> input_alpha,
                  Expr<float> input_ior,
                  Expr<float> input_anisotropy) noexcept {
  if (shader_data.closure == nullptr) {
    return;
  }
  auto &pool = *shader_data.closure;
  const auto initial_sample_weight = abs(average(input_weight));
  /* Cycles deliberately spells this as an early `< cutoff` rejection. Thus a
   * non-finite average reaches closure_alloc; do not replace it with `>=`. */
  $if(!(initial_sample_weight < CLOSURE_WEIGHT_CUTOFF)) {
    const auto allocated = pool.allocate(
        static_cast<std::uint32_t>(CLOSURE_NONE_ID), input_weight);
    $if(allocated.valid) {
      pool.set_sample_weight(allocated.index, initial_sample_weight);
      Float3 weight = input_weight;
      Float3 radius = max(input_radius, make_float3(0.0f));
      Float anisotropy;
      $if(type == static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID)) {
        anisotropy = clamp(input_anisotropy, -0.99f, 0.99f);
      }
      $else { anisotropy = clamp(input_anisotropy, -0.99f, 0.9f); };
      const Float ior = clamp(input_ior, 1.01f, 3.8f);
      Float alpha = input_alpha;
      $if(type ==
          static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID)) {
        alpha = 1.0f;
      };

      Int channel_count = 3;
      Float3 diffuse_weight = make_float3(0.0f);
      const Bool burley_diffuse_ancestor =
          (type == static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID)) &
          ((path_state.flag & path_ray_diffuse_ancestor) != 0u);
      $if(burley_diffuse_ancestor) {
        channel_count = 0;
        diffuse_weight = weight;
      }
      $else {
        const Bool small_x = radius.x < minimum_radius;
        const Bool small_y = radius.y < minimum_radius;
        const Bool small_z = radius.z < minimum_radius;
        diffuse_weight = make_float3(select(0.0f, weight.x, small_x),
                                     select(0.0f, weight.y, small_y),
                                     select(0.0f, weight.z, small_z));
        weight = make_float3(select(weight.x, 0.0f, small_x),
                             select(weight.y, 0.0f, small_y),
                             select(weight.z, 0.0f, small_z));
        radius = make_float3(select(radius.x, 0.0f, small_x),
                             select(radius.y, 0.0f, small_y),
                             select(radius.z, 0.0f, small_z));
        channel_count -=
            cast<int>(small_x) + cast<int>(small_y) + cast<int>(small_z);
      };

      pool.set_normal(allocated.index, normal);
      pool.set_weight(allocated.index, weight);
      $if(channel_count < 3) {
        diffuse_setup(shader_data, normal, diffuse_weight);
      };

      $if(channel_count > 0) {
        pool.set_type(allocated.index, type);
        pool.set_sample_weight(allocated.index, abs(average(weight)) *
                                                    cast<float>(channel_count));
        radius = setup_radius(radius, albedo, ior, type);
        shader_data.flag |= shader_data_bssrdf;
      }
      $else {
        pool.set_type(allocated.index,
                      static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        pool.set_sample_weight(allocated.index, 0.0f);
      };
      pool.set_bssrdf_param(allocated.index, {.radius = radius,
                                              .albedo = albedo,
                                              .anisotropy = anisotropy,
                                              .ior = ior,
                                              .alpha = alpha});
    };
  };
}

void thin_subsurface_setup(ShaderData &shader_data, Expr<luisa::float3> normal,
                           Expr<luisa::float3> weight, Expr<float> anisotropy,
                           Expr<float> roughness,
                           Expr<luisa::float3> color) noexcept {
  if (shader_data.closure == nullptr) {
    return;
  }
  const auto reflection_weight =
      clamp(0.5f * (1.0f - anisotropy), 0.0f, 1.0f) * weight;
  const auto transmission_weight =
      clamp(0.5f * (1.0f + anisotropy), 0.0f, 1.0f) * weight;
  const Bool nonzero = any(reflection_weight != make_float3(0.0f)) |
                       any(transmission_weight != make_float3(0.0f));
  $if(nonzero & (shader_data.closure->left() != 0u)) {
    $if(roughness < 1.0e-5f) {
      diffuse_setup(shader_data, normal, reflection_weight);
      translucent_setup(shader_data, normal, transmission_weight);
    }
    $else {
      const auto param =
          oren_nayar_param(color, dot(normal, shader_data.wi), roughness);
      auto &pool = *shader_data.closure;
      const auto reflection = bsdf_allocate(shader_data, reflection_weight);
      $if(reflection.valid) {
        pool.set_type(reflection.index,
                      static_cast<std::uint32_t>(CLOSURE_BSDF_OREN_NAYAR_ID));
        pool.set_normal(reflection.index, normal);
        pool.set_oren_nayar_param(reflection.index, param);
        shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
      };
      const auto transmission = bsdf_allocate(shader_data, transmission_weight);
      $if(transmission.valid) {
        pool.set_type(
            transmission.index,
            static_cast<std::uint32_t>(CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID));
        pool.set_normal(transmission.index, -normal);
        pool.set_oren_nayar_param(transmission.index, param);
        shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval |
                            shader_data_bsdf_has_transmission;
      };
    };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
