/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_bsdf.h"

#include "cycles_svm_ashikhmin_shirley.h"
#include "cycles_svm_hair.h"
#include "cycles_svm_microfacet.h"
#include "cycles_svm_microfacet_scattering.h"
#include "cycles_svm_principled_hair_chiang.h"
#include "cycles_svm_principled_hair_huang.h"
#include "cycles_svm_ray_portal.h"
#include "cycles_svm_sheen.h"
#include "cycles_svm_simple_closure.h"
#include "cycles_svm_toon.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;

namespace {

[[nodiscard]] constexpr bool closure_enabled(ClosureTypeMask mask,
                                             std::uint32_t type) noexcept {
  return type < 64u && (mask & (ClosureTypeMask{1u} << type)) != 0u;
}

[[nodiscard]] constexpr ClosureTypeMask
closure_bit(std::uint32_t type) noexcept {
  return ClosureTypeMask{1u} << type;
}

/* Exact CLOSURE_IS_BSDF_MICROFACET interval members, including the two
 * authoring-only multi-GGX values. Retained SVM closures have already been
 * normalized by setup, but keeping the complete source predicate here makes
 * host reachability pruning commute with Cycles' device classification. */
inline constexpr auto microfacet_type_mask =
    closure_bit(closure_type::type_microfacet_ggx) |
    closure_bit(closure_type::type_microfacet_beckmann) |
    closure_bit(closure_type::type_microfacet_multi_ggx) |
    closure_bit(closure_type::type_ashikhmin_shirley) |
    closure_bit(closure_type::type_microfacet_beckmann_refraction) |
    closure_bit(closure_type::type_microfacet_ggx_refraction) |
    closure_bit(closure_type::type_thin_glass_transmission) |
    closure_bit(closure_type::type_microfacet_beckmann_glass) |
    closure_bit(closure_type::type_microfacet_ggx_glass) |
    closure_bit(closure_type::type_microfacet_multi_ggx_glass);

[[nodiscard]] Bool is_diffuse(Expr<std::uint32_t> type) noexcept {
  return (type >= closure_type::type_diffuse) &
         (type <= closure_type::type_translucent);
}

[[nodiscard]] Bool is_singular(Expr<std::uint32_t> type) noexcept {
  return (type == closure_type::type_transparent) |
         (type == closure_type::type_ray_portal);
}

[[nodiscard]] Bool is_microfacet(Expr<std::uint32_t> type) noexcept {
  return ((type >= closure_type::type_microfacet_ggx) &
          (type <= closure_type::type_ashikhmin_shirley)) |
         ((type >= closure_type::type_microfacet_beckmann_refraction) &
          (type <= closure_type::type_thin_glass_transmission)) |
         ((type >= closure_type::type_microfacet_beckmann_glass) &
          (type <= closure_type::type_microfacet_multi_ggx_glass));
}

[[nodiscard]] Bool is_curve(const ShaderData &shader_data) noexcept {
  return (shader_data.type & primitive_curve) != 0u;
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float ggx_masking(Expr<float> alpha_squared,
                                Expr<float> cosine) noexcept {
  const auto squared_alpha_tangent =
      alpha_squared * max(1.0f / square(cosine) - 1.0f, 0.0f);
  const auto lambda = 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
  return 1.0f / (1.0f + lambda);
}

[[nodiscard]] Float bump_shadowing_term(const ShaderData &shader_data,
                                        const ShaderClosureCommon &closure,
                                        Expr<luisa::float3> direction,
                                        Expr<bool> is_eval) noexcept {
  Float result = 1.0f;
  $if((!all(closure.N == shader_data.N)) & (!is_curve(shader_data))) {
    const auto cosine_smooth_direction = dot(shader_data.N, direction);
    const auto cosine_smooth_closure = dot(shader_data.N, closure.N);
    const auto cosine_closure_direction = dot(closure.N, direction);
    const auto diffuse = is_diffuse(closure.type);
    const auto crosses_smooth_surface = cosine_smooth_direction *
                                            cosine_smooth_closure *
                                            cosine_closure_direction <
                                        0.0f;
    $if(crosses_smooth_surface & (is_eval | diffuse)) { result = 0.0f; }
    $elif(diffuse &
          ((shader_data.flag & shader_data_use_bump_map_correction) != 0u)) {
      const auto cosine_i = abs(cosine_smooth_direction);
      const auto cosine_d = abs(cosine_smooth_closure);
      $if((cosine_d >= 1.0f) | (cosine_i >= 1.0f)) { result = 1.0f; }
      $elif(cosine_i < 1.0e-6f) { result = 0.0f; }
      $else {
        const auto tangent_d_squared = 1.0f / square(cosine_d) - 1.0f;
        const auto bump_alpha_squared =
            clamp(0.125f * tangent_d_squared, 0.0f, 1.0f);
        result = ggx_masking(bump_alpha_squared, cosine_i);
      };
    };
  };
  return result;
}

[[nodiscard]] Float shift_cos_in(Expr<float> cosine,
                                 Expr<float> frequency) noexcept {
  const auto clamped_cosine = min(cosine, 1.0f);
  const auto angle = acos(clamped_cosine);
  return max(cos(angle * frequency), 0.0f) / clamped_cosine;
}

void assign(BsdfEvaluation &destination,
            const BsdfEvaluation &source) noexcept {
  destination.value = source.value;
  destination.pdf = source.pdf;
}

void assign(BsdfSample &destination, const BsdfSample &source) noexcept {
  destination.value = source.value;
  destination.wo = source.wo;
  destination.pdf = source.pdf;
  destination.sampled_roughness = source.sampled_roughness;
  destination.eta = source.eta;
  destination.label = source.label;
}

[[nodiscard]] BsdfEvaluation
microfacet_eval(const KernelGlobals &kernel_globals, const ClosurePool &pool,
                Expr<std::uint32_t> index, Expr<luisa::float3> wi,
                Expr<luisa::float3> wo, bool ggx) noexcept {
  // Cycles dispatches the Fresnel tag inside microfacet_fresnel(), after the
  // shared half-vector calculation. Dispatching here clones the entire
  // geometry/distribution algorithm for every extra-payload type.
  const auto closure = pool.microfacet(index);
  if (ggx) {
    return bsdf_microfacet_ggx_eval(kernel_globals, closure, wi, wo);
  }
  return bsdf_microfacet_beckmann_eval(kernel_globals, closure, wi, wo);
}

[[nodiscard]] BsdfSample
microfacet_sample(const KernelGlobals &kernel_globals, const ClosurePool &pool,
                  Expr<std::uint32_t> index,
                  Expr<luisa::float3> geometric_normal, Expr<luisa::float3> wi,
                  Expr<luisa::float3> random, bool ggx) noexcept {
  const auto closure = pool.microfacet(index);
  if (ggx) {
    return bsdf_microfacet_ggx_sample(kernel_globals, closure,
                                     geometric_normal, wi, random);
  }
  return bsdf_microfacet_beckmann_sample(kernel_globals, closure,
                                        geometric_normal, wi, random);
}

void set_microfacet_roughness_eta(BsdfRoughnessEta &result,
                                  const ClosurePool &pool,
                                  Expr<std::uint32_t> index,
                                  Expr<luisa::float3> wo) noexcept {
  const auto common = pool.common(index);
  const auto param = pool.microfacet_param(index);
  result.roughness = make_float2(param.alpha_x, param.alpha_y);
  result.eta = select(1.0f, param.ior, dot(common.N, wo) < 0.0f);
}

} // namespace

Float bsdf_get_specular_roughness_squared(
    const ClosurePool &pool, Expr<std::uint32_t> closure_index) noexcept {
  const auto common = pool.common(closure_index);
  Float result = 1.0f;
  $if(is_singular(common.type)) { result = 0.0f; }
  $elif(is_microfacet(common.type)) {
    const auto param = pool.microfacet_param(closure_index);
    result = param.alpha_x * param.alpha_y;
  };
  return result;
}

Float bsdf_get_roughness_pass_squared(
    const ClosurePool &pool, Expr<std::uint32_t> closure_index) noexcept {
  const auto common = pool.common(closure_index);
  Float result = bsdf_get_specular_roughness_squared(pool, closure_index);
  $if((common.type == closure_type::type_oren_nayar) |
      (common.type == closure_type::type_rough_translucent)) {
    const auto roughness = pool.oren_nayar(closure_index).param.roughness;
    const auto squared = roughness * roughness;
    result = squared * squared;
  }
  $elif(closure_type::is_bsdf_diffuse(common.type)) { result = -1.0f; };
  return result;
}

BsdfSample bsdf_sample(const KernelGlobals &kernel_globals,
                       ShaderData &shader_data,
                       Expr<std::uint32_t> closure_index,
                       Expr<luisa::float3> random,
                       ClosureTypeMask closure_types) noexcept {
  auto &pool = *shader_data.closure;
  const auto common = pool.common(closure_index);
  const auto geometric_normal =
      select(shader_data.Ng, common.N, is_curve(shader_data));
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(0.0f),
                    .eta = 0.0f,
                    .label = closure_type::label_none};

  $switch(common.type) {
    if (closure_enabled(closure_types, closure_type::type_diffuse)) {
      $case(closure_type::type_diffuse) {
        assign(result, bsdf_diffuse_sample(common, geometric_normal,
                                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_oren_nayar)) {
      $case(closure_type::type_oren_nayar) {
        assign(result, bsdf_oren_nayar_sample(pool.oren_nayar(closure_index),
                                              geometric_normal, shader_data.wi,
                                              random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_rough_translucent)) {
      $case(closure_type::type_rough_translucent) {
        assign(result, bsdf_rough_translucent_sample(
                           pool.oren_nayar(closure_index), geometric_normal,
                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_translucent)) {
      $case(closure_type::type_translucent) {
        assign(result, bsdf_translucent_sample(common, geometric_normal,
                                               shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_transparent)) {
      $case(closure_type::type_transparent) {
        assign(result, bsdf_transparent_sample(common, geometric_normal,
                                               shader_data.wi));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ray_portal)) {
      $case(closure_type::type_ray_portal){
          /* Cycles kernel_asserts here: ray portals are consumed outside the
           * ordinary BSDF sampler. Release kernels retain LABEL_NONE. */
      };
    }
    if (closure_enabled(closure_types, closure_type::type_microfacet_ggx)) {
      $case(closure_type::type_microfacet_ggx) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, true));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_refraction)) {
      $case(closure_type::type_microfacet_ggx_refraction) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, true));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_glass)) {
      $case(closure_type::type_microfacet_ggx_glass) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, true));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_thin_glass_transmission)) {
      $case(closure_type::type_thin_glass_transmission) {
        assign(result, bsdf_thin_glass_transmission_sample(
                           kernel_globals, pool.microfacet(closure_index),
                           geometric_normal, shader_data.wi, random));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann)) {
      $case(closure_type::type_microfacet_beckmann) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, false));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_refraction)) {
      $case(closure_type::type_microfacet_beckmann_refraction) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, false));
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_glass)) {
      $case(closure_type::type_microfacet_beckmann_glass) {
        assign(result, microfacet_sample(kernel_globals, pool, closure_index,
                                         geometric_normal, shader_data.wi,
                                         random, false));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_shirley)) {
      $case(closure_type::type_ashikhmin_shirley) {
        assign(result, bsdf_ashikhmin_shirley_sample(
                           pool.microfacet(closure_index), geometric_normal,
                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_velvet)) {
      $case(closure_type::type_ashikhmin_velvet) {
        assign(result, bsdf_ashikhmin_velvet_sample(
                           pool.velvet(closure_index), geometric_normal,
                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_diffuse_toon)) {
      $case(closure_type::type_diffuse_toon) {
        assign(result, bsdf_diffuse_toon_sample(pool.toon(closure_index),
                                                geometric_normal,
                                                shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_glossy_toon)) {
      $case(closure_type::type_glossy_toon) {
        assign(result, bsdf_glossy_toon_sample(pool.toon(closure_index),
                                               geometric_normal, shader_data.wi,
                                               random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_reflection)) {
      $case(closure_type::type_hair_reflection) {
        assign(result, bsdf_hair_reflection_sample(
                           pool.hair(closure_index), geometric_normal,
                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_transmission)) {
      $case(closure_type::type_hair_transmission) {
        assign(result, bsdf_hair_transmission_sample(
                           pool.hair(closure_index), geometric_normal,
                           shader_data.wi, random.xy()));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
      $case(closure_type::type_hair_chiang) {
        assign(result, bsdf_hair_chiang_sample(kernel_globals,
                                               pool.chiang_hair(closure_index),
                                               shader_data, random));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
      $case(closure_type::type_hair_huang) {
        assign(result, bsdf_hair_huang_sample(kernel_globals,
                                              pool.huang_hair(closure_index),
                                              shader_data, random));
      };
    }
    if (closure_enabled(closure_types, closure_type::type_sheen)) {
      $case(closure_type::type_sheen) {
        assign(result,
               bsdf_sheen_sample(pool.sheen(closure_index), geometric_normal,
                                 shader_data.wi, random.xy()));
      };
    }
    $default{};
  };

  $if((result.label & closure_type::label_transmit) != 0u) {
    const auto threshold =
        kernel_globals.transparent_roughness_squared_threshold();
    $if((threshold >= 0.0f) &
        ((result.label & closure_type::label_diffuse) == 0u) &
        (bsdf_get_specular_roughness_squared(pool, closure_index) <=
         threshold)) {
      result.label |= closure_type::label_transmit_transparent;
    };
  }
  $elif(result.label != closure_type::label_none) {
    const auto frequency =
        kernel_globals.object_shadow_terminator_shading_offset(
            shader_data.object);
    $if(frequency > 1.0f) {
      result.value *= shift_cos_in(dot(result.wo, common.N), frequency);
    };
    result.value *= bump_shadowing_term(shader_data, common, result.wo, false);
  };
  return result;
}

BsdfRoughnessEta bsdf_roughness_eta(const ClosurePool &pool,
                                    Expr<std::uint32_t> closure_index,
                                    Expr<luisa::float3> wo,
                                    ClosureTypeMask closure_types) noexcept {
  const auto common = pool.common(closure_index);
  BsdfRoughnessEta result{.roughness = make_float2(1.0f), .eta = 1.0f};
  $switch(common.type) {
    if (closure_enabled(closure_types, closure_type::type_transparent)) {
      $case(closure_type::type_transparent) {
        result.roughness = make_float2(0.0f);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ray_portal)) {
      $case(closure_type::type_ray_portal) {
        result.roughness = make_float2(0.0f);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_microfacet_ggx)) {
      $case(closure_type::type_microfacet_ggx) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann)) {
      $case(closure_type::type_microfacet_beckmann) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_refraction)) {
      $case(closure_type::type_microfacet_beckmann_refraction) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_refraction)) {
      $case(closure_type::type_microfacet_ggx_refraction) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_thin_glass_transmission)) {
      $case(closure_type::type_thin_glass_transmission) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_glass)) {
      $case(closure_type::type_microfacet_beckmann_glass) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_glass)) {
      $case(closure_type::type_microfacet_ggx_glass) {
        set_microfacet_roughness_eta(result, pool, closure_index, wo);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_shirley)) {
      $case(closure_type::type_ashikhmin_shirley) {
        const auto param = pool.microfacet_param(closure_index);
        result.roughness = make_float2(param.alpha_x, param.alpha_y);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_reflection)) {
      $case(closure_type::type_hair_reflection) {
        const auto param = pool.hair(closure_index).param;
        result.roughness = make_float2(param.roughness1, param.roughness2);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_transmission)) {
      $case(closure_type::type_hair_transmission) {
        const auto param = pool.hair(closure_index).param;
        result.roughness = make_float2(param.roughness1, param.roughness2);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
      $case(closure_type::type_hair_chiang) {
        const auto alpha = pool.chiang_hair(closure_index).param.m0_roughness;
        result.roughness = make_float2(alpha);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
      $case(closure_type::type_hair_huang) {
        const auto alpha = pool.huang_hair(closure_index).param.roughness;
        result.roughness = make_float2(alpha);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_sheen)) {
      $case(closure_type::type_sheen) {
        const auto alpha = pool.sheen(closure_index).param.roughness;
        result.roughness = make_float2(alpha);
      };
    }
    $default{};
  };
  return result;
}

UInt bsdf_label(const KernelGlobals &kernel_globals, const ClosurePool &pool,
                Expr<std::uint32_t> closure_index, Expr<luisa::float3> wo,
                ClosureTypeMask closure_types) noexcept {
  const auto common = pool.common(closure_index);
  UInt result = closure_type::label_none;
  $switch(common.type) {
    if (closure_enabled(closure_types, closure_type::type_diffuse)) {
      $case(closure_type::type_diffuse) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_bssrdf_burley)) {
      $case(closure_type::type_bssrdf_burley) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_bssrdf_random_walk)) {
      $case(closure_type::type_bssrdf_random_walk) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_bssrdf_random_walk_legacy)) {
      $case(closure_type::type_bssrdf_random_walk_legacy) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_bssrdf_random_walk_skin)) {
      $case(closure_type::type_bssrdf_random_walk_skin) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_oren_nayar)) {
      $case(closure_type::type_oren_nayar) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_translucent)) {
      $case(closure_type::type_translucent) {
        result = closure_type::label_transmit | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_rough_translucent)) {
      $case(closure_type::type_rough_translucent) {
        result = closure_type::label_transmit | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_transparent)) {
      $case(closure_type::type_transparent) {
        result = closure_type::label_transmit | closure_type::label_transparent;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ray_portal)) {
      $case(closure_type::type_ray_portal) {
        result = closure_type::label_transmit | closure_type::label_ray_portal;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_microfacet_ggx)) {
      $case(closure_type::type_microfacet_ggx) {
        const auto glossy = pool.microfacet_param(closure_index).alpha_x *
                                pool.microfacet_param(closure_index).alpha_y >
                            closure_type::microfacet_singular_alpha_product;
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   glossy);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann)) {
      $case(closure_type::type_microfacet_beckmann) {
        const auto param = pool.microfacet_param(closure_index);
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   param.alpha_x * param.alpha_y >
                       closure_type::microfacet_singular_alpha_product);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_refraction)) {
      $case(closure_type::type_microfacet_ggx_refraction) {
        const auto param = pool.microfacet_param(closure_index);
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   param.alpha_x * param.alpha_y >
                       closure_type::microfacet_singular_alpha_product);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_refraction)) {
      $case(closure_type::type_microfacet_beckmann_refraction) {
        const auto param = pool.microfacet_param(closure_index);
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   param.alpha_x * param.alpha_y >
                       closure_type::microfacet_singular_alpha_product);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_glass)) {
      $case(closure_type::type_microfacet_ggx_glass) {
        const auto param = pool.microfacet_param(closure_index);
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   param.alpha_x * param.alpha_y >
                       closure_type::microfacet_singular_alpha_product);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_glass)) {
      $case(closure_type::type_microfacet_beckmann_glass) {
        const auto param = pool.microfacet_param(closure_index);
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            select(closure_type::label_singular, closure_type::label_glossy,
                   param.alpha_x * param.alpha_y >
                       closure_type::microfacet_singular_alpha_product);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_thin_glass_transmission)) {
      $case(closure_type::type_thin_glass_transmission) {
        result = closure_type::label_transmit | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_shirley)) {
      $case(closure_type::type_ashikhmin_shirley) {
        result = closure_type::label_reflect | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_velvet)) {
      $case(closure_type::type_ashikhmin_velvet) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_diffuse_toon)) {
      $case(closure_type::type_diffuse_toon) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_glossy_toon)) {
      $case(closure_type::type_glossy_toon) {
        result = closure_type::label_reflect | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_reflection)) {
      $case(closure_type::type_hair_reflection) {
        result = closure_type::label_reflect | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_transmission)) {
      $case(closure_type::type_hair_transmission) {
        result = closure_type::label_transmit | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
      $case(closure_type::type_hair_chiang) {
        result =
            select(closure_type::label_reflect, closure_type::label_transmit,
                   dot(common.N, wo) < 0.0f) |
            closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
      $case(closure_type::type_hair_huang) {
        result = closure_type::label_reflect | closure_type::label_glossy;
      };
    }
    if (closure_enabled(closure_types, closure_type::type_sheen)) {
      $case(closure_type::type_sheen) {
        result = closure_type::label_reflect | closure_type::label_diffuse;
      };
    }
    $default{};
  };

  $if((result & closure_type::label_transmit) != 0u) {
    const auto threshold =
        kernel_globals.transparent_roughness_squared_threshold();
    $if((threshold >= 0.0f) & (bsdf_get_specular_roughness_squared(
                                   pool, closure_index) <= threshold)) {
      result |= closure_type::label_transmit_transparent;
    };
  };
  return result;
}

BsdfEvaluation bsdf_eval(const KernelGlobals &kernel_globals,
                         ShaderData &shader_data,
                         Expr<std::uint32_t> closure_index,
                         Expr<luisa::float3> wo,
                         ClosureTypeMask closure_types) noexcept {
  auto &pool = *shader_data.closure;
  const auto common = pool.common(closure_index);
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  const auto bump = bump_shadowing_term(shader_data, common, wo, true);
  $if(bump != 0.0f) {
    $switch(common.type) {
      if (closure_enabled(closure_types, closure_type::type_diffuse)) {
        $case(closure_type::type_diffuse) {
          assign(result, bsdf_diffuse_eval(common, shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_oren_nayar)) {
        $case(closure_type::type_oren_nayar) {
          assign(result, bsdf_oren_nayar_eval(pool.oren_nayar(closure_index),
                                              shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_rough_translucent)) {
        $case(closure_type::type_rough_translucent) {
          assign(result,
                 bsdf_rough_translucent_eval(pool.oren_nayar(closure_index),
                                             shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_translucent)) {
        $case(closure_type::type_translucent) {
          assign(result, bsdf_translucent_eval(common, shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_transparent)) {
        $case(closure_type::type_transparent) {
          assign(result, bsdf_transparent_eval(common, shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_ray_portal)) {
        $case(closure_type::type_ray_portal) {
          assign(result, bsdf_ray_portal_eval(pool.ray_portal(closure_index),
                                              shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_microfacet_ggx)) {
        $case(closure_type::type_microfacet_ggx) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, true));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_microfacet_ggx_refraction)) {
        $case(closure_type::type_microfacet_ggx_refraction) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, true));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_microfacet_ggx_glass)) {
        $case(closure_type::type_microfacet_ggx_glass) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, true));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_thin_glass_transmission)) {
        $case(closure_type::type_thin_glass_transmission) {
          assign(result, bsdf_thin_glass_transmission_eval(
                             kernel_globals, pool.microfacet(closure_index),
                             shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_microfacet_beckmann)) {
        $case(closure_type::type_microfacet_beckmann) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, false));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_microfacet_beckmann_refraction)) {
        $case(closure_type::type_microfacet_beckmann_refraction) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, false));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_microfacet_beckmann_glass)) {
        $case(closure_type::type_microfacet_beckmann_glass) {
          assign(result, microfacet_eval(kernel_globals, pool, closure_index,
                                         shader_data.wi, wo, false));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_ashikhmin_shirley)) {
        $case(closure_type::type_ashikhmin_shirley) {
          assign(result,
                 bsdf_ashikhmin_shirley_eval(pool.microfacet(closure_index),
                                             shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_ashikhmin_velvet)) {
        $case(closure_type::type_ashikhmin_velvet) {
          assign(result, bsdf_ashikhmin_velvet_eval(pool.velvet(closure_index),
                                                    shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_diffuse_toon)) {
        $case(closure_type::type_diffuse_toon) {
          assign(result, bsdf_diffuse_toon_eval(pool.toon(closure_index),
                                                shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_glossy_toon)) {
        $case(closure_type::type_glossy_toon) {
          assign(result, bsdf_glossy_toon_eval(pool.toon(closure_index),
                                               shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_hair_reflection)) {
        $case(closure_type::type_hair_reflection) {
          assign(result, bsdf_hair_reflection_eval(pool.hair(closure_index),
                                                   shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types,
                          closure_type::type_hair_transmission)) {
        $case(closure_type::type_hair_transmission) {
          assign(result, bsdf_hair_transmission_eval(pool.hair(closure_index),
                                                     shader_data.wi, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
        $case(closure_type::type_hair_chiang) {
          assign(result, bsdf_hair_chiang_eval(kernel_globals,
                                               pool.chiang_hair(closure_index),
                                               shader_data, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
        $case(closure_type::type_hair_huang) {
          assign(result, bsdf_hair_huang_eval(kernel_globals,
                                              pool.huang_hair(closure_index),
                                              shader_data, wo));
        };
      }
      if (closure_enabled(closure_types, closure_type::type_sheen)) {
        $case(closure_type::type_sheen) {
          assign(result, bsdf_sheen_eval(pool.sheen(closure_index),
                                         shader_data.wi, wo));
        };
      }
      $default{};
    };

    result.value *= bump;
    const auto frequency =
        kernel_globals.object_shadow_terminator_shading_offset(
            shader_data.object);
    $if(frequency > 1.0f) {
      const auto cosine = dot(wo, common.N);
      $if(cosine >= 0.0f) { result.value *= shift_cos_in(cosine, frequency); };
    };
  };
  return result;
}

void bsdf_blur(ClosurePool &pool, Expr<std::uint32_t> closure_index,
               Expr<float> roughness, ClosureTypeMask closure_types) noexcept {
  const auto common = pool.common(closure_index);
  $switch(common.type) {
    if (closure_enabled(closure_types, closure_type::type_microfacet_ggx)) {
      $case(closure_type::type_microfacet_ggx) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_refraction)) {
      $case(closure_type::type_microfacet_ggx_refraction) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_ggx_glass)) {
      $case(closure_type::type_microfacet_ggx_glass) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_thin_glass_transmission)) {
      $case(closure_type::type_thin_glass_transmission) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann)) {
      $case(closure_type::type_microfacet_beckmann) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_refraction)) {
      $case(closure_type::type_microfacet_beckmann_refraction) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types,
                        closure_type::type_microfacet_beckmann_glass)) {
      $case(closure_type::type_microfacet_beckmann_glass) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_ashikhmin_shirley)) {
      $case(closure_type::type_ashikhmin_shirley) {
        auto param = pool.microfacet_param(closure_index);
        param.alpha_x = max(roughness, param.alpha_x);
        param.alpha_y = max(roughness, param.alpha_y);
        pool.set_microfacet_param(closure_index, param);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
      $case(closure_type::type_hair_chiang) {
        auto closure = pool.chiang_hair(closure_index);
        bsdf_hair_chiang_blur(closure, roughness);
        pool.set_chiang_hair_param(closure_index, closure.param);
      };
    }
    if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
      $case(closure_type::type_hair_huang) {
        auto closure = pool.huang_hair(closure_index);
        bsdf_hair_huang_blur(closure, roughness);
        pool.set_huang_hair(closure_index, closure.param, closure.extra);
      };
    }
    $default{};
  };
}

Float3 bsdf_albedo(const KernelGlobals &kernel_globals,
                   const ShaderData &shader_data,
                   Expr<std::uint32_t> closure_index, Expr<bool> reflection,
                   Expr<bool> transmission,
                   ClosureTypeMask closure_types) noexcept {
  const auto &pool = *shader_data.closure;
  const auto common = pool.common(closure_index);
  Float3 result = common.weight;
  if ((closure_types & microfacet_type_mask) != 0u) {
    $if(is_microfacet(common.type)) {
      const auto param = pool.microfacet_param(closure_index);
      Float3 factor = make_float3(0.0f);
      $if(param.fresnel_type ==
          static_cast<std::uint32_t>(MicrofacetFresnel::conductor)) {
        factor = bsdf_microfacet_estimate_albedo(
            kernel_globals, pool.microfacet_conductor(closure_index),
            shader_data.wi, reflection, transmission);
      }
      $elif(param.fresnel_type ==
            static_cast<std::uint32_t>(MicrofacetFresnel::f82_tint)) {
        factor = bsdf_microfacet_estimate_albedo(
            kernel_globals, pool.microfacet_f82_tint(closure_index),
            shader_data.wi, reflection, transmission);
      }
      $else {
        factor = bsdf_microfacet_estimate_albedo(
            kernel_globals, pool.microfacet(closure_index), shader_data.wi,
            reflection, transmission);
      };
      result *= factor;
    };
  }
  if (closure_enabled(closure_types, closure_type::type_hair_chiang)) {
    $if(common.type == closure_type::type_hair_chiang) {
      result *=
          bsdf_hair_chiang_albedo(pool.chiang_hair(closure_index), shader_data);
    };
  }
  if (closure_enabled(closure_types, closure_type::type_hair_huang)) {
    $if(common.type == closure_type::type_hair_huang) {
      result *= bsdf_hair_huang_albedo(pool.huang_hair(closure_index));
    };
  }
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
