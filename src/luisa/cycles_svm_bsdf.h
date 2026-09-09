/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include <cstdint>

namespace psycles::luisa_backend::cycles_svm::detail {

/* Direct Luisa projections of the output arguments of Cycles 5.2.1
 * bsdf_eval() and bsdf_sample(). Values remain unweighted here: the surface
 * one-sample-model fold applies ShaderClosure::weight exactly once. */
struct BsdfEvaluation {
  luisa::compute::Float3 value;
  luisa::compute::Float pdf;
};

struct BsdfSample {
  luisa::compute::Float3 value;
  luisa::compute::Float3 wo;
  luisa::compute::Float pdf;
  luisa::compute::Float2 sampled_roughness;
  luisa::compute::Float eta;
  luisa::compute::UInt label;
};

struct BsdfRoughnessEta {
  luisa::compute::Float2 roughness;
  luisa::compute::Float eta;
};

/* Host/JIT reachability mask. Bit N denotes the exact Cycles ClosureType N;
 * it only suppresses source-switch cases proven absent from a scene and never
 * changes the retained closure ABI or device dispatch model. */
using ClosureTypeMask = std::uint64_t;
inline constexpr ClosureTypeMask all_closure_types = ~ClosureTypeMask{0u};

[[nodiscard]] constexpr ClosureTypeMask closure_types_for_kernel_features(
    std::uint32_t kernel_features) noexcept {
  namespace closure = ::psycles::luisa_backend::cycles_closure;
  auto result = all_closure_types;
  // Direct projection of Cycles features.h: __HAIR__ encloses every hair
  // consumer, __PRINCIPLED_HAIR__ further guards Chiang/Huang, and
  // __SUBSURFACE__ encloses all BSSRDF consumers.
  if ((kernel_features & kernel_feature_hair) == 0u) {
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_reflection);
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_transmission);
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_chiang);
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_huang);
  } else if ((kernel_features & kernel_feature_node_principled_hair) == 0u) {
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_chiang);
    result &= ~(ClosureTypeMask{1u} << closure::type_hair_huang);
  }
  if ((kernel_features & kernel_feature_subsurface) == 0u) {
    result &= ~(ClosureTypeMask{1u} << closure::type_bssrdf_burley);
    result &= ~(ClosureTypeMask{1u} << closure::type_bssrdf_random_walk);
    result &= ~(ClosureTypeMask{1u} <<
                closure::type_bssrdf_random_walk_legacy);
    result &= ~(ClosureTypeMask{1u} <<
                closure::type_bssrdf_random_walk_skin);
  }
  // A reachable Ray Portal producer contributes NODE_PORTAL to the scene
  // feature mask. Its absence proves this closure and its consumer cases
  // unreachable, including scenes that use ordinary Light Path nodes.
  if ((kernel_features & kernel_feature_node_portal) == 0u) {
    result &= ~(ClosureTypeMask{1u} << closure::type_ray_portal);
  }
  return result;
}

[[nodiscard]] luisa::compute::Float bsdf_get_specular_roughness_squared(
    const ClosurePool &pool,
    luisa::compute::Expr<std::uint32_t> closure_index) noexcept;

// Cycles' Roughness data pass intentionally differs from transport
// roughness: diffuse closures without an authored roughness are excluded,
// while Oren-Nayar and Rough Translucent expose their node parameter.
[[nodiscard]] luisa::compute::Float bsdf_get_roughness_pass_squared(
    const ClosurePool &pool,
    luisa::compute::Expr<std::uint32_t> closure_index) noexcept;

[[nodiscard]] BsdfSample bsdf_sample(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<luisa::float3> random,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

[[nodiscard]] BsdfRoughnessEta bsdf_roughness_eta(
    const ClosurePool &pool,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<luisa::float3> wo,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

[[nodiscard]] luisa::compute::UInt bsdf_label(
    const KernelGlobals &kernel_globals, const ClosurePool &pool,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<luisa::float3> wo,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_eval(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<luisa::float3> wo,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

void bsdf_blur(ClosurePool &pool,
               luisa::compute::Expr<std::uint32_t> closure_index,
               luisa::compute::Expr<float> roughness,
               ClosureTypeMask closure_types = all_closure_types) noexcept;

[[nodiscard]] luisa::compute::Float3 bsdf_albedo(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<bool> reflection,
    luisa::compute::Expr<bool> transmission,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
