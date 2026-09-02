/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

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

[[nodiscard]] luisa::compute::Float bsdf_get_specular_roughness_squared(
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
