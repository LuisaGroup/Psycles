/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Direct Luisa projection of Cycles 5.2.1 film/light_passes.h::BsdfEval.
 * Reflection, transmission and volume are mutually exclusive at one bounce,
 * so Cycles retains only diffuse/glossy components in addition to the sum. */
struct SurfaceShaderBsdfEval {
  luisa::compute::Float3 diffuse;
  luisa::compute::Float3 glossy;
  luisa::compute::Float3 sum;
  luisa::compute::Float pdf;
  luisa::compute::Float average_roughness_squared;
};

/* Result of Cycles' surface_shader_bsdf_bssrdf_pick(). `random` contains the
 * exact z-coordinate rescaling used to preserve stratification. Cycles returns
 * only the selected closure pointer and mutates random; the pool index is the
 * corresponding pointer-free representation here. */
struct SurfaceShaderClosurePick {
  luisa::compute::UInt index;
  luisa::compute::Float3 random;
};

/* Exact non-guided result of surface_shader_bsdf_sample_closure(). Selection
 * state remains with SurfaceShaderClosurePick instead of being duplicated
 * across a bounce boundary. */
struct SurfaceShaderBsdfSample {
  SurfaceShaderBsdfEval evaluation;
  luisa::compute::Float3 wo;
  luisa::compute::Float2 sampled_roughness;
  luisa::compute::Float eta;
  luisa::compute::UInt label;
};

[[nodiscard]] SurfaceShaderBsdfEval surface_shader_bsdf_eval(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> wo,
    luisa::compute::Expr<std::uint32_t> light_shader_flags,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

[[nodiscard]] SurfaceShaderClosurePick surface_shader_bsdf_bssrdf_pick(
    const ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] luisa::compute::Float3 surface_shader_bssrdf_sample_weight(
    const ShaderData &shader_data,
    luisa::compute::Expr<std::uint32_t> closure_index) noexcept;

[[nodiscard]] SurfaceShaderBsdfSample surface_shader_bsdf_sample_closure(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const SurfaceShaderClosurePick &pick,
    ClosureTypeMask closure_types = all_closure_types) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
