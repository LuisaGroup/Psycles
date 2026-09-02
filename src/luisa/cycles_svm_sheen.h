/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Cycles 5.2.1 standalone Microfiber Sheen and Ashikhmin Velvet cases. The
 * cursor consumes exactly one SVMNodeSimpleBsdfData record. */
void node_sheen(const KernelGlobals &kernel_globals, Cursor &cursor,
                Stack &stack, luisa::compute::Expr<std::uint32_t> type,
                luisa::compute::Expr<luisa::float3> closure_weight,
                luisa::compute::Expr<float> mix_weight,
                ShaderData &shader_data) noexcept;

/* Exact Cycles Principled SheenBsdf transition. The returned value is the
 * post-LTC closure albedo consumed by closure_layering_weight(). */
[[nodiscard]] luisa::compute::Float3
principled_sheen_setup(const KernelGlobals &kernel_globals,
                       ShaderData &shader_data, const PathState &path_state,
                       luisa::compute::Expr<luisa::float3> input_weight,
                       luisa::compute::Expr<luisa::float3> normal,
                       luisa::compute::Expr<float> roughness) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_sheen_eval(const SheenClosure &closure,
                luisa::compute::Expr<luisa::float3> wi,
                luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_sheen_sample(const SheenClosure &closure,
                  luisa::compute::Expr<luisa::float3> Ng,
                  luisa::compute::Expr<luisa::float3> wi,
                  luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_ashikhmin_velvet_eval(const VelvetClosure &closure,
                           luisa::compute::Expr<luisa::float3> wi,
                           luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_ashikhmin_velvet_sample(
    const VelvetClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
