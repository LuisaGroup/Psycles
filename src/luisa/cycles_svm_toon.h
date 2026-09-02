/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 standalone Diffuse/Glossy Toon SVM transition. The
 * cursor consumes exactly one SVMNodeToonBsdfData record. */
void node_toon(const KernelGlobals &kernel_globals, Cursor &cursor,
               Stack &stack, luisa::compute::Expr<std::uint32_t> type,
               luisa::compute::Expr<luisa::float3> closure_weight,
               luisa::compute::Expr<float> mix_weight, ShaderData &shader_data,
               const PathState &path_state) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_diffuse_toon_eval(const ToonClosure &closure,
                       luisa::compute::Expr<luisa::float3> wi,
                       luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_diffuse_toon_sample(const ToonClosure &closure,
                         luisa::compute::Expr<luisa::float3> Ng,
                         luisa::compute::Expr<luisa::float3> wi,
                         luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_glossy_toon_eval(const ToonClosure &closure,
                      luisa::compute::Expr<luisa::float3> wi,
                      luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_glossy_toon_sample(const ToonClosure &closure,
                        luisa::compute::Expr<luisa::float3> Ng,
                        luisa::compute::Expr<luisa::float3> wi,
                        luisa::compute::Expr<luisa::float2> random) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
