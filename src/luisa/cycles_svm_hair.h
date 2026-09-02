/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 standalone Hair Reflection/Transmission SVM
 * transition. The cursor consumes exactly one SVMNodeHairBsdfData record. */
void node_hair(Cursor &cursor, Stack &stack,
               luisa::compute::Expr<std::uint32_t> type,
               luisa::compute::Expr<luisa::float3> closure_weight,
               luisa::compute::Expr<float> mix_weight,
               ShaderData &shader_data) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_hair_reflection_eval(const HairClosure &closure,
                          luisa::compute::Expr<luisa::float3> wi,
                          luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_hair_transmission_eval(const HairClosure &closure,
                            luisa::compute::Expr<luisa::float3> wi,
                            luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_hair_reflection_sample(
    const HairClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfSample bsdf_hair_transmission_sample(
    const HairClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
