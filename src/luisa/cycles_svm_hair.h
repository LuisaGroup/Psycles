/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Direct Luisa projections of the output tuples of Cycles 5.2.1
 * bsdf_hair_*_eval/sample. The value is the unweighted closure value; the
 * caller applies ShaderClosure::weight exactly as Cycles' integrator does. */
struct HairEvaluation {
  luisa::compute::Float3 value;
  luisa::compute::Float pdf;
};

struct HairSample {
  luisa::compute::Float3 value;
  luisa::compute::Float3 wo;
  luisa::compute::Float pdf;
  luisa::compute::Float2 sampled_roughness;
  luisa::compute::Float eta;
  luisa::compute::UInt label;
};

/* Exact Cycles 5.2.1 standalone Hair Reflection/Transmission SVM
 * transition. The cursor consumes exactly one SVMNodeHairBsdfData record. */
void node_hair(Cursor &cursor, Stack &stack,
               luisa::compute::Expr<std::uint32_t> type,
               luisa::compute::Expr<luisa::float3> closure_weight,
               luisa::compute::Expr<float> mix_weight,
               ShaderData &shader_data) noexcept;

[[nodiscard]] HairEvaluation
bsdf_hair_reflection_eval(const HairClosure &closure,
                          luisa::compute::Expr<luisa::float3> wi,
                          luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] HairEvaluation
bsdf_hair_transmission_eval(const HairClosure &closure,
                            luisa::compute::Expr<luisa::float3> wi,
                            luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] HairSample bsdf_hair_reflection_sample(
    const HairClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] HairSample bsdf_hair_transmission_sample(
    const HairClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
