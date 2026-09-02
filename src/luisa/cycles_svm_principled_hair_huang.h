/* SPDX-FileCopyrightText: 2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Isomorphic Luisa DSL projections of Cycles 5.2.1
 * kernel/closure/bsdf_principled_hair_huang.h. The functions consume and
 * mutate ShaderData::lcg_state at the same control-flow points as Cycles. */
[[nodiscard]] BsdfSample
bsdf_hair_huang_sample(const KernelGlobals &kernel_globals,
                       const HuangHairClosure &closure, ShaderData &shader_data,
                       luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_hair_huang_eval(const KernelGlobals &kernel_globals,
                     const HuangHairClosure &closure, ShaderData &shader_data,
                     luisa::compute::Expr<luisa::float3> wo) noexcept;

void bsdf_hair_huang_blur(HuangHairClosure &closure,
                          luisa::compute::Expr<float> roughness) noexcept;

[[nodiscard]] luisa::compute::Float3
bsdf_hair_huang_albedo(const HuangHairClosure &closure) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
