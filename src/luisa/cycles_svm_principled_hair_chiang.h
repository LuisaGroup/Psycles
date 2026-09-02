/* SPDX-FileCopyrightText: 2018-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Isomorphic Luisa DSL projections of Cycles 5.2.1
 * kernel/closure/bsdf_principled_hair_chiang.h. */
[[nodiscard]] BsdfEvaluation
bsdf_hair_chiang_eval(const KernelGlobals &kernel_globals,
                      const ChiangHairClosure &closure,
                      const ShaderData &shader_data,
                      luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_hair_chiang_sample(const KernelGlobals &kernel_globals,
                        const ChiangHairClosure &closure,
                        const ShaderData &shader_data,
                        luisa::compute::Expr<luisa::float3> random) noexcept;

void bsdf_hair_chiang_blur(ChiangHairClosure &closure,
                           luisa::compute::Expr<float> roughness) noexcept;

[[nodiscard]] luisa::compute::Float3
bsdf_hair_chiang_albedo(const ChiangHairClosure &closure,
                        const ShaderData &shader_data) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
