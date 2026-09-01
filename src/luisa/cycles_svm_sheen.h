/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles Principled SheenBsdf transition. The returned value is the
 * post-LTC closure albedo consumed by closure_layering_weight(). */
[[nodiscard]] luisa::compute::Float3
principled_sheen_setup(const KernelGlobals &kernel_globals,
                       ShaderData &shader_data, const PathState &path_state,
                       luisa::compute::Expr<luisa::float3> input_weight,
                       luisa::compute::Expr<luisa::float3> normal,
                       luisa::compute::Expr<float> roughness) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
