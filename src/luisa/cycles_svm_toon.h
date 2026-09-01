/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 standalone Diffuse/Glossy Toon SVM transition. The
 * cursor consumes exactly one SVMNodeToonBsdfData record. */
void node_toon(const KernelGlobals &kernel_globals, Cursor &cursor,
               Stack &stack, luisa::compute::Expr<std::uint32_t> type,
               luisa::compute::Expr<luisa::float3> closure_weight,
               luisa::compute::Expr<float> mix_weight, ShaderData &shader_data,
               const PathState &path_state) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
