/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 standalone Hair Reflection/Transmission SVM
 * transition. The cursor consumes exactly one SVMNodeHairBsdfData record. */
void node_hair(Cursor &cursor, Stack &stack,
               luisa::compute::Expr<std::uint32_t> type,
               luisa::compute::Expr<luisa::float3> closure_weight,
               luisa::compute::Expr<float> mix_weight,
               ShaderData &shader_data) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
