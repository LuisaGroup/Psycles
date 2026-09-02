/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 standalone Ray Portal SVM transition. The cursor
 * consumes exactly one SVMNodeRayPortalBsdfData record. */
void node_ray_portal(Cursor &cursor, Stack &stack,
                     luisa::compute::Expr<luisa::float3> closure_weight,
                     luisa::compute::Expr<float> mix_weight,
                     ShaderData &shader_data) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_ray_portal_eval(const RayPortalClosure &closure,
                     luisa::compute::Expr<luisa::float3> wi,
                     luisa::compute::Expr<luisa::float3> wo) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
