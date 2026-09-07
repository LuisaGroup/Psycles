/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

// Device-side sizeof/offsetof emitted by cycles_svm_volume_oracle.hip.
// Slots still have ShaderData::closure[]'s 80-byte stride; only the active
// ShaderVolumeClosure prefix and phase-specific parameters are observable.
namespace volume_closure_layout {
inline constexpr std::uint32_t size = 32u;
inline constexpr std::uint32_t henyey_greenstein_g = 20u;
inline constexpr std::uint32_t draine_g = 20u;
inline constexpr std::uint32_t draine_alpha = 24u;
inline constexpr std::uint32_t fournier_forand_c1 = 20u;
inline constexpr std::uint32_t fournier_forand_c2 = 24u;
inline constexpr std::uint32_t fournier_forand_c3 = 28u;
} // namespace volume_closure_layout

void node_closure_volume(const KernelGlobals &kg, Cursor &cursor, Stack &stack,
                         luisa::compute::Expr<luisa::float3> closure_weight,
                         ShaderData &sd,
                         const EvaluationTransition &transition) noexcept;
void node_volume_coefficients(
    const KernelGlobals &kg, Cursor &cursor, Stack &stack,
    luisa::compute::Expr<luisa::float3> scatter_coeffs, ShaderData &sd,
    const PathState &path, const EvaluationTransition &transition) noexcept;
void node_principled_volume(const KernelGlobals &kg, Cursor &cursor,
                            Stack &stack,
                            luisa::compute::Expr<luisa::float3> closure_weight,
                            ShaderData &sd, const PathState &path,
                            const EvaluationTransition &transition) noexcept;
} // namespace psycles::luisa_backend::cycles_svm::detail
