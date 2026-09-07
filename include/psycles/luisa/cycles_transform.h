#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_transform.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_transform {

// Native float projective transform with Cycles' zero-homogeneous-coordinate
// branch. No software rounding, precision annotation or device inversion.
[[nodiscard]] luisa::compute::Float3 perspective(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept;

// Cycles' affine transforms are part of the geometric predicate contract.
// Keeping their fused operation tree explicit prevents a backend matrix
// lowering from introducing an extra rounding step at a triangle surface.
[[nodiscard]] luisa::compute::Float3 point(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept;

[[nodiscard]] luisa::compute::Float3 direction(
    luisa::compute::Expr<luisa::float4x4> transform,
    luisa::compute::Expr<luisa::float3> value) noexcept;

[[nodiscard]] luisa::compute::Float3
direction_transposed(luisa::compute::Expr<luisa::float4x4> transform,
                     luisa::compute::Expr<luisa::float3> value) noexcept;

} // namespace psycles::luisa_backend::cycles_transform
