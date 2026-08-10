#pragma once

#include <psycles/luisa/surface.h>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend::detail {

// Device ABI for the exact post-population SurfaceClosurePoint projection.
// Luisa float3 has 16-byte host/device alignment, so three independent float3
// fields would consume the complete 48-byte budget before scalar state. The
// two float4 records below are an explicit bijection over the nine directional
// components; the remaining scalar tail keeps the full record at 48 bytes.
// This is a semantic record, not a generic register block: adding a dependency
// requires naming it in SurfaceClosurePoint and extending this projection.
struct SurfaceClosurePointCall {
    luisa::float4 geometric_normal_and_shading_x{};
    luisa::float4 shading_yz_and_incoming_xy{};
    float incoming_z{};
    luisa::uint ray_visibility{};
    luisa::uint flags{};
};

static_assert(sizeof(SurfaceClosurePointCall) == 48u);

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceClosurePointCall,
    geometric_normal_and_shading_x,
    shading_yz_and_incoming_xy,
    incoming_z,
    ray_visibility,
    flags) {};

namespace psycles::luisa_backend::detail {

inline constexpr std::uint32_t
    surface_closure_point_use_bump_map_correction = 1u << 0u;
inline constexpr std::uint32_t
    surface_closure_point_back_facing = 1u << 1u;

[[nodiscard]] inline luisa::compute::Var<SurfaceClosurePointCall>
pack_surface_closure_point(
    const SurfaceClosurePoint &point) noexcept {
    luisa::compute::Var<SurfaceClosurePointCall> result;
    result.geometric_normal_and_shading_x =
        luisa::compute::make_float4(
            point.geometric_normal, point.shading_normal.x);
    result.shading_yz_and_incoming_xy =
        luisa::compute::make_float4(
            point.shading_normal.y,
            point.shading_normal.z,
            point.incoming.x,
            point.incoming.y);
    result.incoming_z = point.incoming.z;
    result.ray_visibility = point.ray_visibility;
    const auto bump_flag = luisa::compute::select(
        0u,
        surface_closure_point_use_bump_map_correction,
        point.use_bump_map_correction);
    const auto facing_flag = luisa::compute::select(
        0u,
        surface_closure_point_back_facing,
        point.back_facing);
    result.flags = bump_flag | facing_flag;
    return result;
}

[[nodiscard]] inline SurfaceClosurePoint
unpack_surface_closure_point(
    const luisa::compute::Var<SurfaceClosurePointCall> &point) noexcept {
    const auto geometric_and_shading_x =
        point.geometric_normal_and_shading_x;
    const auto shading_yz_and_incoming_xy =
        point.shading_yz_and_incoming_xy;
    return SurfaceClosurePoint{
        geometric_and_shading_x.xyz(),
        luisa::compute::make_float3(
            geometric_and_shading_x.w,
            shading_yz_and_incoming_xy.x,
            shading_yz_and_incoming_xy.y),
        luisa::compute::make_float3(
            shading_yz_and_incoming_xy.z,
            shading_yz_and_incoming_xy.w,
            point.incoming_z),
        point.ray_visibility,
        (point.flags &
         surface_closure_point_use_bump_map_correction) != 0u,
        (point.flags & surface_closure_point_back_facing) != 0u};
}

}// namespace psycles::luisa_backend::detail
