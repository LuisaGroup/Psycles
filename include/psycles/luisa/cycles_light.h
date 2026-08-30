#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_light.h> through the Psycles::luisa target."
#endif

#include <limits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_light {

// Cycles uses FLT_MAX as ShaderData::ray_length for distant lamps. This is a
// shading sentinel, not a traversal endpoint: shadow rays continue to use the
// backend's finite ray maximum.
inline constexpr auto distant_ray_length =
    std::numeric_limits<float>::max();

// Cycles applies an authored lamp/world bounce limit after selecting an
// analytic emitter from the flat distribution. The boundary is inclusive:
// a light with max_bounces == N remains eligible at path bounce N and is
// rejected only at N + 1. Keep this predicate shared by surface and volume
// NEE so those two paths cannot drift at the boundary.
[[nodiscard]] inline luisa::compute::Bool select_reached_max_bounces(
    luisa::compute::UInt bounce,
    luisa::compute::UInt max_bounces) noexcept {
    return bounce > max_bounces;
}

}// namespace psycles::luisa_backend::cycles_light
