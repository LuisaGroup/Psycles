#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_ray_differential.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_ray_differential {

struct SurfaceBounceUpdate {
    luisa::compute::Float position;
    luisa::compute::Float direction;
};

[[nodiscard]] inline luisa::compute::Float widen_direction(
    luisa::compute::Float previous_direction,
    luisa::compute::Float average_roughness_squared) noexcept {
    using namespace luisa::compute;
    return select(
        previous_direction,
        max(previous_direction,
            sqrt(max(average_roughness_squared, 0.0f))),
        average_roughness_squared > 0.0f);
}

// Cycles initializes a surface NEE shadow ray at the current compact
// positional radius, then widens its angular radius with exactly the same
// directional-mixture measure used by forward BSDF sampling.
[[nodiscard]] inline SurfaceBounceUpdate for_surface_shadow(
    luisa::compute::Float previous_direction,
    luisa::compute::Float surface_position,
    luisa::compute::Float average_roughness_squared) noexcept {
    return {
        .position = surface_position,
        .direction = widen_direction(
            previous_direction, average_roughness_squared)};
}

// Cycles stores ray differentials as two scalar cone radii. A regular surface
// bounce transfers the positional radius to the hit and widens the angular
// radius by the PDF-weighted BSDF roughness. Transparent continuation changes
// neither value because it is the same geometrical ray with an advanced tmin.
[[nodiscard]] inline SurfaceBounceUpdate after_surface_bounce(
    luisa::compute::Float previous_position,
    luisa::compute::Float previous_direction,
    luisa::compute::Float surface_position,
    luisa::compute::Float average_roughness_squared,
    luisa::compute::Bool transparent) noexcept {
    using namespace luisa::compute;
    const auto regular = for_surface_shadow(
        previous_direction,
        surface_position,
        average_roughness_squared);
    return {
        .position = select(
            regular.position, previous_position, transparent),
        .direction = select(
            regular.direction, previous_direction, transparent)};
}

} // namespace psycles::luisa_backend::cycles_ray_differential
