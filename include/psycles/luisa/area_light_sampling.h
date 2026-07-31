#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/area_light_sampling.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/analytic_light_sampling.h>

namespace psycles::luisa_backend {

struct AreaLightSampleInput {
    luisa::compute::Float3 reference;
    luisa::compute::Float3 center;
    luisa::compute::Float3 axis_u;
    luisa::compute::Float3 axis_v;
    luisa::compute::Float3 axis_z;
    luisa::compute::Float length_u;
    luisa::compute::Float length_v;
    luisa::compute::Float spread;
    luisa::compute::Bool ellipse;
    luisa::compute::Bool full_spread;
    luisa::compute::Float2 random;
    luisa::compute::Bool normalize_power;
};

// Host-stage Luisa AST component for Cycles' two analytic area-light
// sampling measures. The segment proposal samples the original primitive in
// area measure. The collision-position method performs the exact spread
// clamp and then selects the resulting rectangle solid-angle or ellipse area
// measure. Both methods retain the original lamp UV and radiometric contract.
class AreaLightSampling {

  public:
    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    from_segment(
        const AreaLightSampleInput
            &input) const noexcept;

    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    from_position(
        const AreaLightSampleInput
            &input) const noexcept;

    // Evaluate Cycles' collision-position measure for a ray that has already
    // intersected the authored area primitive. This applies the same spread
    // clamp as from_position without resampling the known hit point.
    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    from_intersection(
        const AreaLightSampleInput
            &input,
        luisa::compute::Float3 direction,
        luisa::compute::Float distance) const noexcept;
};

}// namespace psycles::luisa_backend
