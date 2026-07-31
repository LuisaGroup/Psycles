#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_analytic_light_sampling.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/analytic_light_sampling.h>

namespace psycles::luisa_backend {

struct VolumePointLightSampleInput {
    luisa::compute::Float3 reference;
    luisa::compute::Float3 center;
    luisa::compute::Float radius;
    luisa::compute::Bool sphere;
    luisa::compute::Float3 axis_x;
    luisa::compute::Float3 axis_y;
    luisa::compute::Float3 axis_z;
    luisa::compute::Float3 axis_scale;
    luisa::compute::Float2 random;
    luisa::compute::Bool normalize_power;
};

struct VolumeSpotLightSampleInput {
    VolumePointLightSampleInput point;
    luisa::compute::Float spot_angle;
    luisa::compute::Float spot_smooth;
};

// Host-stage Luisa AST component for the two distinct Cycles sampling
// contracts used by volume NEE. A segment proposal deliberately keeps
// zero-attenuation spot samples and always samples a visible sphere cap.
// Sampling from the final collision point uses the ordinary finite-emitter
// measure and may instead select the spot spread cone.
class VolumeAnalyticLightSampling {

  public:
    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    point(
        const VolumePointLightSampleInput
            &input) const noexcept;

    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    spot_from_segment(
        const VolumeSpotLightSampleInput
            &input) const noexcept;

    [[nodiscard]]
    analytic_light_sampling::FiniteLightSample
    spot_from_position(
        const VolumeSpotLightSampleInput
            &input) const noexcept;
};

}// namespace psycles::luisa_backend
