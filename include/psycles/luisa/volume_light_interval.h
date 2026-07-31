#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_light_interval.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/volume_direct_sampling.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

struct VolumeLightIntervalResult {
    VolumeDirectSampleInterval interval;
    luisa::compute::Bool valid;
};

struct VolumeSpotIntervalInput {
    luisa::compute::Float3 ray_origin;
    luisa::compute::Float3 ray_direction;
    VolumeDirectSampleInterval interval;
    luisa::compute::Float3 center;
    luisa::compute::Float3 axis_x;
    luisa::compute::Float3 axis_y;
    luisa::compute::Float3 axis_z;
    luisa::compute::Float3 axis_scale;
    luisa::compute::Float radius;
    luisa::compute::Float spot_angle;
};

struct VolumeAreaIntervalInput {
    luisa::compute::Float3 ray_origin;
    luisa::compute::Float3 ray_direction;
    VolumeDirectSampleInterval interval;
    luisa::compute::Float3 center;
    luisa::compute::Float3 axis_u;
    luisa::compute::Float3 axis_v;
    luisa::compute::Float3 axis_z;
    luisa::compute::Float length_u;
    luisa::compute::Float length_v;
    luisa::compute::Float spread;
    luisa::compute::Bool ellipse;
};

struct VolumeTriangleIntervalInput {
    luisa::compute::Float3 ray_origin;
    luisa::compute::Float3 ray_direction;
    VolumeDirectSampleInterval interval;
    luisa::compute::Float3 plane_point;
    // Cycles' authored orientation in world space: transform the local
    // normal by the inverse transpose. This is equivalent to taking the
    // transformed-edge cross product and then applying
    // SD_OBJECT_NEGATIVE_SCALE.
    luisa::compute::Float3 normal;
    luisa::compute::Bool sample_front;
    luisa::compute::Bool sample_back;
};

// Host-stage Luisa AST component for Cycles' volume_valid_direct_ray_segment
// geometry. It clips one closed ray interval to the region that can receive
// energy from a selected finite emitter. Sampling and radiometric evaluation
// remain separate components.
class VolumeLightInterval {

  public:
    [[nodiscard]] VolumeLightIntervalResult spot(
        const VolumeSpotIntervalInput &input) const noexcept;

    [[nodiscard]] VolumeLightIntervalResult area(
        const VolumeAreaIntervalInput &input) const noexcept;

    [[nodiscard]] VolumeLightIntervalResult triangle(
        const VolumeTriangleIntervalInput &input) const noexcept;
};

}// namespace psycles::luisa_backend
