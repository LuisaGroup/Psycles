#pragma once

#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

// Camera dimensions are sampled separately from ray construction. This makes
// the Cycles RNG boundary explicit and prevents geometric camera refactors
// from consuming or reordering path dimensions.
struct CameraDimensionSample {
    UInt cycles_y;
    UInt rng_hash;
    Float2 filter_sample;
    Float3 lens_time_sample;
};

struct CameraRaySample {
    Var<luisa::compute::Ray> ray;
    Float differential_position;
    Float differential_direction;
};

[[nodiscard]] CameraDimensionSample
sample_camera_dimensions(
    const BufferFloat4 &sobol_table,
    const Var<RenderKernelParameters> &parameters,
    UInt full_x,
    UInt full_y,
    UInt sample_index) noexcept;

[[nodiscard]] CameraRaySample construct_camera_ray(
    const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters,
    UInt full_x,
    UInt full_y,
    const CameraDimensionSample &sample,
    CameraProjection projection,
    bool depth_of_field,
    std::uint32_t aperture_blades,
    float aperture_rotation,
    const SafeNormalizeCallable &safe_normalize) noexcept;

}// namespace psycles::luisa_backend::detail
