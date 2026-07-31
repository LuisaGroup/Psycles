#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_direct_sampling.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/volume_stack.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

struct VolumeDirectSampleInterval {
    luisa::compute::Float minimum;
    luisa::compute::Float maximum;
};

struct VolumeDirectSamplingState {
    luisa::compute::UInt method;
    luisa::compute::Float random;
    luisa::compute::Bool use_mis;
    luisa::compute::Bool enabled;
};

struct VolumeEquiangularCoefficients {
    luisa::compute::Float3 light_position;
    VolumeDirectSampleInterval interval;
};

struct VolumeEquiangularSample {
    luisa::compute::Float distance;
    luisa::compute::Float pdf;
    luisa::compute::Bool valid;
};

// Pure Luisa-AST implementation of Cycles' direct-volume technique choice and
// equiangular measure. Geometry components provide the sampled emitter point
// and its visible ray interval; this component owns only probability measure.
class VolumeDirectSampling {

  public:
    [[nodiscard]] VolumeDirectSamplingState prepare(
        luisa::compute::UInt requested_method,
        luisa::compute::Float random,
        luisa::compute::Bool enabled) const noexcept;

    [[nodiscard]] VolumeEquiangularSample sample_equiangular(
        luisa::compute::Float3 ray_origin,
        luisa::compute::Float3 ray_direction,
        const VolumeEquiangularCoefficients &coefficients,
        luisa::compute::Float random) const noexcept;

    [[nodiscard]] luisa::compute::Float equiangular_pdf(
        luisa::compute::Float3 ray_origin,
        luisa::compute::Float3 ray_direction,
        const VolumeEquiangularCoefficients &coefficients,
        luisa::compute::Float distance) const noexcept;

    [[nodiscard]] luisa::compute::Float power_heuristic(
        luisa::compute::Float selected_pdf,
        luisa::compute::Float competing_pdf) const noexcept;
};

}// namespace psycles::luisa_backend
