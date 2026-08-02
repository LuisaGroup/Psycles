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

struct VolumeDirectDirectionSample {
    luisa::compute::Float3 direction;
    luisa::compute::Bool valid;
};

// A scene-light component implements this host-stage phase protocol. Volume
// integration chooses a collision distance first, then asks the provider to
// resample the emitter from that exact point with the original PRNG_LIGHT
// coordinates. Constant emission is evaluated before the receiving phase;
// deferred emission is evaluated only after a non-zero phase and before the
// outer path stage traces the shadow ray. The methods mutate the provider's
// bound VolumeDirectLightSample while Luisa records one fused device AST.
class VolumeDirectLightProvider {

  public:
    virtual ~VolumeDirectLightProvider() noexcept =
        default;

    [[nodiscard]] virtual VolumeDirectDirectionSample
    sample_direction(
        luisa::compute::Float distance)
        const noexcept = 0;

    virtual void evaluate_constant_emission()
        const noexcept = 0;

    virtual void evaluate_deferred_emission(
        luisa::compute::Bool receiving_nonzero)
        const noexcept = 0;
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

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        VolumeDirectDirectionSample)
