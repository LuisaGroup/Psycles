#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_segment.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <memory>

#include <psycles/luisa/heterogeneous_volume_candidate.h>
#include <psycles/luisa/heterogeneous_volume_collision.h>
#include <psycles/luisa/heterogeneous_volume_guiding.h>
#include <psycles/luisa/volume_direct_sampling.h>
#include <psycles/luisa/volume_phase_set.h>

namespace psycles::luisa_backend {

struct HeterogeneousVolumeTransportResult {
    Float3 throughput;
    Float3 emission;
    Float distance;
    Float null_transmittance;
    Float reservoir_random;
    Float optical_depth;
    Float unguided_scatter_probability;
    Float guided_scatter_probability;
    Float majorant_scale;
    UInt next_tracking_rng_offset;
    UInt steps;
    Bool selected_scatter;
    Bool traversal_exhausted;
    Bool step_limit_exceeded;
    Bool majorant_exceeded;
    Bool active;
};

struct HeterogeneousVolumeDirectInput {
    UInt requested_method;
    Float3 light_position;
    VolumeDirectSampleInterval interval;
    Bool enabled;
};

struct HeterogeneousVolumeDirectSample {
    Float3 throughput;
    Float distance;
    Float distance_pdf;
    Float equiangular_pdf;
    Float mis_weight;
    UInt sample_method;
    Bool use_mis;
    Bool scattered;
};

struct HeterogeneousVolumeSegmentInput {
    VolumeMajorantSegmentSequence &segments;
    const HeterogeneousVolumeTrackingRandomSource
        &random;
    const HeterogeneousVolumeCollisionProvider
        &collisions;
    HeterogeneousVolumeGuidingSample guiding;
    HeterogeneousVolumeDirectInput direct;
    const VolumeDirectDirectionProvider
        *direct_direction;
    Float ray_minimum;
    Float ray_maximum;
    Float3 segment_origin;
    Float3 phase_axis;
    Float3 throughput;
    Float direct_random;
    Float reservoir_random;
    Float2 phase_random;
    UInt tracking_rng_offset;
    Bool terminate;
};

struct HeterogeneousVolumeSegmentResult {
    VolumeCoefficients coefficients;
    HeterogeneousVolumeTransportResult transport;
    VolumePhaseSetSample phase;
    HeterogeneousVolumeDirectSample
        direct_transport;
    VolumePhaseSetEvaluation direct_phase;
    Bool scattered;
    Bool phase_failed;
};

// Cycles' heterogeneous weighted-delta segment. Traversal, raw closure
// evaluation, VSPG reservoir selection, direct distance/equiangular transport,
// MIS, phase recovery, and low-throughput roulette remain typed host-stage
// components but are recorded into one fused Luisa AST through this boundary.
class HeterogeneousVolumeSegmentComponent {

  public:
    virtual ~HeterogeneousVolumeSegmentComponent()
        noexcept = default;

    [[nodiscard]] virtual HeterogeneousVolumeSegmentResult
    emit(
        const HeterogeneousVolumeSegmentInput
            &input) const noexcept = 0;

    // Compatibility entry point for focused unguided/no-NEE fixtures. The
    // canonical production boundary above carries every coupled estimator
    // explicitly.
    [[nodiscard]] HeterogeneousVolumeSegmentResult
    emit(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        const HeterogeneousVolumeCollisionProvider
            &collisions,
        Float majorant_scale,
        Float ray_minimum,
        Float ray_maximum,
        Float3 phase_axis,
        Float3 throughput,
        Float reservoir_random,
        Float2 phase_random,
        UInt tracking_rng_offset,
        Bool terminate) const noexcept;
};

[[nodiscard]]
std::unique_ptr<HeterogeneousVolumeSegmentComponent>
make_heterogeneous_volume_segment_component(
    std::size_t closure_allocation_budget);

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeTransportResult)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeDirectInput)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeDirectSample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeSegmentResult)
