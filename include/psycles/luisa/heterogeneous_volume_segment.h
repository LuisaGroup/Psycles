#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_segment.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <memory>

#include <psycles/luisa/heterogeneous_volume_candidate.h>
#include <psycles/luisa/stacked_volume.h>
#include <psycles/luisa/volume_phase_set.h>

namespace psycles::luisa_backend {

// Host-stage source of original closure coefficients at a candidate point.
// Production implementations evaluate the active VolumeStack at `distance`;
// no coefficient grid or material pre-bake is permitted at this boundary.
class HeterogeneousVolumeCollisionProvider {

  public:
    virtual ~HeterogeneousVolumeCollisionProvider()
        noexcept = default;

    [[nodiscard]] virtual VolumeCoefficients
    evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases) const noexcept = 0;
};

struct HeterogeneousVolumeTransportResult {
    Float3 throughput;
    Float3 emission;
    Float distance;
    Float null_transmittance;
    Float reservoir_random;
    Float optical_depth;
    UInt next_tracking_rng_offset;
    UInt steps;
    Bool selected_scatter;
    Bool traversal_exhausted;
    Bool step_limit_exceeded;
    Bool majorant_exceeded;
    Bool active;
};

struct HeterogeneousVolumeSegmentResult {
    VolumeCoefficients coefficients;
    HeterogeneousVolumeTransportResult transport;
    VolumePhaseSetSample phase;
    Bool scattered;
    Bool phase_failed;
};

// Non-guided indirect branch of Cycles' heterogeneous weighted-delta tracker.
// The component includes deterministic absorption/null continuation,
// reservoir-random event selection, defensive majorant correction, emission
// recursion, low-throughput Russian roulette, and final raw-phase recovery.
// VSPG and the independent direct-light estimator are composed separately.
class HeterogeneousVolumeSegmentComponent {

  public:
    virtual ~HeterogeneousVolumeSegmentComponent()
        noexcept = default;

    [[nodiscard]] virtual HeterogeneousVolumeSegmentResult
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
        Bool terminate) const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<HeterogeneousVolumeSegmentComponent>
make_heterogeneous_volume_segment_component(
    std::size_t closure_allocation_budget);

// Production collision provider for original stacked Volume graphs. The base
// state carries path semantics while position is reconstructed from the ray
// and candidate distance on every evaluation.
[[nodiscard]]
std::unique_ptr<HeterogeneousVolumeCollisionProvider>
make_stacked_heterogeneous_volume_collision_provider(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const VolumeStack &stack,
    const ShaderServices &services,
    const VolumeShadingState &base_state,
    Float3 ray_origin,
    Float3 ray_direction);

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeTransportResult)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeSegmentResult)
