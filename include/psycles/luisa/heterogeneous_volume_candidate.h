#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_candidate.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/heterogeneous_volume_tracking.h>
#include <psycles/luisa/volume_majorant_overlap.h>
#include <psycles/sampling/tabulated_sobol.h>

namespace psycles::luisa_backend {

inline constexpr std::uint32_t
    heterogeneous_volume_maximum_steps = 1024u;
inline constexpr std::uint32_t
    heterogeneous_volume_tracking_rng_stride =
        sampling::tabulated_sobol::
            bounce_dimension_count;
static_assert(
    heterogeneous_volume_tracking_rng_stride ==
        16u,
    "Current Cycles allocates 16 dimensions per bounce.");

// Host-stage random-access interface for the copied tracking RNG state.
// Implementations map an explicit dimension offset to Cycles samples; calls
// do not mutate the enclosing path RNG. The walk alone owns the +16
// transition after a candidate.
class HeterogeneousVolumeTrackingRandomSource {

  public:
    virtual ~HeterogeneousVolumeTrackingRandomSource()
        noexcept = default;

    [[nodiscard]] virtual Float scatter_distance(
        UInt rng_offset) const noexcept = 0;

    // Cycles uses the y component of PRNG_VOLUME_SHADE_OFFSET's 2D sample for
    // runtime majorant extrema, not the 1D value at the same dimension.
    [[nodiscard]] virtual Float shade_offset(
        UInt rng_offset) const noexcept = 0;
};

struct HeterogeneousVolumeCandidate {
    VolumeMajorantSegment segment;
    Float distance;
    Float step_distance;
    Float sampled_majorant;
    Float scatter_random;
    Float optical_depth;
    UInt sample_rng_offset;
    UInt next_rng_offset;
    UInt step;
    Bool candidate;
    Bool traversal_exhausted;
    Bool step_limit_exceeded;
};

// Stateful Luisa AST component for Cycles' volume_integrate_advance().
// It owns only candidate free-flight and voxel-boundary continuation:
// collision coefficients, reservoir decisions, VSPG, phase closures, and
// direct-light MIS are independent components composed by the segment stage.
class HeterogeneousVolumeCandidateWalk {

  private:
    VolumeMajorantSegmentSequence &_segments;
    const HeterogeneousVolumeTrackingRandomSource
        &_random;
    HeterogeneousVolumeTracking _tracking;
    Float _majorant_scale;
    Float _distance;
    Float _optical_depth;
    UInt _rng_offset;
    UInt _step;
    Bool _finished;

  public:
    HeterogeneousVolumeCandidateWalk(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        Float majorant_scale,
        Float ray_minimum,
        UInt tracking_rng_offset) noexcept;

    [[nodiscard]] HeterogeneousVolumeCandidate
    advance() noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::HeterogeneousVolumeCandidate)
