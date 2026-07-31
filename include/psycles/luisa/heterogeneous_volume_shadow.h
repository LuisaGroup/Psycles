#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_shadow.h> through the Psycles::luisa target."
#endif

#include <memory>

#include <psycles/luisa/heterogeneous_volume_transmittance.h>

namespace psycles::luisa_backend {

inline constexpr float
    heterogeneous_volume_shadow_throughput_epsilon =
        1.0e-6f;

struct HeterogeneousVolumeShadowResult {
    Float3 throughput;
    UInt next_tracking_rng_offset;
    UInt groups;
    UInt source_segments;
    UInt closure_evaluations;
    Bool traversal_exhausted;
    Bool lookup_complete;
    Bool throughput_terminated;
};

// Cycles' shadow-only majorant coalescing and residual-ratio traversal.
// Adjacent immutable hierarchy segments are merged until
// sigma_range * length >= 1, reducing the mandatory one-sample-per-segment
// cost without changing the original closure evaluations.
class HeterogeneousVolumeShadowComponent {

  public:
    virtual ~HeterogeneousVolumeShadowComponent()
        noexcept = default;

    [[nodiscard]]
    virtual HeterogeneousVolumeShadowResult
    emit(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        const HeterogeneousVolumeCollisionProvider
            &collisions,
        Float3 throughput,
        UInt tracking_rng_offset)
        const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<HeterogeneousVolumeShadowComponent>
make_heterogeneous_volume_shadow_component();

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeShadowResult)
