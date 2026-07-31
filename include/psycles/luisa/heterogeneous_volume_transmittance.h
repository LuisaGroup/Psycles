#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_transmittance.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/heterogeneous_volume_candidate.h>
#include <psycles/luisa/heterogeneous_volume_collision.h>

namespace psycles::luisa_backend {

struct HeterogeneousVolumeTransmittanceResult {
    Float3 transmittance;
    Float probability_mass;
    UInt base_samples;
    UInt independent_estimators;
    UInt evaluations;
};

// Cycles' randomized telescoping residual-ratio estimator for one immutable
// heterogeneous majorant segment. It is used by non-MIS equiangular direct
// sampling and, later, heterogeneous shadow transport. Extinction always
// comes from the original closure graph at each generated point.
class HeterogeneousVolumeTransmittance final {

  public:
    [[nodiscard]]
    HeterogeneousVolumeTransmittanceResult
    evaluate(
        const VolumeMajorantSegment &segment,
        Float minimum,
        Float maximum,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        const HeterogeneousVolumeCollisionProvider
            &collisions,
        UInt rng_offset) const noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeTransmittanceResult)
