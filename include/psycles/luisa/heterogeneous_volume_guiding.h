#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_guiding.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

struct HeterogeneousVolumeGuidingState {
    Float3 scattered_radiance;
    Float3 transmitted_radiance;
    Float majorant_optical_depth;
    Bool enabled;
};

struct HeterogeneousVolumeGuidingSample {
    Float scatter_probability;
    Float majorant_scale;
    Bool enabled;
};

// Exact scalar VSPG measure used by Cycles' heterogeneous null-collision
// integrator. It changes the free-flight rate and final scatter/transmit
// reservoir probabilities; local real/null collision weights remain owned by
// HeterogeneousVolumeTracking.
class HeterogeneousVolumeScatterProbability final {

  public:
    [[nodiscard]] HeterogeneousVolumeGuidingSample
    evaluate(
        const HeterogeneousVolumeGuidingState
            &state) const noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeGuidingState)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::
        HeterogeneousVolumeGuidingSample)
