#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_scatter_probability.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

struct VolumeScatterProbabilityGuidingState {
    Float3 scattered_radiance;
    Float3 transmitted_radiance;
    Float majorant_optical_depth;
    Bool enabled;
};

// Host-stage implementation of Cycles' homogeneous volume scattering
// probability guiding (VSPG) measure. This component changes only the
// scatter-vs-transmit sampling probability; transport weights remain the
// responsibility of HomogeneousVolumeTransport.
class HomogeneousVolumeScatterProbability final {

  private:
    [[nodiscard]] static Float3 _safe_divide(
        Float3 numerator,
        Float3 denominator) noexcept;

  public:
    [[nodiscard]] Float3 evaluate(
        const VolumeCoefficients &coefficients,
        Float distance,
        Bool terminate,
        const VolumeScatterProbabilityGuidingState &guiding) const noexcept;
};

}// namespace psycles::luisa_backend
