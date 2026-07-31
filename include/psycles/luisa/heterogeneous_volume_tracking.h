#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_tracking.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

// A scalar majorant is the measure shared by every spectral channel in
// Cycles' weighted delta tracker. `majorant_exceeded` is not an alternate
// integration mode: it exposes a violated preprocessing contract so callers
// and regression probes can reject an unsound bound.
struct VolumeNullEventCoefficients {
    Float3 sigma_n;
    Float majorant;
    Bool majorant_exceeded;
};

// The complete local state transition at one candidate collision. Keeping
// real and null continuations side by side makes their common normalization
// explicit and prevents either estimator from silently changing measure.
struct HeterogeneousVolumeCollision {
    VolumeNullEventCoefficients null_event;
    Float3 normalized_throughput;
    Float3 emission;
    Float3 channel_pdf;
    Float scatter_probability;
    Float null_probability;
    Float3 scatter_throughput;
    Float3 null_throughput;
    Float null_transmittance;
    Bool absorption_only;
    Bool active;
};

struct HeterogeneousVolumeEventSelection {
    Float random;
    Bool scattered;
};

// Luisa AST component for the local weighted-delta-tracking transition used
// by current Cycles. Spatial traversal, majorant construction, reservoirs,
// phase closures, and direct-light MIS remain separate host-stage components.
// This class owns only the formal null/real collision measure.
class HeterogeneousVolumeTracking {

  private:
    [[nodiscard]] static Float3 _safe_divide_color(
        Float3 numerator,
        Float3 denominator,
        float fallback) noexcept;
    [[nodiscard]] static Float3 _safe_divide_scalar(
        Float3 numerator,
        Float denominator) noexcept;

  public:
    [[nodiscard]] Float candidate_distance(
        Float random,
        Float majorant) const noexcept;

    [[nodiscard]] VolumeNullEventCoefficients
    null_event_coefficients(
        Float3 sigma_t,
        Float sampled_majorant) const noexcept;

    [[nodiscard]] HeterogeneousVolumeCollision
    evaluate_collision(
        const VolumeCoefficients &coefficients,
        Float sampled_majorant,
        Float candidate_distance,
        Float3 throughput,
        Float transmittance) const noexcept;

    [[nodiscard]] HeterogeneousVolumeEventSelection
    select_event(
        Float random,
        Float scatter_probability) const noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeNullEventCoefficients)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::HeterogeneousVolumeCollision)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::HeterogeneousVolumeEventSelection)
