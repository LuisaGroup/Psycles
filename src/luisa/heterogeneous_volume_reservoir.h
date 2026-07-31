#pragma once

#include <psycles/luisa/heterogeneous_volume_guiding.h>

namespace psycles::luisa_backend::detail {

struct HeterogeneousVolumeReservoirCandidate {
    Float3 emission;
    Float distance;
    Float3 throughput;
    Float distance_pdf;
    Bool valid;
};

struct HeterogeneousVolumeReservoirResult {
    Float3 indirect_throughput;
    Float3 emission;
    Float indirect_distance;
    Float3 direct_throughput;
    Float direct_distance;
    Float direct_distance_pdf;
    Float unguided_scatter_probability;
    Float guided_scatter_probability;
    Float random;
    Bool indirect_scatter;
    Bool direct_scatter;
    Bool empty;
};

// Stateful host-stage implementation of Cycles' streaming volume reservoir.
// It owns the conditional scatter-candidate distribution and the final VSPG
// defensive scatter/transmit resampling, but not local collision weights.
class HeterogeneousVolumeReservoir final {

  private:
    Float _total_weight;
    Float _random;
    HeterogeneousVolumeReservoirCandidate
        _candidate;

  public:
    explicit HeterogeneousVolumeReservoir(
        Float random) noexcept;

    [[nodiscard]] Bool empty() const noexcept;
    [[nodiscard]] Float total_weight()
        const noexcept;
    [[nodiscard]] Float random() const noexcept;
    [[nodiscard]]
    HeterogeneousVolumeReservoirCandidate
    candidate() const noexcept;

    void set_random(Float random) noexcept;
    void add(
        Float weight,
        const HeterogeneousVolumeReservoirCandidate
            &candidate) noexcept;

    [[nodiscard]]
    HeterogeneousVolumeReservoirResult
    finalize(
        Float3 transmitted_throughput,
        Float3 transmitted_emission,
        const HeterogeneousVolumeGuidingSample
            &guiding,
        Bool sample_direct_distance) noexcept;
};

}// namespace psycles::luisa_backend::detail

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::detail::
        HeterogeneousVolumeReservoirCandidate)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::detail::
        HeterogeneousVolumeReservoirResult)
