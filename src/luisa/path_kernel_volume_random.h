#pragma once

#include <cstdint>

#include <psycles/luisa/heterogeneous_volume_candidate.h>

namespace psycles::luisa_backend::detail {

inline constexpr std::uint32_t
    heterogeneous_tracking_scramble_seed =
        0xe35fad82u;
inline constexpr std::uint32_t
    heterogeneous_shadow_scramble_seed =
        0x8647ace4u;

// Production mapping from a copied Cycles path RNG state to the dimensions
// consumed by heterogeneous camera and shadow transport. The object exists
// only while the host traces the fused Luisa AST.
class PathVolumeTrackingRandomSource final
    : public HeterogeneousVolumeTrackingRandomSource {

  private:
    const luisa::compute::BufferFloat4
        &_sobol_table;
    UInt _sequence_size;
    UInt _sample_index;
    UInt _rng_hash;

    [[nodiscard]] UInt _dimension(
        UInt rng_offset,
        std::uint32_t dimension) const noexcept;

  public:
    PathVolumeTrackingRandomSource(
        const luisa::compute::BufferFloat4
            &sobol_table,
        UInt sequence_size,
        UInt sample_index,
        UInt rng_hash) noexcept;

    [[nodiscard]] Float scatter_distance(
        UInt rng_offset)
        const noexcept override;
    [[nodiscard]] Float shade_offset(
        UInt rng_offset)
        const noexcept override;
    [[nodiscard]] Float expansion_order(
        UInt rng_offset)
        const noexcept override;
    [[nodiscard]] Float
    transmittance_shade_offset(
        UInt rng_offset)
        const noexcept override;
};

}// namespace psycles::luisa_backend::detail
