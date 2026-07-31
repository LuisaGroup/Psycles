#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_majorant_prepass.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/stacked_volume.h>
#include <psycles/luisa/volume_majorant_hierarchy.h>

namespace psycles::luisa_backend {

inline constexpr std::uint32_t
    volume_majorant_samples_per_cell = 16u;
inline constexpr float
    volume_majorant_voxel_padding = 0.2f;

struct VolumeMajorantGrid {
    Float3 minimum;
    Float3 maximum;
    luisa::compute::Float4x4 object_to_world;
};

struct VolumeMajorantCellExtrema {
    Float minimum;
    Float maximum;
};

// Records the raw Cycles volume-density bake for one 128^3 cell. The original
// SurfaceDispatch graph is evaluated at all sixteen padded Sobol-Burley
// points. This is an acceleration-data producer, not a CPU renderer or a
// closure bake: runtime transport evaluates the same graph again.
class VolumeMajorantPrepass {

  private:
    const SurfaceDispatch &_surfaces;
    const VolumeStackEntryPointProvider &_points;

  public:
    VolumeMajorantPrepass(
        const SurfaceDispatch &surfaces,
        const VolumeStackEntryPointProvider &points) noexcept;

    [[nodiscard]] VolumeMajorantCellExtrema
    evaluate_cell(
        const VolumeStackEntry &entry,
        const ShaderServices &services,
        const VolumeMajorantGrid &grid,
        UInt cell_index) const noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantGrid)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantCellExtrema)
