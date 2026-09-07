#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_majorant_prepass.h> through the Psycles::luisa target."
#endif

#include <cstdint>

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
    UInt resolution;
};

struct VolumeMajorantCellExtrema {
    Float minimum;
    Float maximum;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantGrid)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantCellExtrema)
