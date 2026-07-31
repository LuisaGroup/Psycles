#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_majorant_hierarchy.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

inline constexpr std::uint32_t
    volume_majorant_maximum_depth = 7u;
inline constexpr std::uint32_t
    volume_majorant_grid_resolution =
        1u << volume_majorant_maximum_depth;
inline constexpr float
    volume_majorant_split_threshold = 1.442f;

struct VolumeMajorantExtrema {
    float minimum{};
    float maximum{};
};

struct VolumeMajorantBounds {
    luisa::float3 minimum{};
    luisa::float3 maximum{};
};

struct VolumeMajorantNodeGpu {
    std::int32_t parent{-1};
    std::int32_t first_child{-1};
    float sigma_minimum{};
    float sigma_maximum{};
};

struct VolumeMajorantRootGpu {
    luisa::float3 scale{};
    std::uint32_t node{};
    luisa::float3 translation{};
    // Low Cycles shader identity. Stack entries may carry high surface
    // flags; production root lookup compares their masked identity.
    std::uint32_t shader{~std::uint32_t{0u}};
};

// Contiguous roots belonging to one internal scene instance. A separate final
// range represents the World entry. Keeping the range explicit avoids
// coupling production lookup to host pointer ordering while preserving
// Cycles' one-root-per-object-per-shader identity.
struct VolumeMajorantRootRangeGpu {
    std::uint32_t offset{};
    std::uint32_t count{};
};

struct VolumeMajorantHierarchy {
    VolumeMajorantRootGpu root;
    std::vector<VolumeMajorantNodeGpu> nodes;
};

struct VolumeMajorantBuildResult {
    VolumeMajorantHierarchy hierarchy;
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Host metadata builder matching Cycles' volume Octree construction. The
// input extrema are produced by a Luisa shader-evaluation prepass; this class
// only reduces them into acceleration metadata and never evaluates radiance.
class VolumeMajorantHierarchyBuilder {

  public:
    [[nodiscard]] static constexpr std::size_t
    required_extrema_count() noexcept {
        return static_cast<std::size_t>(
                   volume_majorant_grid_resolution) *
               volume_majorant_grid_resolution *
               volume_majorant_grid_resolution;
    }

    [[nodiscard]] VolumeMajorantBuildResult build(
        const VolumeMajorantBounds &bounds,
        std::span<const VolumeMajorantExtrema>
            extrema,
        float volume_scale = 1.0f) const;
};

}// namespace psycles::luisa_backend

LUISA_STRUCT(
    psycles::luisa_backend::VolumeMajorantNodeGpu,
    parent,
    first_child,
    sigma_minimum,
    sigma_maximum) {};
LUISA_STRUCT(
    psycles::luisa_backend::VolumeMajorantRootGpu,
    scale,
    node,
    translation,
    shader) {};
LUISA_STRUCT(
    psycles::luisa_backend::VolumeMajorantRootRangeGpu,
    offset,
    count) {};
