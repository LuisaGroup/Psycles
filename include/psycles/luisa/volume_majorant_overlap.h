#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_majorant_overlap.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/volume_majorant_traversal.h>
#include <psycles/luisa/volume_stack.h>

namespace psycles::luisa_backend {

struct VolumeMajorantEntrySpace {
    Float3 ray_origin;
    Float3 ray_direction;
    Float object_density;
};

struct VolumeMajorantRuntimeExtrema {
    Float minimum;
    Float maximum;
};

// Host-stage polymorphic boundary for object/world coordinates and the
// runtime extrema policy. The default extrema method is the common Cycles
// camera/no-Light-Path path. A production subclass can emit the exact
// interval re-evaluation for dynamic Light Path nodes without changing the
// overlap measure.
class VolumeMajorantEntryProvider {

  public:
    virtual ~VolumeMajorantEntryProvider() noexcept =
        default;

    [[nodiscard]] virtual VolumeMajorantEntrySpace
    entry_space(
        const VolumeStackEntry &entry,
        Float3 world_ray_origin,
        Float3 world_ray_direction) const noexcept = 0;

    [[nodiscard]] virtual VolumeMajorantRuntimeExtrema
    extrema(
        const VolumeStackEntry &entry,
        const VolumeMajorantLeaf &leaf,
        Float object_density) const noexcept;
};

struct VolumeMajorantSegment {
    Float minimum;
    Float maximum;
    Float sigma_minimum;
    Float sigma_maximum;
    UInt object;
    UInt shader;
    UInt node;
    Bool valid;
    Bool no_overlap;
    Bool lookup_complete;
};

// Exact ordered reduction of the per-entry Cycles hierarchical DDA states.
// One selected traversal persists across segments; every other stack entry is
// reconstructed at the new minimum. Extrema are accumulated in stack order,
// and <= selection deliberately makes the last equal endpoint active.
//
// The resource buffers must be nonempty. Declared counts are checked before
// every dynamic range and root-node lookup; hierarchy topology is validated
// before upload. Missing or malformed coverage invalidates the segment rather
// than silently treating a medium as zero.
class VolumeMajorantOverlapTraversal {

  private:
    struct RootLookup {
        luisa::compute::Var<
            VolumeMajorantRootGpu> root;
        Bool found;
    };

    const luisa::compute::BufferVar<
        VolumeMajorantNodeGpu> &_nodes;
    const luisa::compute::BufferVar<
        VolumeMajorantRootGpu> &_roots;
    const luisa::compute::BufferVar<
        VolumeMajorantRootRangeGpu> &_ranges;
    std::uint32_t _node_count;
    std::uint32_t _root_count;
    std::uint32_t _range_count;
    std::uint32_t _world_range;
    const VolumeStack &_stack;
    const VolumeMajorantEntryProvider &_provider;
    Float3 _world_ray_origin;
    Float3 _world_ray_direction;
    Float _ray_maximum;

    VolumeMajorantTraversal _active;
    VolumeStackEntry _active_entry;
    Float _active_object_density;
    Float _sigma_minimum;
    Float _sigma_maximum;
    Bool _active_valid;
    Bool _no_overlap;
    Bool _lookup_complete;

    [[nodiscard]] RootLookup _find_root(
        const VolumeStackEntry &entry)
        const noexcept;
    void _select_entry(
        Bool condition,
        const VolumeStackEntry &entry,
        Float object_density) noexcept;
    [[nodiscard]] Bool _setup() noexcept;

  public:
    VolumeMajorantOverlapTraversal(
        const luisa::compute::BufferVar<
            VolumeMajorantNodeGpu> &nodes,
        const luisa::compute::BufferVar<
            VolumeMajorantRootGpu> &roots,
        const luisa::compute::BufferVar<
            VolumeMajorantRootRangeGpu> &ranges,
        std::uint32_t node_count,
        std::uint32_t root_count,
        std::uint32_t range_count,
        std::uint32_t world_range,
        const VolumeStack &stack,
        const VolumeMajorantEntryProvider &provider,
        Float3 world_ray_origin,
        Float3 world_ray_direction,
        Float ray_minimum,
        Float ray_maximum) noexcept;

    [[nodiscard]] VolumeMajorantSegment
    current() const noexcept;
    [[nodiscard]] Bool advance() noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantEntrySpace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantRuntimeExtrema)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantSegment)
