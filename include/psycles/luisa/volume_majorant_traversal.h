#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_majorant_traversal.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>
#include <psycles/luisa/volume_majorant_hierarchy.h>

namespace psycles::luisa_backend {

struct VolumeMajorantLeaf {
    Float minimum;
    Float maximum;
    Float sigma_minimum;
    Float sigma_maximum;
    UInt node;
    Bool valid;
};

// Single-octree hierarchical DDA matching the current Cycles bitwise [1, 2)
// traversal. Overlapping-stack reduction is deliberately a separate
// component; this object owns only one root and its adjacent-leaf sequence.
class VolumeMajorantTraversal {

  private:
    const luisa::compute::BufferVar<
        VolumeMajorantNodeGpu> &_nodes;
    UInt _root_node;
    UInt _node;
    Float _ray_maximum;
    Float _minimum;
    Float _maximum;
    Float3 _ray_origin;
    Float3 _ray_direction;
    luisa::compute::UInt3 _current_position;
    UInt _scale;
    UInt _next_scale;
    UInt _octant_mask;
    Bool _hierarchical;
    Bool _valid;

    [[nodiscard]] UInt _octant() const noexcept;
    void _descend() noexcept;
    [[nodiscard]] Float3 _floor_position()
        const noexcept;
    [[nodiscard]] Float _intersect_leaf()
        noexcept;

  public:
    VolumeMajorantTraversal(
        const luisa::compute::BufferVar<
            VolumeMajorantNodeGpu> &nodes,
        const luisa::compute::Var<
            VolumeMajorantRootGpu> &root,
        Float3 ray_origin,
        Float3 ray_direction,
        Float ray_minimum,
        Float ray_maximum) noexcept;

    [[nodiscard]] VolumeMajorantLeaf
    current() const noexcept;

    // Advances to the adjacent leaf, or to the root-majorant tail used by
    // Cycles when a still-active implicit medium extends beyond its bbox.
    [[nodiscard]] Bool advance() noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeMajorantLeaf)
