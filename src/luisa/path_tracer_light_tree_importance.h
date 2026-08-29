#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

// Cycles' volume-segment projection from light/tree.h, exposed at the host
// tracing boundary so its finite 1e12 segment cap has a permanent device
// regression. The ray-infinity sentinel is deliberately a separate concept.
[[nodiscard]] Float3 cycles_light_tree_volume_projection_direction(
    Float3 centroid,
    Float3 point,
    Float3 direction,
    Float3 cone_axis,
    Float distance) noexcept;

// Host-stage component which traces the exact Cycles Light Tree importance
// equations into Luisa DSL. Surface and volume instances are constructed
// separately, so volume mode is a compile-time property of the generated AST.
class LightTreeImportanceComponent {
  private:
    std::shared_ptr<LuisaSceneData> _scene;
    bool _in_volume;

  public:
    LightTreeImportanceComponent(
        std::shared_ptr<LuisaSceneData> scene,
        bool in_volume) noexcept;

    [[nodiscard]] Float2 emitter(
        UInt emitter_index,
        Float3 point,
        Float3 normal_or_direction,
        Float distance,
        Bool has_transmission) const noexcept;

    [[nodiscard]] Float2 node(
        Var<LightTreeNodeGpu> node,
        Float3 point,
        Float3 normal_or_direction,
        Float distance,
        Bool has_transmission) const noexcept;
};

}// namespace psycles::luisa_backend::detail
