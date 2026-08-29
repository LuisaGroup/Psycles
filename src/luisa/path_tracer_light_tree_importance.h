#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

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
