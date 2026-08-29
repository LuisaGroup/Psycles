#pragma once

#include "path_tracer_light_tree_scene.h"

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

struct MeshLightTreeScene {
    std::vector<LightTreeSubtreeInput> subtrees;
    std::vector<LightTreeTopEmitterInput> mesh_emitters;
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Constructs Cycles' quotient of emissive instances by identical mesh-local
// light-tree semantics. Material overrides participate in the quotient key;
// no instance shares a subtree unless every local emitter measure is equal.
class MeshLightTreeSceneComponent {

  public:
    [[nodiscard]] MeshLightTreeScene build(
        const contract::SceneSnapshot &snapshot,
        const LuisaSceneData &scene,
        std::span<const GeometryUpload> geometry_uploads,
        std::span<const EmissiveTriangleGpu> triangles) const noexcept;
};

}// namespace psycles::luisa_backend::detail
