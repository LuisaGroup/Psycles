#pragma once

#include "path_tracer_internal.h"

#include <set>

namespace psycles::luisa_backend::detail {

struct AmbientOcclusionScenePlan {
    bool enabled{};
    bool local_requested{};
    std::uint32_t triangle_instance_count{};

    [[nodiscard]] bool local_enabled() const noexcept {
        return local_requested && triangle_instance_count != 0u;
    }
};

class AmbientOcclusionSceneComponent final {

private:
    AmbientOcclusionScenePlan _plan;

public:
    AmbientOcclusionSceneComponent(
        const contract::SceneSnapshot &snapshot,
        const compiler::MaterialLibrary &materials,
        const std::set<contract::MaterialId> &
            reachable_materials) noexcept;

    void initialize(
        const std::shared_ptr<LuisaSceneData> &scene) const;

    void upload_distance(
        Stream &stream,
        const std::shared_ptr<LuisaSceneData> &scene,
        float distance) const;

    void append_triangle_instance(
        const std::shared_ptr<LuisaSceneData> &scene,
        std::uint32_t primary_instance,
        std::uint32_t geometry_index,
        const Mat4f &transform) const;
};

}// namespace psycles::luisa_backend::detail
