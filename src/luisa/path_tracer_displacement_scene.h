#pragma once

#include "path_tracer_displacement_plan.h"
#include "path_tracer_internal.h"
#include "path_tracer_scene_geometry.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {

struct MeshDisplacementSceneBuildResult {
    std::vector<std::uint32_t> displaced_geometry_indices;
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

class MeshDisplacementSceneComponent {

public:
    [[nodiscard]] MeshDisplacementSceneBuildResult build(
        const std::shared_ptr<LuisaSceneData> &scene,
        Stream &stream,
        const contract::SceneSnapshot &snapshot,
        std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
        const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
        const std::map<contract::GeometryId, std::uint32_t> &triangle_primitive_offsets,
        const std::map<contract::GeometryId, std::uint32_t> &curve_primitive_offsets,
        std::vector<GeometryUpload> &uploads) const;
};

}// namespace psycles::luisa_backend::detail
