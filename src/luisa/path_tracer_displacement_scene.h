#pragma once

#include "path_tracer_displacement_plan.h"
#include "path_tracer_internal.h"

#include <map>
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
        const std::map<contract::GeometryId, std::uint32_t> &
            geometry_indices,
        std::vector<GeometryUpload> &uploads,
        luisa::vector<AttributeBindingGpu> &attribute_bindings,
        luisa::vector<AttributeRangeGpu> &attribute_ranges) const;
};

}// namespace psycles::luisa_backend::detail
