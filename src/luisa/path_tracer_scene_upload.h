#pragma once

#include "path_tracer_internal.h"

#include <string>

namespace psycles::luisa_backend::detail {

struct SceneTableUploadInput {
    luisa::vector<GeometryGpu> &geometries;
    luisa::vector<AttributeBindingGpu> &attribute_bindings;
    luisa::vector<AttributeRangeGpu> &attribute_ranges;
    luisa::vector<InstanceGpu> &instances;
    luisa::vector<MaterialBindingGpu> &geometry_materials;
    luisa::vector<MaterialBindingGpu> &override_materials;
    luisa::vector<LightGpu> &lights;
    luisa::vector<EmissiveTriangleGpu> &emissive_triangles;
    luisa::vector<LightDistributionGpu> &light_distribution;
};

struct SceneTableUploadResult {
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Final host/JIT scene stage. It normalizes logically empty tables to inert
// storage, establishes the unique Cycles-object lookup used by exact-source
// traversal, and uploads every table before the acceleration structure build.
class SceneTableUploadComponent {

  public:
    [[nodiscard]] SceneTableUploadResult upload(
        const std::shared_ptr<LuisaSceneData> &scene,
        Stream &stream,
        SceneTableUploadInput input) const;
};

}// namespace psycles::luisa_backend::detail
