#pragma once

#include "path_tracer_internal.h"

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

[[nodiscard]] std::uint32_t encode_attribute_domain(
    contract::MeshAttributeDomain domain) noexcept;

[[nodiscard]] Vec3f from_luisa(luisa::float3 value) noexcept;

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
    luisa::vector<LightTreeNodeGpu> &light_tree_nodes;
    luisa::vector<LightTreeEmitterGpu> &light_tree_emitters;
    luisa::vector<luisa::uint2> &light_tree_emitter_mappings;
    luisa::vector<luisa::uint4> &light_tree_triangle_lookup;
};

struct SceneTableUploadResult {
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Final host/JIT scene stage. It normalizes logically empty tables and uploads
// every table before the acceleration structure build.
class SceneTableUploadComponent {

  public:
    [[nodiscard]] SceneTableUploadResult upload(
        const std::shared_ptr<LuisaSceneData> &scene,
        Stream &stream,
        SceneTableUploadInput input) const;
};

}// namespace psycles::luisa_backend::detail
