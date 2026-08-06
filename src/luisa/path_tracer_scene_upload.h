#pragma once

#include "path_tracer_internal.h"

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

struct CyclesPrimitiveCompletionPlan;

struct PrimitiveCompletionUpload {
    luisa::vector<PrimitiveCompletionGpu> records;
    luisa::vector<luisa::uint> instances;
};

[[nodiscard]] PrimitiveCompletionUpload
make_primitive_completion_upload(
    const CyclesPrimitiveCompletionPlan &plan);

struct CyclesCompletionSourceLookup {
    luisa::vector<luisa::uint> dense_instances;
    luisa::vector<luisa::uint2> sparse_instances;
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Builds the reverse relation needed only for Cycles' closed-endpoint support
// completion. The common dense encoding is O(1); pathological object-number
// sparsity falls back to O(log S), where S excludes all ordinary instances.
[[nodiscard]] CyclesCompletionSourceLookup
make_cycles_completion_source_lookup(std::span<const InstanceGpu> instances);

struct SceneTableUploadInput {
    luisa::vector<GeometryGpu> &geometries;
    luisa::vector<AttributeBindingGpu> &attribute_bindings;
    luisa::vector<AttributeRangeGpu> &attribute_ranges;
    luisa::vector<InstanceGpu> &instances;
    luisa::vector<PrimitiveCompletionGpu> &primitive_completions;
    luisa::vector<luisa::uint> &primitive_completion_instances;
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

// Final host/JIT scene stage. It normalizes logically empty tables to inert
// storage, establishes the exact-source completion lookup, and uploads every
// table before the acceleration structure build.
class SceneTableUploadComponent {

  public:
    [[nodiscard]] SceneTableUploadResult upload(
        const std::shared_ptr<LuisaSceneData> &scene,
        Stream &stream,
        SceneTableUploadInput input) const;
};

}// namespace psycles::luisa_backend::detail
