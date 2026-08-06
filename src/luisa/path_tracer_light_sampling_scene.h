#pragma once

#include "path_tracer_internal.h"

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

// Complete host representation of the two Cycles light-selection policies.
// The flat CDF and spatial hierarchy share one dense emitter identity:
// emissive triangles, analytic lights, then the optional environment.
struct LightSamplingSceneUpload {
    luisa::vector<LightDistributionGpu> distribution;
    luisa::vector<LightTreeNodeGpu> tree_nodes;
    luisa::vector<LightTreeEmitterGpu> tree_emitters;
    luisa::vector<luisa::uint2> tree_emitter_mappings;
    luisa::vector<luisa::uint4> tree_triangle_lookup;
    std::uint32_t distribution_count{};
    float triangle_area_pdf{};
    float light_selection_pdf{};
    bool environment_in_distribution{};
    std::uint32_t tree_root{~std::uint32_t{0u}};
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }

    [[nodiscard]] bool tree_usable() const noexcept {
        return tree_root < tree_nodes.size() &&
               !tree_emitters.empty() &&
               tree_emitter_mappings.size() == tree_emitters.size();
    }
};

// Builds selection metadata after displacement has finalized GeometryUpload.
// This ordering is required: Cycles constructs emitter bounds and areas from
// the same final vertex support used by intersection, never the source mesh.
[[nodiscard]] LightSamplingSceneUpload
build_light_sampling_scene_upload(
    const contract::SceneSnapshot &snapshot,
    const LuisaSceneData &scene,
    std::span<const GeometryUpload> geometry_uploads,
    std::span<const LightGpu> lights,
    std::span<const EmissiveTriangleGpu> emissive_triangles,
    std::span<const float> emissive_triangle_areas,
    bool include_environment) noexcept;

}// namespace psycles::luisa_backend::detail
