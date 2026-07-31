#pragma once

#include "path_tracer_internal.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {

// Host description retained after material lowering. The scene planner only
// needs stable shader identity and Luisa dispatch coordinates; it never
// inspects or approximates closure values.
struct VolumeMajorantSceneMaterial {
    std::uint32_t surface_tag{};
    std::uint32_t parameter_block{};
    std::uint32_t shader{};
    bool has_volume{};
    bool heterogeneous{};
};

struct VolumeMajorantSceneRoot {
    contract::MaterialId material;
    std::uint32_t range_index{};
    std::uint32_t object{};
    std::uint32_t shader{};
    std::uint32_t surface_tag{};
    std::uint32_t parameter_block{};
    std::uint32_t instance_id{
        invalid_volume_identity};
    VolumeMajorantBounds bounds;
    luisa::float4x4 object_to_world{};
    float volume_scale{1.0f};
    bool heterogeneous{};
};

struct VolumeMajorantScenePlan {
    std::vector<VolumeMajorantSceneRoot> roots;
    std::vector<VolumeMajorantRootRangeGpu> ranges;
    std::uint32_t world_range{
        invalid_volume_identity};
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

struct VolumeMajorantSceneFlattened {
    std::vector<VolumeMajorantNodeGpu> nodes;
    std::vector<VolumeMajorantRootGpu> roots;
    std::vector<VolumeMajorantRootRangeGpu> ranges;
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

struct VolumeMajorantSceneBuildResult {
    std::uint32_t root_count{};
    std::uint32_t node_count{};
    std::uint32_t range_count{};
    std::string diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return diagnostic.empty();
    }
};

// Owns the scene-level one-object/one-shader mapping, exact Luisa extrema
// dispatch, hierarchy assembly, and GPU upload. This is deliberately
// separate from path_tracer_scene.cpp and from the later multi-root transport
// reducer.
class VolumeMajorantSceneComponent {

  public:
    [[nodiscard]] VolumeMajorantScenePlan plan(
        const SceneSnapshot &snapshot,
        const std::map<
            contract::MaterialId,
            VolumeMajorantSceneMaterial>
            &materials) const;

    [[nodiscard]] VolumeMajorantSceneFlattened
    flatten(
        const VolumeMajorantScenePlan &plan,
        std::span<
            const VolumeMajorantHierarchy>
            hierarchies) const;

    [[nodiscard]] VolumeMajorantSceneBuildResult
    build(
        const std::shared_ptr<LuisaSceneData> &scene,
        Stream &stream,
        const VolumeMajorantScenePlan &plan) const;
};

}// namespace psycles::luisa_backend::detail
