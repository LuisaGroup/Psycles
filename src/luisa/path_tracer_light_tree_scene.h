#pragma once

#include "path_tracer_internal.h"

#include <psycles/sampling/light_tree.h>

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

struct LightTreeSceneUpload {
    luisa::vector<LightTreeNodeGpu> nodes;
    luisa::vector<LightTreeEmitterGpu> emitters;
    luisa::vector<luisa::uint2> emitter_mappings;
    std::uint32_t root{sampling::invalid_light_tree_index};
    std::string diagnostic;

    [[nodiscard]] bool usable() const noexcept {
        return diagnostic.empty() &&
               root < nodes.size() &&
               !emitters.empty() &&
               emitter_mappings.size() == emitters.size();
    }
};

// Converts the renderer-neutral hierarchy into the compact device ABI. The
// stable emitter id remains the only identity visible to NEE and forward MIS;
// spatial construction order is private to the tree.
[[nodiscard]] LightTreeSceneUpload
make_light_tree_scene_upload(
    std::span<const sampling::LightTreeEmitter> emitters) noexcept;

[[nodiscard]] sampling::LightTreeEmitter
make_triangle_light_tree_emitter(
    std::uint32_t emitter_id,
    const Mat4f &object_to_world,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2,
    Vec3f emission_estimate,
    contract::EmissionSampling emission_sampling) noexcept;

[[nodiscard]] sampling::LightTreeEmitter
make_analytic_light_tree_emitter(
    std::uint32_t emitter_id,
    const LightGpu &light,
    Vec3f shader_emission_estimate) noexcept;

[[nodiscard]] sampling::LightTreeEmitter
make_environment_light_tree_emitter(
    std::uint32_t emitter_id,
    Vec3f emission_estimate) noexcept;

// Sorted (Cycles object, Cycles primitive, stable emitter id) records used
// only by forward emissive hits. Selection never scans this table.
[[nodiscard]] luisa::vector<luisa::uint4>
make_light_tree_triangle_lookup(
    std::span<const EmissiveTriangleGpu> triangles);

}// namespace psycles::luisa_backend::detail
