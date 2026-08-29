#pragma once

#include "path_tracer_internal.h"

#include <psycles/sampling/light_tree.h>

#include <span>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {

struct LightTreeSceneUpload {
    luisa::vector<LightTreeNodeGpu> nodes;
    luisa::vector<LightTreeEmitterGpu> emitters;
    // Flat-distribution emitter -> top emitter/leaf.
    luisa::vector<luisa::uint2> emitter_mappings;
    // Emissive-triangle emitter -> mesh-local emitter/leaf.
    luisa::vector<luisa::uint2> triangle_emitter_mappings;
    // Concatenated per-proxy local-emitter -> flat triangle emitter maps.
    luisa::vector<luisa::uint> mesh_triangles;
    std::uint32_t root{sampling::invalid_light_tree_index};
    std::string diagnostic;

    [[nodiscard]] bool usable() const noexcept {
        return diagnostic.empty() &&
               root < nodes.size() &&
               !emitters.empty();
    }
};

struct LightTreeSubtreeInput {
    // Constructed once and shared by every semantic mesh instance. Local
    // triangle ids are the tree's dense emitter identity, retained after its
    // private spatial reordering. Keeping the actual tree here also makes the
    // top-level proxy consume exactly the same root measure that is uploaded.
    sampling::CyclesLightTree tree;
    // One representative flat emissive-triangle id for every local emitter.
    // Cycles shares a subtree only between instances of the same semantic
    // mesh, so any representative contains the same local primitive and
    // emission-side contract. Empty retains the generic packed-measure path
    // for renderer-neutral synthetic trees.
    std::vector<std::uint32_t> representative_triangles;
};

struct LightTreeTopEmitterInput {
    sampling::LightTreeEmitter emitter;
    LightTreeEmitterKind kind{LightTreeEmitterKind::direct};
    // Direct: flat distribution id. Mesh: instance index.
    std::uint32_t payload{};
    // Exact leaf source is explicit instead of inferred from distribution
    // ranges. source_index is a LightGpu index for analytic lights; other
    // direct sources do not consume it.
    LightTreeEmitterSource source{LightTreeEmitterSource::measure};
    std::uint32_t source_index{};
    // Mesh-only subtree and one actual triangle id per local emitter id.
    std::uint32_t subtree{sampling::invalid_light_tree_index};
    std::vector<std::uint32_t> triangle_emitters;
};

struct LightTreeHierarchyInput {
    std::vector<LightTreeSubtreeInput> subtrees;
    // Cycles order before spatial construction: local analytic lights, mesh
    // proxies, then distant/background emitters. Emitter ids are reassigned to
    // this dense order by the uploader.
    std::vector<LightTreeTopEmitterInput> top_emitters;
    std::uint32_t distribution_emitter_count{};
    std::uint32_t triangle_emitter_count{};
};

// Converts the renderer-neutral hierarchy into the compact device ABI. The
// stable emitter id remains the only identity visible to NEE and forward MIS;
// spatial construction order is private to the tree.
[[nodiscard]] LightTreeSceneUpload
make_light_tree_scene_upload(
    std::span<const sampling::LightTreeEmitter> emitters) noexcept;

// Flattens the two-level Cycles light-tree quotient. Unique mesh subtrees are
// stored once; every mesh instance remains a distinct top-level emitter whose
// mapping slice resolves a local triangle to the actual sampled instance.
[[nodiscard]] LightTreeSceneUpload
make_light_tree_hierarchy_scene_upload(
    const LightTreeHierarchyInput &input) noexcept;

[[nodiscard]] sampling::LightTreeEmitter
make_triangle_light_tree_emitter(
    std::uint32_t emitter_id,
    const Mat4f &object_to_world,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2,
    Vec3f emission_estimate,
    contract::EmissionSampling emission_sampling,
    bool transform_applied = false) noexcept;

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
