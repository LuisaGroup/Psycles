#include "path_tracer_light_sampling_scene.h"

#include "path_tracer_light_tree_scene.h"
#include "path_tracer_mesh_light_scene.h"
#include "path_tracer_cycles_svm_scene.h"

#include <psycles/compiler/surface_program.h>
#include <psycles/sampling/light_distribution.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Vec3f emission_estimate(
    const LuisaSceneData &scene,
    contract::MaterialId material) {
    if (scene.native_cycles_svm_surface) {
        return cycles_svm_material_metadata(scene, material).emission_estimate;
    }
    const auto *compiled = scene.materials.find(material);
    return compiled == nullptr
               ? Vec3f{1.0f, 1.0f, 1.0f}
               : compiler::estimate_surface_emission(
                     *compiled->surface_program(), compiled->parameters());
}

[[nodiscard]] Vec3f environment_emission_estimate(
    const contract::SceneSnapshot &snapshot,
    const LuisaSceneData &scene) {
    auto estimate = Vec3f{1.0f, 1.0f, 1.0f};
    if (snapshot.world_shader) {
        estimate = emission_estimate(scene, *snapshot.world_shader);
    }
    if (!snapshot.environment || snapshot.environment->pixels.empty()) {
        return estimate;
    }

    // EnvironmentDesc retains the original linear texture, not a baked world
    // shader. Its mean therefore refines Cycles' conservative closure-output
    // estimate without replacing any authored node evaluation.
    Vec3f average{};
    std::size_t finite_count = 0u;
    for (const auto pixel : snapshot.environment->pixels) {
        if (std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
            std::isfinite(pixel.z)) {
            average.x += std::abs(pixel.x);
            average.y += std::abs(pixel.y);
            average.z += std::abs(pixel.z);
            ++finite_count;
        }
    }
    if (finite_count == 0u) {
        return estimate;
    }
    const auto inverse = 1.0f / static_cast<float>(finite_count);
    return {estimate.x * average.x * inverse,
            estimate.y * average.y * inverse,
            estimate.z * average.z * inverse};
}

void append_flat_distribution(
    LightSamplingSceneUpload &result,
    std::span<const float> emissive_triangle_areas,
    std::uint32_t light_count,
    bool include_environment) {
    const auto distribution = sampling::build_cycles_light_distribution(
        emissive_triangle_areas, light_count, include_environment);
    result.distribution.reserve(distribution.entries.size());
    for (std::size_t emitter_id = 0u;
         emitter_id < distribution.entries.size();
         ++emitter_id) {
        const auto &entry = distribution.entries[emitter_id];
        result.distribution.emplace_back(LightDistributionGpu{
            .cumulative = entry.cumulative,
            .selection_pdf = entry.selection_pdf,
            .kind = static_cast<std::uint32_t>(entry.kind),
            .index = entry.index,
            .emitter_id =
                emitter_id < distribution.emitter_count
                    ? static_cast<std::uint32_t>(emitter_id)
                    : ~std::uint32_t{0u}});
    }
    result.distribution_count =
        distribution.usable() ? distribution.emitter_count : 0u;
    result.triangle_area_pdf = distribution.triangle_area_pdf;
    result.light_selection_pdf = distribution.light_selection_pdf;
    result.environment_in_distribution =
        include_environment && result.distribution_count > 0u;
}

}// namespace

std::set<contract::MaterialId>
collect_emission_sampling_materials(
    const LuisaSceneData &scene) {
    std::set<contract::MaterialId> result;
    for (const auto &[material_id, binding] : scene.material_bindings) {
        if (binding.emission_sampling !=
                contract::EmissionSampling::none &&
            (binding.flags & material_flag_may_emit) != 0u) {
            result.emplace(material_id);
        }
    }
    return result;
}

LightSamplingSceneUpload build_light_sampling_scene_upload(
    const contract::SceneSnapshot &snapshot,
    const LuisaSceneData &scene,
    std::span<const GeometryUpload> geometry_uploads,
    std::span<const LightGpu> lights,
    std::span<const Vec3f> analytic_light_emission_estimates,
    std::span<const EmissiveTriangleGpu> emissive_triangles,
    std::span<const float> emissive_triangle_areas,
    bool include_environment) noexcept {
    LightSamplingSceneUpload result;
    try {
        if (emissive_triangles.size() != emissive_triangle_areas.size()) {
            result.diagnostic =
                "emissive triangle identities and areas have different sizes";
            return result;
        }
        if (lights.size() != analytic_light_emission_estimates.size()) {
            result.diagnostic =
                "analytic lights and emission estimates have different sizes";
            return result;
        }
        if (emissive_triangles.size() + lights.size() +
                (include_environment ? 1u : 0u) >=
            static_cast<std::size_t>(
                sampling::invalid_light_tree_index)) {
            result.diagnostic =
                "light population exceeds the 32-bit emitter ABI";
            return result;
        }
        append_flat_distribution(
            result,
            emissive_triangle_areas,
            static_cast<std::uint32_t>(lights.size()),
            include_environment);

        auto meshes = MeshLightTreeSceneComponent{}.build(
            snapshot, scene, geometry_uploads, emissive_triangles);
        if (!meshes.ok()) {
            result.diagnostic = meshes.diagnostic;
            return result;
        }
        LightTreeHierarchyInput hierarchy;
        hierarchy.subtrees = std::move(meshes.subtrees);
        hierarchy.distribution_emitter_count = result.distribution_count;
        hierarchy.triangle_emitter_count =
            static_cast<std::uint32_t>(emissive_triangles.size());
        hierarchy.top_emitters.reserve(
            lights.size() + meshes.mesh_emitters.size() +
            (include_environment ? 1u : 0u));
        std::vector<LightTreeTopEmitterInput> distant;
        for (std::size_t light_index = 0u;
             light_index < lights.size();
             ++light_index) {
            auto emitter = make_analytic_light_tree_emitter(
                0u, lights[light_index],
                analytic_light_emission_estimates[light_index]);
            LightTreeTopEmitterInput direct{
                .emitter = emitter,
                .kind = LightTreeEmitterKind::direct,
                .payload = static_cast<std::uint32_t>(
                    emissive_triangles.size() + light_index),
                .source = LightTreeEmitterSource::analytic_light,
                .source_index = static_cast<std::uint32_t>(light_index)};
            if (emitter.distant) {
                distant.emplace_back(std::move(direct));
            } else {
                hierarchy.top_emitters.emplace_back(std::move(direct));
            }
        }
        hierarchy.top_emitters.insert(
            hierarchy.top_emitters.end(),
            std::make_move_iterator(meshes.mesh_emitters.begin()),
            std::make_move_iterator(meshes.mesh_emitters.end()));
        hierarchy.top_emitters.insert(
            hierarchy.top_emitters.end(),
            std::make_move_iterator(distant.begin()),
            std::make_move_iterator(distant.end()));
        if (include_environment) {
            hierarchy.top_emitters.emplace_back(LightTreeTopEmitterInput{
                .emitter = make_environment_light_tree_emitter(
                    0u, environment_emission_estimate(snapshot, scene)),
                .kind = LightTreeEmitterKind::direct,
                .payload = static_cast<std::uint32_t>(
                    emissive_triangles.size() + lights.size()),
                .source = LightTreeEmitterSource::environment});
        }

        const auto tree = make_light_tree_hierarchy_scene_upload(hierarchy);
        if (tree.usable()) {
            result.tree_nodes = tree.nodes;
            result.tree_emitters = tree.emitters;
            result.tree_emitter_mappings = tree.emitter_mappings;
            result.tree_triangle_emitter_mappings =
                tree.triangle_emitter_mappings;
            result.tree_mesh_triangles = tree.mesh_triangles;
            result.tree_root = tree.root;
            result.tree_triangle_lookup =
                make_light_tree_triangle_lookup(emissive_triangles);
        } else if (!hierarchy.top_emitters.empty() &&
                   !tree.diagnostic.empty()) {
            result.diagnostic = tree.diagnostic;
        }
    } catch (const std::exception &error) {
        result.diagnostic = error.what();
    } catch (...) {
        result.diagnostic = "unknown light-sampling scene construction error";
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
