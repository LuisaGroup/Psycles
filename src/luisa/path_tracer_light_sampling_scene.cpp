#include "path_tracer_light_sampling_scene.h"

#include "path_tracer_light_tree_scene.h"

#include <psycles/compiler/surface_program.h>
#include <psycles/sampling/light_distribution.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Vec3f from_luisa(luisa::float3 value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] std::optional<contract::MaterialId>
triangle_material(
    const contract::TriangleMeshDesc &geometry,
    const contract::InstanceDesc &instance,
    std::size_t primitive_index) noexcept {
    const auto slot =
        primitive_index < geometry.triangle_material_slots.size()
            ? geometry.triangle_material_slots[primitive_index]
            : 0u;
    if (slot < instance.material_overrides.size()) {
        return instance.material_overrides[slot];
    }
    if (geometry.material_slots.empty()) {
        return std::nullopt;
    }
    return geometry.material_slots[std::min<std::size_t>(
        slot, geometry.material_slots.size() - 1u)];
}

[[nodiscard]] Vec3f emission_estimate(
    const LuisaSceneData &scene,
    contract::MaterialId material) noexcept {
    const auto *compiled = scene.materials.find(material);
    return compiled == nullptr
               ? Vec3f{1.0f, 1.0f, 1.0f}
               : compiler::estimate_surface_emission(
                     *compiled->surface_program(), compiled->parameters());
}

[[nodiscard]] Vec3f environment_emission_estimate(
    const contract::SceneSnapshot &snapshot,
    const LuisaSceneData &scene) noexcept {
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
    for (const auto &[material_id, material] :
         scene.materials.materials()) {
        static_cast<void>(material);
        const auto binding =
            scene.material_bindings.find(material_id);
        if (binding != scene.material_bindings.end() &&
            binding->second.emission_sampling !=
                contract::EmissionSampling::none &&
            (binding->second.flags & material_flag_may_emit) != 0u) {
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

        std::vector<const contract::InstanceDesc *> source_instances;
        source_instances.reserve(snapshot.instances.size());
        for (const auto &[id, instance] : snapshot.instances) {
            static_cast<void>(id);
            source_instances.emplace_back(&instance);
        }

        std::vector<sampling::LightTreeEmitter> emitters;
        emitters.reserve(
            emissive_triangles.size() + lights.size() +
            (include_environment ? 1u : 0u));
        for (std::size_t emitter_id = 0u;
             emitter_id < emissive_triangles.size();
             ++emitter_id) {
            const auto &emitter = emissive_triangles[emitter_id];
            if (emitter.instance_index >= source_instances.size() ||
                emitter.geometry_index >= geometry_uploads.size()) {
                result.diagnostic =
                    "emissive triangle references an unavailable instance or geometry";
                return result;
            }
            const auto &instance = *source_instances[emitter.instance_index];
            const auto geometry_iter =
                snapshot.geometries.find(instance.geometry);
            if (geometry_iter == snapshot.geometries.end()) {
                result.diagnostic =
                    "emissive triangle references non-triangle source geometry";
                return result;
            }
            const auto &geometry = geometry_iter->second;
            if (emitter.primitive_index >= geometry.triangles.size()) {
                result.diagnostic =
                    "emissive triangle primitive is outside source geometry";
                return result;
            }
            const auto material = triangle_material(
                geometry, instance, emitter.primitive_index);
            if (!material) {
                result.diagnostic =
                    "emissive triangle has no effective material";
                return result;
            }
            const auto triangle = geometry.triangles[emitter.primitive_index];
            const auto &upload = geometry_uploads[emitter.geometry_index];
            if (triangle[0u] >= upload.positions.size() ||
                triangle[1u] >= upload.positions.size() ||
                triangle[2u] >= upload.positions.size()) {
                result.diagnostic =
                    "emissive triangle is outside final displaced support";
                return result;
            }
            emitters.emplace_back(make_triangle_light_tree_emitter(
                static_cast<std::uint32_t>(emitter_id),
                instance.transform,
                from_luisa(upload.positions[triangle[0u]]),
                from_luisa(upload.positions[triangle[1u]]),
                from_luisa(upload.positions[triangle[2u]]),
                emission_estimate(scene, *material),
                static_cast<contract::EmissionSampling>(
                    emitter.emission_sampling)));
        }

        for (std::size_t light_index = 0u;
             light_index < lights.size();
             ++light_index) {
            emitters.emplace_back(make_analytic_light_tree_emitter(
                static_cast<std::uint32_t>(emitters.size()),
                lights[light_index],
                analytic_light_emission_estimates[light_index]));
        }
        if (include_environment) {
            emitters.emplace_back(make_environment_light_tree_emitter(
                static_cast<std::uint32_t>(emitters.size()),
                environment_emission_estimate(snapshot, scene)));
        }

        const auto tree = make_light_tree_scene_upload(emitters);
        if (tree.usable()) {
            result.tree_nodes = tree.nodes;
            result.tree_emitters = tree.emitters;
            result.tree_emitter_mappings = tree.emitter_mappings;
            result.tree_root = tree.root;
            result.tree_triangle_lookup =
                make_light_tree_triangle_lookup(emissive_triangles);
        } else if (!emitters.empty() && !tree.diagnostic.empty()) {
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
