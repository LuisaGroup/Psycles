#include "path_tracer_mesh_light_scene.h"

#include "path_tracer_scene_geometry.h"

#include <psycles/compiler/surface_program.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <compare>
#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Vec3f from_luisa(luisa::float3 value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3f add(Vec3f a, Vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] float length_squared(Vec3f value) noexcept {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

[[nodiscard]] Vec3f transform_direction(
    const Mat4f &transform,
    Vec3f direction) noexcept {
    return add(
        add(multiply(matrix_axis(transform, 0u), direction.x),
            multiply(matrix_axis(transform, 1u), direction.y)),
        multiply(matrix_axis(transform, 2u), direction.z));
}

[[nodiscard]] bool uniform_scale_squared(
    const Mat4f &transform,
    float &scale_squared) noexcept {
    const auto &e = transform.elements;
    const std::array rows{
        Vec3f{e[0u], e[4u], e[8u]},
        Vec3f{e[1u], e[5u], e[9u]},
        Vec3f{e[2u], e[6u], e[10u]}};
    const std::array columns{
        matrix_axis(transform, 0u),
        matrix_axis(transform, 1u),
        matrix_axis(transform, 2u)};
    const auto reference = length_squared(rows[0u]);
    constexpr auto epsilon = 1.0e-6f;
    const auto equal = [reference](Vec3f value) noexcept {
        return std::abs(length_squared(value) - reference) < epsilon;
    };
    if (reference > 0.0f &&
        std::all_of(rows.begin(), rows.end(), equal) &&
        std::all_of(columns.begin(), columns.end(), equal)) {
        scale_squared = reference;
        return true;
    }
    return false;
}

[[nodiscard]] sampling::LightTreeBounds transform_bounds(
    const sampling::LightTreeBounds &bounds,
    const Mat4f &transform) noexcept {
    if (bounds.empty) {
        return {};
    }
    sampling::LightTreeBounds result;
    for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
        const auto local = Vec3f{
            (corner & 1u) != 0u ? bounds.maximum.x : bounds.minimum.x,
            (corner & 2u) != 0u ? bounds.maximum.y : bounds.minimum.y,
            (corner & 4u) != 0u ? bounds.maximum.z : bounds.minimum.z};
        const auto point = cycles_transform_point(transform, local);
        if (result.empty) {
            result.minimum = point;
            result.maximum = point;
            result.empty = false;
        } else {
            result.minimum = {
                std::min(result.minimum.x, point.x),
                std::min(result.minimum.y, point.y),
                std::min(result.minimum.z, point.z)};
            result.maximum = {
                std::max(result.maximum.x, point.x),
                std::max(result.maximum.y, point.y),
                std::max(result.maximum.z, point.z)};
        }
    }
    return result;
}

[[nodiscard]] sampling::LightTreeMeasure transform_uniform_measure(
    sampling::LightTreeMeasure measure,
    const Mat4f &transform,
    float scale_squared) noexcept {
    measure.bounds = transform_bounds(measure.bounds, transform);
    if (!measure.orientation.empty) {
        measure.orientation.axis = multiply(
            transform_direction(transform, measure.orientation.axis),
            1.0f / std::sqrt(scale_squared));
    }
    measure.energy *= scale_squared;
    return measure;
}

[[nodiscard]] std::optional<contract::MaterialId> triangle_material(
    const contract::TriangleMeshDesc &geometry,
    const contract::InstanceDesc &instance,
    std::size_t primitive) noexcept {
    const auto slot = primitive < geometry.triangle_material_slots.size()
                          ? geometry.triangle_material_slots[primitive]
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

struct TriangleSemantic {
    std::uint32_t primitive{};
    std::uint32_t emission_sampling{};
    std::array<std::uint32_t, 3u> emission{};

    auto operator<=>(const TriangleSemantic &) const noexcept = default;
};

struct MeshSemantic {
    std::uint32_t geometry{};
    bool transform_applied{};
    std::vector<TriangleSemantic> triangles;

    auto operator<=>(const MeshSemantic &) const noexcept = default;
};

struct InstanceTriangle {
    std::uint32_t emitter{};
    const EmissiveTriangleGpu *triangle{};
    Vec3f emission{};
};

[[nodiscard]] TriangleSemantic semantic(
    const InstanceTriangle &triangle) noexcept {
    return {
        .primitive = triangle.triangle->primitive_index,
        .emission_sampling = triangle.triangle->emission_sampling,
        .emission = {
            std::bit_cast<std::uint32_t>(triangle.emission.x),
            std::bit_cast<std::uint32_t>(triangle.emission.y),
            std::bit_cast<std::uint32_t>(triangle.emission.z)}};
}

[[nodiscard]] Vec3f object_centroid(
    const GeometryUpload &geometry,
    const Mat4f &transform) noexcept {
    sampling::LightTreeBounds bounds;
    for (const auto position : geometry.positions) {
        const auto point = cycles_transform_point(transform, from_luisa(position));
        if (bounds.empty) {
            bounds.minimum = point;
            bounds.maximum = point;
            bounds.empty = false;
        } else {
            bounds.minimum = {
                std::min(bounds.minimum.x, point.x),
                std::min(bounds.minimum.y, point.y),
                std::min(bounds.minimum.z, point.z)};
            bounds.maximum = {
                std::max(bounds.maximum.x, point.x),
                std::max(bounds.maximum.y, point.y),
                std::max(bounds.maximum.z, point.z)};
        }
    }
    return bounds.empty
               ? matrix_translation(transform)
               : multiply(add(bounds.minimum, bounds.maximum), 0.5f);
}

}// namespace

MeshLightTreeScene MeshLightTreeSceneComponent::build(
    const contract::SceneSnapshot &snapshot,
    const LuisaSceneData &scene,
    std::span<const GeometryUpload> geometry_uploads,
    std::span<const EmissiveTriangleGpu> triangles) const noexcept {
    MeshLightTreeScene result;
    try {
        std::vector<const contract::InstanceDesc *> instances;
        instances.reserve(snapshot.instances.size());
        for (const auto &[id, instance] : snapshot.instances) {
            static_cast<void>(id);
            instances.emplace_back(&instance);
        }
        std::vector<std::vector<std::uint32_t>> instance_triangles(
            instances.size());
        for (std::size_t emitter = 0u; emitter < triangles.size(); ++emitter) {
            if (triangles[emitter].instance_index >= instances.size()) {
                result.diagnostic =
                    "emissive triangle references an unavailable instance";
                return result;
            }
            instance_triangles[triangles[emitter].instance_index].emplace_back(
                static_cast<std::uint32_t>(emitter));
        }

        std::map<MeshSemantic, std::uint32_t> unique_subtrees;
        for (std::size_t instance_index = 0u;
             instance_index < instances.size();
             ++instance_index) {
            auto &emitter_ids = instance_triangles[instance_index];
            if (emitter_ids.empty()) {
                continue;
            }
            const auto &instance = *instances[instance_index];
            const auto geometry_iter = snapshot.geometries.find(instance.geometry);
            if (geometry_iter == snapshot.geometries.end()) {
                result.diagnostic =
                    "emissive instance references non-triangle source geometry";
                return result;
            }
            const auto &geometry = geometry_iter->second;
            const auto geometry_index = triangles[emitter_ids.front()].geometry_index;
            if (geometry_index >= geometry_uploads.size()) {
                result.diagnostic =
                    "emissive instance references unavailable final geometry";
                return result;
            }
            const auto &upload = geometry_uploads[geometry_index];
            const auto transform_applied =
                !upload.cycles_intersection_positions.empty();
            if (transform_applied &&
                upload.cycles_intersection_positions.size() !=
                    upload.positions.size()) {
                result.diagnostic =
                    "transform-applied light support has inconsistent cardinality";
                return result;
            }
            const auto &light_positions =
                transform_applied
                    ? upload.cycles_intersection_positions
                    : upload.positions;
            std::sort(
                emitter_ids.begin(), emitter_ids.end(),
                [&triangles](std::uint32_t a, std::uint32_t b) noexcept {
                    return triangles[a].primitive_index <
                           triangles[b].primitive_index;
                });

            std::vector<InstanceTriangle> local_triangles;
            local_triangles.reserve(emitter_ids.size());
            for (const auto emitter_id : emitter_ids) {
                const auto &triangle = triangles[emitter_id];
                if (triangle.instance_index != instance_index ||
                    triangle.geometry_index != geometry_index ||
                    triangle.primitive_index >= geometry.triangles.size()) {
                    result.diagnostic =
                        "inconsistent emissive mesh-instance triangle identity";
                    return result;
                }
                const auto material = triangle_material(
                    geometry, instance, triangle.primitive_index);
                if (!material) {
                    result.diagnostic =
                        "emissive triangle has no effective material";
                    return result;
                }
                local_triangles.emplace_back(InstanceTriangle{
                    .emitter = emitter_id,
                    .triangle = &triangle,
                    .emission = emission_estimate(scene, *material)});
            }

            MeshSemantic key{
                .geometry = geometry_index,
                .transform_applied = transform_applied};
            key.triangles.reserve(local_triangles.size());
            std::vector<sampling::LightTreeEmitter> local_emitters;
            local_emitters.reserve(local_triangles.size());
            for (std::size_t local = 0u;
                 local < local_triangles.size();
                 ++local) {
                const auto &entry = local_triangles[local];
                const auto primitive = entry.triangle->primitive_index;
                const auto indices = geometry.triangles[primitive];
                if (indices[0u] >= light_positions.size() ||
                    indices[1u] >= light_positions.size() ||
                    indices[2u] >= light_positions.size()) {
                    result.diagnostic =
                        "emissive triangle is outside final displaced support";
                    return result;
                }
                key.triangles.emplace_back(semantic(entry));
                local_emitters.emplace_back(make_triangle_light_tree_emitter(
                    static_cast<std::uint32_t>(local),
                    transform_applied ? instance.transform : Mat4f{},
                    from_luisa(light_positions[indices[0u]]),
                    from_luisa(light_positions[indices[1u]]),
                    from_luisa(light_positions[indices[2u]]),
                    entry.emission,
                    static_cast<contract::EmissionSampling>(
                        entry.triangle->emission_sampling),
                    transform_applied));
            }

            auto [subtree_iter, inserted] = unique_subtrees.try_emplace(
                std::move(key),
                static_cast<std::uint32_t>(result.subtrees.size()));
            if (inserted) {
                LightTreeSubtreeInput subtree{
                    .tree = sampling::build_cycles_light_subtree(
                        local_emitters)};
                if (!subtree.tree.valid()) {
                    throw std::invalid_argument(
                        "failed to construct mesh-local light tree");
                }
                subtree.representative_triangles.reserve(
                    local_triangles.size());
                for (const auto &triangle : local_triangles) {
                    subtree.representative_triangles.emplace_back(
                        triangle.emitter);
                }
                result.subtrees.emplace_back(std::move(subtree));
            }
            const auto subtree = subtree_iter->second;
            const auto &shared_tree = result.subtrees[subtree].tree;
            auto proxy_measure = shared_tree.nodes[shared_tree.root].measure;
            // Cycles has already applied the object transform to a
            // transform-applied subtree. Its root is the world-space proxy;
            // applying the transform again would square it.
            if (!transform_applied) {
                float scale_squared = 0.0f;
                if (uniform_scale_squared(instance.transform, scale_squared)) {
                    proxy_measure = transform_uniform_measure(
                        proxy_measure, instance.transform, scale_squared);
                } else {
                    proxy_measure = {};
                    for (std::size_t local = 0u;
                         local < local_triangles.size();
                         ++local) {
                        const auto &entry = local_triangles[local];
                        const auto indices =
                            geometry.triangles[entry.triangle->primitive_index];
                        const auto world = make_triangle_light_tree_emitter(
                            static_cast<std::uint32_t>(local),
                            instance.transform,
                            from_luisa(upload.positions[indices[0u]]),
                            from_luisa(upload.positions[indices[1u]]),
                            from_luisa(upload.positions[indices[2u]]),
                            entry.emission,
                            static_cast<contract::EmissionSampling>(
                                entry.triangle->emission_sampling));
                        proxy_measure = sampling::merge_light_tree_measures(
                            proxy_measure, world.measure);
                    }
                }
            }

            LightTreeTopEmitterInput proxy;
            proxy.emitter = {
                .measure = proxy_measure,
                .centroid = object_centroid(upload, instance.transform),
                .distant = false};
            proxy.kind = LightTreeEmitterKind::mesh_instance;
            proxy.payload = static_cast<std::uint32_t>(instance_index);
            proxy.subtree = subtree;
            proxy.triangle_emitters.reserve(local_triangles.size());
            for (const auto &triangle : local_triangles) {
                proxy.triangle_emitters.emplace_back(triangle.emitter);
            }
            result.mesh_emitters.emplace_back(std::move(proxy));
        }
    } catch (const std::exception &error) {
        result = {};
        result.diagnostic = error.what();
    } catch (...) {
        result = {};
        result.diagnostic = "unknown mesh light-tree construction error";
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
