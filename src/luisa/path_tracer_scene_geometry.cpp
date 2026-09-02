#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <unordered_set>

namespace psycles::luisa_backend::detail {

CyclesPrimitiveInterval
CyclesPrimitiveIntervalResolver::resolve(
    std::size_t triangle_count,
    std::optional<std::uint32_t> explicit_offset) noexcept {
    constexpr auto address_space_size =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) +
        1u;
    const auto offset = explicit_offset
                            ? static_cast<std::uint64_t>(
                                  *explicit_offset)
                            : _end;
    if (explicit_offset && offset < _end) {
        return {
            .offset = std::nullopt,
            .error =
                CyclesPrimitiveIntervalError::overlap};
    }
    const auto count =
        static_cast<std::uint64_t>(triangle_count);
    if (offset >= address_space_size ||
        count > address_space_size - offset) {
        return {
            .offset = std::nullopt,
            .error =
                CyclesPrimitiveIntervalError::out_of_range};
    }
    _end = offset + count;
    return {
        .offset =
            static_cast<std::uint32_t>(offset)};
}

namespace {

[[nodiscard]] std::optional<contract::MaterialId>
resolve_surface_material(
    std::span<const contract::MaterialId> geometry_materials,
    std::span<const contract::MaterialId> material_overrides,
    std::uint32_t slot) noexcept {
    if (slot < material_overrides.size()) {
        return material_overrides[slot];
    }
    if (!geometry_materials.empty()) {
        return geometry_materials[std::min<std::size_t>(
            slot, geometry_materials.size() - 1u)];
    }
    return std::nullopt;
}

// Material reachability depends on a primitive only through its raw material
// slot. For geometry G and instance I, let S_G map primitives to slots and
// R_G,I resolve a slot through overrides and the geometry table. Then
// image(R_G,I ∘ S_G) = image(R_G,I | image(S_G)): duplicate primitive
// slots can be eliminated once per geometry before considering any instance.
// This changes the repeated work from instances * primitives to
// primitives + instances * distinct_slots without approximating reachability.
[[nodiscard]] std::vector<std::uint32_t>
collect_triangle_material_slot_image(
    const contract::TriangleMeshDesc &geometry) {
    std::vector<std::uint32_t> result;
    std::unordered_set<std::uint32_t> seen;
    seen.reserve(std::min<std::size_t>(geometry.triangles.size(), 64u));
    for (auto primitive = std::size_t{0u};
         primitive < geometry.triangles.size();
         ++primitive) {
        const auto slot =
            primitive < geometry.triangle_material_slots.size()
                ? geometry.triangle_material_slots[primitive]
                : 0u;
        if (seen.emplace(slot).second) {
            result.emplace_back(slot);
        }
    }
    std::ranges::sort(result);
    return result;
}

[[nodiscard]] std::vector<std::uint32_t>
collect_curve_material_slot_image(
    const contract::CurveGeometryDesc &geometry) {
    std::vector<std::uint32_t> result;
    std::unordered_set<std::uint32_t> seen;
    seen.reserve(std::min<std::size_t>(
        geometry.curve_first_key.size(), 64u));
    for (auto curve = std::size_t{0u};
         curve < geometry.curve_first_key.size();
         ++curve) {
        const auto first = static_cast<std::size_t>(
            geometry.curve_first_key[curve]);
        const auto end =
            curve + 1u < geometry.curve_first_key.size()
                ? static_cast<std::size_t>(
                      geometry.curve_first_key[curve + 1u])
                : geometry.keys.size();
        // Invalid and one-key curves generate no procedural primitive.
        if (first >= geometry.keys.size() ||
            end > geometry.keys.size() || end <= first ||
            end - first < 2u) {
            continue;
        }
        const auto slot =
            curve < geometry.curve_material_slots.size()
                ? geometry.curve_material_slots[curve]
                : 0u;
        if (seen.emplace(slot).second) {
            result.emplace_back(slot);
        }
    }
    std::ranges::sort(result);
    return result;
}

void include_geometry_material_slot_image(
    std::map<
        contract::GeometryId,
        std::set<contract::MaterialId>> &result,
    contract::GeometryId geometry,
    std::span<const std::uint32_t> slots,
    std::span<const contract::MaterialId> geometry_materials,
    std::span<const contract::MaterialId> material_overrides) {
    // The map is the sparse canonical representation of non-empty images.
    // Defer operator[] until a slot resolves so empty/unbound geometry cannot
    // create an observationally meaningless entry or perturb cache identity.
    for (const auto slot : slots) {
        if (const auto material = resolve_surface_material(
                geometry_materials, material_overrides, slot)) {
            result[geometry].emplace(*material);
        }
    }
}

[[nodiscard]] bool material_slot_image_intersects(
    std::span<const std::uint32_t> slots,
    std::span<const contract::MaterialId> geometry_materials,
    std::span<const contract::MaterialId> material_overrides,
    const std::set<contract::MaterialId> &materials) noexcept {
    for (const auto slot : slots) {
        const auto material = resolve_surface_material(
            geometry_materials, material_overrides, slot);
        if (material && materials.contains(*material)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Vec3f cross(Vec3f a, Vec3f b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

[[nodiscard]] float dot(Vec3f a, Vec3f b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3f divide(Vec3f value, float denominator) noexcept {
    return {
        value.x / denominator,
        value.y / denominator,
        value.z / denominator};
}

[[nodiscard]] bool material_blocks_static_transform(
    const contract::SceneSnapshot &scene,
    contract::MaterialId material,
    const std::set<contract::MaterialId> &surface_bssrdf_materials) {
    if (surface_bssrdf_materials.contains(material)) {
        return true;
    }
    const auto iter = scene.materials.find(material);
    return iter != scene.materials.end() &&
           contract::uses_true_displacement(
               iter->second.displacement_method);
}

[[nodiscard]] Vec3f transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        std::fma(
            point.x, e[0u],
            std::fma(
                point.y, e[4u],
                std::fma(point.z, e[8u], e[12u]))),
        std::fma(
            point.x, e[1u],
            std::fma(
                point.y, e[5u],
                std::fma(point.z, e[9u], e[13u]))),
        std::fma(
            point.x, e[2u],
            std::fma(
                point.y, e[6u],
                std::fma(point.z, e[10u], e[14u])))};
}

}// namespace

SceneMaterialReachability build_scene_material_reachability(
    const contract::SceneSnapshot &scene) {
    SceneMaterialReachability result;
    std::map<contract::GeometryId, std::vector<std::uint32_t>>
        triangle_slot_images;
    std::map<contract::GeometryId, std::vector<std::uint32_t>>
        curve_slot_images;
    for (const auto &[instance_id, instance] : scene.instances) {
        static_cast<void>(instance_id);
        // Scene upload gives curve geometry precedence when an invalid scene
        // reuses an id in both geometry maps. Match that rule exactly so the
        // capability image cannot diverge from runtime material resolution.
        if (const auto curve =
                scene.curve_geometries.find(instance.geometry);
            curve != scene.curve_geometries.end()) {
            auto [slots, inserted] =
                curve_slot_images.try_emplace(instance.geometry);
            if (inserted) {
                slots->second =
                    collect_curve_material_slot_image(curve->second);
            }
            include_geometry_material_slot_image(
                result.surface_by_geometry, instance.geometry,
                slots->second,
                curve->second.material_slots,
                instance.material_overrides);
        } else if (const auto mesh =
                       scene.geometries.find(instance.geometry);
                   mesh != scene.geometries.end()) {
            auto [slots, inserted] =
                triangle_slot_images.try_emplace(instance.geometry);
            if (inserted) {
                slots->second =
                    collect_triangle_material_slot_image(mesh->second);
            }
            include_geometry_material_slot_image(
                result.surface_by_geometry, instance.geometry,
                slots->second,
                mesh->second.material_slots,
                instance.material_overrides);
        }
    }
    for (const auto &[geometry, materials] :
         result.surface_by_geometry) {
        static_cast<void>(geometry);
        result.surface_materials.insert(
            materials.begin(), materials.end());
    }
    result.shader_materials = result.surface_materials;
    for (const auto &[light_id, light] : scene.lights) {
        static_cast<void>(light_id);
        if (light.shader) {
            result.shader_materials.emplace(*light.shader);
        }
    }
    if (scene.world_shader) {
        result.shader_materials.emplace(*scene.world_shader);
    }
    return result;
}

std::set<contract::MaterialId>
collect_reachable_surface_materials(
    const contract::SceneSnapshot &scene) {
    return build_scene_material_reachability(scene).surface_materials;
}

std::set<contract::MaterialId> collect_cycles_svm_shader_materials(
    const contract::SceneSnapshot &scene) {
    std::set<contract::MaterialId> result;
    for (const auto &[geometry_id, geometry] : scene.geometries) {
        static_cast<void>(geometry_id);
        result.insert(
            geometry.material_slots.begin(), geometry.material_slots.end());
    }
    for (const auto &[geometry_id, geometry] : scene.curve_geometries) {
        static_cast<void>(geometry_id);
        result.insert(
            geometry.material_slots.begin(), geometry.material_slots.end());
    }
    for (const auto &[instance_id, instance] : scene.instances) {
        static_cast<void>(instance_id);
        result.insert(instance.material_overrides.begin(),
                      instance.material_overrides.end());
    }
    for (const auto &[light_id, light] : scene.lights) {
        static_cast<void>(light_id);
        if (light.shader) {
            result.emplace(*light.shader);
        }
    }
    if (scene.world_shader) {
        result.emplace(*scene.world_shader);
    }
    return result;
}

std::vector<std::uint32_t>
collect_triangle_instances_with_surface_materials(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &materials) {
    std::vector<std::uint32_t> result;
    std::map<contract::GeometryId, std::vector<std::uint32_t>>
        triangle_slot_images;
    auto instance_index = std::uint32_t{0u};
    for (const auto &[instance_id, instance] : scene.instances) {
        static_cast<void>(instance_id);
        // Match scene upload's curve precedence for malformed snapshots that
        // reuse a geometry id in both maps. Such an instance cannot enter the
        // triangle-only Cycles local-intersection domain.
        const auto is_curve =
            scene.curve_geometries.contains(instance.geometry);
        const auto geometry = scene.geometries.find(instance.geometry);
        if (!is_curve && geometry != scene.geometries.end()) {
            auto [slots, inserted] =
                triangle_slot_images.try_emplace(instance.geometry);
            if (inserted) {
                slots->second = collect_triangle_material_slot_image(
                    geometry->second);
            }
            if (material_slot_image_intersects(
                    slots->second, geometry->second.material_slots,
                    instance.material_overrides, materials)) {
                result.emplace_back(instance_index);
            }
        }
        ++instance_index;
    }
    return result;
}

float world_triangle_area(
    const Mat4f &transform,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2) noexcept {
    p0 = transform_point(transform, p0);
    p1 = transform_point(transform, p1);
    p2 = transform_point(transform, p2);
    const Vec3f edge01{
        p1.x - p0.x,
        p1.y - p0.y,
        p1.z - p0.z};
    const Vec3f edge02{
        p2.x - p0.x,
        p2.y - p0.y,
        p2.z - p0.z};
    const Vec3f normal{
        edge01.y * edge02.z - edge01.z * edge02.y,
        edge01.z * edge02.x - edge01.x * edge02.z,
        edge01.x * edge02.y - edge01.y * edge02.x};
    return 0.5f * std::sqrt(
        normal.x * normal.x +
        normal.y * normal.y +
        normal.z * normal.z);
}

Vec3f cycles_transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept {
    return transform_point(transform, point);
}

Mat4f cycles_inverse_transform(
    const Mat4f &transform) noexcept {
    const auto &e = transform.elements;
    auto x = Vec3f{e[0u], e[1u], e[2u]};
    auto y = Vec3f{e[4u], e[5u], e[6u]};
    auto z = Vec3f{e[8u], e[9u], e[10u]};
    const auto translation = Vec3f{e[12u], e[13u], e[14u]};
    auto determinant = dot(x, cross(y, z));
    if (determinant == 0.0f) {
        x.x += 1.0e-8f;
        y.y += 1.0e-8f;
        z.z += 1.0e-8f;
        determinant = dot(x, cross(y, z));
        if (determinant == 0.0f) {
            determinant = std::numeric_limits<float>::max();
        }
    }
    const auto inverse_x = divide(cross(y, z), determinant);
    const auto inverse_y = divide(cross(z, x), determinant);
    const auto inverse_z = divide(cross(x, y), determinant);
    Mat4f result;
    result.elements = {
        inverse_x.x,
        inverse_y.x,
        inverse_z.x,
        0.0f,
        inverse_x.y,
        inverse_y.y,
        inverse_z.y,
        0.0f,
        inverse_x.z,
        inverse_y.z,
        inverse_z.z,
        0.0f,
        -dot(inverse_x, translation),
        -dot(inverse_y, translation),
        -dot(inverse_z, translation),
        1.0f};
    return result;
}

std::vector<CyclesInstanceIntersectionPlan>
build_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &surface_bssrdf_materials) {
    std::map<contract::GeometryId, std::size_t> geometry_users;
    for (const auto &[id, instance] : scene.instances) {
        static_cast<void>(id);
        if (scene.geometries.contains(instance.geometry)) {
            ++geometry_users[instance.geometry];
        }
    }

    std::vector<CyclesInstanceIntersectionPlan> result;
    result.reserve(scene.instances.size());
    for (const auto &[id, instance] : scene.instances) {
        auto transform_applied = false;
        if (const auto geometry_iter = scene.geometries.find(instance.geometry);
            geometry_iter != scene.geometries.end()) {
            const auto &geometry = geometry_iter->second;
            auto blocked = geometry.uses_adaptive_subdivision ||
                           !instance.motion.empty();
            for (const auto material : geometry.material_slots) {
                blocked |= material_blocks_static_transform(
                    scene, material, surface_bssrdf_materials);
            }
            for (const auto material : instance.material_overrides) {
                blocked |= material_blocks_static_transform(
                    scene, material, surface_bssrdf_materials);
            }
            transform_applied = geometry_users[instance.geometry] == 1u &&
                                !blocked;
        }
        result.emplace_back(CyclesInstanceIntersectionPlan{
            .instance = id,
            .transform_applied = transform_applied,
            .world_to_object = cycles_inverse_transform(instance.transform)});
    }

    return result;
}

}// namespace psycles::luisa_backend::detail
