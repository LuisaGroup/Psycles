#include "path_tracer_scene_geometry.h"

#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>

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

[[nodiscard]] bool same_bits(float a, float b) noexcept {
    return std::bit_cast<std::uint32_t>(a) ==
           std::bit_cast<std::uint32_t>(b);
}

[[nodiscard]] bool same_transform_bits(
    const Mat4f &a,
    const Mat4f &b) noexcept {
    for (std::size_t i = 0u; i < a.elements.size(); ++i) {
        if (!same_bits(a.elements[i], b.elements[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool same_support_bits(
    const contract::TriangleMeshDesc &a,
    const contract::TriangleMeshDesc &b) noexcept {
    if (a.positions.size() != b.positions.size() ||
        a.triangles.size() != b.triangles.size()) {
        return false;
    }
    for (std::size_t i = 0u; i < a.positions.size(); ++i) {
        if (!same_bits(a.positions[i].x, b.positions[i].x) ||
            !same_bits(a.positions[i].y, b.positions[i].y) ||
            !same_bits(a.positions[i].z, b.positions[i].z)) {
            return false;
        }
    }
    return a.triangles == b.triangles;
}

void hash_word(std::uint64_t &hash, std::uint32_t word) noexcept {
    constexpr auto prime = std::uint64_t{1099511628211ull};
    for (auto byte = 0u; byte < 4u; ++byte) {
        hash ^= (word >> (byte * 8u)) & 0xffu;
        hash *= prime;
    }
}

[[nodiscard]] std::uint64_t support_hash(
    const contract::TriangleMeshDesc &geometry) noexcept {
    auto hash = std::uint64_t{14695981039346656037ull};
    hash_word(hash, static_cast<std::uint32_t>(geometry.positions.size()));
    hash_word(hash, static_cast<std::uint32_t>(geometry.triangles.size()));
    for (const auto point : geometry.positions) {
        hash_word(hash, std::bit_cast<std::uint32_t>(point.x));
        hash_word(hash, std::bit_cast<std::uint32_t>(point.y));
        hash_word(hash, std::bit_cast<std::uint32_t>(point.z));
    }
    for (const auto triangle : geometry.triangles) {
        hash_word(hash, triangle[0u]);
        hash_word(hash, triangle[1u]);
        hash_word(hash, triangle[2u]);
    }
    return hash;
}

[[nodiscard]] std::uint64_t instance_support_hash(
    std::uint64_t geometry_hash,
    const Mat4f &transform) noexcept {
    auto hash = geometry_hash;
    for (const auto value : transform.elements) {
        hash_word(hash, std::bit_cast<std::uint32_t>(value));
    }
    return hash;
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
           iter->second.has_true_displacement;
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
    std::vector<const contract::InstanceDesc *> ordered_instances;
    ordered_instances.reserve(scene.instances.size());
    std::map<contract::GeometryId, std::size_t> geometry_users;
    for (const auto &[id, instance] : scene.instances) {
        static_cast<void>(id);
        ordered_instances.emplace_back(&instance);
        if (scene.geometries.contains(instance.geometry)) {
            ++geometry_users[instance.geometry];
        }
    }

    std::vector<CyclesInstanceIntersectionPlan> result;
    result.reserve(ordered_instances.size());
    for (std::size_t index = 0u; index < ordered_instances.size(); ++index) {
        const auto &instance = *ordered_instances[index];
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
            .coincident_next = static_cast<std::uint32_t>(index),
            .coincident_count = 1u,
            .transform_applied = transform_applied,
            .world_to_object = cycles_inverse_transform(instance.transform)});
    }

    struct Group {
        contract::GeometryId geometry;
        Mat4f transform;
        std::vector<std::uint32_t> instances;
    };
    std::map<contract::GeometryId, std::uint64_t> geometry_hashes;
    std::vector<Group> groups;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
    for (std::size_t index = 0u; index < ordered_instances.size(); ++index) {
        const auto &instance = *ordered_instances[index];
        const auto geometry_iter = scene.geometries.find(instance.geometry);
        if (geometry_iter == scene.geometries.end()) {
            continue;
        }
        const auto geometry_hash = geometry_hashes.try_emplace(
            instance.geometry,
            support_hash(geometry_iter->second)).first->second;
        const auto hash = instance_support_hash(
            geometry_hash, instance.transform);
        auto &candidate_groups = buckets[hash];
        std::optional<std::size_t> matching_group;
        for (const auto group_index : candidate_groups) {
            const auto &group = groups[group_index];
            if (same_transform_bits(group.transform, instance.transform) &&
                same_support_bits(
                    scene.geometries.at(group.geometry),
                    geometry_iter->second)) {
                matching_group = group_index;
                break;
            }
        }
        if (!matching_group) {
            matching_group = groups.size();
            candidate_groups.emplace_back(*matching_group);
            groups.emplace_back(Group{
                .geometry = instance.geometry,
                .transform = instance.transform,
                .instances = {}});
        }
        groups[*matching_group].instances.emplace_back(
            static_cast<std::uint32_t>(index));
    }
    for (const auto &group : groups) {
        const auto count = static_cast<std::uint32_t>(group.instances.size());
        for (std::size_t i = 0u; i < group.instances.size(); ++i) {
            const auto current = group.instances[i];
            const auto next = group.instances[(i + 1u) % group.instances.size()];
            result[current].coincident_next = next;
            result[current].coincident_count = count;
        }
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
