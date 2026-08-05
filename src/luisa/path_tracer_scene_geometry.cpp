#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <span>
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

void hash_word(std::uint64_t &hash, std::uint32_t word) noexcept {
    constexpr auto prime = std::uint64_t{1099511628211ull};
    for (auto byte = 0u; byte < 4u; ++byte) {
        hash ^= (word >> (byte * 8u)) & 0xffu;
        hash *= prime;
    }
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

[[nodiscard]] bool finite(Vec3f point) noexcept {
    return std::isfinite(point.x) &&
           std::isfinite(point.y) &&
           std::isfinite(point.z);
}

// Exact world-support equality is a finite relation over the final vertex
// array, not an approximate matrix comparison. A small deterministic witness
// routes every equal support into the same bucket because equality of the
// complete array implies equality of each witness. Non-equal supports may
// still collide, so the bucket match always compares every transformed vertex.
[[nodiscard]] std::uint64_t instance_support_hash(
    std::uint32_t support_class,
    std::span<const Vec3f> positions) noexcept {
    auto hash = std::uint64_t{14695981039346656037ull};
    hash_word(hash, support_class);
    if (positions.empty()) {
        return hash;
    }
    const std::array witnesses{
        std::size_t{0u},
        positions.size() / 3u,
        (positions.size() * 2u) / 3u,
        positions.size() - 1u};
    for (const auto index : witnesses) {
        const auto point = positions[index];
        hash_word(hash, std::bit_cast<std::uint32_t>(point.x));
        hash_word(hash, std::bit_cast<std::uint32_t>(point.y));
        hash_word(hash, std::bit_cast<std::uint32_t>(point.z));
    }
    return hash;
}

[[nodiscard]] bool same_world_support_bits(
    std::span<const Vec3f> a,
    std::span<const Vec3f> b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < a.size(); ++index) {
        const auto pa = a[index];
        const auto pb = b[index];
        if (!same_bits(pa.x, pb.x) ||
            !same_bits(pa.y, pb.y) ||
            !same_bits(pa.z, pb.z)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::array<std::uint32_t, 9u>
primitive_support_signature(
    std::span<const Vec3f> world_positions,
    std::array<std::uint32_t, 3u> triangle) noexcept {
    std::array<std::uint32_t, 9u> result{};
    for (auto corner = 0u; corner < 3u; ++corner) {
        const auto point = world_positions[triangle[corner]];
        result[corner * 3u + 0u] =
            std::bit_cast<std::uint32_t>(point.x);
        result[corner * 3u + 1u] =
            std::bit_cast<std::uint32_t>(point.y);
        result[corner * 3u + 2u] =
            std::bit_cast<std::uint32_t>(point.z);
    }
    return result;
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

    return result;
}

bool finalize_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::map<contract::GeometryId, std::uint32_t>
        &final_triangle_support_classes,
    const std::map<contract::GeometryId, CyclesGeometrySupportView>
        &final_supports,
    std::span<CyclesInstanceIntersectionPlan> plan,
    CyclesPrimitiveIntersectionPlan &primitive_plan) {
    constexpr auto maximum_index =
        std::numeric_limits<std::uint32_t>::max();
    primitive_plan = {};
    if (plan.size() != scene.instances.size() ||
        plan.size() > maximum_index) {
        return false;
    }
    for (std::size_t index = 0u; index < plan.size(); ++index) {
        plan[index].coincident_next = static_cast<std::uint32_t>(index);
        plan[index].coincident_count = 1u;
        plan[index].coincident_primitive_offset = 0u;
        plan[index].coincident_primitive_count = 0u;
    }

    struct InstanceSupport {
        std::uint32_t instance{};
        CyclesGeometrySupportView support;
        std::vector<Vec3f> world_positions;
        bool all_finite{true};
    };
    std::map<std::uint32_t, std::vector<InstanceSupport>> support_groups;
    auto index = std::size_t{0u};
    for (const auto &[instance_id, instance] : scene.instances) {
        static_cast<void>(instance_id);
        const auto support_class =
            final_triangle_support_classes.find(instance.geometry);
        if (support_class == final_triangle_support_classes.end()) {
            ++index;
            continue;
        }
        const auto support = final_supports.find(instance.geometry);
        if (support == final_supports.end()) {
            return false;
        }
        const auto view = support->second;
        const auto valid_view =
            view.load_position != nullptr &&
            view.load_triangle != nullptr &&
            view.position_count <= maximum_index &&
            view.triangle_count <= maximum_index &&
            (view.position_count == 0u ||
             view.position_data != nullptr) &&
            (view.triangle_count == 0u ||
             view.triangle_data != nullptr);
        if (!valid_view) {
            return false;
        }
        InstanceSupport entry{
            .instance = static_cast<std::uint32_t>(index),
            .support = view,
            .world_positions = {},
            .all_finite = true};
        entry.world_positions.reserve(view.position_count);
        for (std::size_t vertex = 0u;
             vertex < view.position_count;
             ++vertex) {
            const auto point = transform_point(
                instance.transform, view.position(vertex));
            entry.all_finite &= finite(point);
            entry.world_positions.emplace_back(point);
        }
        for (std::size_t primitive = 0u;
             primitive < view.triangle_count;
             ++primitive) {
            const auto triangle = view.triangle(primitive);
            if (triangle[0u] >= view.position_count ||
                triangle[1u] >= view.position_count ||
                triangle[2u] >= view.position_count) {
                return false;
            }
        }
        support_groups[support_class->second].emplace_back(
            std::move(entry));
        ++index;
    }

    std::vector<std::uint32_t> whole_support_representative(plan.size());
    for (std::size_t instance = 0u; instance < plan.size(); ++instance) {
        whole_support_representative[instance] =
            static_cast<std::uint32_t>(instance);
    }
    std::vector<std::vector<CyclesCoincidentPrimitiveRecord>>
        records_by_instance(plan.size());

    for (auto &[support_class, entries] : support_groups) {
        if (entries.empty()) {
            continue;
        }
        const auto position_count = entries.front().support.position_count;
        const auto triangle_count = entries.front().support.triangle_count;
        for (const auto &entry : entries) {
            if (entry.support.position_count != position_count ||
                entry.support.triangle_count != triangle_count) {
                return false;
            }
        }

        struct WholeGroup {
            std::size_t representative{};
            std::vector<std::uint32_t> instances;
        };
        std::vector<WholeGroup> whole_groups;
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
        for (std::size_t entry_index = 0u;
             entry_index < entries.size();
             ++entry_index) {
            const auto &entry = entries[entry_index];
            if (!entry.all_finite) {
                continue;
            }
            const auto hash = instance_support_hash(
                support_class, entry.world_positions);
            auto &candidates = buckets[hash];
            std::optional<std::size_t> matching_group;
            for (const auto candidate : candidates) {
                const auto &group = whole_groups[candidate];
                if (same_world_support_bits(
                        entries[group.representative].world_positions,
                        entry.world_positions)) {
                    matching_group = candidate;
                    break;
                }
            }
            if (!matching_group) {
                matching_group = whole_groups.size();
                candidates.emplace_back(*matching_group);
                whole_groups.emplace_back(WholeGroup{
                    .representative = entry_index,
                    .instances = {}});
            }
            whole_groups[*matching_group].instances.emplace_back(
                entry.instance);
        }
        for (const auto &group : whole_groups) {
            const auto count =
                static_cast<std::uint32_t>(group.instances.size());
            const auto representative = group.instances.front();
            for (std::size_t member = 0u;
                 member < group.instances.size();
                 ++member) {
                const auto current = group.instances[member];
                const auto next = group.instances[
                    (member + 1u) % group.instances.size()];
                plan[current].coincident_next = next;
                plan[current].coincident_count = count;
                whole_support_representative[current] = representative;
            }
        }

        struct PrimitiveCandidate {
            std::array<std::uint32_t, 9u> signature;
            std::uint32_t instance{};
        };
        std::vector<PrimitiveCandidate> candidates;
        candidates.reserve(entries.size());
        for (std::size_t primitive = 0u;
             primitive < triangle_count;
             ++primitive) {
            candidates.clear();
            for (const auto &entry : entries) {
                const auto triangle = entry.support.triangle(primitive);
                const auto p0 = entry.world_positions[triangle[0u]];
                const auto p1 = entry.world_positions[triangle[1u]];
                const auto p2 = entry.world_positions[triangle[2u]];
                if (!finite(p0) || !finite(p1) || !finite(p2)) {
                    continue;
                }
                candidates.emplace_back(PrimitiveCandidate{
                    .signature = primitive_support_signature(
                        entry.world_positions, triangle),
                    .instance = entry.instance});
            }
            std::sort(
                candidates.begin(), candidates.end(),
                [](const PrimitiveCandidate &a,
                   const PrimitiveCandidate &b) noexcept {
                    if (a.signature != b.signature) {
                        return a.signature < b.signature;
                    }
                    return a.instance < b.instance;
                });
            for (std::size_t first = 0u;
                 first < candidates.size();) {
                auto last = first + 1u;
                while (last < candidates.size() &&
                       candidates[last].signature ==
                           candidates[first].signature) {
                    ++last;
                }
                auto spans_whole_groups = false;
                for (auto member = first + 1u;
                     member < last;
                     ++member) {
                    spans_whole_groups |=
                        whole_support_representative[
                            candidates[member].instance] !=
                        whole_support_representative[
                            candidates[first].instance];
                }
                if (last - first > 1u && spans_whole_groups) {
                    if (primitive_plan.instances.size() > maximum_index -
                            (last - first)) {
                        return false;
                    }
                    const auto alias_offset =
                        static_cast<std::uint32_t>(
                            primitive_plan.instances.size());
                    const auto alias_count =
                        static_cast<std::uint32_t>(last - first);
                    for (auto member = first; member < last; ++member) {
                        primitive_plan.instances.emplace_back(
                            candidates[member].instance);
                    }
                    for (auto member = first; member < last; ++member) {
                        records_by_instance[candidates[member].instance]
                            .emplace_back(CyclesCoincidentPrimitiveRecord{
                                .local_primitive =
                                    static_cast<std::uint32_t>(primitive),
                                .instance_offset = alias_offset,
                                .instance_count = alias_count});
                    }
                }
                first = last;
            }
        }
    }

    for (std::size_t instance = 0u;
         instance < records_by_instance.size();
         ++instance) {
        auto &records = records_by_instance[instance];
        std::sort(
            records.begin(), records.end(),
            [](const CyclesCoincidentPrimitiveRecord &a,
               const CyclesCoincidentPrimitiveRecord &b) noexcept {
                return a.local_primitive < b.local_primitive;
            });
        if (primitive_plan.records.size() >
            maximum_index - records.size()) {
            return false;
        }
        plan[instance].coincident_primitive_offset =
            static_cast<std::uint32_t>(primitive_plan.records.size());
        plan[instance].coincident_primitive_count =
            static_cast<std::uint32_t>(records.size());
        primitive_plan.records.insert(
            primitive_plan.records.end(),
            records.begin(), records.end());
    }
    return true;
}

}// namespace psycles::luisa_backend::detail
