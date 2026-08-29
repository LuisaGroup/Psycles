#include "path_tracer_light_tree_scene.h"

#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto light_tree_pi = 3.14159265358979323846f;
constexpr auto light_tree_half_pi = 0.5f * light_tree_pi;

[[nodiscard]] Vec3f add(Vec3f a, Vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f from_luisa_host(luisa::float3 value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3f subtract(Vec3f a, Vec3f b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] Vec3f multiply(Vec3f a, Vec3f b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

[[nodiscard]] float dot_product(Vec3f a, Vec3f b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3f cross_product(Vec3f a, Vec3f b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] float vector_length(Vec3f value) noexcept {
    return std::sqrt(std::max(dot_product(value, value), 0.0f));
}

[[nodiscard]] Vec3f normalize_or(Vec3f value, Vec3f fallback) noexcept {
    const auto length = vector_length(value);
    return length > 1.0e-20f ? multiply(value, 1.0f / length) : fallback;
}

[[nodiscard]] float average_absolute(Vec3f value) noexcept {
    return (std::abs(value.x) + std::abs(value.y) + std::abs(value.z)) /
           3.0f;
}

[[nodiscard]] sampling::LightTreeBounds bounds_from_points(
    std::span<const Vec3f> points) noexcept {
    sampling::LightTreeBounds result;
    for (const auto point : points) {
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

[[nodiscard]] float transform_determinant(const Mat4f &transform) noexcept {
    return dot_product(
        matrix_axis(transform, 0u),
        cross_product(matrix_axis(transform, 1u), matrix_axis(transform, 2u)));
}

[[nodiscard]] std::uint32_t measure_flags(
    const sampling::LightTreeMeasure &measure,
    bool distant) noexcept {
    return (!measure.bounds.empty
                ? light_tree_measure_has_bounds
                : 0u) |
           (!measure.orientation.empty
                ? light_tree_measure_has_orientation
                : 0u) |
           (distant ? light_tree_measure_is_distant : 0u);
}

[[nodiscard]] luisa::float4 pack_min_energy(
    const sampling::LightTreeMeasure &measure) noexcept {
    return luisa::make_float4(
        measure.bounds.minimum.x,
        measure.bounds.minimum.y,
        measure.bounds.minimum.z,
        measure.energy);
}

[[nodiscard]] luisa::float4 pack_max_theta_o(
    const sampling::LightTreeMeasure &measure) noexcept {
    return luisa::make_float4(
        measure.bounds.maximum.x,
        measure.bounds.maximum.y,
        measure.bounds.maximum.z,
        measure.orientation.theta_o);
}

[[nodiscard]] luisa::float4 pack_cone_theta_e(
    const sampling::LightTreeMeasure &measure) noexcept {
    return luisa::make_float4(
        measure.orientation.axis.x,
        measure.orientation.axis.y,
        measure.orientation.axis.z,
        measure.orientation.theta_e);
}

struct DeviceEmitterIdentity {
    LightTreeEmitterKind kind{LightTreeEmitterKind::direct};
    LightTreeEmitterSource source{LightTreeEmitterSource::measure};
    std::uint32_t payload_0{};
    std::uint32_t payload_1{};
    std::uint32_t payload_2{};
};

struct AppendedTree {
    std::uint32_t root{sampling::invalid_light_tree_index};
    std::vector<luisa::uint2> mappings;
};

[[nodiscard]] std::uint32_t offset_index(
    std::uint32_t index,
    std::uint32_t offset) noexcept {
    return index == sampling::invalid_light_tree_index
               ? sampling::invalid_light_tree_index
               : index + offset;
}

[[nodiscard]] AppendedTree append_tree(
    LightTreeSceneUpload &result,
    const sampling::CyclesLightTree &tree,
    std::span<const DeviceEmitterIdentity> identities) {
    if (!tree.valid() || identities.size() != tree.emitters.size()) {
        throw std::invalid_argument("invalid light-tree append input");
    }
    if (result.nodes.size() + tree.nodes.size() >=
            static_cast<std::size_t>(sampling::invalid_light_tree_index) ||
        result.emitters.size() + tree.emitters.size() >=
            static_cast<std::size_t>(sampling::invalid_light_tree_index)) {
        throw std::length_error("light tree exceeds the 32-bit Luisa device ABI");
    }

    const auto node_offset = static_cast<std::uint32_t>(result.nodes.size());
    const auto emitter_offset =
        static_cast<std::uint32_t>(result.emitters.size());
    result.nodes.reserve(result.nodes.size() + tree.nodes.size());
    for (const auto &node : tree.nodes) {
        const auto distant =
            node.kind == sampling::LightTreeNodeKind::distant;
        result.nodes.emplace_back(LightTreeNodeGpu{
            .bounds_min_energy = pack_min_energy(node.measure),
            .bounds_max_theta_o = pack_max_theta_o(node.measure),
            .cone_axis_theta_e = pack_cone_theta_e(node.measure),
            .topology = luisa::make_uint4(
                offset_index(node.parent, node_offset),
                offset_index(node.left, node_offset),
                offset_index(node.right, node_offset),
                static_cast<std::uint32_t>(node.kind)),
            .emitters = luisa::make_uint4(
                node.emitter_count == 0u
                    ? 0u
                    : node.first_emitter + emitter_offset,
                node.emitter_count,
                measure_flags(node.measure, distant),
                0u)});
    }

    result.emitters.reserve(result.emitters.size() + tree.emitters.size());
    for (const auto &emitter : tree.emitters) {
        const auto &identity = identities[emitter.emitter_id];
        result.emitters.emplace_back(LightTreeEmitterGpu{
            .bounds_min_energy = pack_min_energy(emitter.measure),
            .bounds_max_theta_o = pack_max_theta_o(emitter.measure),
            .cone_axis_theta_e = pack_cone_theta_e(emitter.measure),
            .identity = luisa::make_uint4(
                identity.payload_0,
                measure_flags(emitter.measure, emitter.distant) |
                    light_tree_emitter_kind_bits(identity.kind) |
                    light_tree_emitter_source_bits(identity.source),
                identity.payload_1,
                identity.payload_2)});
    }

    AppendedTree appended;
    appended.root = tree.root + node_offset;
    appended.mappings.resize(tree.emitters.size());
    for (std::size_t emitter = 0u; emitter < tree.emitters.size(); ++emitter) {
        appended.mappings[emitter] = luisa::make_uint2(
            tree.emitter_to_tree[emitter] + emitter_offset,
            tree.emitter_to_leaf[emitter] + node_offset);
    }
    return appended;
}

[[nodiscard]] bool mapping_is_invalid(luisa::uint2 mapping) noexcept {
    return mapping.x == sampling::invalid_light_tree_index ||
           mapping.y == sampling::invalid_light_tree_index;
}

}// namespace

sampling::LightTreeEmitter make_triangle_light_tree_emitter(
    std::uint32_t emitter_id,
    const Mat4f &object_to_world,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2,
    Vec3f emission_estimate,
    contract::EmissionSampling emission_sampling,
    bool transform_applied) noexcept {
    std::array vertices{
        transform_applied ? p0 : cycles_transform_point(object_to_world, p0),
        transform_applied ? p1 : cycles_transform_point(object_to_world, p1),
        transform_applied ? p2 : cycles_transform_point(object_to_world, p2)};
    const auto cross = cross_product(
        subtract(vertices[1u], vertices[0u]),
        subtract(vertices[2u], vertices[0u]));
    const auto area = 0.5f * vector_length(cross);
    const auto front_only =
        emission_sampling == contract::EmissionSampling::front;
    const auto back_only =
        emission_sampling == contract::EmissionSampling::back;
    auto axis = front_only || back_only
                    ? normalize_or(cross, {0.0f, 0.0f, 1.0f})
                    : normalize_or(
                          subtract(vertices[0u], vertices[1u]),
                          {1.0f, 0.0f, 0.0f});
    if (back_only) {
        axis = multiply(axis, -1.0f);
    }
    if ((front_only || back_only) &&
        transform_determinant(object_to_world) < 0.0f) {
        axis = multiply(axis, -1.0f);
    }
    return {
        .measure = {
            .bounds = bounds_from_points(vertices),
            .orientation = {
                .axis = axis,
                .theta_o = front_only || back_only
                               ? 0.0f
                               : light_tree_half_pi,
                .theta_e = light_tree_half_pi,
                .empty = false},
            .energy = area * average_absolute(emission_estimate)},
        .centroid = multiply(
            add(add(vertices[0u], vertices[1u]), vertices[2u]),
            1.0f / 3.0f),
        .emitter_id = emitter_id,
        .distant = false};
}

sampling::LightTreeEmitter make_analytic_light_tree_emitter(
    std::uint32_t emitter_id,
    const LightGpu &light,
    Vec3f shader_emission_estimate) noexcept {
    const auto type = static_cast<contract::LightType>(light.type);
    const auto position = from_luisa_host(light.position);
    const auto axis_x = from_luisa_host(light.axis_x);
    const auto axis_y = from_luisa_host(light.axis_y);
    const auto axis_z = from_luisa_host(light.axis_z);
    auto strength = multiply(
        multiply(from_luisa_host(light.color), light.power),
        shader_emission_estimate);
    const auto normalize = (light.flags & light_flag_normalize) != 0u;
    if (!normalize) {
        auto area = 1.0f;
        if (type == contract::LightType::point ||
            type == contract::LightType::spot) {
            area = light.radius > 0.0f
                       ? 4.0f * light_tree_pi * light.radius * light.radius
                       : 4.0f;
        } else if (type == contract::LightType::area) {
            area = std::abs(light.size_u * light.size_v);
            if ((light.flags & light_flag_ellipse) != 0u) {
                area *= 0.25f * light_tree_pi;
            }
        } else if (type == contract::LightType::distant &&
                   light.angle > 0.0f) {
            const auto sine = std::sin(0.5f * light.angle);
            area = light_tree_pi * sine * sine;
        }
        strength = multiply(strength, area);
    }

    sampling::LightTreeMeasure measure;
    measure.orientation.axis = multiply(axis_z, -1.0f);
    measure.orientation.empty = false;
    bool distant = false;
    if (type == contract::LightType::area) {
        measure.orientation.theta_e = 0.5f * light.spread;
        const auto half_u = multiply(axis_x, 0.5f * light.size_u);
        const auto half_v = multiply(axis_y, 0.5f * light.size_v);
        const std::array corners{
            add(position, add(half_u, half_v)),
            add(position, subtract(half_u, half_v)),
            add(position, subtract(half_v, half_u)),
            subtract(position, add(half_u, half_v))};
        measure.bounds = bounds_from_points(corners);
        strength = multiply(strength, 1.0f / light_tree_pi);
    } else if (type == contract::LightType::point) {
        measure.orientation.theta_o = light_tree_pi;
        measure.orientation.theta_e = light_tree_half_pi;
    } else if (type == contract::LightType::spot) {
        auto theta_e = std::min(
            0.5f * light.spot_angle, light_tree_half_pi);
        if (std::abs(light_tree_half_pi - theta_e) >= 1.0e-6f) {
            theta_e = std::atan(
                std::tan(theta_e) *
                std::max(light.axis_scale.x, light.axis_scale.y) /
                std::max(light.axis_scale.z, 1.0e-20f));
        }
        measure.orientation.theta_e = theta_e;
    } else if (type == contract::LightType::distant) {
        measure.orientation.theta_e = 0.5f * light.angle;
        distant = true;
    }
    if (type == contract::LightType::point ||
        type == contract::LightType::spot) {
        const auto radius = Vec3f{light.radius, light.radius, light.radius};
        const std::array points{
            subtract(position, radius), add(position, radius)};
        measure.bounds = bounds_from_points(points);
        strength = multiply(strength, 0.25f / light_tree_pi);
    }
    measure.energy = average_absolute(strength);
    return {.measure = measure,
            .centroid = position,
            .emitter_id = emitter_id,
            .distant = distant};
}

sampling::LightTreeEmitter make_environment_light_tree_emitter(
    std::uint32_t emitter_id,
    Vec3f emission_estimate) noexcept {
    return {
        .measure = {
            .orientation = {
                .axis = {0.0f, 0.0f, 1.0f},
                .theta_o = light_tree_pi,
                .theta_e = 0.0f,
                .empty = false},
            .energy = light_tree_pi * average_absolute(emission_estimate)},
        .centroid = {0.0f, 0.0f, 1.0f},
        .emitter_id = emitter_id,
        .distant = true};
}

luisa::vector<luisa::uint4> make_light_tree_triangle_lookup(
    std::span<const EmissiveTriangleGpu> triangles) {
    luisa::vector<luisa::uint4> result;
    result.reserve(triangles.size());
    for (std::size_t emitter_id = 0u;
         emitter_id < triangles.size();
         ++emitter_id) {
        result.emplace_back(
            triangles[emitter_id].cycles_object_index,
            triangles[emitter_id].cycles_primitive_index,
            static_cast<std::uint32_t>(emitter_id),
            0u);
    }
    std::sort(
        result.begin(),
        result.end(),
        [](luisa::uint4 a, luisa::uint4 b) noexcept {
            return a.x < b.x ||
                   (a.x == b.x &&
                    (a.y < b.y || (a.y == b.y && a.z < b.z)));
        });
    return result;
}

LightTreeSceneUpload make_light_tree_scene_upload(
    std::span<const sampling::LightTreeEmitter> emitters) noexcept {
    if (emitters.size() >=
        static_cast<std::size_t>(sampling::invalid_light_tree_index)) {
        LightTreeSceneUpload result;
        result.diagnostic = "light tree exceeds the 32-bit emitter ABI";
        return result;
    }
    LightTreeHierarchyInput input;
    input.distribution_emitter_count =
        static_cast<std::uint32_t>(emitters.size());
    input.top_emitters.reserve(emitters.size());
    for (std::size_t index = 0u; index < emitters.size(); ++index) {
        auto emitter = emitters[index];
        emitter.emitter_id = static_cast<std::uint32_t>(index);
        input.top_emitters.emplace_back(LightTreeTopEmitterInput{
            .emitter = emitter,
            .kind = LightTreeEmitterKind::direct,
            .payload = static_cast<std::uint32_t>(index)});
    }
    return make_light_tree_hierarchy_scene_upload(input);
}

LightTreeSceneUpload make_light_tree_hierarchy_scene_upload(
    const LightTreeHierarchyInput &input) noexcept {
    LightTreeSceneUpload result;
    try {
        if (input.distribution_emitter_count >=
                sampling::invalid_light_tree_index ||
            input.triangle_emitter_count >
                input.distribution_emitter_count) {
            result.diagnostic = "invalid light-tree emitter population";
            return result;
        }
        if (input.top_emitters.empty()) {
            return result;
        }

        std::vector<AppendedTree> subtrees;
        subtrees.reserve(input.subtrees.size());
        for (const auto &subtree_input : input.subtrees) {
            if (!subtree_input.representative_triangles.empty() &&
                subtree_input.representative_triangles.size() !=
                    subtree_input.tree.emitters.size()) {
                throw std::invalid_argument(
                    "mesh light subtree source population mismatch");
            }
            const auto &tree = subtree_input.tree;
            if (!tree.valid()) {
                throw std::invalid_argument("empty mesh light subtree");
            }
            std::vector<DeviceEmitterIdentity> identities(
                tree.emitters.size());
            for (std::size_t emitter = 0u;
                 emitter < identities.size();
                 ++emitter) {
                identities[emitter] = {
                    .kind = LightTreeEmitterKind::mesh_triangle,
                    .source =
                        subtree_input.representative_triangles.empty()
                            ? LightTreeEmitterSource::measure
                            : LightTreeEmitterSource::triangle,
                    .payload_0 = static_cast<std::uint32_t>(emitter),
                    .payload_1 =
                        subtree_input.representative_triangles.empty()
                            ? 0u
                            : subtree_input
                                  .representative_triangles[emitter]};
            }
            subtrees.emplace_back(append_tree(result, tree, identities));
        }

        std::vector<sampling::LightTreeEmitter> top_emitters;
        std::vector<DeviceEmitterIdentity> top_identities;
        top_emitters.reserve(input.top_emitters.size());
        top_identities.reserve(input.top_emitters.size());
        for (std::size_t index = 0u;
             index < input.top_emitters.size();
             ++index) {
            const auto &source = input.top_emitters[index];
            auto emitter = source.emitter;
            emitter.emitter_id = static_cast<std::uint32_t>(index);
            top_emitters.emplace_back(emitter);
            if (source.kind == LightTreeEmitterKind::direct) {
                if (source.payload >= input.distribution_emitter_count ||
                    source.subtree != sampling::invalid_light_tree_index ||
                    !source.triangle_emitters.empty()) {
                    throw std::invalid_argument(
                        "invalid direct light-tree emitter metadata");
                }
                top_identities.emplace_back(DeviceEmitterIdentity{
                    .kind = source.kind,
                    .source = source.source,
                    .payload_0 = source.payload,
                    .payload_1 = source.source_index});
                continue;
            }
            if (source.kind != LightTreeEmitterKind::mesh_instance ||
                source.source != LightTreeEmitterSource::measure ||
                source.subtree >= subtrees.size() ||
                source.triangle_emitters.size() !=
                    input.subtrees[source.subtree].tree.emitters.size()) {
                throw std::invalid_argument(
                    "invalid mesh light-tree emitter metadata");
            }
            if (result.mesh_triangles.size() +
                    source.triangle_emitters.size() >=
                static_cast<std::size_t>(sampling::invalid_light_tree_index)) {
                throw std::length_error(
                    "mesh light mapping exceeds the 32-bit device ABI");
            }
            const auto mapping_offset =
                static_cast<std::uint32_t>(result.mesh_triangles.size());
            result.mesh_triangles.insert(
                result.mesh_triangles.end(),
                source.triangle_emitters.begin(),
                source.triangle_emitters.end());
            top_identities.emplace_back(DeviceEmitterIdentity{
                .kind = source.kind,
                .payload_0 = source.payload,
                .payload_1 = subtrees[source.subtree].root,
                .payload_2 = mapping_offset});
        }

        const auto top_tree =
            sampling::build_cycles_light_tree(top_emitters);
        if (!top_tree.valid()) {
            throw std::invalid_argument("invalid top-level light tree");
        }
        const auto top = append_tree(result, top_tree, top_identities);
        result.root = top.root;
        const auto invalid_mapping = luisa::make_uint2(
            sampling::invalid_light_tree_index,
            sampling::invalid_light_tree_index);
        result.emitter_mappings.assign(
            input.distribution_emitter_count, invalid_mapping);
        result.triangle_emitter_mappings.assign(
            input.triangle_emitter_count, invalid_mapping);

        for (std::size_t index = 0u;
             index < input.top_emitters.size();
             ++index) {
            const auto &source = input.top_emitters[index];
            if (source.kind == LightTreeEmitterKind::direct) {
                auto &mapping = result.emitter_mappings[source.payload];
                if (!mapping_is_invalid(mapping)) {
                    throw std::invalid_argument(
                        "duplicate direct light-tree emitter mapping");
                }
                mapping = top.mappings[index];
                continue;
            }
            for (std::size_t local = 0u;
                 local < source.triangle_emitters.size();
                 ++local) {
                const auto triangle = source.triangle_emitters[local];
                if (triangle >= input.triangle_emitter_count ||
                    !mapping_is_invalid(result.emitter_mappings[triangle]) ||
                    !mapping_is_invalid(
                        result.triangle_emitter_mappings[triangle])) {
                    throw std::invalid_argument(
                        "invalid or duplicate mesh triangle mapping");
                }
                result.emitter_mappings[triangle] = top.mappings[index];
                result.triangle_emitter_mappings[triangle] =
                    subtrees[source.subtree].mappings[local];
            }
        }
        if (std::any_of(
                result.emitter_mappings.begin(),
                result.emitter_mappings.end(),
                mapping_is_invalid) ||
            std::any_of(
                result.triangle_emitter_mappings.begin(),
                result.triangle_emitter_mappings.end(),
                mapping_is_invalid)) {
            throw std::invalid_argument(
                "light-tree hierarchy does not cover every emitter");
        }
    } catch (const std::exception &error) {
        result = {};
        result.diagnostic = error.what();
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
