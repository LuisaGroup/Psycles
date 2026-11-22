#include "blender_scene_geometry.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace psycles::adapter::detail {
namespace {

static_assert(std::is_trivially_copyable_v<Vec2f> &&
              sizeof(Vec2f) == 2u * sizeof(float));
static_assert(std::is_trivially_copyable_v<Vec3f> &&
              sizeof(Vec3f) == 3u * sizeof(float));
static_assert(std::is_trivially_copyable_v<Vec4f> &&
              sizeof(Vec4f) == 4u * sizeof(float));
static_assert(std::is_trivially_copyable_v<std::array<std::uint32_t, 2u>> &&
              sizeof(std::array<std::uint32_t, 2u>) == 2u * sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<std::array<std::uint32_t, 3u>> &&
              sizeof(std::array<std::uint32_t, 3u>) == 3u * sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<std::array<std::uint8_t, 4u>> &&
              sizeof(std::array<std::uint8_t, 4u>) == 4u);

[[nodiscard]] contract::MeshAttributeDomain mesh_attribute_domain(
    yyjson_val *value,
    contract::MeshAttributeDomain fallback =
        contract::MeshAttributeDomain::point) {
    const auto name = text(value);
    if (name.empty()) {
        return fallback;
    }
    if (name == "POINT") {
        return contract::MeshAttributeDomain::point;
    }
    if (name == "CORNER") {
        return contract::MeshAttributeDomain::corner;
    }
    if (name == "FACE") {
        return contract::MeshAttributeDomain::face;
    }
    throw std::runtime_error(
        "unsupported mesh attribute domain '" + name + "'");
}

[[nodiscard]] contract::CurveShape parse_curve_shape(yyjson_val *value) {
    const auto name = text(value);
    if (name == "RIBBON" || name == "RIBBONS") {
        return contract::CurveShape::ribbon;
    }
    if (name == "THICK") {
        return contract::CurveShape::thick;
    }
    if (name == "THICK_LINEAR") {
        return contract::CurveShape::thick_linear;
    }
    throw std::runtime_error(
        "unsupported Cycles curve shape '" + name + "'");
}

[[nodiscard]] constexpr std::uint32_t rotate_left(
    std::uint32_t value,
    unsigned int bits) noexcept {
    return (value << bits) | (value >> (32u - bits));
}

} // namespace

std::uint64_t section_offset(
    yyjson_val *geometry,
    const char *name) noexcept {
    return unsigned_number(member(member(geometry, name), "offset"));
}

std::uint32_t cycles_jenkins_hash_uint(std::uint32_t value) noexcept {
    std::uint32_t a = 0xDEADBEEFu + (1u << 2u) + 13u;
    std::uint32_t b = a;
    std::uint32_t c = a;
    a += value;
    c = (c ^ b) - rotate_left(b, 14u);
    a = (a ^ c) - rotate_left(c, 11u);
    b = (b ^ a) - rotate_left(a, 25u);
    c = (c ^ b) - rotate_left(b, 16u);
    a = (a ^ c) - rotate_left(c, 4u);
    b = (b ^ a) - rotate_left(a, 14u);
    c = (c ^ b) - rotate_left(b, 24u);
    return c;
}

std::vector<float> compute_cycles_random_per_island(
    std::size_t point_count,
    std::span<const std::uint32_t> edge_vertices,
    std::span<const std::array<std::uint32_t, 3u>> triangles) {
    if (edge_vertices.size() % 2u != 0u) {
        throw std::runtime_error("mesh island_edges must contain pairs of vertex indices");
    }
    std::vector<std::uint32_t> parents(point_count);
    std::iota(parents.begin(), parents.end(), 0u);
    std::vector<std::uint32_t> ranks(point_count, 0u);

    const auto find_root = [&parents](std::uint32_t vertex_index) noexcept {
        auto root = vertex_index;
        while (parents[root] != root) {
            root = parents[root];
        }
        while (parents[vertex_index] != root) {
            const auto parent = parents[vertex_index];
            parents[vertex_index] = root;
            vertex_index = parent;
        }
        return root;
    };

    const auto join_vertices = [&](std::uint32_t first, std::uint32_t second) {
        if (first >= point_count || second >= point_count) {
            throw std::runtime_error("mesh island edge index out of range");
        }
        auto root_first = find_root(first);
        auto root_second = find_root(second);
        if (root_first == root_second) {
            return;
        }
        if (ranks[root_first] < ranks[root_second]) {
            std::swap(root_first, root_second);
        }
        parents[root_second] = root_first;
        if (ranks[root_first] == ranks[root_second]) {
            ++ranks[root_first];
        }
    };

    for (std::size_t i = 0u; i < edge_vertices.size(); i += 2u) {
        join_vertices(edge_vertices[i], edge_vertices[i + 1u]);
    }

    std::vector<float> result;
    result.reserve(triangles.size());
    for (const auto &triangle : triangles) {
        const auto first_vertex = triangle[0u];
        if (first_vertex >= point_count) {
            throw std::runtime_error("mesh triangle vertex index out of range");
        }
        const auto root = find_root(first_vertex);
        const auto hashed = cycles_jenkins_hash_uint(root);
        result.push_back(static_cast<float>(
            static_cast<double>(hashed) / static_cast<double>(0xFFFFFFFFu)));
    }
    return result;
}

contract::TriangleMeshDesc parse_blender_triangle_mesh(
    yyjson_val *geometry,
    std::span<const std::byte> geometry_bin,
    bool compact_geometry,
    const std::map<std::string, contract::MaterialId, std::less<>> &material_ids,
    contract::MaterialId default_material) {
    using contract::MeshAttributeDomain;
    using contract::TriangleMeshDesc;

    const auto triangle_count = static_cast<std::size_t>(
        unsigned_number(member(geometry, "triangle_count")));
    const auto point_count = static_cast<std::size_t>(unsigned_number(
        member(geometry, compact_geometry ? "point_count" : "vertex_count")));
    const auto corner_count = triangle_count * 3u;
    if (compact_geometry &&
        static_cast<std::size_t>(unsigned_number(
            member(geometry, "corner_count"))) != corner_count) {
        throw std::runtime_error("compact geometry has an invalid corner count");
    }
    const auto domain_count = [&](MeshAttributeDomain domain) noexcept {
        switch (domain) {
            case MeshAttributeDomain::point:
                return point_count;
            case MeshAttributeDomain::corner:
                return corner_count;
            case MeshAttributeDomain::face:
                return triangle_count;
        }
        return std::size_t{0u};
    };

    const auto normal_domain =
        compact_geometry
            ? mesh_attribute_domain(member(geometry, "normal_domain"))
            : MeshAttributeDomain::point;
    const auto uv_domain =
        compact_geometry
            ? mesh_attribute_domain(member(geometry, "uv_domain"))
            : MeshAttributeDomain::point;
    const auto uv_tangent_domain =
        compact_geometry
            ? mesh_attribute_domain(member(geometry, "uv_tangent_domain"))
            : MeshAttributeDomain::point;
    const auto generated_domain =
        compact_geometry
            ? mesh_attribute_domain(member(geometry, "generated_domain"))
            : MeshAttributeDomain::point;

    const auto normal_count = domain_count(normal_domain);
    const auto uv_count = domain_count(uv_domain);
    const auto uv_tangent_count = domain_count(uv_tangent_domain);
    const auto generated_count = domain_count(generated_domain);

    TriangleMeshDesc mesh;
    mesh.name = text(member(geometry, "name"));
    mesh.positions = read_packed_values<Vec3f>(
        geometry_bin, section_offset(geometry, "positions"), point_count);
    mesh.normals.domain = normal_domain;
    mesh.normals.values = read_packed_values<Vec3f>(
        geometry_bin, section_offset(geometry, "normals"), normal_count);

    mesh.uv.domain = uv_domain;
    if (member(geometry, "default_uv_available") != nullptr) {
        mesh.default_uv_available = boolean(
            member(geometry, "default_uv_available"), false);
    } else if (auto *layers = member(geometry, "uv_layers");
               layers != nullptr && yyjson_is_arr(layers)) {
        mesh.default_uv_available = yyjson_arr_size(layers) != 0u;
    }
    mesh.uv.values = read_packed_values<Vec2f>(
        geometry_bin, section_offset(geometry, "uv"), uv_count);

    mesh.uv_tangents.domain = uv_tangent_domain;
    if (auto *uv_tangents_val = member(geometry, "uv_tangents");
        uv_tangents_val != nullptr && !yyjson_is_null(uv_tangents_val)) {
        mesh.uv_tangents.values = read_packed_values<Vec4f>(
            geometry_bin,
            section_offset(geometry, "uv_tangents"),
            uv_tangent_count);
    } else {
        mesh.uv_tangents.values.resize(uv_tangent_count);
    }

    mesh.generated.domain = generated_domain;
    mesh.generated.values = read_packed_values<Vec3f>(
        geometry_bin, section_offset(geometry, "generated"), generated_count);
    if (member(geometry, "generated_transform") != nullptr) {
        mesh.generated_transform = matrix(
            member(geometry, "generated_transform"));
    }

    if (auto *pointiness_source = member(geometry, "pointiness_source");
        pointiness_source != nullptr && !yyjson_is_null(pointiness_source)) {
        const auto edge_count = static_cast<std::size_t>(
            unsigned_number(member(pointiness_source, "edge_count")));
        auto &source = mesh.pointiness_source.emplace();
        source.point_normals = read_packed_values<Vec3f>(
            geometry_bin,
            section_offset(pointiness_source, "point_normals"),
            point_count);
        source.edges = read_packed_values<std::array<std::uint32_t, 2u>>(
            geometry_bin,
            section_offset(pointiness_source, "edges"),
            edge_count);
    }

    if (auto *uv_layers = member(geometry, "uv_layers");
        uv_layers != nullptr && yyjson_is_arr(uv_layers)) {
        yyjson_arr_iter uv_layer_iterator = yyjson_arr_iter_with(uv_layers);
        while (auto *uv_layer = yyjson_arr_iter_next(&uv_layer_iterator)) {
            const auto name = text(member(uv_layer, "name"));
            const auto domain =
                compact_geometry
                    ? mesh_attribute_domain(
                          member(uv_layer, "domain"),
                          MeshAttributeDomain::corner)
                    : MeshAttributeDomain::point;
            const auto value_count = domain_count(domain);
            auto &uv_destination = mesh.uv_layers[name];
            auto &tangent_destination = mesh.uv_tangent_layers[name];
            uv_destination.domain = domain;
            tangent_destination.domain = domain;
            uv_destination.values = read_packed_values<Vec2f>(
                geometry_bin, section_offset(uv_layer, "values"), value_count);
            if (auto *tangents_val = member(uv_layer, "tangents");
                tangents_val != nullptr && !yyjson_is_null(tangents_val)) {
                tangent_destination.values = read_packed_values<Vec4f>(
                    geometry_bin,
                    section_offset(uv_layer, "tangents"),
                    value_count);
            } else {
                tangent_destination.values.resize(value_count);
            }
        }
    }

    if (auto *default_color = member(geometry, "default_color_attribute");
        default_color != nullptr && yyjson_is_str(default_color)) {
        mesh.default_color_attribute = text(default_color);
    }
    if (auto *color_attributes = member(geometry, "color_attributes");
        color_attributes != nullptr && yyjson_is_arr(color_attributes)) {
        yyjson_arr_iter attribute_iterator =
            yyjson_arr_iter_with(color_attributes);
        while (auto *attribute = yyjson_arr_iter_next(&attribute_iterator)) {
            const auto name = text(member(attribute, "name"));
            const auto domain =
                compact_geometry
                    ? mesh_attribute_domain(member(attribute, "domain"))
                    : MeshAttributeDomain::point;
            const auto value_count = domain_count(domain);
            auto &destination = mesh.color_attributes[name];
            destination.domain = domain;
            destination.values = read_packed_values<Vec4f>(
                geometry_bin, section_offset(attribute, "values"), value_count);
            if (text(member(attribute, "data_type")) == "BYTE_COLOR" &&
                domain == MeshAttributeDomain::corner) {
                if (auto *byte_section = member(attribute, "byte_values");
                    byte_section != nullptr && !yyjson_is_null(byte_section)) {
                    auto &raw = mesh.cycles_byte_color_attributes[name];
                    raw.domain = domain;
                    raw.values = read_packed_values<std::array<std::uint8_t, 4u>>(
                        geometry_bin,
                        section_offset(attribute, "byte_values"),
                        value_count);
                }
            }
        }
    }

    mesh.triangles = read_packed_values<std::array<std::uint32_t, 3u>>(
        geometry_bin, section_offset(geometry, "indices"), triangle_count);
    mesh.triangle_material_slots = read_packed_values<std::uint32_t>(
        geometry_bin,
        section_offset(geometry, "triangle_material_slots"),
        triangle_count);

    mesh.triangle_smooth.resize(triangle_count, 0u);
    if (member(geometry, "triangle_smooth") != nullptr) {
        const auto smooth_values = read_packed_values<std::uint32_t>(
            geometry_bin,
            section_offset(geometry, "triangle_smooth"),
            triangle_count);
        for (std::size_t i = 0u; i < triangle_count; ++i) {
            mesh.triangle_smooth[i] =
                static_cast<std::uint8_t>(smooth_values[i] != 0u);
        }
    }

    if (auto *island_random_val = member(geometry, "triangle_random_per_island");
        island_random_val != nullptr && !yyjson_is_null(island_random_val)) {
        mesh.triangle_random_per_island = read_packed_values<float>(
            geometry_bin,
            section_offset(geometry, "triangle_random_per_island"),
            triangle_count);
    } else if (auto *island_edges = member(geometry, "island_edges");
               island_edges != nullptr && !yyjson_is_null(island_edges)) {
        const auto edge_count = static_cast<std::size_t>(
            unsigned_number(member(island_edges, "edge_count")));
        const auto edge_vertices = read_packed_values<std::uint32_t>(
            geometry_bin,
            section_offset(island_edges, "edges"),
            edge_count * 2u);
        mesh.triangle_random_per_island = compute_cycles_random_per_island(
            point_count, edge_vertices, mesh.triangles);
    } else {
        mesh.triangle_random_per_island.assign(triangle_count, 0.0f);
    }

    mesh.cycles_primitive_offset = optional_unsigned_number(
        member(member(geometry, "cycles_sync"), "primitive_offset"));
    mesh.uses_adaptive_subdivision = boolean(
        member(geometry, "uses_adaptive_subdivision"), false);

    auto *slots = member(geometry, "material_slots");
    yyjson_arr_iter slot_iterator = yyjson_arr_iter_with(slots);
    while (auto *slot = yyjson_arr_iter_next(&slot_iterator)) {
        if (yyjson_is_null(slot)) {
            mesh.material_slots.emplace_back(default_material);
            continue;
        }
        const auto iter = material_ids.find(text(slot));
        mesh.material_slots.emplace_back(
            iter == material_ids.end() ? default_material : iter->second);
    }
    if (mesh.material_slots.empty()) {
        mesh.material_slots.emplace_back(default_material);
    }
    return mesh;
}

contract::CurveGeometryDesc parse_blender_curve_geometry(
    yyjson_val *geometry,
    std::span<const std::byte> geometry_bin,
    const std::map<std::string, contract::MaterialId, std::less<>> &material_ids,
    contract::MaterialId default_material) {
    using contract::CurveGeometryDesc;

    const auto key_count = static_cast<std::size_t>(
        unsigned_number(member(geometry, "key_count")));
    const auto curve_count = static_cast<std::size_t>(
        unsigned_number(member(geometry, "curve_count")));

    CurveGeometryDesc curves;
    curves.name = text(member(geometry, "name"));
    curves.shape = parse_curve_shape(member(geometry, "shape"));
    curves.subdivisions = static_cast<std::uint32_t>(
        unsigned_number(member(geometry, "subdivisions"), 2u));
    curves.keys = read_packed_values<Vec4f>(
        geometry_bin, section_offset(geometry, "keys"), key_count);
    curves.curve_first_key = read_packed_values<std::uint32_t>(
        geometry_bin, section_offset(geometry, "curve_first_key"), curve_count);
    curves.curve_material_slots = read_packed_values<std::uint32_t>(
        geometry_bin,
        section_offset(geometry, "curve_material_slots"),
        curve_count);

    if (auto *default_uv = member(geometry, "default_uv_layer");
        default_uv != nullptr && yyjson_is_str(default_uv)) {
        curves.default_uv_layer = text(default_uv);
    }
    if (auto *uv_layers = member(geometry, "uv_layers");
        uv_layers != nullptr && yyjson_is_arr(uv_layers)) {
        yyjson_arr_iter uv_iterator = yyjson_arr_iter_with(uv_layers);
        while (auto *layer = yyjson_arr_iter_next(&uv_iterator)) {
            const auto name = text(member(layer, "name"));
            curves.uv_layers[name] = read_packed_values<Vec2f>(
                geometry_bin, section_offset(layer, "values"), curve_count);
        }
    }
    if (auto *color_attributes = member(geometry, "color_attributes");
        color_attributes != nullptr && yyjson_is_arr(color_attributes)) {
        yyjson_arr_iter color_iterator = yyjson_arr_iter_with(color_attributes);
        while (auto *attribute = yyjson_arr_iter_next(&color_iterator)) {
            const auto name = text(member(attribute, "name"));
            curves.color_attributes[name] = read_packed_values<Vec4f>(
                geometry_bin, section_offset(attribute, "values"), curve_count);
        }
    }

    curves.intercept = read_packed_values<float>(
        geometry_bin, section_offset(geometry, "intercept"), key_count);
    curves.length = read_packed_values<float>(
        geometry_bin, section_offset(geometry, "length"), curve_count);
    curves.random = read_packed_values<float>(
        geometry_bin, section_offset(geometry, "random"), curve_count);
    curves.cycles_curve_offset = optional_unsigned_number(
        member(member(geometry, "cycles_sync"), "curve_offset"));
    curves.cycles_segment_offset = optional_unsigned_number(
        member(member(geometry, "cycles_sync"), "segment_offset"));

    auto *slots = member(geometry, "material_slots");
    yyjson_arr_iter slot_iterator = yyjson_arr_iter_with(slots);
    while (auto *slot = yyjson_arr_iter_next(&slot_iterator)) {
        if (yyjson_is_null(slot)) {
            curves.material_slots.emplace_back(default_material);
            continue;
        }
        const auto iter = material_ids.find(text(slot));
        curves.material_slots.emplace_back(
            iter == material_ids.end() ? default_material : iter->second);
    }
    if (curves.material_slots.empty()) {
        curves.material_slots.emplace_back(default_material);
    }

    std::size_t segment_count = 0u;
    for (std::size_t curve = 0u; curve < curves.curve_first_key.size(); ++curve) {
        const auto begin = static_cast<std::size_t>(curves.curve_first_key[curve]);
        const auto end =
            curve + 1u < curves.curve_first_key.size()
                ? static_cast<std::size_t>(curves.curve_first_key[curve + 1u])
                : curves.keys.size();
        if (end > begin) {
            segment_count += end - begin - 1u;
        }
    }
    if (segment_count !=
        static_cast<std::size_t>(unsigned_number(
            member(geometry, "segment_count")))) {
        throw std::runtime_error("curve geometry has an invalid segment count");
    }
    return curves;
}

} // namespace psycles::adapter::detail
