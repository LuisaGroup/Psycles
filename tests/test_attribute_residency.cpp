#include "../src/luisa/path_tracer_attribute_residency.h"

#include <psycles/compiler/core_nodes.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;
using psycles::Vec2f;
using psycles::Vec4f;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

[[nodiscard]] ShaderGraph attribute_graph(std::uint64_t id) {
    ShaderGraph graph;
    const auto attribute = graph.add_node(
        node_type::vertex_color, "Attribute demand");
    const auto emission = graph.add_node(
        node_type::emission, "Attribute consumer");
    require(
        graph.set_property(
            attribute,
            "AttributeId",
            SocketValue::unsigned_integer(id)) &&
            graph.connect(
                {.node = attribute, .socket = "Color"},
                emission,
                "Color"),
        "could not build attribute graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph named_uv_graph(std::uint64_t id) {
    ShaderGraph graph;
    const auto coordinates = graph.add_node(
        node_type::texture_coordinate, "Named UV demand");
    const auto conversion = graph.add_node(
        node_type::vector_to_color, "UV conversion");
    const auto emission = graph.add_node(
        node_type::emission, "UV consumer");
    require(
        graph.set_property(
            coordinates,
            "UvMapNamed",
            SocketValue::boolean(true)) &&
            graph.set_property(
                coordinates,
                "UvMapId",
                SocketValue::unsigned_integer(id)) &&
            graph.connect(
                {.node = coordinates, .socket = "UV"},
                conversion,
                "Vector") &&
            graph.connect(
                {.node = conversion, .socket = "Color"},
                emission,
                "Color"),
        "could not build named UV graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph named_normal_graph(
    std::uint64_t id,
    std::string_view base) {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::normal_map, "Named tangent demand");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Normal consumer");
    require(
        graph.set_property(
            normal,
            "UvMapNamed",
            SocketValue::boolean(true)) &&
            graph.set_property(
                normal,
                "UvMapId",
                SocketValue::unsigned_integer(id)) &&
            graph.set_property(
                normal,
                "Base",
                SocketValue::string(std::string{base})) &&
            graph.connect(
                {.node = normal, .socket = "Normal"},
                diffuse,
                "Normal"),
        "could not build named tangent graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] MeshAttribute<Vec2f> triangle_uv() {
    return {
        .domain = MeshAttributeDomain::corner,
        .values = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f}}};
}

[[nodiscard]] MeshAttribute<Vec4f> triangle_tangent() {
    return {
        .domain = MeshAttributeDomain::corner,
        .values = {
            {1.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 0.0f, 1.0f}}};
}

[[nodiscard]] MeshAttribute<Vec4f> one_color() {
    return {
        .domain = MeshAttributeDomain::face,
        .values = {{1.0f, 0.5f, 0.25f, 1.0f}}};
}

void test_per_geometry_union_and_tangent_dependency_closure() {
    constexpr MaterialId uv_material{1u};
    constexpr MaterialId tangent_material{2u};
    constexpr MaterialId override_material{3u};
    constexpr MaterialId unreachable_material{4u};
    constexpr GeometryId geometry_id{10u};
    constexpr InstanceId instance_id{11u};
    constexpr std::string_view kept_uv = "KeptUV";
    constexpr std::string_view tangent_uv = "TangentUV";
    constexpr std::string_view dropped_uv = "DroppedUV";
    constexpr std::string_view override_color = "OverrideColor";
    constexpr std::string_view dropped_color = "DroppedColor";

    SceneSnapshot snapshot;
    snapshot.revision = 1u;
    snapshot.materials.emplace(
        uv_material,
        MaterialDesc{
            .name = "named UV",
            .shader = named_uv_graph(uv_attribute_id(kept_uv))});
    snapshot.materials.emplace(
        tangent_material,
        MaterialDesc{
            .name = "original tangent",
            .shader = named_normal_graph(
                uv_undisplaced_tangent_attribute_id(tangent_uv),
                "ORIGINAL")});
    snapshot.materials.emplace(
        override_material,
        MaterialDesc{
            .name = "override attribute",
            .shader = attribute_graph(attribute_id(override_color))});
    snapshot.materials.emplace(
        unreachable_material,
        MaterialDesc{
            .name = "unreachable attribute",
            .shader = attribute_graph(attribute_id(dropped_color))});

    TriangleMeshDesc geometry;
    geometry.name = "residency fixture";
    geometry.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}};
    geometry.triangles = {{0u, 1u, 2u}};
    geometry.material_slots = {uv_material, tangent_material};
    geometry.color_attributes.emplace(
        override_color, one_color());
    geometry.color_attributes.emplace(
        dropped_color, one_color());
    geometry.uv_layers.emplace(
        kept_uv, triangle_uv());
    geometry.uv_layers.emplace(
        tangent_uv, triangle_uv());
    geometry.uv_layers.emplace(
        dropped_uv, triangle_uv());
    geometry.uv_tangent_layers.emplace(
        tangent_uv, triangle_tangent());
    geometry.uv_tangent_layers.emplace(
        dropped_uv, triangle_tangent());
    geometry.pointiness_source.emplace();
    snapshot.geometries.emplace(geometry_id, std::move(geometry));
    snapshot.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "override user",
            .geometry = geometry_id,
            .material_overrides = {override_material}});

    MaterialLibrary materials;
    ShaderCompiler compiler{make_core_node_registry()};
    const auto update = materials.update(snapshot, compiler);
    require(update.committed, "material fixture did not compile");

    const auto plan = build_scene_attribute_residency_plan(
        snapshot, materials);
    const auto &resident = plan.geometry(geometry_id);
    require(
        resident.contains(uv_attribute_id(kept_uv)),
        "base-slot named UV was pruned");
    require(
        resident.contains(attribute_id(override_color)),
        "instance-override attribute was pruned");
    require(
        resident.contains(uv_attribute_id(tangent_uv)),
        "named tangent did not retain its Mikk UV dependency");
    require(
        resident.contains(
            uv_undisplaced_tangent_attribute_id(tangent_uv)),
        "ORIGINAL named tangent was pruned");
    require(
        !resident.contains(uv_tangent_attribute_id(tangent_uv)),
        "DISPLACED tangent was retained by an ORIGINAL-only material");
    require(
        !resident.contains(uv_attribute_id(dropped_uv)) &&
            !resident.contains(attribute_id(dropped_color)) &&
            !resident.contains(cycles_pointiness_attribute_id),
        "unqueried named attributes were retained");
    require(
        plan.source_binding_count == 10u &&
            plan.resident_binding_count == 4u,
        "binding residency census is not exact");
    require(
        plan.source_device_bytes == 416u &&
            plan.resident_device_bytes == 160u,
        "device-byte residency census is not exact");
}

void test_unknown_query_is_conservative_top() {
    ValueInstruction query{
        .operation = ValueOperation::attribute_color,
        .result_type = SocketType::color,
        .operands = {ValueExpressionId{}}};
    SurfaceProgram program{
        1u,
        {},
        {std::move(query)},
        {},
        {}};
    SurfaceParameterBlock parameters{program};
    const auto demand = collect_surface_attribute_demand(
        program, parameters);
    require(
        demand.all && demand.contains(0x123456789abcdef0ull),
        "unresolved runtime attribute ID did not raise demand to top");
}

}// namespace

int main() {
    test_per_geometry_union_and_tangent_dependency_closure();
    test_unknown_query_is_conservative_top();
    return EXIT_SUCCESS;
}
