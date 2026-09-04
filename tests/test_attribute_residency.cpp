#include "../src/luisa/path_tracer_attribute_residency.h"
#include "../src/luisa/path_tracer_scene_geometry.h"

#include <psycles/compiler/core_nodes.h>

#include <cstdlib>
#include <iostream>
#include <string>
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
        node_type::attribute, "Attribute demand");
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

[[nodiscard]] ShaderGraph named_uv_graph(std::string_view attribute) {
    ShaderGraph graph;
    const auto coordinates = graph.add_node(
        node_type::uv_map, "Named UV demand");
    const auto to_vector = graph.add_node(
        node_type::point_to_vector, "UV point conversion");
    const auto to_color = graph.add_node(
        node_type::vector_to_color, "UV conversion");
    const auto emission = graph.add_node(
        node_type::emission, "UV consumer");
    require(
        graph.set_property(
            coordinates,
            "Attribute",
            SocketValue::string(std::string{attribute})) &&
            graph.set_property(
                coordinates,
                "AttributeId",
                SocketValue::unsigned_integer(uv_attribute_id(attribute))) &&
            graph.connect(
                {.node = coordinates, .socket = "UV"},
                to_vector,
                "Point") &&
            graph.connect(
                {.node = to_vector, .socket = "Vector"},
                to_color,
                "Vector") &&
            graph.connect(
                {.node = to_color, .socket = "Color"},
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

void test_named_vector_displacement_demand() {
    constexpr std::string_view uv_map = "VectorDisplacementUV";
    const auto tangent_id =
        uv_undisplaced_tangent_attribute_id(uv_map);
    ShaderGraph graph;
    const auto displacement = graph.add_node(
        node_type::vector_displacement,
        "Named vector displacement demand");
    const auto emission = graph.add_node(
        node_type::emission,
        "Vector displacement surface");
    require(
        graph.set_input(
            displacement,
            "Vector",
            SocketValue::color({0.75f, 0.25f, 0.5f})) &&
            graph.set_property(
                displacement,
                "Attribute",
                SocketValue::string(std::string{uv_map})) &&
            graph.set_property(
                displacement,
                "AttributeNamed",
                SocketValue::boolean(true)) &&
            graph.set_property(
                displacement,
                "AttributeId",
                SocketValue::unsigned_integer(tangent_id)),
        "could not build named vector displacement graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::displacement,
        OutputRef{.node = displacement, .socket = "Displacement"});

    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(graph);
    require(shader.ok(), "named vector displacement graph did not compile");
    const auto surface = compile_surface_program(*shader.program);
    require(surface.ok(), "named vector displacement graph did not lower");
    const SurfaceParameterBlock parameters{*surface.program};
    const auto demand = collect_surface_attribute_demand(
        *surface.program, parameters);
    require(
        !demand.all && demand.contains(tangent_id),
        "named vector displacement tangent was not retained");
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
        .values = {
            {1.0f, 0.5f, 0.25f, 1.0f},
            {1.0f, 0.5f, 0.25f, 1.0f}}};
}

void test_per_geometry_union_and_tangent_dependency_closure() {
    constexpr MaterialId uv_material{1u};
    constexpr MaterialId tangent_material{2u};
    constexpr MaterialId override_material{3u};
    constexpr MaterialId unreachable_material{4u};
    constexpr GeometryId geometry_id{10u};
    constexpr InstanceId base_instance_id{11u};
    constexpr InstanceId override_instance_id{12u};
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
            .shader = named_uv_graph(kept_uv)});
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
    geometry.triangles = {
        {0u, 1u, 2u},
        {0u, 1u, 2u}};
    geometry.material_slots = {uv_material, tangent_material};
    geometry.triangle_material_slots = {0u, 1u};
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
        base_instance_id,
        InstanceDesc{
            .name = "base user",
            .geometry = geometry_id});
    snapshot.instances.emplace(
        override_instance_id,
        InstanceDesc{
            .name = "override user",
            .geometry = geometry_id,
            .material_overrides = {override_material}});

    MaterialLibrary materials;
    ShaderCompiler compiler{make_core_node_registry()};
    const auto reachability =
        build_scene_material_reachability(snapshot);
    const auto update = materials.update(
        snapshot, compiler, reachability.shader_materials);
    if (!update.committed) {
        for (const auto &diagnostic : update.diagnostics) {
            std::cerr << "material " << diagnostic.material.value
                      << ": " << diagnostic.message << '\n';
        }
    }
    require(update.committed, "material fixture did not compile");
    require(
        materials.materials().size() == 3u &&
            materials.find(unreachable_material) == nullptr,
        "unreachable material entered the compiled attribute domain");

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
        // Source: 2 colors * 2 faces + 3 UVs * 3 corners +
        // 2 tangent layers * 2 variants * 6 corners + 3 pointiness values
        // = 640 B. Resident: 1 color * 2 faces + 2 UVs * 3 corners +
        // 1 original-tangent variant * 6 corners = 224 B.
        plan.source_device_bytes == 640u &&
            plan.resident_device_bytes == 224u,
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

void test_curve_domain_attribute_residency() {
    constexpr MaterialId uv_material{21u};
    constexpr MaterialId color_material{22u};
    constexpr MaterialId unreachable_material{23u};
    constexpr GeometryId geometry_id{24u};
    constexpr std::string_view kept_uv = "CurveRootUV";
    constexpr std::string_view dropped_uv = "UnusedCurveUV";
    constexpr std::string_view kept_color = "CurveRootColor";
    constexpr std::string_view dropped_color = "UnusedCurveColor";

    SceneSnapshot snapshot;
    snapshot.materials.emplace(
        uv_material,
        MaterialDesc{
            .name = "curve UV",
            .shader = named_uv_graph(kept_uv)});
    snapshot.materials.emplace(
        color_material,
        MaterialDesc{
            .name = "curve color",
            .shader = attribute_graph(attribute_id(kept_color))});
    snapshot.materials.emplace(
        unreachable_material,
        MaterialDesc{
            .name = "unreachable curve UV",
            .shader = named_uv_graph(dropped_uv)});
    snapshot.curve_geometries.emplace(
        geometry_id,
        CurveGeometryDesc{
            .name = "curve UV residency fixture",
            .keys = {{0.0f, 0.0f, 0.0f, 0.1f},
                     {0.0f, 0.0f, 1.0f, 0.1f},
                     {1.0f, 0.0f, 0.0f, 0.1f},
                     {1.0f, 0.0f, 1.0f, 0.1f}},
            .curve_first_key = {0u, 2u},
            .material_slots = {uv_material, color_material},
            .curve_material_slots = {0u, 1u},
            .default_uv_layer = std::string{kept_uv},
            .uv_layers = {
                {std::string{kept_uv}, {{0.1f, 0.2f}, {0.3f, 0.4f}}},
                {std::string{dropped_uv}, {{0.5f, 0.6f}, {0.7f, 0.8f}}}},
            .color_attributes = {
                {std::string{kept_color},
                 {{0.1f, 0.2f, 0.3f, 1.0f},
                  {0.4f, 0.5f, 0.6f, 1.0f}}},
                {std::string{dropped_color},
                 {{0.7f, 0.8f, 0.9f, 1.0f},
                  {0.2f, 0.3f, 0.4f, 1.0f}}}}});
    snapshot.instances.emplace(
        InstanceId{25u},
        InstanceDesc{.name = "curve UV user", .geometry = geometry_id});

    MaterialLibrary materials;
    ShaderCompiler compiler{make_core_node_registry()};
    const auto reachability = build_scene_material_reachability(snapshot);
    require(
        materials.update(snapshot, compiler, reachability.shader_materials)
            .committed,
        "curve UV material did not compile");
    const auto plan = build_scene_attribute_residency_plan(snapshot, materials);
    const auto &resident = plan.geometry(geometry_id);
    require(
        resident.contains(uv_attribute_id(kept_uv)) &&
            resident.contains(attribute_id(kept_color)) &&
            !resident.contains(uv_attribute_id(dropped_uv)) &&
            !resident.contains(attribute_id(dropped_color)),
        "curve-domain attribute reachability was not exact");
    require(
        plan.source_binding_count == 4u &&
            plan.resident_binding_count == 2u &&
            plan.source_device_bytes == 96u &&
            plan.resident_device_bytes == 48u,
        "curve-domain attribute residency census is not exact");
}

}// namespace

int main() {
    test_per_geometry_union_and_tangent_dependency_closure();
    test_curve_domain_attribute_residency();
    test_unknown_query_is_conservative_top();
    test_named_vector_displacement_demand();
    return EXIT_SUCCESS;
}
