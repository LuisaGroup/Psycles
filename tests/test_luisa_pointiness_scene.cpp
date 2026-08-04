#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/path_tracer.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

[[nodiscard]] ShaderGraph pointiness_shader() {
    ShaderGraph graph;
    const auto geometry =
        graph.add_node(node_type::geometry, "Raw Geometry");
    const auto emission =
        graph.add_node(node_type::emission, "Emission");
    static_cast<void>(graph.set_input(
        emission,
        "Color",
        SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.connect(
        {.node = geometry, .socket = "Pointiness"},
        emission,
        "Strength"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph normal_only_geometry_shader() {
    ShaderGraph graph;
    const auto geometry =
        graph.add_node(node_type::geometry, "Raw Geometry");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    static_cast<void>(graph.connect(
        {.node = geometry, .socket = "Normal"},
        diffuse,
        "Normal"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot pointiness_scene(
    ShaderGraph shader,
    const bool include_source,
    const std::uint64_t revision) {
    constexpr MaterialId material_id{1u};
    constexpr GeometryId geometry_id{2u};
    constexpr InstanceId instance_id{3u};
    constexpr CameraId camera_id{4u};

    SceneSnapshot scene;
    scene.revision = revision;
    scene.materials.emplace(
        material_id,
        MaterialDesc{
            .name = "Pointiness material",
            .shader = std::move(shader)});

    TriangleMeshDesc mesh;
    mesh.name = "Pointiness triangle";
    mesh.positions = {
        {-1.0f, -1.0f, 0.0f},
        {1.0f, -1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}};
    mesh.normals.values.assign(
        mesh.positions.size(),
        Vec3f{0.0f, 0.0f, 1.0f});
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {material_id};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {1u};
    mesh.triangle_random_per_island = {0.0f};
    if (include_source) {
        mesh.pointiness_source = MeshPointinessSource{
            .point_normals = mesh.normals.values,
            .edges = {{0u, 1u}, {1u, 2u}, {2u, 0u}}};
    }
    scene.geometries.emplace(geometry_id, std::move(mesh));
    scene.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "Pointiness triangle",
            .geometry = geometry_id,
            .transform = {}});
    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Pointiness camera",
            .projection = CameraProjection::orthographic,
            .transform = {},
            .orthographic_scale = 2.0f});
    scene.active_camera = camera_id;
    return scene;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device), {}};

    const auto rejected = renderer.compile_scene(
        pointiness_scene(pointiness_shader(), false, 1u));
    if (rejected.ok()) {
        std::cerr << "missing Pointiness source was accepted on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    auto named_diagnostic = false;
    for (const auto &diagnostic : rejected.diagnostics) {
        named_diagnostic |=
            diagnostic.message.find("Geometry.Pointiness") !=
            std::string::npos;
    }
    if (!named_diagnostic) {
        std::cerr << "missing Pointiness source has no named diagnostic on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    // Blender Geometry nodes expose every output in the raw graph. Using
    // Normal must not make the dead Pointiness output an attribute demand.
    // This is the topology found in Barbershop's Razor_Blade_Wood.001.
    const auto normal_only = renderer.compile_scene(
        pointiness_scene(
            normal_only_geometry_shader(), false, 2u));
    if (!normal_only.ok()) {
        for (const auto &diagnostic : normal_only.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        std::cerr
            << "dead Geometry.Pointiness output required a source on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto accepted = renderer.compile_scene(
        pointiness_scene(pointiness_shader(), true, 3u));
    if (!accepted.ok()) {
        for (const auto &diagnostic : accepted.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        std::cerr << "valid Pointiness source was rejected on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
