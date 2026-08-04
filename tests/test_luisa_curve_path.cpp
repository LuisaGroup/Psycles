#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
    auto result = Mat4f{};
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] ShaderGraph hair_intercept_emission() {
    ShaderGraph graph;
    const auto hair = graph.add_node(node_type::hair_info, "Native Hair Info");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    static_cast<void>(graph.set_input(
        emission, "Color", SocketValue::color({0.25f, 0.5f, 1.0f})));
    static_cast<void>(graph.connect(
        {.node = hair, .socket = "Intercept"}, emission, "Strength"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph diffuse_shader() {
    ShaderGraph graph;
    const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    static_cast<void>(graph.set_input(
        diffuse, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        diffuse, "Roughness", SocketValue::floating(0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot make_emissive_curve_scene() {
    constexpr MaterialId material_id{1u};
    constexpr GeometryId geometry_id{2u};
    constexpr InstanceId instance_id{3u};
    constexpr CameraId camera_id{4u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        material_id,
        MaterialDesc{
            .name = "Hair Intercept emission",
            .shader = hair_intercept_emission(),
            .cycles_shader_index = 11u});
    scene.curve_geometries.emplace(
        geometry_id,
        CurveGeometryDesc{
            .name = "Native Cycles ribbon",
            .shape = CurveShape::ribbon,
            .subdivisions = 3u,
            .keys = {
                {-1.0f, 0.0f, 0.0f, 0.4f},
                {1.0f, 0.0f, 0.0f, 0.4f}},
            .curve_first_key = {0u},
            .material_slots = {material_id},
            .curve_material_slots = {0u},
            .intercept = {0.375f, 0.375f},
            .length = {2.0f},
            .random = {0.25f},
            .cycles_curve_offset = 23u,
            .cycles_segment_offset = 31u});
    scene.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "Native Cycles ribbon",
            .geometry = geometry_id,
            .transform = {},
            .cycles_object_index = 7u});
    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Ribbon camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .orthographic_scale = 0.1f,
            .near_clip = 0.01f,
            .far_clip = 10.0f});
    scene.active_camera = camera_id;
    return scene;
}

[[nodiscard]] SceneSnapshot make_shadow_scene(bool with_curve) {
    constexpr MaterialId receiver_material{10u};
    constexpr MaterialId blocker_material{11u};
    constexpr GeometryId receiver_geometry{12u};
    constexpr GeometryId blocker_geometry{13u};
    constexpr CameraId camera_id{14u};

    SceneSnapshot scene;
    scene.revision = with_curve ? 3u : 2u;
    scene.materials.emplace(
        receiver_material,
        MaterialDesc{
            .name = "Diffuse receiver",
            .shader = diffuse_shader(),
            .cycles_shader_index = 12u});
    scene.materials.emplace(
        blocker_material,
        MaterialDesc{
            .name = "Opaque ribbon blocker",
            .shader = diffuse_shader(),
            .cycles_shader_index = 13u});

    TriangleMeshDesc receiver;
    receiver.name = "Shadow receiver";
    receiver.positions = {
        {-2.0f, -2.0f, 0.0f},
        {2.0f, -2.0f, 0.0f},
        {2.0f, 2.0f, 0.0f},
        {-2.0f, 2.0f, 0.0f}};
    receiver.normals.values.assign(
        receiver.positions.size(), Vec3f{0.0f, 0.0f, 1.0f});
    receiver.triangles = {{0u, 1u, 2u}, {0u, 2u, 3u}};
    receiver.material_slots = {receiver_material};
    receiver.triangle_material_slots = {0u, 0u};
    receiver.triangle_smooth = {0u, 0u};
    receiver.cycles_primitive_offset = 40u;
    scene.geometries.emplace(receiver_geometry, std::move(receiver));
    scene.instances.emplace(
        InstanceId{15u},
        InstanceDesc{
            .name = "Shadow receiver",
            .geometry = receiver_geometry,
            .transform = {},
            .cycles_object_index = 8u});

    if (with_curve) {
        scene.curve_geometries.emplace(
            blocker_geometry,
            CurveGeometryDesc{
                .name = "Shadow-blocking Cycles ribbon",
                .shape = CurveShape::ribbon,
                .subdivisions = 3u,
                .keys = {
                    {0.5f, -1.0f, 1.0f, 0.2f},
                    {0.5f, 1.0f, 1.0f, 0.2f}},
                .curve_first_key = {0u},
                .material_slots = {blocker_material},
                .curve_material_slots = {0u},
                .intercept = {0.0f, 1.0f},
                .length = {2.0f},
                .random = {0.5f},
                .cycles_curve_offset = 50u,
                .cycles_segment_offset = 60u});
        scene.instances.emplace(
            InstanceId{16u},
            InstanceDesc{
                .name = "Shadow-blocking Cycles ribbon",
                .geometry = blocker_geometry,
                .transform = {},
                .cycles_object_index = 9u});
    }

    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Shadow camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .orthographic_scale = 0.02f,
            .near_clip = 0.01f,
            .far_clip = 10.0f});
    scene.active_camera = camera_id;
    scene.lights.emplace(
        LightId{17u},
        LightDesc{
            .name = "Shadow point light",
            .type = LightType::point,
            .transform = translated(1.0f, 0.0f, 2.0f),
            .color = {1.0f, 1.0f, 1.0f},
            .power = 20.0f,
            .size = 0.0f,
            .normalize = true,
            .is_sphere = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask = all_ray_visibility,
            .cycles_shader_index = 14u,
            .cycles_object_index = 10u});
    return scene;
}

[[nodiscard]] RenderSettings make_settings() {
    return {
        .full_extent = {.width = 1u, .height = 1u},
        .window = {},
        .seed = 0x6d2b79f5u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::box,
        .filter_width = 1.0f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 0u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 8u,
            .film_exposure = 1.0f,
            .reflective_caustics = true,
            .refractive_caustics = true},
        .passes = {{.kind = PassKind::combined,
                    .name = "Combined",
                    .light_group = {},
                    .channels = 4u}}};
}

[[nodiscard]] bool close(float actual, float expected) noexcept {
    return std::abs(actual - expected) <= 2.0e-5f;
}

[[nodiscard]] std::optional<std::array<float, 4u>> render_pixel(
    psycles::luisa_backend::LuisaPathTracerBackend &renderer,
    const SceneSnapshot &scene,
    std::string_view backend,
    std::string_view label) {
    auto compilation = renderer.compile_scene(scene);
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return std::nullopt;
    }
    auto session = renderer.create_session(*compilation.scene, make_settings());
    psycles::io::MemoryOutputSink sink;
    if (!session ||
        !session->render_samples(
            {.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, sink)) {
        std::cerr << label << " render failed on " << backend << '\n';
        return std::nullopt;
    }
    const auto *combined = sink.find(PassKind::combined);
    if (combined == nullptr || combined->pixels.size() != 4u) {
        std::cerr << label << " Combined pass has an invalid shape on "
                  << backend << '\n';
        return std::nullopt;
    }
    return std::array{
        combined->pixels[0u],
        combined->pixels[1u],
        combined->pixels[2u],
        combined->pixels[3u]};
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true, .max_samples_per_dispatch = 1u}};
    const auto emissive = render_pixel(
        renderer, make_emissive_curve_scene(), backend, "native curve path");
    if (!emissive) {
        return EXIT_FAILURE;
    }
    constexpr std::array expected{0.09375f, 0.1875f, 0.375f, 1.0f};
    for (auto channel = std::size_t{0u}; channel < expected.size(); ++channel) {
        if (!close((*emissive)[channel], expected[channel])) {
            std::cerr << "native Hair Info path mismatch on " << backend
                      << " channel " << channel << ": got "
                      << (*emissive)[channel] << ", expected "
                      << expected[channel] << '\n';
            return EXIT_FAILURE;
        }
    }

    const auto unblocked = render_pixel(
        renderer, make_shadow_scene(false), backend, "unblocked shadow");
    const auto blocked = render_pixel(
        renderer, make_shadow_scene(true), backend, "ribbon shadow");
    if (!unblocked || !blocked) {
        return EXIT_FAILURE;
    }
    for (auto channel = std::size_t{0u}; channel < 3u; ++channel) {
        if ((*unblocked)[channel] <= 0.01f ||
            !close((*blocked)[channel], 0.0f)) {
            std::cerr << "native ribbon shadow mismatch on " << backend
                      << " channel " << channel << ": unblocked="
                      << (*unblocked)[channel] << ", blocked="
                      << (*blocked)[channel] << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
