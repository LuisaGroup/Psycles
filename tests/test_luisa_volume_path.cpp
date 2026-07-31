#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

[[nodiscard]] ShaderGraph volume_boundary_shader() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(
            node_type::transparent_bsdf,
            "Transparent boundary");
    const auto absorption =
        graph.add_node(
            node_type::volume_absorption,
            "Homogeneous absorption");
    static_cast<void>(graph.set_input(
        transparent,
        "Color",
        SocketValue::color(
            {1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        absorption,
        "Color",
        SocketValue::color(
            {0.2f, 0.5f, 0.8f})));
    static_cast<void>(graph.set_input(
        absorption,
        "Density",
        SocketValue::floating(0.5f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = transparent,
            .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = absorption,
            .socket = "Volume"});
    return graph;
}

[[nodiscard]] ShaderGraph heterogeneous_volume_shader() {
    auto graph = ShaderGraph{};
    const auto transparent =
        graph.add_node(
            node_type::transparent_bsdf,
            "Transparent boundary");
    const auto coordinates =
        graph.add_node(
            node_type::texture_coordinate,
            "Texture Coordinate");
    const auto scalar =
        graph.add_node(
            node_type::vector_to_scalar,
            "Generated to density");
    const auto absorption =
        graph.add_node(
            node_type::volume_absorption,
            "Heterogeneous absorption");
    static_cast<void>(graph.connect(
        {.node = coordinates,
         .socket = "Generated"},
        scalar,
        "Vector"));
    static_cast<void>(graph.connect(
        {.node = scalar,
         .socket = "Value"},
        absorption,
        "Density"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = transparent,
            .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = absorption,
            .socket = "Volume"});
    return graph;
}

[[nodiscard]] ShaderGraph world_shader() {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(
            node_type::emission,
            "White world");
    static_cast<void>(graph.set_input(
        emission,
        "Color",
        SocketValue::color(
            {1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        emission,
        "Strength",
        SocketValue::floating(1.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = emission,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] TriangleMeshDesc volume_box(
    MaterialId material) {
    TriangleMeshDesc mesh;
    mesh.name = "Cycles homogeneous volume box";
    mesh.positions = {
        {-2.0f, -2.0f, -2.0f},
        {2.0f, -2.0f, -2.0f},
        {2.0f, 2.0f, -2.0f},
        {-2.0f, 2.0f, -2.0f},
        {-2.0f, -2.0f, 2.0f},
        {2.0f, -2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f},
        {-2.0f, 2.0f, 2.0f}};
    mesh.triangles = {
        {0u, 3u, 2u},
        {0u, 2u, 1u},
        {4u, 5u, 6u},
        {4u, 6u, 7u},
        {0u, 1u, 5u},
        {0u, 5u, 4u},
        {3u, 7u, 6u},
        {3u, 6u, 2u},
        {0u, 4u, 7u},
        {0u, 7u, 3u},
        {1u, 2u, 6u},
        {1u, 6u, 5u}};
    mesh.material_slots = {material};
    mesh.triangle_material_slots.assign(
        mesh.triangles.size(), 0u);
    mesh.triangle_smooth.assign(
        mesh.triangles.size(), 0u);
    return mesh;
}

[[nodiscard]] SceneSnapshot make_scene() {
    constexpr MaterialId volume_material{1u};
    constexpr MaterialId world_material{2u};
    constexpr GeometryId geometry{3u};
    constexpr InstanceId instance{4u};
    constexpr CameraId camera{5u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        volume_material,
        MaterialDesc{
            .name =
                "Cycles homogeneous absorption boundary",
            .shader = volume_boundary_shader(),
            .cycles_shader_index = 0u});
    scene.materials.emplace(
        world_material,
        MaterialDesc{
            .name = "Cycles white world",
            .shader = world_shader(),
            .cycles_shader_index = 1u});
    scene.geometries.emplace(
        geometry,
        volume_box(volume_material));
    scene.instances.emplace(
        instance,
        InstanceDesc{
            .name = "Volume box",
            .geometry = geometry,
            .transform = {},
            .cycles_object_index = 0u});
    scene.cameras.emplace(
        camera,
        CameraDesc{
            .name = "Inside-volume orthographic camera",
            .projection =
                CameraProjection::orthographic,
            .transform = {},
            .sensor_fit =
                CameraSensorFit::automatic,
            .orthographic_scale = 1.0f,
            .near_clip = 0.1f,
            .far_clip = 100.0f});
    scene.active_camera = camera;
    scene.world_shader = world_material;
    scene.cycles_background_object_index = 1u;
    scene.world_sampling = WorldSampling::none;
    return scene;
}

[[nodiscard]] RenderSettings make_settings() {
    return {
        .full_extent = {
            .width = 4u,
            .height = 4u},
        .window = {},
        .seed = 11939u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::box,
        .filter_width = 1.0f,
        .pass_alpha_threshold = 0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 1u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 8u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold = 0.01f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::
                    forward_path_tracing},
        .passes = {
            {.kind = PassKind::combined,
             .name = "Combined",
             .channels = 4u},
            {.kind = PassKind::volume_direct,
             .name = "Volume Direct",
             .channels = 3u},
            {.kind = PassKind::volume_indirect,
             .name = "Volume Indirect",
             .channels = 3u},
            {.kind = PassKind::environment,
             .name = "Environment",
             .channels = 3u}}};
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 3.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    psycles::luisa_backend::LuisaPathTracerBackend
        renderer{
            std::move(device),
            {.next_event_estimation = false,
             .max_samples_per_dispatch = 1u}};
    auto heterogeneous = make_scene();
    heterogeneous.materials
        .at(MaterialId{1u})
        .shader = heterogeneous_volume_shader();
    const auto rejected =
        renderer.compile_scene(heterogeneous);
    auto found_heterogeneous_diagnostic = false;
    for (const auto &diagnostic :
         rejected.diagnostics) {
        found_heterogeneous_diagnostic =
            found_heterogeneous_diagnostic ||
            diagnostic.message.find(
                "spatially varying Volume closure") !=
                std::string::npos;
    }
    if (rejected.ok() ||
        !found_heterogeneous_diagnostic) {
        std::cerr
            << "heterogeneous volume was not rejected "
               "before homogeneous path compilation\n";
        return EXIT_FAILURE;
    }
    auto compilation =
        renderer.compile_scene(make_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic :
             compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }

    const auto settings = make_settings();
    auto session =
        renderer.create_session(
            *compilation.scene, settings);
    if (!session) {
        std::cerr
            << "could not create volume render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink sink;
    if (!session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            sink)) {
        std::cerr << "volume render failed\n";
        return EXIT_FAILURE;
    }

    const auto *combined =
        sink.find(PassKind::combined);
    const auto *environment =
        sink.find(PassKind::environment);
    const auto *volume_direct =
        sink.find(PassKind::volume_direct);
    const auto *volume_indirect =
        sink.find(PassKind::volume_indirect);
    if (combined == nullptr ||
        environment == nullptr ||
        volume_direct == nullptr ||
        volume_indirect == nullptr) {
        std::cerr
            << "volume path passes were not produced\n";
        return EXIT_FAILURE;
    }

    // Official Cycles 5.2.0 CPU, one sample, raw Combined pass. The camera
    // starts at its 0.1 near clip, so the finite in-box segment is 1.9 units.
    constexpr std::array cycles_combined{
        0.4676664173603058f,
        0.6218850612640381f,
        0.8269591331481934f,
        1.0f};
    const auto pixel_count =
        static_cast<std::size_t>(
            settings.full_extent.width) *
        settings.full_extent.height;
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel * combined->channels;
        const auto pass_base =
            pixel * environment->channels;
        for (auto channel = std::size_t{0u};
             channel < cycles_combined.size();
             ++channel) {
            if (!approximately_equal(
                    combined->pixels[
                        combined_base + channel],
                    cycles_combined[channel])) {
                std::cerr
                    << "Cycles volume-segment regression "
                       "failed on "
                    << backend << " pixel " << pixel
                    << " channel " << channel
                    << ": got "
                    << combined->pixels[
                           combined_base + channel]
                    << ", expected "
                    << cycles_combined[channel]
                    << '\n';
                return EXIT_FAILURE;
            }
        }
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            if (!approximately_equal(
                    environment->pixels[
                        pass_base + channel],
                    cycles_combined[channel]) ||
                !approximately_equal(
                    volume_direct->pixels[
                        pass_base + channel],
                    0.0f) ||
                !approximately_equal(
                    volume_indirect->pixels[
                        pass_base + channel],
                    0.0f)) {
                std::cerr
                    << "volume pass routing regression "
                       "failed on "
                    << backend << " pixel " << pixel
                    << " channel " << channel
                    << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
