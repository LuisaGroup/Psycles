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

[[nodiscard]] ShaderGraph
volume_scatter_shader() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(
            node_type::transparent_bsdf,
            "Transparent boundary");
    const auto scatter =
        graph.add_node(
            node_type::volume_scatter,
            "Homogeneous isotropic scatter");
    static_cast<void>(
        graph.set_input(
            transparent,
            "Color",
            SocketValue::color(
                {1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        graph.set_input(
            scatter,
            "Color",
            SocketValue::color(
                {1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        graph.set_input(
            scatter,
            "Density",
            SocketValue::floating(
                0.5f)));
    static_cast<void>(
        graph.set_input(
            scatter,
            "Anisotropy",
            SocketValue::floating(
                0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = transparent,
            .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = scatter,
            .socket = "Volume"});
    return graph;
}

[[nodiscard]] ShaderGraph
emission_shader() {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(
            node_type::emission,
            "Raw mesh emission closure");
    static_cast<void>(
        graph.set_input(
            emission,
            "Color",
            SocketValue::color(
                {1.0f, 1.0f, 1.0f})));
    static_cast<void>(
        graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(
                10.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = emission,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph
black_world_shader() {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(
            node_type::emission,
            "Black world");
    static_cast<void>(
        graph.set_input(
            emission,
            "Color",
            SocketValue::color(
                {0.0f, 0.0f, 0.0f})));
    static_cast<void>(
        graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(
                0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = emission,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] TriangleMeshDesc
volume_box(
    MaterialId material) {
    TriangleMeshDesc mesh;
    mesh.name =
        "Cycles homogeneous volume box";
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

[[nodiscard]] TriangleMeshDesc
emitter_triangle(
    MaterialId material) {
    TriangleMeshDesc mesh;
    mesh.name =
        "Cycles finite-volume triangle mesh";
    mesh.positions = {
        {0.6f, -0.6f, -0.8f},
        {1.4f, -0.6f, -0.8f},
        {1.0f, 0.2f, -0.8f}};
    mesh.triangles = {
        {0u, 1u, 2u}};
    mesh.material_slots = {material};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {0u};
    return mesh;
}

[[nodiscard]] Mat4f reflected_x() noexcept {
    Mat4f result;
    result.elements[0u] = -1.0f;
    return result;
}

[[nodiscard]] SceneSnapshot
make_scene(
    std::uint32_t emitter_visibility =
        all_ray_visibility) {
    constexpr MaterialId
        volume_material{1u};
    constexpr MaterialId
        world_material{2u};
    constexpr MaterialId
        emitter_material{3u};
    constexpr GeometryId
        volume_geometry{4u};
    constexpr GeometryId
        emitter_geometry{5u};
    constexpr InstanceId
        volume_instance{6u};
    constexpr InstanceId
        emitter_instance{7u};
    constexpr CameraId camera{8u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        volume_material,
        MaterialDesc{
            .name =
                "Cycles homogeneous isotropic volume boundary",
            .shader =
                volume_scatter_shader(),
            .volume_sampling =
                VolumeSampling::
                    multiple_importance,
            .cycles_shader_index =
                0u});
    scene.materials.emplace(
        world_material,
        MaterialDesc{
            .name =
                "Cycles black world",
            .shader =
                black_world_shader(),
            .cycles_shader_index =
                1u});
    scene.materials.emplace(
        emitter_material,
        MaterialDesc{
            .name =
                "Cycles raw mesh emission closure",
            .shader =
                emission_shader(),
            .emission_sampling =
                EmissionSampling::front,
            .cycles_shader_index =
                2u});
    scene.geometries.emplace(
        volume_geometry,
        volume_box(
            volume_material));
    scene.geometries.emplace(
        emitter_geometry,
        emitter_triangle(
            emitter_material));
    scene.instances.emplace(
        volume_instance,
        InstanceDesc{
            .name = "Volume box",
            .geometry =
                volume_geometry,
            .transform = {},
            .cycles_object_index =
                0u});
    scene.instances.emplace(
        emitter_instance,
        InstanceDesc{
            .name =
                "Negative-scale mesh emitter",
            .geometry =
                emitter_geometry,
            .transform =
                reflected_x(),
            .visibility_mask =
                emitter_visibility,
            .cycles_object_index =
                1u});
    scene.cameras.emplace(
        camera,
        CameraDesc{
            .name =
                "Inside-volume orthographic camera",
            .projection =
                CameraProjection::
                    orthographic,
            .transform = {},
            .sensor_fit =
                CameraSensorFit::
                    automatic,
            .orthographic_scale =
                1.0f,
            .near_clip = 0.1f,
            .far_clip = 100.0f});
    scene.active_camera = camera;
    scene.world_shader =
        world_material;
    scene.world_sampling =
        WorldSampling::none;
    scene.cycles_background_object_index =
        2u;
    return scene;
}

[[nodiscard]] RenderSettings
make_settings() {
    return {
        .full_extent = {
            .width = 4u,
            .height = 4u},
        .window = {},
        .seed = 11939u,
        .transparent_background = false,
        .pixel_filter =
            PixelFilter::box,
        .filter_width = 1.0f,
        .pass_alpha_threshold =
            0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces =
                1u,
            .volume_bounces = 0u,
            .transparent_min_bounces =
                0u,
            .transparent_max_bounces =
                8u,
            .sample_clamp_direct =
                0.0f,
            .sample_clamp_indirect =
                0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold =
                0.01f,
            .reflective_caustics =
                true,
            .refractive_caustics =
                true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::
                    multiple_importance_sampling},
        .passes = {
            {.kind = PassKind::combined,
             .name = "Combined",
             .channels = 4u},
            {.kind =
                 PassKind::volume_direct,
             .name = "Volume Direct",
             .channels = 3u}}};
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    // Preserve native backend math; structural visibility and pass routing
    // remain separately asserted by this fixture.
    float tolerance = 1.0e-5f) noexcept {
    return std::abs(actual - expected) <=
           tolerance;
}

[[nodiscard]] bool render_scene(
    psycles::luisa_backend::
        LuisaPathTracerBackend &renderer,
    SceneSnapshot scene,
    const RenderSettings &settings,
    psycles::io::MemoryOutputSink
        &sink) {
    const auto compilation =
        renderer.compile_scene(scene);
    if (!compilation.ok()) {
        for (const auto &diagnostic :
             compilation.diagnostics) {
            std::cerr
                << diagnostic.message
                << '\n';
        }
        return false;
    }
    auto session =
        renderer.create_session(
            *compilation.scene,
            settings);
    return session &&
           session->render_samples(
               {.first = 0u,
                .count = 1u,
                .offset = 0u,
                .total = 1u},
               sink);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1
                ? argv[1]
                : "fallback"};
    luisa::compute::Context
        context{argv[0]};
    auto device =
        context.create_device(
            backend);
    psycles::luisa_backend::
        LuisaPathTracerBackend renderer{
            std::move(device),
            {.next_event_estimation =
                 true,
             .max_samples_per_dispatch =
                 8u}};
    const auto settings =
        make_settings();
    psycles::io::MemoryOutputSink
        sink;
    if (!render_scene(
            renderer,
            make_scene(),
            settings,
            sink)) {
        std::cerr
            << "mesh-volume render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *combined =
        sink.find(
            PassKind::combined);
    const auto *volume =
        sink.find(
            PassKind::volume_direct);
    if (combined == nullptr ||
        volume == nullptr) {
        std::cerr
            << "mesh-volume passes missing on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    // Official Cycles main b82c3f0, one Tabulated Sobol sample. This fixture
    // uses a raw Emission closure on one FRONT-sampled triangle under a
    // negative-determinant instance. It pins triangle_light_sample<true>,
    // plane-half-space clipping, triangle_light_sample<false>, raw shader
    // evaluation, phase/light MIS, and target-primitive shadow exclusion.
    constexpr std::array cycles{
        0.016576273f,
        0.012104260f,
        0.003122555f,
        0.004680551f,
        0.024927052f,
        0.017955169f,
        0.016594326f,
        0.001506036f,
        0.044854816f,
        0.019945480f,
        0.010872677f,
        0.004508528f,
        0.065161705f,
        0.022143248f,
        0.013146726f,
        0.006225511f};
    auto passed = true;
    for (auto pixel = std::size_t{0u};
         pixel < cycles.size();
         ++pixel) {
        const auto combined_base =
            pixel *
            combined->channels;
        const auto volume_base =
            pixel *
            volume->channels;
        for (auto channel =
                 std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto combined_value =
                combined->pixels[
                    combined_base +
                    channel];
            const auto volume_value =
                volume->pixels[
                    volume_base +
                    channel];
            if (!approximately_equal(
                    combined_value,
                    cycles[pixel]) ||
                !approximately_equal(
                    volume_value,
                    cycles[pixel])) {
                std::cerr
                    << "Cycles mesh-volume regression failed on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << ": got "
                    << combined_value
                    << " (volume "
                    << volume_value
                    << "), expected "
                    << cycles[pixel]
                    << '\n';
                passed = false;
            }
        }
        if (!approximately_equal(
                combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "mesh-volume alpha failed on "
                << backend << " pixel "
                << pixel << '\n';
            passed = false;
        }
    }

    psycles::io::MemoryOutputSink
        hidden_sink;
    const auto without_scatter =
        all_ray_visibility &
        ~visibility_bit(
            RayVisibility::
                volume_scatter);
    if (!render_scene(
            renderer,
            make_scene(
                without_scatter),
            settings,
            hidden_sink)) {
        std::cerr
            << "hidden mesh-volume render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *hidden_volume =
        hidden_sink.find(
            PassKind::volume_direct);
    if (hidden_volume == nullptr) {
        return EXIT_FAILURE;
    }
    for (const auto value :
         hidden_volume->pixels) {
        if (value != 0.0f) {
            std::cerr
                << "volume-scatter visibility leak on "
                << backend << ": got "
                << value << '\n';
            passed = false;
            break;
        }
    }

#if defined(PSYCLES_WITH_OPENIMAGEIO)
    if (argc > 2) {
        std::string error;
        if (!psycles::io::
                write_multilayer_exr(
                    sink.images(),
                    argv[2],
                    "ViewLayer",
                    &error)) {
            std::cerr
                << "could not write mesh-volume EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
#endif
    return passed
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
