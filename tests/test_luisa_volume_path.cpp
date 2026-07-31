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

[[nodiscard]] Mat4f translated(
    float x,
    float y,
    float z) noexcept {
    auto result = Mat4f{};
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

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

[[nodiscard]] ShaderGraph volume_scatter_shader() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(
            node_type::transparent_bsdf,
            "Transparent boundary");
    const auto scatter =
        graph.add_node(
            node_type::volume_scatter,
            "Homogeneous isotropic scatter");
    static_cast<void>(graph.set_input(
        transparent,
        "Color",
        SocketValue::color(
            {1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        scatter,
        "Color",
        SocketValue::color(
            {1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        scatter,
        "Density",
        SocketValue::floating(0.5f)));
    static_cast<void>(graph.set_input(
        scatter,
        "Anisotropy",
        SocketValue::floating(0.0f)));
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

[[nodiscard]] ShaderGraph black_world_shader() {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(
            node_type::emission,
            "Black world");
    static_cast<void>(graph.set_input(
        emission,
        "Color",
        SocketValue::color(
            {0.0f, 0.0f, 0.0f})));
    static_cast<void>(graph.set_input(
        emission,
        "Strength",
        SocketValue::floating(0.0f)));
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

[[nodiscard]] SceneSnapshot make_direct_scene() {
    auto scene = make_scene();
    scene.revision = 2u;
    scene.materials
        .at(MaterialId{1u})
        .name =
        "Cycles homogeneous isotropic volume boundary";
    scene.materials
        .at(MaterialId{1u})
        .shader = volume_scatter_shader();
    scene.materials
        .at(MaterialId{2u})
        .name = "Cycles black world";
    scene.materials
        .at(MaterialId{2u})
        .shader = black_world_shader();
    scene.cycles_background_object_index = 2u;
    scene.lights.emplace(
        LightId{6u},
        LightDesc{
            .name = "Cycles unit distant light",
            .type = LightType::distant,
            .transform = {},
            .color = {1.0f, 1.0f, 1.0f},
            .power = 1.0f,
            .angle = 0.0f,
            .normalize = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask =
                all_ray_visibility,
            .cycles_shader_index = 2u,
            .cycles_object_index = 1u});
    return scene;
}

[[nodiscard]] SceneSnapshot
make_point_direct_scene() {
    auto scene = make_direct_scene();
    scene.revision = 3u;
    scene.materials
        .at(MaterialId{1u})
        .volume_sampling =
        VolumeSampling::
            multiple_importance;
    scene.lights.clear();
    scene.lights.emplace(
        LightId{6u},
        LightDesc{
            .name =
                "Cycles finite-volume point light",
            .type = LightType::point,
            .transform = translated(
                0.6f, -0.25f, -0.8f),
            .color = {1.0f, 1.0f, 1.0f},
            .power = 10.0f,
            .size = 0.0f,
            .normalize = true,
            .is_sphere = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask =
                all_ray_visibility,
            .cycles_shader_index = 2u,
            .cycles_object_index = 1u});
    return scene;
}

[[nodiscard]] SceneSnapshot
make_spot_direct_scene() {
    auto scene = make_direct_scene();
    scene.revision = 4u;
    scene.materials
        .at(MaterialId{1u})
        .volume_sampling =
        VolumeSampling::
            multiple_importance;
    scene.lights.clear();
    scene.lights.emplace(
        LightId{6u},
        LightDesc{
            .name =
                "Cycles finite-volume spot light",
            .type = LightType::spot,
            .transform = translated(
                0.2f, -0.1f, -0.4f),
            .color = {1.0f, 1.0f, 1.0f},
            .power = 10.0f,
            .size = 0.0f,
            .spot_angle = 0.9f,
            .spot_smooth = 0.35f,
            .normalize = true,
            .is_sphere = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask =
                all_ray_visibility,
            .cycles_shader_index = 2u,
            .cycles_object_index = 1u});
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

[[nodiscard]] RenderSettings make_direct_settings() {
    auto settings = make_settings();
    settings.integrator
        .direct_light_sampling =
        DirectLightSampling::
            multiple_importance_sampling;
    return settings;
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
            {.next_event_estimation = true,
             .max_samples_per_dispatch = 8u}};
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

    auto transparent_settings = settings;
    transparent_settings.transparent_background =
        true;
    auto transparent_session =
        renderer.create_session(
            *compilation.scene,
            transparent_settings);
    if (!transparent_session) {
        std::cerr
            << "could not create transparent volume "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        transparent_sink;
    if (!transparent_session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            transparent_sink)) {
        std::cerr
            << "transparent volume render failed\n";
        return EXIT_FAILURE;
    }
    const auto *transparent_combined =
        transparent_sink.find(
            PassKind::combined);
    const auto *transparent_environment =
        transparent_sink.find(
            PassKind::environment);
    if (transparent_combined == nullptr ||
        transparent_environment == nullptr) {
        std::cerr
            << "transparent volume passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
    // Cycles stores average(throughput) as raw Combined transparency and
    // converts it to alpha only after sample normalization. The environment
    // remains available as a pass, but transparent-film rays do not write it
    // to Combined RGB.
    constexpr auto cycles_transparent_alpha =
        0.361163139f;
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel *
            transparent_combined->channels;
        const auto pass_base =
            pixel *
            transparent_environment->channels;
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            if (!approximately_equal(
                    transparent_combined
                        ->pixels[
                            combined_base +
                            channel],
                    0.0f) ||
                !approximately_equal(
                    transparent_environment
                        ->pixels[
                            pass_base +
                            channel],
                    cycles_combined[channel])) {
                std::cerr
                    << "transparent volume routing "
                       "regression failed on "
                    << backend << " pixel " << pixel
                    << " channel " << channel
                    << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!approximately_equal(
                transparent_combined->pixels[
                    combined_base + 3u],
                cycles_transparent_alpha)) {
            std::cerr
                << "Cycles transparent volume alpha "
                   "regression failed on "
                << backend << " pixel " << pixel
                << ": got "
                << transparent_combined->pixels[
                       combined_base + 3u]
                << ", expected "
                << cycles_transparent_alpha
                << '\n';
            return EXIT_FAILURE;
        }
    }

    auto direct_compilation =
        renderer.compile_scene(
            make_direct_scene());
    if (!direct_compilation.ok()) {
        for (const auto &diagnostic :
             direct_compilation.diagnostics) {
            std::cerr << diagnostic.message
                      << '\n';
        }
        return EXIT_FAILURE;
    }
    const auto direct_settings =
        make_direct_settings();
    auto direct_session =
        renderer.create_session(
            *direct_compilation.scene,
            direct_settings);
    if (!direct_session) {
        std::cerr
            << "could not create volume-direct "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        direct_sink;
    if (!direct_session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            direct_sink)) {
        std::cerr
            << "volume-direct render failed\n";
        return EXIT_FAILURE;
    }
    const auto *direct_combined =
        direct_sink.find(
            PassKind::combined);
    const auto *direct_volume =
        direct_sink.find(
            PassKind::volume_direct);
    const auto *direct_volume_indirect =
        direct_sink.find(
            PassKind::volume_indirect);
    const auto *direct_environment =
        direct_sink.find(
            PassKind::environment);
    if (direct_combined == nullptr ||
        direct_volume == nullptr ||
        direct_volume_indirect == nullptr ||
        direct_environment == nullptr) {
        std::cerr
            << "volume-direct passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
    // Official Blender 5.2.0/Cycles CPU, Tabulated Sobol, one sample,
    // raw 32-bit Combined EXR. This scene exercises primary-ray VSPG's
    // empty-history defensive probability, distance sampling, isotropic
    // phase evaluation, distant-light NEE/MIS, and volume shadow
    // transmittance without any material pre-baking.
    constexpr std::array cycles_direct{
        0.015102289f,
        0.011529196f,
        0.010234529f,
        0.016950374f,
        0.016323633f,
        0.010089879f,
        0.017004959f,
        0.008192022f,
        0.016794045f,
        0.012923940f,
        0.014676524f,
        0.012400300f,
        0.009641428f,
        0.009475062f,
        0.015544447f,
        0.016437037f};
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel *
            direct_combined->channels;
        const auto pass_base =
            pixel *
            direct_volume->channels;
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto combined_value =
                direct_combined->pixels[
                    combined_base + channel];
            const auto direct_value =
                direct_volume->pixels[
                    pass_base + channel];
            if (!approximately_equal(
                    combined_value,
                    cycles_direct[pixel]) ||
                !approximately_equal(
                    direct_value,
                    cycles_direct[pixel]) ||
                !approximately_equal(
                    direct_volume_indirect
                        ->pixels[
                            pass_base +
                            channel],
                    0.0f) ||
                !approximately_equal(
                    direct_environment
                        ->pixels[
                            pass_base +
                            channel],
                    0.0f)) {
                std::cerr
                    << "Cycles volume-direct regression "
                       "failed on "
                    << backend << " pixel " << pixel
                    << " channel " << channel
                    << ": got combined "
                    << combined_value
                    << ", direct "
                    << direct_value
                    << ", expected "
                    << cycles_direct[pixel]
                    << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!approximately_equal(
                direct_combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "volume-direct alpha regression "
                   "failed on "
                << backend << " pixel " << pixel
                << '\n';
            return EXIT_FAILURE;
        }
    }

    auto point_compilation =
        renderer.compile_scene(
            make_point_direct_scene());
    if (!point_compilation.ok()) {
        for (const auto &diagnostic :
             point_compilation.diagnostics) {
            std::cerr << diagnostic.message
                      << '\n';
        }
        return EXIT_FAILURE;
    }
    auto point_session =
        renderer.create_session(
            *point_compilation.scene,
            direct_settings);
    if (!point_session) {
        std::cerr
            << "could not create point-volume "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        point_sink;
    if (!point_session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            point_sink)) {
        std::cerr
            << "point-volume render failed\n";
        return EXIT_FAILURE;
    }
    const auto *point_combined =
        point_sink.find(PassKind::combined);
    const auto *point_volume =
        point_sink.find(
            PassKind::volume_direct);
    if (point_combined == nullptr ||
        point_volume == nullptr) {
        std::cerr
            << "point-volume passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
    // Official Blender 5.2.0/Cycles CPU. The finite emitter is sampled once
    // for the equiangular reference point, MIS remaps the volume scatter
    // dimension before indirect free flight, and the same emitter/random.xy
    // pair is sampled again from the resulting direct collision point.
    constexpr std::array cycles_point_mis{
        0.0151212150f,
        0.0245463103f,
        0.0328574479f,
        0.0312816277f,
        0.0163814686f,
        0.0329394266f,
        0.0488485657f,
        0.0545895584f,
        0.0167177897f,
        0.0477123372f,
        0.1035088152f,
        0.3427364230f,
        0.0208319221f,
        0.0349823460f,
        0.0848462731f,
        0.1400486678f};
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel *
            point_combined->channels;
        const auto pass_base =
            pixel *
            point_volume->channels;
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto expected =
                cycles_point_mis[pixel];
            const auto combined_value =
                point_combined->pixels[
                    combined_base + channel];
            const auto direct_value =
                point_volume->pixels[
                    pass_base + channel];
            if (!approximately_equal(
                    combined_value,
                    expected) ||
                !approximately_equal(
                    direct_value,
                    expected)) {
                std::cerr
                    << "Cycles finite point-volume "
                       "regression failed on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << ": got "
                    << combined_value
                    << ", expected "
                    << expected << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!approximately_equal(
                point_combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "point-volume alpha regression "
                   "failed on "
                << backend << " pixel "
                << pixel << '\n';
            return EXIT_FAILURE;
        }
    }

    auto spot_compilation =
        renderer.compile_scene(
            make_spot_direct_scene());
    if (!spot_compilation.ok()) {
        for (const auto &diagnostic :
             spot_compilation.diagnostics) {
            std::cerr << diagnostic.message
                      << '\n';
        }
        return EXIT_FAILURE;
    }
    auto spot_session =
        renderer.create_session(
            *spot_compilation.scene,
            direct_settings);
    if (!spot_session) {
        std::cerr
            << "could not create spot-volume "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        spot_sink;
    if (!spot_session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            spot_sink)) {
        std::cerr
            << "spot-volume render failed\n";
        return EXIT_FAILURE;
    }
    const auto *spot_combined =
        spot_sink.find(PassKind::combined);
    const auto *spot_volume =
        spot_sink.find(
            PassKind::volume_direct);
    if (spot_combined == nullptr ||
        spot_volume == nullptr) {
        std::cerr
            << "spot-volume passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
    // Official Blender 5.2.0/Cycles CPU, one Tabulated Sobol sample. The
    // proposal uses spot_light_sample<true>, clips the volume interval to the
    // enclosing cone, and reuses PRNG_LIGHT.xy with spot_light_sample<false>
    // at the selected collision.
    constexpr std::array cycles_spot_mis{
        0.0f,
        0.0f,
        0.00207081134f,
        0.0f,
        0.0f,
        0.00470471708f,
        0.000760584546f,
        0.00427631568f,
        0.0f,
        0.00933369156f,
        0.205294624f,
        0.0373432599f,
        0.000263685535f,
        0.00348850037f,
        0.0109063750f,
        0.00531436736f};
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel *
            spot_combined->channels;
        const auto pass_base =
            pixel *
            spot_volume->channels;
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto expected =
                cycles_spot_mis[pixel];
            const auto combined_value =
                spot_combined->pixels[
                    combined_base + channel];
            const auto direct_value =
                spot_volume->pixels[
                    pass_base + channel];
            if (!approximately_equal(
                    combined_value,
                    expected) ||
                !approximately_equal(
                    direct_value,
                    expected)) {
                std::cerr
                    << "Cycles finite spot-volume "
                       "regression failed on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << ": got "
                    << combined_value
                    << ", expected "
                    << expected << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!approximately_equal(
                spot_combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "spot-volume alpha regression "
                   "failed on "
                << backend << " pixel "
                << pixel << '\n';
            return EXIT_FAILURE;
        }
    }

    auto guided_session =
        renderer.create_session(
            *direct_compilation.scene,
            direct_settings);
    if (!guided_session) {
        std::cerr
            << "could not create VSPG history "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        guided_sink;
    psycles::io::MemoryOutputSink
        guided_partial_sink;
    if (!guided_session->render_samples(
            {.first = 0u,
             .count = 3u,
             .offset = 0u,
             .total = 4u},
            guided_partial_sink) ||
        !guided_session->render_samples(
            {.first = 3u,
             .count = 1u,
             .offset = 0u,
             .total = 4u},
            guided_sink)) {
        std::cerr
            << "VSPG history render failed\n";
        return EXIT_FAILURE;
    }
    const auto *guided_combined =
        guided_sink.find(
            PassKind::combined);
    const auto *guided_volume =
        guided_sink.find(
            PassKind::volume_direct);
    const auto *guided_partial_combined =
        guided_partial_sink.find(
            PassKind::combined);
    if (guided_combined == nullptr ||
        guided_partial_combined == nullptr ||
        guided_volume == nullptr) {
        std::cerr
            << "VSPG history passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
    constexpr auto diagnostic_pixel = 14u;
    constexpr auto cycles_partial_diagnostic =
        0.010638036f;
    const auto partial_diagnostic =
        guided_partial_combined->pixels[
            diagnostic_pixel *
            guided_partial_combined->channels];
    if (!approximately_equal(
            partial_diagnostic,
            cycles_partial_diagnostic,
            1.0e-6f)) {
        std::cerr
            << "Cycles VSPG partial-history "
               "regression failed on "
            << backend << ": got "
            << partial_diagnostic
            << ", expected "
            << cycles_partial_diagnostic
            << '\n';
        return EXIT_FAILURE;
    }
    auto fused_session =
        renderer.create_session(
            *direct_compilation.scene,
            direct_settings);
    if (!fused_session) {
        std::cerr
            << "could not create fused VSPG "
               "render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink
        fused_sink;
    if (!fused_session->render_samples(
            {.first = 0u,
             .count = 4u,
             .offset = 0u,
             .total = 4u},
            fused_sink)) {
        std::cerr
            << "fused VSPG render failed\n";
        return EXIT_FAILURE;
    }
    const auto *fused_combined =
        fused_sink.find(
            PassKind::combined);
    const auto *fused_volume =
        fused_sink.find(
            PassKind::volume_direct);
    if (fused_combined == nullptr ||
        fused_volume == nullptr) {
        std::cerr
            << "fused VSPG passes were not "
               "produced\n";
        return EXIT_FAILURE;
    }
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    if (argc > 3) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                guided_sink.images(),
                argv[3],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write VSPG history "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
#endif
    // Official Blender 5.2.0/Cycles CPU, four Tabulated Sobol samples.
    // Cycles filters VSPG history after cumulative samples one and two,
    // then uses those quantized spatial guides for later primary segments.
    // The renderer dispatch limit is deliberately eight, so this also
    // regresses the independent power-of-two history update schedule.
    constexpr std::array cycles_guided{
        0.013084637f,
        0.012581268f,
        0.012184560f,
        0.013252687f,
        0.014589819f,
        0.012482958f,
        0.011810919f,
        0.011361255f,
        0.013536201f,
        0.012845095f,
        0.012335613f,
        0.011992678f,
        0.011454757f,
        0.011619815f,
        0.011370411f,
        0.012845693f};
    for (auto pixel = std::size_t{0u};
         pixel < pixel_count;
         ++pixel) {
        const auto combined_base =
            pixel *
            guided_combined->channels;
        const auto volume_base =
            pixel *
            guided_volume->channels;
        for (auto channel = std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto combined_value =
                guided_combined->pixels[
                    combined_base + channel];
            const auto volume_value =
                guided_volume->pixels[
                    volume_base + channel];
            const auto fused_combined_value =
                fused_combined->pixels[
                    combined_base + channel];
            const auto fused_volume_value =
                fused_volume->pixels[
                    volume_base + channel];
            if (!approximately_equal(
                    combined_value,
                    cycles_guided[pixel],
                    1.0e-6f) ||
                !approximately_equal(
                    volume_value,
                    cycles_guided[pixel],
                    1.0e-6f) ||
                !approximately_equal(
                    fused_combined_value,
                    cycles_guided[pixel],
                    1.0e-6f) ||
                !approximately_equal(
                    fused_volume_value,
                    cycles_guided[pixel],
                    1.0e-6f)) {
                std::cerr
                    << "Cycles VSPG history regression "
                       "failed on "
                    << backend << " pixel " << pixel
                    << " channel " << channel
                    << ": got combined "
                    << combined_value
                    << ", direct "
                    << volume_value
                    << ", fused combined "
                    << fused_combined_value
                    << ", fused direct "
                    << fused_volume_value
                    << ", expected "
                    << cycles_guided[pixel]
                    << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!approximately_equal(
                guided_combined->pixels[
                    combined_base + 3u],
                1.0f) ||
            !approximately_equal(
                fused_combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "VSPG history alpha regression "
                   "failed on "
                << backend << " pixel " << pixel
                << '\n';
            return EXIT_FAILURE;
        }
    }
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    if (argc > 5) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                spot_sink.images(),
                argv[5],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write spot-volume "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 4) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                point_sink.images(),
                argv[4],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write point-volume "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 2) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                direct_sink.images(),
                argv[2],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write volume-direct "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
#endif
    return EXIT_SUCCESS;
}
