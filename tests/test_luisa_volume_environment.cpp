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
#include <utility>

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
environment_shader() {
    ShaderGraph graph;
    const auto coordinates =
        graph.add_node(
            node_type::texture_coordinate,
            "World Texture Coordinate");
    const auto point_to_vector =
        graph.add_node(
            node_type::point_to_vector,
            "World Generated point to vector");
    const auto gradient =
        graph.add_node(
            node_type::gradient_texture,
            "Linear world gradient");
    const auto remap =
        graph.add_node(
            node_type::math,
            "0.25 * gradient + 0.5");
    const auto gray =
        graph.add_node(
            node_type::scalar_to_color,
            "Gradient to color");
    const auto background =
        graph.add_node(
            node_type::emission,
            "Raw world Background closure");
    static_cast<void>(
        graph.set_property(
            gradient,
            "GradientType",
            SocketValue::string(
                "LINEAR")));
    static_cast<void>(
        graph.set_property(
            remap,
            "Operation",
            SocketValue::string(
                "MULTIPLY_ADD")));
    static_cast<void>(
        graph.set_input(
            remap,
            "B",
            SocketValue::floating(
                0.25f)));
    static_cast<void>(
        graph.set_input(
            remap,
            "C",
            SocketValue::floating(
                0.5f)));
    static_cast<void>(
        graph.set_input(
            background,
            "Strength",
            SocketValue::floating(
                1.0f)));
    static_cast<void>(
        graph.connect(
            {.node = coordinates,
             .socket = "Generated"},
            point_to_vector,
            "Point"));
    static_cast<void>(
        graph.connect(
            {.node = point_to_vector,
             .socket = "Vector"},
            gradient,
            "Vector"));
    static_cast<void>(
        graph.connect(
            {.node = gradient,
             .socket = "Factor"},
            remap,
            "A"));
    static_cast<void>(
        graph.connect(
            {.node = remap,
             .socket = "Value"},
            gray,
            "Value"));
    static_cast<void>(
        graph.connect(
            {.node = gray,
             .socket = "Color"},
            background,
            "Color"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = background,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph
constant_environment_shader() {
    ShaderGraph graph;
    const auto background =
        graph.add_node(
            node_type::emission,
            "Constant raw world Background closure");
    static_cast<void>(
        graph.set_input(
            background,
            "Color",
            SocketValue::color(
                {0.8f, 0.4f, 0.2f})));
    static_cast<void>(
        graph.set_input(
            background,
            "Strength",
            SocketValue::floating(
                1.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = background,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph
bump_feature_mask_environment_shader() {
    ShaderGraph graph;
    const auto displacement =
        graph.add_node(
            node_type::vector_displacement,
            "World Vector Displacement feature-mask oracle");
    const auto color =
        graph.add_node(
            node_type::vector_to_color,
            "World displacement to color");
    const auto background =
        graph.add_node(
            node_type::emission,
            "World displacement emission");
    const auto configured =
        graph.set_input(
            displacement,
            "Vector",
            SocketValue::color(
                {1.0f, 0.0f, 0.0f})) &&
        graph.set_input(
            displacement,
            "Midlevel",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            displacement,
            "Scale",
            SocketValue::floating(1.0f)) &&
        graph.set_property(
            displacement,
            "Space",
            SocketValue::string("WORLD")) &&
        graph.set_property(
            displacement,
            "AttributeNamed",
            SocketValue::boolean(false)) &&
        graph.set_property(
            displacement,
            "AttributeId",
            SocketValue::unsigned_integer(0u)) &&
        graph.connect(
            {.node = displacement,
             .socket = "Displacement"},
            color,
            "Vector") &&
        graph.connect(
            {.node = color,
             .socket = "Color"},
            background,
            "Color") &&
        graph.set_input(
            background,
            "Strength",
            SocketValue::floating(1.0f));
    if (!configured) {
        std::abort();
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = background,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] TriangleMeshDesc
volume_box(MaterialId material) {
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

[[nodiscard]] SceneSnapshot
make_scene(
    std::uint32_t world_visibility =
        all_ray_visibility,
    bool spatial_world = true) {
    constexpr MaterialId
        volume_material{1u};
    constexpr MaterialId
        world_material{2u};
    constexpr GeometryId
        volume_geometry{3u};
    constexpr InstanceId
        volume_instance{4u};
    constexpr CameraId camera{5u};

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
                "Cycles raw environment closure",
            .shader =
                spatial_world
                    ? environment_shader()
                    : constant_environment_shader(),
            .cycles_shader_index =
                1u});
    scene.geometries.emplace(
        volume_geometry,
        volume_box(
            volume_material));
    scene.instances.emplace(
        volume_instance,
        InstanceDesc{
            .name = "Volume box",
            .geometry =
                volume_geometry,
            .transform = {},
            .cycles_object_index =
                0u});
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
        WorldSampling::manual;
    scene.world_sample_map_resolution =
        4u;
    scene.world_visibility_mask =
        world_visibility;
    scene.cycles_background_object_index =
        1u;
    return scene;
}

[[nodiscard]] SceneSnapshot
bump_feature_mask_scene() {
    constexpr MaterialId world_material{2u};
    constexpr CameraId camera{5u};
    auto scene = make_scene();
    scene.materials.at(world_material).shader =
        bump_feature_mask_environment_shader();

    // Column-major camera-to-world transform. Local -Z maps to the
    // non-axis-aligned world ray normalize((1, 2, 3)), ruling out an
    // accidental axis-basis zero. Cycles 5.2.1 still returns exact zero:
    // KERNEL_FEATURE_NODE_MASK_SURFACE_BACKGROUND excludes NODE_BUMP, and
    // svm_node_vector_displacement writes a zero vector in that branch.
    scene.cameras.at(camera).transform.elements = {
        0.89442719f, -0.44721359f, 0.0f, 0.0f,
        -0.35856858f, -0.71713716f, 0.59761430f, 0.0f,
        -0.26726124f, -0.53452248f, -0.80178373f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};
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
    float tolerance = 8.0e-6f) noexcept {
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

[[nodiscard]] bool all_zero(
    const psycles::io::PassImage
        &image) noexcept {
    for (const auto value :
         image.pixels) {
        if (value != 0.0f) {
            return false;
        }
    }
    return true;
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
            << "environment-volume render failed on "
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
            << "environment-volume passes missing on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    // Official Cycles main b82c3f0, one Tabulated Sobol sample. The source
    // world is a raw Generated -> Linear Gradient -> Multiply Add ->
    // Background closure under MANUAL 4x2 importance-map sampling. These
    // arrays jointly pin the infinite-emitter segment dummy, distance
    // sampling, collision-point direction sample, selection and directional
    // PDFs, raw closure evaluation, phase MIS, and forward background
    // transport.
    constexpr std::array cycles_combined{
        0.446771294f,
        0.191449106f,
        0.492332697f,
        0.501667857f,
        0.450594395f,
        0.248026997f,
        0.196257144f,
        0.459130287f,
        0.449371487f,
        0.471367568f,
        0.444093287f,
        0.435745567f,
        0.133926883f,
        0.154041812f,
        0.094614848f,
        0.179454282f};
    constexpr std::array cycles_volume{
        0.036814660f,
        0.191449106f,
        0.082376063f,
        0.091711238f,
        0.040637758f,
        0.248026997f,
        0.196257144f,
        0.049173657f,
        0.039414864f,
        0.061410930f,
        0.034136653f,
        0.025788946f,
        0.133926883f,
        0.154041812f,
        0.094614848f,
        0.179454282f};
    auto passed = true;
    for (auto pixel = std::size_t{0u};
         pixel <
         cycles_combined.size();
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
            const auto expected_combined =
                cycles_combined[pixel];
            const auto expected_volume =
                cycles_volume[pixel];
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
                    expected_combined) ||
                !approximately_equal(
                    volume_value,
                    expected_volume)) {
                std::cerr
                    << "Cycles environment-volume regression failed on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << ": got "
                    << combined_value
                    << " (volume "
                    << volume_value
                    << "), expected "
                    << expected_combined
                    << " (volume "
                    << expected_volume
                    << ")\n";
                passed = false;
            }
        }
        if (!approximately_equal(
                combined->pixels[
                    combined_base + 3u],
                1.0f)) {
            std::cerr
                << "environment-volume alpha failed on "
                << backend << " pixel "
                << pixel << '\n';
            passed = false;
        }
    }

    // Latest Cycles disables Background MIS for a constant raw world even
    // when its authored policy is MANUAL; MANUAL only selects importance-map
    // resolution. Pin that structural eligibility rule separately so the
    // spatial oracle above cannot pass through an always-enabled emitter.
    psycles::io::MemoryOutputSink
        constant_sink;
    if (!render_scene(
            renderer,
            make_scene(
                all_ray_visibility,
                false),
            settings,
            constant_sink)) {
        std::cerr
            << "constant environment render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *constant_combined =
        constant_sink.find(
            PassKind::combined);
    const auto *constant_volume =
        constant_sink.find(
            PassKind::volume_direct);
    if (constant_combined == nullptr ||
        constant_volume == nullptr) {
        return EXIT_FAILURE;
    }
    constexpr std::array
        cycles_constant_combined_red{
            0.655930638f,
            0.351366013f,
            0.655930638f,
            0.655930638f,
            0.655930638f,
            0.369103521f,
            0.248031095f,
            0.655930638f,
            0.655930638f,
            0.655930638f,
            0.655930638f,
            0.655930638f,
            0.170339391f,
            0.277375728f,
            0.285650074f,
            0.315094978f};
    constexpr std::array
        cycles_constant_volume_red{
            0.0f,
            0.351366013f,
            0.0f,
            0.0f,
            0.0f,
            0.369103521f,
            0.248031095f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.170339391f,
            0.277375728f,
            0.285650074f,
            0.315094978f};
    constexpr std::array
        constant_channel_scale{
            1.0f, 0.5f, 0.25f};
    for (auto pixel = std::size_t{0u};
         pixel <
         cycles_constant_combined_red
             .size();
         ++pixel) {
        const auto combined_base =
            pixel *
            constant_combined->channels;
        const auto volume_base =
            pixel *
            constant_volume->channels;
        for (auto channel =
                 std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto scale =
                constant_channel_scale[
                    channel];
            const auto combined_value =
                constant_combined->pixels[
                    combined_base +
                    channel];
            const auto volume_value =
                constant_volume->pixels[
                    volume_base +
                    channel];
            if (!approximately_equal(
                    combined_value,
                    cycles_constant_combined_red[
                        pixel] *
                        scale) ||
                !approximately_equal(
                    volume_value,
                    cycles_constant_volume_red[
                        pixel] *
                        scale)) {
                std::cerr
                    << "constant World emitter eligibility failed on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << ": got "
                    << combined_value
                    << " (volume "
                    << volume_value
                    << ")\n";
                passed = false;
            }
        }
    }

    psycles::io::MemoryOutputSink
        no_scatter_sink;
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
            no_scatter_sink)) {
        std::cerr
            << "volume-hidden environment render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *no_scatter_combined =
        no_scatter_sink.find(
            PassKind::combined);
    const auto *no_scatter_volume =
        no_scatter_sink.find(
            PassKind::volume_direct);
    if (no_scatter_combined == nullptr ||
        no_scatter_volume == nullptr ||
        !all_zero(*no_scatter_volume)) {
        std::cerr
            << "world volume-scatter visibility leaked on "
            << backend << '\n';
        passed = false;
    } else if (all_zero(
                   *no_scatter_combined)) {
        std::cerr
            << "volume-scatter visibility incorrectly hid camera background on "
            << backend << '\n';
        passed = false;
    }

    psycles::io::MemoryOutputSink
        no_camera_sink;
    const auto without_camera =
        all_ray_visibility &
        ~visibility_bit(
            RayVisibility::camera);
    if (!render_scene(
            renderer,
            make_scene(
                without_camera),
            settings,
            no_camera_sink)) {
        std::cerr
            << "camera-hidden environment render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *no_camera_combined =
        no_camera_sink.find(
            PassKind::combined);
    const auto *no_camera_volume =
        no_camera_sink.find(
            PassKind::volume_direct);
    if (no_camera_combined == nullptr ||
        no_camera_volume == nullptr) {
        return EXIT_FAILURE;
    }
    auto saw_volume = false;
    for (auto pixel = std::size_t{0u};
         pixel <
         cycles_volume.size();
         ++pixel) {
        const auto combined_base =
            pixel *
            no_camera_combined->channels;
        const auto volume_base =
            pixel *
            no_camera_volume->channels;
        for (auto channel =
                 std::size_t{0u};
             channel < 3u;
             ++channel) {
            const auto combined_value =
                no_camera_combined->pixels[
                    combined_base +
                    channel];
            const auto volume_value =
                no_camera_volume->pixels[
                    volume_base +
                    channel];
            saw_volume |=
                volume_value != 0.0f;
            if (!approximately_equal(
                    combined_value,
                    volume_value)) {
                std::cerr
                    << "world camera visibility leaked on "
                    << backend << " pixel "
                    << pixel << " channel "
                    << channel << '\n';
                passed = false;
            }
        }
    }
    if (!saw_volume) {
        std::cerr
            << "camera visibility incorrectly hid volume lighting on "
            << backend << '\n';
        passed = false;
    }

    psycles::io::MemoryOutputSink
        bump_feature_mask_sink;
    if (!render_scene(
            renderer,
            bump_feature_mask_scene(),
            settings,
            bump_feature_mask_sink)) {
        std::cerr
            << "world Vector Displacement feature-mask render failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *bump_feature_mask_combined =
        bump_feature_mask_sink.find(
            PassKind::combined);
    const auto *bump_feature_mask_volume =
        bump_feature_mask_sink.find(
            PassKind::volume_direct);
    if (bump_feature_mask_combined == nullptr ||
        bump_feature_mask_volume == nullptr ||
        !all_zero(*bump_feature_mask_volume)) {
        std::cerr
            << "world Vector Displacement feature-mask volume pass failed on "
            << backend << '\n';
        passed = false;
    } else {
        for (auto pixel = std::size_t{0u};
             pixel < bump_feature_mask_combined->pixels.size() /
                         bump_feature_mask_combined->channels;
             ++pixel) {
            const auto base =
                pixel * bump_feature_mask_combined->channels;
            for (auto channel = std::size_t{0u};
                 channel < 3u;
                 ++channel) {
                if (bump_feature_mask_combined->pixels[base + channel] !=
                    0.0f) {
                    std::cerr
                        << "Cycles 5.2.1 world NODE_BUMP mask regression "
                        << "failed on " << backend << " pixel " << pixel
                        << " channel " << channel << ": got "
                        << bump_feature_mask_combined->pixels[base + channel]
                        << ", expected 0\n";
                    passed = false;
                }
            }
            if (bump_feature_mask_combined->pixels[base + 3u] != 1.0f) {
                std::cerr
                    << "world Vector Displacement feature-mask alpha failed on "
                    << backend << " pixel " << pixel << '\n';
                passed = false;
            }
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
                << "could not write environment-volume EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
#endif
    return passed
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
