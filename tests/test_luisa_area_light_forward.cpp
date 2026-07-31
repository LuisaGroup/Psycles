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

constexpr auto pi = 3.14159265358979323846f;
constexpr std::string_view oracle_color_attribute{
    "Cycles rectangle oracle color"};
constexpr std::string_view point_identity_attribute{
    "Cycles rectangle point identity"};
constexpr std::string_view face_identity_attribute{
    "Cycles rectangle face identity"};

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

[[nodiscard]] ShaderGraph diffuse_shader() {
    ShaderGraph graph;
    const auto oracle_color =
        graph.add_node(node_type::vertex_color, "Corner Color");
    const auto point_identity =
        graph.add_node(node_type::vertex_color, "Point Identity");
    const auto face_identity =
        graph.add_node(node_type::vertex_color, "Face Identity");
    const auto multiply_point =
        graph.add_node(node_type::multiply_color, "Multiply Point");
    const auto multiply_face =
        graph.add_node(node_type::multiply_color, "Multiply Face");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    const auto bind_attribute =
        [&](NodeId node, std::string_view name) {
            static_cast<void>(graph.set_property(
                node,
                "AttributeId",
                SocketValue::unsigned_integer(
                    attribute_id(name))));
        };
    bind_attribute(oracle_color, oracle_color_attribute);
    bind_attribute(point_identity, point_identity_attribute);
    bind_attribute(face_identity, face_identity_attribute);
    static_cast<void>(graph.connect(
        {.node = oracle_color, .socket = "Color"},
        multiply_point,
        "A"));
    static_cast<void>(graph.connect(
        {.node = point_identity, .socket = "Color"},
        multiply_point,
        "B"));
    static_cast<void>(graph.connect(
        {.node = multiply_point, .socket = "Color"},
        multiply_face,
        "A"));
    static_cast<void>(graph.connect(
        {.node = face_identity, .socket = "Color"},
        multiply_face,
        "B"));
    static_cast<void>(graph.connect(
        {.node = multiply_face, .socket = "Color"},
        diffuse,
        "Color"));
    static_cast<void>(graph.set_input(
        diffuse,
        "Roughness",
        SocketValue::floating(0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot make_scene() {
    constexpr MaterialId material_id{1u};
    constexpr GeometryId geometry_id{2u};
    constexpr InstanceId instance_id{3u};
    constexpr CameraId camera_id{4u};
    constexpr LightId light_id{5u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        material_id,
        MaterialDesc{
            .name = "Cycles rectangle oracle diffuse",
            .shader = diffuse_shader(),
            .cycles_shader_index = 6u});

    TriangleMeshDesc mesh;
    mesh.name = "Cycles rectangle oracle plane";
    mesh.positions = {
        {-4.0f, -4.0f, 0.0f},
        {4.0f, -4.0f, 0.0f},
        {4.0f, 4.0f, 0.0f},
        {-4.0f, 4.0f, 0.0f}};
    mesh.normals.values.assign(
        mesh.positions.size(),
        Vec3f{0.0f, 0.0f, 1.0f});
    mesh.uv.values = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}};
    mesh.uv.domain = MeshAttributeDomain::corner;
    auto &oracle_color =
        mesh.color_attributes[
            std::string{oracle_color_attribute}];
    oracle_color.domain = MeshAttributeDomain::corner;
    oracle_color.values.assign(
        mesh.uv.values.size(),
        Vec4f{
            0.6200000047683716f,
            0.4099999964237213f,
            0.23000000417232513f,
            1.0f});
    auto &point_identity =
        mesh.color_attributes[
            std::string{point_identity_attribute}];
    point_identity.domain = MeshAttributeDomain::point;
    point_identity.values.assign(
        mesh.positions.size(),
        Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    auto &face_identity =
        mesh.color_attributes[
            std::string{face_identity_attribute}];
    face_identity.domain = MeshAttributeDomain::face;
    face_identity.values.assign(
        2u,
        Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    mesh.triangles = {
        {0u, 1u, 2u},
        {0u, 2u, 3u}};
    mesh.material_slots = {material_id};
    mesh.triangle_material_slots = {0u, 0u};
    mesh.triangle_smooth = {0u, 0u};
    mesh.triangle_random_per_island = {
        0.874585747718811f,
        0.874585747718811f};
    scene.geometries.emplace(
        geometry_id, std::move(mesh));
    scene.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "Cycles rectangle oracle plane",
            .geometry = geometry_id,
            .transform = {},
            .cycles_object_index = 0u});

    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Cycles rectangle oracle camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .field_of_view = 0.4710899591445923f,
            .horizontal_field_of_view =
                0.6911112070083618f,
            .sensor_fit = CameraSensorFit::automatic,
            .orthographic_scale = 2.200000047683716f,
            .near_clip = 0.10000000149011612f,
            .far_clip = 1000.0f});
    scene.active_camera = camera_id;

    scene.lights.emplace(
        light_id,
        LightDesc{
            .name = "Cycles rectangle oracle light",
            .type = LightType::area,
            .transform = translated(
                0.3700000047683716f,
                -0.20999999344348907f,
                1.399999976158142f),
            .color = {
                0.36000001430511475f,
                0.7200000286102295f,
                1.0f},
            .power = 37.0f,
            .size = 0.800000011920929f,
            .size_y = 0.5f,
            .spread = pi,
            .normalize = true,
            .ellipse = false,
            .is_sphere = false,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask =
                all_ray_visibility &
                ~visibility_bit(RayVisibility::camera),
            .is_shadow_catcher = true,
            .cycles_shader_index = 5u,
            .cycles_object_index = 1u});
    scene.world_sampling = WorldSampling::automatic;
    return scene;
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-6f) noexcept {
    return std::abs(actual - expected) <= tolerance;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .max_samples_per_dispatch = 1u}};
    auto compilation = renderer.compile_scene(make_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }

    constexpr std::uint32_t width = 32u;
    constexpr std::uint32_t height = 32u;
    const RenderSettings settings{
        .full_extent = {.width = width, .height = height},
        .window = {},
        .seed = 20903u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::blackman_harris,
        .filter_width = 1.5f,
        .pass_alpha_threshold = 0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 0u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 8u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 10.0f,
            .filter_glossy = 1.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold =
                0.009999999776482582f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::
                    multiple_importance_sampling},
        .passes = {
            {.kind = PassKind::combined,
             .name = "Combined",
             .light_group = {},
             .channels = 4u}}};
    auto session =
        renderer.create_session(*compilation.scene, settings);
    if (!session) {
        std::cerr << "could not create Luisa render session\n";
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink sink;
    if (!session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            sink)) {
        std::cerr << "Luisa render failed\n";
        return EXIT_FAILURE;
    }
    const auto *combined = sink.find(PassKind::combined);
    if (combined == nullptr || combined->channels != 4u ||
        combined->pixels.size() !=
            static_cast<std::size_t>(width) *
                height * combined->channels) {
        std::cerr << "Combined pass has an invalid shape\n";
        return EXIT_FAILURE;
    }

    std::array<double, 3u> means{};
    for (std::size_t pixel = 0u;
         pixel <
         static_cast<std::size_t>(width) * height;
         ++pixel) {
        const auto base = pixel * combined->channels;
        for (std::size_t channel = 0u;
             channel < means.size();
             ++channel) {
            means[channel] +=
                combined->pixels[base + channel];
        }
    }
    for (auto &mean : means) {
        mean /= static_cast<double>(width * height);
    }
    constexpr std::array cycles_means{
        0.21808527410030365,
        0.2884353697299957,
        0.22472937405109406};
    for (std::size_t channel = 0u;
         channel < means.size();
         ++channel) {
        if (std::abs(means[channel] -
                     cycles_means[channel]) >
            2.0e-6) {
            std::cerr
                << "Cycles full-frame mean regression failed on "
                << backend << " channel " << channel
                << ": got " << means[channel]
                << ", expected " << cycles_means[channel]
                << '\n';
            return EXIT_FAILURE;
        }
    }

    constexpr std::uint32_t hit_x = 20u;
    constexpr std::uint32_t hit_y = 14u;
    const auto hit_base =
        (static_cast<std::size_t>(hit_y) * width + hit_x) *
        combined->channels;
    constexpr std::array cycles_hit{
        0.37258365750312805f,
        0.4927719235420227f,
        0.38393476605415344f,
        1.0f};
    for (std::size_t channel = 0u;
         channel < cycles_hit.size();
         ++channel) {
        if (!approximately_equal(
                combined->pixels[hit_base + channel],
                cycles_hit[channel])) {
            std::cerr
                << "Cycles BSDF-forward rectangle hit regression "
                   "failed on "
                << backend << " channel " << channel
                << ": got "
                << combined->pixels[hit_base + channel]
                << ", expected " << cycles_hit[channel]
                << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
