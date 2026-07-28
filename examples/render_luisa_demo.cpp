#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

constexpr auto pi = 3.14159265358979323846f;

[[nodiscard]] Mat4f transform(
    Vec3f x,
    Vec3f y,
    Vec3f z,
    Vec3f translation) noexcept {
    return Mat4f{{
        x.x, x.y, x.z, 0.0f,
        y.x, y.y, y.z, 0.0f,
        z.x, z.y, z.z, 0.0f,
        translation.x, translation.y, translation.z, 1.0f}};
}

[[nodiscard]] Vec3f subtract(Vec3f a, Vec3f b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f v, float s) noexcept {
    return {v.x * s, v.y * s, v.z * s};
}

[[nodiscard]] float dot(Vec3f a, Vec3f b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3f cross(Vec3f a, Vec3f b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

[[nodiscard]] Vec3f normalize(Vec3f v) noexcept {
    return multiply(
        v, 1.0f / std::sqrt(std::max(dot(v, v), 1.0e-20f)));
}

[[nodiscard]] Mat4f look_at(
    Vec3f eye,
    Vec3f target) noexcept {
    const auto forward = normalize(subtract(target, eye));
    const auto right =
        normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
    const auto backward = multiply(forward, -1.0f);
    const auto up = normalize(cross(backward, right));
    return transform(right, up, backward, eye);
}

[[nodiscard]] ShaderGraph diffuse(Vec3f color, float roughness) {
    ShaderGraph graph;
    const auto node =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    static_cast<void>(
        graph.set_input(node, "Color", SocketValue::color(color)));
    static_cast<void>(graph.set_input(
        node, "Roughness", SocketValue::floating(roughness)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = node, .socket = "Closure"});
    return graph;
}

void add_quad(
    TriangleMeshDesc &mesh,
    std::array<Vec3f, 4u> p,
    Vec3f normal,
    std::uint32_t material) {
    const auto base =
        static_cast<std::uint32_t>(mesh.positions.size());
    for (const auto value : p) {
        mesh.positions.emplace_back(value);
        mesh.normals.emplace_back(normal);
    }
    mesh.uv.insert(
        mesh.uv.end(),
        {{0.0f, 0.0f},
         {1.0f, 0.0f},
         {1.0f, 1.0f},
         {0.0f, 1.0f}});
    mesh.triangles.emplace_back(
        std::array{base, base + 1u, base + 2u});
    mesh.triangles.emplace_back(
        std::array{base, base + 2u, base + 3u});
    mesh.triangle_material_slots.emplace_back(material);
    mesh.triangle_material_slots.emplace_back(material);
}

[[nodiscard]] TriangleMeshDesc room(
    const std::vector<MaterialId> &materials) {
    TriangleMeshDesc mesh;
    mesh.name = "Luisa Cornell room";
    mesh.material_slots = materials;
    add_quad(
        mesh,
        {{{-3.0f, 0.0f, 2.5f},
          {3.0f, 0.0f, 2.5f},
          {3.0f, 0.0f, -3.0f},
          {-3.0f, 0.0f, -3.0f}}},
        {0.0f, 1.0f, 0.0f},
        0u);
    add_quad(
        mesh,
        {{{-3.0f, 0.0f, -3.0f},
          {3.0f, 0.0f, -3.0f},
          {3.0f, 4.5f, -3.0f},
          {-3.0f, 4.5f, -3.0f}}},
        {0.0f, 0.0f, 1.0f},
        0u);
    add_quad(
        mesh,
        {{{-3.0f, 0.0f, 2.5f},
          {-3.0f, 0.0f, -3.0f},
          {-3.0f, 4.5f, -3.0f},
          {-3.0f, 4.5f, 2.5f}}},
        {1.0f, 0.0f, 0.0f},
        1u);
    add_quad(
        mesh,
        {{{3.0f, 0.0f, -3.0f},
          {3.0f, 0.0f, 2.5f},
          {3.0f, 4.5f, 2.5f},
          {3.0f, 4.5f, -3.0f}}},
        {-1.0f, 0.0f, 0.0f},
        2u);
    return mesh;
}

[[nodiscard]] TriangleMeshDesc cube(MaterialId material) {
    TriangleMeshDesc mesh;
    mesh.name = "Unit cube";
    mesh.material_slots = {material};
    add_quad(
        mesh,
        {{{-1.0f, -1.0f, 1.0f},
          {1.0f, -1.0f, 1.0f},
          {1.0f, 1.0f, 1.0f},
          {-1.0f, 1.0f, 1.0f}}},
        {0.0f, 0.0f, 1.0f},
        0u);
    add_quad(
        mesh,
        {{{1.0f, -1.0f, -1.0f},
          {-1.0f, -1.0f, -1.0f},
          {-1.0f, 1.0f, -1.0f},
          {1.0f, 1.0f, -1.0f}}},
        {0.0f, 0.0f, -1.0f},
        0u);
    add_quad(
        mesh,
        {{{-1.0f, -1.0f, -1.0f},
          {-1.0f, -1.0f, 1.0f},
          {-1.0f, 1.0f, 1.0f},
          {-1.0f, 1.0f, -1.0f}}},
        {-1.0f, 0.0f, 0.0f},
        0u);
    add_quad(
        mesh,
        {{{1.0f, -1.0f, 1.0f},
          {1.0f, -1.0f, -1.0f},
          {1.0f, 1.0f, -1.0f},
          {1.0f, 1.0f, 1.0f}}},
        {1.0f, 0.0f, 0.0f},
        0u);
    add_quad(
        mesh,
        {{{-1.0f, 1.0f, 1.0f},
          {1.0f, 1.0f, 1.0f},
          {1.0f, 1.0f, -1.0f},
          {-1.0f, 1.0f, -1.0f}}},
        {0.0f, 1.0f, 0.0f},
        0u);
    return mesh;
}

[[nodiscard]] SceneSnapshot make_scene() {
    constexpr MaterialId white{1u};
    constexpr MaterialId red{2u};
    constexpr MaterialId green{3u};
    constexpr MaterialId blue{4u};
    constexpr GeometryId room_id{10u};
    constexpr GeometryId cube_id{11u};
    constexpr CameraId camera_id{20u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        white,
        MaterialDesc{
            .name = "Plaster",
            .shader = diffuse({0.72f, 0.72f, 0.72f}, 0.35f)});
    scene.materials.emplace(
        red,
        MaterialDesc{
            .name = "Red wall",
            .shader = diffuse({0.72f, 0.06f, 0.035f}, 0.45f)});
    scene.materials.emplace(
        green,
        MaterialDesc{
            .name = "Green wall",
            .shader = diffuse({0.04f, 0.54f, 0.12f}, 0.45f)});
    scene.materials.emplace(
        blue,
        MaterialDesc{
            .name = "Blue block",
            .shader = diffuse({0.035f, 0.18f, 0.85f}, 0.16f)});
    scene.geometries.emplace(
        room_id, room({white, red, green}));
    scene.geometries.emplace(cube_id, cube(blue));
    scene.instances.emplace(
        InstanceId{30u},
        InstanceDesc{
            .name = "Room",
            .geometry = room_id,
            .transform = {},
            .motion = {},
            .material_overrides = {},
            .visibility_mask = ~std::uint32_t{0u}});
    scene.instances.emplace(
        InstanceId{31u},
        InstanceDesc{
            .name = "Block",
            .geometry = cube_id,
            .transform = transform(
                {0.72f, 0.0f, 0.0f},
                {0.0f, 1.15f, 0.0f},
                {0.0f, 0.0f, 0.72f},
                {0.35f, 1.15f, -0.55f}),
            .motion = {},
            .material_overrides = {},
            .visibility_mask = ~std::uint32_t{0u}});
    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Camera",
            .projection = CameraProjection::perspective,
            .transform =
                look_at({0.0f, 2.05f, 7.0f}, {0.0f, 1.5f, -0.8f}),
            .field_of_view = 47.0f * pi / 180.0f,
            .horizontal_field_of_view = 47.0f * pi / 180.0f,
            .sensor_fit = CameraSensorFit::vertical,
            .orthographic_scale = 1.0f,
            .near_clip = 1.0e-3f,
            .far_clip = 100.0f});
    scene.active_camera = camera_id;
    scene.lights.emplace(
        LightId{40u},
        LightDesc{
            .name = "Ceiling softbox",
            .type = LightType::area,
            .transform = transform(
                {1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                {0.0f, 1.0f, 0.0f},
                {-0.4f, 4.25f, -0.6f}),
            .color = {1.0f, 0.82f, 0.62f},
            .power = 18.0f,
            .size = 1.8f,
            .spread = pi,
            .shader = std::nullopt});
    scene.lights.emplace(
        LightId{41u},
        LightDesc{
            .name = "World",
            .type = LightType::background,
            .transform = {},
            .color = {0.008f, 0.012f, 0.025f},
            .power = 1.0f,
            .size = 0.0f,
            .spread = pi,
            .shader = std::nullopt});
    return scene;
}

template<typename T>
[[nodiscard]] std::optional<T> parse_unsigned(
    std::string_view text) {
    T result{};
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return result;
}

}// namespace

int main(int argc, char **argv) {
    const auto output = std::filesystem::path{
        argc > 1 ? argv[1] : "psycles-luisa.ppm"};
    const auto backend_name =
        std::string_view{argc > 2 ? argv[2] : "fallback"};
    auto width = std::uint32_t{320u};
    auto height = std::uint32_t{240u};
    auto samples = std::uint32_t{64u};
    if (argc > 3) {
        auto parsed = parse_unsigned<std::uint32_t>(argv[3]);
        if (!parsed || *parsed == 0u) {
            return EXIT_FAILURE;
        }
        width = *parsed;
    }
    if (argc > 4) {
        auto parsed = parse_unsigned<std::uint32_t>(argv[4]);
        if (!parsed || *parsed == 0u) {
            return EXIT_FAILURE;
        }
        height = *parsed;
    }
    if (argc > 5) {
        auto parsed = parse_unsigned<std::uint32_t>(argv[5]);
        if (!parsed || *parsed == 0u) {
            return EXIT_FAILURE;
        }
        samples = *parsed;
    }

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend_name);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true}};
    auto compilation = renderer.compile_scene(make_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }
    const RenderSettings settings{
        .full_extent = {.width = width, .height = height},
        .window = {},
        .seed = 0x5a17c9e3u,
        .transparent_background = false,
        .integrator = {
            .max_bounces = 7u,
            .min_bounces = 2u,
            .diffuse_bounces = 7u,
            .glossy_bounces = 7u,
            .transmission_bounces = 7u},
        .passes = {
            {.kind = PassKind::combined,
             .name = "Combined",
             .light_group = {},
             .channels = 4u},
            {.kind = PassKind::normal,
             .name = "Normal",
             .light_group = {},
             .channels = 3u},
            {.kind = PassKind::albedo,
             .name = "Albedo",
             .light_group = {},
             .channels = 3u}}};
    auto session =
        renderer.create_session(*compilation.scene, settings);
    if (!session) {
        return EXIT_FAILURE;
    }
    psycles::io::MemoryOutputSink sink;
    const auto begin = std::chrono::steady_clock::now();
    if (!session->render_samples(
            {.first = 0u,
             .count = samples,
             .offset = 0u,
             .total = samples},
            sink)) {
        return EXIT_FAILURE;
    }
    const auto *combined = sink.find(PassKind::combined);
    const auto *normal = sink.find(PassKind::normal);
    const auto *albedo = sink.find(PassKind::albedo);
    if (combined == nullptr || normal == nullptr ||
        albedo == nullptr ||
        !psycles::io::write_ppm(
            *combined,
            output,
            {.exposure_stops = -0.25f,
             .apply_aces_tonemap = true,
             .apply_srgb_transfer = true})) {
        return EXIT_FAILURE;
    }
    const auto stem = output.parent_path() / output.stem();
    const auto combined_linear =
        std::filesystem::path{stem.string() + "-combined.pfm"};
    const auto normal_linear =
        std::filesystem::path{stem.string() + "-normal.pfm"};
    const auto albedo_linear =
        std::filesystem::path{stem.string() + "-albedo.pfm"};
    if (!psycles::io::write_pfm(*combined, combined_linear) ||
        !psycles::io::write_pfm(*normal, normal_linear) ||
        !psycles::io::write_pfm(*albedo, albedo_linear)) {
        return EXIT_FAILURE;
    }
    const auto seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin)
            .count();
    std::cout << "Luisa/" << backend_name << " rendered "
              << width << 'x' << height << " at "
              << samples << " spp in " << seconds
              << " s: " << output << '\n'
              << "Linear Combined: " << combined_linear << '\n'
              << "Linear Normal:   " << normal_linear << '\n'
              << "Linear Albedo:   " << albedo_linear << '\n';
    return EXIT_SUCCESS;
}
