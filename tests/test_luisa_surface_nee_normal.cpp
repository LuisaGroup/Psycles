#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include "path_tracer_internal.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
    Mat4f result;
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] ShaderGraph tilted_thin_glass_shader() {
    ShaderGraph graph;
    const auto normal_color = graph.add_node(
        node_type::constant_color,
        "Linked Cycles closure normal");
    const auto normal_vector = graph.add_node(
        node_type::color_to_vector,
        "Color to closure-normal vector");
    const auto normal = graph.add_node(
        node_type::vector_to_normal,
        "Raw closure normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Tilted thin-glass receiver");
    const auto configured =
        graph.set_input(
            normal_color,
            "Color",
            SocketValue::color({0.6f, 0.0f, 0.8f})) &&
        graph.connect(
            {.node = normal_color, .socket = "Color"},
            normal_vector,
            "Color") &&
        graph.connect(
            {.node = normal_vector, .socket = "Vector"},
            normal,
            "Vector") &&
        graph.connect(
            {.node = normal, .socket = "Normal"},
            principled,
            "Normal") &&
        graph.set_input(
            principled,
            "BaseColor",
            SocketValue::color({0.25f, 0.64f, 1.0f})) &&
        graph.set_input(
            principled,
            "Roughness",
            SocketValue::floating(0.35f)) &&
        graph.set_input(
            principled,
            "TransmissionWeight",
            SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled,
            "IOR",
            SocketValue::floating(1.5f)) &&
        graph.set_input(
            principled,
            "ThinWall",
            SocketValue::boolean(true));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure tilted thin-glass shader"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph subsurface_shader() {
    ShaderGraph graph;
    const auto bssrdf = graph.add_node(
        node_type::subsurface_scattering,
        "Reachable BSSRDF capability");
    const auto configured =
        graph.set_input(
            bssrdf,
            "Color",
            SocketValue::color({0.7f, 0.25f, 0.1f})) &&
        graph.set_input(
            bssrdf,
            "Normal",
            SocketValue::normal({0.0f, 0.0f, 1.0f})) &&
        graph.set_input(
            bssrdf,
            "Scale",
            SocketValue::floating(1.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure BSSRDF capability shader"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = bssrdf, .socket = "Closure"});
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
            .name = "Cycles linked-normal thin glass",
            .shader = tilted_thin_glass_shader(),
            .cycles_shader_index = 0u});

    TriangleMeshDesc mesh;
    mesh.name = "Linked-normal NEE receiver";
    mesh.positions = {
        {-4.0f, -4.0f, 0.0f},
        {4.0f, -4.0f, 0.0f},
        {0.0f, 4.0f, 0.0f}};
    mesh.normals.values.assign(
        mesh.positions.size(),
        Vec3f{0.0f, 0.0f, 1.0f});
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {material_id};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {1u};
    mesh.triangle_random_per_island = {0.0f};
    scene.geometries.emplace(geometry_id, std::move(mesh));
    scene.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "Linked-normal NEE receiver",
            .geometry = geometry_id,
            .transform = {},
            .cycles_object_index = 0u});

    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Linked-normal NEE camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .orthographic_scale = 1.0f,
            .near_clip = 0.1f,
            .far_clip = 100.0f});
    scene.active_camera = camera_id;

    scene.lights.emplace(
        light_id,
        LightDesc{
            .name = "Cycles linked-normal point light",
            .type = LightType::point,
            .transform = translated(0.3f, -0.2f, 2.0f),
            .color = {1.0f, 0.8f, 0.6f},
            .power = 40.0f,
            .size = 0.0f,
            .normalize = true,
            .is_sphere = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask = all_ray_visibility,
            .cycles_shader_index = 1u,
            .cycles_object_index = 1u});
    scene.world_sampling = WorldSampling::none;
    return scene;
}

[[nodiscard]] RenderSettings make_settings() {
    return {
        .full_extent = {.width = 1u, .height = 1u},
        .window = {},
        .seed = 0u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::box,
        .filter_width = 1.0f,
        .pass_alpha_threshold = 0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 0u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 0u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold = 0.01f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::multiple_importance_sampling},
        .passes = {{
            .kind = PassKind::combined,
            .name = "Combined",
            .light_group = {},
            .channels = 4u}}};
}

[[nodiscard]] bool compiled_scene_has_subsurface(
    const SceneCompilation &compilation) {
    const auto *compiled = dynamic_cast<
        const psycles::luisa_backend::detail::LuisaCompiledScene *>(
        compilation.scene.get());
    if (compiled == nullptr || !compiled->data()) {
        throw std::runtime_error{
            "compiled scene does not expose Luisa scene data"};
    }
    return compiled->data()->has_subsurface;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
            .max_samples_per_dispatch = 1u}};

    {
        auto unreachable = make_scene();
        unreachable.materials.emplace(
            MaterialId{99u},
            MaterialDesc{
                .name = "Unreachable BSSRDF",
                .shader = subsurface_shader(),
                .cycles_shader_index = 2u});
        const auto capability =
            renderer.compile_scene(unreachable);
        if (!capability.ok() ||
            compiled_scene_has_subsurface(capability)) {
            std::cerr
                << "unreachable BSSRDF enabled path transport on "
                << backend << '\n';
            return EXIT_FAILURE;
        }
    }
    {
        auto reachable = make_scene();
        reachable.materials.at(MaterialId{1u}).shader =
            subsurface_shader();
        const auto capability =
            renderer.compile_scene(reachable);
        if (!capability.ok() ||
            !compiled_scene_has_subsurface(capability)) {
            std::cerr
                << "reachable BSSRDF did not enable path transport on "
                << backend << '\n';
            return EXIT_FAILURE;
        }
    }

    const auto compilation = renderer.compile_scene(make_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }
    auto session = renderer.create_session(
        *compilation.scene, make_settings());
    if (!session) {
        std::cerr << "could not create linked-normal NEE session on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink output;
    if (!session->render_samples(
            {.first = 0u,
                .count = 1u,
                .offset = 0u,
                .total = 1u},
            output)) {
        std::cerr << "linked-normal NEE render failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto *combined = output.find(PassKind::combined);
    if (combined == nullptr || combined->channels != 4u ||
        combined->pixels.size() != 4u) {
        std::cerr << "linked-normal NEE Combined pass has invalid shape on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto &pixel = combined->pixels;
    const auto finite = std::isfinite(pixel[0u]) &&
                        std::isfinite(pixel[1u]) &&
                        std::isfinite(pixel[2u]) &&
                        std::isfinite(pixel[3u]);
    // This focused fixture pins the production-kernel liveness failure. Its
    // quantitative Cycles oracle remains the full
    // principled_thin_wall_surface probe, where both oblique lights exercise
    // non-zero reflection and transmission and every pass has a strict gate.
    if (!finite || pixel[3u] != 1.0f) {
        std::cerr << "linked-normal NEE liveness/value regression on "
                  << backend << ": {" << pixel[0u] << ", "
                  << pixel[1u] << ", " << pixel[2u] << ", "
                  << pixel[3u] << "}\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
