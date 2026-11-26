#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/luisa/volume_shadow_interval.h>

#include "../src/luisa/path_tracer_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace luisa::compute;

void set_test_env(const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr && value[0] != '\0') {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct ScopedEnvGuard {
    std::string key;
    std::string old_value;
    bool had_old{false};

    ScopedEnvGuard(const char *k, const char *v) : key{k} {
        if (const char *prev = std::getenv(k); prev != nullptr) {
            had_old = true;
            old_value = prev;
        }
        set_test_env(key.c_str(), v);
    }
    ~ScopedEnvGuard() {
        set_test_env(key.c_str(), had_old ? old_value.c_str() : "");
    }
};

[[nodiscard]] ShaderGraph black_world_shader() {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(node_type::emission, "Black world");
    static_cast<void>(graph.set_input(
        emission, "Color", SocketValue::color({0.0f, 0.0f, 0.0f})));
    static_cast<void>(graph.set_input(
        emission, "Strength", SocketValue::floating(0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph volume_scatter_shader(float density) {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(node_type::transparent_bsdf, "Transparent boundary");
    const auto scatter =
        graph.add_node(node_type::volume_scatter, "Homogeneous scatter");
    static_cast<void>(graph.set_input(
        transparent, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        scatter, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        scatter, "Density", SocketValue::floating(density)));
    static_cast<void>(graph.set_input(
        scatter, "Anisotropy", SocketValue::floating(0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = transparent, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{.node = scatter, .socket = "Volume"});
    return graph;
}

[[nodiscard]] ShaderGraph volume_absorption_shader(float density) {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(node_type::transparent_bsdf, "Transparent boundary");
    const auto absorption =
        graph.add_node(node_type::volume_absorption, "Homogeneous absorption");
    static_cast<void>(graph.set_input(
        transparent, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        absorption, "Color", SocketValue::color({0.0f, 0.0f, 0.0f})));
    static_cast<void>(graph.set_input(
        absorption, "Density", SocketValue::floating(density)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = transparent, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{.node = absorption, .socket = "Volume"});
    return graph;
}

[[nodiscard]] TriangleMeshDesc unit_box_mesh(MaterialId material, float xy_half,
                                             float z_min, float z_max) {
    TriangleMeshDesc mesh;
    mesh.name = "Volume box";
    mesh.positions = {
        {-xy_half, -xy_half, z_min},
        {xy_half, -xy_half, z_min},
        {xy_half, xy_half, z_min},
        {-xy_half, xy_half, z_min},
        {-xy_half, -xy_half, z_max},
        {xy_half, -xy_half, z_max},
        {xy_half, xy_half, z_max},
        {-xy_half, xy_half, z_max}};
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
    mesh.triangle_material_slots.assign(mesh.triangles.size(), 0u);
    mesh.triangle_smooth.assign(mesh.triangles.size(), 0u);
    return mesh;
}

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
    Mat4f result{};
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] Mat4f scaled_translated(float sx, float sy, float sz,
                                      float tx, float ty, float tz) noexcept {
    Mat4f result{};
    result.elements[0u] = sx;
    result.elements[5u] = sy;
    result.elements[10u] = sz;
    result.elements[12u] = tx;
    result.elements[13u] = ty;
    result.elements[14u] = tz;
    return result;
}

// Rotated 180 degrees around X so local -Z points along +Z: shadow rays travel along (0, 0, -1).
[[nodiscard]] Mat4f distant_light_along_plus_z() noexcept {
    Mat4f result{};
    result.elements[5u] = -1.0f;
    result.elements[10u] = -1.0f;
    return result;
}

[[nodiscard]] RenderSettings make_test_settings(std::uint32_t transparent_max = 16u) {
    return {
        .full_extent = {.width = 8u, .height = 8u},
        .window = {},
        .seed = 4242u,
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
            .transparent_max_bounces = transparent_max,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::multiple_importance_sampling},
        .passes = {
            {.kind = PassKind::combined, .name = "Combined", .channels = 4u},
            {.kind = PassKind::volume_direct,
             .name = "Volume Direct",
             .channels = 3u}}};
}

[[nodiscard]] SceneSnapshot make_base_scatter_scene(std::uint64_t revision) {
    constexpr MaterialId scatter_mat{1u};
    constexpr MaterialId world_mat{2u};
    constexpr GeometryId scatter_geom{1u};
    constexpr InstanceId scatter_inst{1u};
    constexpr CameraId camera{1u};

    SceneSnapshot scene;
    scene.revision = revision;
    scene.materials.emplace(
        scatter_mat,
        MaterialDesc{
            .name = "Scatter slab",
            .shader = volume_scatter_shader(0.5f),
            .cycles_shader_index = 0u});
    scene.materials.emplace(
        world_mat,
        MaterialDesc{
            .name = "Black world",
            .shader = black_world_shader(),
            .cycles_shader_index = 1u});
    // Camera at origin looks down -Z. Scattering medium lies in z in [-2.0, -0.5].
    scene.geometries.emplace(
        scatter_geom, unit_box_mesh(scatter_mat, 10.0f, -2.0f, -0.5f));
    scene.instances.emplace(
        scatter_inst,
        InstanceDesc{
            .name = "Scatter slab",
            .geometry = scatter_geom,
            .transform = {},
            .cycles_object_index = 0u});
    scene.cameras.emplace(
        camera,
        CameraDesc{
            .name = "Ortho cam",
            .projection = CameraProjection::orthographic,
            .transform = {},
            .sensor_fit = CameraSensorFit::automatic,
            .orthographic_scale = 1.0f,
            .near_clip = 0.05f,
            .far_clip = 100.0f});
    scene.active_camera = camera;
    scene.world_shader = world_mat;
    scene.world_sampling = WorldSampling::none;
    return scene;
}

[[nodiscard]] SceneSnapshot make_multi_instance_absorber_scene(
    std::uint64_t revision, bool with_absorbers) {
    auto scene = make_base_scatter_scene(revision);
    constexpr MaterialId absorber_mat{3u};
    constexpr GeometryId slab_geom{2u};
    std::uint32_t next_obj = 1u;
    if (with_absorbers) {
        scene.materials.emplace(
            absorber_mat,
            MaterialDesc{
                .name = "Absorber slab",
                .shader = volume_absorption_shader(1.0f),
                .cycles_shader_index = 2u});
        // Single shared mesh of thickness 0.5 in Z ([-0.25, +0.25]), instanced 3 times
        // with narrow gaps so all 3 distinct instances share identical prim IDs 0..11.
        scene.geometries.emplace(
            slab_geom, unit_box_mesh(absorber_mat, 10.0f, -0.25f, 0.25f));
        const std::array<float, 3u> z_centers{-2.50f, -3.0002f, -3.5004f};
        for (std::size_t i = 0u; i < z_centers.size(); ++i) {
            scene.instances.emplace(
                InstanceId{10u + i},
                InstanceDesc{
                    .name = "Absorber instance " + std::to_string(i),
                    .geometry = slab_geom,
                    .transform = translated(0.0f, 0.0f, z_centers[i]),
                    .cycles_object_index = next_obj++});
        }
    }
    scene.lights.emplace(
        LightId{1u},
        LightDesc{
            .name = "Distant light",
            .type = LightType::distant,
            .transform = distant_light_along_plus_z(),
            .color = {1.0f, 1.0f, 1.0f},
            .power = 2.0f,
            .angle = 0.0f,
            .normalize = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask = all_ray_visibility,
            .cycles_shader_index = with_absorbers ? 3u : 2u,
            .cycles_object_index = next_obj++});
    scene.cycles_background_object_index = next_obj;
    return scene;
}

[[nodiscard]] SceneSnapshot make_nested_instance_absorber_scene(
    std::uint64_t revision) {
    auto scene = make_base_scatter_scene(revision);
    constexpr MaterialId nested_mat{3u};
    constexpr GeometryId unit_cube_geom{2u};
    scene.materials.emplace(
        nested_mat,
        MaterialDesc{
            .name = "Nested absorber (sigma_a=0.7)",
            .shader = volume_absorption_shader(0.7f),
            .cycles_shader_index = 2u});
    // Shared unit cube in z in [-0.5, 0.5]; distinct nested instances differ only by
    // transform, keeping geometry, material, and primitive indices 0..11 identical.
    scene.geometries.emplace(
        unit_cube_geom, unit_box_mesh(nested_mat, 1.0f, -0.5f, 0.5f));
    // Outer covers z in [-3.8, -2.6] (thickness 1.2, sigma_a=0.7 -> tau_outer=0.84).
    scene.instances.emplace(
        InstanceId{20u},
        InstanceDesc{
            .name = "Outer nested absorber",
            .geometry = unit_cube_geom,
            .transform = scaled_translated(10.0f, 10.0f, 1.2f, 0.0f, 0.0f, -3.2f),
            .cycles_object_index = 1u});
    // Inner covers z in [-3.6, -2.8] (thickness 0.8, sigma_a=0.7 -> tau_inner=0.56).
    // Combined optical depth across the nested boundary crossings is 1.40.
    scene.instances.emplace(
        InstanceId{21u},
        InstanceDesc{
            .name = "Inner nested absorber",
            .geometry = unit_cube_geom,
            .transform = scaled_translated(10.0f, 10.0f, 0.8f, 0.0f, 0.0f, -3.2f),
            .cycles_object_index = 2u});
    scene.lights.emplace(
        LightId{1u},
        LightDesc{
            .name = "Distant light",
            .type = LightType::distant,
            .transform = distant_light_along_plus_z(),
            .color = {1.0f, 1.0f, 1.0f},
            .power = 2.0f,
            .angle = 0.0f,
            .normalize = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask = all_ray_visibility,
            .cycles_shader_index = 3u,
            .cycles_object_index = 3u});
    scene.cycles_background_object_index = 4u;
    return scene;
}

struct RenderedResult {
    std::vector<float> volume_direct;
    std::uint64_t guard_hits{0ull};
};

[[nodiscard]] bool run_render(
    LuisaPathTracerBackend &renderer,
    const SceneSnapshot &scene,
    const RenderSettings &settings,
    RenderedResult &out) {
    auto compilation = renderer.compile_scene(scene);
    if (!compilation.ok()) {
        for (const auto &d : compilation.diagnostics) {
            std::cerr << d.message << '\n';
        }
        return false;
    }
    auto session = renderer.create_session(*compilation.scene, settings);
    if (!session) {
        std::cerr << "failed to create render session\n";
        return false;
    }
    psycles::io::MemoryOutputSink sink;
    if (!session->render_samples(
            {.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, sink)) {
        std::cerr << "render_samples failed\n";
        return false;
    }
    const auto *vol = sink.find(PassKind::volume_direct);
    if (vol == nullptr || vol->pixels.empty()) {
        std::cerr << "Volume Direct pass missing\n";
        return false;
    }
    out.volume_direct.assign(vol->pixels.begin(), vol->pixels.end());
    auto *compiled_impl =
        dynamic_cast<const psycles::luisa_backend::detail::LuisaCompiledScene *>(
            compilation.scene.get());
    out.guard_hits = compiled_impl != nullptr
                         ? compiled_impl->data()->last_volume_shadow_guard_hits
                         : 0ull;
    return true;
}

[[nodiscard]] bool test_interval_cursor_and_same_primitive(Device &device,
                                                          Stream &stream) {
    // Verify monotonic advance against repeated / zero / regressive committed_t
    // and verify same_primitive rejects distinct instances sharing prim index.
    constexpr std::uint32_t N = 20u;
    auto out_buf = device.create_buffer<float>(N);
    auto flag_buf = device.create_buffer<luisa::uint>(4u);
    Kernel1D k = [&]() noexcept {
        VolumeShadowIntervalCursor cursor{0.5f};
        Float prev_tmin = 0.5f;
        std::vector<float> fake_hits = {0.0f, 0.25f, 0.5f, 0.5f, 0.5f, 0.5001f};
        for (std::uint32_t i = 0u; i < 16u; ++i) {
            const float raw_hit = i < fake_hits.size() ? fake_hits[i] : 0.0f;
            const Float committed_t = max(prev_tmin, Expr<float>{raw_hit});
            const Float next_tmin = cursor.advance(committed_t);
            out_buf->write(i, next_tmin - prev_tmin);
            prev_tmin = next_tmin;
        }
        flag_buf->write(
            0u, select(0u, 1u, surface_ray::same_primitive(10u, 3u, 10u, 3u)));
        flag_buf->write(
            1u, select(0u, 1u, surface_ray::same_primitive(10u, 3u, 11u, 3u)));
        flag_buf->write(
            2u, select(0u, 1u, surface_ray::same_primitive(10u, 3u, 10u, 4u)));
        flag_buf->write(
            3u, select(0u, 1u, prev_tmin > 0.5f));
    };
    auto shader = device.compile(k, ShaderOption{.enable_cache = false});
    std::array<float, N> deltas{};
    std::array<luisa::uint, 4u> flags{};
    stream << shader().dispatch(1u)
           << out_buf.copy_to(deltas.data())
           << flag_buf.copy_to(flags.data())
           << synchronize();
    for (std::uint32_t i = 0u; i < 16u; ++i) {
        if (!(deltas[i] > 0.0f)) {
            std::cerr << "interval cursor failed to strictly advance at step "
                      << i << ": delta=" << deltas[i] << '\n';
            return false;
        }
    }
    if (flags[0] != 1u || flags[1] != 0u || flags[2] != 0u || flags[3] != 1u) {
        std::cerr << "same_primitive contract failed: flags=[" << flags[0]
                  << "," << flags[1] << "," << flags[2] << "," << flags[3]
                  << "]\n";
        return false;
    }
    return true;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};

    {
        auto dev = context.create_device(backend);
        auto st = dev.create_stream();
        if (!test_interval_cursor_and_same_primitive(dev, st)) {
            return EXIT_FAILURE;
        }
        std::cout << "[PASS] interval cursor monotonic advance & same_primitive (" << backend << ")\n";
    }

    const auto settings = make_test_settings();
    RenderedResult ref_res{};
    RenderedResult multi_res{};
    RenderedResult nested_res{};

    {
        auto dev = context.create_device(backend);
        LuisaPathTracerBackend renderer{
            std::move(dev),
            {.next_event_estimation = true, .max_samples_per_dispatch = 8u}};
        if (!run_render(renderer, make_multi_instance_absorber_scene(1u, false),
                        settings, ref_res)) {
            return EXIT_FAILURE;
        }
        if (ref_res.guard_hits != 0ull) {
            std::cerr << "unexpected guard hits in reference scene: "
                      << ref_res.guard_hits << '\n';
            return EXIT_FAILURE;
        }
        if (!run_render(renderer, make_multi_instance_absorber_scene(2u, true),
                        settings, multi_res)) {
            return EXIT_FAILURE;
        }
        if (multi_res.guard_hits != 0ull) {
            std::cerr << "unexpected guard hits in multi-instance scene: "
                      << multi_res.guard_hits << '\n';
            return EXIT_FAILURE;
        }
        if (!run_render(renderer, make_nested_instance_absorber_scene(3u),
                        settings, nested_res)) {
            return EXIT_FAILURE;
        }
        if (nested_res.guard_hits != 0ull) {
            std::cerr << "unexpected guard hits in nested-instance scene: "
                      << nested_res.guard_hits << '\n';
            return EXIT_FAILURE;
        }
    }

    const float expected_multi_ratio = std::exp(-1.5f);
    const float expected_nested_ratio = std::exp(-1.4f);
    double ref_sum = 0.0;
    double multi_sum = 0.0;
    double nested_sum = 0.0;
    for (std::size_t i = 0u; i < ref_res.volume_direct.size(); i += 3u) {
        const float r = ref_res.volume_direct[i];
        const float m = multi_res.volume_direct[i];
        const float n = nested_res.volume_direct[i];
        ref_sum += r;
        multi_sum += m;
        nested_sum += n;
        if (r > 1.0e-5f) {
            const float mr = m / r;
            const float nr = n / r;
            if (std::abs(mr - expected_multi_ratio) > 1.0e-3f) {
                std::cerr << "distinct-instance crossing transmittance mismatch at pixel "
                          << (i / 3u) << ": got ratio " << mr
                          << ", expected " << expected_multi_ratio << '\n';
                return EXIT_FAILURE;
            }
            if (std::abs(nr - expected_nested_ratio) > 1.0e-3f) {
                std::cerr << "nested-instance crossing transmittance mismatch at pixel "
                          << (i / 3u) << ": got ratio " << nr
                          << ", expected " << expected_nested_ratio << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    if (!(ref_sum > 1.0e-3)) {
        std::cerr << "reference volume_direct radiance unexpectedly near zero\n";
        return EXIT_FAILURE;
    }
    std::cout << "[PASS] distinct-instance transmittance ratio=" << (multi_sum / ref_sum)
              << " (expected " << expected_multi_ratio << ", guard_hits=" << multi_res.guard_hits << ")\n";
    std::cout << "[PASS] nested-instance transmittance ratio=" << (nested_sum / ref_sum)
              << " (expected " << expected_nested_ratio << ", guard_hits=" << nested_res.guard_hits << ")\n";

    // Force step guard trigger on the 7-boundary multi-instance scene by capping steps to 3.
    {
        ScopedEnvGuard max_steps_guard{"PSYCLES_VOLUME_SHADOW_MAX_STEPS", "3"};
        ScopedEnvGuard no_cache_guard{"PSYCLES_DISABLE_SHADER_CACHE", "1"};
        auto dev = context.create_device(backend);
        LuisaPathTracerBackend renderer{
            std::move(dev),
            {.next_event_estimation = true, .max_samples_per_dispatch = 8u}};
        RenderedResult guarded_res{};
        if (!run_render(renderer, make_multi_instance_absorber_scene(4u, true),
                        settings, guarded_res)) {
            return EXIT_FAILURE;
        }
        if (guarded_res.guard_hits == 0ull) {
            std::cerr << "expected volume shadow guard_hits > 0 when capped at 3 steps\n";
            return EXIT_FAILURE;
        }
        std::cout << "[PASS] step-limit guard triggered and logged (guard_hits="
                  << guarded_res.guard_hits << " on " << backend << ")\n";
    }

    return EXIT_SUCCESS;
}
