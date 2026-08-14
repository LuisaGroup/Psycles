#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_trace_schema.h>
#include <psycles/luisa/path_tracer.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
namespace trace_schema = psycles::luisa_backend::path_trace_schema;

constexpr auto width = 3u;
constexpr auto height = 2u;
constexpr auto sample_count = 8u;
constexpr auto traced_sample = 5u;

constexpr std::array pass_kinds{PassKind::combined,
    PassKind::normal,
    PassKind::albedo,
    PassKind::glossy_color,
    PassKind::transmission_color,
    PassKind::emission,
    PassKind::environment,
    PassKind::diffuse_direct,
    PassKind::diffuse_indirect,
    PassKind::glossy_direct,
    PassKind::glossy_indirect,
    PassKind::transmission_direct,
    PassKind::transmission_indirect,
    PassKind::volume_direct,
    PassKind::volume_indirect,
    PassKind::sample_count};

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
    Mat4f result;
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] ShaderGraph surface_shader() {
    ShaderGraph graph;
  const auto principled =
      graph.add_node(node_type::principled_bsdf, "Per-sample dispatch surface");
    const auto configured =
      graph.set_input(principled, "BaseColor",
            SocketValue::color({0.23f, 0.51f, 0.71f})) &&
      graph.set_input(principled, "Metallic", SocketValue::floating(0.17f)) &&
      graph.set_input(principled, "Roughness", SocketValue::floating(0.31f)) &&
      graph.set_input(principled, "EmissionColor",
            SocketValue::color({0.04f, 0.015f, 0.007f})) &&
      graph.set_input(principled, "EmissionStrength",
            SocketValue::floating(0.25f));
    if (!configured) {
        std::abort();
    }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph world_shader() {
    ShaderGraph graph;
  const auto emission =
      graph.add_node(node_type::emission, "Per-sample dispatch world");
    const auto configured =
      graph.set_input(emission, "Color",
            SocketValue::color({0.12f, 0.18f, 0.27f})) &&
      graph.set_input(emission, "Strength", SocketValue::floating(0.35f));
    if (!configured) {
        std::abort();
    }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot make_scene() {
    constexpr MaterialId surface_material{1u};
    constexpr MaterialId world_material{2u};
    constexpr GeometryId geometry{3u};
    constexpr InstanceId instance{4u};
    constexpr CameraId camera{5u};
    constexpr LightId light{6u};

    SceneSnapshot scene;
    scene.revision = 1u;
  scene.materials.emplace(surface_material,
                          MaterialDesc{.name = "Per-sample dispatch surface",
            .shader = surface_shader(),
            .cycles_shader_index = 0u});
  scene.materials.emplace(world_material,
                          MaterialDesc{.name = "Per-sample dispatch world",
            .shader = world_shader(),
            .cycles_shader_index = 1u});

    TriangleMeshDesc mesh;
    mesh.name = "Per-sample dispatch receiver";
    mesh.positions = {
      {-4.0f, -4.0f, 0.0f}, {4.0f, -4.0f, 0.0f}, {0.0f, 4.0f, 0.0f}};
  mesh.normals.values.assign(mesh.positions.size(), Vec3f{0.0f, 0.0f, 1.0f});
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {surface_material};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {1u};
    mesh.triangle_random_per_island = {0.0f};
    scene.geometries.emplace(geometry, std::move(mesh));
  scene.instances.emplace(instance,
                          InstanceDesc{.name = "Per-sample dispatch receiver",
            .geometry = geometry,
            .transform = {},
            .cycles_object_index = 0u});

  scene.cameras.emplace(camera,
                        CameraDesc{.name = "Per-sample dispatch camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .orthographic_scale = 2.0f,
            .near_clip = 0.1f,
            .far_clip = 100.0f});
    scene.active_camera = camera;

  scene.lights.emplace(light,
                       LightDesc{.name = "Per-sample dispatch point light",
            .type = LightType::point,
            .transform = translated(0.4f, -0.25f, 2.0f),
            .color = {1.0f, 0.73f, 0.41f},
            .power = 35.0f,
            .size = 0.1f,
            .normalize = true,
            .is_sphere = true,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask = all_ray_visibility,
            .cycles_shader_index = 2u,
            .cycles_object_index = 1u});
    scene.world_shader = world_material;
    scene.world_sampling = WorldSampling::automatic;
    scene.cycles_background_object_index = 2u;
    return scene;
}

[[nodiscard]] RenderSettings make_settings() {
    RenderSettings settings{
        .full_extent = {.width = width, .height = height},
        .window = {},
        .seed = 11939u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::box,
        .filter_width = 1.0f,
        .pass_alpha_threshold = 0.5f,
      .integrator = {.max_bounces = 2u,
            .min_bounces = 0u,
            .diffuse_bounces = 1u,
            .glossy_bounces = 1u,
            .transmission_bounces = 1u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 2u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold = 0.01f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::multiple_importance_sampling}};
    settings.passes.reserve(pass_kinds.size());
    for (const auto kind : pass_kinds) {
    settings.passes.emplace_back(
        PassRequest{.kind = kind,
            .name = "dispatch-regression",
                    .channels = kind == PassKind::combined       ? 4u
                                : kind == PassKind::sample_count ? 1u
                                  : 3u});
    }
    return settings;
}

class TraceSink final : public psycles::luisa_backend::LuisaPathTraceSink {

public:
    std::optional<psycles::luisa_backend::LuisaPathTrace> trace;

  void write(const psycles::luisa_backend::LuisaPathTrace &value) override {
        trace = value;
    }
};

struct RenderResult {
    psycles::io::MemoryOutputSink output;
  std::optional<psycles::luisa_backend::LuisaPathTrace> trace;
};

[[nodiscard]] std::optional<RenderResult>
render(luisa::compute::Context &context, std::string_view backend,
    psycles::luisa_backend::LuisaPathScheduler scheduler,
       std::uint32_t samples_per_dispatch, bool split_request,
       bool staged_surface_sorting = true, bool path_trace_enabled = true,
       bool staged_direct_light_queue = false,
       std::uint32_t wavefront_tail_megakernel_threshold = 4096u,
       std::uint32_t wavefront_counter_readback_batch_size = 4u,
       std::uint32_t wavefront_counter_readback_pipeline_depth = 2u,
       std::uint32_t wavefront_frame_capacity = 128u) {
    auto device = context.create_device(backend);
  auto trace_sink = path_trace_enabled ? std::make_shared<TraceSink>()
                                       : std::shared_ptr<TraceSink>{};
  const auto trace_request =
      trace_sink
          ? std::optional{psycles::luisa_backend::LuisaPathTraceRequest{
                .pixel_x = 1u,
                .pixel_y = 1u,
                .sample = traced_sample,
                .sink = trace_sink}}
          : std::optional<psycles::luisa_backend::LuisaPathTraceRequest>{};
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .scheduler = scheduler,
         .wavefront_frame_capacity = wavefront_frame_capacity,
         .wavefront_graph_worker_count = 5u,
         .wavefront_graph_selective_scheduling =
             scheduler ==
             psycles::luisa_backend::LuisaPathScheduler::wavefront_graph,
         .wavefront_counter_readback_batch_size =
             wavefront_counter_readback_batch_size,
         .wavefront_counter_readback_pipeline_depth =
             wavefront_counter_readback_pipeline_depth,
         .wavefront_tail_megakernel_threshold =
             wavefront_tail_megakernel_threshold,
         .staged_surface_sorting = staged_surface_sorting,
       .staged_direct_light_queue = staged_direct_light_queue,
         .persistent_worker_count = 128u,
         .persistent_block_size = 64u,
         .persistent_fetch_size = 4u,
         .max_samples_per_dispatch = samples_per_dispatch,
       .path_trace = trace_request}};
    auto compilation = renderer.compile_scene(make_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return std::nullopt;
    }
  auto session = renderer.create_session(*compilation.scene, make_settings());
    if (!session) {
        return std::nullopt;
    }

    psycles::io::MemoryOutputSink output;
    if (split_request) {
        psycles::io::MemoryOutputSink partial;
        if (!session->render_samples(
            {.first = 0u, .count = 3u, .offset = 0u, .total = sample_count},
                partial) ||
        !session->render_samples({.first = 3u,
                 .count = sample_count - 3u,
                 .offset = 0u,
                 .total = sample_count},
                output)) {
            return std::nullopt;
        }
  } else if (!session->render_samples({.first = 0u,
                    .count = sample_count,
                    .offset = 0u,
                    .total = sample_count},
                   output)) {
        return std::nullopt;
    }
  if (trace_sink && !trace_sink->trace) {
        return std::nullopt;
    }
  return RenderResult{.output = std::move(output),
                      .trace = trace_sink ? std::move(trace_sink->trace)
                                          : std::nullopt};
}

[[nodiscard]] bool same_bits(float lhs, float rhs) noexcept {
  return std::bit_cast<std::uint32_t>(lhs) == std::bit_cast<std::uint32_t>(rhs);
}

[[nodiscard]] bool close(float lhs, float rhs,
                         float tolerance = 2.0e-5f) noexcept {
    return std::isfinite(lhs) && std::isfinite(rhs) &&
           std::abs(lhs - rhs) <=
               tolerance * std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

[[nodiscard]] bool compare_outputs(const RenderResult &reference,
                                   const RenderResult &candidate, bool exact,
                                   std::string_view label) {
    for (const auto kind : pass_kinds) {
        const auto *expected = reference.output.find(kind);
        const auto *actual = candidate.output.find(kind);
        if (expected == nullptr || actual == nullptr ||
            expected->channels != actual->channels ||
            expected->extent.width != actual->extent.width ||
            expected->extent.height != actual->extent.height ||
            expected->pixels.size() != actual->pixels.size()) {
            std::cerr << label << " changed pass shape "
                      << static_cast<std::uint32_t>(kind) << '\n';
            return false;
        }
        for (auto i = std::size_t{0u}; i < expected->pixels.size(); ++i) {
            const auto matches = exact ?
                                     same_bits(expected->pixels[i],
                                               actual->pixels[i]) :
                                     close(expected->pixels[i],
                                           actual->pixels[i]);
            if (!matches) {
                std::cerr << label << " changed pass "
                          << static_cast<std::uint32_t>(kind) << " value " << i
                          << ": expected " << expected->pixels[i] << ", got "
                          << actual->pixels[i] << '\n';
                return false;
            }
        }
    }
    if (reference.trace.has_value() != candidate.trace.has_value()) {
        std::cerr << label << " changed path-trace availability\n";
        return false;
    }
    if (reference.trace) {
        for (auto slot = std::size_t{0u}; slot < reference.trace->slots.size();
             ++slot) {
            for (auto component = std::size_t{0u};
                 component < reference.trace->slots[slot].size(); ++component) {
                const auto expected = reference.trace->slots[slot][component];
                const auto actual = candidate.trace->slots[slot][component];
                const auto matches = exact ? same_bits(expected, actual)
                                           : close(expected, actual);
                if (!matches) {
                    std::cerr
                        << label << " changed path trace at slot " << slot
                        << ", component " << component << ": expected "
                        << expected << " (0x" << std::hex
                        << std::bit_cast<std::uint32_t>(expected) << "), got "
                        << std::dec << actual << " (0x" << std::hex
                        << std::bit_cast<std::uint32_t>(actual) << ")\n"
                        << std::dec;
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool validate_reference(const RenderResult &result) {
    const auto *samples = result.output.find(PassKind::sample_count);
    const auto *combined = result.output.find(PassKind::combined);
    const auto *normal = result.output.find(PassKind::normal);
    const auto *albedo = result.output.find(PassKind::albedo);
    if (samples == nullptr || combined == nullptr || normal == nullptr ||
        albedo == nullptr || samples->pixels.size() != width * height) {
        return false;
    }
  if (!std::all_of(samples->pixels.begin(), samples->pixels.end(),
            [](float value) noexcept {
                return value == static_cast<float>(sample_count);
            })) {
        std::cerr << "sample-count pass did not record every (pixel, sample)\n";
        return false;
    }
    const auto has_energy = [](const auto &image) noexcept {
        return std::any_of(
        image.pixels.begin(), image.pixels.end(), [](float value) noexcept {
                return std::isfinite(value) && std::abs(value) > 1.0e-6f;
            });
    };
  if (!has_energy(*combined) || !has_energy(*normal) || !has_energy(*albedo)) {
        std::cerr << "dispatch fixture did not exercise primary film passes\n";
        return false;
    }
  if (!result.trace) {
    std::cerr << "reference path trace is missing\n";
    return false;
  }
  const auto &rng =
      result.trace->slots[trace_schema::index(trace_schema::GlobalSlot::rng)];
  if (result.trace->sample != traced_sample ||
        rng[0u] != static_cast<float>(traced_sample)) {
        std::cerr << "dispatch.z did not preserve the absolute sample index\n";
        return false;
    }
    return true;
}

}// namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};

    const auto reference = render(
      context, backend, psycles::luisa_backend::LuisaPathScheduler::megakernel,
      sample_count, false);
    // The serial film path is the deterministic diagnostic oracle. Splitting
    // the same ordered sample range across host dispatches must remain
    // bit-exact for every pass; this catches accidental changes to exact-hash
    // validation without imposing bit identity on floating-point atomics.
    const auto deterministic = render(
      context, backend, psycles::luisa_backend::LuisaPathScheduler::megakernel,
      sample_count, true);
  const auto single_plane =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::megakernel_per_sample,
             1u, false);
  const auto per_sample =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::megakernel_per_sample,
             sample_count, false);
  // Same atomic per-(pixel,sample) film topology as the deferred shadow
  // coroutine below, but without coroutine cuts. This isolates the semantic
  // effect of moving BSDF continuation construction before the shadow state
  // machine from host chunking and serial-vs-atomic accumulation order.
  const auto per_sample_no_trace =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::megakernel_per_sample,
             sample_count, false, true, false);
  const auto chunked =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::megakernel_per_sample,
             sample_count, true);
    const auto wavefront = render(
      context, backend, psycles::luisa_backend::LuisaPathScheduler::wavefront,
      sample_count, false);
  const auto graph_wavefront =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_graph,
             sample_count, false, true, true, false, 0u);
  // Exercise the CoroGraph-derived state-machine tail on the same graph and
  // all-pass film contract. The threshold exceeds this fixture's 48 logical
  // invocations, so any residual work after admission is eligible.
  const auto graph_wavefront_tail =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_graph,
             sample_count, false, true, true, false, 128u, 1u, 1u);
  const auto staged_wavefront =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_staged,
             sample_count, false);
  const auto staged_wavefront_unsorted =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_staged,
             sample_count, false, false);
  // Same staged per-(pixel,sample) topology, with only the host/JIT
  // direct-light execution policy changed. No diagnostic trace is requested
  // because it intentionally keeps shadow evaluation inline and ordered.
  const auto staged_direct_inline =
      render(context, backend,
        psycles::luisa_backend::LuisaPathScheduler::wavefront_staged,
             sample_count, false, true, false, false);
  const auto staged_direct_queued =
      render(context, backend,
        psycles::luisa_backend::LuisaPathScheduler::wavefront_staged,
             sample_count, false, true, false, true);
  const auto staged_direct_queued_chunked =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_staged, 3u,
             true, true, false, true);
  // A capacity below the six simultaneously resident pixel frames forces
  // refill and direct-light queue back-pressure across the 48 logical
  // (pixel, sample) instances. Runtime SoA offsets must preserve the same film
  // while the shader structure remains capacity-independent.
  const auto staged_direct_queued_small_capacity =
      render(context, backend,
             psycles::luisa_backend::LuisaPathScheduler::wavefront_staged,
             sample_count, false, true, false, true, 4096u, 4u, 2u, 3u);
    const auto persistent = render(
      context, backend, psycles::luisa_backend::LuisaPathScheduler::persistent,
      sample_count, false);
    if (!reference || !deterministic || !single_plane || !per_sample ||
      !per_sample_no_trace || !chunked || !wavefront || !graph_wavefront ||
      !graph_wavefront_tail || !staged_wavefront ||
      !staged_wavefront_unsorted || !staged_direct_inline ||
      !staged_direct_queued || !staged_direct_queued_chunked ||
      !staged_direct_queued_small_capacity || !persistent ||
        !validate_reference(*reference) ||
      !compare_outputs(*reference, *deterministic, true,
            "deterministic serial chunking") ||
      !compare_outputs(*reference, *single_plane, false,
            "single-plane atomic dispatch") ||
      !compare_outputs(*reference, *per_sample, false,
            "batched per-sample dispatch") ||
      !compare_outputs(*reference, *chunked, false,
            "chunked per-sample dispatch") ||
      !compare_outputs(*per_sample_no_trace, *staged_direct_inline, false,
                       "deferred shadow after surface continuation") ||
      !compare_outputs(*reference, *wavefront, false, "wavefront dispatch") ||
      !compare_outputs(*reference, *graph_wavefront, false,
                       "graph wavefront dispatch") ||
      !compare_outputs(*reference, *graph_wavefront_tail, false,
                       "graph wavefront tail dispatch") ||
      !compare_outputs(*reference, *staged_wavefront, false,
            "staged wavefront dispatch") ||
      !compare_outputs(*reference, *staged_wavefront_unsorted, false,
            "staged wavefront dispatch without surface sorting") ||
      !compare_outputs(*staged_direct_inline, *staged_direct_queued, false,
                       "queued direct-light visibility") ||
      !compare_outputs(*staged_direct_inline, *staged_direct_queued_chunked,
                       false, "chunked queued direct-light visibility") ||
      !compare_outputs(*staged_direct_inline,
                       *staged_direct_queued_small_capacity, false,
                       "small-capacity queued direct-light visibility") ||
      !compare_outputs(*reference, *persistent, false, "persistent dispatch")) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
