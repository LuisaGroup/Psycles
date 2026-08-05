#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <cmath>
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

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
  Mat4f result;
  result.elements[12u] = x;
  result.elements[13u] = y;
  result.elements[14u] = z;
  return result;
}

[[nodiscard]] ShaderGraph displaced_shader() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Cycles displacement geometry");
  const auto normal_to_vector = graph.add_node(node_type::normal_to_vector,
                                               "Normal to displacement vector");
  const auto scale =
      graph.add_node(node_type::vector_math, "Scale displacement normal");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Visible diffuse surface");
  static_cast<void>(
      graph.set_property(scale, "Operation", SocketValue::string("SCALE")));
  static_cast<void>(
      graph.set_input(scale, "Scale", SocketValue::floating(0.5f)));
  static_cast<void>(graph.connect({.node = geometry, .socket = "Normal"},
                                  normal_to_vector, "Normal"));
  static_cast<void>(graph.connect(
      {.node = normal_to_vector, .socket = "Vector"}, scale, "A"));
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  graph.set_root(ShaderDomain::displacement,
                 OutputRef{.node = scale, .socket = "Vector"});
  return graph;
}

[[nodiscard]] SceneSnapshot displacement_scene() {
  constexpr MaterialId material_id{1u};
  constexpr GeometryId geometry_id{2u};
  constexpr InstanceId instance_id{3u};
  constexpr CameraId camera_id{4u};

  SceneSnapshot scene;
  scene.revision = 1u;
  scene.materials.emplace(
      material_id,
      MaterialDesc{.name = "Cycles flat triangle displacement",
                   .shader = displaced_shader(),
                   .displacement_method = DisplacementMethod::displacement});

  TriangleMeshDesc mesh;
  mesh.name = "Flat-authored smooth displacement triangle";
  mesh.positions = {
      {-10.0f, -10.0f, 0.0f}, {10.0f, -10.0f, 0.0f}, {0.0f, 10.0f, 0.0f}};
  // Cycles evaluates true displacement at vertices with
  // SHADER_SMOOTH_NORMAL set even when the final triangle is flat. The
  // deliberately different point and face normals make that stage
  // contract observable in final acceleration geometry.
  mesh.normals.values.assign(mesh.positions.size(), Vec3f{0.0f, 0.6f, 0.8f});
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {material_id};
  mesh.triangle_material_slots = {0u};
  mesh.triangle_smooth = {0u};
  mesh.triangle_random_per_island = {0.0f};
  scene.geometries.emplace(geometry_id, std::move(mesh));
  scene.instances.emplace(instance_id,
                          InstanceDesc{.name = "Displaced triangle",
                                       .geometry = geometry_id,
                                       .transform = {}});
  scene.cameras.emplace(camera_id,
                        CameraDesc{.name = "Displacement camera",
                                   .projection = CameraProjection::orthographic,
                                   .transform = translated(0.0f, 0.0f, 3.0f),
                                   .orthographic_scale = 1.0f,
                                   .near_clip = 0.1f,
                                   .far_clip = 100.0f});
  scene.active_camera = camera_id;
  scene.world_sampling = WorldSampling::none;
  return scene;
}

[[nodiscard]] RenderSettings render_settings() {
  return {.full_extent = {.width = 1u, .height = 1u},
          .window = {},
          .seed = 0u,
          .transparent_background = false,
          .pixel_filter = PixelFilter::box,
          .filter_width = 1.0f,
          .pass_alpha_threshold = 0.5f,
          .integrator = {.max_bounces = 1u,
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
                             DirectLightSampling::forward_path_tracing},
          .passes = {{.kind = PassKind::combined,
                      .name = "Combined",
                      .light_group = {},
                      .channels = 4u}}};
}

class CapturingPathTraceSink final
    : public psycles::luisa_backend::LuisaPathTraceSink {

private:
  std::optional<psycles::luisa_backend::LuisaPathTrace> _trace;

public:
  void write(const psycles::luisa_backend::LuisaPathTrace &trace) override {
    _trace = trace;
  }

  [[nodiscard]] const auto &trace() const noexcept { return _trace; }
};

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  luisa::compute::Context context{argv[0]};
  auto device = context.create_device(backend);
  auto trace_sink = std::make_shared<CapturingPathTraceSink>();
  psycles::luisa_backend::LuisaPathTracerBackend renderer{
      std::move(device),
      {.next_event_estimation = false,
       .max_samples_per_dispatch = 1u,
       .path_trace = psycles::luisa_backend::LuisaPathTraceRequest{
           .pixel_x = 0u, .pixel_y = 0u, .sample = 0u, .sink = trace_sink}}};

  auto compilation = renderer.compile_scene(displacement_scene());
  if (!compilation.ok()) {
    for (const auto &diagnostic : compilation.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    return EXIT_FAILURE;
  }
  auto session = renderer.create_session(*compilation.scene, render_settings());
  if (!session) {
    std::cerr << "could not create displacement session on " << backend << '\n';
    return EXIT_FAILURE;
  }
  psycles::io::MemoryOutputSink output;
  if (!session->render_samples(
          {.first = 0u, .count = 1u, .offset = 0u, .total = 1u}, output)) {
    std::cerr << "displacement render failed on " << backend << '\n';
    return EXIT_FAILURE;
  }
  if (!trace_sink->trace()) {
    std::cerr << "displacement trace was not delivered on " << backend << '\n';
    return EXIT_FAILURE;
  }

  using psycles::luisa_backend::path_trace_schema::EventSlot;
  using psycles::luisa_backend::path_trace_schema::index;
  const auto &surface =
      trace_sink->trace()->slots[index(0u, EventSlot::surface_p)];
  if (surface[3u] != 1.0f || std::abs(surface[2u] - 0.4f) > 2.0e-6f) {
    std::cerr << "Cycles forced-smooth displacement regression on " << backend
              << ": surface z=" << surface[2u] << ", expected 0.4\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
