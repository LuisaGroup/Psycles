#include "cycles_lamp_routing_fixture.h"
#include "path_kernel_builder.h"
#include "path_kernel_transitions.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/path_tracer.h>

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/luisa-compute.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace psycles::luisa_backend::detail;
namespace contract = psycles::contract;
namespace f = psycles::test_support::lamp_routing;
namespace path = psycles::luisa_backend::cycles_path_state;

template <typename Function> Function unbound() {
  return Function{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

PathKernelConfig make_config(std::shared_ptr<LuisaSceneData> scene) {
  return {
      .scene = scene,
      .light_transport = make_light_transport_callables(
          contract::DirectLightSampling::multiple_importance_sampling),
      .light_distribution_sample = unbound<LightDistributionSampleCallable>(),
      .light_tree = {unbound<LightTreeSampleCallable>(),
                     unbound<LightTreeSampleCallable>(),
                     unbound<LightTreePdfCallable>(),
                     unbound<LightTreePdfCallable>(),
                     unbound<LightTreeForwardPdfCallable>(),
                     unbound<LightTreeTriangleEmitterCallable>()},
      .surfaces = make_surface_callables(scene),
      .background_sampling = std::make_shared<BackgroundSamplingDistribution>(),
      .shade_shadow_surface = unbound<EvaluateShadowSurfaceCallable>(),
      .trace_shadow = unbound<TraceShadowCallable>()};
}

contract::SceneSnapshot fixture() {
  contract::SceneSnapshot scene;
  const contract::CameraId camera{1};
  scene.cameras.emplace(camera, contract::CameraDesc{});
  scene.active_camera = camera;
  // A genuine, off-ray mesh keeps the production scene traversal bound.
  // Every fixture ray misses it, matching the supplied Cycles BVH result.
  contract::ShaderGraph graph;
  const auto closure = graph.add_node(
      psycles::compiler::node_type::null_closure, "Off-ray material");
  graph.set_root(contract::ShaderDomain::surface,
                 contract::OutputRef{closure, "Closure"});
  const contract::MaterialId material{1};
  const contract::GeometryId geometry{1};
  scene.materials.emplace(material,
                          contract::MaterialDesc{.shader = std::move(graph)});
  contract::TriangleMeshDesc mesh;
  mesh.positions = {{10, 0, 10}, {11, 0, 10}, {10, 1, 10}};
  mesh.triangles = {{0, 1, 2}};
  mesh.material_slots = {material};
  scene.geometries.emplace(geometry, std::move(mesh));
  scene.instances.emplace(
      contract::InstanceId{1},
      contract::InstanceDesc{.geometry = geometry, .cycles_object_index = 0});
  for (unsigned i = 0; i < f::light_count; ++i) {
    psycles::Mat4f transform;
    transform.elements[0] = transform.elements[10] = f::axis;
    transform.elements[2] = -f::axis;
    transform.elements[8] = f::axis;
    transform.elements[14] = float(i + 1);
    scene.lights.emplace(
        contract::LightId{i + 1},
        contract::LightDesc{.name = "Dark spot endpoint",
                            .type = contract::LightType::spot,
                            .transform = transform,
                            .color = {1, 1, 1},
                            .power = 1,
                            .size = f::radius,
                            .spot_angle = f::spot_angle,
                            .spot_smooth = 0.5f,
                            .normalize = true,
                            .is_sphere = true,
                            .use_mis = true,
                            .cycles_object_index = f::first_light_object + i});
  }
  return scene;
}

bool run(const char *program, const char *backend) {
  struct Oracle {
    std::array<unsigned, 12> value{};
    float tmin{};
  };
  std::array<Oracle, f::cases> expected{};
  std::ifstream source{PSYCLES_LAMP_ROUTING_ORACLE};
  for (unsigned mode = 0; mode < f::modes; ++mode) {
    for (unsigned index = 0; index < f::transparent_limits.size(); ++index) {
      char tag{};
      unsigned m{}, i{};
      source >> tag >> m >> i;
      auto &row = expected[mode * f::transparent_limits.size() + index];
      for (auto &value : row.value) {
        source >> value;
      }
      source >> row.tmin;
      if (!source || tag != 'Q' || m != mode || i != index) {
        return false;
      }
    }
  }
  std::string extra;
  if (source >> extra) {
    return false;
  }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  psycles::luisa_backend::LuisaPathTracerBackend compiler{
      Device{device.impl_shared()}};
  const auto compilation = compiler.compile_scene(fixture());
  for (const auto &d : compilation.diagnostics) {
    std::cerr << d.message << '\n';
  }
  if (!compilation.ok()) {
    return false;
  }
  const auto scene =
      dynamic_cast<const LuisaCompiledScene &>(*compilation.scene).data();
  if (scene->light_count != f::light_count) {
    return false;
  }
  auto config = make_config(scene);
  const PathKernelPipeline pipeline{config};
  constexpr unsigned paths = 43;
  auto output = device.create_buffer<luisa::uint4>(paths);
  auto distances = device.create_buffer<luisa::float4>(paths);
  auto dummy = device.create_buffer<float>(1);
  auto counts = device.create_buffer<unsigned>(1);
  const auto record = [&](UInt limit, UInt mode,
                          const Var<Buffer<luisa::uint4>> &out,
                          const BufferFloat4 &floats, const BufferFloat &unused,
                          const BufferUInt &unused_counts,
                          PathCoroutineCutPolicy policy) {
    UInt first = 0;
    Var<RenderKernelParameters> parameters;
    parameters.transparent_max_bounces = limit;
    parameters.sobol_sequence_size = 1;
    parameters.camera_transform = parameters.camera_inverse_transform =
        make_float4x4(1);
    PathKernelInvocation invocation{.config = config,
                                    .film_accumulation =
                                        PathFilmAccumulation::serial,
                                    .combined = floats,
                                    .normal = floats,
                                    .albedo = floats,
                                    .light_passes = floats,
                                    .sample_count = unused_counts,
                                    .volume_guiding_raw = floats,
                                    .volume_guiding_denoised = unused_counts,
                                    .path_trace = floats,
                                    .sample_first = first,
                                    .sobol_table = floats,
                                    .filter_table = unused,
                                    .parameters = parameters};
    PathSampleContext sample{.invocation = invocation};
    sample.ray =
        make_ray(make_float3(0), make_float3(0, 0, 1), 0.0f, f::maximum);
    sample.throughput = make_float3(1);
    sample.ray_source_object = f::initial_object;
    sample.ray_source_primitive = f::initial_primitive;
    sample.path_flags = path::flag_mis_skip | path::flag_reflect;
    sample.cycles_path_visibility =
        select(path::visibility_diffuse, path::visibility_camera, mode == 0);
    sample.cycles_rng_offset = f::rng_offset;
    pipeline.emit(sample, policy);
    out.write(dispatch_x(),
              make_uint4(sample.transparent_depth, sample.cycles_rng_offset,
                         sample.ray_source_object,
                         sample.ray_source_primitive));
    floats.write(dispatch_x(),
                 make_float4(sample.ray->t_min(), sample.radiance));
  };
  bool passed = true;
  const auto check_state = [&](const char *layout, unsigned mode, unsigned index,
                               const auto &actual, const auto &real) {
    const auto &e = expected[mode * f::transparent_limits.size() + index];
    for (unsigned i = 0; i < paths; ++i) {
      const auto v = actual[i];
      if (v.x != e.value[3] || v.y != e.value[4] || v.z != e.value[5] ||
          v.w != e.value[6] || !std::isfinite(real[i].x) ||
          std::abs(real[i].x - e.tmin) > 1.0e-5f || real[i].y != 0 ||
          real[i].z != 0 || real[i].w != 0) {
        if (i == 0) {
          std::cerr << "state layout=" << layout << " mode=" << mode
                    << " limit=" << f::transparent_limits[index]
                    << " actual=" << v.x << ',' << v.y << ',' << v.z
                    << ',' << v.w << " tmin=" << real[i].x << '\n';
        }
        passed = false;
      }
    }
  };
  // The same production pipeline without suspension separates ordinary
  // backend control flow from continuation/frame transfer. Both are checked
  // against Cycles, not against each other's output.
  Kernel1D monolithic{[&](UInt limit, UInt mode, Var<Buffer<luisa::uint4>> out,
                          BufferFloat4 floats, BufferFloat unused,
                          BufferUInt unused_counts) {
    record(limit, mode, out, floats, unused, unused_counts,
           PathCoroutineCutPolicy::none);
  }};
  auto direct = device.compile(
      monolithic, {.enable_cache = false, .enable_fast_math = true});
  for (unsigned mode = 0; mode < f::modes; ++mode) {
    for (unsigned index = 0; index < f::transparent_limits.size(); ++index) {
      std::array<luisa::uint4, paths> actual{};
      std::array<luisa::float4, paths> real{};
      stream << direct(f::transparent_limits[index], mode, output, distances,
                       dummy, counts).dispatch(paths)
             << output.copy_to(actual.data())
             << distances.copy_to(real.data()) << synchronize();
      check_state("monolithic", mode, index, actual, real);
    }
  }
  Coroutine<void(unsigned, unsigned, Buffer<luisa::uint4>,
                 Buffer<luisa::float4>, Buffer<float>, Buffer<unsigned>)>
      coroutine{[&](UInt limit, UInt mode, Var<Buffer<luisa::uint4>> out,
                    BufferFloat4 floats, BufferFloat unused,
                    BufferUInt unused_counts) {
        record(limit, mode, out, floats, unused, unused_counts,
               PathCoroutineCutPolicy::cycles_wavefront);
      }};
  for (bool soa : {true, false}) {
    WavefrontCoroScheduler scheduler{
        device,
        coroutine,
        {.thread_count = 19,
         .global_memory_soa = soa,
         .report_stats = true,
         .shader_option = {.enable_cache = false, .enable_fast_math = true},
         .execution_block_size = 32,
         .largest_continuation_first = true,
         .incremental_continuation_counts = true}};
    for (unsigned mode = 0; mode < f::modes; ++mode) {
      for (unsigned index = 0; index < f::transparent_limits.size(); ++index) {
        const auto &e = expected[mode * f::transparent_limits.size() + index];
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
          std::array<luisa::uint4, paths> actual{};
          std::array<luisa::float4, paths> real{};
          stream << scheduler(f::transparent_limits[index], mode, output,
                              distances, dummy, counts)
                        .dispatch(paths)
                 << output.copy_to(actual.data())
                 << distances.copy_to(real.data()) << synchronize();
          const auto &stats = scheduler.last_dispatch_stats();
          unsigned stage = 0;
          for (const auto name : {path_transition::intersect_closest,
                                  path_transition::shade_light_forward,
                                  path_transition::shade_background}) {
            const auto *node = coroutine.graph().node_by_name(name);
            const auto visits =
                node ? stats.continuations[node->index].executed_count : 0;
            if (visits != std::uint64_t(e.value[stage++]) * paths) {
              std::cerr << "soa=" << soa << " mode=" << mode
                        << " limit=" << f::transparent_limits[index]
                        << " queue=" << name << " visits=" << visits
                        << " expected=" << e.value[stage - 1] * paths << '\n';
              passed = false;
            }
          }
          check_state(soa ? "soa" : "aos", mode, index, actual, real);
        }
      }
    }
  }
  std::cout << "Production lamp routing: " << f::cases
            << " cases, monolithic/SoA/AoS, repeated capacity refills; passed=" << passed
            << '\n';
  return passed;
}
} // namespace
int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
