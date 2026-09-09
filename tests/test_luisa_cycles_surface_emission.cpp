#include "cycles_surface_emission_fixture.h"
#include "cycles_surface_emission_test_support.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
namespace f = psycles::test_support::surface_emission_fixture;
namespace d = psycles::test_support::surface_emission;
namespace path = psycles::luisa_backend::cycles_path_state;
namespace closure = psycles::luisa_backend::cycles_closure;

struct Oracle {
  luisa::float3 emission;
  std::array<float, f::film_lanes> film{};
  std::array<unsigned, f::film_lanes> native_write_counts{};
};

bool camera_integration(Device &device, const Oracle &oracle) {
  namespace c = psycles::contract;
  namespace b = psycles::luisa_backend;
  c::SceneSnapshot scene;
  c::ShaderGraph graph;
  const auto node = graph.add_node(psycles::compiler::node_type::emission,
                                   "Native emission film");
  if (!graph.set_input(node, "Color", c::SocketValue::color({2, 4, 1})) ||
      !graph.set_input(node, "Strength", c::SocketValue::floating(1))) {
    return false;
  }
  graph.set_root(c::ShaderDomain::surface, c::OutputRef{node, "Closure"});
  scene.materials.emplace(
      c::MaterialId{1},
      c::MaterialDesc{.shader = std::move(graph), .cycles_shader_index = 0});
  c::TriangleMeshDesc mesh;
  mesh.positions = {{-10, -10, 0}, {10, -10, 0}, {0, 10, 0}};
  mesh.triangles = {{0, 1, 2}};
  mesh.material_slots = {c::MaterialId{1}};
  scene.geometries.emplace(c::GeometryId{1}, std::move(mesh));
  scene.instances.emplace(
      c::InstanceId{1},
      c::InstanceDesc{.geometry = c::GeometryId{1}, .cycles_object_index = 0});
  psycles::Mat4f transform;
  transform.elements[14] = 3;
  scene.cameras.emplace(
      c::CameraId{1},
      c::CameraDesc{.projection = c::CameraProjection::orthographic,
                    .transform = transform,
                    .orthographic_scale = 1});
  scene.active_camera = c::CameraId{1};
  scene.world_sampling = c::WorldSampling::none;
  for (bool nee : {false, true}) {
    for (auto scheduler : {b::LuisaPathScheduler::megakernel,
                           b::LuisaPathScheduler::wavefront_staged}) {
      // Keep a single native Device. Vulkan's process-global Volk dispatch
      // tables do not permit a second live Device alongside the probe.
      b::LuisaPathTracerBackend renderer{Device{device.impl_shared()},
                                         {.next_event_estimation = nee,
                                          .scheduler = scheduler,
                                          .wavefront_frame_capacity = 32u,
                                          .max_samples_per_dispatch = 4u}};
      auto compiled = renderer.compile_scene(scene);
      for (const auto &d : compiled.diagnostics) {
        std::cerr << d.message << '\n';
      }
      if (!compiled.ok()) {
        return false;
      }
      c::RenderSettings settings{.full_extent = {2, 2},
                                 .pixel_filter = c::PixelFilter::box,
                                 .filter_width = 1.0f};
      settings.integrator.max_bounces = 1;
      // Both scene/integrator.cpp and Psycles scene upload multiply authored
      // average-channel clamp settings by three for the kernel's RGB sum.
      settings.integrator.sample_clamp_direct = f::direct_limit / 3.0f;
      settings.integrator.sample_clamp_indirect = f::indirect_limit / 3.0f;
      settings.integrator.direct_light_sampling =
          nee ? c::DirectLightSampling::multiple_importance_sampling
              : c::DirectLightSampling::forward_path_tracing;
      settings.passes = {{.kind = c::PassKind::combined, .channels = 4},
                         {.kind = c::PassKind::emission, .channels = 3}};
      auto session = renderer.create_session(*compiled.scene, settings);
      if (!session) {
        return false;
      }
      psycles::io::MemoryOutputSink output;
      if (!session->render_samples(
              {.first = 0, .count = 4, .offset = 0, .total = 4}, output)) {
        return false;
      }
      for (auto kind : {c::PassKind::combined, c::PassKind::emission}) {
        const auto image = output.find(kind);
        if (!image) {
          return false;
        }
        const unsigned offset = kind == c::PassKind::combined ? 0 : 36;
        for (unsigned p = 0; p < 4; ++p) {
          for (unsigned lane = 0; lane < 3; ++lane) {
            const auto value = image->pixels[p * image->channels + lane];
            const auto expected = oracle.film[offset + lane];
            if (!std::isfinite(value) || std::abs(value - expected) > 2e-6f) {
              std::cerr << "Full surface emission nee=" << nee
                        << " scheduler=" << unsigned(scheduler)
                        << " lane=" << lane << " actual=" << value
                        << " Cycles=" << expected << '\n';
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

bool run(const char *program, const char *backend) {
  std::array<Oracle, f::oracle_cases> oracle{};
  std::ifstream source{PSYCLES_SURFACE_EMISSION_ORACLE};
  for (unsigned i = 0; i < f::oracle_cases; ++i) {
    char tag{};
    unsigned index{};
    auto &r = oracle[i];
    source >> tag >> index >> r.emission.x >> r.emission.y >> r.emission.z;
    for (auto &v : r.film) {
      source >> v;
    }
    for (auto &v : r.native_write_counts) {
      source >> v;
    }
    if (!source || tag != 'E' || index != i) {
      return false;
    }
  }
  std::string extra;
  if (source >> extra) {
    return false;
  }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto film = device.create_buffer<luisa::float4>(f::film_lanes / 4);
  auto trace = device.create_buffer<luisa::float4>(1);
  auto count = device.create_buffer<unsigned>(1);
  auto unused = device.create_buffer<float>(1);
  unsigned passed = 0, total = 0;
  // Eligibility cannot depend on whether forward-only or NEE rendering was
  // selected. Serial and per-sample atomics consume the same operation.
  for (bool nee : {false, true}) {
    auto config = d::config(nee, false);
    for (auto mode :
         {d::PathFilmAccumulation::atomic, d::PathFilmAccumulation::serial}) {
      Kernel1D kernel = [&](BufferFloat4 output, BufferFloat4 diagnostic,
                            BufferUInt count, BufferFloat unused, UInt flags,
                            Bool exit, UInt path_flags, UInt depth,
                            Float3 emission) {
        d::record(config, mode, output, diagnostic, count, unused, flags, exit,
                  path_flags, depth, emission, UInt{0u}, Float{f::direct_limit},
                  Float{f::indirect_limit});
      };
      auto shader =
          device.compile(kernel, ShaderOption{.enable_cache = false,
                                              .enable_fast_math = true});
      for (unsigned i = 0; i < f::cases; ++i) {
        const auto in = f::inputs[i];
        const auto flags = in.emission ? closure::runtime_emission : 0u;
        const auto path_flags = in.route == 0  ? 0u
                                : in.route < 3 ? path::flag_surface_pass
                                               : path::flag_volume_pass;
        const auto depth = in.route == 0 ? 0u : in.route % 2 ? 1u : 3u;
        std::array<luisa::float4, f::film_lanes / 4> actual{};
        stream << film.copy_from(actual.data())
               << shader(film, trace, count, unused, flags, bool(in.exit),
                         path_flags, depth, oracle[i].emission)
                      .dispatch(1u)
               << film.copy_to(actual.data()) << synchronize();
        bool good = true;
        for (unsigned lane = 0; lane < f::film_lanes; ++lane) {
          const auto v = actual[lane / 4][lane % 4];
          const auto e = oracle[i].film[lane];
          if (!std::isfinite(v) || std::abs(v - e) > 2.0e-6f) {
            std::cerr << "nee=" << nee
                      << " atomic=" << (mode == d::PathFilmAccumulation::atomic)
                      << " case=" << i << " lane=" << lane << " actual=" << v
                      << " Cycles=" << e << '\n';
            good = false;
          }
        }
        passed += good;
        ++total;
      }
    }
  }
  std::cout << "Original Cycles surface emission film: " << passed << '/'
            << total << '\n';
  return passed == total && camera_integration(device, oracle[f::cases]);
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
