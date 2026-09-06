#include "cycles_svm_native_shadow_fixture.h"
#include "path_kernel_direct_light_task.h"
#include "path_tracer_bsdf_tables.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shadow.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <type_traits>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;
namespace abi = psycles::compiler::cycles_svm;
constexpr unsigned count = native_shadow_inputs.size();
constexpr unsigned batch_count = native_shadow_batches.size();

template <typename CallableType> CallableType unbound() {
  return CallableType{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

template <typename T, std::size_t Extent>
auto upload(Device &device, Stream &stream, std::span<T, Extent> values) {
  auto buffer = device.create_buffer<std::remove_const_t<T>>(values.size());
  stream << buffer.copy_from(values.data()) << synchronize();
  return buffer;
}

bool close(float actual, float expected) {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             5.0e-5f * std::max(std::abs(expected), 1.0e-5f);
}

bool run(const char *program, const char *backend, bool no_cache) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  ShaderOption options;
  options.enable_cache = !no_cache;
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &runtime = *scene->cycles_svm;
  runtime.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  runtime.objects = std::make_unique<CyclesSvmObjectRuntime>();
  std::array<abi::KernelObject, count> objects{};
  std::array<unsigned, count> flags{}, shaders{};
  std::array<abi::KernelShader, count> shader_records{};
  std::array<InstanceGpu, count> instances{};
  std::array<abi::packed_uint3, count> indices{};
  std::array<abi::packed_float3, count * 3u> vertices{};
  std::array<abi::packed_normal, count * 3u> normals{};
  std::array<luisa::float4, count * 3u> rays{};
  std::array<luisa::uint4, count> identities{};
  for (auto i = 0u; i < count; ++i) {
    const auto &c = native_shadow_inputs[i];
    const auto &g = c.geometry;
    objects[i].tfm = g.tfm;
    objects[i].itfm = g.itfm;
    objects[i].position_offset = int(i * 3u);
    objects[i].normal_offset =
        (g.object_flags & abi::SD_OBJECT_HAS_CORNER_NORMALS) ? 0 : int(i * 3u);
    objects[i].primitive_type = abi::PRIMITIVE_TRIANGLE;
    flags[i] = g.object_flags;
    shaders[i] = g.shader;
    shader_records[i].flags = c.shader_flags;
    // Reverse instance IDs and use local primitive zero. Neither index may
    // accidentally be interpreted as Cycles' global object/primitive index.
    instances[count - 1u - i].cycles_object_index = i;
    instances[count - 1u - i].cycles_primitive_offset = i;
    indices[i] = {0u, 1u, 2u};
    for (auto v = 0u; v < 3u; ++v) {
      vertices[i * 3u + v] = g.vertices[v];
      normals[i * 3u + v] = abi::pack_geometry_normal(g.normals[v]);
    }
    rays[i * 3u] = luisa::make_float4(g.ray_P.x, g.ray_P.y, g.ray_P.z, g.time);
    rays[i * 3u + 1u] =
        luisa::make_float4(g.ray_D.x, g.ray_D.y, g.ray_D.z, g.distance);
    rays[i * 3u + 2u] = luisa::make_float4(g.dP, g.dD, c.u, c.v);
    identities[i] =
        luisa::make_uint4(g.sample, g.rng_hash, g.rng_offset, count - 1u - i);
  }
  runtime.objects->object_buffer = upload(device, stream, std::span{objects});
  runtime.objects->object_flag_buffer =
      upload(device, stream, std::span{flags});
  runtime.geometry->triangle_vertex_buffer =
      upload(device, stream, std::span{vertices});
  runtime.geometry->triangle_index_buffer =
      upload(device, stream, std::span{indices});
  runtime.geometry->triangle_shader_buffer =
      upload(device, stream, std::span{shaders});
  runtime.geometry->attribute_normal_buffer =
      upload(device, stream, std::span{normals});
  runtime.kernel_shader_buffer =
      upload(device, stream, std::span{shader_records});
  scene->instance_buffer = upload(device, stream, std::span{instances});
  const auto table = make_cycles_bsdf_table_values({});
  scene->cycles_bsdf_table_buffer = upload(device, stream, std::span{table});
  const auto words = make_native_shadow_image();
  runtime.word_buffer = upload(device, stream, std::span{words});
  runtime.kernel_features = abi::kernel_feature_path_tracing;
  for (const auto node :
       {abi::NODE_END, abi::NODE_SHADER_JUMP, abi::NODE_VALUE_F,
        abi::NODE_LIGHT_PATH, abi::NODE_CLOSURE_SET_WEIGHT,
        abi::NODE_CLOSURE_BSDF}) {
    runtime.compilation.table.node_types_used[node] = true;
  }
  auto ray_buffer = upload(device, stream, std::span{rays});
  auto identity_buffer = upload(device, stream, std::span{identities});
  const auto evaluate = make_cycles_svm_shadow_surface_callable(scene);
  Kernel1D<Buffer<luisa::float4>, Buffer<luisa::uint4>, Buffer<luisa::float4>,
           Buffer<luisa::uint4>>
      single = [scene, evaluate](BufferFloat4 rays, BufferUInt4 ids,
                                 BufferFloat4 out, BufferUInt4 meta) noexcept {
        const auto i = dispatch_id().x;
        const auto p = rays.read(i * 3u);
        const auto d = rays.read(i * 3u + 1u);
        const auto uv = rays.read(i * 3u + 2u);
        const auto id = ids.read(i);
        auto ray = make_ray(p.xyz(), d.xyz(), 0.0f, 100.0f);
        Var<ShadowIntersectionCall> hit;
        hit.instance = id.w;
        hit.distance = d.w;
        hit.barycentric = uv.zw();
        Var<ShaderEvaluationStateCall> path;
        path.ray_depth = 7u;
        path.transparent_depth = 3u;
        path.diffuse_depth = 2u;
        path.glossy_depth = 4u;
        path.transmission_depth = 1u;
        auto state = make_shadow_shader_context(path, p.w, id.x, id.y, id.z);
        Var<RenderKernelParameters> params;
        params.camera_transform = params.camera_inverse_transform =
            make_float4x4(1.0f);
        const PathCyclesSvmKernelGlobals kg{
            scene, params, scene->camera.projection, true, true};
        const auto setup = setup_cycles_svm_shadow_shader_data(
            scene, kg, ray, hit, uv.x, uv.y, state, params);
        const auto &sd = setup.shader_data;
        out.write(i * 8u, make_float4(sd.P, sd.time));
        out.write(i * 8u + 1u, make_float4(sd.N, sd.ray_length));
        out.write(i * 8u + 2u, make_float4(sd.Ng, sd.dP));
        out.write(i * 8u + 3u, make_float4(sd.wi, sd.dI));
        out.write(i * 8u + 4u, make_float4(sd.dPdu, sd.u));
        out.write(i * 8u + 5u, make_float4(sd.dPdv, sd.v));
        out.write(i * 8u + 6u,
                  make_float4(sd.du.dx, sd.du.dy, sd.dv.dx, sd.dv.dy));
        const auto surface = evaluate(ray, hit, uv.x, uv.y, state, params);
        out.write(i * 8u + 7u,
                  make_float4(surface->transmittance,
                              surface->volume_boundary.cast<float>()));
        meta.write(i * 2u,
                   make_uint4(sd.shader, sd.flag, sd.object_flag, sd.type));
        meta.write(i * 2u + 1u, make_uint4(surface->object, surface->primitive,
                                           sd.lcg_state, surface->kind));
      };
  auto shader = device.compile(single, options);
  auto output = device.create_buffer<luisa::float4>(count * 8u);
  auto metadata = device.create_buffer<luisa::uint4>(count * 2u);
  std::array<luisa::float4, count * 8u> values{};
  std::array<luisa::uint4, count * 2u> integers{};
  stream
      << shader(ray_buffer, identity_buffer, output, metadata).dispatch(count)
      << output.copy_to(values.data()) << metadata.copy_to(integers.data())
      << synchronize();

  std::array<DirectLightTaskCall, batch_count> tasks{};
  std::array<ShadowIntersectionBatchCall, batch_count> batches{};
  for (auto i = 0u; i < batch_count; ++i) {
    const auto &c = native_shadow_batches[i];
    const auto &g = native_shadow_inputs[0].geometry;
    auto &t = tasks[i];
    t.ray_origin = luisa::make_float3(g.ray_P.x, g.ray_P.y, g.ray_P.z);
    t.ray_direction = luisa::make_float3(g.ray_D.x, g.ray_D.y, g.ray_D.z);
    t.ray_maximum = 100.0f;
    t.ray_dP = g.dP;
    t.ray_dD = g.dD;
    t.ray_time = g.time;
    t.sample_index = g.sample;
    t.rng_hash = g.rng_hash;
    t.rng_offset = c.rng_offset;
    t.path_depth = 7u;
    t.diffuse_depth = 2u;
    t.glossy_depth = 4u;
    t.transmission_depth = 1u;
    t.transparent_depth = c.transparent_bounce;
    t.volume_bounds_bounce = c.volume_bounds_bounce;
    t.shadow_throughput =
        luisa::make_float3(c.throughput.x, c.throughput.y, c.throughput.z);
    // Film finalization must not multiply this pre-shadow diagnostic field
    // into the complete shadow throughput a second time.
    t.unshadowed_contribution = luisa::make_float3(3.0f, 5.0f, 7.0f);
    t.source_object = i; // Prepared collector input, not a geometric self hit.
    batches[i].count = batches[i].total = c.count;
    for (auto j = 0u; j < 4u; ++j) {
      const auto object = c.objects[j];
      const auto &g = native_shadow_inputs[object];
      batches[i].hits[j] = {.instance = count - 1u - object,
                            .primitive = 0u,
                            .hit_type = 1u,
                            .distance = j < c.count ? c.distances[j] : 100.0f,
                            .barycentric = luisa::make_float2(g.u, g.v)};
    }
  }
  auto task_buffer = upload(device, stream, std::span{tasks});
  auto batch_buffer = upload(device, stream, std::span{batches});
  // Prepared intersection results isolate the production SHADE_SHADOW state
  // machine from traversal. A full batch is followed by a genuine empty batch.
  LocalShadowIntersectionCallable collect =
      [batch_buffer = batch_buffer.view()](Var<luisa::compute::Ray> ray,
                                           UInt source, UInt, UInt, UInt,
                                           UInt) noexcept {
        auto batch = batch_buffer->read(source);
        $if(ray->t_min() != 0.0f) { batch.count = batch.total = 0u; };
        return batch;
      };
  auto intersection =
      std::make_shared<ShadowIntersectionComponent>(std::move(collect));
  const auto light_transport = make_light_transport_callables(
      psycles::contract::DirectLightSampling::next_event_estimation);
  const DirectLightTaskEvaluator evaluator{
      .intersect_shadow = intersection,
      .shade_shadow_surface = evaluate,
      .trace_shadow = make_fused_shadow_trace_callable(intersection, evaluate),
      .light_sample_roulette = unbound<LightSampleRouletteCallable>(),
      .clamp_contribution = light_transport.clamp_light_contribution,
      .split_scattered_light = unbound<SplitScatteredLightCallable>()};
  Kernel1D<Buffer<DirectLightTaskCall>, Buffer<ShadowIntersectionBatchCall>,
           Buffer<luisa::float4>, Buffer<luisa::uint4>, Buffer<luisa::float4>>
      batch_kernel = [evaluator](BufferVar<DirectLightTaskCall> tasks,
                                 BufferVar<ShadowIntersectionBatchCall> batches,
                                 BufferFloat4 out, BufferUInt4 meta,
                                 BufferFloat4 fused) noexcept {
        const auto i = dispatch_id().x;
        auto task = tasks.read(i);
        Var<RenderKernelParameters> params;
        params.transparent_max_bounces = 65535u;
        params.sample_clamp_direct = params.sample_clamp_indirect =
            std::numeric_limits<float>::max();
        params.camera_transform = params.camera_inverse_transform =
            make_float4x4(1.0f);
        const auto traced = evaluator.trace(task, params);
        fused.write(i * 2u, make_float4(traced->throughput, 0.0f));
        const auto step = evaluator.shade_shadow(task, batches.read(i), params);
        fused.write(i * 2u + 1u,
                    make_float4(evaluator.contribution(
                                    task, task.shadow_throughput, params),
                                0.0f));
        out.write(i, make_float4(task.shadow_throughput, task.ray_minimum));
        meta.write(i * 2u, make_uint4(step.continue_shadow.cast<unsigned>(),
                                      step.visible.cast<unsigned>(),
                                      task.transparent_depth, task.rng_offset));
        meta.write(
            i * 2u + 1u,
            make_uint4(task.volume_bounds_bounce,
                       (!step.continue_shadow & !step.visible).cast<unsigned>(),
                       0u, 0u));
      };
  auto batch_shader = device.compile(batch_kernel, options);
  auto batch_out = device.create_buffer<luisa::float4>(batch_count);
  auto batch_meta = device.create_buffer<luisa::uint4>(batch_count * 2u);
  auto fused_out = device.create_buffer<luisa::float4>(batch_count * 2u);
  std::array<luisa::float4, batch_count> batch_values{};
  std::array<luisa::float4, batch_count * 2u> fused_values{};
  std::array<luisa::uint4, batch_count * 2u> batch_integers{};
  stream << batch_shader(task_buffer, batch_buffer, batch_out, batch_meta,
                         fused_out)
                .dispatch(batch_count)
         << batch_out.copy_to(batch_values.data())
         << batch_meta.copy_to(batch_integers.data())
         << fused_out.copy_to(fused_values.data()) << synchronize();

  std::ifstream oracle{PSYCLES_NATIVE_SHADOW_ORACLE};
  bool passed = bool(oracle);
  for (auto i = 0u; i < count + batch_count; ++i) {
    const bool surface = i < count;
    const auto index = surface ? i : i - count;
    char kind{};
    unsigned scenario{};
    oracle >> kind >> scenario;
    passed &= scenario == index && kind == (surface ? 'S' : 'B');
    std::array<float, 32u> expected_values{};
    for (auto j = 0u; j < (surface ? 32u : 4u); ++j) {
      auto &expected = expected_values[j];
      oracle >> expected;
      const auto actual = surface ? values[index * 8u + j / 4u][j % 4u]
                                  : batch_values[index][j];
      if (!close(actual, expected)) {
        std::cerr << kind << ' ' << index << " float=" << j
                  << " actual=" << actual << " expected=" << expected << '\n';
        passed = false;
      }
    }
    std::array<unsigned, 8u> expected_meta{};
    for (auto j = 0u; j < 8u; ++j) {
      oracle >> expected_meta[j];
      const auto actual = surface ? integers[index * 2u + j / 4u][j % 4u]
                                  : batch_integers[index * 2u + j / 4u][j % 4u];
      if (actual != expected_meta[j]) {
        std::cerr << kind << ' ' << index << " uint=" << j
                  << " actual=" << actual << " expected=" << expected_meta[j]
                  << '\n';
        passed = false;
      }
    }
    if (!surface) {
      for (auto j = 0u; j < 3u; ++j) {
        const auto expected =
            expected_meta[5] == 1u ? 0.0f : expected_values[j];
        if (!close(fused_values[index * 2u][j], expected)) {
          std::cerr << "F " << index << " float=" << j
                    << " actual=" << fused_values[index * 2u][j]
                    << " expected=" << expected << '\n';
          passed = false;
        }
        if (!close(fused_values[index * 2u + 1u][j], expected_values[j])) {
          std::cerr << "Film " << index << " float=" << j
                    << " actual=" << fused_values[index * 2u + 1u][j]
                    << " expected=" << expected_values[j] << '\n';
          passed = false;
        }
      }
    }
  }
  std::string trailing;
  passed &= bool(oracle) && !(oracle >> trailing);
  if (passed)
    std::cout << "Native shadow: 25 setup/SVM and 14 staged/fused/film Cycles "
                 "oracle cases passed\n";
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback",
             argc > 2 && std::string_view{argv[2]} == "--no-cache")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
