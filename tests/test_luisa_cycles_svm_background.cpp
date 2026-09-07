#include "cycles_svm_background_fixture.h"
#include "path_tracer_cycles_svm_background.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_light.h"
#include "path_tracer_environment.h"

#include <psycles/sampling/background_distribution.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace abi = psycles::compiler::cycles_svm;
constexpr unsigned count = background_inputs.size();

template <typename T> auto upload(Device &device, Stream &stream, const T &values) {
  auto buffer = device.create_buffer<typename T::value_type>(values.size());
  stream << buffer.copy_from(values.data()) << synchronize();
  return buffer;
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &r = *scene->cycles_svm;
  r.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  r.objects = std::make_unique<CyclesSvmObjectRuntime>();
  const auto dummy = [&]<typename T>(Buffer<T> &buffer) {
    buffer = upload(device, stream, std::array<T, 1u>{});
  };
  dummy(r.objects->object_buffer); dummy(r.objects->object_flag_buffer);
  auto &g = *r.geometry;
  dummy(g.attribute_map_buffer); dummy(g.attribute_float_buffer);
  dummy(g.attribute_float2_buffer); dummy(g.attribute_float3_buffer);
  dummy(g.attribute_float4_buffer); dummy(g.attribute_uchar4_buffer);
  dummy(g.attribute_normal_buffer); dummy(g.triangle_vertex_buffer);
  dummy(g.triangle_index_buffer); dummy(g.triangle_shader_buffer);
  dummy(g.curve_key_buffer); dummy(g.curve_buffer); dummy(g.point_buffer);
  r.compilation.table.words = background_words();
  r.compilation.table.peak_stack_usage = 4u;
  for (auto tag : {abi::NODE_SHADER_JUMP, abi::NODE_GEOMETRY, abi::NODE_LIGHT_PATH,
                   abi::NODE_EMISSION_WEIGHT, abi::NODE_CLOSURE_BACKGROUND, abi::NODE_END}) {
    r.compilation.table.node_types_used[tag] = true;
  }
  r.word_buffer = upload(device, stream, r.compilation.table.words);
  std::array<abi::KernelShader, 3u> records{};
  records[0].flags = background_shader_flags;
  records[1].flags = records[2].flags = abi::SD_HAS_CONSTANT_EMISSION;
  records[2].constant_emission = {0.25f, -0.5f, 2.0f};
  r.kernel_shader_buffer = upload(device, stream, records);
  scene->cycles_background_shader_id = background_shader;
  scene->cycles_background_object_index = 13u;
  scene->background_map_weight = 1.0f;
  scene->background_map_width = background_width;
  scene->background_map_height = background_height;
  scene->background_guided_sun_weight = 4.0f;
  // Deliberately present metadata must not override or add to the native
  // graph. There is no texture, legacy MaterialLibrary or parameter buffer.
  scene->environment_suns.push_back({{0, 0, 1}, {100, 100, 100}, 0.5f});
  std::vector<luisa::float4> rays;
  std::vector<luisa::uint2> states;
  for (const auto &c : background_inputs) {
    rays.push_back(luisa::make_float4(c.origin[0], c.origin[1], c.origin[2], c.time));
    rays.push_back(luisa::make_float4(c.direction[0], c.direction[1], c.direction[2], c.differential));
    rays.push_back(luisa::make_float4(c.u, c.v, 0.0f, 0.0f));
    states.push_back(luisa::make_uint2(c.visibility, c.flag));
  }
  auto input = upload(device, stream, rays);
  auto state_input = upload(device, stream, states);
  auto output = device.create_buffer<luisa::float4>(count * 11u);
  auto metadata = device.create_buffer<luisa::uint4>(count * 3u);
  std::ifstream oracle{PSYCLES_BACKGROUND_ORACLE};
  bool passed = true;
  const auto close = [&](float actual, float expected, unsigned mode, unsigned row, unsigned field) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1.0e-6f + 8.0e-5f * std::abs(expected)) {
      std::cerr << "Background " << mode << ':' << row << " field " << field
                << ": " << actual << " expected " << expected << '\n';
      passed = false;
    }
  };
  std::array<luisa::float4, count> baked{};
  for (auto mode = 0u; mode < 3u; ++mode) {
    const auto evaluation = static_cast<CyclesSvmBackgroundEvaluation>(mode);
    Kernel1D kernel = [scene, mode, evaluation](BufferFloat4 rays, BufferUInt2 states,
                                                BufferFloat4 out, BufferUInt4 meta) {
      const auto i = dispatch_x();
      const auto p = rays.read(3u * i), d = rays.read(3u * i + 1u), uv = rays.read(3u * i + 2u);
      const auto s = states.read(i);
      Float3 origin = p.xyz(), direction = d.xyz();
      Float differential = d.w, time = p.w;
      const svm::PathState state{
          mode == 0u ? s.x : UInt{0u},
          (mode == 0u ? s.y : UInt{0u}) | svm::path_ray_emission |
              (mode == 2u ? svm::path_ray_importance_bake : 0u),
          3u, 5u, 2u, 1u, 4u, 0u};
      Var<RenderKernelParameters> params;
      params.camera_transform = params.camera_inverse_transform = make_float4x4(1.0f);
      const PathCyclesSvmKernelGlobals kg{scene, params, scene->camera.projection, true, true};
      if (mode == 2u) {
        const auto ray = cycles_svm_background_bake_ray(uv.x, uv.y, background_width, background_height);
        origin = make_float3(0.0f); direction = ray.direction;
        differential = ray.differential; time = 0.5f;
      }
      auto sd = setup_cycles_svm_background_shader_data(
          *scene, origin, direction, differential, time, state.visibility, evaluation);
      if (mode == 1u) {
        Var<DirectLightTaskCall> task;
        task.light_object = scene->cycles_background_object_index;
        task.ray_origin = origin; task.ray_direction = direction;
        task.ray_dD = differential; task.ray_time = time;
        sd = setup_cycles_svm_light_shader_data(scene, kg, task, params).shader_data;
      }
      out.write(i * 11u, make_float4(sd.P, sd.time));
      out.write(i * 11u + 1u, make_float4(sd.ray_P, sd.ray_length));
      out.write(i * 11u + 2u, make_float4(sd.N, sd.dP));
      out.write(i * 11u + 3u, make_float4(sd.Ng, sd.dI));
      out.write(i * 11u + 4u, make_float4(sd.wi, sd.u));
      out.write(i * 11u + 5u, make_float4(sd.dPdu, sd.v));
      out.write(i * 11u + 6u, make_float4(sd.dPdv, 0.0f));
      out.write(i * 11u + 7u, make_float4(sd.du.dx, sd.du.dy, sd.dv.dx, sd.dv.dy));
      meta.write(i * 3u, make_uint4(sd.shader, sd.flag, sd.object_flag, sd.type));
      meta.write(i * 3u + 1u, make_uint4(sd.object, sd.prim, state.visibility, state.flag));
      const auto identity = make_float4x4(1.0f);
      const svm::TransformState transforms{identity, identity, identity, identity};
      svm::EvaluationResult result;
      svm::eval_nodes(kg, *scene->cycles_svm->word_buffer, abi::SHADER_TYPE_SURFACE,
          0u, cycles_svm_background_feature_mask(evaluation),
          scene->cycles_svm->compilation.table.node_types_used,
          transforms, sd, state, result, 4u);
      out.write(i * 11u + 8u, make_float4(sd.closure_emission_background, sd.flag.cast<float>()));
      out.write(i * 11u + 9u, make_float4(evaluate_environment_emission(
          scene, params, origin, direction, differential, time, state, 0u, evaluation), 0.0f));
      if (mode == 2u) {
        out.write(i * 11u + 10u, make_float4(evaluate_background_importance(scene, params, uv.x, uv.y), 0.0f));
      }
      meta.write(i * 3u + 2u, make_uint4(result.status, result.final_offset,
          kg.background_use_sun_guiding().cast<unsigned>(), 0u));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::array<luisa::float4, count * 11u> actual{};
    std::array<luisa::uint4, count * 3u> meta{};
    stream << shader(input, state_input, output, metadata).dispatch(count)
           << output.copy_to(actual.data()) << metadata.copy_to(meta.data()) << synchronize();
    for (auto i = 0u; i < count; ++i) {
      char kind{}; unsigned oracle_mode{}, row{};
      if (!(oracle >> kind >> oracle_mode >> row) || kind != 'B' || oracle_mode != mode || row != i) { return false; }
      std::array<float, 36u> expected{};
      for (auto &v : expected) { if (!(oracle >> v)) { return false; } }
      for (auto j = 0u; j < 36u; ++j) {
        close(actual[i * 11u + j / 4u][j % 4u], expected[j], mode, i, j);
      }
      for (auto j = 0u; j < 8u; ++j) {
        unsigned expected_meta{};
        if (!(oracle >> expected_meta) || meta[i * 3u + j / 4u][j % 4u] != expected_meta) {
          std::cerr << "Background integer state mismatch " << mode << ':' << i << ':' << j << '\n';
          passed = false;
        }
      }
      for (auto j = 0u; j < 3u; ++j) { close(actual[i * 11u + 9u][j], expected[32u + j], mode, i, 36u + j); }
      if (meta[i * 3u + 2u].x != unsigned(svm::EvaluationStatus::ended) ||
          meta[i * 3u + 2u].y != r.compilation.table.words.size() ||
          meta[i * 3u + 2u].z != 1u) {
        const auto m = meta[i * 3u + 2u];
        std::cerr << "Background PC/status/sun-guiding mismatch " << mode << ':' << i
                  << " status=" << m.x << " PC=" << m.y << " guiding=" << m.z << '\n';
        passed = false;
      }
      if (mode == 2u) { baked[i] = actual[i * 11u + 10u]; }
    }
  }
  for (auto i = 0u; i < count; ++i) {
    char kind{}; unsigned row{};
    if (!(oracle >> kind >> row) || kind != 'I' || row != i) { return false; }
    for (auto j = 0u; j < 3u; ++j) {
      float expected{}; if (!(oracle >> expected)) { return false; }
      close(baked[i][j], expected, 3u, i, j);
    }
  }
  // Camera-dependent Window output is baked only after render parameters
  // exist. Keep both sessions' CDFs alive and verify the second bake cannot
  // mutate the first. The reference radiance is Cycles' GPU bake above;
  // the separately tested distribution builder is not a shader evaluator.
  scene->camera.projection = psycles::contract::CameraProjection::orthographic;
  scene->background_map_width = 4u; scene->background_map_height = 2u;
  r.compilation.table.words = background_words(true);
  r.word_buffer = upload(device, stream, r.compilation.table.words);
  r.compilation.table.node_types_used[abi::NODE_GEOMETRY] = false;
  r.compilation.table.node_types_used[abi::NODE_TEX_COORD] = true;
  std::array<std::shared_ptr<const BackgroundSamplingDistribution>, 2u> distributions;
  std::array<psycles::sampling::CyclesBackgroundMapDistribution, 2u> expected_distributions;
  for (auto variant = 0u; variant < 2u; ++variant) {
    std::array<psycles::Vec3f, 8u> colors{};
    for (auto i = 0u; i < 8u; ++i) {
      char kind{}; unsigned v{}, row{};
      if (!(oracle >> kind >> v >> row >> colors[i].x >> colors[i].y >> colors[i].z) ||
          kind != 'W' || v != variant || row != i) { return false; }
    }
    expected_distributions[variant] =
        psycles::sampling::build_cycles_background_map_distribution(colors, 4u, 2u);
    RenderKernelParameters params{};
    params.camera_transform = params.camera_inverse_transform = luisa::make_float4x4(1.0f);
    params.full_width = variant == 0u ? 1920u : 960u;
    params.full_height = 1080u;
    params.camera_world_to_ndc = luisa::make_float4x4(
        luisa::make_float4(variant == 0u ? 0.25f : 0.5f, 0, 0, 0),
        luisa::make_float4(0, variant == 0u ? 0.5f : 0.25f, 0, 0),
        luisa::make_float4(0, 0, 1, 0), luisa::make_float4(0.5f, 0.5f, 0, 1));
    distributions[variant] = build_background_sampling_distribution(scene, stream, params);
  }
  if (distributions[0]->conditional.handle() == distributions[1]->conditional.handle()) {
    std::cerr << "Render sessions share a camera-dependent background CDF\n";
    passed = false;
  }
  for (auto variant = 0u; variant < 2u; ++variant) {
    std::array<luisa::float2, 10u> conditional{};
    std::array<luisa::float2, 3u> marginal{};
    stream << distributions[variant]->conditional.copy_to(conditional.data())
           << distributions[variant]->marginal.copy_to(marginal.data()) << synchronize();
    const auto &expected = expected_distributions[variant];
    for (auto i = 0u; i < conditional.size(); ++i) {
      close(conditional[i].x, expected.conditional[i].function, 5u, variant, i * 2u);
      close(conditional[i].y, expected.conditional[i].cumulative, 5u, variant, i * 2u + 1u);
    }
    for (auto i = 0u; i < marginal.size(); ++i) {
      close(marginal[i].x, expected.marginal[i].function, 6u, variant, i * 2u);
      close(marginal[i].y, expected.marginal[i].cumulative, 6u, variant, i * 2u + 1u);
    }
  }
  // These shader IDs have no jump entries. Even constant black must take the
  // KernelShader fast path, not execute an out-of-range SVM stream.
  for (unsigned constant = 1u; constant <= 2u; ++constant) {
    scene->cycles_background_shader_id = constant | background_shader;
    scene->background_guided_sun_weight = 0.0f;
    Kernel1D kernel = [scene](BufferFloat4 out) {
      Var<RenderKernelParameters> params;
      params.camera_transform = params.camera_inverse_transform = make_float4x4(1.0f);
      const svm::PathState state{1u, svm::path_ray_emission};
      const PathCyclesSvmKernelGlobals kg{scene, params, scene->camera.projection, true, true};
      out.write(0u, make_float4(evaluate_environment_emission(scene, params,
          make_float3(1.0f), make_float3(0.0f, 0.0f, 1.0f), 0.5f, 0.5f,
          state, 0u, CyclesSvmBackgroundEvaluation::forward), kg.background_use_sun_guiding().cast<float>()));
    };
    auto shader = device.compile(kernel);
    luisa::float4 actual{};
    stream << shader(output).dispatch(1u) << output.view(0u, 1u).copy_to(&actual) << synchronize();
    const auto expected = constant == 1u ? luisa::make_float3(0.0f) : luisa::make_float3(0.25f, -0.5f, 2.0f);
    for (auto j = 0u; j < 3u; ++j) { close(actual[j], expected[j], 4u, constant, j); }
    close(actual.w, 0.0f, 4u, constant, 3u);
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
