#include "cycles_svm_light_emission_fixture.h"
#include "path_kernel_builder.h"
#include "path_kernel_direct_light_task.h"
#include "path_tracer_bsdf_tables.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>
#include <psycles/luisa/cycles_svm.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <type_traits>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
using psycles::test_support::light_emission_cases;
constexpr auto case_count = static_cast<unsigned>(light_emission_cases.size());

// The fixture executes only SHADE_LIGHT_NEE. Other stages are deliberately
// unbound; accidentally invoking one is an error, not a reference evaluator.
template <typename Function> Function unbound() {
  return Function{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

template <typename T, std::size_t Extent>
auto upload(Device &device, Stream &stream, std::span<T, Extent> values) {
  auto buffer = device.create_buffer<std::remove_const_t<T>>(values.size());
  stream << buffer.copy_from(values.data()) << synchronize();
  return buffer;
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->native_cycles_svm_surface = true;
  scene->emissive_triangle_count = case_count;
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  auto &runtime = *scene->cycles_svm;
  runtime.geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  runtime.objects = std::make_unique<CyclesSvmObjectRuntime>();
  auto &used = runtime.compilation.table.node_types_used;
  for (const auto node :
       {abi::NODE_SHADER_JUMP, abi::NODE_VALUE_F, abi::NODE_CLOSURE_BSDF,
        abi::NODE_CLOSURE_SET_WEIGHT, abi::NODE_END}) {
    used[node] = true;
  }
  runtime.compilation.table.peak_stack_usage = 1u;
  const auto image = psycles::test_support::make_light_emission_image();
  runtime.word_buffer = upload(device, stream, std::span{image});
  std::array<abi::KernelShader, case_count> shaders{};
  runtime.kernel_shader_buffer =
      upload(device, stream, std::span<const abi::KernelShader>{shaders});
  abi::KernelObject object{};
  object.tfm = object.itfm = {{1.0f, 0.0f, 0.0f, 0.0f},
                              {0.0f, 1.0f, 0.0f, 0.0f},
                              {0.0f, 0.0f, 1.0f, 0.0f}};
  object.primitive_type = abi::PRIMITIVE_TRIANGLE;
  runtime.objects->object_buffer =
      upload(device, stream, std::span<const abi::KernelObject>{&object, 1u});
  constexpr std::array<unsigned, 1u> flags{0u};
  runtime.objects->object_flag_buffer =
      upload(device, stream, std::span{flags});
  constexpr std::array<abi::packed_float3, 3u> vertices{
      abi::packed_float3{-2.0f, -2.0f, 0.0f},
      {2.0f, -2.0f, 0.0f},
      {0.0f, 2.0f, 0.0f}};
  runtime.geometry->triangle_vertex_buffer =
      upload(device, stream, std::span{vertices});
  std::array<abi::packed_uint3, case_count> indices{};
  std::array<unsigned, case_count> triangle_shaders{};
  std::array<float, case_count> cosines{};
  for (auto i = 0u; i < case_count; ++i) {
    indices[i] = {0u, 1u, 2u};
    triangle_shaders[i] = i;
    cosines[i] = light_emission_cases[i].cosine;
  }
  runtime.geometry->triangle_index_buffer =
      upload(device, stream, std::span<const abi::packed_uint3>{indices});
  runtime.geometry->triangle_shader_buffer =
      upload(device, stream, std::span<const unsigned>{triangle_shaders});
  const std::array<abi::packed_normal, 3u> normals{
      abi::pack_geometry_normal({0.0f, 0.0f, 1.0f}),
      abi::pack_geometry_normal({0.0f, 0.0f, 1.0f}),
      abi::pack_geometry_normal({0.0f, 0.0f, 1.0f})};
  runtime.geometry->attribute_normal_buffer =
      upload(device, stream, std::span{normals});
  const auto table = make_cycles_bsdf_table_values({});
  scene->cycles_bsdf_table_buffer = upload(device, stream, std::span{table});
  auto cosine_buffer = upload(device, stream, std::span<const float>{cosines});

  PathKernelConfig config{
      .scene = scene,
      .light_transport = {unbound<SafeNormalizeCallable>(),
                          unbound<ForwardLightWeightCallable>(),
                          unbound<NeeLightWeightCallable>(),
                          unbound<ClampLightContributionCallable>(),
                          LightSampleRouletteCallable{
                              [](Float3, Float, Float) { return 1.0f; }},
                          unbound<LightComponentRatioCallable>(),
                          unbound<SplitScatteredLightCallable>()},
      .light_distribution_sample = unbound<LightDistributionSampleCallable>(),
      .light_tree = {unbound<LightTreeSampleCallable>(),
                     unbound<LightTreeSampleCallable>(),
                     unbound<LightTreePdfCallable>(),
                     unbound<LightTreePdfCallable>(),
                     unbound<LightTreeForwardPdfCallable>(),
                     unbound<LightTreeTriangleEmitterCallable>()},
      .surfaces = {nullptr, unbound<SurfacePreparationCallable>(),
                   unbound<SurfaceEvaluateLightCallable>(),
                   unbound<SurfaceConstantEmissionCallable>(),
                   unbound<SurfaceEmissionCallable>(),
                   unbound<SurfaceSampleCallable>(),
                   unbound<SurfaceClosureTraceCallable>(),
                   unbound<SurfaceSampleTraceCallable>(),
                   unbound<SurfaceBssrdfNormalCallable>()},
      .environment = {unbound<EnvironmentConstantCallable>(),
                      unbound<EnvironmentBaseCallable>(),
                      {},
                      unbound<EnvironmentSunCallable>()},
      .shade_shadow_surface = unbound<EvaluateShadowSurfaceCallable>(),
      .trace_shadow = unbound<TraceShadowCallable>()};
  const auto evaluator = make_direct_light_task_evaluator(config);
  Kernel1D<Buffer<float>, Buffer<luisa::float4>> kernel =
      [evaluator](BufferFloat cosines, BufferFloat4 output) noexcept {
        const auto i = dispatch_id().x;
        const auto cosine = cosines.read(i);
        const auto incoming =
            make_float3(sqrt(1.0f - cosine * cosine), 0.0f, cosine);
        Var<DirectLightTaskCall> task;
        task.ray_origin = incoming;
        task.ray_direction = -incoming;
        task.ray_maximum = 1.0f;
        task.light_object = 0u;
        task.light_primitive = i;
        task.unshadowed_contribution = make_float3(1.0f);
        task.nee_path_throughput = make_float3(1.0f);
        // No pre-evaluated shader is supplied. Production SHADE_LIGHT_NEE
        // must reconstruct the light hit and execute the native SVM here.
        task.light_shader = make_float3(0.0f);
        Var<RenderKernelParameters> parameters;
        parameters.camera_transform = make_float4x4(1.0f);
        parameters.camera_inverse_transform = make_float4x4(1.0f);
        parameters.full_width = 64u;
        parameters.full_height = 64u;
        const auto active = evaluator.shade_light_nee(task, parameters);
        output.write(i, make_float4(task.light_shader, active.cast<float>()));
      };
  auto shader = device.compile(kernel);
  auto output = device.create_buffer<luisa::float4>(case_count);
  std::array<luisa::float4, case_count> values{};
  stream << shader(cosine_buffer, output).dispatch(case_count)
         << output.copy_to(values.data()) << synchronize();

  std::ifstream oracle{PSYCLES_LIGHT_EMISSION_ORACLE};
  bool passed = static_cast<bool>(oracle);
  for (auto i = 0u; i < case_count && oracle; ++i) {
    unsigned scenario{}, flags{}, offset{}, count{}, left{};
    luisa::float3 emission{}, extinction{};
    oracle >> scenario >> emission.x >> emission.y >> emission.z >> flags >>
        extinction.x >> extinction.y >> extinction.z >> offset >> count >> left;
    const auto actual = values[i];
    const auto near = [](float a, float b) {
      return std::isfinite(a) &&
             std::abs(a - b) <= 5.0e-5f * std::max(std::abs(b), 1.0e-8f);
    };
    const auto active =
        emission.x != 0.0f || emission.y != 0.0f || emission.z != 0.0f;
    if (!oracle || scenario != i || !near(actual.x, emission.x) ||
        !near(actual.y, emission.y) || !near(actual.z, emission.z) ||
        actual.w != static_cast<float>(active)) {
      std::cerr << "Native NEE case " << i << " got (" << actual.x << ','
                << actual.y << ',' << actual.z << ") expected (" << emission.x
                << ',' << emission.y << ',' << emission.z << ")\n";
      passed = false;
    }
  }
  return passed;
}

} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
