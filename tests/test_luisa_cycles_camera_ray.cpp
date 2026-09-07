#include "cycles_camera_projection_test_support.h"
#include "path_tracer_camera.h"
#include "path_tracer_cycles_svm_kernel_globals.h"

#include <psycles/compiler/cycles_camera.h>
#include <psycles/compiler/cycles_transform.h>
#include <psycles/luisa/camera_sampling.h>
#include <psycles/sampling/pixel_filter.h>
#include <luisa/xir/translators/ast2xir.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto output = device.create_buffer<luisa::float4>(5u);
  auto scene = std::make_shared<LuisaSceneData>();
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->cycles_svm->geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  scene->cycles_svm->objects = std::make_unique<CyclesSvmObjectRuntime>();
  // Deliberately simple input lookup table. The expected rays are sampled by
  // original Cycles at the resulting raster positions, not a CPU evaluator.
  psycles::sampling::PixelFilterTable filter{};
  for (unsigned i = 0u; i < filter.size(); ++i) {
    filter[i] = float(i) / float(filter.size() - 1u);
  }
  auto filter_buffer = device.create_buffer<float>(filter.size());
  stream << filter_buffer.copy_from(filter.data()) << synchronize();
  const SafeNormalizeCallable safe = [](Float3 v, Float3 fallback) {
    Float3 result = fallback;
    $if(dot(v, v) > 1.0e-20f) { result = normalize(v); };
    return result;
  };
  std::ifstream oracle{PSYCLES_CAMERA_RAY_ORACLE};
  if (!oracle) { return false; }
  bool passed = true;
  for (unsigned i = 0u; i < camera_projection_inputs.size(); ++i) {
    const auto &c = camera_projection_inputs[i];
    RenderKernelParameters p{};
    p.full_width = c.width; p.full_height = c.height;
    const auto setup = psycles::compiler::make_cycles_camera_projection(
        camera_projection_description(c), c.width, c.height);
    p.camera_raster_to_camera = to_luisa(setup.raster_to_camera);
    p.camera_world_to_ndc = to_luisa(setup.world_to_ndc);
    p.camera_dx = to_luisa(setup.dx); p.camera_dy = to_luisa(setup.dy);
    p.camera_inv_aperture_ratio = 1.0f / c.aperture_ratio;
    p.camera_near = c.near_clip; p.camera_far = c.far_clip;
    p.camera_aperture_radius = c.aperture; p.camera_focal_distance = c.focus;
    p.camera_aperture_ratio = c.aperture_ratio;
    p.camera_transform = to_luisa(c.transform);
    p.camera_inverse_transform = to_luisa(
        psycles::compiler::cycles_inverse_affine_transform(c.transform));
    Kernel1D<RenderKernelParameters, luisa::float2, Buffer<float>, Buffer<luisa::float4>>
        kernel = [c, safe, scene](Var<RenderKernelParameters> parameters, Float2 raster,
                            BufferFloat filter_table, BufferFloat4 out) {
      const UInt x = cast<unsigned>(floor(raster.x));
      const UInt y = cast<unsigned>(floor(raster.y));
      CameraDimensionSample sample{
          y, UInt{0u}, raster - floor(raster),
          make_float3(0.5f, camera_lens_sample.x, camera_lens_sample.y)};
      const auto ray = construct_camera_ray(
          filter_table, parameters, x, parameters.full_height - 1u - y, sample,
          c.orthographic ? CameraProjection::orthographic : CameraProjection::perspective,
          c.aperture > 0.0f, c.blades, c.rotation, safe);
      out.write(0u, make_float4(ray.ray->origin(), ray.differential_position));
      out.write(1u, make_float4(ray.ray->direction(), ray.differential_direction));
      out.write(2u, make_float4(ray.ray->t_min(), ray.ray->t_max(), 0.5f, 0.0f));
      const PathCyclesSvmKernelGlobals kg{scene, parameters,
          c.orthographic ? CameraProjection::orthographic : CameraProjection::perspective,
          true, true};
      const auto normal = make_float3(0.0f, 0.0f, 1.0f);
      const auto zero = make_float3(0.0f);
      const auto identity = make_float4x4(1.0f);
      psycles::luisa_backend::cycles_svm::ShaderData sd{
          ray.ray->origin(), normal, normal, -ray.ray->direction(),
          1u, 0u, 0u, 0u, 0u, 0.0f, 0.0f, 0u,
          0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          zero, zero, identity, identity};
      sd.object = 0u;
      out.write(3u, make_float4(kg.camera_world_to_ndc(
          sd, ray.ray->origin() + 2.0f * ray.ray->direction()), 0.0f));
      sd.object = psycles::luisa_backend::cycles_svm::object_none;
      out.write(4u, make_float4(kg.camera_world_to_ndc(sd, ray.ray->direction()), 0.0f));
    };
    auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
    for (const auto *function : module->function_list()) {
      if (const auto *definition = function->definition()) {
        definition->traverse_instructions([&](const xir::Instruction *instruction) {
          if (instruction->isa<xir::ArithmeticInst>() &&
              static_cast<const xir::ArithmeticInst *>(instruction)->op() ==
                  xir::ArithmeticOp::MATRIX_INVERSE) {
            std::cerr << "Camera/Window service emitted a device matrix inverse\n";
            passed = false;
          }
        });
      }
    }
    auto shader = device.compile(kernel, {.enable_cache = false, .enable_fast_math = true});
    for (unsigned j = 0u; j < camera_raster_fractions.size(); ++j) {
      unsigned scenario{}, sample{};
      luisa::float2 raster{};
      std::array<luisa::float4, 5u> expected{}, actual{};
      oracle >> scenario >> sample >> raster.x >> raster.y;
      for (auto &v : expected) { oracle >> v.x >> v.y >> v.z >> v.w; }
      if (!oracle || scenario != i || sample != j) { return false; }
      stream << shader(p, raster, filter_buffer, output).dispatch(1u)
             << output.copy_to(actual.data()) << synchronize();
      for (unsigned k = 0u; k < 20u; ++k) {
        const auto a = actual[k / 4u][k % 4u], e = expected[k / 4u][k % 4u];
        // Numerical tolerance, not a bit-exact gate. Compact differentials
        // involve cancellation in the original host precomputation.
        if (!std::isfinite(a) || std::abs(a - e) > 5.0e-6f * std::max(std::abs(e), 1.0f)) {
          std::cerr << "camera=" << i << " sample=" << j << " lane=" << k
                    << " actual=" << a << " Cycles=" << e << '\n';
          passed = false;
        }
      }
    }
  }
  std::string trailing;
  if (oracle >> trailing) { return false; }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? EXIT_SUCCESS : EXIT_FAILURE;
}
