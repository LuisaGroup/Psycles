#include "cycles_camera_transform_fixture.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_light.h"

#include <psycles/compiler/cycles_transform.h>

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
using psycles::test_support::camera_transform_cases;
namespace abi = psycles::compiler::cycles_svm;

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->cycles_svm->geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  scene->cycles_svm->objects = std::make_unique<CyclesSvmObjectRuntime>();
  scene->cycles_background_shader_id = 0u;
  scene->cycles_background_object_index = 0u;
  scene->cycles_svm->kernel_shader_buffer =
      device.create_buffer<abi::KernelShader>(1u);
  // The production setup records both sides of its runtime background branch.
  // Even its inactive object-resource arguments must have valid bindings.
  scene->cycles_svm->objects->object_buffer =
      device.create_buffer<abi::KernelObject>(1u);
  scene->cycles_svm->objects->object_flag_buffer = device.create_buffer<unsigned>(1u);
  const abi::KernelShader shader_record{};
  const abi::KernelObject object_record{};
  const unsigned object_flags{};
  stream << scene->cycles_svm->kernel_shader_buffer->copy_from(&shader_record)
         << scene->cycles_svm->objects->object_buffer.copy_from(&object_record)
         << scene->cycles_svm->objects->object_flag_buffer.copy_from(&object_flags)
         << synchronize();

  Kernel1D<RenderKernelParameters, luisa::float3, Buffer<luisa::float4>>
      kernel = [scene](Var<RenderKernelParameters> parameters, Float3 position,
                       BufferFloat4 output) noexcept {
        const PathCyclesSvmKernelGlobals kg{
            scene, parameters, psycles::contract::CameraProjection::orthographic,
            true, true};
        Var<DirectLightTaskCall> task;
        task.light_object = 0u;
        task.ray_direction = position;
        auto light = setup_cycles_svm_light_shader_data(scene, kg, task, parameters);
        for (auto c = 0u; c < 4u; ++c) {
          output.write(c, light.transforms.world_to_camera[c]);
        }
        const auto ndc = kg.camera_world_to_ndc(light.shader_data, position);
        output.write(4u, make_float4(ndc, 0.0f));
      };

  // This is the actual production KernelGlobals/NEE adapter, not a test-only
  // matrix load. Camera updates are kernel arguments; no compile-time constant
  // folding is allowed to hide a device inversion from the regression.
  auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
  unsigned inverses{};
  for (const auto *function : module->function_list()) {
    if (const auto *definition = function->definition()) {
      definition->traverse_instructions([&](const xir::Instruction *inst) {
        inverses += inst->isa<xir::ArithmeticInst>() &&
                    static_cast<const xir::ArithmeticInst *>(inst)->op() ==
                        xir::ArithmeticOp::MATRIX_INVERSE;
      });
    }
  }
  bool passed = inverses == 0u;
  if (!passed) {
    std::cerr << "Production camera adapters emit " << inverses
              << " device matrix inversions; Cycles uploads worldtocamera\n";
  }
  auto shader = device.compile(kernel);
  auto output = device.create_buffer<luisa::float4>(5u);
  std::ifstream oracle{PSYCLES_CAMERA_TRANSFORM_ORACLE};
  if (!oracle) {
    std::cerr << "Missing Cycles camera transform oracle\n";
    return false;
  }
  for (auto i = 0u; i < camera_transform_cases.size(); ++i) {
    const auto &input = camera_transform_cases[i];
    RenderKernelParameters parameters{};
    parameters.camera_transform = to_luisa(input.camera_to_world);
    parameters.camera_inverse_transform = to_luisa(
        psycles::compiler::cycles_inverse_affine_transform(input.camera_to_world));
    parameters.full_width = parameters.full_height = 1u;
    parameters.camera_ortho_vertical_span = 2.0f;
    std::array<luisa::float4, 5u> actual{};
    stream << shader(parameters, to_luisa(input.position), output).dispatch(1u)
           << output.copy_to(actual.data()) << synchronize();
    unsigned scenario{};
    oracle >> scenario;
    passed &= scenario == i;
    for (auto j = 0u; j < 20u; ++j) {
      float expected{};
      oracle >> expected;
      const auto value = actual[j / 4u][j % 4u];
      if (!std::isfinite(value) ||
          std::abs(value - expected) > 5.0e-5f * std::max(1.0f, std::abs(expected))) {
        std::cerr << "Camera case=" << i << " lane=" << j << " got=" << value
                  << " expected=" << expected << '\n';
        passed = false;
      }
    }
    if (!oracle) {
      std::cerr << "Malformed Cycles camera transform oracle\n";
      return false;
    }
  }
  std::string trailing;
  if (oracle >> trailing) {
    std::cerr << "Unexpected trailing camera oracle data\n";
    passed = false;
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? EXIT_SUCCESS : EXIT_FAILURE;
}
