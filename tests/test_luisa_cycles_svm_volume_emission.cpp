#include "cycles_svm_volume_emission_fixture.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
using namespace luisa::compute;
using namespace psycles::test_support;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace detail = psycles::luisa_backend::detail;

struct Expected { luisa::float3 emission; unsigned flag; };

void require(bool condition, const char *message) {
  if (!condition) { throw std::runtime_error{message}; }
}

auto read_expected() {
  std::array<std::array<Expected, 3u>, volume_emission_case_count> expected{};
  std::ifstream file{PSYCLES_VOLUME_EMISSION_RUNTIME};
  for (auto i = 0u; i < expected.size(); ++i) {
    for (auto domain = 0u; domain < 3u; ++domain) {
      unsigned index{}, phase{};
      auto &value = expected[i][domain];
      file >> index >> phase >> value.emission.x >> value.emission.y >>
          value.emission.z >> value.flag;
      require(bool(file) && index == i && phase == domain,
              "invalid external Cycles HIP emission oracle");
    }
  }
  return expected;
}

void run(Device &device, bool have_object_table) {
  const auto words = read_volume_emission_words(PSYCLES_VOLUME_EMISSION_WORDS);
  const auto expected = read_expected();
  auto stream = device.create_stream();
  auto word_buffer = device.create_buffer<unsigned>(words.size());
  auto output = device.create_buffer<luisa::float4>(volume_emission_case_count);
  auto metadata = device.create_buffer<luisa::uint4>(volume_emission_case_count);
  auto scene = std::make_shared<detail::LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->cycles_svm = std::make_unique<detail::CyclesSvmRuntime>();
  scene->cycles_svm->geometry = std::make_unique<detail::CyclesSvmGeometryRuntime>();
  scene->cycles_svm->objects = std::make_unique<detail::CyclesSvmObjectRuntime>();
  auto &objects = scene->cycles_svm->objects->object_buffer;
  objects = device.create_buffer<abi::KernelObject>(volume_emission_densities.size());
  std::array<abi::KernelObject, volume_emission_densities.size()> records{};
  for (auto i = 0u; i < records.size(); ++i) {
    records[i].volume_density = volume_emission_densities[i];
  }
  stream << word_buffer.copy_from(words.data()) << objects.copy_from(records.data());

  constexpr std::array domains{abi::SHADER_TYPE_VOLUME, abi::SHADER_TYPE_SURFACE,
                                abi::SHADER_TYPE_DISPLACEMENT};
  // Exact Cycles kernel/features.h domain masks. The light-path camera test
  // uses visibility, not a rewritten per-domain shader or folded input.
  constexpr std::array masks{
      abi::kernel_feature_node_emission | abi::kernel_feature_node_volume |
          abi::kernel_feature_node_voronoi_extra | abi::kernel_feature_node_light_path |
          abi::kernel_feature_node_portal,
      svm::kernel_feature_node_mask_surface,
      abi::kernel_feature_node_voronoi_extra | abi::kernel_feature_node_bump |
          abi::kernel_feature_node_bump_state | abi::kernel_feature_node_portal};
  constexpr std::array final_offsets{22u, 5u, 23u};
  std::array<bool, abi::NODE_NUM> used{};
  for (const auto node : {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_LIGHT_PATH,
                         abi::NODE_MATH, abi::NODE_EMISSION_WEIGHT,
                         abi::NODE_CLOSURE_EMISSION}) { used[node] = true; }
  for (auto domain = 0u; domain < domains.size(); ++domain) {
    Kernel1D<Buffer<unsigned>, Buffer<luisa::float4>, Buffer<luisa::uint4>> kernel =
        [=](BufferUInt source, BufferFloat4 result, BufferVar<luisa::uint4> meta) {
          const UInt i = dispatch_x();
          const UInt object = select(i % 4u - 1u, svm::object_none, i % 4u == 0u);
          const Bool accumulate = (i & 8u) != 0u;
          const auto identity = make_float4x4(1.0f);
          Var<detail::RenderKernelParameters> parameters;
          parameters.camera_transform = identity;
          parameters.camera_inverse_transform = identity;
          const detail::PathCyclesSvmKernelGlobals scene_kg{
              scene, parameters, psycles::contract::CameraProjection::perspective,
              true, true};
          const DefaultCyclesSvmKernelGlobals empty_kg;
          const svm::KernelGlobals &kg = have_object_table
              ? static_cast<const svm::KernelGlobals &>(scene_kg)
              : static_cast<const svm::KernelGlobals &>(empty_kg);
          svm::ShaderData sd{make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, 1.0f),
                            0u, 0u, svm::shader_data_is_volume_shader_eval |
                            select(0u, svm::shader_data_emission, accumulate),
                            0u, ~0u, 0.0f, 0.0f, object, 0.0f, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            make_float3(0.0f), make_float3(0.0f), identity, identity};
          sd.closure_emission_background = make_float3(0.125f, -0.25f, 0.5f);
          const svm::TransformState transforms{identity, identity, identity, identity};
          const svm::PathState path{
              select(0u, svm::path_ray_visibility_camera, (i & 4u) == 0u), 0u};
          svm::EvaluationResult status;
          svm::eval_nodes(kg, source, domains[domain], 0u, masks[domain], used,
                          transforms, sd, path, status);
          result.write(i, make_float4(sd.closure_emission_background, 1.0f));
          meta.write(i, make_uint4(sd.flag, status.status, status.final_offset, 0u));
        };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    std::array<luisa::float4, volume_emission_case_count> values{};
    std::array<luisa::uint4, volume_emission_case_count> states{};
    stream << shader(word_buffer, output, metadata).dispatch(volume_emission_case_count)
           << output.copy_to(values.data()) << metadata.copy_to(states.data())
           << synchronize();
    for (auto i = 0u; i < values.size(); ++i) {
      auto reference = expected[i][domain];
      const auto missing_density = !have_object_table && domain == 0u &&
                                   volume_emission_object(i) != svm::object_none;
      if (missing_density) {
        // A missing host/JIT service is not a Cycles configuration. It must
        // fail closed before changing the emission state, including when the
        // dynamic strength is zero. OBJECT_NONE needs no table and still uses
        // the external oracle unchanged.
        reference = {.emission = {0.125f, -0.25f, 0.5f},
                     .flag = svm::shader_data_is_volume_shader_eval |
                         (volume_emission_accumulate(i) ? svm::shader_data_emission : 0u)};
      }
      bool same = states[i].x == reference.flag &&
                  states[i].y == static_cast<unsigned>(missing_density
                      ? svm::EvaluationStatus::unsupported_node
                      : svm::EvaluationStatus::ended) &&
                  states[i].z == (missing_density ? 21u : final_offsets[domain]);
      for (auto lane = 0u; lane < 3u; ++lane) {
        same &= std::isfinite(values[i][lane]) &&
                std::abs(values[i][lane] - reference.emission[lane]) <=
                    2.0e-6f * std::max(1.0f, std::abs(reference.emission[lane]));
      }
      if (!same) {
        std::cerr << "case=" << i << " domain=" << domain
                  << " flag/status/PC=" << states[i].x << '/' << states[i].y << '/'
                  << states[i].z << '\n';
        throw std::runtime_error{"native volume emission differs from external Cycles HIP"};
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    run(device, true);
    run(device, false);
    std::cout << "48 Cycles HIP volume emission/domain cases and 48 missing-service cases passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
