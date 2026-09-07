#include "cycles_svm_subsurface_exit_fixture.h"
#include "cycles_svm_subsurface.h"
#include "cycles_svm_surface_shader.h"
#include "cycles_svm_simple_closure.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <psycles/luisa/cycles_closure.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace detail = svm::detail;
namespace abi = psycles::compiler::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;
using psycles::test_support::subsurface_exit_inputs;
constexpr auto count = unsigned(subsurface_exit_inputs.size());

svm::ShaderData make_shader_data(svm::ClosurePool *pool) {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          normalize(make_float3(0.3f, -0.4f, 1.0f)),
          svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.0f, 0.0f,
          0u, 0.0f, 1.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f), identity, identity, 0u, pool};
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  std::array<luisa::uint4, count> controls{};
  std::array<luisa::float4, count> randoms{};
  for (auto i = 0u; i < count; ++i) {
    const auto &input = subsurface_exit_inputs[i];
    controls[i] = luisa::make_uint4(input.flags, input.normal_case,
                                   input.light_flags, 0u);
    randoms[i] = luisa::make_float4(input.random[0], input.random[1],
                                    input.random[2], 0.0f);
  }
  auto control_buffer = device.create_buffer<luisa::uint4>(count);
  auto random_buffer = device.create_buffer<luisa::float4>(count);
  auto output = device.create_buffer<luisa::float4>(count * 7u);
  auto metadata = device.create_buffer<luisa::uint4>(count * 2u);
  Kernel1D kernel = [](BufferUInt4 controls, BufferFloat4 randoms,
                       BufferFloat4 out, BufferUInt4 meta) {
    const auto i = dispatch_x();
    const auto input = controls.read(i);
    svm::ClosurePool pool{4u};
    auto sd = make_shader_data(&pool);
    sd.flag = input.x | unsigned(abi::SD_EMISSION | abi::SD_HOLDOUT |
                                 abi::SD_BSSRDF);
    detail::diffuse_setup(sd, sd.N, make_float3(0.2f, 0.3f, 0.4f));
    $if(input.y != 1u) {
      const auto a = pool.allocate(closure::type_bssrdf_random_walk,
                                   make_float3(0.6f, -0.9f, 0.6f));
      const auto b = pool.allocate(closure::type_bssrdf_random_walk_skin,
                                   make_float3(-0.2f, -0.4f, -0.6f));
      pool.set_sample_weight(a.index, 0.1f);
      pool.set_sample_weight(b.index, 0.4f);
      pool.set_normal(a.index, make_float3(0.6f, 0.0f, 0.8f));
      pool.set_normal(b.index, make_float3(0.0f, 0.8f, 0.6f));
      $if(input.y == 2u) {
        pool.set_weight(b.index, make_float3(0.6f, -0.9f, 0.6f));
        pool.set_normal(b.index, make_float3(-0.6f, 0.0f, -0.8f));
      }
      $elif(input.y == 3u) {
        pool.set_normal(a.index, make_float3(0.96f, 0.0f, 0.28f));
        pool.set_normal(b.index, make_float3(0.96f, 0.0f, 0.28f));
      };
    };
    pool.set_left(0u);
    detail::subsurface_shader_data_setup(sd);
    const auto pick = detail::surface_shader_bsdf_bssrdf_pick(
        sd, randoms.read(i).xyz());
    psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
    constexpr auto mask = detail::ClosureTypeMask{1u} << closure::type_diffuse;
    const auto sampled = detail::surface_shader_bsdf_sample_closure(
        kg, sd, pick, mask);
    const auto evaluated = detail::surface_shader_bsdf_eval(
        kg, sd, sampled.wo, input.z, mask);
    const auto common = pool.common(0u);
    out.write(i * 7u, make_float4(sd.N, common.sample_weight));
    out.write(i * 7u + 1u, make_float4(common.N, sampled.evaluation.pdf));
    out.write(i * 7u + 2u, make_float4(sampled.evaluation.sum, evaluated.pdf));
    out.write(i * 7u + 3u, make_float4(evaluated.sum,
                                      evaluated.average_roughness_squared));
    out.write(i * 7u + 4u, make_float4(sampled.wo, sampled.eta));
    out.write(i * 7u + 5u, make_float4(sampled.evaluation.diffuse,
                                      sampled.sampled_roughness.x));
    out.write(i * 7u + 6u, make_float4(evaluated.diffuse, pick.random.z));
    meta.write(i * 2u, make_uint4(sd.flag, pool.count(), pool.left(), common.type));
    UInt exit_evaluations = 0u, ordinary_evaluations = 0u;
    $if(detail::surface_shader_material_eval_required(true, input.x)) {
      exit_evaluations += 1u;
    };
    $if(detail::surface_shader_material_eval_required(false, input.x)) {
      ordinary_evaluations += 1u;
    };
    meta.write(i * 2u + 1u, make_uint4(sampled.label, pick.index,
                                       exit_evaluations, ordinary_evaluations));
  };
  auto shader = device.compile(kernel);
  std::array<luisa::float4, count * 7u> values{};
  std::array<luisa::uint4, count * 2u> integers{};
  stream << control_buffer.copy_from(controls.data())
         << random_buffer.copy_from(randoms.data())
         << shader(control_buffer, random_buffer, output, metadata).dispatch(count)
         << output.copy_to(values.data()) << metadata.copy_to(integers.data())
         << synchronize();
  std::ifstream oracle{PSYCLES_SUBSURFACE_EXIT_ORACLE};
  bool passed = true;
  for (auto i = 0u; i < count; ++i) {
    unsigned row{};
    if (!(oracle >> row) || row != i) { return false; }
    for (auto j = 0u; j < 28u; ++j) {
      float expected{};
      if (!(oracle >> expected)) { return false; }
      const auto actual = values[i * 7u + j / 4u][j % 4u];
      if (!std::isfinite(actual) ||
          std::abs(actual - expected) > 2.0e-5f * std::max(1.0f, std::abs(expected))) {
        std::cerr << "SSS exit " << i << " float " << j << ": " << actual
                  << " expected " << expected << '\n';
        passed = false;
      }
    }
    for (auto j = 0u; j < 8u; ++j) {
      unsigned expected{};
      if (!(oracle >> expected)) { return false; }
      const auto actual = integers[i * 2u + j / 4u][j % 4u];
      if (actual != expected) {
        std::cerr << "SSS exit " << i << " uint " << j << ": " << actual
                  << " expected " << expected << '\n';
        passed = false;
      }
    }
  }
  std::string trailing;
  passed &= !(oracle >> trailing) && oracle.eof();
  if (passed) { std::cout << "Cycles HIP SSS exit: " << count << " cases passed\n"; }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1;
}
