// Original Cycles 5.2.1 SSS exit state and BSDF, executed on HIP.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 \
//   -O3 -ffast-math -I include -I /path/to/blender/intern/cycles \
//   tools/cycles_svm_subsurface_exit_oracle.hip -o /tmp/sss-exit-oracle
#include "../tests/cycles_svm_subsurface_exit_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/subsurface.h"
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using psycles::test_support::SubsurfaceExitInput;
using psycles::test_support::subsurface_exit_inputs;
constexpr auto count = unsigned(subsurface_exit_inputs.size());
static_assert(SHADER_USE_MIS == (1u << 28u));
static_assert(SHADER_EXCLUDE_DIFFUSE == (1u << 27u));

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(const SubsurfaceExitInput *inputs, float4 *out,
                       uint4 *meta) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) { return; }
  const auto input = inputs[i];
  ShaderData sd{};
  sd.N = sd.Ng = make_float3(0.0f, 0.0f, 1.0f);
  sd.wi = normalize(make_float3(0.3f, -0.4f, 1.0f));
  sd.type = PRIMITIVE_TRIANGLE;
  sd.object = 0;
  sd.flag = input.flags | SD_EMISSION | SD_HOLDOUT | SD_BSSRDF;
  sd.num_closure_left = 4;
  bsdf_diffuse_setup(&sd, sd.N, make_float3(0.2f, 0.3f, 0.4f));
  if (input.normal_case != 1u) {
    auto &a = sd.closure[sd.num_closure++];
    a.type = CLOSURE_BSSRDF_RANDOM_WALK_ID;
    a.weight = make_float3(0.6f, -0.9f, 0.6f);
    a.sample_weight = 0.1f;
    a.N = make_float3(0.6f, 0.0f, 0.8f);
    auto &b = sd.closure[sd.num_closure++];
    b.type = CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID;
    b.weight = make_float3(-0.2f, -0.4f, -0.6f);
    b.sample_weight = 0.4f;
    b.N = make_float3(0.0f, 0.8f, 0.6f);
    if (input.normal_case == 2u) {
      b.weight = a.weight;
      b.N = -a.N;
    } else if (input.normal_case == 3u) {
      a.N = b.N = make_float3(0.96f, 0.0f, 0.28f);
    }
    sd.num_closure_left -= 2;
  }
  // Simulate an exhausted extra-payload tail. Exit setup must recover the
  // entire capacity, independently of old live closure and extra counts.
  sd.num_closure_left = 0;
  subsurface_shader_data_setup(nullptr, &sd);
  auto random = make_float3(input.random[0], input.random[1], input.random[2]);
  const auto *selected = surface_shader_bsdf_bssrdf_pick(&sd, &random);
  BsdfEval sampled{};
  float3 wo = zero_float3();
  float pdf = 0.0f, eta = 0.0f, roughness = 0.0f;
  float2 sampled_roughness = zero_float2();
  const auto label = surface_shader_bsdf_sample_closure(
      nullptr, &sd, selected, random, &sampled, &wo, &pdf,
      &sampled_roughness, &eta, roughness);
  BsdfEval evaluated{};
  const auto eval_pdf = surface_shader_bsdf_eval(
      nullptr, 0, &sd, wo, &evaluated, input.light_flags, roughness);
  out[i * 7u] = make_float4(sd.N, sd.closure[0].sample_weight);
  out[i * 7u + 1u] = make_float4(sd.closure[0].N, pdf);
  out[i * 7u + 2u] = make_float4(sampled.sum, eval_pdf);
  out[i * 7u + 3u] = make_float4(evaluated.sum, roughness);
  out[i * 7u + 4u] = make_float4(wo, eta);
  out[i * 7u + 5u] = make_float4(sampled.diffuse, sampled_roughness.x);
  out[i * 7u + 6u] = make_float4(evaluated.diffuse, random.z);
  meta[i * 2u] = make_uint4(sd.flag, sd.num_closure, sd.num_closure_left,
                            sd.closure[0].type);
  meta[i * 2u + 1u] = make_uint4(label, selected - sd.closure,
      (!(PATH_RAY_SUBSURFACE_RANDOM_WALK & PATH_RAY_SUBSURFACE) ||
       (input.flags & SD_HAS_BSSRDF_BUMP)) ? 1u : 0u,
      (!(0u & PATH_RAY_SUBSURFACE) ||
       (input.flags & SD_HAS_BSSRDF_BUMP)) ? 1u : 0u);
}
} // namespace

int main() {
  KernelObject object{};
  object.shadow_terminator_shading_offset = 1.0f;
  KernelObject *objects{};
  check(hipMalloc(&objects, sizeof(object)));
  check(hipMemcpy(objects, &object, sizeof(object), hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.data.max_closures = 4;
  params.objects = objects;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  SubsurfaceExitInput *inputs{};
  float4 *values{};
  uint4 *metadata{};
  check(hipMalloc(&inputs, sizeof(subsurface_exit_inputs)));
  check(hipMalloc(&values, count * 7u * sizeof(float4)));
  check(hipMalloc(&metadata, count * 2u * sizeof(uint4)));
  check(hipMemcpy(inputs, subsurface_exit_inputs.data(), sizeof(subsurface_exit_inputs),
                  hipMemcpyHostToDevice));
  probe<<<(count + 63u) / 64u, 64u>>>(inputs, values, metadata);
  check(hipGetLastError());
  std::vector<float4> floats(count * 7u);
  std::vector<uint4> integers(count * 2u);
  check(hipMemcpy(floats.data(), values, floats.size() * sizeof(float4), hipMemcpyDeviceToHost));
  check(hipMemcpy(integers.data(), metadata, integers.size() * sizeof(uint4), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < count; ++i) {
    std::printf("%u", i);
    for (auto j = 0u; j < 7u; ++j) {
      const auto v = floats[i * 7u + j];
      std::printf(" %.9g %.9g %.9g %.9g", v.x, v.y, v.z, v.w);
    }
    for (auto j = 0u; j < 2u; ++j) {
      const auto v = integers[i * 2u + j];
      std::printf(" %u %u %u %u", v.x, v.y, v.z, v.w);
    }
    std::puts("");
  }
  check(hipFree(inputs));
  check(hipFree(values));
  check(hipFree(metadata));
  check(hipFree(objects));
}
