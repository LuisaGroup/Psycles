// Calls Cycles 5.2.1 itself on HIP; no reference shading formula is duplicated.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_surface_roughness_oracle.hip -o /tmp/surface-roughness-oracle
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/surface_shader.h"

#include "../tests/cycles_surface_roughness_cases.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

using psycles::test_support::RoughnessCase;
using psycles::test_support::RoughnessClosureInput;
using psycles::test_support::roughness_cases;
constexpr unsigned scenario_count = roughness_cases.size();

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__device__ void append(ShaderData &sd, const RoughnessClosureInput &input) {
  static_assert(sizeof(OrenNayarBsdf) <= sizeof(ShaderClosure));
  static_assert(sizeof(MicrofacetBsdf) <= sizeof(ShaderClosure));
  const auto type = static_cast<ClosureType>(input.type);
  auto &closure = sd.closure[sd.num_closure++];
  closure.type = type;
  closure.weight = make_float3(input.weight[0], input.weight[1], input.weight[2]);
  closure.N = make_float3(0.0f, 0.0f, 1.0f);
  if (type == CLOSURE_BSDF_OREN_NAYAR_ID ||
      type == CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID) {
    reinterpret_cast<OrenNayarBsdf *>(&closure)->param.roughness = input.roughness;
  }
  if (CLOSURE_IS_BSDF_MICROFACET(type)) {
    auto *bsdf = reinterpret_cast<MicrofacetBsdf *>(&closure);
    bsdf->alpha_x = input.alpha_x;
    bsdf->alpha_y = input.alpha_y;
  }
}

__global__ void probe(const RoughnessCase *cases, float *values) {
  const auto scenario = threadIdx.x;
  if (scenario >= scenario_count) { return; }
  ShaderData sd{};
  sd.N = make_float3(0.0f, 0.0f, 1.0f);
  for (auto i = 0u; i < cases[scenario].count; ++i) {
    append(sd, cases[scenario].closures[i]);
  }
  values[scenario] = surface_shader_average_roughness(&sd);
}

}  // namespace

int main() {
  RoughnessCase *cases{};
  check(hipMalloc(&cases, sizeof(roughness_cases)));
  check(hipMemcpy(cases, roughness_cases.data(), sizeof(roughness_cases),
                  hipMemcpyHostToDevice));
  float *values{};
  check(hipMalloc(&values, scenario_count * sizeof(float)));
  probe<<<1u, 64u>>>(cases, values);
  check(hipGetLastError());
  std::array<float, scenario_count> host{};
  check(hipMemcpy(host.data(), values, sizeof(host), hipMemcpyDeviceToHost));
  for (unsigned i = 0u; i < scenario_count; ++i) {
    std::printf("%u %.9g\n", i, host[i]);
  }
  check(hipFree(values));
  check(hipFree(cases));
}
