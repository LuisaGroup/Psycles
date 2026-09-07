// Original Cycles 5.2.1 volume-only boundary transitions on their actual GPU SoA.
#include "../tests/cycles_volume_boundary_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/bvh/bvh.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
#include "kernel/integrator/shade_surface.h"
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace psycles::test_support;
__device__ VolumeBoundaryInput prepared[volume_boundary_inputs.size()];

void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}
template<typename T> T *allocate(unsigned count) {
  T *out{};
  check(hipMalloc(&out, count * sizeof(T)));
  return out;
}

__global__ void probe(float4 *out, uint4 *meta) {
  const auto i = threadIdx.x;
  if (i >= volume_boundary_inputs.size()) { return; }
  const auto input = prepared[i];
  const IntegratorState state{int(i)};
  INTEGRATOR_STATE_WRITE(state, path, volume_bounds_bounce) = input.bounds;
  INTEGRATOR_STATE_WRITE(state, path, rng_offset) = 33u;
  INTEGRATOR_STATE_WRITE(state, path, continuation_probability) = input.probability;
  INTEGRATOR_STATE_WRITE(state, path, throughput) = make_float3(0.25f, 0.5f, 0.75f);
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.125f;
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->ray_length = input.distance;
  int label = LABEL_NONE;
  if (!integrate_surface_terminate(state, input.flag)) {
    label = integrate_surface_volume_only_bounce(state, sd);
  }
  out[i] = make_float4(INTEGRATOR_STATE(state, path, throughput),
                       INTEGRATOR_STATE(state, ray, tmin));
  meta[i] = make_uint4(INTEGRATOR_STATE(state, path, volume_bounds_bounce),
                       INTEGRATOR_STATE(state, path, rng_offset), label, 0u);
}

int main() {
  constexpr auto count = unsigned(volume_boundary_inputs.size());
  KernelParamsHIP parameters{};
  auto &state = parameters.integrator_state;
  state.path.volume_bounds_bounce = allocate<uint16_t>(count);
  state.path.rng_offset = allocate<uint16_t>(count);
  state.path.continuation_probability = allocate<float>(count);
  state.path.throughput = allocate<PackedSpectrum>(count);
  state.ray.tmin = allocate<float>(count);
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &parameters, sizeof(parameters)));
  check(hipMemcpyToSymbol(HIP_SYMBOL(prepared), volume_boundary_inputs.data(),
                          sizeof(volume_boundary_inputs)));
  auto *output = allocate<float4>(count);
  auto *metadata = allocate<uint4>(count);
  probe<<<1u, 64u>>>(output, metadata);
  check(hipGetLastError());
  std::vector<float4> rows(count);
  std::vector<uint4> states(count);
  check(hipMemcpy(rows.data(), output, count * sizeof(float4), hipMemcpyDeviceToHost));
  check(hipMemcpy(states.data(), metadata, count * sizeof(uint4), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < count; ++i) {
    const auto a = rows[i];
    const auto b = states[i];
    std::printf("%u %.9g %.9g %.9g %.9g %u %u %u\n", i, a.x, a.y, a.z, a.w,
                b.x, b.y, b.z);
  }
  check(hipDeviceReset());
}
