// Cycles 5.2.1 HIP oracle: execute the original shader lookup and sort-key
// macro. This does not implement a reference renderer or sorting algorithm.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_wavefront_sort_oracle.hip -o /tmp/wavefront-sort-oracle
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/bvh/util.h"
#include "kernel/integrator/state_flow.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {
constexpr unsigned count = 6u;
constexpr unsigned shader_count = 39u;
constexpr unsigned partition_size = 65536u;

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(uint4 *output, const unsigned *states) {
  const unsigned row = threadIdx.x;
  if (row >= count) {
    return;
  }
  const int type = row < 4u ? PRIMITIVE_TRIANGLE : PRIMITIVE_CURVE;
  const int prim = row < 4u ? row : row - 4u;
  const auto shader = intersection_get_shader_from_isect_prim(nullptr, prim, type);
  output[row] =
      make_uint4(row, states[row], shader, INTEGRATOR_SORT_KEY(shader, states[row]));
}
} // namespace

int main() {
  const std::array<unsigned, 4u> triangles{
      12u | SHADER_SMOOTH_NORMAL | SHADER_CAST_SHADOW, 19u | SHADER_CAST_SHADOW,
      12u | SHADER_CAST_SHADOW, 38u | SHADER_USE_MIS};
  std::array<KernelCurve, 2u> curves{};
  curves[0].shader_id = 3u | SHADER_CAST_SHADOW;
  curves[1].shader_id = 6u | SHADER_USE_MIS;
  const std::array<unsigned, count> states{0u,       65535u,  65536u,
                                           1048575u, 131072u, 1048575u};
  unsigned *device_triangles{}, *device_states{};
  KernelCurve *device_curves{};
  uint4 *device_output{};
  check(hipMalloc(&device_triangles, sizeof(triangles)));
  check(hipMalloc(&device_curves, sizeof(curves)));
  check(hipMalloc(&device_states, sizeof(states)));
  check(hipMalloc(&device_output, count * sizeof(uint4)));
  check(hipMemcpy(device_triangles, triangles.data(), sizeof(triangles),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(device_curves, curves.data(), sizeof(curves), hipMemcpyHostToDevice));
  check(hipMemcpy(device_states, states.data(), sizeof(states), hipMemcpyHostToDevice));
  auto params = std::make_unique<KernelParamsHIP>();
  params->tri_shader = device_triangles;
  params->curves = device_curves;
  params->data.max_shaders = shader_count;
  params->integrator_state.sort_partition_divisor = partition_size;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), params.get(), sizeof(*params)));
  probe<<<1u, 32u>>>(device_output, device_states);
  check(hipGetLastError());
  std::array<uint4, count> result{};
  check(hipMemcpy(result.data(), device_output, sizeof(result), hipMemcpyDeviceToHost));
  std::printf("configuration %u %u\n", shader_count, partition_size);
  for (const auto &row : result) {
    std::printf("key %u %u %u %u\n", row.x, row.y, row.z, row.w);
  }
  check(hipFree(device_output));
  check(hipFree(device_states));
  check(hipFree(device_curves));
  check(hipFree(device_triangles));
}
