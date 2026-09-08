// Observe original Cycles GPU remainder and Noise under its HIPRT flags.
// hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 -ffast-math \
//   -I /path/to/blender/intern/cycles tools/cycles_fmod_noise_oracle.hip -o PROBE
// Dynamic device inputs prevent host folding; no shader formula is copied.
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/svm/noise.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void probe(const float4 *coordinates, const float *divisors,
                      float4 *outputs, unsigned count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) { return; }
  const auto p = coordinates[index];
  const auto divisor = divisors[index];
  const auto vector3 = fmod(make_float3(p.x, p.y, p.z), divisor);
  outputs[index * 4] = make_float4(fmodf(p.x, divisor), vector3.x, vector3.y, vector3.z);
  outputs[index * 4 + 1] = fmod(p, divisor);
  outputs[index * 4 + 2] = make_float4(snoise_1d(p.x), snoise_2d(make_float2(p.x, p.y)),
      snoise_3d(make_float3(p.x, p.y, p.z)), snoise_4d(p));
  outputs[index * 4 + 3] = make_float4(fmodf(p.x, 100000.0f), fmodf(p.y, 100000.0f),
      fmodf(p.z, 100000.0f), fmodf(p.w, 100000.0f));
}
} // namespace

int main() {
  constexpr unsigned count = 8;
  const std::array<float4, count> coordinates{{
      {5.75f, -5.75f, 0.125f, -0.125f},
      {1.0e17f, -1.0e17f, 1.0e12f, -1.0e12f},
      {99999.5f, 100000.0f, 100000.5f, -100000.5f},
      {999999.5f, 1000000.0f, 1000000.5f, -1000000.5f},
      {1.0e17f, -1.0e17f, 1.0e30f, -1.0e30f},
      {0.0f, -0.0f, 2.0f, -2.0f},
      {0.3f, 0.7f, -0.3f, -0.7f},
      {12345.625f, -12345.625f, 1024.25f, -1024.25f}}};
  const std::array<float, count> divisors{{2.0f, 6.283185307179586f, 100000.0f,
      100000.0f, 100000.0f, 2.0f, 0.1f, 7.0f}};
  float4 *device_coordinates{}, *device_outputs{};
  float *device_divisors{};
  check(hipMalloc(&device_coordinates, sizeof(coordinates)));
  check(hipMalloc(&device_divisors, sizeof(divisors)));
  check(hipMalloc(&device_outputs, count * 4 * sizeof(float4)));
  check(hipMemcpy(device_coordinates, coordinates.data(), sizeof(coordinates), hipMemcpyHostToDevice));
  check(hipMemcpy(device_divisors, divisors.data(), sizeof(divisors), hipMemcpyHostToDevice));
  probe<<<1, 32>>>(device_coordinates, device_divisors, device_outputs, count);
  check(hipGetLastError());
  std::array<float4, count * 4> output;
  check(hipMemcpy(output.data(), device_outputs, sizeof(output), hipMemcpyDeviceToHost));
  for (unsigned index = 0; index < count; ++index) {
    for (unsigned slot = 0; slot < 4; ++slot) {
      const auto value = output[index * 4 + slot];
      std::array<unsigned, 4> bits;
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(bits.data(), &value, sizeof(value));
      std::printf("%u %u %08x %08x %08x %08x %.9g %.9g %.9g %.9g\n", index, slot,
          bits[0], bits[1], bits[2], bits[3], value.x, value.y, value.z, value.w);
    }
  }
  check(hipFree(device_outputs));
  check(hipFree(device_divisors));
  check(hipFree(device_coordinates));
}
