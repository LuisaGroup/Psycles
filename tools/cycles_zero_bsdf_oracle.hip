// Original Cycles HIP sampler witness: a valid sampling density is not a
// proof that the returned BSDF contribution is nonzero.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 \
//   -O3 -ffast-math -I /path/to/blender/intern/cycles \
//   tools/cycles_zero_bsdf_oracle.hip -o /tmp/zero-bsdf-oracle
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/util/colorspace.h"
#include "kernel/closure/bsdf_ashikhmin_velvet.h"

#include <cstdio>
#include <cstdlib>

namespace {
void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}

__global__ void probe(float4 *result) {
  VelvetBsdf bsdf{};
  bsdf.N = make_float3(0, 0, 1);
  bsdf.sigma = threadIdx.x == 0 ? 0.01f : 0.5f;
  bsdf_ashikhmin_velvet_setup(&bsdf);
  Spectrum value{};
  float3 direction{};
  float pdf{};
  const auto label = bsdf_ashikhmin_velvet_sample(
      reinterpret_cast<const ShaderClosure *>(&bsdf), bsdf.N, bsdf.N,
      make_float2(0.1f, 0.5f), &value, &direction, &pdf);
  result[threadIdx.x * 2] = make_float4(spectrum_to_rgb(value), pdf);
  result[threadIdx.x * 2 + 1] = make_float4(direction, float(label));
}
} // namespace

int main() {
  float4 *device{};
  check(hipMalloc(&device, 4 * sizeof(float4)));
  probe<<<1, 2>>>(device);
  check(hipGetLastError());
  float4 result[4]{};
  check(hipMemcpy(result, device, sizeof(result), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 4; ++i) {
    const auto v = result[i];
    std::printf("%u %.9g %.9g %.9g %.9g\n", i, v.x, v.y, v.z, v.w);
  }
  check(hipFree(device));
}
