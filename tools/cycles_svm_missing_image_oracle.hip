// Original Cycles 5.2.1 GPU failed-image behavior, not a reference sampler.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles this_file -o /tmp/oracle
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/svm/image.h"
// clang-format on

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr unsigned case_count = 5u * 3u * 4u;

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(float4 *output) {
  const auto index = threadIdx.x;
  if (index >= case_count) { return; }
  const auto image = index / 12u;
  const auto coordinate = (index / 4u) % 3u;
  const auto flags = index % 4u;
  const auto uv = coordinate == 0u ? make_float2(0.25f, 0.75f) :
                  coordinate == 1u ? make_float2(-2.0f, 3.0f) :
                                     make_float2(0.0f, 1.0f);
  ShaderDataTinyStorage storage{};
  output[index] = svm_image_texture(nullptr, AS_SHADER_DATA(&storage),
      image == 4u ? KERNEL_IMAGE_NONE : int(image), dual2{uv}, flags);
}

__global__ void initialize_images(KernelImageTexture *images) {
  const auto index = threadIdx.x;
  constexpr ExtensionType extensions[]{EXTENSION_REPEAT, EXTENSION_EXTEND,
                                        EXTENSION_CLIP, EXTENSION_MIRROR};
  images[index] = KernelImageTexture{};
  images[index].image_info_id = KERNEL_IMAGE_NONE;
  images[index].tile_descriptor_offset = KERNEL_TILE_LOAD_NONE;
  images[index].extension = extensions[index];
}
} // namespace

int main() {
  KernelImageTexture *device_images{};
  check(hipMalloc(&device_images, 4u * sizeof(KernelImageTexture)));
  initialize_images<<<1u, 4u>>>(device_images);
  check(hipGetLastError());
  KernelParamsHIP params{};
  params.image_textures = device_images;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  float4 *output{};
  check(hipMalloc(&output, case_count * sizeof(float4)));
  probe<<<1u, 64u>>>(output);
  check(hipGetLastError());
  std::array<float4, case_count> host{};
  check(hipMemcpy(host.data(), output, sizeof(host), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < case_count; ++i) {
    const auto value = host[i];
    std::printf("%u %.9g %.9g %.9g %.9g\n", i, value.x, value.y, value.z, value.w);
  }
  check(hipFree(output));
  check(hipFree(device_images));
}
