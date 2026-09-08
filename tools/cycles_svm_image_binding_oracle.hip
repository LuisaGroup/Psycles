// Original Cycles 5.2.1 image sampler, not an independent evaluator.
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
constexpr unsigned handle_count = 16u;
constexpr unsigned coordinate_count = 5u;
constexpr unsigned case_count = handle_count * coordinate_count;

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void initialize_images(KernelImageTexture *images) {
  const auto index = threadIdx.x;
  images[index] = KernelImageTexture{};
  images[index].image_info_id = index;
  images[index].tile_descriptor_offset = KERNEL_TILE_LOAD_NONE;
}

__global__ void probe(float4 *output) {
  const auto index = threadIdx.x;
  if (index >= case_count) { return; }
  constexpr float2 coordinates[]{
      {0.375f, 0.5f}, {-0.125f, 0.5f}, {1.125f, 0.5f},
      {0.125f, -0.5f}, {0.875f, 1.5f}};
  ShaderDataTinyStorage storage{};
  // This fixture checks backend sampler binding over Psycles' top-down image
  // upload, not the shader-to-image orientation adapter. Give original Cycles
  // the same physical storage coordinate. Whole-scene Cycles comparisons keep
  // their original UVs and remain the end-to-end oracle for that adapter.
  const auto uv = coordinates[index % coordinate_count];
  output[index] = svm_image_texture(nullptr, AS_SHADER_DATA(&storage),
      int(index / coordinate_count), dual2{make_float2(uv.x, 1.0f - uv.y)}, 0u);
}
} // namespace

int main() {
  // Inputs match the integration fixture's two-texel, linear-data image.
  constexpr std::array pixels{uchar4{255u, 0u, 0u, 255u},
                               uchar4{0u, 255u, 0u, 255u}};
  hipArray_t array{};
  const auto channel = hipCreateChannelDesc<uchar4>();
  check(hipMallocArray(&array, &channel, 2u, 1u));
  check(hipMemcpy2DToArray(array, 0u, 0u, pixels.data(), sizeof(pixels),
                          sizeof(pixels), 1u, hipMemcpyHostToDevice));
  constexpr std::array interpolation{INTERPOLATION_CLOSEST, INTERPOLATION_LINEAR,
                                      INTERPOLATION_CUBIC, INTERPOLATION_SMART};
  constexpr std::array extension{EXTENSION_REPEAT, EXTENSION_EXTEND,
                                  EXTENSION_CLIP, EXTENSION_MIRROR};
  constexpr std::array addresses{hipAddressModeWrap, hipAddressModeClamp,
                                  hipAddressModeBorder, hipAddressModeMirror};
  std::array<hipTextureObject_t, handle_count> textures{};
  std::array<KernelImageInfo, handle_count> info{};
  for (auto i = 0u; i < handle_count; ++i) {
    hipResourceDesc resource{};
    resource.resType = hipResourceTypeArray;
    resource.res.array.array = array;
    hipTextureDesc sampler{};
    sampler.addressMode[0] = sampler.addressMode[1] = sampler.addressMode[2] = addresses[i % 4u];
    sampler.filterMode = i / 4u == 0u ? hipFilterModePoint : hipFilterModeLinear;
    sampler.readMode = hipReadModeNormalizedFloat;
    sampler.normalizedCoords = 1;
    check(hipCreateTextureObject(&textures[i], &resource, &sampler, nullptr));
    info[i].data = reinterpret_cast<uint64_t>(textures[i]);
    info[i].data_type = IMAGE_DATA_TYPE_BYTE4;
    info[i].interpolation = interpolation[i / 4u];
    info[i].extension = extension[i % 4u];
    info[i].width = 2u;
    info[i].height = 1u;
    info[i].inv_width = 0.5f;
    info[i].inv_height = 1.0f;
  }
  KernelImageTexture *device_images{};
  KernelImageInfo *device_info{};
  check(hipMalloc(&device_images, handle_count * sizeof(KernelImageTexture)));
  check(hipMalloc(&device_info, sizeof(info)));
  check(hipMemcpy(device_info, info.data(), sizeof(info), hipMemcpyHostToDevice));
  initialize_images<<<1u, handle_count>>>(device_images);
  check(hipGetLastError());
  KernelParamsHIP params{};
  params.image_textures = device_images;
  params.image_info = device_info;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  float4 *output{};
  check(hipMalloc(&output, case_count * sizeof(float4)));
  probe<<<1u, 128u>>>(output);
  check(hipGetLastError());
  std::array<float4, case_count> host{};
  check(hipMemcpy(host.data(), output, sizeof(host), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < case_count; ++i) {
    const auto value = host[i];
    std::printf("%u %.9g %.9g %.9g %.9g\n", i, value.x, value.y, value.z, value.w);
  }
  check(hipFree(output));
  check(hipFree(device_info));
  check(hipFree(device_images));
  for (const auto texture : textures) { check(hipDestroyTextureObject(texture)); }
  check(hipFreeArray(array));
}
