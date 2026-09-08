// Descriptor-layout experiment only. No renderer, SVM, or software sampler.
// Native HIP tex2D is the GPU control; all experimental variants issue the
// same AMD image intrinsic with descriptors copied from real HIP objects.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Image = unsigned __attribute__((ext_vector_type(8)));
using Sampler = unsigned __attribute__((ext_vector_type(4)));
using Color = float __attribute__((ext_vector_type(4)));
extern "C" __device__ Color native_sample(unsigned, float, float, Image, Sampler,
                                         bool, unsigned, unsigned)
    __asm("llvm.amdgcn.image.sample.lz.2d.v4f32.f32.v8i32.v4i32");

constexpr unsigned image_count = 256, descriptor_count = 4096;
constexpr unsigned texel_side = 64, lane_count = 1u << 20, iterations = 16;
constexpr unsigned repetitions = 9;
constexpr unsigned layout_count = 7;
struct alignas(32) NativePrefix { unsigned words[20]; };
static_assert(HIP_IMAGE_OBJECT_SIZE_DWORD == 12 &&
              HIP_SAMPLER_OBJECT_OFFSET_DWORD == 12 &&
              HIP_SAMPLER_OBJECT_SIZE_DWORD == 8);
__device__ Image load_image(const void *pointer) {
  return *(const Image __attribute__((address_space(1))) *)pointer;
}
__device__ Sampler load_sampler(const void *pointer) {
  return *(const Sampler __attribute__((address_space(1))) *)pointer;
}
struct Slot {
  std::uint64_t buffer, size;
  const Image *image;
  std::uint64_t levels_sampler, size_xy;
  std::uint64_t tex3d, levels3d, size3d_xy, size3d_z;
};
static_assert(sizeof(Slot) == 72);
struct alignas(32) Combined {
  Image image;
  Sampler sampler;
};
static_assert(sizeof(Image) == 32 && sizeof(Sampler) == 16);
static_assert(sizeof(Combined) == 64);

void checked(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
template<typename T> T *upload(const std::vector<T> &data) {
  T *result{};
  checked(hipMalloc(&result, data.size() * sizeof(T)));
  checked(hipMemcpy(result, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
  return result;
}

template<unsigned layout>
__global__ void probe(const Slot *slots, const Image *images, const Sampler *samplers,
                      const Combined *combined, const hipTextureObject_t *objects,
                      const hipTextureObject_t *clones,
                      const unsigned *indices, float4 *output) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto base = indices[tid];
  auto x = float(tid & 1023u) * (1.5f / 1024.f) - .25f;
  auto y = float((tid >> 10u) & 1023u) * (1.5f / 1024.f) - .25f;
  Color sum{};
  #pragma unroll 1
  for (auto i = 0u; i < iterations; ++i) {
    const auto index = (base + i * 131u) & (descriptor_count - 1u);
    Color value;
    if constexpr (layout == 6) {
      // In this fixture a descriptor index identifies an immutable image /
      // sampler pair. Equal indices therefore imply equal SRDs. Group on
      // this key before loading descriptor vectors, preserving every lane's
      // own coordinates. No uniform-texture assumption is made.
      bool pending = true;
      while (pending) {
        const auto chosen = __builtin_amdgcn_readfirstlane(index);
        if (index == chosen) {
          const auto slot = slots[chosen];
          const auto sampler = samplers[(slot.levels_sampler >> 8u) & 15u];
          const auto image = load_image(slot.image);
          value = native_sample(15u, x, y, image, sampler, false, 0u, 0u);
          pending = false;
        }
      }
    } else if constexpr (layout == 3 || layout == 4) {
      auto sampled = tex2D<float4>((layout == 3 ? objects : clones)[index], x, y);
      value = Color{sampled.x, sampled.y, sampled.z, sampled.w};
    } else {
      Image image;
      Sampler sampler;
      if constexpr (layout == 5) {
        image = load_image(objects[index]);
        sampler = load_sampler(reinterpret_cast<const std::byte *>(objects[index]) + 48u);
      } else if constexpr (layout == 2) {
        image = combined[index].image;
        sampler = combined[index].sampler;
      } else {
        auto code = (slots[index].levels_sampler >> 8u) & 15u;
        sampler = samplers[code];
        if constexpr (layout == 0) { image = load_image(slots[index].image); }
        else { image = images[index]; }
      }
      value = native_sample(15u, x, y, image, sampler, false, 0u, 0u);
    }
    sum += value;
    // A real data-dependent coordinate chain prevents independent fetch
    // scheduling from hiding every load-to-sample latency in this experiment.
    x = x * .97f + .013f + value.x * .01f;
    y = y * .97f + .017f + value.y * .01f;
  }
  output[tid] = make_float4(sum.x, sum.y, sum.z, sum.w);
}

int main() {
  hipDeviceProp_t property{};
  checked(hipGetDeviceProperties(&property, 0));
  std::fprintf(stderr, "device=%s arch=%s lanes=%u samples_per_lane=%u descriptors=%u actual_images=%u repeats=%u\n",
      property.name, property.gcnArchName, lane_count, iterations, descriptor_count,
      image_count, repetitions);
  const std::array addresses{hipAddressModeWrap, hipAddressModeClamp,
                             hipAddressModeBorder, hipAddressModeMirror};
  std::vector<hipArray_t> arrays(image_count);
  std::vector<hipTextureObject_t> native_objects(image_count * addresses.size());
  std::vector<uchar4> pixels(texel_side * texel_side);
  for (auto index = 0u; index < image_count; ++index) {
    for (auto p = 0u; p < pixels.size(); ++p) {
      pixels[p] = make_uchar4((p * 17u + index * 19u) & 255u,
          (p * 13u + index * 23u) & 255u, (p * 3u + index * 7u) & 255u, 255u);
    }
    auto channel = hipCreateChannelDesc<uchar4>();
    checked(hipMallocArray(&arrays[index], &channel, texel_side, texel_side));
    checked(hipMemcpy2DToArray(arrays[index], 0, 0, pixels.data(), texel_side * 4u,
                               texel_side * 4u, texel_side, hipMemcpyHostToDevice));
    for (auto address = 0u; address < addresses.size(); ++address) {
      hipResourceDesc resource{};
      resource.resType = hipResourceTypeArray;
      resource.res.array.array = arrays[index];
      hipTextureDesc sampler{};
      sampler.addressMode[0] = sampler.addressMode[1] = sampler.addressMode[2] = addresses[address];
      sampler.filterMode = hipFilterModeLinear;
      sampler.readMode = hipReadModeNormalizedFloat;
      sampler.normalizedCoords = 1;
      checked(hipCreateTextureObject(&native_objects[index + address * image_count],
                                      &resource, &sampler, nullptr));
    }
  }
  std::vector<Image> images(descriptor_count);
  std::vector<Sampler> samplers(16);
  std::vector<Combined> combined(descriptor_count);
  std::vector<hipTextureObject_t> objects(descriptor_count);
  std::vector<NativePrefix> prefixes(descriptor_count);
  for (auto i = 0u; i < descriptor_count; ++i) {
    objects[i] = native_objects[i % native_objects.size()];
    checked(hipMemcpy(&prefixes[i], objects[i], 80u, hipMemcpyDeviceToHost));
    checked(hipMemcpy(&images[i], objects[i], sizeof(Image), hipMemcpyDeviceToHost));
    auto address = (i / image_count) & 3u;
    Sampler sampler;
    checked(hipMemcpy(&sampler, reinterpret_cast<const std::byte *>(objects[i]) +
                       HIP_SAMPLER_OBJECT_OFFSET_DWORD * sizeof(unsigned),
                       sizeof(Sampler), hipMemcpyDeviceToHost));
    if (i < image_count * addresses.size() && i % image_count == 0) {
      samplers[address] = sampler;
    } else if (std::memcmp(&sampler, &samplers[address], sizeof(Sampler))) {
      std::fprintf(stderr, "sampler unexpectedly depends on image at %u\n", i);
      return 1;
    }
    combined[i] = {images[i], sampler};
  }
  auto device_images = upload(images);
  auto device_samplers = upload(samplers);
  auto device_combined = upload(combined);
  auto device_objects = upload(objects);
  auto device_prefixes = upload(prefixes);
  std::vector<hipTextureObject_t> clones(descriptor_count);
  for (auto i = 0u; i < descriptor_count; ++i) {
    // Device tex2D reads only this SRD prefix. These are NOT runtime objects:
    // never pass a cloned prefix to hipDestroyTextureObject or other host APIs.
    clones[i] = reinterpret_cast<hipTextureObject_t>(device_prefixes + i);
  }
  auto device_clones = upload(clones);
  std::vector<Slot> slots(descriptor_count);
  for (auto i = 0u; i < descriptor_count; ++i) {
    slots[i].image = device_images + i;
    slots[i].levels_sampler = 1u | (((i / image_count) & 3u) << 8u);
    slots[i].size_xy = std::uint64_t{texel_side} | (std::uint64_t{texel_side} << 32u);
  }
  auto device_slots = upload(slots);
  float4 *output{};
  checked(hipMalloc(&output, lane_count * sizeof(float4)));
  std::vector<unsigned> indices(lane_count);
  auto device_indices = upload(indices);
  std::vector<float4> reference(lane_count), candidate(lane_count);
  hipEvent_t start{}, finish{};
  checked(hipEventCreate(&start));
  checked(hipEventCreate(&finish));
  auto launch = [&](unsigned layout) {
    #define LAUNCH(n) probe<n><<<lane_count / 256u, 256u>>>(device_slots, device_images, \
        device_samplers, device_combined, device_objects, device_clones, device_indices, output)
    switch (layout) {
      case 0: LAUNCH(0); break;
      case 1: LAUNCH(1); break;
      case 2: LAUNCH(2); break;
      case 3: LAUNCH(3); break;
      case 4: LAUNCH(4); break;
      case 5: LAUNCH(5); break;
      case 6: LAUNCH(6); break;
    }
    #undef LAUNCH
    checked(hipGetLastError());
  };
  std::puts("pattern,round,order,layout,milliseconds");
  for (auto pattern = 0u; pattern < 3u; ++pattern) {
    for (auto i = 0u; i < lane_count; ++i) {
      // 0: identical ID within a wave; 1: neighboring per-lane IDs;
      // 2: deterministic hash across the full descriptor range.
      auto index = pattern == 0u ? i >> 5u : i;
      if (pattern == 2u) {
        index ^= index >> 16u; index *= 0x7feb352du;
        index ^= index >> 15u; index *= 0x846ca68bu;
        index ^= index >> 16u;
      }
      indices[i] = index & (descriptor_count - 1u);
    }
    checked(hipMemcpy(device_indices, indices.data(), indices.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    launch(3);
    checked(hipMemcpy(reference.data(), output, lane_count * sizeof(float4), hipMemcpyDeviceToHost));
    for (auto layout = 0u; layout < layout_count; ++layout) {
      if (layout == 3u) { continue; }
      launch(layout);
      checked(hipMemcpy(candidate.data(), output, lane_count * sizeof(float4), hipMemcpyDeviceToHost));
      float max_error = 0.f;
      for (auto i = 0u; i < lane_count; ++i) {
        const auto a = std::array{reference[i].x, reference[i].y, reference[i].z, reference[i].w};
        const auto b = std::array{candidate[i].x, candidate[i].y, candidate[i].z, candidate[i].w};
        for (auto c = 0u; c < 4u; ++c) {
          const auto finite = [](float value) {
            return (std::bit_cast<std::uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
          };
          if (!finite(a[c]) || !finite(b[c]) || std::abs(a[c] - b[c]) > 1e-5f) {
            std::fprintf(stderr, "GPU control mismatch pattern=%u layout=%u lane=%u component=%u expected=%.9g actual=%.9g\n",
                pattern, layout, i, c, a[c], b[c]);
            return 1;
          }
          max_error = std::max(max_error, std::abs(a[c] - b[c]));
        }
      }
      std::fprintf(stderr, "validated pattern=%u layout=%u rgba_lanes=%u max_abs=%.9g\n",
          pattern, layout, lane_count * 4u, max_error);
    }
    for (auto warm = 0u; warm < 2u; ++warm) {
      for (auto layout = 0u; layout < layout_count; ++layout) { launch(layout); }
    }
    checked(hipDeviceSynchronize());
    for (auto round = 0u; round < repetitions; ++round) {
      for (auto order = 0u; order < layout_count; ++order) {
        auto layout = (round + order) % layout_count;
        checked(hipEventRecord(start));
        launch(layout);
        checked(hipEventRecord(finish));
        checked(hipEventSynchronize(finish));
        float milliseconds{};
        checked(hipEventElapsedTime(&milliseconds, start, finish));
        std::printf("%u,%u,%u,%u,%.9g\n", pattern, round, order, layout, milliseconds);
        std::fflush(stdout);
      }
    }
  }
  checked(hipEventDestroy(start)); checked(hipEventDestroy(finish));
  checked(hipFree(output)); checked(hipFree(device_indices));
  checked(hipFree(device_slots)); checked(hipFree(device_images));
  checked(hipFree(device_samplers)); checked(hipFree(device_combined));
  checked(hipFree(device_objects));
  checked(hipFree(device_clones)); checked(hipFree(device_prefixes));
  for (auto object : native_objects) { checked(hipDestroyTextureObject(object)); }
  for (auto array : arrays) { checked(hipFreeArray(array)); }
}
