#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdlib>

void checked(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}
void describe(const char *name, const void *pointer) {
  hipPointerAttribute_t hip{};
  checked(hipPointerGetAttributes(&hip, pointer));
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  auto status = hsa_amd_pointer_info(pointer, &info, nullptr, nullptr, nullptr);
  if (status != HSA_STATUS_SUCCESS || info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
    std::fprintf(stderr, "HSA pointer information unavailable: %u\n", status);
    std::exit(1);
  }
  char owner[64]{};
  hsa_device_type_t type{};
  if (hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_NAME, owner) != HSA_STATUS_SUCCESS ||
      hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS) {
    std::exit(1);
  }
  std::printf("%s hip_type=%u managed=%u hip_flags=%u hsa_type=%u owner=%s owner_device=%u hsa_flags=%u fine=%u coarse=%u pool_allocation_bytes=%zu\n",
      name, static_cast<unsigned>(hip.type), hip.isManaged, hip.allocationFlags,
      info.type, owner, type, info.global_flags,
      !!(info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED),
      !!(info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED),
      info.sizeInBytes);
}
int main() {
  hipArray_t array{};
  auto channel = hipCreateChannelDesc<uchar4>();
  checked(hipMallocArray(&array, &channel, 64u, 64u));
  hipResourceDesc resource{};
  resource.resType = hipResourceTypeArray;
  resource.res.array.array = array;
  hipTextureDesc sampler{};
  sampler.filterMode = hipFilterModeLinear;
  sampler.readMode = hipReadModeNormalizedFloat;
  sampler.normalizedCoords = 1;
  hipTextureObject_t object{};
  checked(hipCreateTextureObject(&object, &resource, &sampler, nullptr));
  void *device_descriptor{};
  checked(hipMalloc(&device_descriptor, 32u));
  describe("native_hip_texture", object);
  describe("hip_malloc_descriptor", device_descriptor);
  checked(hipFree(device_descriptor));
  checked(hipDestroyTextureObject(object));
  checked(hipFreeArray(array));
}
