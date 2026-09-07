// Invoke the external Cycles 5.2.1 interpreter on the externally dumped image.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -I /path/to/blender/intern/cycles \
//   tools/cycles_svm_volume_emission_oracle.hip -o /tmp/volume-emission-oracle
#include "../tests/cycles_svm_volume_emission_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
// clang-format on

#include <cstdio>
#include <cstdlib>

using namespace psycles::test_support;

void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}

__global__ void probe(float4 *output) {
  const auto i = threadIdx.x;
  if (i >= volume_emission_case_count) { return; }
  const auto object = i % 4u == 0u ? ~0u : i % 4u - 1u;
  const auto camera = (i & 4u) == 0u;
  const auto accumulate = (i & 8u) != 0u;
  const auto initialize = [&] (ShaderData *sd) {
    sd->object = object;
    sd->shader = 0u;
    sd->flag = SD_IS_VOLUME_SHADER_EVAL | (accumulate ? SD_EMISSION : 0u);
    sd->closure_emission_background = make_float3(0.125f, -0.25f, 0.5f);
  };
  const auto visibility = static_cast<PathRayVisibility>(
      camera ? PATH_RAY_VISIBILITY_CAMERA : 0u);
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  initialize(sd);
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_VOLUME, SHADER_TYPE_VOLUME>(
      nullptr, IntegratorState{0}, sd, nullptr, visibility, PATH_RAY_FLAG_NONE);
  output[i * 3u] = make_float4(sd->closure_emission_background, float(sd->flag));
  initialize(sd);
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
      nullptr, IntegratorState{0}, sd, nullptr, visibility, PATH_RAY_FLAG_NONE);
  output[i * 3u + 1u] = make_float4(sd->closure_emission_background, float(sd->flag));
  initialize(sd);
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_DISPLACEMENT, SHADER_TYPE_DISPLACEMENT>(
      nullptr, IntegratorState{0}, sd, nullptr, visibility, PATH_RAY_FLAG_NONE);
  output[i * 3u + 2u] = make_float4(sd->closure_emission_background, float(sd->flag));
}

int main(int argc, char **argv) {
  if (argc != 2) { return 1; }
  const auto words = read_volume_emission_words(argv[1]);
  KernelParamsHIP parameters{};
  unsigned *device_words{};
  check(hipMalloc(&device_words, words.size() * sizeof(unsigned)));
  check(hipMemcpy(device_words, words.data(), words.size() * sizeof(unsigned),
                  hipMemcpyHostToDevice));
  parameters.svm_nodes = device_words;
  std::array<KernelObject, volume_emission_densities.size()> objects{};
  for (auto i = 0u; i < objects.size(); ++i) {
    objects[i].volume_density = volume_emission_densities[i];
  }
  KernelObject *device_objects{};
  check(hipMalloc(&device_objects, sizeof(objects)));
  check(hipMemcpy(device_objects, objects.data(), sizeof(objects), hipMemcpyHostToDevice));
  parameters.objects = device_objects;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &parameters, sizeof(parameters)));
  float4 *device_output{};
  check(hipMalloc(&device_output, volume_emission_case_count * 3u * sizeof(float4)));
  probe<<<1u, 32u>>>(device_output);
  check(hipGetLastError());
  std::array<float4, volume_emission_case_count * 3u> output{};
  check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < volume_emission_case_count; ++i) {
    for (auto domain = 0u; domain < 3u; ++domain) {
      const auto value = output[i * 3u + domain];
      std::printf("%u %u %.9g %.9g %.9g %u\n", i, domain,
                  value.x, value.y, value.z, unsigned(value.w));
    }
  }
  check(hipFree(device_output));
  check(hipFree(device_objects));
  check(hipFree(device_words));
}
