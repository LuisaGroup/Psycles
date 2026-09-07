// Run the committed, externally compiled Cycles word images through the
// original Cycles 5.2.1 HIP interpreter. No closure formula is reimplemented.
// hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 \
//   -ffast-math -I /path/to/blender/intern/cycles this_file -o /tmp/oracle
// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#include "kernel/svm/svm.h"
// clang-format on

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr unsigned case_count = 48u;
struct Output {
  unsigned flag, count, left, type;
  float values[10];
};

void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

__global__ void probe(Output *output) {
  const auto i = threadIdx.x;
  if (i >= case_count) { return; }
  constexpr unsigned visibility[]{PATH_RAY_VISIBILITY_CAMERA,
                                   PATH_RAY_VISIBILITY_DIFFUSE,
                                   PATH_RAY_VISIBILITY_GLOSSY,
                                   PATH_RAY_VISIBILITY_SHADOW};
  constexpr unsigned capacities[]{0u, 1u, 64u};
  ShaderData sd{};
  sd.shader = i / 12u;
  sd.object = OBJECT_NONE;
  sd.prim = PRIM_NONE;
  sd.type = i < 24u ? PRIMITIVE_TRIANGLE : PRIMITIVE_VOLUME;
  sd.flag = i < 24u ? 0u : SD_IS_VOLUME_SHADER_EVAL;
  sd.N = sd.Ng = sd.wi = make_float3(0.0f, 0.0f, 1.0f);
  sd.num_closure_left = capacities[i % 3u];
  const auto ray_visibility = visibility[(i / 3u) % 4u];
  if (i < 24u) {
    svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
        nullptr, IntegratorState{0}, &sd, nullptr, ray_visibility, PATH_RAY_FLAG_NONE);
  } else {
    svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_VOLUME, SHADER_TYPE_VOLUME>(
        nullptr, IntegratorState{0}, &sd, nullptr, ray_visibility, PATH_RAY_FLAG_NONE);
  }
  auto &out = output[i];
  out = {};
  out.flag = sd.flag;
  out.count = sd.num_closure;
  out.left = sd.num_closure_left;
  const auto put = [&](unsigned offset, float3 value) {
    out.values[offset] = value.x;
    out.values[offset + 1u] = value.y;
    out.values[offset + 2u] = value.z;
  };
  put(0u, sd.closure_transparent_extinction);
  put(3u, sd.closure_emission_background);
  if (sd.num_closure != 0) {
    out.type = sd.closure[0].type;
    put(6u, sd.closure[0].weight);
    out.values[9] = sd.closure[0].sample_weight;
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) { return 1; }
  std::ifstream input{argv[1]};
  std::vector<unsigned> words(16u);
  constexpr std::array names{"SurfaceAdd", "SurfaceMix", "VolumeAdd", "VolumeMix"};
  for (unsigned shader = 0u; shader < names.size(); ++shader) {
    std::string name;
    unsigned count;
    if (!(input >> name >> std::dec >> count) || name != names[shader] || count < 4u) {
      return 2;
    }
    std::vector<unsigned> local(count);
    for (auto &word : local) { if (!(input >> std::hex >> word)) { return 2; } }
    const auto start = static_cast<unsigned>(words.size());
    words[shader * 4u] = local[0];
    for (auto j = 1u; j < 4u; ++j) { words[shader * 4u + j] = local[j] - 4u + start; }
    words.insert(words.end(), local.begin() + 4u, local.end());
  }
  unsigned *device_words{};
  Output *device_output{};
  check(hipMalloc(&device_words, words.size() * sizeof(unsigned)));
  check(hipMalloc(&device_output, case_count * sizeof(Output)));
  check(hipMemcpy(device_words, words.data(), words.size() * sizeof(unsigned), hipMemcpyHostToDevice));
  KernelParamsHIP params{};
  params.svm_nodes = device_words;
  params.data.film.is_rec709 = true;
  params.data.integrator.caustics_reflective = true;
  params.data.integrator.caustics_refractive = true;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  probe<<<1u, 64u>>>(device_output);
  check(hipGetLastError());
  std::array<Output, case_count> results;
  check(hipMemcpy(results.data(), device_output, sizeof(results), hipMemcpyDeviceToHost));
  for (auto i = 0u; i < case_count; ++i) {
    const auto &row = results[i];
    std::printf("%u %u %u %u %u", i, row.flag, row.count, row.left, row.type);
    for (float value : row.values) { std::printf(" %.9g", value); }
    std::printf("\n");
  }
  check(hipFree(device_output));
  check(hipFree(device_words));
}
