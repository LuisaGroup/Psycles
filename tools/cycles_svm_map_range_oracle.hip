// Original Cycles interpreter; all expected radiance comes from GPU execution.
#include "../tests/cycles_svm_map_range_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#if defined(PSYCLES_MAP_RANGE_FLOOR_ULP)
// Sensitivity runs only: perturb the input to floor in map_range.h by one
// representable float. Pre-include its dependencies to leave every other node
// and helper untouched; the central oracle never enables this macro.
#include "kernel/svm/node_types.h"
#include "kernel/svm/util.h"
static_assert(PSYCLES_MAP_RANGE_FLOOR_ULP == -1 || PSYCLES_MAP_RANGE_FLOOR_ULP == 1);
ccl_device float map_range_oracle_adjacent_floor(float x) {
  return floorf(nextafterf(x, PSYCLES_MAP_RANGE_FLOOR_ULP > 0 ? 0x1.fffffep127f : -0x1.fffffep127f));
}
#define floorf map_range_oracle_adjacent_floor
#include "kernel/svm/map_range.h"
#undef floorf
#endif
#include "kernel/svm/svm.h"
// clang-format on

#include <cstdio>
#include <cstdlib>

using namespace psycles::test_support;
__device__ MapRangePoint points[map_range_points.size()];
void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}

__global__ void probe(float4 *out) {
  const auto i = threadIdx.x;
  if (i >= map_range_points.size()) {
    return;
  }
  const auto point = points[i];
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->shader = 0;
  sd->object = OBJECT_NONE;
  sd->P = make_float3(point.x, point.y, point.z);
  const auto initialize = [&] {
    sd->flag = 0;
    sd->closure_emission_background = make_float3(-1.0f, -2.0f, -3.0f);
  };
  initialize();
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_SURFACE, SHADER_TYPE_SURFACE>(
      nullptr, IntegratorState{0}, sd, nullptr, PATH_RAY_VISIBILITY_CAMERA,
      PATH_RAY_FLAG_NONE);
  out[i * 3u] = make_float4(sd->closure_emission_background, float(sd->flag));
  initialize();
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_VOLUME, SHADER_TYPE_VOLUME>(
      nullptr, IntegratorState{0}, sd, nullptr, PATH_RAY_VISIBILITY_CAMERA,
      PATH_RAY_FLAG_NONE);
  out[i * 3u + 1u] =
      make_float4(sd->closure_emission_background, float(sd->flag));
  initialize();
  svm_eval_nodes<KERNEL_FEATURE_NODE_MASK_DISPLACEMENT,
                 SHADER_TYPE_DISPLACEMENT>(nullptr, IntegratorState{0}, sd,
                                           nullptr, PATH_RAY_VISIBILITY_CAMERA,
                                           PATH_RAY_FLAG_NONE);
  out[i * 3u + 2u] =
      make_float4(sd->closure_emission_background, float(sd->flag));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  const auto images = read_map_range_images(argv[1]);
  unsigned *words{};
  float4 *out{};
  constexpr auto count = map_range_points.size();
  check(hipMalloc(&words, 256u * sizeof(unsigned)));
  check(hipMalloc(&out, count * 3u * sizeof(float4)));
  KernelParamsHIP params{};
  params.svm_nodes = words;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  check(hipMemcpyToSymbol(HIP_SYMBOL(points), map_range_points.data(),
                          sizeof(map_range_points)));
  for (auto model = 0u; model < images.size(); ++model) {
    const auto &image = images[model].words;
    check(hipMemcpy(words, image.data(), image.size() * sizeof(unsigned),
                    hipMemcpyHostToDevice));
    probe<<<1u, 32u>>>(out);
    check(hipGetLastError());
    std::array<float4, count * 3u> values{};
    check(hipMemcpy(values.data(), out, sizeof(values), hipMemcpyDeviceToHost));
    for (auto i = 0u; i < count; ++i) {
      for (auto domain = 0u; domain < 3u; ++domain) {
        const auto value = values[i * 3u + domain];
        std::printf("%u %u %u %.9g %.9g %.9g %u\n", model, i, domain, value.x,
                    value.y, value.z, unsigned(value.w));
      }
    }
  }
  check(hipFree(out));
  check(hipFree(words));
}
