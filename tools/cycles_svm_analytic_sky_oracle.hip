// Original Cycles 5.2.1 interpreter on independently dumped analytic-sky
// images.
#include "../tests/cycles_svm_analytic_sky_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/integrator/state_util.h"
#if defined(PSYCLES_SKY_COSINE_ULP)
// Optional rounding-sensitivity run, never the central oracle. Pre-include
// every sky.h dependency so that the macro below affects only that header's
// sky_angle_between call, not direction_to_spherical or other math helpers.
#include "kernel/image.h"
#include "kernel/svm/node_types.h"
#include "kernel/svm/types.h"
#include "kernel/svm/util.h"
#include "kernel/util/colorspace.h"
#include "kernel/util/differential.h"
#include "util/color.h"
#include "util/defines.h"
static_assert(PSYCLES_SKY_COSINE_ULP == -1 || PSYCLES_SKY_COSINE_ULP == 1);
ccl_device float sky_oracle_adjacent_cosine_acos(float x) {
  return safe_acosf(nextafterf(x, PSYCLES_SKY_COSINE_ULP > 0 ? 0x1.fffffep127f : -0x1.fffffep127f));
}
#define safe_acosf sky_oracle_adjacent_cosine_acos
#include "kernel/svm/sky.h"
#undef safe_acosf
#endif
#include "kernel/svm/svm.h"
// clang-format on

#include <cstdio>
#include <cstdlib>

using namespace psycles::test_support;
__device__ SkyDirection directions[analytic_sky_directions.size()];
void check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(1);
  }
}

__global__ void probe(float4 *out) {
  const auto i = threadIdx.x;
  if (i >= analytic_sky_directions.size()) {
    return;
  }
  const auto direction = directions[i];
  ShaderDataTinyStorage storage{};
  auto *sd = AS_SHADER_DATA(&storage);
  sd->shader = 0;
  sd->object = OBJECT_NONE;
  sd->P = make_float3(direction.x, direction.y, direction.z);
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
  if (argc != 3) {
    return 1;
  }
  KernelParamsHIP params{};
  const auto row = [](SkyDirection v) {
    return make_float4(v.x, v.y, v.z, 0.0f);
  };
  params.data.film.xyz_to_r = row(analytic_sky_xyz_to_rgb[0]);
  params.data.film.xyz_to_g = row(analytic_sky_xyz_to_rgb[1]);
  params.data.film.xyz_to_b = row(analytic_sky_xyz_to_rgb[2]);
  check(hipMemcpyToSymbol(HIP_SYMBOL(directions),
                          analytic_sky_directions.data(),
                          sizeof(analytic_sky_directions)));
  unsigned *words{};
  float4 *out{};
  constexpr auto count = analytic_sky_directions.size();
  check(hipMalloc(&words, 52u * sizeof(unsigned)));
  check(hipMalloc(&out, count * 3u * sizeof(float4)));
  params.svm_nodes = words;
  check(hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
  for (auto model = 0u; model < 2u; ++model) {
    const auto image = read_analytic_sky_words(argv[model + 1u]);
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
