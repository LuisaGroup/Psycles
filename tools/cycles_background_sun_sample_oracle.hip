// Cycles 5.2.1's actual background mixture sampler, executed on HIP.
#include "../tests/cycles_background_sun_sample_fixture.h"

// clang-format off
#include "kernel/device/hip/compat.h"
#include "kernel/device/hip/globals.h"
#include "kernel/device/gpu/image.h"
#include "kernel/geom/shader_data.h"
#include "kernel/light/light.h"
// clang-format on

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {
using namespace psycles::test_support;
constexpr auto count = unsigned(background_sun_randoms.size());
void check(hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(e));
    std::exit(1);
  }
}
__global__ void probe(const float2 *randoms, float4 *out) {
  const auto i = threadIdx.x;
  if (i >= count) {
    return;
  }
  auto random = randoms[i];
  // Keep the selected mixture component a sun, including near its RNG edge.
  if (kernel_data.background.map_weight != 0.0f) {
    random.x *= 0.8f;
  }
  float pdf;
  const auto direction =
      background_light_sample(nullptr, zero_float3(), random, &pdf);
  out[i] = make_float4(direction, pdf);
}
} // namespace

int main() {
  std::array<float2, count> randoms{};
  for (auto i = 0u; i < count; ++i) {
    randoms[i] =
        make_float2(background_sun_randoms[i][0], background_sun_randoms[i][1]);
  }
  const std::array<float2, 6u> conditional{
      make_float2(1.0f, 0.0f), make_float2(1.0f, 0.5f),
      make_float2(1.0f, 1.0f), make_float2(1.0f, 0.0f),
      make_float2(1.0f, 0.5f), make_float2(1.0f, 1.0f)};
  const std::array<float2, 3u> marginal{make_float2(1.0f, 0.0f),
                                        make_float2(1.0f, 0.5f),
                                        make_float2(1.0f, 1.0f)};
  const auto upload = [](const auto &values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    T *device{};
    check(hipMalloc(&device, values.size() * sizeof(T)));
    check(hipMemcpy(device, values.data(), values.size() * sizeof(T),
                    hipMemcpyHostToDevice));
    return device;
  };
  const auto device_randoms = upload(randoms);
  KernelParamsHIP params{};
  params.light_background_conditional_cdf = upload(conditional);
  params.light_background_marginal_cdf = upload(marginal);
  params.data.background.map_res_x = params.data.background.map_res_y = 2;
  params.data.background.sun =
      make_float4(background_sun_axis_radius[0], background_sun_axis_radius[1],
                  background_sun_axis_radius[2], background_sun_axis_radius[3]);
  params.data.background.sun_weight = 4.0f;
  float4 *out{};
  check(hipMalloc(&out, count * sizeof(float4)));
  std::array<float4, count> values{};
  for (auto mixture = 0u; mixture < 2u; ++mixture) {
    params.data.background.map_weight = float(mixture);
    check(
        hipMemcpyToSymbol(HIP_SYMBOL(kernel_params), &params, sizeof(params)));
    probe<<<1, 32>>>(device_randoms, out);
    check(hipGetLastError());
    check(hipMemcpy(values.data(), out, sizeof(values), hipMemcpyDeviceToHost));
    for (auto i = 0u; i < count; ++i) {
      const auto v = values[i];
      std::printf("%u %u %.9g %.9g %.9g %.9g\n", mixture, i, v.x, v.y, v.z,
                  v.w);
    }
  }
  check(hipDeviceReset());
}
