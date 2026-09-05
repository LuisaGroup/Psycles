#include "cycles_background_sun_sample_fixture.h"
#include "path_kernel_environment_light.h"

#include <luisa/luisa-compute.h>
#include <psycles/luisa/background_sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace {
using namespace luisa::compute;
using namespace psycles::test_support;
namespace bg = psycles::luisa_backend::background_sampling;
namespace detail = psycles::luisa_backend::detail;
constexpr auto count = unsigned(background_sun_randoms.size());

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr std::array<luisa::float2, 6u> conditional{luisa::float2{1.0f, 0.0f},
                                                      {1.0f, 0.5f},
                                                      {1.0f, 1.0f},
                                                      {1.0f, 0.0f},
                                                      {1.0f, 0.5f},
                                                      {1.0f, 1.0f}};
  constexpr std::array<luisa::float2, 3u> marginal{
      luisa::float2{1.0f, 0.0f}, {1.0f, 0.5f}, {1.0f, 1.0f}};
  std::array<luisa::float2, count> randoms{};
  for (auto i = 0u; i < count; ++i) {
    randoms[i] = luisa::make_float2(background_sun_randoms[i][0],
                                    background_sun_randoms[i][1]);
  }
  const auto upload = [&](const auto &values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    auto buffer = device.create_buffer<T>(values.size());
    stream << buffer.copy_from(values.data()) << synchronize();
    return buffer;
  };
  auto scene = std::make_shared<detail::LuisaSceneData>();
  scene->background_conditional_cdf = upload(conditional);
  scene->background_marginal_cdf = upload(marginal);
  scene->background_map_width = scene->background_map_height = 2u;
  scene->background_guided_sun_weight = 4.0f;
  scene->background_guided_sun_axis = luisa::make_float3(
      background_sun_axis_radius[0], background_sun_axis_radius[1],
      background_sun_axis_radius[2]);
  scene->background_guided_sun_radius = background_sun_axis_radius[3];
  // Enter the production portal-capable route, but no portal is eligible.
  // Cycles then uses exactly the same sun/map mixture as the direct helper.
  scene->background_portal_weight = 1.0f;
  scene->light_buffer = device.create_buffer<detail::LightGpu>(1u);
  const auto environment = detail::make_environment_light_component();
  auto rng = upload(randoms);
  auto output = device.create_buffer<luisa::float4>(2u * count);
  bool passed = true;
  std::ifstream oracle{PSYCLES_BACKGROUND_SUN_ORACLE};
  for (auto mixture = 0u; mixture < 2u; ++mixture) {
    scene->background_map_weight = float(mixture);
    Kernel1D kernel = [mixture, scene, environment](
                          BufferFloat2 cdf, BufferFloat2 marginal,
                          BufferFloat2 rng, BufferFloat4 out) noexcept {
      const auto i = dispatch_x();
      auto random = rng.read(i);
      if (mixture != 0u) {
        random.x *= 0.8f;
      }
      const auto sample =
          bg::sample(cdf, marginal, 2u, 2u, float(mixture), 4.0f,
                     make_float3(background_sun_axis_radius[0],
                                 background_sun_axis_radius[1],
                                 background_sun_axis_radius[2]),
                     background_sun_axis_radius[3], random);
      out.write(i, make_float4(sample.direction, sample.pdf));
      const auto portal_route = environment->from_position(
          scene, make_float3(0.0f), random, 1.0f, 0u, 0u);
      out.write(count + i,
                make_float4(portal_route.direction, portal_route.pdf));
    };
    auto shader = device.compile(kernel);
    std::array<luisa::float4, 2u * count> values{};
    stream << shader(scene->background_conditional_cdf,
                     scene->background_marginal_cdf, rng, output)
                  .dispatch(count)
           << output.copy_to(values.data()) << synchronize();
    for (auto i = 0u; i < count; ++i) {
      unsigned expected_mixture{}, expected_index{};
      luisa::float4 expected{};
      if (!(oracle >> expected_mixture >> expected_index >> expected.x >>
            expected.y >> expected.z >> expected.w) ||
          expected_mixture != mixture || expected_index != i) {
        std::cerr << "Malformed Cycles background sun oracle\n";
        return false;
      }
      for (auto route = 0u; route < 2u; ++route) {
        for (auto j = 0u; j < 4u; ++j) {
          const auto actual = values[route * count + i][j];
          if (!std::isfinite(actual) ||
              std::abs(actual - expected[j]) >
                  5.0e-5f * std::max(std::abs(expected[j]), 1.0e-5f)) {
            std::cerr << "Sun mixture=" << mixture << " route=" << route
                      << " case=" << i << " lane=" << j << " got=" << actual
                      << " Cycles=" << expected[j] << '\n';
            passed = false;
          }
        }
      }
    }
  }
  oracle >> std::ws;
  return passed && oracle.eof();
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
