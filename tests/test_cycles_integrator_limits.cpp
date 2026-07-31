#include "../src/luisa/cycles_integrator_limits.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using psycles::luisa_backend::detail::cycles_device_path_step_cap;
using psycles::luisa_backend::detail::cycles_kernel_bounce_limits;
using psycles::luisa_backend::detail::cycles_path_step_limit;
using psycles::luisa_backend::detail::cycles_synced_bounce_limit;
using psycles::luisa_backend::detail::CyclesSceneBounceLimits;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_scene_to_kernel_limit_mapping() {
  for (auto scene_limit = 0u; scene_limit <= 1024u; ++scene_limit) {
    const auto kernel_limit = cycles_synced_bounce_limit(scene_limit);
    require(kernel_limit == scene_limit + 1u,
            "representable scene limit did not gain the "
            "direct-light bounce");

    for (auto depth = 0u; depth <= scene_limit + 2u; ++depth) {
      require((depth >= kernel_limit) == (depth > scene_limit),
              "maximum-bounce termination boundary changed");
      require((depth > kernel_limit) == (depth > scene_limit + 1u),
              "minimum-bounce roulette boundary changed");
    }
  }

  constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
  require(cycles_synced_bounce_limit(maximum) == maximum,
          "unrepresentable plus-one limit did not saturate");
}

void test_structural_mapping() {
  constexpr CyclesSceneBounceLimits scene{.maximum = 8u,
                                          .minimum = 0u,
                                          .maximum_diffuse = 2u,
                                          .maximum_glossy = 3u,
                                          .maximum_transmission = 4u,
                                          .maximum_volume = 6u,
                                          .transparent_minimum = 5u,
                                          .transparent_maximum = 9u};
  constexpr auto kernel = cycles_kernel_bounce_limits(scene);
  static_assert(kernel.maximum == 9u);
  static_assert(kernel.minimum == 1u);
  static_assert(kernel.maximum_diffuse == 3u);
  static_assert(kernel.maximum_glossy == 4u);
  static_assert(kernel.maximum_transmission == 5u);
  static_assert(kernel.maximum_volume == 7u);
  static_assert(kernel.transparent_minimum == 6u);
  static_assert(kernel.transparent_maximum == 9u);
  static_assert(kernel.maximum_path_steps == 19u);
}

void test_path_step_upper_bound() {
  for (auto scene_maximum = 0u; scene_maximum <= 32u; ++scene_maximum) {
    const auto synced_maximum = cycles_synced_bounce_limit(scene_maximum);
    for (auto transparent_maximum = 0u; transparent_maximum <= 32u;
         ++transparent_maximum) {
      const auto required =
          static_cast<std::uint64_t>(scene_maximum) + 1u +
          static_cast<std::uint64_t>(
              transparent_maximum == 0u ? 1u : transparent_maximum) +
          1u;
      require(cycles_path_step_limit(synced_maximum, transparent_maximum) ==
                  required,
              "uncapped path did not reserve every scatter "
              "and its terminal intersection");
    }
  }

  require(cycles_path_step_limit(std::numeric_limits<std::uint32_t>::max(),
                                 std::numeric_limits<std::uint32_t>::max()) ==
              cycles_device_path_step_cap,
          "malformed scene limits escaped the device path cap");

  // Lone Monk's current values: UI max=8, transparent max=8. Nine opaque
  // scatters, eight transparent scatters, and one terminal intersection.
  require(
      cycles_kernel_bounce_limits({.maximum = 8u, .transparent_maximum = 8u})
              .maximum_path_steps == 18u,
      "Lone Monk path-step boundary regressed");
}

} // namespace

int main() {
  test_scene_to_kernel_limit_mapping();
  test_structural_mapping();
  test_path_step_upper_bound();
  std::cout << "All Cycles integrator-limit invariants passed.\n";
  return EXIT_SUCCESS;
}
