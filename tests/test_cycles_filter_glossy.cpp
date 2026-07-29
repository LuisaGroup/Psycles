#include "../src/luisa/cycles_filter_glossy.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using psycles::luisa_backend::detail::cycles_filter_glossy_device_scale;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_scene_to_device_mapping() {
  require(cycles_filter_glossy_device_scale(0.0f) ==
              std::numeric_limits<float>::max(),
          "zero scene filter did not map to Cycles' disabled sentinel");
  require(cycles_filter_glossy_device_scale(-1.0f) ==
              std::numeric_limits<float>::max(),
          "negative scene filter did not map to the disabled sentinel");
  require(cycles_filter_glossy_device_scale(0.5f) == 2.0f,
          "scene filter reciprocal changed");
  require(cycles_filter_glossy_device_scale(2.0f) == 0.5f,
          "scene filter reciprocal changed");
  require(cycles_filter_glossy_device_scale(0.25f) >
              cycles_filter_glossy_device_scale(0.5f),
          "device threshold did not decrease as scene filtering increased");
}

} // namespace

int main() {
  test_scene_to_device_mapping();
  std::cout << "All Cycles Filter Glossy setting invariants passed.\n";
  return EXIT_SUCCESS;
}
