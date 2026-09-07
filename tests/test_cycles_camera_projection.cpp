#include "cycles_camera_projection_test_support.h"
#include <psycles/compiler/cycles_camera.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

int main() {
  using namespace psycles::test_support;
  std::ifstream oracle{PSYCLES_CAMERA_PROJECTION_ORACLE};
  bool passed = bool(oracle);
  for (unsigned i = 0u; i < camera_projection_inputs.size(); ++i) {
    const auto &input = camera_projection_inputs[i];
    auto camera = camera_projection_description(input);
    unsigned scenario{};
    std::array<float, 40u> expected{};
    oracle >> scenario;
    for (auto &v : expected) { oracle >> v; }
    if (!oracle || scenario != i) { return 1; }
    // Both physical and legacy angle-only bundles must preserve the original
    // camera semantics. Roundoff from angle reconstruction is tolerated.
    for (unsigned legacy = 0u; legacy < 2u; ++legacy) {
      if (legacy) { camera.sensor.reset(); }
      const auto p = psycles::compiler::make_cycles_camera_projection(camera, input.width, input.height);
      std::array<float, 40u> actual{};
      for (unsigned r = 0u; r < 4u; ++r) {
        for (unsigned c = 0u; c < 4u; ++c) {
          actual[r * 4u + c] = p.raster_to_camera.elements[c * 4u + r];
          actual[24u + r * 4u + c] = p.world_to_ndc.elements[c * 4u + r];
        }
      }
      actual[16] = p.dx.x; actual[17] = p.dx.y; actual[18] = p.dx.z;
      actual[20] = p.dy.x; actual[21] = p.dy.y; actual[22] = p.dy.z;
      for (unsigned j = 0u; j < actual.size(); ++j) {
        if (!std::isfinite(actual[j]) ||
            std::abs(actual[j] - expected[j]) > 5.0e-6f * std::max(1.0f, std::abs(expected[j]))) {
          std::cerr << "camera=" << i << " legacy=" << legacy << " lane=" << j
                    << " actual=" << actual[j] << " Cycles=" << expected[j] << '\n';
          passed = false;
        }
      }
    }
  }
  std::string trailing;
  if (oracle >> trailing) { return 1; }
  return passed ? 0 : 1;
}
