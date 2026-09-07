#pragma once

#include <array>

namespace psycles::test_support {
struct VolumeBoundaryInput {
  unsigned bounds;
  unsigned flag;
  float probability;
  float distance;
};
inline constexpr auto volume_boundary_inputs = [] {
  std::array<VolumeBoundaryInput, 36> cases{};
  unsigned i = 0;
  for (const auto bounds : {0u, 1023u, 1024u}) {
    for (const auto flag : {0u, 1u << 10u}) {
      for (const auto probability : {0.0f, 0.25f, 1.0f}) {
        for (const auto distance : {0.0f, 0.875f}) {
          cases[i++] = {bounds, flag, probability, distance};
        }
      }
    }
  }
  return cases;
}();
} // namespace psycles::test_support
