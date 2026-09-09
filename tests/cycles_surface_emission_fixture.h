#pragma once

// Inputs only; output values and native film write counts come from the
// original Cycles HIP program, never a CPU light/film implementation.
#include <array>

namespace psycles::test_support::surface_emission_fixture {
inline constexpr unsigned routes = 5;
inline constexpr unsigned kinds = 4;
inline constexpr unsigned cases = 2 * 2 * routes * kinds;
inline constexpr unsigned oracle_cases = cases + 1;
inline constexpr unsigned film_lanes = 40;
inline constexpr float direct_limit = 2.0f;
inline constexpr float indirect_limit = 1.5f;
struct Input {
  unsigned emission, exit, route, kind;
};
inline constexpr auto inputs = [] {
  std::array<Input, oracle_cases> result{};
  unsigned i = 0;
  for (unsigned e = 0; e < 2; ++e) {
    for (unsigned x = 0; x < 2; ++x) {
      for (unsigned r = 0; r < routes; ++r) {
        for (unsigned k = 0; k < kinds; ++k) {
          result[i++] = {e, x, r, k};
        }
      }
    }
  }
  // Unity camera throughput for the full-renderer integration gate.
  result[cases] = {1, 0, 0, 4};
  return result;
}();
} // namespace psycles::test_support::surface_emission_fixture
