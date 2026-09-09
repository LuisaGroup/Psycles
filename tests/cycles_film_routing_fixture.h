#pragma once

// Boundary inputs only. Shader-state and film expectations are captured from
// original Cycles GPU code by the observer, not calculated on the host.
#include <array>

namespace psycles::test_support::film_routing_fixture {
enum Operation : unsigned { surface_nee, volume_nee, emission, background };
inline constexpr unsigned operations = 4, routes = 5, depths = 3, kinds = 4;
inline constexpr unsigned cases = operations * routes * depths * kinds;
inline constexpr unsigned film_lanes = 52;
using RGB = std::array<float, 3>;
struct Input {
  unsigned operation, route, bounce, kind;
  RGB contribution, diffuse_weight, glossy_weight;
  RGB bsdf_diffuse, bsdf_glossy, bsdf_sum;
};
inline constexpr auto inputs = [] {
  std::array<Input, cases> result{};
  unsigned i = 0;
  for (unsigned op = 0; op < operations; ++op) {
    for (unsigned route = 0; route < routes; ++route) {
      for (unsigned depth : {0u, 1u, 3u}) {
        for (unsigned kind = 0; kind < kinds; ++kind) {
          auto &v = result[i++];
          v = {op, route, depth, kind,
               {2, 4, 1}, {.25f, .125f, .5f}, {.125f, .25f, .25f},
               {1, 1, 2}, {1, 2, 1}, {4, 8, 4}};
          if (kind == 1) {
            v.contribution = {3, -4, 1};
            v.bsdf_sum = {0, 8, 4};
            v.bsdf_diffuse = {0, 1, 2};
            v.bsdf_glossy = {0, 2, 1};
          } else if (kind == 2) {
            v.contribution = {0, 0, 0};
          } else if (kind == 3) {
            v.bsdf_diffuse = {1e-25f, 1e-25f, 2e-25f};
            v.bsdf_glossy = {1e-25f, 2e-25f, 1e-25f};
            v.bsdf_sum = {4e-25f, 8e-25f, 4e-25f};
          }
        }
      }
    }
  }
  return result;
}();
} // namespace psycles::test_support::film_routing_fixture
