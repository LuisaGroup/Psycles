#pragma once

#include <array>

namespace psycles::test_support::light_endpoint {

// Authored inputs only. Expected intersections come from original Cycles GPU
// lights_intersect_impl<true>, not a host intersection/transport
// implementation.
struct Input {
  unsigned type; // 0 point, 1 spot, 2 area
  bool sphere;
  bool ellipse;
  std::array<float, 3> origin;
  std::array<float, 3> direction;
  float radius{0.5f};
  float minimum{0.0f};
};

inline constexpr std::array inputs{
    Input{0, true, false, {0, 0, -3}, {0, 0, 1}},
    Input{0, false, false, {0, 0, -3}, {0, 0, 1}},
    Input{1, true, false, {0, 0, -3}, {0, 0, 1}},
    Input{1, true, false, {3, 0, 0}, {-1, 0, 0}}, // Outside spot emission cone.
    Input{1, false, false, {3, 0, 0}, {-1, 0, 0}},
    Input{2, false, false, {0, 0, -3}, {0, 0, 1}},
    Input{2, false, false, {3, 0, -3}, {-0.6f, 0, 0.8f}}, // Outside spread.
    Input{2, false, true, {3, 0, -3}, {-0.6f, 0, 0.8f}},
    Input{2, false, false, {0, 0, 3}, {0, 0, -1}}, // Back-facing area.
    Input{0, true, false, {0, 0, -3}, {0, 0, 1}, 0.0f},
    Input{1, true, false, {0.1f, 0, 0}, {1, 0, 0}}, // One-sided spot rejection.
    Input{0, true, false, {0, 0, -3}, {0, 0, 1}, 0.5f, 4.0f},
};

// Camera; diffuse; excluded diffuse; excluded camera; diffuse without MIS.
inline constexpr unsigned modes = 5u;
inline constexpr unsigned count = unsigned(inputs.size()) * modes;
inline constexpr float maximum = 100.0f;
inline constexpr float spot_angle = 0.6f;
inline constexpr float spot_smooth = 0.5f;
inline constexpr float spread = 0.4f;
inline constexpr float length = 2.0f;

} // namespace psycles::test_support::light_endpoint
