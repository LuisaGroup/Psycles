#pragma once

#include <array>

namespace psycles::test_support::lamp_routing {

// Authored inputs, not a local path-state reference implementation.
inline constexpr std::array<unsigned, 4> transparent_limits{1, 2, 3, 5};
inline constexpr unsigned light_count = 3;
inline constexpr unsigned first_light_object = 3;
inline constexpr unsigned initial_object = 17;
inline constexpr unsigned initial_primitive = 29;
inline constexpr unsigned rng_offset = 72;
inline constexpr unsigned modes = 2; // Camera and diffuse reflection.
inline constexpr unsigned cases = transparent_limits.size() * modes;
inline constexpr float maximum = 100.0f;
inline constexpr float radius = 0.5f;
inline constexpr float axis = 0.7071067811865475f;
inline constexpr float spot_angle = 0.2f;

} // namespace psycles::test_support::lamp_routing
