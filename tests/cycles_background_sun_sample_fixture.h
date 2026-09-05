#pragma once

#include <array>

namespace psycles::test_support {

// Inputs only. These near-edge RNG pairs exercise the distinction between a
// sampling PDF and a separately evaluated PDF of the rounded direction.
inline constexpr std::array<float, 4u> background_sun_axis_radius{
    0.3009074926376343f, -0.5211871266365051f, 0.7986355423927307f,
    0.008726646192371845f};
inline constexpr float below_one = 0x1.fffffep-1f;
inline constexpr std::array<std::array<float, 2u>, 18u> background_sun_randoms{
    {{0.0f, 0.0f},
     {0.0f, 0.25f},
     {0.0f, 0.5f},
     {0.0f, 0.75f},
     {0.0f, below_one},
     {0.25f, 0.0f},
     {0.5f, 0.0f},
     {0.75f, 0.0f},
     {below_one, 0.0f},
     {below_one, 0.25f},
     {below_one, 0.5f},
     {below_one, 0.75f},
     {below_one, below_one},
     {0.25f, below_one},
     {0.5f, below_one},
     {0.75f, below_one},
     {0.1f, 0.2f},
     {0.5f, 0.5f}}};

} // namespace psycles::test_support
