#pragma once

#include <psycles/core/math.h>

#include <array>

namespace psycles::test_support {

struct CameraTransformInput {
  Mat4f camera_to_world;
  Vec3f position;
};

// Inputs only. Expected transforms and node outputs come from the original
// Cycles HIP implementation, never a second camera evaluator in the test.
inline constexpr std::array<CameraTransformInput, 8u> camera_transform_cases{{
    {{}, {1.0f, -2.0f, 6.0f}},
    {{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, -4, 2, 1}},
     {1.0f, -2.0f, 6.0f}},
    {{{0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 3, -4, 2, 1}},
     {1.0f, -2.0f, 6.0f}},
    {{{2, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 3, 0, -1, 2, 4, 1}},
     {1.0f, -2.0f, 6.0f}},
    {{{-2, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 3, 0, -1, 2, 4, 1}},
     {1.0f, -2.0f, 6.0f}},
    {{{1, 0.25f, 0, 0, -0.5f, 2, 0.125f, 0,
       0.25f, -0.75f, 1.5f, 0, 0.3f, -0.4f, 0.2f, 1}},
     {-0.7f, 1.3f, 2.1f}},
    // Singular affine transforms exercise Cycles' diagonal regularization.
    {{{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, -1, 2, 1}},
     {2.0f, 3.0f, 4.0f}},
    {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 2, 1}},
     {2.0f, 3.0f, 4.0f}},
}};

} // namespace psycles::test_support
