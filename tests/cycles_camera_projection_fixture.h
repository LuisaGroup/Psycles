#pragma once

#include <psycles/core/math.h>

#include <array>
#include <cstdint>

namespace psycles::test_support {

// Inputs only. The host fixture invokes Cycles' original projection utilities;
// the device fixture invokes its original camera_sample_* functions on HIP.
struct CameraProjectionInput {
  unsigned width{1440u}, height{1080u};
  unsigned fit{}; // 0 = AUTO, 1 = HORIZONTAL, 2 = VERTICAL.
  bool orthographic{};
  float lens{25.5f}, sensor_width{36.0f}, sensor_height{24.0f};
  Vec2f pixel_aspect{1.0f, 1.0f};
  Vec2f shift{0.0f, 0.07099999487400055f};
  float near_clip{1.119999885559082f}, far_clip{50.0f};
  float ortho_scale{4.0f};
  float aperture{(25.5f * 1.0e-3f) / (2.0f * 11.199999809265137f)};
  float focus{2.636000633239746f};
  unsigned blades{};
  float rotation{}, aperture_ratio{1.0f};
  // Blender convention, looking down local -Z. This is the evaluated,
  // scale-free camera transform from Lone Monk, frame 4.
  Mat4f transform{{1, 0, 0, 0, 0, -4.371138828673793e-8f, 1, 0,
                  0, -1, -4.371138828673793e-8f, 0,
                  20, 7.512925148010254f, 0.8999999761581421f, 1}};
};

inline constexpr auto camera_projection_inputs = [] {
  std::array<CameraProjectionInput, 24u> inputs{};
  for (unsigned i = 0; i < inputs.size(); ++i) {
    auto &c = inputs[i];
    c.fit = (i / 2u) % 3u;
    c.orthographic = i >= 12u;
    if (i % 2u) { c.width = 900u; c.height = 1600u; }
    if (i % 6u >= 2u) { c.shift = {-0.23f, 0.31f}; }
    if (i % 6u >= 4u) { c.pixel_aspect = {1.6f, 0.8f}; }
    if (i % 4u == 1u) { c.aperture = 0.0f; }
    if (i % 4u == 2u) { c.aperture = 0.4f; c.aperture_ratio = 1.8f; }
    c.blades = std::array{0u, 1u, 2u, 5u}[i % 4u];
    c.rotation = 0.13f;
    if (i % 6u == 5u) {
      c.transform = Mat4f{{0.6f, 0.8f, 0, 0, -0.8f, 0.6f, 0, 0,
                          0, 0, 1, 0, -3, 2, 7, 1}};
      c.near_clip = 0.001f;
      c.far_clip = 1000.0f;
    }
  }
  return inputs;
}();

inline constexpr std::array<Vec2f, 6u> camera_raster_fractions{{
    {0.0f, 0.0f}, {0.5f, 1.0f / 3.0f}, {0.999f, 0.999f},
    {0.173f, 0.791f}, {0.501f, 0.499f}, {0.999f, 0.001f}}};
inline constexpr Vec2f camera_lens_sample{
    0.37103718519210815f, 0.4975980520248413f};

} // namespace psycles::test_support
