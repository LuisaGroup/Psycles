#pragma once

#include "cycles_camera_projection_fixture.h"
#include <psycles/contract/scene.h>

#include <cmath>

namespace psycles::test_support {

inline contract::CameraDesc camera_projection_description(const CameraProjectionInput &c) {
  contract::CameraDesc result;
  result.projection = c.orthographic ? contract::CameraProjection::orthographic
                                     : contract::CameraProjection::perspective;
  result.transform = c.transform;
  result.field_of_view = 2.0f * std::atan(c.sensor_height * 0.5f / c.lens);
  result.horizontal_field_of_view = 2.0f * std::atan(c.sensor_width * 0.5f / c.lens);
  result.sensor_fit = c.fit == 0u ? contract::CameraSensorFit::automatic
      : c.fit == 1u ? contract::CameraSensorFit::horizontal : contract::CameraSensorFit::vertical;
  result.sensor = contract::CameraSensor{c.lens, c.sensor_width, c.sensor_height};
  result.pixel_aspect = c.pixel_aspect;
  result.orthographic_scale = c.ortho_scale;
  result.lens_shift_x = c.shift.x; result.lens_shift_y = c.shift.y;
  result.near_clip = c.near_clip; result.far_clip = c.far_clip;
  result.aperture_radius = c.aperture; result.focal_distance = c.focus;
  result.aperture_blades = c.blades; result.aperture_rotation = c.rotation;
  result.aperture_ratio = c.aperture_ratio;
  return result;
}
} // namespace psycles::test_support
