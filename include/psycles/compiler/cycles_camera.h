#pragma once

#include <psycles/contract/scene.h>

namespace psycles::compiler {

struct CyclesCameraProjection {
  Mat4f raster_to_camera;
  Mat4f world_to_ndc;
  Vec3f dx, dy;
  float aspect{};
  float horizontal_tangent{}, vertical_tangent{};
  float orthographic_vertical_span{};
  float shift_x{}, shift_y{};
};

// Static, non-stereo final-render camera setup, following BlenderSync's
// viewplane/sensor fitting and Camera::update in Cycles 5.2.1. Raster rows use
// Cycles' lower-left convention. All matrix inversions are host setup work.
[[nodiscard]] CyclesCameraProjection make_cycles_camera_projection(
    const contract::CameraDesc &camera, unsigned width, unsigned height);

} // namespace psycles::compiler
