/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
// Camera projection equations follow Blender Cycles 5.2.1.
#include <psycles/compiler/cycles_camera.h>
#include <psycles/compiler/cycles_transform.h>

#include <algorithm>
#include <cmath>

namespace psycles::compiler {
namespace {

float &element(Mat4f &m, unsigned r, unsigned c) { return m.elements[c * 4u + r]; }
float element(const Mat4f &m, unsigned r, unsigned c) { return m.elements[c * 4u + r]; }

Mat4f multiply(const Mat4f &a, const Mat4f &b) {
  Mat4f result;
  for (unsigned r = 0u; r < 4u; ++r) {
    for (unsigned c = 0u; c < 4u; ++c) {
      element(result, r, c) =
          (element(a, r, 0) * element(b, 0, c) + element(a, r, 1) * element(b, 1, c)) +
          (element(a, r, 2) * element(b, 2, c) + element(a, r, 3) * element(b, 3, c));
    }
  }
  return result;
}

Mat4f scale(float x, float y, float z) {
  Mat4f result;
  result.elements[0] = x; result.elements[5] = y; result.elements[10] = z;
  return result;
}

// Cycles util/projection_inverse.h: pivoted forward elimination and backward
// substitution. This host-only projection inverse is distinct from the
// existing affine Transform inverse; substituting one for the other changes
// the rastertocamera representation and near-plane homogeneous division.
Mat4f inverse_projection(Mat4f matrix) {
  /* SPDX-License-Identifier: BSD-3-Clause
   * Adapted from Cycles util/projection_inverse.h and its upstream:
   * Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
   * Digital Ltd. LLC. All rights reserved.
   * See docs/licenses/cycles-camera-projection-BSD-3-Clause.txt. */
  Mat4f result;
  for (unsigned i = 0u; i < 4u; ++i) {
    auto pivot = i;
    auto magnitude = std::abs(element(matrix, i, i));
    for (auto j = i + 1u; j < 4u; ++j) {
      const auto candidate = std::abs(element(matrix, j, i));
      if (candidate > magnitude) { pivot = j; magnitude = candidate; }
    }
    if (magnitude == 0.0f) { return {}; }
    if (pivot != i) {
      for (unsigned j = 0u; j < 4u; ++j) {
        std::swap(element(matrix, i, j), element(matrix, pivot, j));
        std::swap(element(result, i, j), element(result, pivot, j));
      }
    }
    for (auto j = i + 1u; j < 4u; ++j) {
      const float f = element(matrix, j, i) / element(matrix, i, i);
      for (unsigned k = 0u; k < 4u; ++k) {
        element(matrix, j, k) -= f * element(matrix, i, k);
        element(result, j, k) -= f * element(result, i, k);
      }
    }
  }
  for (int i = 3; i >= 0; --i) {
    const float diagonal = element(matrix, i, i);
    if (diagonal == 0.0f) { return {}; }
    for (unsigned j = 0u; j < 4u; ++j) {
      element(matrix, i, j) /= diagonal;
      element(result, i, j) /= diagonal;
    }
    for (int j = 0; j < i; ++j) {
      const float f = element(matrix, j, i);
      for (unsigned k = 0u; k < 4u; ++k) {
        element(matrix, j, k) -= f * element(matrix, i, k);
        element(result, j, k) -= f * element(result, i, k);
      }
    }
  }
  return result;
}

Vec3f project(const Mat4f &m, float x, float y) {
  const auto dot = [&](unsigned r) {
    return (element(m, r, 0) * x + element(m, r, 1) * y) + element(m, r, 3);
  };
  const auto w = dot(3u);
  const auto reciprocal = w != 0.0f ? 1.0f / w : 0.0f;
  return {dot(0u) * reciprocal, dot(1u) * reciprocal, dot(2u) * reciprocal};
}

Vec3f world_direction(const Mat4f &m, Vec3f v) {
  // CameraDesc retains Blender -Z; KernelCamera uses +Z.
  const auto &e = m.elements;
  return {e[0] * v.x + e[4] * v.y - e[8] * v.z,
          e[1] * v.x + e[5] * v.y - e[9] * v.z,
          e[2] * v.x + e[6] * v.y - e[10] * v.z};
}
} // namespace

CyclesCameraProjection make_cycles_camera_projection(
    const contract::CameraDesc &camera, unsigned width, unsigned height) {
  width = std::max(width, 1u); height = std::max(height, 1u);
  const float xratio = float(width) * camera.pixel_aspect.x;
  const float yratio = float(height) * camera.pixel_aspect.y;
  const bool horizontal = camera.sensor_fit == contract::CameraSensorFit::horizontal ||
      (camera.sensor_fit == contract::CameraSensorFit::automatic && xratio > yratio);
  float ratio = horizontal ? xratio / yratio : yratio / xratio;
  float x = horizontal ? ratio : 1.0f, y = horizontal ? 1.0f : ratio;
  const bool orthographic = camera.projection == contract::CameraProjection::orthographic;
  const bool vertical_sensor = camera.sensor_fit == contract::CameraSensorFit::vertical;
  const float fov = camera.sensor
      ? 2.0f * std::atan((0.5f * (vertical_sensor ? camera.sensor->height_mm
                                                              : camera.sensor->width_mm)) /
                         camera.sensor->lens_mm / ratio)
      : 2.0f * std::atan(std::tan(0.5f * (vertical_sensor ? camera.field_of_view
                                                       : camera.horizontal_field_of_view)) / ratio);
  CyclesCameraProjection result;
  result.aspect = xratio / yratio;
  result.horizontal_tangent = std::tan(0.5f * fov) * x;
  result.vertical_tangent = std::tan(0.5f * fov) * y;
  result.shift_x = camera.lens_shift_x * ratio / x;
  result.shift_y = camera.lens_shift_y * ratio / y;
  result.orthographic_vertical_span = horizontal
      ? camera.orthographic_scale / result.aspect : camera.orthographic_scale;
  if (orthographic) {
    const float factor = camera.orthographic_scale / (ratio * 2.0f);
    x *= factor; y *= factor; ratio = camera.orthographic_scale / 2.0f;
  }
  const float shift_x = 2.0f * (ratio * camera.lens_shift_x);
  const float shift_y = 2.0f * (ratio * camera.lens_shift_y);
  const float left = -x + shift_x, right = x + shift_x;
  const float bottom = -y + shift_y, top = y + shift_y;
  Mat4f translation;
  translation.elements[12] = -left; translation.elements[13] = -bottom;
  const auto screentondc = multiply(scale(1.0f / (right - left), 1.0f / (top - bottom), 1), translation);
  const auto rastertoscreen = cycles_inverse_affine_transform(
      multiply(scale(float(width), float(height), 1), screentondc));
  const float n = camera.near_clip, f = camera.far_clip;
  Mat4f projection;
  if (orthographic) {
    projection = scale(1, 1, 1.0f / (f - n));
  } else {
    projection.elements = {1, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, f / (f - n), 1, 0, 0, -f * n / (f - n), 0};
    const float inverse_angle = 1.0f / std::tan(0.5f * fov);
    projection = multiply(scale(inverse_angle, inverse_angle, 1), projection);
  }
  result.raster_to_camera = multiply(inverse_projection(projection), rastertoscreen);
  auto camera_to_world = camera.transform;
  for (unsigned r = 0u; r < 3u; ++r) { camera_to_world.elements[8u + r] *= -1.0f; }
  result.world_to_ndc = multiply(screentondc, multiply(
      projection, cycles_inverse_affine_transform(camera_to_world)));
  const auto &m = result.raster_to_camera;
  Vec3f dx{m.elements[0], m.elements[1], m.elements[2]};
  Vec3f dy{m.elements[4], m.elements[5], m.elements[6]};
  if (!orthographic) {
    const auto center = project(m, 0, 0), px = project(m, 1, 0), py = project(m, 0, 1);
    dx = {px.x - center.x, px.y - center.y, px.z - center.z};
    dy = {py.x - center.x, py.y - center.y, py.z - center.z};
  }
  result.dx = world_direction(camera.transform, dx);
  result.dy = world_direction(camera.transform, dy);
  return result;
}

} // namespace psycles::compiler
