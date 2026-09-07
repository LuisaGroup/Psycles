// Original Cycles 5.2.1 host camera projection. No CPU shading or rendering.
// Compile against the oracle's libcycles_util and guardedalloc libraries;
// see docs/validation/2026-09-07/camera-projection/README.md.
#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "util/boundbox.h"
#include "util/projection.h"
#include "../tests/cycles_camera_projection_fixture.h"

#include <cstdio>

int main() {
  using namespace ccl;
  unsigned index{};
  for (const auto &input : psycles::test_support::camera_projection_inputs) {
    const float xratio = float(input.width) * input.pixel_aspect.x;
    const float yratio = float(input.height) * input.pixel_aspect.y;
    const bool horizontal = input.fit == 1u || (input.fit == 0u && xratio > yratio);
    const float sensor = input.fit == 2u ? input.sensor_height : input.sensor_width;
    float aspectratio = horizontal ? xratio / yratio : yratio / xratio;
    auto aspect = horizontal ? make_float2(aspectratio, 1) : make_float2(1, aspectratio);
    if (input.orthographic) {
      aspect *= input.ortho_scale / (aspectratio * 2.0f);
      aspectratio = input.ortho_scale / 2.0f;
    }
    auto viewplane = BoundBox2D(aspect).offset(
        2.0f * aspectratio * make_float2(input.shift.x, input.shift.y));
    // Full-frame final render: viewport_camera_border = [0,1]^2.
    BoundBox2D border;
    border.left = border.bottom = 0.0f;
    border.right = border.top = 1.0f;
    const auto fulltoborder = transform_from_viewplane(border);
    const auto ndctoraster = transform_scale(input.width, input.height, 1) *
                            transform_inverse(fulltoborder);
    const auto screentondc = fulltoborder * transform_from_viewplane(viewplane);
    const auto rastertoscreen = transform_inverse(ndctoraster * screentondc);
    const float fov = 2.0f * atanf((0.5f * sensor) / input.lens / aspectratio);
    const auto projection = input.orthographic
        ? projection_orthographic(input.near_clip, input.far_clip)
        : projection_perspective(fov, input.near_clip, input.far_clip);
    const auto rtc = projection_inverse(projection) * rastertoscreen;
    const auto &e = input.transform.elements;
    const Transform ctw{make_float4(e[0], e[4], -e[8], e[12]),
                         make_float4(e[1], e[5], -e[9], e[13]),
                         make_float4(e[2], e[6], -e[10], e[14])};
    const auto worldtondc = screentondc * (projection * transform_inverse(ctw));
    auto dx = input.orthographic
        ? transform_perspective_direction(&rtc, make_float3(1, 0, 0))
        : transform_perspective(&rtc, make_float3(1, 0, 0)) -
          transform_perspective(&rtc, make_float3(0, 0, 0));
    auto dy = input.orthographic
        ? transform_perspective_direction(&rtc, make_float3(0, 1, 0))
        : transform_perspective(&rtc, make_float3(0, 1, 0)) -
          transform_perspective(&rtc, make_float3(0, 0, 0));
    dx = transform_direction(&ctw, dx);
    dy = transform_direction(&ctw, dy);
    std::printf("%u", index++);
    for (const auto &r : {rtc.x, rtc.y, rtc.z, rtc.w,
                          make_float4(dx, 0), make_float4(dy, 0),
                          worldtondc.x, worldtondc.y, worldtondc.z, worldtondc.w}) {
      std::printf(" %.9g %.9g %.9g %.9g", r.x, r.y, r.z, r.w);
    }
    std::puts("");
  }
}
