#include "path_tracer_curve_scene.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using psycles::contract::CurveGeometryDesc;
using psycles::contract::CurveShape;
using psycles::contract::MaterialId;
using psycles::luisa_backend::detail::build_curve_geometry_upload;
using psycles::luisa_backend::detail::CurveSegmentGpu;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

void expect_segment(const CurveSegmentGpu &segment, std::uint32_t key_before,
                    std::uint32_t key_begin, std::uint32_t key_end,
                    std::uint32_t key_after, std::uint32_t curve_index,
                    std::uint32_t cycles_curve_index,
                    std::uint32_t cycles_segment_index) {
  expect(segment.key_before == key_before, "wrong preceding key");
  expect(segment.key_begin == key_begin, "wrong segment-begin key");
  expect(segment.key_end == key_end, "wrong segment-end key");
  expect(segment.key_after == key_after, "wrong following key");
  expect(segment.curve_index == curve_index, "wrong local curve index");
  expect(segment.cycles_curve_index == cycles_curve_index,
         "wrong global Cycles curve index");
  expect(segment.cycles_segment_index == cycles_segment_index,
         "wrong global Cycles segment index");
}

void test_typed_segments_and_cycles_identity() {
  const CurveGeometryDesc geometry{
      .name = "Two Curves",
      .shape = CurveShape::ribbon,
      .subdivisions = 3u,
      .keys = {{-8.0f, 0.0f, 0.0f, 0.0f},
               {0.0f, 0.0f, 0.0f, 0.2f},
               {0.0f, 0.0f, 0.0f, 0.2f},
               {-8.0f, 0.0f, 0.0f, 0.0f},
               {2.0f, 0.0f, 0.0f, 0.1f},
               {3.0f, 0.0f, 0.0f, 0.1f},
               {4.0f, 0.0f, 0.0f, 0.1f}},
      .curve_first_key = {0u, 4u},
      .material_slots = {MaterialId{41u}, MaterialId{43u}},
      .curve_material_slots = {1u, 0u},
      .default_uv_layer = "RootUV",
      .uv_layers = {{"DetailUV", {{0.6f, 0.7f}, {0.8f, 0.9f}}},
                    {"RootUV", {{0.1f, 0.2f}, {0.3f, 0.4f}}}},
      .color_attributes = {{"RootColor",
                            {{0.1f, 0.2f, 0.3f, 1.0f},
                             {0.4f, 0.5f, 0.6f, 1.0f}}}},
      .intercept = {0.0f, 0.25f, 0.75f, 1.0f, 0.0f, 0.5f, 1.0f},
      .length = {8.0f, 2.0f},
      .random = {}};

  const auto upload = build_curve_geometry_upload(geometry, 17u, 31u);
  expect(upload.keys.size() == 7u, "curve keys were not preserved");
  expect(upload.segments.size() == 5u, "wrong segment count");
  expect(upload.bounds.size() == 5u, "wrong AABB count");
  expect(upload.material_slots.size() == 2u, "wrong material-slot count");
  expect(upload.default_uv_layer == "RootUV", "default UV layer changed");
  expect(upload.uv_layers.size() == 2u, "named UV layers were dropped");
  const auto root_uv = upload.uv_layers.at("RootUV")[1u];
  expect(root_uv.x == 0.3f && root_uv.y == 0.4f,
         "curve-domain UV values changed");
  expect(upload.color_attributes.size() == 1u,
         "curve-domain colors were dropped");
  const auto root_color = upload.color_attributes.at("RootColor")[1u];
  expect(root_color.x == 0.4f && root_color.y == 0.5f && root_color.z == 0.6f &&
             root_color.w == 1.0f,
         "curve-domain color values changed");
  expect(upload.intercept.size() == 7u, "wrong Intercept domain");
  expect(upload.length.size() == 2u, "wrong Length domain");
  expect(upload.random.size() == 2u, "wrong Random domain");

  expect_segment(upload.segments[0u], 0u, 0u, 1u, 2u, 0u, 17u, 31u);
  expect_segment(upload.segments[1u], 0u, 1u, 2u, 3u, 0u, 17u, 32u);
  expect_segment(upload.segments[2u], 1u, 2u, 3u, 3u, 0u, 17u, 33u);
  expect_segment(upload.segments[3u], 4u, 4u, 5u, 6u, 1u, 18u, 34u);
  expect_segment(upload.segments[4u], 4u, 5u, 6u, 6u, 1u, 18u, 35u);

  expect(upload.material_slots[0u] == 1u, "first material slot changed");
  expect(upload.material_slots[1u] == 0u, "second material slot changed");
  expect(upload.intercept[2u] == 0.75f, "Intercept value changed");
  expect(upload.intercept[6u] == 1.0f, "final Intercept value changed");
  expect(upload.length[1u] == 2.0f, "Length value changed");
  expect(upload.random[0u] == 0.0f, "missing Random was not zero");
  expect(upload.random[1u] == 0.0f, "missing Random was not zero");
}

void test_bounds_match_cycles_segment_radius_rule() {
  const CurveGeometryDesc geometry{.name = "Analytic Extremum",
                                   .shape = CurveShape::ribbon,
                                   .keys = {{-8.0f, 0.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 0.0f, 0.2f},
                                            {0.0f, 0.0f, 0.0f, 0.2f},
                                            {-8.0f, 0.0f, 0.0f, 0.0f}},
                                   .curve_first_key = {0u}};
  const auto upload = build_curve_geometry_upload(geometry, 0u, 0u);

  // The middle span is the degenerate quadratic x(u) = 4u(1-u). Cycles'
  // curvebounds only solves derivative roots when the cubic coefficient is
  // non-zero, so this exact source case retains the endpoint interval [0, 0]
  // even though the mathematical center reaches one. Hair::Curve::bounds_grow
  // then expands it by max(radius[key_begin], radius[key_end]) == 0.2; it also
  // deliberately does not spline the radius controls, whose cubic would peak
  // at 0.225 here.
  const auto &bounds = upload.bounds[1u];
  const auto near = [](float lhs, float rhs) noexcept {
    return std::abs(lhs - rhs) <= 1.0e-6f;
  };
  expect(near(bounds.packed_min[0u], -0.2f),
         "wrong Cycles segment AABB minimum");
  expect(near(bounds.packed_max[0u], 0.2f),
         "Cycles degenerate-quadratic bound changed");
  expect(near(bounds.packed_min[1u], -0.2f) &&
             near(bounds.packed_min[2u], -0.2f) &&
             near(bounds.packed_max[1u], 0.2f) &&
             near(bounds.packed_max[2u], 0.2f),
         "Cycles active-endpoint radius expansion changed");
}

} // namespace

int main() {
  try {
    test_typed_segments_and_cycles_identity();
    test_bounds_match_cycles_segment_radius_rule();
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "curve geometry upload test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
