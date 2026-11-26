/* SPDX-FileCopyrightText: 2024-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

inline constexpr float radial_pi = 3.14159265358979323846f;
inline constexpr float radial_tau = 6.28318530717958647692f;

[[nodiscard]] Float sqr(Expr<float> x) noexcept { return x * x; }

[[nodiscard]] Float safe_div(Expr<float> num, Expr<float> den) noexcept {
  return select(0.0f, num / den, den != 0.0f);
}

[[nodiscard]] Float radial_atan(Expr<float> y, Expr<float> x) noexcept {
  return select(0.0f, atan(y / x), x != 0.0f);
}

[[nodiscard]] Float4 calculate_out_variables_irregular_circular(
    bool normalize_r_gon_parameter, Expr<float> r_gon_sides,
    Expr<float> r_gon_roundness, Expr<float2> coord,
    Expr<float> l_coord) noexcept {
  const Float x_axis_A_coord =
      atan2(coord.y, coord.x) + select(0.0f, radial_tau, coord.y < 0.0f);
  const Float segment_divider_A_angle_bisector = radial_pi / r_gon_sides;
  const Float segment_divider_A_next_segment_divider =
      2.0f * segment_divider_A_angle_bisector;
  const Float segment_id =
      floor(x_axis_A_coord / segment_divider_A_next_segment_divider);
  const Float segment_divider_A_coord =
      x_axis_A_coord - segment_id * segment_divider_A_next_segment_divider;
  const Float segment_divider_A_bevel_start =
      segment_divider_A_angle_bisector -
      atan((1.0f - r_gon_roundness) * tan(segment_divider_A_angle_bisector));

  const Float last_angle_bisector_A_x_axis =
      radial_pi - floor(r_gon_sides) * segment_divider_A_angle_bisector;
  const Float last_segment_divider_A_x_axis =
      2.0f * last_angle_bisector_A_x_axis;
  const Float inner_last_bevel_start_A_x_axis =
      last_angle_bisector_A_x_axis -
      atan((1.0f - r_gon_roundness) * tan(last_angle_bisector_A_x_axis));
  const Float l_last_circle_radius =
      r_gon_roundness * tan(last_angle_bisector_A_x_axis) /
      tan(0.5f *
          (segment_divider_A_angle_bisector + last_angle_bisector_A_x_axis));
  const Float inner_last_cos =
      cos(last_angle_bisector_A_x_axis - inner_last_bevel_start_A_x_axis);
  const Float2 last_circle_center = make_float2(
      (cos(inner_last_bevel_start_A_x_axis) / inner_last_cos) -
          l_last_circle_radius * cos(last_angle_bisector_A_x_axis),
      l_last_circle_radius * sin(last_angle_bisector_A_x_axis) -
          (sin(inner_last_bevel_start_A_x_axis) / inner_last_cos));
  const Float2 outer_last_bevel_start =
      last_circle_center +
      l_last_circle_radius *
          make_float2(cos(segment_divider_A_angle_bisector),
                      sin(segment_divider_A_angle_bisector));
  const Float x_axis_A_outer_last_bevel_start =
      radial_atan(outer_last_bevel_start.y, outer_last_bevel_start.x);
  const Float outer_last_bevel_start_A_angle_bisector =
      segment_divider_A_angle_bisector - x_axis_A_outer_last_bevel_start;

  Float4 result = make_float4(0.0f);

  $if((x_axis_A_coord >= x_axis_A_outer_last_bevel_start) &
      (x_axis_A_coord <
       radial_tau - last_segment_divider_A_x_axis -
           x_axis_A_outer_last_bevel_start)) {
    const Float bevel_start_A_angle_bisector =
        segment_divider_A_angle_bisector - segment_divider_A_bevel_start;

    $if(((segment_divider_A_coord >= segment_divider_A_bevel_start) &
         (segment_divider_A_coord <
          segment_divider_A_next_segment_divider -
              segment_divider_A_bevel_start)) |
        (x_axis_A_coord >=
         radial_tau - last_segment_divider_A_x_axis -
             segment_divider_A_bevel_start) |
        (x_axis_A_coord < segment_divider_A_bevel_start)) {
      /* Regular straight part. */
      const Float l_angle_bisector =
          l_coord *
          cos(segment_divider_A_angle_bisector - segment_divider_A_coord);
      Float r_gon_parameter =
          l_angle_bisector *
          tan(abs(segment_divider_A_angle_bisector - segment_divider_A_coord));
      $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
        r_gon_parameter *= -1.0f;
      };
      if (normalize_r_gon_parameter) {
        $if((x_axis_A_coord >=
             radial_tau - last_segment_divider_A_x_axis -
                 segment_divider_A_angle_bisector) |
            (x_axis_A_coord < segment_divider_A_angle_bisector)) {
          const Float effective_roundness =
              1.0f -
              tan(segment_divider_A_angle_bisector -
                  x_axis_A_outer_last_bevel_start) /
                  tan(segment_divider_A_angle_bisector);
          const Float spline_start_outer_last_bevel_start =
              (1.0f - effective_roundness) * x_axis_A_outer_last_bevel_start;
          const Float normalize_based_on_l_angle_bisector =
              l_angle_bisector * tan(outer_last_bevel_start_A_angle_bisector) +
              spline_start_outer_last_bevel_start *
                  (0.5f * l_angle_bisector + 0.5f) +
              effective_roundness * x_axis_A_outer_last_bevel_start;
          r_gon_parameter = safe_div(r_gon_parameter,
                                     normalize_based_on_l_angle_bisector);
        }
        $else {
          const Float spline_start_A_bevel_start =
              (1.0f - r_gon_roundness) * segment_divider_A_bevel_start;
          const Float normalize_based_on_l_angle_bisector =
              l_angle_bisector * tan(bevel_start_A_angle_bisector) +
              spline_start_A_bevel_start * (0.5f * l_angle_bisector + 0.5f) +
              r_gon_roundness * segment_divider_A_bevel_start;
          r_gon_parameter = safe_div(r_gon_parameter,
                                     normalize_based_on_l_angle_bisector);
        };
      }
      result = make_float4(
          l_angle_bisector, r_gon_parameter, 0.0f,
          segment_id * segment_divider_A_next_segment_divider +
              segment_divider_A_angle_bisector);
    }
    $else {
      /* Regular rounded part. */
      const Float nearest_segment_divider_SA_coord =
          segment_divider_A_coord -
          select(0.0f, segment_divider_A_next_segment_divider,
                 segment_divider_A_coord > segment_divider_A_angle_bisector);
      const Float l_circle_center =
          (1.0f - r_gon_roundness) / cos(segment_divider_A_angle_bisector);
      const Float c_proj =
          cos(nearest_segment_divider_SA_coord) * l_circle_center;
      const Float l_coord_R_l_bevel_start =
          c_proj + sqrt(max(sqr(c_proj) + sqr(r_gon_roundness) -
                                sqr(l_circle_center),
                            0.0f));
      const Float l_angle_bisector = safe_div(l_coord, l_coord_R_l_bevel_start);
      const Float spline_start_A_bevel_start =
          (1.0f - r_gon_roundness) * segment_divider_A_bevel_start;
      const Float coord_A_bevel_start =
          segment_divider_A_bevel_start - abs(nearest_segment_divider_SA_coord);
      Float r_gon_parameter = l_coord * sin(bevel_start_A_angle_bisector);
      $if(coord_A_bevel_start < spline_start_A_bevel_start) {
        const Float l_cos = l_coord * cos(bevel_start_A_angle_bisector);
        r_gon_parameter +=
            l_cos * coord_A_bevel_start +
            0.5f * (1.0f - l_cos) *
                safe_div(sqr(coord_A_bevel_start), spline_start_A_bevel_start);
      }
      $else {
        r_gon_parameter +=
            spline_start_A_bevel_start *
                (0.5f * l_coord * cos(bevel_start_A_angle_bisector) - 0.5f) +
            coord_A_bevel_start;
      };
      $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
        r_gon_parameter *= -1.0f;
      };
      if (normalize_r_gon_parameter) {
        const Float normalize_based_on_l_angle_bisector =
            l_angle_bisector * tan(bevel_start_A_angle_bisector) +
            spline_start_A_bevel_start * (0.5f * l_angle_bisector + 0.5f) +
            r_gon_roundness * segment_divider_A_bevel_start;
        const Float normalize_based_on_l_coord =
            l_coord * sin(bevel_start_A_angle_bisector) +
            spline_start_A_bevel_start *
                (0.5f * l_coord * cos(bevel_start_A_angle_bisector) + 0.5f) +
            r_gon_roundness * segment_divider_A_bevel_start;
        r_gon_parameter = safe_div(
            r_gon_parameter,
            lerp(normalize_based_on_l_angle_bisector, normalize_based_on_l_coord,
                 coord_A_bevel_start / segment_divider_A_bevel_start));
      }
      result = make_float4(
          l_angle_bisector, r_gon_parameter, 0.0f,
          segment_id * segment_divider_A_next_segment_divider +
              segment_divider_A_angle_bisector);
    };
  }
  $else {
    const Float inner_last_bevel_start_A_last_angle_bisector =
        last_angle_bisector_A_x_axis - inner_last_bevel_start_A_x_axis;

    $if((x_axis_A_coord >=
         radial_tau - last_segment_divider_A_x_axis +
             inner_last_bevel_start_A_x_axis) &
        (x_axis_A_coord < radial_tau - inner_last_bevel_start_A_x_axis)) {
      /* Irregular straight part. */
      const Float l_angle_bisector_R_l_last_angle_bisector =
          cos(segment_divider_A_angle_bisector) /
          cos(last_angle_bisector_A_x_axis);
      const Float l_last_angle_bisector =
          l_coord * cos(last_angle_bisector_A_x_axis - segment_divider_A_coord);
      const Float l_angle_bisector =
          l_angle_bisector_R_l_last_angle_bisector * l_last_angle_bisector;
      const Float spline_start_A_bevel_start =
          (1.0f - r_gon_roundness) * inner_last_bevel_start_A_x_axis;
      Float r_gon_parameter =
          l_angle_bisector *
          tan(abs(last_angle_bisector_A_x_axis - segment_divider_A_coord));
      $if(segment_divider_A_coord < last_angle_bisector_A_x_axis) {
        r_gon_parameter *= -1.0f;
      };
      if (normalize_r_gon_parameter) {
        const Float normalize_based_on_l_l_angle_bisector =
            l_angle_bisector_R_l_last_angle_bisector *
            (l_last_angle_bisector *
                 tan(inner_last_bevel_start_A_last_angle_bisector) +
             spline_start_A_bevel_start *
                 (0.5f * l_last_angle_bisector + 0.5f) +
             r_gon_roundness * inner_last_bevel_start_A_x_axis);
        r_gon_parameter = safe_div(r_gon_parameter,
                                   normalize_based_on_l_l_angle_bisector);
      }
      result = make_float4(
          l_angle_bisector, r_gon_parameter, 0.0f,
          segment_id * segment_divider_A_next_segment_divider +
              last_angle_bisector_A_x_axis);
    }
    $else {
      /* Irregular rounded part. */
      Float nearest_segment_divider_MSA_coord = atan2(coord.y, coord.x);
      $if((x_axis_A_coord >=
           radial_tau - last_segment_divider_A_x_axis -
               x_axis_A_outer_last_bevel_start) &
          (x_axis_A_coord < radial_tau - last_angle_bisector_A_x_axis)) {
        nearest_segment_divider_MSA_coord += last_segment_divider_A_x_axis;
        nearest_segment_divider_MSA_coord *= -1.0f;
      };
      const Float proj =
          sin(nearest_segment_divider_MSA_coord) * last_circle_center.y +
          cos(nearest_segment_divider_MSA_coord) * last_circle_center.x;
      const Float l_coord_R_l_last_angle_bisector =
          proj + sqrt(max(sqr(proj) + sqr(l_last_circle_radius) -
                              sqr(last_circle_center.x) -
                              sqr(last_circle_center.y),
                          0.0f));
      const Float l_angle_bisector_R_l_last_angle_bisector =
          cos(segment_divider_A_angle_bisector) /
          cos(last_angle_bisector_A_x_axis);
      const Float l_last_angle_bisector =
          safe_div(l_coord, l_coord_R_l_last_angle_bisector);
      const Float l_angle_bisector =
          l_angle_bisector_R_l_last_angle_bisector * l_last_angle_bisector;
      Float r_gon_parameter = 0.0f;
      Float x_axis_A_angle_bisector = 0.0f;

      $if(nearest_segment_divider_MSA_coord < 0.0f) {
        const Float spline_start_A_bevel_start =
            (1.0f - r_gon_roundness) * inner_last_bevel_start_A_x_axis;
        const Float coord_A_bevel_start =
            inner_last_bevel_start_A_x_axis -
            abs(nearest_segment_divider_MSA_coord);
        r_gon_parameter =
            l_angle_bisector_R_l_last_angle_bisector * l_coord *
            sin(inner_last_bevel_start_A_last_angle_bisector);
        $if(coord_A_bevel_start < spline_start_A_bevel_start) {
          const Float l_cos =
              l_coord * cos(inner_last_bevel_start_A_last_angle_bisector);
          r_gon_parameter +=
              l_angle_bisector_R_l_last_angle_bisector *
              (l_cos * coord_A_bevel_start +
               0.5f * (1.0f - l_cos) *
                   safe_div(sqr(coord_A_bevel_start),
                            spline_start_A_bevel_start));
        }
        $else {
          r_gon_parameter +=
              l_angle_bisector_R_l_last_angle_bisector *
              (spline_start_A_bevel_start *
                   (0.5f * l_coord *
                        cos(inner_last_bevel_start_A_last_angle_bisector) -
                    0.5f) +
               coord_A_bevel_start);
        };
        $if(segment_divider_A_coord < last_angle_bisector_A_x_axis) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          const Float normalize_based_on_l_l_angle_bisector =
              l_last_angle_bisector *
                  tan(inner_last_bevel_start_A_last_angle_bisector) +
              spline_start_A_bevel_start *
                  (0.5f * l_last_angle_bisector + 0.5f) +
              r_gon_roundness * inner_last_bevel_start_A_x_axis;
          const Float normalize_based_on_l_coord =
              l_coord * sin(inner_last_bevel_start_A_last_angle_bisector) +
              spline_start_A_bevel_start *
                  (0.5f * l_coord *
                       cos(inner_last_bevel_start_A_last_angle_bisector) +
                   0.5f) +
              r_gon_roundness * inner_last_bevel_start_A_x_axis;
          r_gon_parameter = safe_div(
              r_gon_parameter,
              l_angle_bisector_R_l_last_angle_bisector *
                  lerp(normalize_based_on_l_l_angle_bisector,
                       normalize_based_on_l_coord,
                       safe_div(coord_A_bevel_start,
                                inner_last_bevel_start_A_x_axis)));
        }
        x_axis_A_angle_bisector =
            segment_id * segment_divider_A_next_segment_divider +
            last_angle_bisector_A_x_axis;
      }
      $else {
        const Float effective_roundness =
            1.0f -
            tan(segment_divider_A_angle_bisector -
                x_axis_A_outer_last_bevel_start) /
                tan(segment_divider_A_angle_bisector);
        const Float spline_start_outer_last_bevel_start =
            (1.0f - effective_roundness) * x_axis_A_outer_last_bevel_start;
        const Float coord_A_bevel_start =
            x_axis_A_outer_last_bevel_start -
            abs(nearest_segment_divider_MSA_coord);
        r_gon_parameter =
            l_coord * sin(outer_last_bevel_start_A_angle_bisector);
        $if(coord_A_bevel_start < spline_start_outer_last_bevel_start) {
          const Float l_cos =
              l_coord * cos(outer_last_bevel_start_A_angle_bisector);
          r_gon_parameter +=
              l_cos * coord_A_bevel_start +
              0.5f * (1.0f - l_cos) *
                  safe_div(sqr(coord_A_bevel_start),
                           spline_start_outer_last_bevel_start);
        }
        $else {
          r_gon_parameter +=
              spline_start_outer_last_bevel_start *
                  (0.5f * l_coord *
                       cos(outer_last_bevel_start_A_angle_bisector) -
                   0.5f) +
              coord_A_bevel_start;
        };
        $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          const Float normalize_based_on_l_angle_bisector =
              l_angle_bisector * tan(outer_last_bevel_start_A_angle_bisector) +
              spline_start_outer_last_bevel_start *
                  (0.5f * l_angle_bisector + 0.5f) +
              effective_roundness * x_axis_A_outer_last_bevel_start;
          const Float normalize_based_on_l_coord =
              l_coord * sin(outer_last_bevel_start_A_angle_bisector) +
              spline_start_outer_last_bevel_start *
                  (0.5f * l_coord *
                       cos(outer_last_bevel_start_A_angle_bisector) +
                   0.5f) +
              effective_roundness * x_axis_A_outer_last_bevel_start;
          r_gon_parameter = safe_div(
              r_gon_parameter,
              lerp(normalize_based_on_l_angle_bisector,
                   normalize_based_on_l_coord,
                   safe_div(coord_A_bevel_start,
                            x_axis_A_outer_last_bevel_start)));
        }
        x_axis_A_angle_bisector =
            segment_id * segment_divider_A_next_segment_divider +
            segment_divider_A_angle_bisector;
      };
      result = make_float4(l_angle_bisector, r_gon_parameter, 0.0f,
                           x_axis_A_angle_bisector);
    };
  };
  return result;
}

[[nodiscard]] Float4 calculate_out_variables(bool normalize_r_gon_parameter,
                                             Expr<float> r_gon_sides,
                                             Expr<float> r_gon_roundness,
                                             Expr<float2> coord) noexcept {
  const Float l_coord = sqrt(sqr(coord.x) + sqr(coord.y));
  Float4 out_variables = make_float4(0.0f);

  $if(fract(r_gon_sides) == 0.0f) {
    const Float x_axis_A_coord =
        atan2(coord.y, coord.x) + select(0.0f, radial_tau, coord.y < 0.0f);
    const Float segment_divider_A_angle_bisector = radial_pi / r_gon_sides;
    const Float segment_divider_A_next_segment_divider =
        2.0f * segment_divider_A_angle_bisector;
    const Float segment_id =
        floor(x_axis_A_coord / segment_divider_A_next_segment_divider);
    const Float segment_divider_A_coord =
        x_axis_A_coord - segment_id * segment_divider_A_next_segment_divider;

    $if(r_gon_roundness == 0.0f) {
      const Float l_angle_bisector =
          l_coord *
          cos(segment_divider_A_angle_bisector - segment_divider_A_coord);
      Float r_gon_parameter =
          l_angle_bisector *
          tan(abs(segment_divider_A_angle_bisector - segment_divider_A_coord));
      $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
        r_gon_parameter *= -1.0f;
      };
      if (normalize_r_gon_parameter) {
        $if(r_gon_sides != 2.0f) {
          r_gon_parameter = safe_div(
              r_gon_parameter,
              l_angle_bisector * tan(segment_divider_A_angle_bisector));
        };
      }
      const Float max_unit_parameter = select(
          0.0f, tan(segment_divider_A_angle_bisector), r_gon_sides != 2.0f);
      out_variables = make_float4(
          l_angle_bisector, r_gon_parameter, max_unit_parameter,
          segment_id * segment_divider_A_next_segment_divider +
              segment_divider_A_angle_bisector);
    }
    $elif(r_gon_roundness == 1.0f) {
      Float r_gon_parameter =
          abs(segment_divider_A_angle_bisector - segment_divider_A_coord);
      $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
        r_gon_parameter *= -1.0f;
      };
      if (normalize_r_gon_parameter) {
        r_gon_parameter /= segment_divider_A_angle_bisector;
      }
      out_variables = make_float4(
          l_coord, r_gon_parameter, segment_divider_A_angle_bisector,
          segment_id * segment_divider_A_next_segment_divider +
              segment_divider_A_angle_bisector);
    }
    $else {
      const Float segment_divider_A_bevel_start =
          segment_divider_A_angle_bisector -
          atan((1.0f - r_gon_roundness) *
               tan(segment_divider_A_angle_bisector));
      const Float bevel_start_A_angle_bisector =
          segment_divider_A_angle_bisector - segment_divider_A_bevel_start;

      $if((segment_divider_A_coord >=
           segment_divider_A_next_segment_divider -
               segment_divider_A_bevel_start) |
          (segment_divider_A_coord < segment_divider_A_bevel_start)) {
        const Float nearest_segment_divider_SA_coord =
            segment_divider_A_coord -
            select(0.0f, segment_divider_A_next_segment_divider,
                   segment_divider_A_coord > segment_divider_A_angle_bisector);
        const Float l_circle_center =
            (1.0f - r_gon_roundness) / cos(segment_divider_A_angle_bisector);
        const Float c_proj =
            cos(nearest_segment_divider_SA_coord) * l_circle_center;
        const Float l_coord_R_l_bevel_start =
            c_proj + sqrt(max(sqr(c_proj) + sqr(r_gon_roundness) -
                                  sqr(l_circle_center),
                              0.0f));
        const Float l_angle_bisector =
            safe_div(l_coord, l_coord_R_l_bevel_start);
        const Float spline_start_A_bevel_start =
            (1.0f - r_gon_roundness) * segment_divider_A_bevel_start;
        const Float coord_A_bevel_start =
            segment_divider_A_bevel_start -
            abs(nearest_segment_divider_SA_coord);
        Float r_gon_parameter = l_coord * sin(bevel_start_A_angle_bisector);
        $if(coord_A_bevel_start < spline_start_A_bevel_start) {
          const Float l_cos = l_coord * cos(bevel_start_A_angle_bisector);
          r_gon_parameter +=
              l_cos * coord_A_bevel_start +
              0.5f * (1.0f - l_cos) *
                  safe_div(sqr(coord_A_bevel_start),
                           spline_start_A_bevel_start);
        }
        $else {
          r_gon_parameter +=
              spline_start_A_bevel_start *
                  (0.5f * l_coord * cos(bevel_start_A_angle_bisector) - 0.5f) +
              coord_A_bevel_start;
        };
        $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          const Float normalize_based_on_l_angle_bisector =
              l_angle_bisector * tan(bevel_start_A_angle_bisector) +
              spline_start_A_bevel_start * (0.5f * l_angle_bisector + 0.5f) +
              r_gon_roundness * segment_divider_A_bevel_start;
          const Float normalize_based_on_l_coord =
              l_coord * sin(bevel_start_A_angle_bisector) +
              spline_start_A_bevel_start *
                  (0.5f * l_coord * cos(bevel_start_A_angle_bisector) + 0.5f) +
              r_gon_roundness * segment_divider_A_bevel_start;
          r_gon_parameter = safe_div(
              r_gon_parameter,
              lerp(normalize_based_on_l_angle_bisector,
                   normalize_based_on_l_coord,
                   safe_div(coord_A_bevel_start,
                            segment_divider_A_bevel_start)));
        }
        const Float max_unit_parameter =
            tan(bevel_start_A_angle_bisector) + spline_start_A_bevel_start +
            r_gon_roundness * segment_divider_A_bevel_start;
        out_variables = make_float4(
            l_angle_bisector, r_gon_parameter, max_unit_parameter,
            segment_id * segment_divider_A_next_segment_divider +
                segment_divider_A_angle_bisector);
      }
      $else {
        const Float l_angle_bisector =
            l_coord *
            cos(segment_divider_A_angle_bisector - segment_divider_A_coord);
        const Float spline_start_A_bevel_start =
            (1.0f - r_gon_roundness) * segment_divider_A_bevel_start;
        Float r_gon_parameter =
            l_angle_bisector *
            tan(abs(segment_divider_A_angle_bisector -
                    segment_divider_A_coord));
        $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          const Float normalize_based_on_l_angle_bisector =
              l_angle_bisector * tan(bevel_start_A_angle_bisector) +
              spline_start_A_bevel_start * (0.5f * l_angle_bisector + 0.5f) +
              r_gon_roundness * segment_divider_A_bevel_start;
          r_gon_parameter =
              safe_div(r_gon_parameter, normalize_based_on_l_angle_bisector);
        }
        const Float max_unit_parameter =
            tan(bevel_start_A_angle_bisector) + spline_start_A_bevel_start +
            r_gon_roundness * segment_divider_A_bevel_start;
        out_variables = make_float4(
            l_angle_bisector, r_gon_parameter, max_unit_parameter,
            segment_id * segment_divider_A_next_segment_divider +
                segment_divider_A_angle_bisector);
      };
    };
  }
  $else {
    $if(r_gon_roundness == 0.0f) {
      const Float x_axis_A_coord =
          atan2(coord.y, coord.x) + select(0.0f, radial_tau, coord.y < 0.0f);
      const Float segment_divider_A_angle_bisector = radial_pi / r_gon_sides;
      const Float segment_divider_A_next_segment_divider =
          2.0f * segment_divider_A_angle_bisector;
      const Float segment_id =
          floor(x_axis_A_coord / segment_divider_A_next_segment_divider);
      const Float segment_divider_A_coord =
          x_axis_A_coord - segment_id * segment_divider_A_next_segment_divider;
      const Float last_angle_bisector_A_x_axis =
          radial_pi - floor(r_gon_sides) * segment_divider_A_angle_bisector;
      const Float last_segment_divider_A_x_axis =
          2.0f * last_angle_bisector_A_x_axis;

      $if(x_axis_A_coord < radial_tau - last_segment_divider_A_x_axis) {
        const Float l_angle_bisector =
            l_coord *
            cos(segment_divider_A_angle_bisector - segment_divider_A_coord);
        Float r_gon_parameter =
            l_angle_bisector *
            tan(abs(segment_divider_A_angle_bisector -
                    segment_divider_A_coord));
        $if(segment_divider_A_coord < segment_divider_A_angle_bisector) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          r_gon_parameter = safe_div(
              r_gon_parameter,
              l_angle_bisector * tan(segment_divider_A_angle_bisector));
        }
        out_variables = make_float4(
            l_angle_bisector, r_gon_parameter,
            tan(segment_divider_A_angle_bisector),
            segment_id * segment_divider_A_next_segment_divider +
                segment_divider_A_angle_bisector);
      }
      $else {
        const Float l_angle_bisector_R_l_last_angle_bisector =
            cos(segment_divider_A_angle_bisector) /
            cos(last_angle_bisector_A_x_axis);
        const Float l_last_angle_bisector =
            l_coord * cos(last_angle_bisector_A_x_axis - segment_divider_A_coord);
        const Float l_angle_bisector =
            l_angle_bisector_R_l_last_angle_bisector * l_last_angle_bisector;
        Float r_gon_parameter =
            l_angle_bisector *
            tan(abs(last_angle_bisector_A_x_axis - segment_divider_A_coord));
        $if(segment_divider_A_coord < last_angle_bisector_A_x_axis) {
          r_gon_parameter *= -1.0f;
        };
        if (normalize_r_gon_parameter) {
          r_gon_parameter = safe_div(
              r_gon_parameter,
              l_angle_bisector * tan(last_angle_bisector_A_x_axis));
        }
        out_variables = make_float4(
            l_angle_bisector, r_gon_parameter,
            tan(last_angle_bisector_A_x_axis),
            segment_id * segment_divider_A_next_segment_divider +
                last_angle_bisector_A_x_axis);
      };
    }
    $else {
      out_variables = calculate_out_variables_irregular_circular(
          normalize_r_gon_parameter, r_gon_sides, r_gon_roundness, coord,
          l_coord);
    };
  };

  $if((coord.x == 0.0f) & (coord.y == 0.0f)) { out_variables.y = 0.0f; };

  if (normalize_r_gon_parameter) {
    out_variables.y = 0.5f * out_variables.y + 0.5f;
  } else {
    out_variables.x -= 1.0f;
  }

  return out_variables;
}

[[nodiscard]] Float calculate_out_segment_id(Expr<float> r_gon_sides,
                                             Expr<float2> coord) noexcept {
  return floor(r_gon_sides * ((atan2(coord.y, coord.x) / radial_tau) +
                              select(0.0f, 1.0f, coord.y < 0.0f)));
}

} // namespace

void node_radial_tiling(Cursor &cursor, Stack &stack) noexcept {
  const auto vector_x = cursor.word();
  const auto vector_y = cursor.word();
  const auto vector_z = cursor.word();
  const auto sides_bits = cursor.word();
  const auto roundness_bits = cursor.word();
  const auto packed0 = cursor.word();
  const auto normalize_word = cursor.byte(packed0, 0u);
  const auto segment_coordinates_offset = cursor.byte(packed0, 1u);
  const auto segment_id_offset = cursor.byte(packed0, 2u);
  const auto max_unit_parameter_offset = cursor.byte(packed0, 3u);
  const auto packed1 = cursor.word();
  const auto x_axis_A_angle_bisector_offset = cursor.byte(packed1, 0u);

  const Float3 coord3 =
      stack_load_input_float3(stack, vector_x, vector_y, vector_z);
  const Float2 coord = make_float2(coord3.x, coord3.y);
  const Float r_gon_sides =
      max(stack_load_input_float(stack, sides_bits), 2.0f);
  const Float r_gon_roundness =
      clamp(stack_load_input_float(stack, roundness_bits), 0.0f, 1.0f);

  const auto run_variables = [&](bool normalize_param) noexcept {
    const Float4 out_variables = calculate_out_variables(
        normalize_param, r_gon_sides, r_gon_roundness, coord);
    stack_store_float3_if_valid(
        stack, segment_coordinates_offset,
        make_float3(out_variables.y, out_variables.x, 0.0f));
    stack_store_float_if_valid(stack, max_unit_parameter_offset,
                               out_variables.z);
    stack_store_float_if_valid(stack, x_axis_A_angle_bisector_offset,
                               out_variables.w);
  };

  if (const auto const_norm = try_const_uint(normalize_word)) {
    run_variables(*const_norm != 0u);
  } else {
    $if(normalize_word != 0u) { run_variables(true); }
    $else { run_variables(false); };
  }

  $if(segment_id_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, segment_id_offset,
                      calculate_out_segment_id(r_gon_sides, coord));
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
