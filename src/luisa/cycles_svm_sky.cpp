/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/spherical_geometry.h>

#include <array>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

constexpr auto pi = 3.14159265358979323846f;
constexpr auto half_pi = 0.5f * pi;
constexpr auto inverse_two_pi = 0.5f / pi;
constexpr auto two_over_pi = 2.0f / pi;

[[nodiscard]] Float2 direction_to_spherical(Expr<luisa::float3> direction) {
  return make_float2(
      acos(clamp(direction.z, -1.0f, 1.0f)),
      spherical_geometry::canonical_direction_azimuth(direction));
}

[[nodiscard]] Float3 spherical_to_direction(Float theta, Float phi) {
  const auto sin_theta = sin(theta);
  return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos(theta));
}

[[nodiscard]] std::array<Float, 9u> sky_configuration(Cursor &cursor) noexcept {
  // Braced initialization sequences the word loads. All 27 coefficient words
  // belong to the typed payload even when Preetham uses only five per channel.
  return {cursor.floating(), cursor.floating(), cursor.floating(),
          cursor.floating(), cursor.floating(), cursor.floating(),
          cursor.floating(), cursor.floating(), cursor.floating()};
}

[[nodiscard]] Float sky_perez(const std::array<Float, 9u> &c, Expr<float> theta,
                              Expr<float> gamma) noexcept {
  const auto ct = cos(theta);
  const auto cg = cos(gamma);
  return (1.0f + c[0] * exp(c[1] / ct)) *
         (1.0f + c[2] * exp(c[3] * gamma) + c[4] * cg * cg);
}

[[nodiscard]] Float sky_radiance_internal(const std::array<Float, 9u> &c,
                                          Expr<float> theta,
                                          Expr<float> gamma) noexcept {
  const auto ct = cos(theta);
  const auto cg = cos(gamma);
  const auto exp_m = exp(c[4] * gamma);
  const auto ray_m = cg * cg;
  const auto mie_m =
      (1.0f + ray_m) / pow(1.0f + c[8] * c[8] - 2.0f * c[8] * cg, 1.5f);
  const auto zenith = sqrt(ct);
  return (1.0f + c[0] * exp(c[1] / (ct + 0.01f))) *
         (c[2] + c[3] * exp_m + c[5] * ray_m + c[6] * mie_m + c[7] * zenith);
}

[[nodiscard]] Float3
sky_radiance_analytic(Cursor &cursor, const KernelGlobals &kg,
                      Expr<std::uint32_t> type,
                      Expr<luisa::float3> direction) noexcept {
  const auto sunphi = cursor.floating();
  const auto suntheta = cursor.floating();
  const auto radiance_x = cursor.floating();
  const auto radiance_y = cursor.floating();
  const auto radiance_z = cursor.floating();
  const auto config_x = sky_configuration(cursor);
  const auto config_y = sky_configuration(cursor);
  const auto config_z = sky_configuration(cursor);
  const auto spherical = direction_to_spherical(direction);
  const auto phi = -spherical.y + half_pi;
  // Cycles computes the angular separation before clamping the query to the
  // horizon. Moving this clamp above gamma changes below-horizon radiance.
  const auto cospsi = sin(spherical.x) * sin(suntheta) * cos(sunphi - phi) +
                      cos(spherical.x) * cos(suntheta);
  const auto gamma = acos(clamp(cospsi, -1.0f, 1.0f));
  const auto theta = min(spherical.x, half_pi - 0.001f);
  Float3 rgb;
  $if(type == static_cast<unsigned>(NODE_SKY_PREETHAM)) {
    const auto x = radiance_y * sky_perez(config_y, theta, gamma);
    const auto y = radiance_z * sky_perez(config_z, theta, gamma);
    const auto Y = radiance_x * sky_perez(config_x, theta, gamma);
    Float X = 0.0f;
    Float Z = 0.0f;
    $if(y != 0.0f) { X = (x / y) * Y; };
    $if((y != 0.0f) & (Y != 0.0f)) { Z = (1.0f - x - y) / y * Y; };
    rgb = max(xyz_to_rgb(kg, make_float3(X, Y, Z)), make_float3(0.0f));
  }
  $else {
    const auto x = sky_radiance_internal(config_x, theta, gamma) * radiance_x;
    const auto y = sky_radiance_internal(config_y, theta, gamma) * radiance_y;
    const auto z = sky_radiance_internal(config_z, theta, gamma) * radiance_z;
    rgb = max(xyz_to_rgb(kg, make_float3(x, y, z)), make_float3(0.0f)) *
          (2.0f * pi / 683.0f);
  };
  return rgb;
}

[[nodiscard]] Float3
sky_radiance_nishita(Cursor &cursor, const KernelGlobals &kernel_globals,
                     ShaderData &shader_data, const PathState &path_state,
                     Expr<luisa::float3> direction) noexcept {

  const auto pixel_bottom_x = cursor.floating();
  const auto pixel_bottom_y = cursor.floating();
  const auto pixel_bottom_z = cursor.floating();
  const auto pixel_top_x = cursor.floating();
  const auto pixel_top_y = cursor.floating();
  const auto pixel_top_z = cursor.floating();
  const auto pixel_bottom =
      make_float3(pixel_bottom_x, pixel_bottom_y, pixel_bottom_z);
  const auto pixel_top = make_float3(pixel_top_x, pixel_top_y, pixel_top_z);
  const auto sun_elevation = cursor.floating();
  const auto sun_rotation = cursor.floating();
  const auto angular_diameter = cursor.floating();
  const auto sun_intensity = cursor.floating();
  const auto earth_intersection_angle = cursor.floating();
  const auto texture_id = cursor.word().bitcast<std::int32_t>();

  const auto spherical = direction_to_spherical(direction);
  const auto sun_direction =
      spherical_to_direction(sun_elevation - half_pi, sun_rotation - half_pi);
  const auto sun_direction_angle =
      spherical_geometry::precise_angle(direction, sun_direction);
  const auto half_angular = angular_diameter * 0.5f;
  const auto direction_elevation = half_pi - spherical.x;

  Float3 xyz = make_float3(0.0f);
  const auto draw_sun = (angular_diameter >= 0.0f) &
                        (sun_direction_angle < half_angular) &
                        (direction_elevation > earth_intersection_angle) &
                        !(((path_state.flag & path_ray_importance_bake) != 0u) &
                          kernel_globals.background_use_sun_guiding());
  $if(draw_sun) {
    const auto y =
        (direction_elevation - sun_elevation) / angular_diameter + 0.5f;
    const auto normalized_angle = sun_direction_angle / half_angular;
    const auto limb_darkening =
        1.0f - 0.6f * (1.0f - sqrt(1.0f - normalized_angle * normalized_angle));
    xyz = lerp(pixel_bottom, pixel_top, y) * (sun_intensity * limb_darkening);
  };

  const auto x =
      fract((-spherical.y - half_pi + sun_rotation) * inverse_two_pi);
  const auto y = copysign(sqrt(abs(direction_elevation) * two_over_pi),
                          direction_elevation) *
                     0.5f +
                 0.5f;
  const Dual2 uv{.val = make_float2(x, y),
                 .dx = make_float2(0.0f),
                 .dy = make_float2(0.0f)};
  xyz +=
      kernel_globals.kernel_image_interp_with_udim(shader_data, texture_id, uv)
          .xyz();

  return max(xyz_to_rgb(kernel_globals, xyz), make_float3(0.0f));
}

} // namespace

void node_tex_sky(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals, ShaderData &shader_data,
                  const PathState &path_state) noexcept {
  const auto sky_type = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto direction_offset = cursor.byte(packed_offsets, 0u);
  const auto output_offset = cursor.byte(packed_offsets, 1u);
  const auto direction = stack_load_float3(stack, direction_offset);
  Float3 rgb;
  $if((sky_type == static_cast<unsigned>(NODE_SKY_PREETHAM)) |
      (sky_type == static_cast<unsigned>(NODE_SKY_HOSEK))) {
    rgb = sky_radiance_analytic(cursor, kernel_globals, sky_type, direction);
  }
  $else {
    rgb = sky_radiance_nishita(cursor, kernel_globals, shader_data, path_state,
                               direction);
  };
  stack_store_float3(stack, output_offset, rgb);
}

} // namespace psycles::luisa_backend::cycles_svm::detail
