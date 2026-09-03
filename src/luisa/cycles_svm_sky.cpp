/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/spherical_geometry.h>

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

} // namespace

void node_tex_sky(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  ShaderData &shader_data,
                  const PathState &path_state) noexcept {
  const auto sky_type = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto direction_offset = cursor.byte(packed_offsets, 0u);
  const auto output_offset = cursor.byte(packed_offsets, 1u);

  // This compiler emits NODE_TEX_SKY only for the two Nishita modes. Keeping
  // the runtime guard makes a malformed or externally supplied legacy stream
  // fail at the semantic boundary instead of misreading its 32-word payload
  // as the 12-word Nishita record.
  $if((sky_type != static_cast<std::uint32_t>(NODE_SKY_SINGLE_SCATTERING)) &
      (sky_type != static_cast<std::uint32_t>(NODE_SKY_MULTIPLE_SCATTERING))) {
    dsl::unreachable("non-Nishita NODE_TEX_SKY reached Nishita SVM handler");
  };

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

  const auto direction = stack_load_float3(stack, direction_offset);
  const auto spherical = direction_to_spherical(direction);
  const auto sun_direction = spherical_to_direction(
      sun_elevation - half_pi, sun_rotation - half_pi);
  const auto sun_direction_angle =
      spherical_geometry::precise_angle(direction, sun_direction);
  const auto half_angular = angular_diameter * 0.5f;
  const auto direction_elevation = half_pi - spherical.x;

  Float3 xyz = make_float3(0.0f);
  const auto draw_sun =
      (angular_diameter >= 0.0f) &
      (sun_direction_angle < half_angular) &
      (direction_elevation > earth_intersection_angle) &
      !(((path_state.flag & path_ray_importance_bake) != 0u) &
        kernel_globals.background_use_sun_guiding());
  $if(draw_sun) {
    const auto y =
        (direction_elevation - sun_elevation) / angular_diameter + 0.5f;
    const auto normalized_angle = sun_direction_angle / half_angular;
    const auto limb_darkening =
        1.0f - 0.6f *
                   (1.0f - sqrt(1.0f - normalized_angle * normalized_angle));
    xyz = lerp(pixel_bottom, pixel_top, y) *
          (sun_intensity * limb_darkening);
  };

  const auto x = fract(
      (-spherical.y - half_pi + sun_rotation) * inverse_two_pi);
  const auto y =
      copysign(sqrt(abs(direction_elevation) * two_over_pi),
               direction_elevation) *
          0.5f +
      0.5f;
  const Dual2 uv{.val = make_float2(x, y),
                 .dx = make_float2(0.0f),
                 .dy = make_float2(0.0f)};
  xyz += kernel_globals
             .kernel_image_interp_with_udim(shader_data, texture_id, uv)
             .xyz();

  stack_store_float3(
      stack, output_offset,
      max(xyz_to_rgb(kernel_globals, xyz), make_float3(0.0f)));
}

} // namespace psycles::luisa_backend::cycles_svm::detail
