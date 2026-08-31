#include "surface_geometry_context.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

Float3 surface_geometry_tangent(const SurfacePoint &point) noexcept {
  Float3 tangent = native_vector_math::normalize_unchecked(point.dpdu);
  const auto has_mesh_generated =
      !point.is_curve & (point.geometry_index != ~0u);
  $if(has_mesh_generated) {
    const auto object_tangent = make_float3(-(point.generated.y - 0.5f),
                                            point.generated.x - 0.5f, 0.0f);
    const auto world_tangent = native_vector_math::normalize_unchecked(
        point.normal_to_world_x * object_tangent.x +
        point.normal_to_world_y * object_tangent.y +
        point.normal_to_world_z * object_tangent.z);
    tangent = cross(
        point.shading_normal,
        native_vector_math::normalize_unchecked(
            cross(world_tangent, point.shading_normal)));
  };
  return tangent;
}

} // namespace psycles::luisa_backend::detail
