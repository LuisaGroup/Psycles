#include "surface_geometry_context.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

Float3 surface_geometry_tangent(const SurfacePoint &point) noexcept {
  Float3 tangent = normalize(point.dpdu);
  const auto has_mesh_generated =
      !point.is_curve & (point.geometry_index != ~0u);
  $if(has_mesh_generated) {
    const auto object_tangent = make_float3(-(point.generated.y - 0.5f),
                                            point.generated.x - 0.5f, 0.0f);
    const auto world_tangent =
        normalize(point.normal_to_world_x * object_tangent.x +
                  point.normal_to_world_y * object_tangent.y +
                  point.normal_to_world_z * object_tangent.z);
    tangent = cross(point.shading_normal,
                    normalize(cross(world_tangent, point.shading_normal)));
  };
  return tangent;
}

} // namespace psycles::luisa_backend::detail
