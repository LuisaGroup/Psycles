#pragma once

#include "path_tracer_lighting.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct CyclesTriangleSurfaceInput {
  luisa::compute::Float4x4 object_to_world;
  luisa::compute::Float4x4 normal_to_world;
  Bool transform_applied;
  Float2 barycentric;
  Float3 ray_direction;
  Bool smooth;
  Float3 p0;
  Float3 p1;
  Float3 p2;
  Float3 final_p0;
  Float3 final_p1;
  Float3 final_p2;
  Float3 n0;
  Float3 n1;
  Float3 n2;
};

// One coherent triangle representation for all post-intersection geometry.
// Cycles bakes single-user object transforms into mesh vertices before BVH
// construction. Once that happens, hit reconstruction and Ng must consume
// those same final vertices instead of re-evaluating an algebraically
// equivalent object-space interpolation through the instance matrix.
struct CyclesTriangleSurface {
  Float3 p0;
  Float3 p1;
  Float3 p2;
  Float3 n0;
  Float3 n1;
  Float3 n2;
  Float3 world_p0;
  Float3 world_p1;
  Float3 world_p2;
  Float3 object_position;
  Float3 position;
  Float3 object_geometric_normal;
  Float3 geometric_normal;
  Float3 object_shading_normal;
  Float3 shading_normal;
  Bool back_facing;
};

struct CyclesTriangleSurfaceDerivatives {
  Float3 dpdu;
  Float3 dpdv;
};

// Cycles names these derivatives dPdu/dPdv, but for triangles u and v are
// barycentric coordinates rather than a texture map. Therefore
//
//   dPdu = p1 - p0, dPdv = p2 - p0,
//
// in the same final world-space support used for hit reconstruction. Both
// vectors change sign on a back face. Keeping this projection beside the
// support resolver prevents an unrelated UV tangent frame from being
// substituted for the geometric derivative contract.
[[nodiscard]] CyclesTriangleSurfaceDerivatives
cycles_triangle_surface_derivatives(
    const CyclesTriangleSurface &surface) noexcept;

class CyclesTriangleSurfaceComponent {

public:
  virtual ~CyclesTriangleSurfaceComponent() noexcept = default;

  [[nodiscard]] virtual CyclesTriangleSurface
  resolve(const CyclesTriangleSurfaceInput &input,
          const SafeNormalizeCallable &safe_normalize) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const CyclesTriangleSurfaceComponent>
make_cycles_triangle_surface_component();

} // namespace psycles::luisa_backend::detail
