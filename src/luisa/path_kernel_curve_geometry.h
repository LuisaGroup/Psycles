#pragma once

#include "path_kernel_curve_primitive.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct CurveGeometryContext {
  CurvePrimitiveContext primitive;
  CurveRibbonIntersection intersection;
  Float3 object_position;
  Float3 position;
  Float3 object_shading_normal;
  Float3 shading_normal;
  Float3 geometric_normal;
  Float3 object_dpdu;
  Float3 dpdu;
  Float3 dpdv;
  Float intercept;
  Float length;
  Float thickness;
  Float3 tangent_normal;
  Float random;
};

// Reconstructs Cycles ShaderData for a committed ribbon segment. The
// procedural acceleration hit stores only t, so this typed component repeats
// the exact segment intersection to recover u/v before emitting the shader
// setup algebra from curve_shader_setup and curve.h.
class CurveGeometryComponent {

public:
  virtual ~CurveGeometryComponent() noexcept = default;

  [[nodiscard]] virtual CurveGeometryContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id, Expr<std::uint32_t> segment_id,
       const Var<luisa::compute::Ray> &world_ray,
       Expr<float> committed_distance) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const CurveGeometryComponent>
make_curve_geometry_component();

} // namespace psycles::luisa_backend::detail
