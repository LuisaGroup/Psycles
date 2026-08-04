#pragma once

#include "curve_ribbon_component.h"
#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct CurveSegmentContext {
  UInt instance_id;
  UInt segment_id;
  Var<InstanceGpu> instance;
  Var<GeometryGpu> geometry;
  Var<CurveSegmentGpu> segment;
  CurveControlPoints control_points;
};

struct CurvePrimitiveContext {
  CurveSegmentContext curve;
  UInt material_slot;
  Var<MaterialBindingGpu> material_binding;
  UInt cycles_surface_shader;
  UInt cycles_object_index;
  UInt cycles_primitive_index;
  Bool has_volume;

  [[nodiscard]] VolumeStackEntry volume_stack_entry() const noexcept;
};

// Curves use a segment AABB as the acceleration primitive while Cycles uses
// the containing curve as its shading and self-intersection primitive. This
// component is the single typed mapping between those two address spaces.
class CurvePrimitiveComponent {

public:
  virtual ~CurvePrimitiveComponent() noexcept = default;

  [[nodiscard]] virtual CurveSegmentContext
  emit_segment(const std::shared_ptr<LuisaSceneData> &scene,
               Expr<std::uint32_t> instance_id,
               Expr<std::uint32_t> segment_id) const noexcept = 0;

  [[nodiscard]] virtual CurvePrimitiveContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id,
       Expr<std::uint32_t> segment_id) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const CurvePrimitiveComponent>
make_curve_primitive_component();

} // namespace psycles::luisa_backend::detail
