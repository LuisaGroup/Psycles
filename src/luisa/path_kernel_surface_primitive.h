#pragma once

#include "path_tracer_lighting.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct SurfacePrimitiveGeometryContext {
  Var<InstanceGpu> instance;
  Float3 p0;
  Float3 p1;
  Float3 p2;
  Float3 n0;
  Float3 n1;
  Float3 n2;
  luisa::compute::Float4x4 object_to_world;
  luisa::compute::Float4x4 world_to_object;
  Float3 wp0;
  Float3 wp1;
  Float3 wp2;
  Float3 hit_position;
  Float3 object_hit_position;
  Float differential_radius;
  Bool is_curve;
  Bool triangle_smooth;
  UInt surface_tag;
  UInt cycles_surface_shader;
  UInt cycles_object_index;
  UInt cycles_primitive_index;
  VolumeStackEntry volume_stack_entry;
  Bool surface_has_volume;
  SurfacePoint point;
};

// Runtime primitive dispatch assembled at the host/JIT stage. Each dynamic
// branch expands a strongly typed triangle or curve component; no weak
// float4 payload or device virtual dispatch is introduced.
class SurfacePrimitiveGeometryComponent {

public:
  virtual ~SurfacePrimitiveGeometryComponent() noexcept = default;

  [[nodiscard]] virtual SurfacePrimitiveGeometryContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit,
       const Var<luisa::compute::Ray> &ray, Expr<float> ray_dP,
       Expr<float> ray_dD,
       const SafeNormalizeCallable &safe_normalize) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const SurfacePrimitiveGeometryComponent>
make_surface_primitive_geometry_component();

} // namespace psycles::luisa_backend::detail
