#pragma once

#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct PrimitiveMaterialContext {
  Var<MaterialBindingGpu> binding;
  UInt cycles_surface_shader;
  UInt cycles_object_index;
  Bool has_volume;
  // These are deliberately distinct. may_emit controls whether a visible
  // closure can contribute at the endpoint; triangle_emission_sampling is
  // non-NONE only when this exact primitive belongs to the sampled mesh-light
  // population and therefore competes in forward-hit MIS.
  Bool may_emit;
  Bool has_transparent_shadow;
  UInt triangle_emission_sampling;
};

// Resolves the effective per-primitive material while Luisa records the
// kernel AST. Mesh faces and curves share this exact override, shader-identity,
// and volume-capability boundary.
class PrimitiveMaterialComponent {

public:
  virtual ~PrimitiveMaterialComponent() noexcept = default;

  // Bindless primitive payload layout is part of material selection, not of
  // surface geometry evaluation. Keeping these address translations here
  // gives shading, volume traversal, and scheduler metadata one definition
  // for the primitive -> material-slot relation.
  [[nodiscard]] virtual UInt
  triangle_material_slot(const std::shared_ptr<LuisaSceneData> &scene,
                         const Var<GeometryGpu> &geometry,
                         Expr<std::uint32_t> primitive_id) const noexcept = 0;

  [[nodiscard]] virtual UInt
  curve_material_slot(const std::shared_ptr<LuisaSceneData> &scene,
                      const Var<GeometryGpu> &geometry,
                      const Var<CurveSegmentGpu> &segment) const noexcept = 0;

  [[nodiscard]] virtual Var<MaterialBindingGpu>
  resolve_binding(const std::shared_ptr<LuisaSceneData> &scene,
                  const Var<InstanceGpu> &instance,
                  const Var<GeometryGpu> &geometry,
                  Expr<std::uint32_t> material_slot) const noexcept = 0;

  [[nodiscard]] virtual UInt
  cycles_object_index(Expr<std::uint32_t> instance_id,
                      const Var<InstanceGpu> &instance) const noexcept = 0;

  [[nodiscard]] virtual PrimitiveMaterialContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id, const Var<InstanceGpu> &instance,
       const Var<GeometryGpu> &geometry, Expr<std::uint32_t> material_slot,
       Expr<bool> smooth) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const PrimitiveMaterialComponent>
make_primitive_material_component();

} // namespace psycles::luisa_backend::detail
