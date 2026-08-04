#pragma once

#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct PrimitiveMaterialContext {
  Var<MaterialBindingGpu> binding;
  UInt cycles_surface_shader;
  UInt cycles_object_index;
  Bool has_volume;
};

// Resolves the effective per-primitive material while Luisa records the
// kernel AST. Mesh faces and curves share this exact override, shader-identity,
// and volume-capability boundary.
class PrimitiveMaterialComponent {

public:
  virtual ~PrimitiveMaterialComponent() noexcept = default;

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
