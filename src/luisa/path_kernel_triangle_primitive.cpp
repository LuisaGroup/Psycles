#include "path_kernel_triangle_primitive.h"

#include "cycles_shader_identity.h"
#include "path_kernel_primitive_material.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BindlessTrianglePrimitiveComponent final
    : public TrianglePrimitiveComponent {

private:
  std::shared_ptr<const PrimitiveMaterialComponent> _material;

public:
  BindlessTrianglePrimitiveComponent()
      : _material{make_primitive_material_component()} {}

  TrianglePrimitiveContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id,
       Expr<std::uint32_t> primitive_id) const noexcept override {
    UInt resolved_instance_id = instance_id;
    UInt resolved_primitive_id = primitive_id;
    Var<InstanceGpu> instance =
        scene->instance_buffer->read(resolved_instance_id);
    Var<GeometryGpu> geometry =
        scene->geometry_buffer->read(instance.geometry_index);
    Var<Triangle> triangle =
        scene->heap->buffer<Triangle>(geometry.bindless_base)
            .read(resolved_primitive_id);
    UInt material_slot = _material->triangle_material_slot(
        scene, geometry, resolved_primitive_id);
    Bool smooth = scene->heap->buffer<luisa::uint>(geometry.bindless_base + 8u)
                      .read(resolved_primitive_id) != 0u;

    auto material = _material->emit(scene, resolved_instance_id, instance,
                                    geometry, material_slot, smooth);
    UInt cycles_primitive_index =
        geometry.cycles_primitive_offset + resolved_primitive_id;
    return {.instance_id = std::move(resolved_instance_id),
            .primitive_id = std::move(resolved_primitive_id),
            .instance = std::move(instance),
            .geometry = std::move(geometry),
            .triangle = std::move(triangle),
            .material_slot = std::move(material_slot),
            .smooth = std::move(smooth),
            .material_binding = std::move(material.binding),
            .cycles_surface_shader = std::move(material.cycles_surface_shader),
            .cycles_object_index = std::move(material.cycles_object_index),
            .cycles_primitive_index = std::move(cycles_primitive_index),
            .has_volume = std::move(material.has_volume),
            .may_emit = std::move(material.may_emit),
            .triangle_emission_sampling =
                std::move(material.triangle_emission_sampling)};
  }
};

} // namespace

VolumeStackEntry TrianglePrimitiveContext::volume_stack_entry() const noexcept {
  const auto valid = has_volume & ((material_binding.cycles_shader_index !=
                                    cycles_shader_identity::invalid_index) |
                                   (material_binding.material_identity !=
                                    cycles_shader_identity::invalid_index));
  return {.object = cycles_object_index,
          .shader = cycles_surface_shader,
          .surface_tag = material_binding.surface_tag,
          .parameter_block = material_binding.parameter_block,
          .instance_id = instance_id,
          .sample_method = material_binding.volume_sampling,
          .valid = valid};
}

std::shared_ptr<const TrianglePrimitiveComponent>
make_triangle_primitive_component() {
  return std::make_shared<BindlessTrianglePrimitiveComponent>();
}

} // namespace psycles::luisa_backend::detail
