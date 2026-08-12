#include "path_kernel_primitive_material.h"

#include "cycles_shader_identity.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BindlessPrimitiveMaterialComponent final
    : public PrimitiveMaterialComponent {

public:
  UInt triangle_material_slot(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<GeometryGpu> &geometry,
      Expr<std::uint32_t> primitive_id) const noexcept override {
    return scene->heap
        ->buffer<luisa::uint>(geometry.bindless_base + 4u)
        .read(primitive_id);
  }

  UInt curve_material_slot(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<GeometryGpu> &geometry,
      const Var<CurveSegmentGpu> &segment) const noexcept override {
    return scene->heap
        ->buffer<luisa::uint>(geometry.bindless_base + 4u)
        .read(segment.curve_index);
  }

  Var<MaterialBindingGpu> resolve_binding(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<InstanceGpu> &instance,
      const Var<GeometryGpu> &geometry,
      Expr<std::uint32_t> material_slot) const noexcept override {
    UInt resolved_material_slot = material_slot;
    Var<MaterialBindingGpu> binding = scene->geometry_material_buffer->read(
        geometry.material_offset +
        min(resolved_material_slot, max(geometry.material_count, 1u) - 1u));
    $if(resolved_material_slot < instance.override_count) {
      binding = scene->override_material_buffer->read(instance.override_offset +
                                                      resolved_material_slot);
    };
    return binding;
  }

  UInt cycles_object_index(
      Expr<std::uint32_t> instance_id,
      const Var<InstanceGpu> &instance) const noexcept override {
    UInt resolved_instance_id = instance_id;
    return select(resolved_instance_id, instance.cycles_object_index,
                  instance.cycles_object_index !=
                      cycles_shader_identity::invalid_index);
  }

  PrimitiveMaterialContext emit(const std::shared_ptr<LuisaSceneData> &scene,
                                Expr<std::uint32_t> instance_id,
                                const Var<InstanceGpu> &instance,
                                const Var<GeometryGpu> &geometry,
                                Expr<std::uint32_t> material_slot,
                                Expr<bool> smooth) const noexcept override {
    UInt resolved_instance_id = instance_id;
    auto binding = resolve_binding(
        scene, instance, geometry, material_slot);

    const auto has_cycles_shader_identity =
        binding.cycles_shader_index != cycles_shader_identity::invalid_index;
    const auto base_shader_identity =
        select(binding.material_identity, binding.cycles_shader_index,
               has_cycles_shader_identity);
    UInt cycles_surface_shader =
        base_shader_identity | cycles_shader_identity::cast_shadow |
        select(0u, cycles_shader_identity::smooth_normal, smooth);
    UInt object_index = cycles_object_index(resolved_instance_id, instance);
    Bool has_volume = (binding.flags & material_flag_has_volume) != 0u;
    const Bool may_emit = (binding.flags & material_flag_may_emit) != 0u;
    const UInt emission_sampling =
        (binding.flags &
         material_emission_sampling_mask) >>
        material_emission_sampling_shift;
    const Bool light_visible =
        (instance.visibility_mask & mesh_light_sampling_visibility) != 0u;
    const Bool is_triangle = geometry.primitive_kind == geometry_kind_triangle;
    const Bool participates_in_triangle_sampling =
        may_emit & light_visible & is_triangle &
        (emission_sampling !=
         static_cast<std::uint32_t>(contract::EmissionSampling::none));
    UInt triangle_emission_sampling =
        select(static_cast<std::uint32_t>(contract::EmissionSampling::none),
               emission_sampling, participates_in_triangle_sampling);
    return {.binding = std::move(binding),
            .cycles_surface_shader = std::move(cycles_surface_shader),
            .cycles_object_index = std::move(object_index),
            .has_volume = std::move(has_volume),
            .may_emit = std::move(may_emit),
            .triangle_emission_sampling =
                std::move(triangle_emission_sampling)};
  }
};

} // namespace

std::shared_ptr<const PrimitiveMaterialComponent>
make_primitive_material_component() {
  return std::make_shared<BindlessPrimitiveMaterialComponent>();
}

} // namespace psycles::luisa_backend::detail
