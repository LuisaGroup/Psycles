#include "path_kernel_triangle_primitive.h"

#include "cycles_shader_identity.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BindlessTrianglePrimitiveComponent final
    : public TrianglePrimitiveComponent {

  public:
    TrianglePrimitiveContext emit(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id)
        const noexcept override {
        UInt resolved_instance_id = instance_id;
        UInt resolved_primitive_id = primitive_id;
        Var<InstanceGpu> instance =
            scene->instance_buffer->read(
                resolved_instance_id);
        Var<GeometryGpu> geometry =
            scene->geometry_buffer->read(
                instance.geometry_index);
        Var<Triangle> triangle =
            scene->heap
                ->buffer<Triangle>(
                    geometry.bindless_base)
                .read(resolved_primitive_id);
        UInt material_slot =
            scene->heap
                ->buffer<luisa::uint>(
                    geometry.bindless_base + 4u)
                .read(resolved_primitive_id);
        Bool smooth =
            scene->heap
                ->buffer<luisa::uint>(
                    geometry.bindless_base + 8u)
                .read(resolved_primitive_id) != 0u;

        Var<MaterialBindingGpu> material_binding =
            scene->geometry_material_buffer->read(
                geometry.material_offset +
                min(
                    material_slot,
                    max(
                        geometry.material_count,
                        1u) -
                        1u));
        $if(material_slot <
            instance.override_count) {
            material_binding =
                scene->override_material_buffer->read(
                    instance.override_offset +
                    material_slot);
        };

        const auto has_cycles_shader_identity =
            material_binding.cycles_shader_index !=
            cycles_shader_identity::invalid_index;
        const auto base_shader_identity =
            select(
                material_binding.material_identity,
                material_binding.cycles_shader_index,
                has_cycles_shader_identity);
        UInt cycles_surface_shader =
            base_shader_identity |
            cycles_shader_identity::cast_shadow |
            select(
                0u,
                cycles_shader_identity::smooth_normal,
                smooth);
        UInt cycles_object_index =
            select(
                resolved_instance_id,
                instance.cycles_object_index,
                instance.cycles_object_index !=
                    cycles_shader_identity::
                        invalid_index);
        Bool has_volume =
            (material_binding.flags &
             material_flag_has_volume) != 0u;
        return {
            .instance_id =
                std::move(resolved_instance_id),
            .primitive_id =
                std::move(resolved_primitive_id),
            .instance = std::move(instance),
            .geometry = std::move(geometry),
            .triangle = std::move(triangle),
            .material_slot =
                std::move(material_slot),
            .smooth = std::move(smooth),
            .material_binding =
                std::move(material_binding),
            .cycles_surface_shader =
                std::move(cycles_surface_shader),
            .cycles_object_index =
                std::move(cycles_object_index),
            .has_volume =
                std::move(has_volume)};
    }
};

}// namespace

VolumeStackEntry
TrianglePrimitiveContext::
    volume_stack_entry() const noexcept {
    const auto valid =
        has_volume &
        ((material_binding.cycles_shader_index !=
          cycles_shader_identity::invalid_index) |
         (material_binding.material_identity !=
          cycles_shader_identity::invalid_index));
    return {
        .object = cycles_object_index,
        .shader = cycles_surface_shader,
        .surface_tag =
            material_binding.surface_tag,
        .parameter_block =
            material_binding.parameter_block,
        .instance_id = instance_id,
        .valid = valid};
}

std::shared_ptr<const TrianglePrimitiveComponent>
make_triangle_primitive_component() {
    return std::make_shared<
        BindlessTrianglePrimitiveComponent>();
}

}// namespace psycles::luisa_backend::detail
