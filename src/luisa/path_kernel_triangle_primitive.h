#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/volume_stack.h>

#include <memory>

namespace psycles::luisa_backend::detail {

// Shared host-stage result for every operation which starts from a committed
// triangle. Keeping material override selection and Cycles identity assembly
// here makes ordinary surface shading and volume-only traversal use exactly
// the same semantic boundary.
struct TrianglePrimitiveContext {
    UInt instance_id;
    UInt primitive_id;
    Var<InstanceGpu> instance;
    Var<GeometryGpu> geometry;
    Var<Triangle> triangle;
    UInt material_slot;
    Bool smooth;
    Var<MaterialBindingGpu> material_binding;
    UInt cycles_surface_shader;
    UInt cycles_object_index;
    UInt cycles_primitive_index;
    Bool has_volume;
    Bool may_emit;
    Bool has_transparent_shadow;
    UInt triangle_emission_sampling;

    [[nodiscard]] VolumeStackEntry
    volume_stack_entry() const noexcept;
};

class TrianglePrimitiveComponent {

  public:
    virtual ~TrianglePrimitiveComponent() noexcept =
        default;

    [[nodiscard]] virtual TrianglePrimitiveContext
    emit(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id)
        const noexcept = 0;
};

[[nodiscard]] inline luisa::compute::Float4x4 instance_normal_to_world(
    const std::shared_ptr<LuisaSceneData> &scene,
    Expr<std::uint32_t> instance_id,
    const Var<InstanceGpu> &instance) noexcept {
    luisa::compute::Float4x4 normal_to_world =
        luisa::compute::make_float4x4(1.0f);
    $if(instance.cycles_transform_applied != 0u) {
        normal_to_world = transpose(instance.cycles_world_to_object);
    }
    $else {
        normal_to_world = transpose(
            inverse(scene->accel->instance_transform(instance_id)));
    };
    return normal_to_world;
}

[[nodiscard]]
std::shared_ptr<const TrianglePrimitiveComponent>
make_triangle_primitive_component();

}// namespace psycles::luisa_backend::detail
