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
    Bool has_volume;

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

[[nodiscard]]
std::shared_ptr<const TrianglePrimitiveComponent>
make_triangle_primitive_component();

}// namespace psycles::luisa_backend::detail
