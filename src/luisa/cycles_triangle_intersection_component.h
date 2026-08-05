#pragma once

#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct CyclesTriangleIntersection {
    Bool valid;
    Float distance;
    Float2 barycentric;
};

// Builds the Cycles/Embree Pluecker triangle relation in Luisa DSL. The host
// component boundary keeps source-representation policy out of the traversal
// orchestration while still emitting one backend-specialized shader AST.
class CyclesTriangleIntersectionComponent {

  public:
    virtual ~CyclesTriangleIntersectionComponent() noexcept = default;

    [[nodiscard]] virtual CyclesTriangleIntersection intersect(
        const Var<luisa::compute::Ray> &world_ray,
        Expr<luisa::float4x4> world_to_object,
        Expr<std::uint32_t> transform_applied,
        Expr<luisa::float3> p0,
        Expr<luisa::float3> p1,
        Expr<luisa::float3> p2) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<
    const CyclesTriangleIntersectionComponent>
make_cycles_triangle_intersection_component();

}// namespace psycles::luisa_backend::detail
