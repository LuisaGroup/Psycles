#pragma once

#include "path_kernel_triangle_primitive.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct TriangleGeometryContext {
    TrianglePrimitiveContext primitive;
    Float3 p0;
    Float3 p1;
    Float3 p2;
    // Final support used by Cycles traversal. For transform-applied meshes
    // these vertices are already in world space; otherwise they alias p0-p2.
    Float3 cycles_intersection_p0;
    Float3 cycles_intersection_p1;
    Float3 cycles_intersection_p2;
    Float3 n0;
    Float3 n1;
    Float3 n2;
    Float2 uv0;
    Float2 uv1;
    Float2 uv2;
    Float4 tangent0;
    Float4 tangent1;
    Float4 tangent2;
    Float3 undisplaced_p0;
    Float3 undisplaced_p1;
    Float3 undisplaced_p2;
    Float3 undisplaced_n0;
    Float3 undisplaced_n1;
    Float3 undisplaced_n2;
    Float4 undisplaced_tangent0;
    Float4 undisplaced_tangent1;
    Float4 undisplaced_tangent2;
    Float3 generated0;
    Float3 generated1;
    Float3 generated2;
    Float random_per_island;
};

class TriangleGeometryComponent {

public:
    virtual ~TriangleGeometryComponent() noexcept = default;

    [[nodiscard]] virtual TriangleGeometryContext emit(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_index)
        const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const TriangleGeometryComponent>
make_triangle_geometry_component();

}// namespace psycles::luisa_backend::detail
