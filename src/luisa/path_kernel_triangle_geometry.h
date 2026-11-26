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
