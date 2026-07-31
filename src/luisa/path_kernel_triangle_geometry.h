#pragma once

#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct TriangleGeometryContext {
    Var<GeometryGpu> geometry;
    Var<Triangle> triangle;
    Float3 p0;
    Float3 p1;
    Float3 p2;
    Float3 n0;
    Float3 n1;
    Float3 n2;
    Float2 uv0;
    Float2 uv1;
    Float2 uv2;
    Float4 tangent0;
    Float4 tangent1;
    Float4 tangent2;
    Float3 generated0;
    Float3 generated1;
    Float3 generated2;
    UInt material_slot;
    Float random_per_island;
    Bool smooth;
};

class TriangleGeometryComponent {

public:
    virtual ~TriangleGeometryComponent() noexcept = default;

    [[nodiscard]] virtual TriangleGeometryContext emit(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> geometry_index,
        Expr<std::uint32_t> primitive_index)
        const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const TriangleGeometryComponent>
make_triangle_geometry_component();

}// namespace psycles::luisa_backend::detail
