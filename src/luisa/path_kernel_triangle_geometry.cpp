#include "path_kernel_triangle_geometry.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BindlessTriangleGeometryComponent final
    : public TriangleGeometryComponent {

  private:
    std::shared_ptr<const TrianglePrimitiveComponent>
        _primitive{
            make_triangle_primitive_component()};

public:
    TriangleGeometryContext emit(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_index)
        const noexcept override {
        auto primitive = _primitive->emit(
            scene,
            instance_id,
            primitive_index);
        auto &geometry = primitive.geometry;
        auto &triangle = primitive.triangle;
        const auto positions =
            scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 1u);
        const auto cycles_intersection_positions =
            scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 9u);
        return {
            .primitive = std::move(primitive),
            .p0 = positions.read(triangle.i0),
            .p1 = positions.read(triangle.i1),
            .p2 = positions.read(triangle.i2),
            .cycles_intersection_p0 =
                cycles_intersection_positions.read(triangle.i0),
            .cycles_intersection_p1 =
                cycles_intersection_positions.read(triangle.i1),
            .cycles_intersection_p2 =
                cycles_intersection_positions.read(triangle.i2)};
    }
};

}// namespace

std::shared_ptr<const TriangleGeometryComponent>
make_triangle_geometry_component() {
    return std::make_shared<
        BindlessTriangleGeometryComponent>();
}

}// namespace psycles::luisa_backend::detail
