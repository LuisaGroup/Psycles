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
        const auto corner = primitive_index * 3u;
        const auto attribute_index =
            [&](UInt point_index,
                std::uint32_t corner_offset,
                std::uint32_t corner_flag) noexcept {
                return select(
                    point_index,
                    corner + corner_offset,
                    (geometry.attribute_domains &
                     corner_flag) != 0u);
            };
        const auto normal_i0 = attribute_index(
            triangle.i0, 0u, geometry_normal_corner);
        const auto normal_i1 = attribute_index(
            triangle.i1, 1u, geometry_normal_corner);
        const auto normal_i2 = attribute_index(
            triangle.i2, 2u, geometry_normal_corner);
        const auto uv_i0 = attribute_index(
            triangle.i0, 0u, geometry_uv_corner);
        const auto uv_i1 = attribute_index(
            triangle.i1, 1u, geometry_uv_corner);
        const auto uv_i2 = attribute_index(
            triangle.i2, 2u, geometry_uv_corner);
        const auto tangent_i0 = attribute_index(
            triangle.i0, 0u, geometry_uv_tangent_corner);
        const auto tangent_i1 = attribute_index(
            triangle.i1, 1u, geometry_uv_tangent_corner);
        const auto tangent_i2 = attribute_index(
            triangle.i2, 2u, geometry_uv_tangent_corner);
        const auto generated_i0 = attribute_index(
            triangle.i0, 0u, geometry_generated_corner);
        const auto generated_i1 = attribute_index(
            triangle.i1, 1u, geometry_generated_corner);
        const auto generated_i2 = attribute_index(
            triangle.i2, 2u, geometry_generated_corner);
        const auto positions =
            scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 1u);
        const auto normals =
            scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 2u);
        const auto uv =
            scene->heap->buffer<luisa::float2>(
                geometry.bindless_base + 3u);
        const auto generated =
            scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 5u);
        const auto tangents =
            scene->heap->buffer<luisa::float4>(
                geometry.bindless_base + 7u);
        return {
            .primitive = std::move(primitive),
            .p0 = positions.read(triangle.i0),
            .p1 = positions.read(triangle.i1),
            .p2 = positions.read(triangle.i2),
            .n0 = normals.read(normal_i0),
            .n1 = normals.read(normal_i1),
            .n2 = normals.read(normal_i2),
            .uv0 = uv.read(uv_i0),
            .uv1 = uv.read(uv_i1),
            .uv2 = uv.read(uv_i2),
            .tangent0 = tangents.read(tangent_i0),
            .tangent1 = tangents.read(tangent_i1),
            .tangent2 = tangents.read(tangent_i2),
            .generated0 = generated.read(generated_i0),
            .generated1 = generated.read(generated_i1),
            .generated2 = generated.read(generated_i2),
            .random_per_island =
                scene->heap
                    ->buffer<float>(
                        geometry.bindless_base + 6u)
                    .read(primitive_index)};
    }
};

}// namespace

std::shared_ptr<const TriangleGeometryComponent>
make_triangle_geometry_component() {
    return std::make_shared<
        BindlessTriangleGeometryComponent>();
}

}// namespace psycles::luisa_backend::detail
