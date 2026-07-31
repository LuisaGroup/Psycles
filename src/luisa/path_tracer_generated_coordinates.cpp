#include "path_tracer_generated_coordinates.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace psycles::luisa_backend::detail {

Vec3f GeneratedCoordinateMapping::apply(
    Vec3f point) const noexcept {
    const auto &e = object_to_generated.elements;
    return {
        e[0u] * point.x +
            e[4u] * point.y +
            e[8u] * point.z +
            e[12u],
        e[1u] * point.x +
            e[5u] * point.y +
            e[9u] * point.z +
            e[13u],
        e[2u] * point.x +
            e[6u] * point.y +
            e[10u] * point.z +
            e[14u]};
}

GeneratedCoordinateMapping
make_generated_coordinate_mapping(
    const contract::TriangleMeshDesc &geometry) noexcept {
    if (geometry.generated_transform) {
        return {
            .object_to_generated =
                *geometry.generated_transform};
    }
    if (geometry.positions.empty()) {
        return {};
    }

    auto bounds_min = geometry.positions.front();
    auto bounds_max = geometry.positions.front();
    for (const auto position : geometry.positions) {
        bounds_min.x =
            std::min(bounds_min.x, position.x);
        bounds_min.y =
            std::min(bounds_min.y, position.y);
        bounds_min.z =
            std::min(bounds_min.z, position.z);
        bounds_max.x =
            std::max(bounds_max.x, position.x);
        bounds_max.y =
            std::max(bounds_max.y, position.y);
        bounds_max.z =
            std::max(bounds_max.z, position.z);
    }
    const auto axis =
        [](float minimum, float maximum) noexcept {
            const auto extent = maximum - minimum;
            if (std::abs(extent) <= 1.0e-20f) {
                return std::pair{0.0f, 0.5f};
            }
            const auto scale = 1.0f / extent;
            return std::pair{
                scale, -minimum * scale};
        };
    const auto [scale_x, offset_x] =
        axis(bounds_min.x, bounds_max.x);
    const auto [scale_y, offset_y] =
        axis(bounds_min.y, bounds_max.y);
    const auto [scale_z, offset_z] =
        axis(bounds_min.z, bounds_max.z);

    Mat4f transform;
    transform.elements = {
        scale_x, 0.0f, 0.0f, 0.0f,
        0.0f, scale_y, 0.0f, 0.0f,
        0.0f, 0.0f, scale_z, 0.0f,
        offset_x, offset_y, offset_z, 1.0f};
    return {
        .object_to_generated =
            std::move(transform)};
}

}// namespace psycles::luisa_backend::detail
