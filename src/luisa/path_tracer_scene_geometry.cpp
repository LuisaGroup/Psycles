#include "path_tracer_scene_geometry.h"

#include <cmath>
#include <limits>

namespace psycles::luisa_backend::detail {

CyclesPrimitiveInterval
CyclesPrimitiveIntervalResolver::resolve(
    std::size_t triangle_count,
    std::optional<std::uint32_t> explicit_offset) noexcept {
    constexpr auto address_space_size =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) +
        1u;
    const auto offset = explicit_offset
                            ? static_cast<std::uint64_t>(
                                  *explicit_offset)
                            : _end;
    if (explicit_offset && offset < _end) {
        return {
            .offset = std::nullopt,
            .error =
                CyclesPrimitiveIntervalError::overlap};
    }
    const auto count =
        static_cast<std::uint64_t>(triangle_count);
    if (offset >= address_space_size ||
        count > address_space_size - offset) {
        return {
            .offset = std::nullopt,
            .error =
                CyclesPrimitiveIntervalError::out_of_range};
    }
    _end = offset + count;
    return {
        .offset =
            static_cast<std::uint32_t>(offset)};
}

namespace {

[[nodiscard]] Vec3f transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        e[0u] * point.x + e[4u] * point.y +
            e[8u] * point.z + e[12u],
        e[1u] * point.x + e[5u] * point.y +
            e[9u] * point.z + e[13u],
        e[2u] * point.x + e[6u] * point.y +
            e[10u] * point.z + e[14u]};
}

}// namespace

float world_triangle_area(
    const Mat4f &transform,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2) noexcept {
    p0 = transform_point(transform, p0);
    p1 = transform_point(transform, p1);
    p2 = transform_point(transform, p2);
    const Vec3f edge01{
        p1.x - p0.x,
        p1.y - p0.y,
        p1.z - p0.z};
    const Vec3f edge02{
        p2.x - p0.x,
        p2.y - p0.y,
        p2.z - p0.z};
    const Vec3f normal{
        edge01.y * edge02.z - edge01.z * edge02.y,
        edge01.z * edge02.x - edge01.x * edge02.z,
        edge01.x * edge02.y - edge01.y * edge02.x};
    return 0.5f * std::sqrt(
        normal.x * normal.x +
        normal.y * normal.y +
        normal.z * normal.z);
}

}// namespace psycles::luisa_backend::detail
