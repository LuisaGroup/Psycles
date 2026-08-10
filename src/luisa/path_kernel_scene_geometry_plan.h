#pragma once

#include <cstddef>

namespace psycles::luisa_backend::detail {

// Geometry populations are immutable while a render kernel is recorded.
// Keep primitive reachability in the host/JIT stage so an absent acceleration
// primitive kind does not leave its complete material, geometry, or ray-query
// callback body behind a device-side hit-kind test.
struct ScenePrimitiveStagePlan {
    bool triangles{};
    bool curves{};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return !triangles && !curves;
    }

    [[nodiscard]] constexpr bool mixed() const noexcept {
        return triangles && curves;
    }
};

[[nodiscard]] constexpr ScenePrimitiveStagePlan
make_scene_primitive_stage_plan(
    std::size_t triangle_geometry_count,
    std::size_t curve_geometry_count) noexcept {
    return {
        .triangles = triangle_geometry_count != 0u,
        .curves = curve_geometry_count != 0u};
}

}// namespace psycles::luisa_backend::detail
