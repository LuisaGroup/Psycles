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

// Exact source completion is a traversal property rather than a primitive
// kind. The scene upload creates a dense or sparse source lookup if and only if
// at least one triangle instance has coincident or partial-overlap support.
// Keep that finite proof beside primitive reachability so ordinary triangle
// scenes can record the singleton exact resolver without the completion
// lookup, binary search, and alias loop.
struct SceneTraversalStagePlan {
    ScenePrimitiveStagePlan primitives{};
    bool triangle_completion{};

    // Completion is a refinement of triangle traversal, never an
    // independent stage. Canonicalizing at every public construction
    // boundary keeps the finite plan domain closed even for manually built
    // test/plugin plans.
    [[nodiscard]] constexpr SceneTraversalStagePlan
    canonicalized() const noexcept {
        auto result = *this;
        result.triangle_completion &= result.primitives.triangles;
        return result;
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

[[nodiscard]] constexpr SceneTraversalStagePlan
make_scene_traversal_stage_plan(
    std::size_t triangle_geometry_count,
    std::size_t curve_geometry_count,
    std::size_t completion_source_dense_count,
    std::size_t completion_source_sparse_count) noexcept {
    const auto primitives =
        make_scene_primitive_stage_plan(
            triangle_geometry_count,
            curve_geometry_count);
    return SceneTraversalStagePlan{
        .primitives = primitives,
        .triangle_completion =
            primitives.triangles &&
            (completion_source_dense_count != 0u ||
             completion_source_sparse_count != 0u)}
        .canonicalized();
}

}// namespace psycles::luisa_backend::detail
