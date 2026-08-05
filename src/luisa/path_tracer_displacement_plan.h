#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include <psycles/contract/scene.h>

namespace psycles::luisa_backend::detail {

struct CyclesDisplacementVertexEvaluation {
    std::uint32_t vertex_index{};
    std::uint32_t primitive_index{};
    std::uint32_t corner_index{};
    contract::MaterialId material{};
};

struct CyclesMeshDisplacementPlan {
    std::vector<CyclesDisplacementVertexEvaluation> evaluations;
    // Cycles recomputes vertex normals only for DISPLACEMENT triangles;
    // BOTH intentionally preserves undisplaced normals for automatic bump.
    std::vector<bool> true_displacement_triangles;

    [[nodiscard]] bool empty() const noexcept {
        return evaluations.empty();
    }
};

// Reproduce Cycles GeometryManager::fill_shader_input ownership exactly:
// triangles and their corners are visited in source order, bump-only
// materials are skipped, and the first eligible corner owns each shared
// vertex's one displacement evaluation.
[[nodiscard]] CyclesMeshDisplacementPlan
make_cycles_mesh_displacement_plan(
    const contract::TriangleMeshDesc &geometry,
    const std::map<contract::MaterialId,
                   contract::MaterialDesc> &materials,
    std::span<const contract::MaterialId> material_overrides = {});

}// namespace psycles::luisa_backend::detail
