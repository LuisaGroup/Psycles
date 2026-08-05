#include "path_tracer_displacement_plan.h"

#include <algorithm>
#include <optional>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::optional<contract::MaterialId>
triangle_material(
    const contract::TriangleMeshDesc &geometry,
    std::size_t primitive_index,
    std::span<const contract::MaterialId> material_overrides) noexcept {
    const auto slot =
        primitive_index < geometry.triangle_material_slots.size()
            ? geometry.triangle_material_slots[primitive_index]
            : 0u;
    if (slot < material_overrides.size()) {
        return material_overrides[slot];
    }
    if (geometry.material_slots.empty()) {
        return std::nullopt;
    }
    return geometry.material_slots[std::min<std::size_t>(
        slot, geometry.material_slots.size() - 1u)];
}

}// namespace

CyclesMeshDisplacementPlan make_cycles_mesh_displacement_plan(
    const contract::TriangleMeshDesc &geometry,
    const std::map<contract::MaterialId,
                   contract::MaterialDesc> &materials,
    std::span<const contract::MaterialId> material_overrides) {
    CyclesMeshDisplacementPlan result;
    result.true_displacement_triangles.resize(
        geometry.triangles.size(), false);
    std::vector<bool> done(geometry.positions.size(), false);

    for (std::size_t primitive_index = 0u;
         primitive_index < geometry.triangles.size();
         ++primitive_index) {
        const auto material_id = triangle_material(
            geometry, primitive_index, material_overrides);
        if (!material_id) {
            continue;
        }
        const auto material = materials.find(*material_id);
        if (material == materials.end() ||
            !contract::uses_true_displacement(
                material->second.displacement_method) ||
            !material->second.shader
                 .root(contract::ShaderDomain::displacement)) {
            continue;
        }
        result.true_displacement_triangles[primitive_index] =
            material->second.displacement_method ==
            contract::DisplacementMethod::displacement;

        const auto &triangle = geometry.triangles[primitive_index];
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            const auto vertex = triangle[corner];
            if (vertex >= done.size() || done[vertex]) {
                continue;
            }
            done[vertex] = true;
            result.evaluations.emplace_back(
                CyclesDisplacementVertexEvaluation{
                    .vertex_index = vertex,
                    .primitive_index =
                        static_cast<std::uint32_t>(primitive_index),
                    .corner_index = corner,
                    .material = *material_id});
        }
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
