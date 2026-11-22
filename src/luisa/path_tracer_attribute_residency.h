#pragma once

#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>

namespace psycles::luisa_backend::detail {

struct SurfaceAttributeDemand {
    bool all{};
    std::set<std::uint64_t> ids;

    [[nodiscard]] bool contains(std::uint64_t id) const noexcept;
};

struct GeometryAttributeResidency {
    SurfaceAttributeDemand demand;

    [[nodiscard]] bool contains(std::uint64_t id) const noexcept {
        return demand.contains(id);
    }
};

struct SceneAttributeResidencyPlan {
    std::map<contract::GeometryId, GeometryAttributeResidency> geometries;
    std::size_t source_binding_count{};
    std::size_t resident_binding_count{};
    std::uint64_t source_device_bytes{};
    std::uint64_t resident_device_bytes{};

    [[nodiscard]] const GeometryAttributeResidency &geometry(
        contract::GeometryId id) const noexcept;
};

// Native shader requests use Cycles' scene-assigned names/standard IDs, not
// the contract's transport hashes. Project those requests onto the source
// upload slots (including MikkTSpace inputs). The union is Geometry::used_shaders,
// including unused slots and overrides, matching the native attribute-map packer.
[[nodiscard]] SceneAttributeResidencyPlan
build_scene_attribute_residency_plan(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t> &material_shader_indices);

}// namespace psycles::luisa_backend::detail
