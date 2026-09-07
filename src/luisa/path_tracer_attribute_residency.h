#pragma once

#include <psycles/compiler/material_library.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>

namespace psycles::luisa_backend::detail {

// Host/JIT proof domain for named geometry attributes. A finite set denotes
// the exact IDs which a material can query. `all` is the conservative top
// element used when an ID cannot be proven constant before shader execution.
struct SurfaceAttributeDemand {
    bool all{};
    std::set<std::uint64_t> ids;

    [[nodiscard]] bool contains(std::uint64_t id) const noexcept;
    void merge(const SurfaceAttributeDemand &other);
};

[[nodiscard]] SurfaceAttributeDemand collect_surface_attribute_demand(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceParameterBlock &parameters);

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

// Computes the least conservative per-geometry union allowed by the exact
// primitive material image: only slots selected by real triangle primitives,
// resolved through each instance's overrides and Cycles' last-slot clamp, are
// roots. Missing programs inside that reachable image raise the affected
// geometry to the conservative top element rather than making pruning
// unsound.
[[nodiscard]] SceneAttributeResidencyPlan
build_scene_attribute_residency_plan(
    const contract::SceneSnapshot &snapshot,
    const compiler::MaterialLibrary &materials);

// Native shader requests use Cycles' scene-assigned names/standard IDs, not
// the contract's transport hashes. Project those requests onto the source
// upload slots (including MikkTSpace inputs) without invoking SurfaceProgram.
// The union is Geometry::used_shaders, including unused slots and overrides,
// exactly like the native attribute-map packer; it is not primitive reachability.
[[nodiscard]] SceneAttributeResidencyPlan
build_scene_attribute_residency_plan(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t> &material_shader_indices);

}// namespace psycles::luisa_backend::detail
