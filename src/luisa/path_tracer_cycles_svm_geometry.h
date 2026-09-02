#pragma once

#include "cycles_svm_scene_image.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>
#include <psycles/compiler/cycles_svm_object_scene.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <map>
#include <span>

namespace psycles::luisa_backend::detail {

struct GeometryUpload;
struct CyclesInstanceIntersectionPlan;

// Build after displacement has finalized `mesh_uploads`. The source object
// domain determines geometry order, shader request vectors are merged with
// Cycles' first-insertion-wins set algebra, and all typed arrays are committed
// only if every interval and source relation is valid.
[[nodiscard]] CyclesSvmGeometrySceneImage build_cycles_svm_geometry_scene_image(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const compiler::cycles_svm::ObjectIdentityPlan &object_identities,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets);

} // namespace psycles::luisa_backend::detail
