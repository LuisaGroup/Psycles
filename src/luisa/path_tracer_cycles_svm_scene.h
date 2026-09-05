#pragma once

#include "path_tracer_internal.h"

#include <memory>
#include <string>

namespace psycles::luisa_backend::detail {

struct CyclesInstanceIntersectionPlan;

// Checked lookup in the compiled Cycles used-shader domain. Missing native
// state is an invalid scene, never permission to consult the legacy cache.
[[nodiscard]] const compiler::cycles_svm::ShaderCompileMetadata &
cycles_svm_material_metadata(const LuisaSceneData &scene,
                             contract::MaterialId material);

// Adapt native KernelShader flags to the existing geometry binding ABI.
// Storage addresses remain renderer-owned; all shader semantics come from
// the same table that will be uploaded for native SVM execution.
[[nodiscard]] MaterialBinding make_cycles_svm_material_binding(
    const LuisaSceneData &scene, contract::MaterialId material,
    std::uint32_t surface_tag, std::uint32_t parameter_block,
    std::uint32_t material_identity);

// Compile the snapshot's Cycles used-shader domain directly from validated
// source graphs. A retained legacy MaterialLibrary is neither a prerequisite
// nor an authority for any shader image produced by this transaction.
[[nodiscard]] std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic);

void upload_cycles_svm_runtime(Stream &stream,
                               CyclesSvmRuntime &runtime) noexcept;

// Completes geometry- and object-owned DeviceScene images only after mesh
// displacement has finalized the source uploads. A false result leaves both
// runtime pointers null, so no caller can observe a partially finalized table.
[[nodiscard]] bool finalize_cycles_svm_scene_runtime(
    const std::shared_ptr<LuisaSceneData> &scene,
    const contract::SceneSnapshot &snapshot,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets,
    std::string &diagnostic);

void upload_cycles_svm_scene_runtime(Stream &stream,
                                     CyclesSvmRuntime &runtime) noexcept;

} // namespace psycles::luisa_backend::detail
