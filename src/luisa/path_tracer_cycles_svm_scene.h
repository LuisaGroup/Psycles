#pragma once

#include "path_tracer_internal.h"

#include <memory>
#include <string>

namespace psycles::luisa_backend::detail {

struct CyclesInstanceIntersectionPlan;

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
