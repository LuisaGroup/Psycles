#pragma once

#include "path_tracer_internal.h"

#include <memory>
#include <string>

namespace psycles::luisa_backend::detail {

[[nodiscard]] std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic);

void upload_cycles_svm_runtime(Stream &stream,
                               CyclesSvmRuntime &runtime) noexcept;

// Completes the geometry-owned DeviceScene image only after mesh displacement
// has finalized the source uploads. A false result leaves `runtime.geometry`
// null, so no caller can observe a partially allocated typed table set.
[[nodiscard]] bool finalize_cycles_svm_geometry_runtime(
    const std::shared_ptr<LuisaSceneData> &scene,
    const contract::SceneSnapshot &snapshot,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets,
    std::string &diagnostic);

void upload_cycles_svm_geometry_runtime(Stream &stream,
                                        CyclesSvmRuntime &runtime) noexcept;

} // namespace psycles::luisa_backend::detail
