#pragma once

#include "cycles_svm_scene_image.h"

#include <map>
#include <span>

namespace psycles::luisa_backend::detail {

struct GeometryUpload;
struct CyclesInstanceIntersectionPlan;

// Finalizes the dense Cycles object tables from the same post-displacement
// geometry transaction that produced `geometry_image`. Every positional input
// is accompanied by a scene identity map; no resource or source-object ordinal
// is inferred from container order.
[[nodiscard]] CyclesSvmObjectSceneImage build_cycles_svm_object_scene_image(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::ObjectIdentityPlan &object_identities,
    const compiler::cycles_svm::ParticleTableImage &particles,
    const CyclesSvmGeometrySceneImage &geometry_image,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices);

} // namespace psycles::luisa_backend::detail
