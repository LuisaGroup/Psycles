#include "path_tracer_subsurface_scene.h"

#include "path_tracer_scene_geometry.h"

namespace psycles::luisa_backend::detail {

bool SubsurfaceScenePlan::contains(
    std::uint32_t primary_instance) const noexcept {
    return primary_instance < triangle_instance_mask.size() &&
           triangle_instance_mask[primary_instance] != 0u;
}

SubsurfaceScenePlan SubsurfaceSceneComponent::plan(
    const contract::SceneSnapshot &snapshot,
    const std::set<contract::MaterialId> &
        bssrdf_materials) const {
    SubsurfaceScenePlan result;
    result.triangle_instance_mask.resize(snapshot.instances.size(), 0u);
    const auto selected =
        collect_triangle_instances_with_surface_materials(
            snapshot, bssrdf_materials);
    for (const auto primary_instance : selected) {
        if (primary_instance < result.triangle_instance_mask.size()) {
            result.triangle_instance_mask[primary_instance] = 1u;
        }
    }
    result.triangle_instance_count =
        static_cast<std::uint32_t>(selected.size());
    return result;
}

void SubsurfaceSceneComponent::initialize_accel(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SubsurfaceScenePlan &plan) const {
    scene->subsurface_instance_count =
        plan.triangle_instance_count;
    if (scene->has_subsurface) {
        LUISA_INFO(
            "Psycles BSSRDF local-intersection domain: triangle_instances={}/{}.",
            plan.triangle_instance_count,
            plan.triangle_instance_mask.size());
    }
    if (plan.triangle_instance_count != 0u) {
        scene->subsurface_accel.emplace(
            scene->device.create_accel());
    }
}

void SubsurfaceSceneComponent::append_triangle_instance(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SubsurfaceScenePlan &plan,
    std::uint32_t primary_instance,
    std::uint32_t geometry_index,
    const Mat4f &transform) const {
    if (!plan.contains(primary_instance)) {
        return;
    }
    LUISA_ASSERT(
        scene->subsurface_accel &&
            geometry_index < scene->geometries.size(),
        "Invalid BSSRDF local-intersection scene plan.");
    // Cycles local intersection ignores normal path visibility and traverses
    // the complete object BLAS. Retain programmable candidates for reservoir
    // and self filtering, and map the compact TLAS instance back to the
    // primary instance ordinal through its user id.
    scene->subsurface_accel->emplace_back(
        scene->geometries[geometry_index].mesh,
        to_luisa(transform),
        std::uint8_t{0xffu},
        false,
        primary_instance);
}

}// namespace psycles::luisa_backend::detail
