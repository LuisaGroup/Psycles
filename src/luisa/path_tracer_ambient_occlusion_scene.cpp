#include "path_tracer_ambient_occlusion_scene.h"

#include <array>

namespace psycles::luisa_backend::detail {

AmbientOcclusionSceneComponent::AmbientOcclusionSceneComponent(
    const compiler::cycles_svm::CompiledShaderTable &compilation) noexcept {
    // Native AO is not admitted by the compiler yet. Do not consult an
    // unrelated SurfaceProgram to silently supply raytrace state. Adding the
    // native opcode must also implement its scene/provider boundary here.
    LUISA_ASSERT(!compilation.table.node_types_used[
        compiler::cycles_svm::NODE_AMBIENT_OCCLUSION],
        "Native SVM AO requires its native scene/provider implementation.");
}

void AmbientOcclusionSceneComponent::initialize(
    const std::shared_ptr<LuisaSceneData> &scene) const {
    scene->has_ambient_occlusion = _plan.enabled;
    scene->has_local_ambient_occlusion = _plan.local_requested;
    if (_plan.enabled) {
        scene->ambient_occlusion_distance_buffer.emplace(
            scene->device.create_buffer<float>(1u));
    }
    if (_plan.local_enabled()) {
        LUISA_INFO(
            "Psycles Only Local AO domain: triangle_instances={}.",
            _plan.triangle_instance_count);
        scene->ambient_occlusion_local_accel.emplace(
            scene->device.create_accel(static_scene_accel_option()));
    }
}

void AmbientOcclusionSceneComponent::upload_distance(
    Stream &stream,
    const std::shared_ptr<LuisaSceneData> &scene,
    float distance) const {
    if (!_plan.enabled) {
        return;
    }
    LUISA_ASSERT(
        scene->ambient_occlusion_distance_buffer,
        "AO distance buffer was not initialized.");
    const std::array values{distance};
    stream << scene->ambient_occlusion_distance_buffer->copy_from(
        luisa::span{values});
}

void AmbientOcclusionSceneComponent::append_triangle_instance(
    const std::shared_ptr<LuisaSceneData> &scene,
    std::uint32_t primary_instance,
    std::uint32_t geometry_index,
    const Mat4f &transform) const {
    if (!_plan.local_enabled()) {
        return;
    }
    LUISA_ASSERT(
        scene->ambient_occlusion_local_accel &&
            geometry_index < scene->geometries.size(),
        "Invalid Only Local AO scene plan.");
    scene->ambient_occlusion_local_accel->emplace_back(
        scene->geometries[geometry_index].mesh,
        to_luisa(transform),
        std::uint8_t{0xffu},
        false,
        primary_instance);
}

}// namespace psycles::luisa_backend::detail
