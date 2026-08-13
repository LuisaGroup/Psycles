#include "path_tracer_scene_upload.h"

#include "cycles_shader_identity.h"
#include <cstdint>

namespace psycles::luisa_backend::detail {
namespace {

void provide_inert_storage(SceneTableUploadInput input) {
    if (input.light_distribution.empty()) {
        input.light_distribution.emplace_back(LightDistributionGpu{});
    }
    if (input.light_tree_nodes.empty()) {
        input.light_tree_nodes.emplace_back(LightTreeNodeGpu{});
    }
    if (input.light_tree_emitters.empty()) {
        input.light_tree_emitters.emplace_back(LightTreeEmitterGpu{});
    }
    if (input.light_tree_emitter_mappings.empty()) {
        input.light_tree_emitter_mappings.emplace_back(
            ~luisa::uint{0u}, ~luisa::uint{0u});
    }
    if (input.light_tree_triangle_lookup.empty()) {
        input.light_tree_triangle_lookup.emplace_back(
            ~luisa::uint{0u},
            ~luisa::uint{0u},
            ~luisa::uint{0u},
            0u);
    }
    if (input.lights.empty()) {
        input.lights.emplace_back(LightGpu{});
    }
    if (input.emissive_triangles.empty()) {
        input.emissive_triangles.emplace_back(EmissiveTriangleGpu{});
    }
    if (input.geometry_materials.empty()) {
        input.geometry_materials.emplace_back(MaterialBindingGpu{
            .cycles_shader_index = cycles_shader_identity::invalid_index});
    }
    if (input.override_materials.empty()) {
        input.override_materials.emplace_back(MaterialBindingGpu{
            .cycles_shader_index = cycles_shader_identity::invalid_index});
    }
    // Empty-world renders are valid Cycles scenes. Luisa buffers cannot be
    // zero-sized; these records stay unreachable without a committed hit.
    if (input.geometries.empty()) {
        input.geometries.emplace_back(GeometryGpu{});
    }
    if (input.attribute_bindings.empty()) {
        input.attribute_bindings.emplace_back(AttributeBindingGpu{});
    }
    if (input.attribute_ranges.empty()) {
        input.attribute_ranges.emplace_back(AttributeRangeGpu{});
    }
    if (input.instances.empty()) {
        input.instances.emplace_back(InstanceGpu{});
    }
}

}// namespace

std::uint32_t encode_attribute_domain(
    contract::MeshAttributeDomain domain) noexcept {
    switch (domain) {
        case contract::MeshAttributeDomain::point:
            return attribute_domain_point;
        case contract::MeshAttributeDomain::corner:
            return attribute_domain_corner;
        case contract::MeshAttributeDomain::face:
            return attribute_domain_face;
    }
    return attribute_domain_point;
}

Vec3f from_luisa(luisa::float3 value) noexcept {
    return {value.x, value.y, value.z};
}

SceneTableUploadResult
SceneTableUploadComponent::upload(const std::shared_ptr<LuisaSceneData> &scene,
                                  Stream &stream,
                                  SceneTableUploadInput input) const {
    provide_inert_storage(input);
    scene->geometry_buffer =
        scene->device.create_buffer<GeometryGpu>(input.geometries.size());
    if (!scene->attribute_binding_buffer) {
        scene->attribute_binding_buffer =
            scene->device.create_buffer<AttributeBindingGpu>(
                input.attribute_bindings.size());
        scene->heap.emplace_on_update(scene->attribute_binding_slot,
                                      scene->attribute_binding_buffer);
    }
    if (!scene->attribute_range_buffer) {
        scene->attribute_range_buffer =
            scene->device.create_buffer<AttributeRangeGpu>(
                input.attribute_ranges.size());
        scene->heap.emplace_on_update(scene->attribute_range_slot,
                                      scene->attribute_range_buffer);
    }
    scene->instance_buffer =
        scene->device.create_buffer<InstanceGpu>(input.instances.size());
    scene->geometry_material_buffer =
        scene->device.create_buffer<MaterialBindingGpu>(
            input.geometry_materials.size());
    scene->override_material_buffer =
        scene->device.create_buffer<MaterialBindingGpu>(
            input.override_materials.size());
    scene->light_buffer =
        scene->device.create_buffer<LightGpu>(input.lights.size());
    scene->emissive_triangle_buffer =
        scene->device.create_buffer<EmissiveTriangleGpu>(
            input.emissive_triangles.size());
    scene->light_distribution_buffer =
        scene->device.create_buffer<LightDistributionGpu>(
            input.light_distribution.size());
    scene->light_tree_node_buffer =
        scene->device.create_buffer<LightTreeNodeGpu>(
            input.light_tree_nodes.size());
    scene->light_tree_emitter_buffer =
        scene->device.create_buffer<LightTreeEmitterGpu>(
            input.light_tree_emitters.size());
    scene->light_tree_emitter_mapping_buffer =
        scene->device.create_buffer<luisa::uint2>(
            input.light_tree_emitter_mappings.size());
    scene->light_tree_triangle_lookup_buffer =
        scene->device.create_buffer<luisa::uint4>(
            input.light_tree_triangle_lookup.size());

    stream << scene->geometry_buffer.copy_from(luisa::span{input.geometries})
           << scene->attribute_binding_buffer.copy_from(
                  luisa::span{input.attribute_bindings})
           << scene->attribute_range_buffer.copy_from(
                  luisa::span{input.attribute_ranges})
           << scene->instance_buffer.copy_from(luisa::span{input.instances})
           << scene->geometry_material_buffer.copy_from(
                  luisa::span{input.geometry_materials})
           << scene->override_material_buffer.copy_from(
                  luisa::span{input.override_materials})
           << scene->light_buffer.copy_from(luisa::span{input.lights})
           << scene->emissive_triangle_buffer.copy_from(
                  luisa::span{input.emissive_triangles})
           << scene->light_distribution_buffer.copy_from(
                  luisa::span{input.light_distribution})
           << scene->light_tree_node_buffer.copy_from(
                  luisa::span{input.light_tree_nodes})
           << scene->light_tree_emitter_buffer.copy_from(
                  luisa::span{input.light_tree_emitters})
           << scene->light_tree_emitter_mapping_buffer.copy_from(
                  luisa::span{input.light_tree_emitter_mappings})
           << scene->light_tree_triangle_lookup_buffer.copy_from(
                  luisa::span{input.light_tree_triangle_lookup})
           << scene->texture_heap.update() << scene->heap.update()
           << scene->accel.build();
    if (scene->subsurface_accel) {
        stream << scene->subsurface_accel->build();
    }
    stream << synchronize();
    return {};
}

}// namespace psycles::luisa_backend::detail
