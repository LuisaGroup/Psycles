#include "path_tracer_scene_upload.h"

#include "cycles_shader_identity.h"
#include <cstdint>
#include <limits>

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

SceneTraversalTableBuildResult
build_scene_traversal_tables(SceneTraversalTableBuildInput input) {
    SceneTraversalTableBuildResult result;
    constexpr auto uint_maximum =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    if (input.geometry_materials.size() > uint_maximum ||
        input.override_materials.size() > uint_maximum ||
        input.instances.size() > uint_maximum) {
        result.diagnostic =
            "Scene traversal table exceeds the 32-bit device address space.";
        return result;
    }
    const auto material_flag_count =
        input.geometry_materials.size() + input.override_materials.size();
    if (material_flag_count > uint_maximum) {
        result.diagnostic =
            "Scene traversal table exceeds the 32-bit device address space.";
        return result;
    }

    result.material_flags.reserve(material_flag_count);
    for (const auto &binding : input.geometry_materials) {
        result.material_flags.emplace_back(binding.flags);
    }
    for (const auto &binding : input.override_materials) {
        result.material_flags.emplace_back(binding.flags);
    }

    const auto base_material_count =
        static_cast<std::uint32_t>(input.geometry_materials.size());
    result.instances.reserve(input.instances.size());
    for (auto instance_index = std::size_t{0u};
         instance_index < input.instances.size(); ++instance_index) {
        const auto &instance = input.instances[instance_index];
        if (instance.geometry_index >= input.geometries.size()) {
            result.diagnostic =
                "Scene traversal instance references an unavailable geometry.";
            return result;
        }
        const auto &geometry = input.geometries[instance.geometry_index];
        const auto geometry_material_count =
            std::max(geometry.material_count, 1u);
        if (geometry.material_offset > input.geometry_materials.size() ||
            geometry_material_count >
                input.geometry_materials.size() - geometry.material_offset) {
            result.diagnostic =
                "Scene traversal geometry material range is out of bounds.";
            return result;
        }
        if (instance.override_offset > input.override_materials.size() ||
            instance.override_count >
                input.override_materials.size() - instance.override_offset) {
            result.diagnostic =
                "Scene traversal instance override range is out of bounds.";
            return result;
        }
        if (instance.cycles_primitive_offset !=
            geometry.cycles_primitive_offset) {
            result.diagnostic =
                "Scene traversal instance and geometry primitive offsets "
                "disagree.";
            return result;
        }
        if (geometry.primitive_kind >
                scene_traversal_primitive_kind_mask ||
            geometry.curve_subdivision_level >
                scene_traversal_curve_subdivision_maximum) {
            result.diagnostic =
                "Scene traversal primitive metadata cannot be packed exactly.";
            return result;
        }

        const auto cycles_object_index =
            instance.cycles_object_index !=
                    cycles_shader_identity::invalid_index
                ? instance.cycles_object_index
                : static_cast<std::uint32_t>(instance_index);
        result.instances.emplace_back(SceneTraversalInstanceGpu{
            .bindless_base = geometry.bindless_base,
            .geometry_material_offset = geometry.material_offset,
            .geometry_material_count = geometry_material_count,
            .override_material_offset =
                base_material_count + instance.override_offset,
            .override_material_count = instance.override_count,
            .cycles_object_index = cycles_object_index,
            .cycles_primitive_offset = instance.cycles_primitive_offset,
            .primitive_kind_and_curve_subdivision =
                pack_scene_traversal_primitive(
                    geometry.primitive_kind,
                    geometry.curve_subdivision_level)});
    }
    return result;
}

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
    auto traversal_tables = build_scene_traversal_tables(
        {.geometries = std::span<const GeometryGpu>{
             input.geometries.data(), input.geometries.size()},
         .instances = std::span<const InstanceGpu>{
             input.instances.data(), input.instances.size()},
         .geometry_materials = std::span<const MaterialBindingGpu>{
             input.geometry_materials.data(), input.geometry_materials.size()},
         .override_materials = std::span<const MaterialBindingGpu>{
             input.override_materials.data(), input.override_materials.size()}});
    if (!traversal_tables.ok()) {
        return {.diagnostic = std::move(traversal_tables.diagnostic)};
    }
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
    scene->traversal_instance_buffer =
        scene->device.create_buffer<SceneTraversalInstanceGpu>(
            traversal_tables.instances.size());
    scene->traversal_material_flags_buffer =
        scene->device.create_buffer<luisa::uint>(
            traversal_tables.material_flags.size());
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
           << scene->traversal_instance_buffer.copy_from(
                  luisa::span{traversal_tables.instances})
           << scene->traversal_material_flags_buffer.copy_from(
                  luisa::span{traversal_tables.material_flags})
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
