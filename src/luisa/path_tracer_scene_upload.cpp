#include "path_tracer_scene_upload.h"

#include "cycles_shader_identity.h"
#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::vector<luisa::uint2>
build_cycles_object_instance_map(
    const luisa::vector<InstanceGpu> &instances) {
    std::vector<luisa::uint2> result;
    result.reserve(instances.size());
    for (std::size_t index = 0u; index < instances.size(); ++index) {
        const auto explicit_index = instances[index].cycles_object_index;
        const auto object_index =
            explicit_index != cycles_shader_identity::invalid_index
                ? explicit_index
                : static_cast<std::uint32_t>(index);
        result.emplace_back(
            object_index, static_cast<std::uint32_t>(index));
    }
    std::sort(result.begin(), result.end(),
              [](luisa::uint2 a, luisa::uint2 b) noexcept {
                  return a.x < b.x;
              });
    return result;
}

[[nodiscard]] bool object_indices_are_unique(
    const std::vector<luisa::uint2> &mapping) noexcept {
    for (std::size_t index = 1u; index < mapping.size(); ++index) {
        if (mapping[index - 1u].x == mapping[index].x) {
            return false;
        }
    }
    return true;
}

void provide_inert_storage(SceneTableUploadInput input) {
    if (input.light_distribution.empty()) {
        input.light_distribution.emplace_back(LightDistributionGpu{});
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
    if (input.coincident_primitives.empty()) {
        input.coincident_primitives.emplace_back(CoincidentPrimitiveGpu{});
    }
    if (input.coincident_primitive_instances.empty()) {
        input.coincident_primitive_instances.emplace_back(0u);
    }
}

}// namespace

CoincidentPrimitiveUpload make_coincident_primitive_upload(
    const CyclesPrimitiveIntersectionPlan &plan) {
    CoincidentPrimitiveUpload result;
    result.records.reserve(plan.records.size());
    for (const auto &record : plan.records) {
        result.records.emplace_back(CoincidentPrimitiveGpu{
            .local_primitive = record.local_primitive,
            .instance_offset = record.instance_offset,
            .instance_count = record.instance_count});
    }
    result.instances.reserve(plan.instances.size());
    for (const auto instance : plan.instances) {
        result.instances.emplace_back(instance);
    }
    return result;
}

SceneTableUploadResult SceneTableUploadComponent::upload(
    const std::shared_ptr<LuisaSceneData> &scene,
    Stream &stream,
    SceneTableUploadInput input) const {
    provide_inert_storage(input);
    const auto object_instance_map =
        build_cycles_object_instance_map(input.instances);
    if (!object_indices_are_unique(object_instance_map)) {
        return {.diagnostic =
                    "Cycles object identities must be unique across scene "
                    "instances"};
    }
    scene->cycles_object_instance_map_count =
        static_cast<std::uint32_t>(object_instance_map.size());

    scene->geometry_buffer =
        scene->device.create_buffer<GeometryGpu>(input.geometries.size());
    if (!scene->attribute_binding_buffer) {
        scene->attribute_binding_buffer =
            scene->device.create_buffer<AttributeBindingGpu>(
                input.attribute_bindings.size());
        scene->heap.emplace_on_update(
            scene->attribute_binding_slot,
            scene->attribute_binding_buffer);
    }
    if (!scene->attribute_range_buffer) {
        scene->attribute_range_buffer =
            scene->device.create_buffer<AttributeRangeGpu>(
                input.attribute_ranges.size());
        scene->heap.emplace_on_update(
            scene->attribute_range_slot,
            scene->attribute_range_buffer);
    }
    scene->instance_buffer =
        scene->device.create_buffer<InstanceGpu>(input.instances.size());
    scene->coincident_primitive_buffer =
        scene->device.create_buffer<CoincidentPrimitiveGpu>(
            input.coincident_primitives.size());
    scene->coincident_primitive_instance_buffer =
        scene->device.create_buffer<luisa::uint>(
            input.coincident_primitive_instances.size());
    scene->cycles_object_instance_map_buffer =
        scene->device.create_buffer<luisa::uint2>(object_instance_map.size());
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

    stream << scene->geometry_buffer.copy_from(
                  luisa::span{input.geometries})
           << scene->attribute_binding_buffer.copy_from(
                  luisa::span{input.attribute_bindings})
           << scene->attribute_range_buffer.copy_from(
                  luisa::span{input.attribute_ranges})
           << scene->instance_buffer.copy_from(
                  luisa::span{input.instances})
           << scene->coincident_primitive_buffer.copy_from(
                  luisa::span{input.coincident_primitives})
           << scene->coincident_primitive_instance_buffer.copy_from(
                  luisa::span{input.coincident_primitive_instances})
           << scene->cycles_object_instance_map_buffer.copy_from(
                  luisa::span{object_instance_map})
           << scene->geometry_material_buffer.copy_from(
                  luisa::span{input.geometry_materials})
           << scene->override_material_buffer.copy_from(
                  luisa::span{input.override_materials})
           << scene->light_buffer.copy_from(luisa::span{input.lights})
           << scene->emissive_triangle_buffer.copy_from(
                  luisa::span{input.emissive_triangles})
           << scene->light_distribution_buffer.copy_from(
                  luisa::span{input.light_distribution})
           << scene->texture_heap.update()
           << scene->heap.update()
           << scene->accel.build()
           << synchronize();
    return {};
}

}// namespace psycles::luisa_backend::detail
