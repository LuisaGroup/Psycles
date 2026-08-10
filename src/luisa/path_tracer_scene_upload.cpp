#include "path_tracer_scene_upload.h"

#include "cycles_shader_identity.h"
#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::uint32_t
cycles_object_index(const InstanceGpu &instance,
                    std::size_t fallback_index) noexcept {
    return instance.cycles_object_index != cycles_shader_identity::invalid_index
               ? instance.cycles_object_index
               : static_cast<std::uint32_t>(fallback_index);
}

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
    if (input.primitive_completions.empty()) {
        input.primitive_completions.emplace_back(PrimitiveCompletionGpu{});
    }
    if (input.primitive_completion_instances.empty()) {
        input.primitive_completion_instances.emplace_back(0u);
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

CyclesCompletionSourceLookup
make_cycles_completion_source_lookup(std::span<const InstanceGpu> instances) {
    CyclesCompletionSourceLookup result;
    std::vector<std::uint32_t> object_indices;
    object_indices.reserve(instances.size());
    result.sparse_instances.reserve(instances.size());
    for (std::size_t index = 0u; index < instances.size(); ++index) {
        const auto object_index = cycles_object_index(instances[index], index);
        object_indices.emplace_back(object_index);
        if (std::max(instances[index].coincident_count, 1u) > 1u ||
            instances[index].primitive_completion_count != 0u) {
            result.sparse_instances.emplace_back(
                object_index, static_cast<std::uint32_t>(index));
        }
    }
    std::sort(object_indices.begin(), object_indices.end());
    if (std::adjacent_find(object_indices.begin(), object_indices.end()) !=
        object_indices.end()) {
        result.diagnostic =
            "Cycles object identities must be unique across scene "
            "instances";
        result.sparse_instances.clear();
        return result;
    }
    std::sort(
        result.sparse_instances.begin(), result.sparse_instances.end(),
        [](luisa::uint2 a, luisa::uint2 b) noexcept { return a.x < b.x; });
    if (result.sparse_instances.empty()) {
        return result;
    }

    constexpr std::uint64_t minimum_dense_entries = 4096u;
    constexpr std::uint64_t maximum_dense_entries = 1u << 20u;
    const auto proportional_entries = std::min<std::uint64_t>(
        maximum_dense_entries,
        static_cast<std::uint64_t>(instances.size()) * 4u);
    const auto dense_limit =
        std::max(minimum_dense_entries, proportional_entries);
    const auto required_entries =
        static_cast<std::uint64_t>(result.sparse_instances.back().x) + 1u;
    if (required_entries <= dense_limit) {
        result.dense_instances.assign(
            static_cast<std::size_t>(required_entries),
            cycles_shader_identity::invalid_index);
        for (const auto entry : result.sparse_instances) {
            result.dense_instances[entry.x] = entry.y;
        }
        result.sparse_instances.clear();
    }
    return result;
}

PrimitiveCompletionUpload
make_primitive_completion_upload(const CyclesPrimitiveCompletionPlan &plan) {
    PrimitiveCompletionUpload result;
    result.records.reserve(plan.records.size());
    for (const auto &record : plan.records) {
        result.records.emplace_back(
            PrimitiveCompletionGpu{.local_primitive = record.local_primitive,
                                   .instance_offset = record.instance_offset,
                                   .instance_count = record.instance_count});
    }
    result.instances.reserve(plan.instances.size());
    for (const auto instance : plan.instances) {
        result.instances.emplace_back(instance);
    }
    return result;
}

SceneTableUploadResult
SceneTableUploadComponent::upload(const std::shared_ptr<LuisaSceneData> &scene,
                                  Stream &stream,
                                  SceneTableUploadInput input) const {
    provide_inert_storage(input);
    const auto completion_source_lookup =
        make_cycles_completion_source_lookup(input.instances);
    if (!completion_source_lookup.ok()) {
        return {.diagnostic = completion_source_lookup.diagnostic};
    }
    scene->cycles_completion_source_dense_count = static_cast<std::uint32_t>(
        completion_source_lookup.dense_instances.size());
    scene->cycles_completion_source_sparse_count = static_cast<std::uint32_t>(
        completion_source_lookup.sparse_instances.size());
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
    scene->primitive_completion_buffer =
        scene->device.create_buffer<PrimitiveCompletionGpu>(
            input.primitive_completions.size());
    scene->primitive_completion_instance_buffer =
        scene->device.create_buffer<luisa::uint>(
            input.primitive_completion_instances.size());
    if (scene->cycles_completion_source_dense_count != 0u) {
        scene->cycles_completion_source_dense_buffer =
            scene->device.create_buffer<luisa::uint>(
                scene->cycles_completion_source_dense_count);
    }
    if (scene->cycles_completion_source_sparse_count != 0u) {
        scene->cycles_completion_source_sparse_buffer =
            scene->device.create_buffer<luisa::uint2>(
                scene->cycles_completion_source_sparse_count);
    }
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

    if (scene->cycles_completion_source_dense_count != 0u) {
        stream << scene->cycles_completion_source_dense_buffer.copy_from(
            luisa::span{completion_source_lookup.dense_instances});
    }
    if (scene->cycles_completion_source_sparse_count != 0u) {
        stream << scene->cycles_completion_source_sparse_buffer.copy_from(
            luisa::span{completion_source_lookup.sparse_instances});
    }
    stream << scene->geometry_buffer.copy_from(luisa::span{input.geometries})
           << scene->attribute_binding_buffer.copy_from(
                  luisa::span{input.attribute_bindings})
           << scene->attribute_range_buffer.copy_from(
                  luisa::span{input.attribute_ranges})
           << scene->instance_buffer.copy_from(luisa::span{input.instances})
           << scene->primitive_completion_buffer.copy_from(
                  luisa::span{input.primitive_completions})
           << scene->primitive_completion_instance_buffer.copy_from(
                  luisa::span{input.primitive_completion_instances})
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
           << scene->accel.build() << synchronize();
    return {};
}

}// namespace psycles::luisa_backend::detail
