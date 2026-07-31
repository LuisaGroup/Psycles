#include "path_tracer_volume_majorant_scene.h"

#include "cycles_shader_identity.h"
#include "path_kernel_volume_point.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/volume_majorant_prepass.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::optional<VolumeMajorantBounds>
object_bounds(
    const contract::TriangleMeshDesc &geometry)
    noexcept {
    if (geometry.positions.empty()) {
        return std::nullopt;
    }
    auto minimum = geometry.positions.front();
    auto maximum = geometry.positions.front();
    for (const auto point : geometry.positions) {
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    }
    if (minimum == maximum) {
        // Cycles skips only an entirely collapsed mesh bound. One or two
        // degenerate axes are retained and become the root-majorant path.
        return std::nullopt;
    }
    return VolumeMajorantBounds{
        .minimum =
            {minimum.x, minimum.y, minimum.z},
        .maximum =
            {maximum.x, maximum.y, maximum.z}};
}

[[nodiscard]] float cycles_volume_scale(
    const Mat4f &transform) noexcept {
    constexpr auto inverse_sqrt_three =
        0.57735026918962576451f;
    const auto &e = transform.elements;
    const Vec3f transformed{
        (e[0u] + e[4u] + e[8u]) *
            inverse_sqrt_three,
        (e[1u] + e[5u] + e[9u]) *
            inverse_sqrt_three,
        (e[2u] + e[6u] + e[10u]) *
            inverse_sqrt_three};
    return std::sqrt(
        transformed.x * transformed.x +
        transformed.y * transformed.y +
        transformed.z * transformed.z);
}

[[nodiscard]] std::set<contract::MaterialId>
effective_materials(
    const contract::TriangleMeshDesc &geometry,
    const contract::InstanceDesc &instance) {
    std::set<contract::MaterialId> result;
    const auto slot_count =
        std::max(
            geometry.material_slots.size(),
            instance.material_overrides.size());
    for (auto slot = std::size_t{0u};
         slot < slot_count;
         ++slot) {
        if (slot <
            instance.material_overrides.size()) {
            result.emplace(
                instance.material_overrides[slot]);
        } else if (
            !geometry.material_slots.empty()) {
            result.emplace(
                geometry.material_slots[
                    std::min(
                        slot,
                        geometry.material_slots.size() -
                            1u)]);
        }
    }
    return result;
}

[[nodiscard]] bool finite(
    const VolumeMajorantBounds &bounds) noexcept {
    return
        std::isfinite(bounds.minimum.x) &&
        std::isfinite(bounds.minimum.y) &&
        std::isfinite(bounds.minimum.z) &&
        std::isfinite(bounds.maximum.x) &&
        std::isfinite(bounds.maximum.y) &&
        std::isfinite(bounds.maximum.z);
}

[[nodiscard]] bool finite(
    const Mat4f &transform) noexcept {
    return std::all_of(
        transform.elements.begin(),
        transform.elements.end(),
        [](float value) noexcept {
            return std::isfinite(value);
        });
}

[[nodiscard]] std::string root_label(
    const VolumeMajorantSceneRoot &root) {
    return "volume majorant root for material " +
           std::to_string(root.material.value) +
           " and range " +
           std::to_string(root.range_index);
}

[[nodiscard]] std::optional<std::string>
validate_local_hierarchy(
    const VolumeMajorantSceneRoot &planned,
    const VolumeMajorantHierarchy &source) {
    const auto label = root_label(planned);
    if (source.nodes.empty() ||
        source.root.node >= source.nodes.size()) {
        return label +
               " has an invalid local root";
    }
    if (source.nodes[source.root.node].parent != -1) {
        return label +
               " root has a parent";
    }

    std::vector<std::uint8_t> visited(
        source.nodes.size(), 0u);
    std::vector<std::uint32_t> pending{
        source.root.node};
    auto visited_count = std::size_t{0u};
    while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        if (visited[index] != 0u) {
            return label +
                   " is cyclic or shares a child";
        }
        visited[index] = 1u;
        ++visited_count;

        const auto &node = source.nodes[index];
        if (!std::isfinite(node.sigma_minimum) ||
            !std::isfinite(node.sigma_maximum) ||
            node.sigma_minimum < 0.0f ||
            node.sigma_maximum <
                node.sigma_minimum) {
            return label +
                   " has invalid node extrema";
        }
        if (node.parent < -1 ||
            node.first_child < -1) {
            return label +
                   " uses a negative index other than the "
                   "tree sentinel";
        }
        if (node.first_child < 0) {
            continue;
        }

        const auto first_child =
            static_cast<std::size_t>(
                node.first_child);
        constexpr auto child_count =
            std::size_t{8u};
        if (first_child >
                source.nodes.size() ||
            source.nodes.size() -
                    first_child <
                child_count) {
            return label +
                   " has an incomplete octree child block";
        }
        for (auto child = std::size_t{0u};
             child < child_count;
             ++child) {
            const auto child_index =
                first_child + child;
            if (source.nodes[child_index].parent !=
                static_cast<std::int32_t>(
                    index)) {
                return label +
                       " has a child with a mismatched "
                       "parent";
            }
            pending.emplace_back(
                static_cast<std::uint32_t>(
                    child_index));
        }
    }
    if (visited_count != source.nodes.size()) {
        return label +
               " contains nodes unreachable from its root";
    }
    return std::nullopt;
}

}// namespace

VolumeMajorantScenePlan
VolumeMajorantSceneComponent::plan(
    const SceneSnapshot &snapshot,
    const std::map<
        contract::MaterialId,
        VolumeMajorantSceneMaterial>
        &materials) const {
    VolumeMajorantScenePlan result;
    const auto instance_count =
        snapshot.instances.size();
    if (instance_count >=
        static_cast<std::size_t>(
            invalid_volume_identity)) {
        result.diagnostic =
            "volume majorant instance range count exceeds "
            "the 32-bit device contract";
        return result;
    }
    result.world_range =
        static_cast<std::uint32_t>(
            instance_count);
    result.ranges.resize(instance_count + 1u);

    auto append_material =
        [&](contract::MaterialId material_id,
            std::uint32_t range_index,
            std::uint32_t object,
            std::uint32_t instance_id,
            const VolumeMajorantBounds &bounds,
            const Mat4f &object_to_world,
            float volume_scale) -> bool {
            const auto material =
                materials.find(material_id);
            if (material == materials.end() ||
                !material->second.has_volume) {
                return true;
            }
            if (result.roots.size() >=
                static_cast<std::size_t>(
                    invalid_volume_identity)) {
                result.diagnostic =
                    "volume majorant root count exceeds "
                    "the 32-bit device contract";
                return false;
            }
            result.roots.emplace_back(
                VolumeMajorantSceneRoot{
                    .material = material_id,
                    .range_index = range_index,
                    .object = object,
                    .shader =
                        material->second.shader &
                        cycles_shader_identity::
                            shader_mask,
                    .surface_tag =
                        material->second.surface_tag,
                    .parameter_block =
                        material->second.parameter_block,
                    .instance_id = instance_id,
                    .bounds = bounds,
                    .object_to_world =
                        to_luisa(object_to_world),
                    .volume_scale = volume_scale,
                    .heterogeneous =
                        material->second
                            .heterogeneous});
            return true;
        };

    auto instance_index = std::uint32_t{0u};
    for (const auto &[instance_id, instance] :
         snapshot.instances) {
        static_cast<void>(instance_id);
        auto &range = result.ranges[instance_index];
        range.offset =
            static_cast<std::uint32_t>(
                result.roots.size());
        const auto geometry =
            snapshot.geometries.find(
                instance.geometry);
        if (geometry !=
            snapshot.geometries.end()) {
            const auto bounds =
                object_bounds(geometry->second);
            if (bounds) {
                const auto volume_scale =
                    cycles_volume_scale(
                        instance.transform);
                if (!finite(*bounds) ||
                    !finite(instance.transform) ||
                    !std::isfinite(volume_scale)) {
                    result.diagnostic =
                        "volume majorant object bounds, "
                        "transform, and scale must be finite";
                    return result;
                }
                const auto object =
                    instance.cycles_object_index
                        .value_or(instance_index);
                for (const auto material :
                     effective_materials(
                         geometry->second,
                         instance)) {
                    if (!append_material(
                            material,
                            instance_index,
                            object,
                            instance_index,
                            *bounds,
                            instance.transform,
                            volume_scale)) {
                        return result;
                    }
                }
            }
        }
        range.count =
            static_cast<std::uint32_t>(
                result.roots.size()) -
            range.offset;
        ++instance_index;
    }

    auto &world =
        result.ranges[result.world_range];
    world.offset =
        static_cast<std::uint32_t>(
            result.roots.size());
    if (snapshot.world_shader) {
        constexpr VolumeMajorantBounds world_bounds{
            .minimum =
                {-10000.0f, -10000.0f, -10000.0f},
            .maximum =
                {10000.0f, 10000.0f, 10000.0f}};
        if (!append_material(
                *snapshot.world_shader,
                result.world_range,
                snapshot
                    .cycles_background_object_index
                    .value_or(
                        invalid_volume_identity),
                invalid_volume_identity,
                world_bounds,
                Mat4f{},
                1.0f)) {
            return result;
        }
    }
    world.count =
        static_cast<std::uint32_t>(
            result.roots.size()) -
        world.offset;
    return result;
}

VolumeMajorantSceneFlattened
VolumeMajorantSceneComponent::flatten(
    const VolumeMajorantScenePlan &plan,
    std::span<
        const VolumeMajorantHierarchy>
        hierarchies) const {
    VolumeMajorantSceneFlattened result;
    if (!plan.ok()) {
        result.diagnostic = plan.diagnostic;
        return result;
    }
    if (plan.roots.size() >
            static_cast<std::size_t>(
                invalid_volume_identity) ||
        plan.ranges.empty() ||
        plan.world_range !=
            plan.ranges.size() - 1u) {
        result.diagnostic =
            "volume majorant plan violates the root/range "
            "device contract";
        return result;
    }
    if (hierarchies.size() !=
        plan.roots.size()) {
        result.diagnostic =
            "volume majorant hierarchy count does not "
            "match the planned roots";
        return result;
    }
    result.ranges = plan.ranges;
    auto range_cursor = std::uint64_t{0u};
    for (auto range_index = std::size_t{0u};
         range_index < result.ranges.size();
         ++range_index) {
        const auto range =
            result.ranges[range_index];
        const auto end =
            static_cast<std::uint64_t>(
                range.offset) +
            range.count;
        if (range.offset != range_cursor ||
            end > plan.roots.size()) {
            result.diagnostic =
                "volume majorant root ranges do not form "
                "an ordered partition";
            return result;
        }
        for (auto root_index = range_cursor;
             root_index < end;
             ++root_index) {
            if (plan.roots[
                    static_cast<std::size_t>(
                        root_index)]
                    .range_index !=
                range_index) {
                result.diagnostic =
                    "volume majorant root identity does not "
                    "match its range partition";
                return result;
            }
        }
        range_cursor = end;
    }
    if (range_cursor != plan.roots.size()) {
        result.diagnostic =
            "volume majorant root ranges do not cover the "
            "planned root array";
        return result;
    }

    for (auto root_index = std::size_t{0u};
         root_index < hierarchies.size();
         ++root_index) {
        const auto &source =
            hierarchies[root_index];
        const auto node_offset =
            result.nodes.size();
        constexpr auto maximum_node_count =
            static_cast<std::size_t>(
                std::numeric_limits<
                    std::int32_t>::max());
        if (node_offset >
                maximum_node_count ||
            source.nodes.size() >
                maximum_node_count -
                    node_offset) {
            result.diagnostic =
                "flattened volume majorant nodes exceed "
                "the signed 32-bit Cycles index contract";
            return result;
        }
        const auto new_node_count =
            node_offset +
            source.nodes.size();
        if (const auto diagnostic =
                validate_local_hierarchy(
                    plan.roots[root_index],
                    source)) {
            result.diagnostic = *diagnostic;
            return result;
        }

        auto root = source.root;
        root.node +=
            static_cast<std::uint32_t>(
                node_offset);
        root.shader =
            plan.roots[root_index].shader;
        result.roots.emplace_back(root);
        result.nodes.reserve(new_node_count);
        for (auto node : source.nodes) {
            const auto adjust =
                [&](std::int32_t index)
                -> std::optional<std::int32_t> {
                    if (index < 0) {
                        return -1;
                    }
                    if (static_cast<std::size_t>(
                            index) >=
                        source.nodes.size()) {
                        return std::nullopt;
                    }
                    return static_cast<std::int32_t>(
                        node_offset +
                        static_cast<std::size_t>(
                            index));
                };
            const auto parent =
                adjust(node.parent);
            const auto first_child =
                adjust(node.first_child);
            if (!parent || !first_child) {
                result.diagnostic =
                    root_label(
                        plan.roots[root_index]) +
                    " contains an out-of-range node link";
                return result;
            }
            node.parent = *parent;
            node.first_child = *first_child;
            result.nodes.emplace_back(node);
        }
    }
    return result;
}

VolumeMajorantSceneBuildResult
VolumeMajorantSceneComponent::build(
    const std::shared_ptr<LuisaSceneData> &scene,
    Stream &stream,
    const VolumeMajorantScenePlan &plan) const {
    VolumeMajorantSceneBuildResult result;
    if (!scene) {
        result.diagnostic =
            "volume majorant scene resource is null";
        return result;
    }
    if (!plan.ok()) {
        result.diagnostic = plan.diagnostic;
        return result;
    }
    scene->volume_majorant_world_range =
        plan.world_range;
    scene->volume_majorant_root_count = 0u;
    scene->volume_majorant_node_count = 0u;
    scene->volume_majorant_range_count = 0u;
    if (plan.roots.empty()) {
        return result;
    }

    const auto extrema_count =
        VolumeMajorantHierarchyBuilder::
            required_extrema_count();
    auto extrema_buffer =
        scene->device.create_buffer<luisa::float2>(
            extrema_count);
    luisa::vector<luisa::float2> readback(
        extrema_count);
    auto points =
        make_scene_volume_stack_entry_point_provider(
            scene);
    Kernel1D evaluate =
        [scene, points](
            luisa::compute::BufferVar<
                luisa::float2> output,
            UInt object,
            UInt shader,
            UInt surface_tag,
            UInt parameter_block,
            UInt instance_id,
            Float3 grid_minimum,
            Float3 grid_maximum,
            luisa::compute::Float4x4
                object_to_world,
            UInt grid_resolution) noexcept {
            set_block_size(64u, 1u, 1u);
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            VolumeMajorantPrepass prepass{
                scene->surfaces, *points};
            const VolumeStackEntry entry{
                .object = object,
                .shader = shader,
                .surface_tag = surface_tag,
                .parameter_block =
                    parameter_block,
                .instance_id = instance_id,
                .sample_method =
                    volume_sample_distance,
                .valid = true};
            const VolumeMajorantGrid grid{
                .minimum = grid_minimum,
                .maximum = grid_maximum,
                .object_to_world =
                    object_to_world,
                .resolution =
                    grid_resolution};
            const auto extrema =
                prepass.evaluate_cell(
                    entry,
                    services,
                    grid,
                    dispatch_x());
            output.write(
                dispatch_x(),
                make_float2(
                    extrema.minimum,
                    extrema.maximum));
        };
    luisa::compute::ShaderOption shader_options;
    shader_options.enable_cache = true;
    shader_options.enable_fast_math = false;
    auto shader =
        scene->device.compile(
            evaluate, shader_options);

    std::vector<VolumeMajorantHierarchy>
        hierarchies;
    hierarchies.reserve(plan.roots.size());
    std::vector<VolumeMajorantExtrema>
        extrema(extrema_count);
    const VolumeMajorantHierarchyBuilder builder;
    for (const auto &root : plan.roots) {
        const auto resolution =
            root.heterogeneous
                ? volume_majorant_grid_resolution
                : volume_majorant_homogeneous_resolution;
        const auto root_extrema_count =
            VolumeMajorantHierarchyBuilder::
                required_extrema_count(
                    resolution);
        stream
            << shader(
                   extrema_buffer,
                   root.object,
                   root.shader,
                   root.surface_tag,
                   root.parameter_block,
                   root.instance_id,
                   root.bounds.minimum,
                   root.bounds.maximum,
                   root.object_to_world,
                   resolution)
                   .dispatch(root_extrema_count)
            << extrema_buffer
                   .view(
                       0u,
                       root_extrema_count)
                   .copy_to(
                       luisa::span{
                           readback.data(),
                           root_extrema_count})
            << synchronize();
        for (auto index = std::size_t{0u};
             index < root_extrema_count;
             ++index) {
            extrema[index] = {
                .minimum =
                    readback[index].x,
                .maximum =
                    readback[index].y};
        }
        auto built =
            builder.build(
                root.bounds,
                std::span{
                    extrema.data(),
                    root_extrema_count},
                root.volume_scale,
                resolution);
        if (!built.ok()) {
            result.diagnostic =
                root_label(root) + ": " +
                built.diagnostic;
            return result;
        }
        hierarchies.emplace_back(
            std::move(built.hierarchy));
    }

    auto flattened =
        flatten(plan, hierarchies);
    if (!flattened.ok()) {
        result.diagnostic =
            std::move(flattened.diagnostic);
        return result;
    }
    scene->volume_majorant_node_buffer =
        scene->device.create_buffer<
            VolumeMajorantNodeGpu>(
            flattened.nodes.size());
    scene->volume_majorant_root_buffer =
        scene->device.create_buffer<
            VolumeMajorantRootGpu>(
            flattened.roots.size());
    scene->volume_majorant_range_buffer =
        scene->device.create_buffer<
            VolumeMajorantRootRangeGpu>(
            flattened.ranges.size());
    stream
        << scene->volume_majorant_node_buffer
               .copy_from(
                   luisa::span{
                       flattened.nodes})
        << scene->volume_majorant_root_buffer
               .copy_from(
                   luisa::span{
                       flattened.roots})
        << scene->volume_majorant_range_buffer
               .copy_from(
                   luisa::span{
                       flattened.ranges})
        << synchronize();
    scene->volume_majorant_root_count =
        static_cast<std::uint32_t>(
            flattened.roots.size());
    scene->volume_majorant_node_count =
        static_cast<std::uint32_t>(
            flattened.nodes.size());
    scene->volume_majorant_range_count =
        static_cast<std::uint32_t>(
            flattened.ranges.size());
    result.root_count =
        scene->volume_majorant_root_count;
    result.node_count =
        scene->volume_majorant_node_count;
    result.range_count =
        scene->volume_majorant_range_count;
    return result;
}

}// namespace psycles::luisa_backend::detail
