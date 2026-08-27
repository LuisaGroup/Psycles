#include "path_tracer_volume_metadata.h"

#include <psycles/luisa/volume_stack.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Vec3f transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        e[0u] * point.x +
            e[4u] * point.y +
            e[8u] * point.z +
            e[12u],
        e[1u] * point.x +
            e[5u] * point.y +
            e[9u] * point.z +
            e[13u],
        e[2u] * point.x +
            e[6u] * point.y +
            e[10u] * point.z +
            e[14u]};
}

void grow(
    WorldBounds &bounds,
    Vec3f point) noexcept {
    if (!bounds.valid) {
        bounds.minimum = point;
        bounds.maximum = point;
        bounds.valid = true;
        return;
    }
    bounds.minimum.x =
        std::min(bounds.minimum.x, point.x);
    bounds.minimum.y =
        std::min(bounds.minimum.y, point.y);
    bounds.minimum.z =
        std::min(bounds.minimum.z, point.z);
    bounds.maximum.x =
        std::max(bounds.maximum.x, point.x);
    bounds.maximum.y =
        std::max(bounds.maximum.y, point.y);
    bounds.maximum.z =
        std::max(bounds.maximum.z, point.z);
}

void grow(
    WorldBounds &bounds,
    Vec3f point,
    float radius) noexcept {
    const auto extent = std::max(radius, 0.0f);
    grow(
        bounds,
        Vec3f{
            point.x - extent,
            point.y - extent,
            point.z - extent});
    grow(
        bounds,
        Vec3f{
            point.x + extent,
            point.y + extent,
            point.z + extent});
}

[[nodiscard]] WorldBounds object_bounds(
    const contract::TriangleMeshDesc &geometry,
    const Mat4f &transform) noexcept {
    WorldBounds local;
    for (const auto point : geometry.positions) {
        grow(local, point);
    }
    if (!local.valid) {
        return {};
    }
    WorldBounds world;
    for (const auto x :
         {local.minimum.x, local.maximum.x}) {
        for (const auto y :
             {local.minimum.y, local.maximum.y}) {
            for (const auto z :
                 {local.minimum.z,
                  local.maximum.z}) {
                grow(
                    world,
                    transform_point(
                        transform,
                        Vec3f{x, y, z}));
            }
        }
    }
    return world;
}

[[nodiscard]] bool instance_has_volume(
    const contract::TriangleMeshDesc &geometry,
    const contract::InstanceDesc &instance,
    const std::set<contract::MaterialId>
        &volume_materials) noexcept {
    const auto slot_count =
        std::max(
            geometry.material_slots.size(),
            instance.material_overrides.size());
    for (auto slot = std::size_t{0u};
         slot < slot_count;
         ++slot) {
        std::optional<contract::MaterialId>
            material;
        if (slot <
            instance.material_overrides.size()) {
            material =
                instance.material_overrides[slot];
        } else if (
            !geometry.material_slots.empty()) {
            material =
                geometry.material_slots[
                    std::min(
                        slot,
                        geometry
                                .material_slots
                                .size() -
                            1u)];
        }
        if (material &&
            volume_materials.contains(*material)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] float maximum_aperture_radius(
    const CameraVolumeBoundsQuery &camera)
    noexcept {
    const auto ratio =
        std::max(camera.aperture_ratio, 1.0e-8f);
    return camera.aperture_radius /
           std::min(ratio, 1.0f);
}

[[nodiscard]] WorldBounds camera_bounds(
    const CameraVolumeBoundsQuery &camera)
    noexcept {
    WorldBounds bounds;
    const auto near_clip =
        std::max(camera.near_clip, 0.0f);
    const auto aperture =
        std::max(
            maximum_aperture_radius(camera),
            0.0f);
    if (camera.projection ==
        contract::CameraProjection::panorama) {
        grow(
            bounds,
            transform_point(
                camera.transform,
                Vec3f{}),
            aperture + near_clip);
        return bounds;
    }

    const auto scaled_dof =
        aperture > 0.0f &&
                camera.focal_distance > 0.0f
            ? aperture *
                  (near_clip /
                   camera.focal_distance)
            : 0.0f;
    const auto extend =
        aperture +
        std::max(near_clip, scaled_dof);
    const std::array screen_points{
        Vec2f{-1.0f + 2.0f * camera.shift_x,
              1.0f + 2.0f * camera.shift_y},
        Vec2f{-1.0f + 2.0f * camera.shift_x,
              -1.0f + 2.0f * camera.shift_y},
        Vec2f{1.0f + 2.0f * camera.shift_x,
              -1.0f + 2.0f * camera.shift_y},
        Vec2f{1.0f + 2.0f * camera.shift_x,
              1.0f + 2.0f * camera.shift_y},
        Vec2f{2.0f * camera.shift_x,
              2.0f * camera.shift_y}};
    const auto point_count =
        camera.projection ==
                contract::CameraProjection::
                    perspective
            ? screen_points.size()
            : screen_points.size() - 1u;
    for (auto index = std::size_t{0u};
         index < point_count;
         ++index) {
        const auto screen = screen_points[index];
        Vec3f local;
        if (camera.projection ==
            contract::CameraProjection::
                perspective) {
            local = {
                screen.x *
                    camera.horizontal_tangent *
                    near_clip,
                screen.y *
                    camera.vertical_tangent *
                    near_clip,
                -near_clip};
        } else {
            local = {
                screen.x *
                    camera.orthographic_vertical_span *
                    camera.aspect * 0.5f,
                screen.y *
                    camera.orthographic_vertical_span *
                    0.5f,
                -near_clip};
        }
        grow(
            bounds,
            transform_point(
                camera.transform, local),
            extend);
    }
    return bounds;
}

}// namespace

std::uint32_t
cycles_program_closure_allocation_count(
    const compiler::SurfaceProgram &program) noexcept {
    auto count = std::uint32_t{0u};
    const auto add =
        [&](std::uint32_t amount) noexcept {
            count = std::min(
                cycles_max_closure_allocations,
                count + std::min(
                            amount,
                            cycles_max_closure_allocations -
                                count));
        };
    for (const auto &closure :
         program.closure_instructions()) {
        switch (closure.operation) {
            case compiler::ClosureOperation::
                null_closure:
            case compiler::ClosureOperation::add:
            case compiler::ClosureOperation::mix:
                break;
            case compiler::ClosureOperation::principled:
                add(12u);
                break;
            case compiler::ClosureOperation::diffuse:
            case compiler::ClosureOperation::translucent:
            case compiler::ClosureOperation::glass:
            case compiler::ClosureOperation::refraction:
            case compiler::ClosureOperation::sheen_microfiber:
            case compiler::ClosureOperation::sheen_ashikhmin:
            case compiler::ClosureOperation::emission:
            case compiler::ClosureOperation::transparent:
                add(1u);
                break;
            case compiler::ClosureOperation::subsurface:
                // Cycles reserves the BSSRDF plus per-channel diffuse
                // fallback storage for radii below BSSRDF_MIN_RADIUS.
                add(3u);
                break;
            case compiler::ClosureOperation::glossy:
                // Cycles counts MULTI_GGX as a two-closure family. The
                // lowered instruction retains this static Distribution
                // property as preserve_ggx_energy.
                add(
                    closure.preserve_ggx_energy
                        ? 2u
                        : 1u);
                break;
            case compiler::ClosureOperation::metallic_f82:
            case compiler::ClosureOperation::metallic_conductor:
                // ShaderGraph::get_num_closures reserves one MicrofacetBsdf
                // slot plus one Fresnel payload slot for either conductor
                // model, independent of the MULTI_GGX distribution flag.
                add(2u);
                break;
        }
    }
    for (const auto &volume :
         program.volume_instructions()) {
        switch (volume.operation) {
            case compiler::VolumeOperation::null_volume:
            case compiler::VolumeOperation::add:
            case compiler::VolumeOperation::mix:
                break;
            case compiler::VolumeOperation::absorption:
            case compiler::VolumeOperation::scatter:
            case compiler::VolumeOperation::coefficients:
            case compiler::VolumeOperation::principled:
                // Official ShaderGraph::get_num_closures() reserves one
                // MAX_VOLUME_STACK_SIZE block for every volume closure node.
                add(
                    cycles_volume_node_closure_allocations);
                break;
            case compiler::VolumeOperation::emission:
                // EmissionNode has CLOSURE_EMISSION_ID in Cycles and counts
                // as one closure even when reached from the volume output;
                // unlike phase closures it does not reserve a volume block.
                add(1u);
                break;
        }
    }
    return count;
}

std::uint32_t
cycles_scene_closure_allocation_budget(
    const compiler::MaterialLibrary &materials) noexcept {
    auto maximum = std::uint32_t{1u};
    for (const auto &[id, material] :
         materials.materials()) {
        static_cast<void>(id);
        maximum = std::max(
            maximum,
            cycles_program_closure_allocation_count(
                *material.surface_program()));
    }
    return std::min(
        maximum,
        cycles_max_closure_allocations);
}

bool WorldBounds::intersects(
    const WorldBounds &other) const noexcept {
    return valid && other.valid &&
           minimum.x <= other.maximum.x &&
           maximum.x >= other.minimum.x &&
           minimum.y <= other.maximum.y &&
           maximum.y >= other.minimum.y &&
           minimum.z <= other.maximum.z &&
           maximum.z >= other.minimum.z;
}

bool VolumeSceneMetadata::has_volumes()
    const noexcept {
    return has_world_volume || !objects.empty();
}

VolumeSceneMetadata
VolumeSceneMetadataComponent::analyze(
    const contract::SceneSnapshot &snapshot,
    const std::set<contract::MaterialId>
        &volume_materials) const {
    VolumeSceneMetadata metadata;
    metadata.has_world_volume =
        snapshot.world_shader &&
        volume_materials.contains(
            *snapshot.world_shader);
    for (const auto &[instance_id, instance] :
         snapshot.instances) {
        const auto geometry =
            snapshot.geometries.find(
                instance.geometry);
        if (geometry ==
                snapshot.geometries.end() ||
            !instance_has_volume(
                geometry->second,
                instance,
                volume_materials)) {
            continue;
        }
        const auto bounds =
            object_bounds(
                geometry->second,
                instance.transform);
        if (!bounds.valid) {
            continue;
        }
        metadata.objects.emplace_back(
            VolumeObjectMetadata{
                .instance = instance_id,
                .bounds = bounds,
                .intersects_volume =
                    false});
    }
    for (auto first = std::size_t{0u};
         first < metadata.objects.size();
         ++first) {
        for (auto second = first + 1u;
             second < metadata.objects.size();
             ++second) {
            if (metadata.objects[first]
                    .bounds
                    .intersects(
                        metadata.objects[second]
                            .bounds)) {
                metadata.objects[first]
                    .intersects_volume = true;
                metadata.objects[second]
                    .intersects_volume = true;
            }
        }
    }
    if (!metadata.has_volumes()) {
        return metadata;
    }
    auto stack_size = std::uint32_t{2u};
    auto has_volume_object = false;
    for (const auto &object :
         metadata.objects) {
        if (object.intersects_volume ||
            !has_volume_object) {
            stack_size += 1u;
        }
        has_volume_object = true;
        if (stack_size ==
            maximum_volume_stack_size) {
            break;
        }
    }
    metadata.stack_size =
        std::min(
            stack_size,
            maximum_volume_stack_size);
    return metadata;
}

bool VolumeSceneMetadataComponent::
    camera_may_be_inside_volume(
        const VolumeSceneMetadata &metadata,
        const CameraVolumeBoundsQuery &camera)
        const noexcept {
    if (metadata.objects.empty()) {
        return false;
    }
    const auto viewplane =
        camera_bounds(camera);
    return std::any_of(
        metadata.objects.begin(),
        metadata.objects.end(),
        [&](const VolumeObjectMetadata &object) {
            return viewplane.intersects(
                object.bounds);
        });
}

}// namespace psycles::luisa_backend::detail
