#pragma once

#include <psycles/compiler/material_library.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/surface.h>
#include <psycles/luisa/volume_stack.h>

#include <cstdint>
#include <set>
#include <vector>

namespace psycles::luisa_backend::detail {

struct WorldBounds {
    Vec3f minimum{};
    Vec3f maximum{};
    bool valid{};

    [[nodiscard]] bool
    intersects(const WorldBounds &other) const noexcept;
};

struct VolumeObjectMetadata {
    contract::InstanceId instance;
    WorldBounds bounds;
    bool intersects_volume{};
};

struct VolumeSceneMetadata {
    std::vector<VolumeObjectMetadata> objects;
    bool has_world_volume{};
    // Zero is Psycles' host-specialization sentinel. A volume-enabled scene
    // stores Cycles' actual size, including background and terminator slots.
    std::uint32_t stack_size{};
    // Cycles' kernel_data.max_closures, specialized from reachable shader
    // graphs and capped by MAX_CLOSURE. Volume evaluation consumes this
    // global budget across every active stack entry.
    std::uint32_t closure_allocation_budget{1u};

    [[nodiscard]] bool has_volumes() const noexcept;
};

inline constexpr std::uint32_t
    cycles_max_closure_allocations =
        maximum_surface_closure_capacity;
inline constexpr std::uint32_t
    cycles_volume_node_closure_allocations =
        maximum_volume_stack_size;

[[nodiscard]] std::uint32_t
cycles_program_closure_allocation_count(
    const compiler::SurfaceProgram &program) noexcept;

[[nodiscard]] std::uint32_t
cycles_scene_closure_allocation_budget(
    const compiler::MaterialLibrary &materials) noexcept;

struct CameraVolumeBoundsQuery {
    contract::CameraProjection projection{
        contract::CameraProjection::perspective};
    Mat4f transform;
    float aspect{1.0f};
    float horizontal_tangent{};
    float vertical_tangent{};
    float orthographic_scale{1.0f};
    float shift_x{};
    float shift_y{};
    float near_clip{};
    float aperture_radius{};
    float focal_distance{};
    float aperture_ratio{1.0f};
};

// Host-side scene analysis matching Cycles' conservative stack allocation and
// camera-view-plane overlap test. This only specializes Luisa AST construction;
// it never evaluates radiance or acts as a reference renderer.
class VolumeSceneMetadataComponent {

  public:
    [[nodiscard]] VolumeSceneMetadata
    analyze(
        const contract::SceneSnapshot &snapshot,
        const std::set<contract::MaterialId>
            &volume_materials) const;

    [[nodiscard]] bool
    camera_may_be_inside_volume(
        const VolumeSceneMetadata &metadata,
        const CameraVolumeBoundsQuery &camera)
        const noexcept;
};

}// namespace psycles::luisa_backend::detail
