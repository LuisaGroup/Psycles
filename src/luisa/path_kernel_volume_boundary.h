#pragma once

#include "path_kernel_triangle_primitive.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct TriangleVolumeBoundary {
    TrianglePrimitiveContext primitive;
    Float3 geometric_normal;
    Bool back_facing;
};

// Minimal triangle reconstruction for volume-only traversal. It deliberately
// omits shading normals, UVs, tangents, and arbitrary attributes: Cycles'
// enclosure probe needs only SD_HAS_VOLUME, exact stack identity, and the
// geometric front/back orientation.
class TriangleVolumeBoundaryComponent {

  public:
    virtual ~TriangleVolumeBoundaryComponent() noexcept =
        default;

    [[nodiscard]] virtual Bool has_volume(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id)
        const noexcept = 0;

    [[nodiscard]] virtual TriangleVolumeBoundary
    resolve(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id,
        Expr<luisa::float3> ray_direction)
        const noexcept = 0;
};

struct CameraVolumeStackInitialization {
    UInt intersection_count;
    UInt enclosed_count;
};

class CameraVolumeStackComponent {

  public:
    virtual ~CameraVolumeStackComponent() noexcept =
        default;

    [[nodiscard]]
    virtual CameraVolumeStackInitialization
    initialize(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<luisa::float3> camera_origin,
        Expr<std::uint32_t> visibility,
        VolumeStack &stack) const noexcept = 0;
};

[[nodiscard]]
std::shared_ptr<const TriangleVolumeBoundaryComponent>
make_triangle_volume_boundary_component(
    std::shared_ptr<const TrianglePrimitiveComponent>
        primitive = {});

[[nodiscard]]
std::shared_ptr<const CameraVolumeStackComponent>
make_camera_volume_stack_component();

}// namespace psycles::luisa_backend::detail
