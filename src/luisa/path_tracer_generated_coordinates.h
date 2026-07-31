#pragma once

#include <psycles/contract/scene.h>

namespace psycles::luisa_backend::detail {

// One host-side representation serves both paths of Cycles' Generated
// contract: surface fallback values and volume interior coordinates.
struct GeneratedCoordinateMapping {
    Mat4f object_to_generated;

    [[nodiscard]] Vec3f apply(
        Vec3f object_position) const noexcept;
};

[[nodiscard]] GeneratedCoordinateMapping
make_generated_coordinate_mapping(
    const contract::TriangleMeshDesc &geometry) noexcept;

}// namespace psycles::luisa_backend::detail
