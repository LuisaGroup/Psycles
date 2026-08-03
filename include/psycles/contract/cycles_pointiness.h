#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <psycles/core/math.h>

namespace psycles::contract {

// Construct Cycles' ATTR_STD_POINTINESS point-domain attribute from the
// evaluated mesh-sync inputs. Invalid cardinalities or edge indices throw
// std::invalid_argument rather than silently changing topology.
[[nodiscard]] std::vector<float> make_cycles_pointiness_attribute(
    std::span<const Vec3f> positions,
    std::span<const Vec3f> point_normals,
    std::span<const std::array<std::uint32_t, 2u>> edges);

}// namespace psycles::contract
