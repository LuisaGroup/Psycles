#pragma once

#include "blender_scene_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace psycles::adapter::detail {

template <typename T>
[[nodiscard]] std::vector<T> read_packed_values(
    std::span<const std::byte> binary,
    std::uint64_t offset,
    std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("binary scene section is too large");
    }
    const auto byte_count = count * sizeof(T);
    if (offset > binary.size() ||
        byte_count > binary.size() - static_cast<std::size_t>(offset)) {
        throw std::runtime_error("geometry binary section exceeds buffer");
    }
    std::vector<T> result(count);
    if (byte_count > 0u) {
        std::memcpy(
            result.data(),
            binary.data() + static_cast<std::size_t>(offset),
            byte_count);
    }
    return result;
}

[[nodiscard]] std::uint64_t section_offset(
    yyjson_val *geometry,
    const char *name) noexcept;

/// Exact Cycles / Blender `hash_uint` (Jenkins lookup3 single-uint mix).
[[nodiscard]] std::uint32_t cycles_jenkins_hash_uint(
    std::uint32_t value) noexcept;

/// Derives per-triangle random-per-island floats via DisjointSet over edges
/// and `cycles_jenkins_hash_uint(find_root(tri[0]))`.
[[nodiscard]] std::vector<float> compute_cycles_random_per_island(
    std::size_t point_count,
    std::span<const std::uint32_t> edge_vertices,
    std::span<const std::array<std::uint32_t, 3u>> triangles);

[[nodiscard]] contract::TriangleMeshDesc parse_blender_triangle_mesh(
    yyjson_val *geometry,
    std::span<const std::byte> geometry_bin,
    bool compact_geometry,
    const std::map<std::string, contract::MaterialId, std::less<>> &material_ids,
    contract::MaterialId default_material);

[[nodiscard]] contract::CurveGeometryDesc parse_blender_curve_geometry(
    yyjson_val *geometry,
    std::span<const std::byte> geometry_bin,
    const std::map<std::string, contract::MaterialId, std::less<>> &material_ids,
    contract::MaterialId default_material);

} // namespace psycles::adapter::detail
