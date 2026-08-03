#include <psycles/contract/cycles_pointiness.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace psycles::contract {
namespace {

[[nodiscard]] float coordinate_sum(const Vec3f value) noexcept {
    return value.x + value.y + value.z;
}

[[nodiscard]] Vec3f add(const Vec3f a, const Vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f subtract(const Vec3f a, const Vec3f b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3f negate(const Vec3f value) noexcept {
    return {-value.x, -value.y, -value.z};
}

[[nodiscard]] Vec3f divide(const Vec3f value, const float divisor) noexcept {
    return {value.x / divisor, value.y / divisor, value.z / divisor};
}

[[nodiscard]] float dot(const Vec3f a, const Vec3f b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] float length_squared(const Vec3f value) noexcept {
    return dot(value, value);
}

[[nodiscard]] Vec3f normalize(const Vec3f value) noexcept {
    return divide(value, std::sqrt(length_squared(value)));
}

[[nodiscard]] Vec3f safe_normalize(const Vec3f value) noexcept {
    const auto length = std::sqrt(length_squared(value));
    return length != 0.0f ? divide(value, length) : value;
}

[[nodiscard]] float safe_acos(const float value) noexcept {
    return std::acos(std::clamp(value, -1.0f, 1.0f));
}

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> canonical_edge(
    std::uint32_t first,
    std::uint32_t second) noexcept {
    if (first > second) {
        std::swap(first, second);
    }
    return {first, second};
}

}// namespace

std::vector<float> make_cycles_pointiness_attribute(
    const std::span<const Vec3f> positions,
    const std::span<const Vec3f> point_normals,
    const std::span<const std::array<std::uint32_t, 2u>> edges) {
    if (point_normals.size() != positions.size()) {
        throw std::invalid_argument{
            "Cycles Pointiness point-normal cardinality mismatch"};
    }
    for (const auto edge : edges) {
        if (edge[0u] >= positions.size() || edge[1u] >= positions.size()) {
            throw std::invalid_argument{
                "Cycles Pointiness edge index is out of range"};
        }
    }
    if (positions.empty()) {
        return {};
    }

    // Step 1: reproduce Cycles' coordinate-sum ordering and duplicate-chain
    // construction. The exact-position reverse-index rule is important: it
    // makes every chain terminate at the lowest original vertex index.
    std::vector<std::uint32_t> sorted_indices(positions.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0u);
    std::sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [&](const auto first, const auto second) noexcept {
            if (positions[first] == positions[second]) {
                return first > second;
            }
            return coordinate_sum(positions[first]) <
                   coordinate_sum(positions[second]);
        });

    std::vector<std::uint32_t> original_index(positions.size());
    constexpr auto epsilon = std::numeric_limits<float>::epsilon();
    for (std::size_t sorted = 0u; sorted < sorted_indices.size(); ++sorted) {
        const auto vertex = sorted_indices[sorted];
        const auto position = positions[vertex];
        auto original = vertex;
        for (auto other_sorted = sorted + 1u;
             other_sorted < sorted_indices.size();
             ++other_sorted) {
            const auto other = sorted_indices[other_sorted];
            const auto other_position = positions[other];
            if (coordinate_sum(other_position) - coordinate_sum(position) >
                3.0f * epsilon) {
                break;
            }
            if (length_squared(subtract(other_position, position)) < epsilon) {
                original = other;
                break;
            }
        }
        original_index[vertex] = original;
    }
    for (std::size_t vertex = 0u; vertex < original_index.size(); ++vertex) {
        auto original = original_index[vertex];
        while (original != original_index[original]) {
            original = original_index[original];
        }
        original_index[vertex] = original;
    }

    // Step 2: weld point normals before normalization. normalize(), rather
    // than safe_normalize(), intentionally mirrors Cycles' host path.
    std::vector<Vec3f> welded_normals(positions.size());
    for (std::size_t vertex = 0u; vertex < positions.size(); ++vertex) {
        const auto original = original_index[vertex];
        welded_normals[original] =
            add(welded_normals[original], point_normals[vertex]);
    }
    for (std::size_t vertex = 0u; vertex < positions.size(); ++vertex) {
        welded_normals[vertex] = normalize(welded_normals[original_index[vertex]]);
    }

    // Step 3: deduplicate edges after welding and measure the one-ring angle.
    std::vector<std::uint32_t> counters(positions.size(), 0u);
    std::vector<float> raw(positions.size(), 0.0f);
    std::vector<Vec3f> edge_accumulation(positions.size());
    std::set<std::pair<std::uint32_t, std::uint32_t>> visited_edges;
    for (const auto edge : edges) {
        const auto first = original_index[edge[0u]];
        const auto second = original_index[edge[1u]];
        if (!visited_edges.insert(canonical_edge(first, second)).second) {
            continue;
        }
        const auto direction = safe_normalize(
            subtract(positions[second], positions[first]));
        edge_accumulation[first] = add(edge_accumulation[first], direction);
        edge_accumulation[second] =
            add(edge_accumulation[second], negate(direction));
        ++counters[first];
        ++counters[second];
    }
    constexpr auto inverse_pi = 0.31830988618379067154f;
    for (std::size_t vertex = 0u; vertex < positions.size(); ++vertex) {
        if (original_index[vertex] != vertex || counters[vertex] == 0u) {
            continue;
        }
        const auto average_edge = divide(
            edge_accumulation[vertex],
            static_cast<float>(counters[vertex]));
        raw[vertex] = safe_acos(dot(welded_normals[vertex], average_edge)) *
                      inverse_pi;
    }

    // Step 4: add the one-ring neighbors once to approximate a two-ring
    // neighborhood, then copy the welded originals back to all duplicates.
    auto values = raw;
    std::fill(counters.begin(), counters.end(), 0u);
    visited_edges.clear();
    for (const auto edge : edges) {
        const auto first = original_index[edge[0u]];
        const auto second = original_index[edge[1u]];
        if (!visited_edges.insert(canonical_edge(first, second)).second) {
            continue;
        }
        values[first] += raw[second];
        values[second] += raw[first];
        ++counters[first];
        ++counters[second];
    }
    for (std::size_t vertex = 0u; vertex < positions.size(); ++vertex) {
        values[vertex] /= static_cast<float>(counters[vertex] + 1u);
    }
    for (std::size_t vertex = 0u; vertex < positions.size(); ++vertex) {
        values[vertex] = values[original_index[vertex]];
    }
    return values;
}

}// namespace psycles::contract
