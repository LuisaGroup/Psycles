#include <psycles/luisa/volume_majorant_hierarchy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace psycles::luisa_backend {
namespace {

struct GridBox {
    std::array<std::uint32_t, 3u> minimum{};
    std::array<std::uint32_t, 3u> maximum{};
};

[[nodiscard]] bool finite(
    luisa::float3 value) noexcept {
    return std::isfinite(value.x) &&
           std::isfinite(value.y) &&
           std::isfinite(value.z);
}

[[nodiscard]] luisa::float3 subtract(
    luisa::float3 a,
    luisa::float3 b) noexcept {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z};
}

[[nodiscard]] VolumeMajorantExtrema merge(
    VolumeMajorantExtrema a,
    VolumeMajorantExtrema b) noexcept {
    return {
        .minimum =
            std::min(a.minimum, b.minimum),
        .maximum =
            std::max(a.maximum, b.maximum)};
}

class HierarchyBuildState {

  private:
    std::span<const VolumeMajorantExtrema>
        _grid;
    VolumeMajorantBounds _bounds;
    float _volume_scale;
    VolumeMajorantHierarchy _hierarchy;

    [[nodiscard]] static std::size_t
    flatten(
        std::uint32_t x,
        std::uint32_t y,
        std::uint32_t z) noexcept {
        constexpr auto resolution =
            static_cast<std::size_t>(
                volume_majorant_grid_resolution);
        return static_cast<std::size_t>(x) +
               resolution *
                   (static_cast<std::size_t>(y) +
                    resolution *
                        static_cast<std::size_t>(z));
    }

    [[nodiscard]] VolumeMajorantExtrema
    extrema(const GridBox &box) const noexcept {
        auto result = VolumeMajorantExtrema{
            .minimum =
                std::numeric_limits<float>::max(),
            .maximum =
                -std::numeric_limits<float>::max()};
        for (auto z = box.minimum[2u];
             z < box.maximum[2u];
             ++z) {
            for (auto y = box.minimum[1u];
                 y < box.maximum[1u];
                 ++y) {
                for (auto x = box.minimum[0u];
                     x < box.maximum[0u];
                     ++x) {
                    result = merge(
                        result,
                        _grid[flatten(x, y, z)]);
                }
            }
        }
        return result;
    }

    [[nodiscard]] float node_diagonal(
        const GridBox &box) const noexcept {
        const auto root_size =
            subtract(
                _bounds.maximum,
                _bounds.minimum);
        constexpr auto inverse_resolution =
            1.0f /
            static_cast<float>(
                volume_majorant_grid_resolution);
        const auto x =
            root_size.x *
            static_cast<float>(
                box.maximum[0u] -
                box.minimum[0u]) *
            inverse_resolution;
        const auto y =
            root_size.y *
            static_cast<float>(
                box.maximum[1u] -
                box.minimum[1u]) *
            inverse_resolution;
        const auto z =
            root_size.z *
            static_cast<float>(
                box.maximum[2u] -
                box.minimum[2u]) *
            inverse_resolution;
        return std::sqrt(x * x + y * y + z * z);
    }

    [[nodiscard]] bool should_split(
        const GridBox &box,
        std::uint32_t depth,
        VolumeMajorantExtrema sigma) const noexcept {
        return depth <
                   volume_majorant_maximum_depth &&
               (sigma.maximum - sigma.minimum) *
                       node_diagonal(box) *
                       _volume_scale >
                   volume_majorant_split_threshold;
    }

    void build_node(
        std::uint32_t node_index,
        const GridBox &box,
        std::uint32_t depth) {
        const auto sigma = extrema(box);
        _hierarchy.nodes[node_index]
            .sigma_minimum = sigma.minimum;
        _hierarchy.nodes[node_index]
            .sigma_maximum = sigma.maximum;
        if (!should_split(
                box, depth, sigma)) {
            return;
        }

        const std::array center{
            (box.minimum[0u] +
             box.maximum[0u]) /
                2u,
            (box.minimum[1u] +
             box.maximum[1u]) /
                2u,
            (box.minimum[2u] +
             box.maximum[2u]) /
                2u};
        const auto first_child =
            static_cast<std::uint32_t>(
                _hierarchy.nodes.size());
        _hierarchy.nodes[node_index]
            .first_child =
            static_cast<std::int32_t>(
                first_child);
        _hierarchy.nodes.resize(
            _hierarchy.nodes.size() + 8u);
        for (auto child = 0u;
             child < 8u;
             ++child) {
            auto &node =
                _hierarchy.nodes[
                    first_child + child];
            node.parent =
                static_cast<std::int32_t>(
                    node_index);
            GridBox child_box;
            for (auto axis = 0u;
                 axis < 3u;
                 ++axis) {
                const auto high =
                    ((child >> axis) & 1u) !=
                    0u;
                child_box.minimum[axis] =
                    high
                        ? center[axis]
                        : box.minimum[axis];
                child_box.maximum[axis] =
                    high
                        ? box.maximum[axis]
                        : center[axis];
            }
            build_node(
                first_child + child,
                child_box,
                depth + 1u);
        }
    }

  public:
    HierarchyBuildState(
        std::span<const VolumeMajorantExtrema>
            grid,
        VolumeMajorantBounds bounds,
        float volume_scale) noexcept
        : _grid{grid},
          _bounds{bounds},
          _volume_scale{volume_scale} {}

    [[nodiscard]] VolumeMajorantHierarchy
    build() {
        _hierarchy.nodes.emplace_back();
        const GridBox root_box{
            .minimum = {0u, 0u, 0u},
            .maximum = {
                volume_majorant_grid_resolution,
                volume_majorant_grid_resolution,
                volume_majorant_grid_resolution}};
        build_node(0u, root_box, 0u);

        const auto size =
            subtract(
                _bounds.maximum,
                _bounds.minimum);
        const luisa::float3 scale{
            1.0f / size.x,
            1.0f / size.y,
            1.0f / size.z};
        _hierarchy.root = {
            .scale = scale,
            .node = 0u,
            .translation = {
                -_bounds.minimum.x * scale.x +
                    1.0f,
                -_bounds.minimum.y * scale.y +
                    1.0f,
                -_bounds.minimum.z * scale.z +
                    1.0f},
            .shader = ~std::uint32_t{0u}};
        return std::move(_hierarchy);
    }
};

}// namespace

VolumeMajorantBuildResult
VolumeMajorantHierarchyBuilder::build(
    const VolumeMajorantBounds &bounds,
    std::span<const VolumeMajorantExtrema>
        extrema,
    float volume_scale) const {
    VolumeMajorantBuildResult result;
    if (extrema.size() !=
        required_extrema_count()) {
        result.diagnostic =
            "volume majorant extrema grid must contain "
            "exactly 128^3 cells";
        return result;
    }
    if (!finite(bounds.minimum) ||
        !finite(bounds.maximum) ||
        bounds.maximum.x <
            bounds.minimum.x ||
        bounds.maximum.y <
            bounds.minimum.y ||
        bounds.maximum.z <
            bounds.minimum.z ||
        (bounds.maximum.x ==
             bounds.minimum.x &&
         bounds.maximum.y ==
             bounds.minimum.y &&
         bounds.maximum.z ==
             bounds.minimum.z)) {
        result.diagnostic =
            "volume majorant bounds must be finite and "
            "not fully degenerate";
        return result;
    }
    if (!std::isfinite(volume_scale) ||
        volume_scale < 0.0f) {
        result.diagnostic =
            "volume majorant density scale must be finite "
            "and nonnegative";
        return result;
    }
    const auto valid_extrema =
        std::all_of(
            extrema.begin(),
            extrema.end(),
            [](VolumeMajorantExtrema value) {
                return std::isfinite(
                           value.minimum) &&
                       std::isfinite(
                           value.maximum) &&
                       value.minimum >= 0.0f &&
                       value.maximum >=
                           value.minimum;
            });
    if (!valid_extrema) {
        result.diagnostic =
            "volume majorant extrema must be finite, "
            "nonnegative, and ordered";
        return result;
    }
    result.hierarchy =
        HierarchyBuildState{
            extrema,
            bounds,
            volume_scale}
            .build();
    return result;
}

}// namespace psycles::luisa_backend
