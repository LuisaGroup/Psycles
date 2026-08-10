#pragma once

#include "path_tracer_internal.h"

#include <array>

namespace psycles::luisa_backend::detail {

// Canonical Cycles TextureInterpolator transfer function. Interpolation and
// extension are immutable shader-graph metadata, so host specialization keeps
// device control flow out of the recorded shader. Texture handles and UVs
// remain Luisa expressions.
template<typename TextureHeap>
[[nodiscard]] Float4 sample_cycles_texture_2d(
    const TextureHeap &textures,
    Expr<std::uint32_t> handle,
    Expr<luisa::float2> uv,
    std::uint32_t interpolation,
    std::uint32_t extension) noexcept {
    // Implement Cycles' TextureInterpolator explicitly. Relying on a backend
    // sampler makes clip-border, mirror, and cubic behavior backend-dependent
    // and differs from Cycles at texel boundaries.
    auto texture = textures->tex2d(handle);
    auto size = texture.size();
    Int width = cast<int>(size.x);
    Int height = cast<int>(size.y);

    const auto split_coordinate =
        [](Float coordinate, Int &index) noexcept -> Float {
        // Match Cycles' frac(): truncation followed by an explicit negative
        // correction, including negative integers.
        index = cast<int>(coordinate) -
                select(0, 1, coordinate < 0.0f);
        return coordinate - cast<float>(index);
    };
    const auto wrap_periodic =
        [](Int coordinate, Int extent) noexcept -> Int {
        auto wrapped = coordinate % extent;
        return select(wrapped,
                      wrapped + extent,
                      wrapped < 0);
    };
    const auto wrap_mirror =
        [](Int coordinate, Int extent) noexcept -> Int {
        auto adjusted = coordinate +
                        select(0, 1, coordinate < 0);
        auto period = abs(adjusted) % (2 * extent);
        return select(period,
                      2 * extent - period - 1,
                      period >= extent);
    };
    const auto read_clip =
        [&](Int x, Int y) noexcept -> Float4 {
        Float4 value = make_float4(0.0f);
        $if ((x >= 0) & (x < width) &
             (y >= 0) & (y < height)) {
            value = texture.read(make_uint2(
                cast<uint>(x), cast<uint>(y)));
        };
        return value;
    };
    const auto wrap_coordinate =
        [&](Int coordinate, Int extent) noexcept -> Int {
        if (extension == 0u) {
            return wrap_periodic(coordinate, extent);
        }
        if (extension == 2u) {
            return clamp(coordinate, 0, extent - 1);
        }
        if (extension == 3u) {
            return wrap_mirror(coordinate, extent);
        }
        return coordinate;
    };
    const auto read_wrapped =
        [&](Int x, Int y) noexcept -> Float4 {
        return read_clip(wrap_coordinate(x, width),
                         wrap_coordinate(y, height));
    };

    auto coordinate = def(uv);
    if (interpolation == 0u) {
        Int x;
        Int y;
        static_cast<void>(split_coordinate(
            coordinate.x * cast<float>(width), x));
        static_cast<void>(split_coordinate(
            coordinate.y * cast<float>(height), y));
        return read_wrapped(x, y);
    }

    Int x;
    Int y;
    auto tx = split_coordinate(
        coordinate.x * cast<float>(width) - 0.5f, x);
    auto ty = split_coordinate(
        coordinate.y * cast<float>(height) - 0.5f, y);

    if (interpolation == 1u) {
        auto x1 = x + 1;
        auto y1 = y + 1;
        auto row0 = (1.0f - tx) * read_wrapped(x, y) +
                    tx * read_wrapped(x1, y);
        auto row1 = (1.0f - tx) * read_wrapped(x, y1) +
                    tx * read_wrapped(x1, y1);
        return (1.0f - ty) * row0 + ty * row1;
    }

    // Cycles treats both Cubic and Smart as cubic. These are its exact cubic
    // B-spline weights.
    const auto cubic_weights = [](Float t) noexcept {
        return std::array<Float, 4u>{
            (((-1.0f / 6.0f) * t + 0.5f) * t -
             0.5f) *
                    t +
                (1.0f / 6.0f),
            ((0.5f * t - 1.0f) * t) * t +
                (2.0f / 3.0f),
            ((-0.5f * t + 0.5f) * t + 0.5f) *
                    t +
                (1.0f / 6.0f),
            (1.0f / 6.0f) * t * t * t};
    };
    auto wx = cubic_weights(tx);
    auto wy = cubic_weights(ty);
    const auto cubic_row = [&](Int row) noexcept {
        return wx[0u] * read_wrapped(x - 1, row) +
               wx[1u] * read_wrapped(x, row) +
               wx[2u] * read_wrapped(x + 1, row) +
               wx[3u] * read_wrapped(x + 2, row);
    };
    return wy[0u] * cubic_row(y - 1) +
           wy[1u] * cubic_row(y) +
           wy[2u] * cubic_row(y + 1) +
           wy[3u] * cubic_row(y + 2);
}

}// namespace psycles::luisa_backend::detail
