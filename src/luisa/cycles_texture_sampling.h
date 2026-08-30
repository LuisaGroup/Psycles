#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

[[nodiscard]] constexpr luisa::compute::SamplerAddress
cycles_texture_sampler_address(std::uint32_t extension) noexcept {
    // Surface-program canonical extension codes are Repeat, Clip, Extend,
    // Mirror. This is the same quotient used by Cycles when it creates the
    // backend texture object: wrap, border, clamp, mirror.
    switch (extension) {
        case 0u:
            return luisa::compute::SamplerAddress::REPEAT;
        case 1u:
            return luisa::compute::SamplerAddress::ZERO;
        case 2u:
            return luisa::compute::SamplerAddress::EDGE;
        case 3u:
            return luisa::compute::SamplerAddress::MIRROR;
        default:
            std::abort();
    }
}

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
    auto texture = textures->tex2d(handle);
    const auto address = cycles_texture_sampler_address(extension);
    if (interpolation == 0u) {
        return texture.sample(
            uv, luisa::compute::SamplerFilter::POINT, address);
    }
    if (interpolation == 1u) {
        return texture.sample(
            uv, luisa::compute::SamplerFilter::LINEAR_POINT, address);
    }

    // Cycles' fast bicubic reconstruction is a separable cubic B-spline
    // factored into four native bilinear samples. The factorization is exact:
    // g0=w0+w1, g1=w2+w3 and h selects the bilinear coordinate whose two
    // hardware weights reproduce each adjacent cubic pair.
    const auto cubic_w0 = [](Float a) noexcept {
        return (1.0f / 6.0f) *
               (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
    };
    const auto cubic_w1 = [](Float a) noexcept {
        return (1.0f / 6.0f) *
               (a * a * (3.0f * a - 6.0f) + 4.0f);
    };
    const auto cubic_w2 = [](Float a) noexcept {
        return (1.0f / 6.0f) *
               (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
    };
    const auto cubic_w3 = [](Float a) noexcept {
        return (1.0f / 6.0f) * a * a * a;
    };
    const auto cubic_axis = [&](Float coordinate, Float extent) noexcept {
        const auto x = coordinate * extent - 0.5f;
        const auto px = luisa::compute::floor(x);
        const auto f = x - px;
        const auto w0 = cubic_w0(f);
        const auto w1 = cubic_w1(f);
        const auto w2 = cubic_w2(f);
        const auto w3 = cubic_w3(f);
        const auto g0 = w0 + w1;
        const auto g1 = w2 + w3;
        const auto h0 = w1 / g0 - 1.0f;
        const auto h1 = w3 / g1 + 1.0f;
        return make_float4(
            (px + h0 + 0.5f) / extent,
            (px + h1 + 0.5f) / extent,
            g0,
            g1);
    };
    const auto size = texture.size();
    const auto x = cubic_axis(uv.x, cast<float>(size.x));
    const auto y = cubic_axis(uv.y, cast<float>(size.y));
    const auto sample = [&](Float u, Float v) noexcept {
        return texture.sample(
            make_float2(u, v),
            luisa::compute::SamplerFilter::LINEAR_POINT,
            address);
    };
    return y.z * (x.z * sample(x.x, y.x) +
                  x.w * sample(x.y, y.x)) +
           y.w * (x.z * sample(x.x, y.y) +
                  x.w * sample(x.y, y.y));
}

}// namespace psycles::luisa_backend::detail
