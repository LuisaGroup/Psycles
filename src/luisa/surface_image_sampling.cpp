#include "surface_image_sampling.h"

#include "surface_color_encoding.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

template <typename T>
[[nodiscard]] T srgb_to_linear_exact(T value) noexcept {
    // Branchless exact IEC 61966-2-1 sRGB-to-linear covering negative, SDR,
    // and HDR (> 1.0) inputs without per-channel scalar HDR fallback branches.
    const auto low = max(value, 0.0f) * (1.0f / 12.92f);
    const auto high = pow(max(value + 0.055f, 1.0e-30f) * (1.0f / 1.055f), 2.4f);
    return select(high, low, value < 0.04045f);
}

[[nodiscard]] const luisa::compute::Callable<luisa::float3(luisa::float3)> &
srgb_to_linear_float3_callable() noexcept {
    static auto callable = []() noexcept {
        luisa::compute::Callable result{
            [](Float3 value) noexcept {
                return srgb_to_linear_exact(value);
            }};
        result.set_name("cycles_srgb_to_linear_float3");
        return result;
    }();
    return callable;
}

}// namespace

Float srgb_to_linear(Float value) noexcept {
    return srgb_to_linear_exact(value);
}

Float3 srgb_to_linear(Float3 value) noexcept {
    return srgb_to_linear_float3_callable()(value);
}

Float4 decode_surface_image_sample(
    Float4 sampled,
    Bool unassociate_alpha,
    Bool encoded_as_srgb) noexcept {
    $if(unassociate_alpha) {
        const auto alpha = sampled.w;
        const auto should_unassociate =
            (alpha != 0.0f) & (alpha != 1.0f);
        const auto safe_alpha =
            select(1.0f, alpha, should_unassociate);
        const auto unassoc = sampled.xyz() / safe_alpha;
        // Clamp sRGB cutout textures after unassociation so lossy BC7 alpha/color
        // reconstruction noise near alpha == 0 cannot blow up pow(x, 2.4).
        const auto safe_unassoc =
            select(unassoc, clamp(unassoc, 0.0f, 1.0f), encoded_as_srgb);
        sampled = make_float4(
            select(
                sampled.xyz(),
                safe_unassoc,
                should_unassociate),
            alpha);
    };
    $if(encoded_as_srgb) {
        // Cycles filters associated encoded texels first, optionally
        // unassociates alpha, and only then decodes the RGB channels.
        sampled = make_float4(
            srgb_to_linear(sampled.xyz()),
            sampled.w);
    };
    return sampled;
}

}// namespace psycles::luisa_backend::detail
