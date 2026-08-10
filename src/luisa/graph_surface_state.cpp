#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float scalar(
    compiler::ValueExpressionId id,
    const TracedValues &values) noexcept {
    return get(id, values.values).scalar();
}

[[nodiscard]] Float3 vector(
    compiler::ValueExpressionId id,
    const TracedValues &values) noexcept {
    return get(id, values.values).vector();
}

[[nodiscard]] ULong unsigned_integer(
    compiler::ValueExpressionId id,
    const TracedValues &values) noexcept {
    return get(id, values.values).unsigned_integer();
}

[[nodiscard]] Float sample_weight(Float3 value) noexcept {
    // Cycles stores fabsf(average(weight)) in ShaderClosure. Ordinary
    // BSDF allocation has already clamped negative spectral components;
    // transparent closure setup applies the absolute value directly.
    return abs((value.x + value.y + value.z) / 3.0f);
}

[[nodiscard]] TransparentClosureState transparent_closure_state(
    Float3 weight) noexcept {
    // bsdf_transparent_setup differs from ordinary bsdf_alloc: it retains
    // signed spectral weights and applies the cutoff to fabs(average).
    const auto candidate_sample_weight = sample_weight(weight);
    const auto allocated = candidate_sample_weight >=
                           cycles_closure::closure_weight_cutoff;
    return {
        .weight = select(make_float3(0.0f), weight, allocated),
        .sample_weight = select(
            0.0f, candidate_sample_weight, allocated)};
}

[[nodiscard]] Float3 bsdf_allocated_weight(
    Float3 value) noexcept {
    // Cycles bsdf_alloc() removes negative spectral weight and skips
    // closures whose average weight is below CLOSURE_WEIGHT_CUTOFF.
    value = max(value, make_float3(0.0f));
    auto average = (value.x + value.y + value.z) / 3.0f;
    return select(
        make_float3(0.0f),
        value,
        average >= 1.0e-5f);
}

[[nodiscard]] Float pass_weight(Float3 value) noexcept {
    // Cycles data passes use fabsf(average(sc->weight)), which differs
    // from the lobe-selection weight when signed closure weights are
    // present.
    return abs((value.x + value.y + value.z) / 3.0f);
}

[[nodiscard]] Float max_component(
    Float3 value) noexcept {
    return max(value.x, max(value.y, value.z));
}

[[nodiscard]] Float srgb_to_linear(
    Float value) noexcept {
    auto linear_segment =
        max(value, 0.0f) * (1.0f / 12.92f);
    auto power_segment = pow(
        (value + 0.055f) * (1.0f / 1.055f),
        2.4f);
    return select(
        power_segment,
        linear_segment,
        value < 0.04045f);
}

[[nodiscard]] Float3 srgb_to_linear(
    Float3 value) noexcept {
    return make_float3(
        srgb_to_linear(value.x),
        srgb_to_linear(value.y),
        srgb_to_linear(value.z));
}

[[nodiscard]] Float cycles_table_1d(
    const ShaderServices &services,
    Float x,
    Expr<std::uint32_t> offset,
    std::uint32_t size) noexcept {
    auto coordinate =
        clamp(x, 0.0f, 1.0f) *
        static_cast<float>(size - 1u);
    auto index = min(
        cast<luisa::uint>(coordinate),
        size - 1u);
    auto next = min(index + 1u, size - 1u);
    auto t = coordinate - cast<float>(index);
    auto data0 =
        services.cycles_bsdf_data(index + offset);
    auto data1 =
        services.cycles_bsdf_data(next + offset);
    return lerp(data0, data1, t);
}

[[nodiscard]] Float cycles_table_2d(
    const ShaderServices &services,
    Float x,
    Float y,
    Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size) noexcept {
    auto coordinate =
        clamp(y, 0.0f, 1.0f) *
        static_cast<float>(y_size - 1u);
    auto index = min(
        cast<luisa::uint>(coordinate),
        y_size - 1u);
    auto next = min(index + 1u, y_size - 1u);
    auto t = coordinate - cast<float>(index);
    auto data0 = cycles_table_1d(
        services,
        x,
        offset + x_size * index,
        x_size);
    auto data1 = cycles_table_1d(
        services,
        x,
        offset + x_size * next,
        x_size);
    return lerp(data0, data1, t);
}

[[nodiscard]] Float cycles_table_3d(
    const ShaderServices &services,
    Float x,
    Float y,
    Float z,
    Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size,
    std::uint32_t z_size) noexcept {
    auto coordinate =
        clamp(z, 0.0f, 1.0f) *
        static_cast<float>(z_size - 1u);
    auto index = min(
        cast<luisa::uint>(coordinate),
        z_size - 1u);
    auto next = min(index + 1u, z_size - 1u);
    auto t = coordinate - cast<float>(index);
    auto slice_stride = x_size * y_size;
    auto data0 = cycles_table_2d(
        services,
        x,
        y,
        offset + slice_stride * index,
        x_size,
        y_size);
    auto data1 = cycles_table_2d(
        services,
        x,
        y,
        offset + slice_stride * next,
        x_size,
        y_size);
    return lerp(data0, data1, t);
}

[[nodiscard]] Float3 rgb_to_hsv(
    const ShaderServices &services,
    Float3 rgb) noexcept {
    if (const auto provider =
            services.surface_color_transform_provider()) {
        return provider->rgb_to_hsv(rgb);
    }
    return rgb_to_hsv_inline(rgb);
}

[[nodiscard]] Float3 hsv_to_rgb(
    const ShaderServices &services,
    Float3 hsv) noexcept {
    if (const auto provider =
            services.surface_color_transform_provider()) {
        return provider->hsv_to_rgb(hsv);
    }
    return hsv_to_rgb_inline(hsv);
}

[[nodiscard]] Float3 rgb_to_hsl(
    const ShaderServices &services,
    Float3 rgb) noexcept {
    if (const auto provider =
            services.surface_color_transform_provider()) {
        return provider->rgb_to_hsl(rgb);
    }
    return rgb_to_hsl_inline(rgb);
}

[[nodiscard]] Float3 hsl_to_rgb(
    const ShaderServices &services,
    Float3 hsl) noexcept {
    if (const auto provider =
            services.surface_color_transform_provider()) {
        return provider->hsl_to_rgb(hsl);
    }
    return hsl_to_rgb_inline(hsl);
}

[[nodiscard]] Float3 separate_color(
    const ShaderServices &services,
    Float3 color,
    std::uint64_t mode) noexcept {
    return mode == 1u
               ? rgb_to_hsv(services, color)
               : mode == 2u
                     ? rgb_to_hsl(services, color)
                     : color;
}

[[nodiscard]] Float3 combine_color(
    const ShaderServices &services,
    Float3 channels,
    std::uint64_t mode) noexcept {
    return mode == 1u
               ? hsv_to_rgb(services, channels)
               : mode == 2u
                     ? hsl_to_rgb(services, channels)
                     : channels;
}

}// namespace psycles::luisa_backend::detail
