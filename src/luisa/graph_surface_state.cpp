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

}// namespace psycles::luisa_backend::detail
