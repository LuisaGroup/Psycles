#include "graph_surface_internal.h"

#include <luisa/dsl/func.h>
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

namespace {

[[nodiscard]] const luisa::compute::Callable<float(float)> &
srgb_to_linear_hdr_callable() noexcept {
    static auto callable = []() noexcept {
        luisa::compute::Callable result{
            [](Float value) noexcept {
                Float out = 0.0f;
                $outline_with_name("cycles_srgb_to_linear_hdr") {
                    out = pow((value + 0.055f) * (1.0f / 1.055f), 2.4f);
                };
                return out;
            }};
        result.set_name("cycles_srgb_to_linear_hdr");
        return result;
    }();
    return callable;
}

template <typename T>
[[nodiscard]] T srgb_to_linear_unit_poly(T value) noexcept {
    // On [0.04045, 1.0], t = (x + 0.055) / 1.055 in [0.09047, 1.0] and
    // q = sqrt(sqrt(t)) in [0.54844, 1.0]. With u = q - 1,
    // t^2.4 = t^2 * (1 + u * P4(u)) matches pow(t, 2.4f) within 3.6e-7
    // (<= 2 ULP) and evaluates exact 1.0 at x = 1.0 without invoking pow().
    const auto t = (value + 0.055f) * (1.0f / 1.055f);
    const auto q = sqrt(sqrt(max(t, 0.0f)));
    const auto u = q - 1.0f;
    const auto p =
        ((((-0.0229220558f * u + 0.0167627893f) * u - 0.0649761409f) * u +
          0.479943097f) *
             u +
         1.59999943f) *
            u +
        1.0f;
    return select(
        t * t * p,
        max(value, 0.0f) * (1.0f / 12.92f),
        value < 0.04045f);
}

}// namespace

[[nodiscard]] Float srgb_to_linear(
    Float value) noexcept {
    Float result = srgb_to_linear_unit_poly(value);
    $if(value > 1.0f) {
        result = srgb_to_linear_hdr_callable()(value);
    };
    return result;
}

[[nodiscard]] Float3 srgb_to_linear(
    Float3 value) noexcept {
    Float3 result = srgb_to_linear_unit_poly(value);
    $if(any(value > 1.0f)) {
        const auto &hdr = srgb_to_linear_hdr_callable();
        $if(value.x > 1.0f) { result.x = hdr(value.x); };
        $if(value.y > 1.0f) { result.y = hdr(value.y); };
        $if(value.z > 1.0f) { result.z = hdr(value.z); };
    };
    return result;
}

}// namespace psycles::luisa_backend::detail
