#include "surface_math.h"

namespace psycles::luisa_backend::detail {

Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept {
    const auto valid = dot(value, value) > 1.0e-20f;
    auto selected = select(fallback, value, valid);
    const auto fallback_valid =
        dot(selected, selected) > 1.0e-20f;
    selected = select(
        make_float3(0.0f, 0.0f, 1.0f),
        selected,
        fallback_valid);
    return normalize(selected);
}

}// namespace psycles::luisa_backend::detail
