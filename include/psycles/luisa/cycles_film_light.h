#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_film_light.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_film_light {

// Cycles' film_clamp_light first maps every non-finite contribution component
// to zero and only then applies the direct/indirect sample limit. Keeping the
// two operations in one DSL contract prevents Combined and the split light
// passes from observing different versions of the same path contribution.
[[nodiscard]] inline luisa::compute::Float3 ensure_finite(
    luisa::compute::Float3 value) noexcept {
    using namespace luisa::compute;
    const auto finite_x = !luisa::compute::dsl::isnan(value.x) & !luisa::compute::dsl::isinf(value.x);
    const auto finite_y = !luisa::compute::dsl::isnan(value.y) & !luisa::compute::dsl::isinf(value.y);
    const auto finite_z = !luisa::compute::dsl::isnan(value.z) & !luisa::compute::dsl::isinf(value.z);
    return make_float3(
        select(0.0f, value.x, finite_x),
        select(0.0f, value.y, finite_y),
        select(0.0f, value.z, finite_z));
}

[[nodiscard]] inline luisa::compute::Float3 clamp(
    luisa::compute::Float3 contribution,
    luisa::compute::UInt depth,
    luisa::compute::Float direct_limit,
    luisa::compute::Float indirect_limit) noexcept {
    using namespace luisa::compute;
    contribution = ensure_finite(contribution);
    const Float limit = select(
        direct_limit,
        indirect_limit,
        depth > 0u);
    const Float magnitude =
        abs(contribution.x) +
        abs(contribution.y) +
        abs(contribution.z);
    const Bool should_clamp =
        (limit > 0.0f) & (magnitude > limit);
    return select(
        contribution,
        contribution *
            (limit / max(magnitude, 1.0e-20f)),
        should_clamp);
}

}// namespace psycles::luisa_backend::cycles_film_light
