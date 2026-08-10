#include "principled_specular_state.h"

namespace psycles::luisa_backend::detail {

PrincipledSpecularState populate_principled_specular_state(
    const PrincipledSpecularStateInput &input) noexcept {
    const auto incoming = safe_normalize(
        input.incoming, input.surface_shading_normal);
    const auto correction_enabled =
        input.use_bump_map_correction &
        !all(input.surface_geometric_normal == input.normal);
    const auto glossy_normal = select(
        input.normal,
        ensure_valid_specular_reflection(
            input.surface_geometric_normal,
            incoming,
            input.normal),
        correction_enabled);
    return {
        .glossy_normal = glossy_normal,
        .incoming_cosine = clamp(
            dot(glossy_normal, incoming), 0.0f, 1.0f),
        .roughness = clamp(input.roughness, 0.0f, 1.0f),
        .specular_tint = max(
            input.specular_tint, make_float3(0.0f))};
}

}// namespace psycles::luisa_backend::detail
