#pragma once

#include <psycles/core/math.h>
#include <luisa/dsl/sugar.h>

#include <cmath>
#include <limits>

namespace psycles::luisa_backend {

// Cycles scene/light.cpp::copy_to_kernel owns these values. They are rebuilt
// with immutable light/transform scene data, never derived per shading point.
struct AreaLightParameters {
    float tan_half_spread{};
    float normalize_spread{};
};

struct SpotLightParameters {
    float cos_half_spot_angle{};
    float half_cot_half_spot_angle{};
    float spot_smooth{}; // Reciprocal blend width, not the authored smooth.
    float cos_half_larger_spread{};
    float ray_segment_dp{};
};

[[nodiscard]] inline AreaLightParameters make_area_light_parameters(float spread) noexcept {
    constexpr float pi = 3.1415926535897932f;
    constexpr float maximum = std::numeric_limits<float>::max();
    const auto half = 0.5f * std::fmax(spread, 0.0f);
    const auto tangent = spread == pi ? maximum : std::tan(half);
    const auto normalization = half > 0.0f ?
        (half > 0.05f ? 1.0f / (tangent - half) : 3.0f / std::pow(half, 3.0f)) : maximum;
    return {tangent, normalization};
}

[[nodiscard]] inline SpotLightParameters make_spot_light_parameters(
    float angle, float smooth, float radius, Vec3f axis_x, Vec3f axis_y, Vec3f axis_z) noexcept {
    const auto cosine = std::cos(0.5f * angle);
    const auto tangent = std::tan(0.5f * angle);
    const auto squared_length = [](Vec3f v) { return v.x*v.x + v.y*v.y + v.z*v.z; };
    const auto x = squared_length(axis_x), y = squared_length(axis_y), z = squared_length(axis_z);
    const auto t2 = tangent * tangent;
    return {cosine, 0.5f / tangent, 1.0f / ((1.0f - cosine) * smooth),
            1.0f / std::sqrt(1.0f + t2 * std::fmax(x, y) / z),
            radius * std::sqrt(1.0f + z / (t2 * std::fmin(x, y)))};
}
} // namespace psycles::luisa_backend

LUISA_STRUCT(psycles::luisa_backend::AreaLightParameters, tan_half_spread, normalize_spread) {};
LUISA_STRUCT(psycles::luisa_backend::SpotLightParameters, cos_half_spot_angle,
             half_cot_half_spot_angle, spot_smooth, cos_half_larger_spread, ray_segment_dp) {};
