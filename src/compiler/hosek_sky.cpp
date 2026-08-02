#include "hosek_sky.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <sky_hosek.h>

namespace psycles::compiler::detail {

[[nodiscard]] std::vector<float> cook_hosek_wilkie_sky(
    float sun_x,
    float sun_y,
    float sun_z,
    float turbidity,
    float ground_albedo) {
    constexpr auto pi_over_two =
        1.57079632679489661923;
    const auto length = std::sqrt(
        static_cast<double>(sun_x) * sun_x +
        static_cast<double>(sun_y) * sun_y +
        static_cast<double>(sun_z) * sun_z);
    const auto inverse_length =
        length > 1.0e-20 ? 1.0 / length : 0.0;
    const std::array normalized_direction{
        static_cast<float>(sun_x * inverse_length),
        static_cast<float>(sun_y * inverse_length),
        static_cast<float>(
            length > 1.0e-20
                ? sun_z * inverse_length
                : 1.0)};

    // Cycles clamps a legacy Hosek sun to the horizon before retaining its
    // spherical direction. Reconstruct the Cartesian direction from those
    // clamped angles so gamma uses the same sun as the coefficient cook.
    const auto theta = std::clamp(
        std::acos(std::clamp(
            static_cast<double>(normalized_direction[2]),
            -1.0,
            1.0)),
        0.0,
        pi_over_two);
    const auto phi = std::atan2(
        static_cast<double>(normalized_direction[0]),
        static_cast<double>(normalized_direction[1]));
    const std::array sun_direction{
        static_cast<float>(std::sin(theta) * std::sin(phi)),
        static_cast<float>(std::sin(theta) * std::cos(phi)),
        static_cast<float>(std::cos(theta))};

    // Blender exposes this model over the physically tabulated domain.
    // Keep the host-side cook inside that domain before indexing the
    // upstream Hosek-Wilkie coefficient tables.
    const auto bounded_turbidity = std::clamp(
        static_cast<double>(turbidity), 1.0, 10.0);
    const auto bounded_albedo = std::clamp(
        static_cast<double>(ground_albedo), 0.0, 1.0);
    const auto solar_elevation = pi_over_two - theta;

    auto *state = SKY_arhosek_xyz_skymodelstate_alloc_init(
        bounded_turbidity,
        bounded_albedo,
        solar_elevation);
    if (state == nullptr) {
        return {};
    }

    std::vector<float> result;
    result.reserve(33u);
    result.insert(
        result.end(),
        sun_direction.begin(),
        sun_direction.end());
    for (std::size_t channel = 0u; channel < 3u; ++channel) {
        result.emplace_back(
            static_cast<float>(state->radiances[channel]));
    }
    for (std::size_t channel = 0u; channel < 3u; ++channel) {
        for (std::size_t coefficient = 0u;
             coefficient < 9u;
             ++coefficient) {
            result.emplace_back(static_cast<float>(
                state->configs[channel][coefficient]));
        }
    }
    SKY_arhosekskymodelstate_free(state);
    return result;
}

}// namespace psycles::compiler::detail
