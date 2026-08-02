#pragma once

#include <vector>

namespace psycles::compiler::detail {

// Layout: normalized sun direction (3), XYZ radiance (3), then the
// nine Hosek-Wilkie configuration coefficients for X, Y, and Z (27).
[[nodiscard]] std::vector<float> cook_hosek_wilkie_sky(
    float sun_x,
    float sun_y,
    float sun_z,
    float turbidity,
    float ground_albedo);

}// namespace psycles::compiler::detail
