#pragma once

#include <array>
#include <cstddef>

#include <psycles/contract/render.h>

namespace psycles::sampling {

// Cycles stores reconstruction-filter inverse CDFs in an interpolated lookup
// table. The even size and symmetric construction have observable endpoint
// behavior, so this is part of the camera sampling contract rather than an
// interchangeable numerical approximation.
inline constexpr std::size_t pixel_filter_table_size = 1024u;
using PixelFilterTable = std::array<float, pixel_filter_table_size>;

[[nodiscard]] PixelFilterTable
make_pixel_filter_table(contract::PixelFilter filter, float width);

} // namespace psycles::sampling
