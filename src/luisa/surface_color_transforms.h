#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Canonical Cycles-compatible implementations. These functions are the
// single semantic source used by both standalone GraphSurface lowering and
// production shared callables.
[[nodiscard]] Float3 rgb_to_hsv_inline(Float3 rgb) noexcept;
[[nodiscard]] Float3 hsv_to_rgb_inline(Float3 hsv) noexcept;
[[nodiscard]] Float3 rgb_to_hsl_inline(Float3 rgb) noexcept;
[[nodiscard]] Float3 hsl_to_rgb_inline(Float3 hsl) noexcept;

}// namespace psycles::luisa_backend::detail
