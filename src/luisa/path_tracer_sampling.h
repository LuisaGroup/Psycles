#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using PixelFilterCallable = Callable<float(float, float)>;

[[nodiscard]] PixelFilterCallable make_pixel_filter_callable(
    contract::PixelFilter pixel_filter) noexcept;

}// namespace psycles::luisa_backend::detail
