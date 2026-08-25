#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float srgb_to_linear(Float value) noexcept;
[[nodiscard]] Float3 srgb_to_linear(Float3 value) noexcept;

}// namespace psycles::luisa_backend::detail
