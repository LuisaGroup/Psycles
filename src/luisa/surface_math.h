#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept;

}// namespace psycles::luisa_backend::detail
