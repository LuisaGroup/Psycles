#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Cycles' dielectric Fresnel primitive shared by closure setup and direct SVM
// context evaluation. This declaration deliberately has no dependency on the
// legacy graph-value object model.
[[nodiscard]] Float fresnel_dielectric_cos(Float cosine, Float eta) noexcept;

} // namespace psycles::luisa_backend::detail
