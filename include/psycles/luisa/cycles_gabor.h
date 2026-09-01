#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_gabor.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_gabor {

// Direct Luisa DSL projection of Cycles 5.2.1's SVM Gabor evaluator. The
// components are Value, Phase, and Intensity in Cycles output order.
[[nodiscard]] Float3 evaluate(
    luisa::compute::Expr<std::uint32_t> gabor_type,
    Float3 coordinates,
    Float3 orientation_3d,
    Float scale,
    Float frequency,
    Float anisotropy,
    Float orientation_2d) noexcept;

} // namespace psycles::luisa_backend::cycles_gabor
