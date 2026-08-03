#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_wave.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_wave {

// Wave Texture uses the normalized three-dimensional Cycles fBm directly,
// without the Noise Texture node's input clamps or coordinate distortion.
// Keep that expensive AST in one shared Callable so every material and both
// Wave outputs reuse the same device function.
void prepare_distortion_noise() noexcept;

[[nodiscard]] Float distortion_noise(
    Float3 point,
    Float detail,
    Float roughness) noexcept;

}// namespace psycles::luisa_backend::cycles_wave
