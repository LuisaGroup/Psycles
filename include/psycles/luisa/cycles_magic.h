#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_magic.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_magic {

inline constexpr std::uint32_t maximum_depth = 10u;

// Depth is Blender node storage and therefore known while Luisa traces the
// shader AST. One specialized Callable is emitted for each used depth; the
// authored Vector, Scale, and Distortion remain device expressions.
void prepare(std::uint32_t depth) noexcept;

[[nodiscard]] Float3 evaluate(
    std::uint32_t depth,
    Float3 vector,
    Float scale,
    Float distortion) noexcept;

}// namespace psycles::luisa_backend::cycles_magic
