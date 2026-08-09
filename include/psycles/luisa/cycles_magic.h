#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_magic.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_magic {

inline constexpr std::uint32_t maximum_depth = 10u;

// Depth is material data. One callable retains a structured runtime loop so
// authored depth never creates a new shader AST or cache identity.
void prepare() noexcept;

[[nodiscard]] Float3 evaluate(
    UInt depth,
    Float3 vector,
    Float scale,
    Float distortion) noexcept;

}// namespace psycles::luisa_backend::cycles_magic
