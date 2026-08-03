#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_blocks.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

// Lossless value ABI for passing one canonical closure across a shared Luisa
// Callable boundary. These four SSA matrix values are never dynamically
// indexed and are not a device closure array. Integer identity and Boolean
// setup bits are preserved by bitcast.
struct SurfaceClosureBlocks {
    luisa::compute::Float4x4 block_0;
    luisa::compute::Float4x4 block_1;
    luisa::compute::Float4x4 block_2;
    luisa::compute::Float4x4 block_3;
};

[[nodiscard]] SurfaceClosureBlocks pack_surface_closure(
    const SurfaceClosureRecord &closure) noexcept;

[[nodiscard]] SurfaceClosureRecord unpack_surface_closure(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float4x4> block_2,
    Expr<luisa::float4x4> block_3) noexcept;

}// namespace psycles::luisa_backend
