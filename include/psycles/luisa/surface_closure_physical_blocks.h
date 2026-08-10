#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_physical_blocks.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

#include <cstddef>

namespace psycles::luisa_backend {

// Lossless callable ABI for the exact post-population physical projection.
// Its 48 scalar lanes fill three matrices exactly. A fourth block would mean
// that setup/AOV state crossed the dependency cut again and is guarded by the
// focused regression.
inline constexpr std::size_t
    surface_closure_physical_block_count = 3u;

struct SurfaceClosurePhysicalBlocks {
    luisa::compute::Float4x4 block_0;
    luisa::compute::Float4x4 block_1;
    luisa::compute::Float4x4 block_2;
};

[[nodiscard]] SurfaceClosurePhysicalBlocks
pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float4x4> block_2) noexcept;

}// namespace psycles::luisa_backend
