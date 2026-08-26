#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_physical_blocks.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

#include <cstddef>

namespace psycles::luisa_backend {

// Tagged-union callable ABI for the exact post-population physical
// projection. The (kind, lobe) tag selects a conservative payload family;
// fields outside that family are canonicalized to
// SurfaceClosureRecord::zero(). Each family contains every field observable
// by any of its member tags. The two matrices contain 32 scalar lanes. This
// is sufficient because the widest member is glass/refraction:
//
//   identity/flags 4 + weight/allocation 4 + sample weight 1 +
//   normal/roughness 4 + color 3 + dielectric payload 16 = 32.
//
// For every tag k, pack is injective on the fields observable for k and
// unpack(pack(x)) restricted to those fields equals x. Retaining an
// unobservable field inside the selected conservative family is harmless;
// fields belonging only to another family never alias it. A third block
// would mean mutually exclusive family payloads had been combined again and
// is guarded by the focused regression.
inline constexpr std::size_t
    surface_closure_physical_block_count = 2u;

struct SurfaceClosurePhysicalBlocks {
    luisa::compute::Float4x4 block_0;
    luisa::compute::Float4x4 block_1;
};

[[nodiscard]] SurfaceClosurePhysicalBlocks
pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1) noexcept;

}// namespace psycles::luisa_backend
