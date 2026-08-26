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

// Decoded discriminant and fields shared by every member of the physical
// closure tagged union. The last vector is intentionally named after its
// representation rather than either logical interpretation: block_0 stores
// Color for ordinary/BSSRDF closures and evaluation_scale for the dielectric
// family. Decoding that lane is independent of block_1; choosing its semantic
// meaning is part of payload elimination below.
//
// Keeping this type separate is a staging invariant, not merely a compact
// storage convenience. A consumer may inspect the tag and common fields
// without introducing any expression which depends on block_1. This lets a
// runtime kind branch dominate the mutually exclusive payload loads, matching
// the demand-loaded ShaderClosure representation used by Cycles.
struct SurfaceClosurePhysicalCommonRecord {
    UInt kind;
    UInt lobe;
    Float3 weight;
    Float allocation_weight;
    Float sample_weight;
    Bool setup_valid;
    Float3 color_or_evaluation_scale;
    Float3 normal;
    Float roughness;
    Bool preserve_ggx_energy;
    Bool beckmann;
    UInt bssrdf_method;
};

[[nodiscard]] SurfaceClosurePhysicalBlocks
pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalCommonRecord
unpack_surface_closure_physical_common(
    Expr<luisa::float4x4> block_0) noexcept;

// Eliminators for the physical tagged union. Let P_k be the projection onto
// fields observable by closure kind k. For every encoded physical closure x,
// the implementation invariant is
//
//   P_k(unpack_family_k(pack(x))) == P_k(x).
//
// The common-only eliminator deliberately has no block_1 argument. The other
// eliminators contain no runtime family selects: their host/JIT caller must
// establish the corresponding tag before recording the call. This mirrors
// Cycles' `ShaderClosure *` cast after its type switch and keeps mutually
// exclusive payload expressions out of each other's live ranges.
[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalCommonRecord &common) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_general(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_bssrdf(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

// Generic inverse retained for storage round trips and diagnostics. Hot
// consumers should dispatch on common.kind and use one eliminator above.
[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_payload(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalRecord
unpack_surface_closure_physical(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1) noexcept;

}// namespace psycles::luisa_backend
