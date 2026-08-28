#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_physical_blocks.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

#include <cstddef>
#include <functional>

namespace psycles::luisa_backend {

// Tagged-union callable ABI for the exact post-population physical
// projection. Its discrete identity is the same pair used by Cycles'
// ShaderClosure/MicrofacetBsdf: ClosureType and MicrofacetFresnel. Authoring
// kind/lobe, allocation state, setup flags, distribution flags and BSSRDF
// method are deliberately absent: after retention they are either encoded by
// that pair, represented by retained-prefix membership, or unobservable.
// Fields outside the selected payload family are canonicalized to zero. The
// two matrices contain 32 scalar lanes. This is sufficient because the widest
// members are general and glass/refraction:
//
//   identity/reserved 4 + weight/reserved 4 + sample weight 1 +
//   normal/roughness 4 + color 3 + payload 16 = 32.
//
// For every tag k, pack is injective on the fields observable for k and
// unpack(pack(x)) restricted to those fields equals x. Retaining an
// unobservable field inside the selected conservative family is harmless;
// fields belonging only to another family never alias it. Thin-film lanes
// are further canonicalized by the exact observable tags: Principled
// Metallic/Dielectric, standalone Metallic F82/Conductor, and Glass. A third
// block would mean mutually exclusive family payloads had been combined again
// and is guarded by the focused regression.
inline constexpr std::size_t
    surface_closure_physical_block_count = 2u;

struct SurfaceClosurePhysicalBlocks {
    luisa::compute::Float4x4 block_0;
    luisa::compute::Float4x4 block_1;
};

// Host/JIT thunk for a physical payload read. Consumers invoke this only
// while recording the matching runtime family branch, so a Local read (or any
// equivalent storage operation) is dominated by the tag test in the emitted
// device CFG. This is deliberately not a device callable or an IR entity.
using SurfaceClosurePhysicalPayloadLoader =
    std::function<luisa::compute::Float4x4()>;

// Decoded discriminants and fields shared by every member of the physical
// closure tagged union. `closure_type` is the post-setup Cycles ClosureType;
// `microfacet_fresnel` is Cycles' MicrofacetFresnel ABI value and is none for
// closure families which do not observe it. The last vector is intentionally
// named after its representation rather than either logical interpretation:
// block_0 stores
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
    UInt closure_type;
    UInt microfacet_fresnel;
    Float3 weight;
    Float sample_weight;
    Float3 color_or_evaluation_scale;
    Float3 normal;
    Float roughness;
};

// The physical closure representation is a tagged sum, not a product of all
// possible payloads. These types make that distinction explicit at the C++
// staging boundary:
//
//   Physical = Common x (Unit + General + Hair + Dielectric + Bssrdf).
//
// A family consumer receives exactly one of the records below. It therefore
// cannot name a field owned by another family, even accidentally. These are
// host-side bundles of Luisa expressions; they do not define an additional
// device-memory ABI and do not introduce an IR entity.
struct SurfaceClosurePhysicalCommonOnlyRecord {
    SurfaceClosurePhysicalCommonRecord common;
};

struct SurfaceClosurePhysicalGeneralPayload {
    // Post-setup scattering never observes authored diffuse roughness or
    // metallic weight: closure identity/lobe already encodes the selected
    // branch. Their former lanes carry the film parameters without growing
    // the two-block tagged-union ABI.
    Float thin_film_thickness;
    Float thin_film_ior;
    Float ior;
    Float3 specular_tint;
    Float sheen_transform_a;
    Float sheen_transform_b;
    Float3 evaluation_scale;
    Float3 microfacet_tangent;
    Float microfacet_alpha_x;
    Float microfacet_alpha_y;
};

struct SurfaceClosurePhysicalGeneralRecord {
    SurfaceClosurePhysicalCommonRecord common;
    SurfaceClosurePhysicalGeneralPayload payload;
};

struct SurfaceClosurePhysicalHairPayload {
    Float3 tangent;
    Float roughness_u;
    Float roughness_v;
    Float offset;
};

struct SurfaceClosurePhysicalHairRecord {
    SurfaceClosurePhysicalCommonRecord common;
    SurfaceClosurePhysicalHairPayload payload;
};

struct SurfaceClosurePhysicalDielectricPayload {
    Float ior;
    // Glass color is fully represented by reflection/transmission tint and
    // evaluation scale after setup. Two of its former spare lanes carry the
    // film parameters; the third remains canonical zero.
    Float thin_film_thickness;
    Float thin_film_ior;
    Float3 fresnel_f0;
    Float3 fresnel_f90;
    Float3 reflection_tint;
    Float3 transmission_tint;
};

struct SurfaceClosurePhysicalDielectricRecord {
    SurfaceClosurePhysicalCommonRecord common;
    SurfaceClosurePhysicalDielectricPayload payload;
};

struct SurfaceClosurePhysicalBssrdfPayload {
    Float3 radius;
    Float3 albedo;
    Float bssrdf_ior;
    Float roughness;
    Float anisotropy;
};

struct SurfaceClosurePhysicalBssrdfRecord {
    SurfaceClosurePhysicalCommonRecord common;
    SurfaceClosurePhysicalBssrdfPayload payload;
};

// Expression-only projections used by direct/expanded consumers. They obey
// the same family interfaces as the packed-storage decoders below, so the
// scattering algebra has one implementation regardless of its producer.
[[nodiscard]] SurfaceClosurePhysicalCommonRecord
project_surface_closure_physical_common(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalCommonOnlyRecord
project_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalGeneralRecord
project_surface_closure_physical_general(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalHairRecord
project_surface_closure_physical_hair(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalDielectricRecord
project_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalBssrdfRecord
project_surface_closure_physical_bssrdf(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalBlocks
pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

[[nodiscard]] SurfaceClosurePhysicalCommonRecord
unpack_surface_closure_physical_common(
    Expr<luisa::float4x4> block_0) noexcept;

// Conservative payload classes are a function of the retained post-setup
// ClosureType. The predicates are pairwise disjoint over the represented
// Cycles type domain; type_none and common-only closures match none. Keeping
// one predicate per class lets host reachability omit impossible checks while
// recording a specialized shader AST.
[[nodiscard]] Bool surface_closure_uses_general_payload(
    UInt closure_type) noexcept;
[[nodiscard]] Bool surface_closure_uses_hair_payload(
    UInt closure_type) noexcept;
[[nodiscard]] Bool surface_closure_uses_dielectric_payload(
    UInt closure_type) noexcept;
[[nodiscard]] Bool surface_closure_uses_bssrdf_payload(
    UInt closure_type) noexcept;

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
[[nodiscard]] SurfaceClosurePhysicalCommonOnlyRecord
unpack_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalCommonRecord &common) noexcept;

[[nodiscard]] SurfaceClosurePhysicalGeneralRecord
unpack_surface_closure_physical_general(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalHairRecord
unpack_surface_closure_physical_hair(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalDielectricRecord
unpack_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

[[nodiscard]] SurfaceClosurePhysicalBssrdfRecord
unpack_surface_closure_physical_bssrdf(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1) noexcept;

}// namespace psycles::luisa_backend
