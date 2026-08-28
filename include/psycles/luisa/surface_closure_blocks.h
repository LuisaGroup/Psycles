#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_blocks.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

// Lossless value ABI for passing one canonical closure across a shared Luisa
// Callable boundary. These four SSA matrix values are never dynamically
// indexed and are not a device closure array. identity.x/y are the exact
// ClosureType/MicrofacetFresnel pair; integer identity and Boolean setup bits
// are preserved by bitcast. Authoring kind/lobe have no lane in this ABI.
struct SurfaceClosureBlocks {
    luisa::compute::Float4x4 block_0;
    luisa::compute::Float4x4 block_1;
    luisa::compute::Float4x4 block_2;
    luisa::compute::Float4x4 block_3;
};

// Semantic row view of the complete closure ABI. This is the sole inverse
// map from matrix lanes to closure fields: consumers which only need a row
// projection can use it without unpacking and then repacking a full record.
struct SurfaceClosureBlockRows {
    luisa::compute::UInt4 identity;
    Float4 weight_allocation_weight;
    Float4 albedo_sample_weight;
    Float4 reflection_albedo_roughness;
    Float4 transmission_albedo_diffuse_roughness;
    Float4 color_metallic;
    Float4 normal_ior;
    Float4 specular_tint_ior_level;
    Float4 evaluation_scale_sheen_transform_a;
    Float4 fresnel_f0_sheen_transform_b;
    Float4 fresnel_f90_microfacet_alpha_x;
    Float4 reflection_tint_microfacet_alpha_y;
    Float4 transmission_tint_bssrdf_ior;
    Float4 bssrdf_radius_anisotropy;
    Float4 bssrdf_albedo_roughness;
    Float4 microfacet_tangent_reserved;
};

[[nodiscard]] SurfaceClosureBlocks pack_surface_closure(
    const SurfaceClosureRecord &closure) noexcept;

[[nodiscard]] SurfaceClosureBlockRows unpack_surface_closure_block_rows(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float4x4> block_2,
    Expr<luisa::float4x4> block_3) noexcept;

[[nodiscard]] SurfaceClosureRecord unpack_surface_closure(
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float4x4> block_2,
    Expr<luisa::float4x4> block_3) noexcept;

}// namespace psycles::luisa_backend
