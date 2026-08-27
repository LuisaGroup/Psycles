#include <psycles/luisa/surface_closure_blocks.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr std::uint32_t setup_valid_bit = 1u << 0u;
inline constexpr std::uint32_t preserve_ggx_energy_bit = 1u << 1u;
inline constexpr std::uint32_t beckmann_bit = 1u << 2u;

}// namespace

SurfaceClosureBlocks pack_surface_closure(
    const SurfaceClosureRecord &closure) noexcept {
    UInt flags = 0u;
    flags |= select(0u, setup_valid_bit, closure.setup_valid);
    flags |= select(
        0u,
        preserve_ggx_energy_bit,
        closure.preserve_ggx_energy);
    flags |= select(0u, beckmann_bit, closure.beckmann);
    return {
        .block_0 = make_float4x4(
            make_uint4(
                closure.kind,
                closure.lobe,
                flags,
                closure.bssrdf_method)
                .bitcast<luisa::float4>(),
            make_float4(
                closure.weight,
                closure.allocation_weight),
            make_float4(
                closure.albedo,
                closure.sample_weight),
            make_float4(
                closure.reflection_albedo,
                closure.roughness)),
        .block_1 = make_float4x4(
            make_float4(
                closure.transmission_albedo,
                closure.diffuse_roughness),
            make_float4(
                closure.color,
                closure.metallic),
            make_float4(
                closure.normal,
                closure.ior),
            make_float4(
                closure.specular_tint,
                closure.specular_ior_level)),
        .block_2 = make_float4x4(
            make_float4(
                closure.evaluation_scale,
                closure.sheen_transform_a),
            make_float4(
                closure.fresnel_f0,
                closure.sheen_transform_b),
            make_float4(
                closure.fresnel_f90,
                closure.microfacet_alpha_x),
            make_float4(
                closure.reflection_tint,
                closure.microfacet_alpha_y)),
        .block_3 = make_float4x4(
            make_float4(
                closure.transmission_tint,
                closure.bssrdf_ior),
            make_float4(
                closure.bssrdf_radius,
                closure.bssrdf_anisotropy),
            make_float4(
                closure.bssrdf_albedo,
                closure.bssrdf_roughness),
            make_float4(
                closure.microfacet_tangent,
                0.0f))};
}

SurfaceClosureRecord unpack_surface_closure(
    Expr<luisa::float4x4> block_0_expression,
    Expr<luisa::float4x4> block_1_expression,
    Expr<luisa::float4x4> block_2_expression,
    Expr<luisa::float4x4> block_3_expression) noexcept {
    const auto rows = unpack_surface_closure_block_rows(
        block_0_expression,
        block_1_expression,
        block_2_expression,
        block_3_expression);
    const auto flags = rows.identity.z;
    return {
        .kind = rows.identity.x,
        .lobe = rows.identity.y,
        .weight = rows.weight_allocation_weight.xyz(),
        .allocation_weight = rows.weight_allocation_weight.w,
        .sample_weight = rows.albedo_sample_weight.w,
        .setup_valid =
            (flags & setup_valid_bit) != 0u,
        .albedo = rows.albedo_sample_weight.xyz(),
        .reflection_albedo = rows.reflection_albedo_roughness.xyz(),
        .transmission_albedo =
            rows.transmission_albedo_diffuse_roughness.xyz(),
        .color = rows.color_metallic.xyz(),
        .normal = rows.normal_ior.xyz(),
        .roughness = rows.reflection_albedo_roughness.w,
        .microfacet_tangent = rows.microfacet_tangent_reserved.xyz(),
        .microfacet_alpha_x = rows.fresnel_f90_microfacet_alpha_x.w,
        .microfacet_alpha_y = rows.reflection_tint_microfacet_alpha_y.w,
        .diffuse_roughness =
            rows.transmission_albedo_diffuse_roughness.w,
        .metallic = rows.color_metallic.w,
        .ior = rows.normal_ior.w,
        .specular_ior_level = rows.specular_tint_ior_level.w,
        .specular_tint = rows.specular_tint_ior_level.xyz(),
        .sheen_transform_a = rows.evaluation_scale_sheen_transform_a.w,
        .sheen_transform_b = rows.fresnel_f0_sheen_transform_b.w,
        .evaluation_scale = rows.evaluation_scale_sheen_transform_a.xyz(),
        .fresnel_f0 = rows.fresnel_f0_sheen_transform_b.xyz(),
        .fresnel_f90 = rows.fresnel_f90_microfacet_alpha_x.xyz(),
        .reflection_tint = rows.reflection_tint_microfacet_alpha_y.xyz(),
        .transmission_tint = rows.transmission_tint_bssrdf_ior.xyz(),
        .preserve_ggx_energy =
            (flags & preserve_ggx_energy_bit) != 0u,
        .beckmann =
            (flags & beckmann_bit) != 0u,
        .bssrdf_method = rows.identity.w,
        .bssrdf_radius = rows.bssrdf_radius_anisotropy.xyz(),
        .bssrdf_albedo = rows.bssrdf_albedo_roughness.xyz(),
        .bssrdf_ior = rows.transmission_tint_bssrdf_ior.w,
        .bssrdf_roughness = rows.bssrdf_albedo_roughness.w,
        .bssrdf_anisotropy = rows.bssrdf_radius_anisotropy.w};
}

SurfaceClosureBlockRows unpack_surface_closure_block_rows(
    Expr<luisa::float4x4> block_0_expression,
    Expr<luisa::float4x4> block_1_expression,
    Expr<luisa::float4x4> block_2_expression,
    Expr<luisa::float4x4> block_3_expression) noexcept {
    const auto block_0 =
        luisa::compute::Float4x4{block_0_expression};
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    const auto block_2 =
        luisa::compute::Float4x4{block_2_expression};
    const auto block_3 =
        luisa::compute::Float4x4{block_3_expression};
    return {
        .identity = block_0[0u].bitcast<luisa::uint4>(),
        .weight_allocation_weight = block_0[1u],
        .albedo_sample_weight = block_0[2u],
        .reflection_albedo_roughness = block_0[3u],
        .transmission_albedo_diffuse_roughness = block_1[0u],
        .color_metallic = block_1[1u],
        .normal_ior = block_1[2u],
        .specular_tint_ior_level = block_1[3u],
        .evaluation_scale_sheen_transform_a = block_2[0u],
        .fresnel_f0_sheen_transform_b = block_2[1u],
        .fresnel_f90_microfacet_alpha_x = block_2[2u],
        .reflection_tint_microfacet_alpha_y = block_2[3u],
        .transmission_tint_bssrdf_ior = block_3[0u],
        .bssrdf_radius_anisotropy = block_3[1u],
        .bssrdf_albedo_roughness = block_3[2u],
        .microfacet_tangent_reserved = block_3[3u]};
}

}// namespace psycles::luisa_backend
