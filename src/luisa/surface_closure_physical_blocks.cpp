#include <psycles/luisa/surface_closure_physical_blocks.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr std::uint32_t setup_valid_bit = 1u << 0u;
inline constexpr std::uint32_t preserve_ggx_energy_bit = 1u << 1u;
inline constexpr std::uint32_t beckmann_bit = 1u << 2u;

}// namespace

SurfaceClosurePhysicalBlocks pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
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
                closure.color,
                closure.sample_weight),
            make_float4(
                closure.normal,
                closure.roughness)),
        .block_1 = make_float4x4(
            make_float4(
                closure.specular_tint,
                closure.diffuse_roughness),
            make_float4(
                closure.evaluation_scale,
                closure.metallic),
            make_float4(
                closure.fresnel_f0,
                closure.ior),
            make_float4(
                closure.fresnel_f90,
                closure.sheen_transform_a)),
        .block_2 = make_float4x4(
            make_float4(
                closure.reflection_tint,
                closure.sheen_transform_b),
            make_float4(
                closure.transmission_tint,
                closure.bssrdf_ior),
            make_float4(
                closure.bssrdf_radius,
                closure.bssrdf_anisotropy),
            make_float4(
                closure.bssrdf_albedo,
                closure.bssrdf_roughness))};
}

SurfaceClosurePhysicalRecord unpack_surface_closure_physical(
    Expr<luisa::float4x4> block_0_expression,
    Expr<luisa::float4x4> block_1_expression,
    Expr<luisa::float4x4> block_2_expression) noexcept {
    const auto block_0 =
        luisa::compute::Float4x4{block_0_expression};
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    const auto block_2 =
        luisa::compute::Float4x4{block_2_expression};
    const auto identity =
        block_0[0u].bitcast<luisa::uint4>();
    const auto flags = identity.z;
    return {
        .kind = identity.x,
        .lobe = identity.y,
        .weight = block_0[1u].xyz(),
        .allocation_weight = block_0[1u].w,
        .sample_weight = block_0[2u].w,
        .setup_valid =
            (flags & setup_valid_bit) != 0u,
        .color = block_0[2u].xyz(),
        .normal = block_0[3u].xyz(),
        .roughness = block_0[3u].w,
        .diffuse_roughness = block_1[0u].w,
        .metallic = block_1[1u].w,
        .ior = block_1[2u].w,
        .specular_tint = block_1[0u].xyz(),
        .sheen_transform_a = block_1[3u].w,
        .sheen_transform_b = block_2[0u].w,
        .evaluation_scale = block_1[1u].xyz(),
        .fresnel_f0 = block_1[2u].xyz(),
        .fresnel_f90 = block_1[3u].xyz(),
        .reflection_tint = block_2[0u].xyz(),
        .transmission_tint = block_2[1u].xyz(),
        .preserve_ggx_energy =
            (flags & preserve_ggx_energy_bit) != 0u,
        .beckmann =
            (flags & beckmann_bit) != 0u,
        .bssrdf_method = identity.w,
        .bssrdf_radius = block_2[2u].xyz(),
        .bssrdf_albedo = block_2[3u].xyz(),
        .bssrdf_ior = block_2[1u].w,
        .bssrdf_roughness = block_2[3u].w,
        .bssrdf_anisotropy = block_2[2u].w};
}

}// namespace psycles::luisa_backend
