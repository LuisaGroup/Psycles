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
    const auto glass_payload =
        (closure.kind == static_cast<std::uint32_t>(
                             SurfaceClosureKind::glass)) |
        (closure.kind == static_cast<std::uint32_t>(
                             SurfaceClosureKind::refraction));
    const auto bssrdf_payload =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::bssrdf);

    // block_0 is the common tagged record. Glass uses the otherwise
    // unobservable Color lanes for evaluation_scale; its Color is preserved
    // in the three spare scalar lanes of the dielectric payload below.
    const auto common_color = select(
        closure.color, closure.evaluation_scale, glass_payload);

    auto payload_0 = make_float4(
        closure.specular_tint,
        closure.diffuse_roughness);
    auto payload_1 = make_float4(
        closure.evaluation_scale,
        closure.metallic);
    auto payload_2 = make_float4(
        closure.sheen_transform_a,
        closure.sheen_transform_b,
        closure.ior,
        0.0f);
    Float4 payload_3 = make_float4(0.0f);

    payload_0 = select(
        payload_0,
        make_float4(closure.fresnel_f0, closure.ior),
        glass_payload);
    payload_1 = select(
        payload_1,
        make_float4(closure.fresnel_f90, closure.color.x),
        glass_payload);
    payload_2 = select(
        payload_2,
        make_float4(closure.reflection_tint, closure.color.y),
        glass_payload);
    payload_3 = select(
        payload_3,
        make_float4(closure.transmission_tint, closure.color.z),
        glass_payload);

    payload_0 = select(
        payload_0,
        make_float4(
            closure.bssrdf_radius,
            closure.bssrdf_anisotropy),
        bssrdf_payload);
    payload_1 = select(
        payload_1,
        make_float4(
            closure.bssrdf_albedo,
            closure.bssrdf_roughness),
        bssrdf_payload);
    payload_2 = select(
        payload_2,
        make_float4(
            closure.bssrdf_ior,
            closure.ior,
            0.0f,
            0.0f),
        bssrdf_payload);
    payload_3 = select(
        payload_3,
        make_float4(0.0f),
        bssrdf_payload);
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
                common_color,
                closure.sample_weight),
            make_float4(
                closure.normal,
                closure.roughness)),
        .block_1 = make_float4x4(
            payload_0,
            payload_1,
            payload_2,
            payload_3)};
}

SurfaceClosurePhysicalCommonRecord
unpack_surface_closure_physical_common(
    Expr<luisa::float4x4> block_0_expression) noexcept {
    const auto block_0 =
        luisa::compute::Float4x4{block_0_expression};
    const auto identity =
        block_0[0u].bitcast<luisa::uint4>();
    const auto flags = identity.z;
    return {
        .kind = identity.x,
        .lobe = identity.y,
        .weight = block_0[1u].xyz(),
        .allocation_weight = block_0[1u].w,
        .sample_weight = block_0[2u].w,
        .setup_valid = (flags & setup_valid_bit) != 0u,
        .color_or_evaluation_scale = block_0[2u].xyz(),
        .normal = block_0[3u].xyz(),
        .roughness = block_0[3u].w,
        .preserve_ggx_energy =
            (flags & preserve_ggx_energy_bit) != 0u,
        .beckmann = (flags & beckmann_bit) != 0u,
        .bssrdf_method = identity.w};
}

SurfaceClosurePhysicalCommonRecord
project_surface_closure_physical_common(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .kind = closure.kind,
        .lobe = closure.lobe,
        .weight = closure.weight,
        .allocation_weight = closure.allocation_weight,
        .sample_weight = closure.sample_weight,
        .setup_valid = closure.setup_valid,
        .color_or_evaluation_scale = select(
            closure.color,
            closure.evaluation_scale,
            (closure.kind == static_cast<std::uint32_t>(
                                 SurfaceClosureKind::glass)) |
                (closure.kind == static_cast<std::uint32_t>(
                                     SurfaceClosureKind::refraction))),
        .normal = closure.normal,
        .roughness = closure.roughness,
        .preserve_ggx_energy = closure.preserve_ggx_energy,
        .beckmann = closure.beckmann,
        .bssrdf_method = closure.bssrdf_method};
}

SurfaceClosurePhysicalCommonOnlyRecord
project_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {.common = project_surface_closure_physical_common(closure)};
}

SurfaceClosurePhysicalGeneralRecord
project_surface_closure_physical_general(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .common = project_surface_closure_physical_common(closure),
        .payload = {
            .diffuse_roughness = closure.diffuse_roughness,
            .metallic = closure.metallic,
            .ior = closure.ior,
            .specular_tint = closure.specular_tint,
            .sheen_transform_a = closure.sheen_transform_a,
            .sheen_transform_b = closure.sheen_transform_b,
            .evaluation_scale = closure.evaluation_scale}};
}

SurfaceClosurePhysicalDielectricRecord
project_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .common = project_surface_closure_physical_common(closure),
        .payload = {
            .color = closure.color,
            .ior = closure.ior,
            .fresnel_f0 = closure.fresnel_f0,
            .fresnel_f90 = closure.fresnel_f90,
            .reflection_tint = closure.reflection_tint,
            .transmission_tint = closure.transmission_tint}};
}

SurfaceClosurePhysicalBssrdfRecord
project_surface_closure_physical_bssrdf(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .common = project_surface_closure_physical_common(closure),
        .payload = {
            .radius = closure.bssrdf_radius,
            .albedo = closure.bssrdf_albedo,
            .bssrdf_ior = closure.bssrdf_ior,
            .roughness = closure.bssrdf_roughness,
            .anisotropy = closure.bssrdf_anisotropy}};
}

SurfaceClosurePhysicalCommonOnlyRecord
unpack_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalCommonRecord &common) noexcept {
    return {.common = common};
}

SurfaceClosurePhysicalGeneralRecord
unpack_surface_closure_physical_general(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    return {
        .common = common,
        .payload = {
            .diffuse_roughness = block_1[0u].w,
            .metallic = block_1[1u].w,
            .ior = block_1[2u].z,
            .specular_tint = block_1[0u].xyz(),
            .sheen_transform_a = block_1[2u].x,
            .sheen_transform_b = block_1[2u].y,
            .evaluation_scale = block_1[1u].xyz()}};
}

SurfaceClosurePhysicalDielectricRecord
unpack_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    return {
        .common = common,
        .payload = {
            .color = make_float3(
                block_1[1u].w,
                block_1[2u].w,
                block_1[3u].w),
            .ior = block_1[0u].w,
            .fresnel_f0 = block_1[0u].xyz(),
            .fresnel_f90 = block_1[1u].xyz(),
            .reflection_tint = block_1[2u].xyz(),
            .transmission_tint = block_1[3u].xyz()}};
}

SurfaceClosurePhysicalBssrdfRecord
unpack_surface_closure_physical_bssrdf(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    return {
        .common = common,
        .payload = {
            .radius = block_1[0u].xyz(),
            .albedo = block_1[1u].xyz(),
            .bssrdf_ior = block_1[2u].x,
            .roughness = block_1[1u].w,
            .anisotropy = block_1[0u].w}};
}

SurfaceClosurePhysicalRecord
unpack_surface_closure_physical_payload(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    const auto glass_payload =
        (common.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::glass)) |
        (common.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::refraction));
    const auto bssrdf_payload =
        common.kind == static_cast<std::uint32_t>(
                           SurfaceClosureKind::bssrdf);
    const auto specialized_payload =
        glass_payload | bssrdf_payload;
    const auto zero3 = make_float3(0.0f);
    const auto glass_color = make_float3(
        block_1[1u].w,
        block_1[2u].w,
        block_1[3u].w);
    const auto general_ior = block_1[2u].z;
    const auto bssrdf_ior = block_1[2u].x;
    return {
        .kind = common.kind,
        .lobe = common.lobe,
        .weight = common.weight,
        .allocation_weight = common.allocation_weight,
        .sample_weight = common.sample_weight,
        .setup_valid = common.setup_valid,
        .color = select(
            common.color_or_evaluation_scale,
            glass_color,
            glass_payload),
        .normal = common.normal,
        .roughness = common.roughness,
        .diffuse_roughness = select(
            block_1[0u].w, 0.0f, specialized_payload),
        .metallic = select(
            block_1[1u].w, 0.0f, specialized_payload),
        .ior = select(
            select(general_ior, block_1[0u].w, glass_payload),
            block_1[2u].y,
            bssrdf_payload),
        .specular_tint = select(
            block_1[0u].xyz(), zero3, specialized_payload),
        .sheen_transform_a = select(
            block_1[2u].x, 0.0f, specialized_payload),
        .sheen_transform_b = select(
            block_1[2u].y, 0.0f, specialized_payload),
        .evaluation_scale = select(
            select(
                block_1[1u].xyz(),
                common.color_or_evaluation_scale,
                glass_payload),
            make_float3(1.0f),
            bssrdf_payload),
        .fresnel_f0 = select(
            zero3, block_1[0u].xyz(), glass_payload),
        .fresnel_f90 = select(
            zero3, block_1[1u].xyz(), glass_payload),
        .reflection_tint = select(
            zero3, block_1[2u].xyz(), glass_payload),
        .transmission_tint = select(
            zero3, block_1[3u].xyz(), glass_payload),
        .preserve_ggx_energy = common.preserve_ggx_energy,
        .beckmann = common.beckmann,
        .bssrdf_method = common.bssrdf_method,
        .bssrdf_radius = select(
            zero3, block_1[0u].xyz(), bssrdf_payload),
        .bssrdf_albedo = select(
            zero3, block_1[1u].xyz(), bssrdf_payload),
        .bssrdf_ior = select(
            1.4f, bssrdf_ior, bssrdf_payload),
        .bssrdf_roughness = select(
            1.0f, block_1[1u].w, bssrdf_payload),
        .bssrdf_anisotropy = select(
            0.0f, block_1[0u].w, bssrdf_payload)};
}

SurfaceClosurePhysicalRecord unpack_surface_closure_physical(
    Expr<luisa::float4x4> block_0_expression,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto common = unpack_surface_closure_physical_common(
        block_0_expression);
    return unpack_surface_closure_physical_payload(
        common, block_1_expression);
}

}// namespace psycles::luisa_backend
