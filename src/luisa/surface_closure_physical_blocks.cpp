#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

SurfaceClosurePhysicalBlocks pack_surface_closure_physical(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    const UInt closure_type{closure.closure_type};
    const UInt microfacet_fresnel{closure.microfacet_fresnel};
    const auto general_payload =
        surface_closure_uses_general_payload(closure_type);
    const auto hair_payload =
        surface_closure_uses_hair_payload(closure_type);
    const auto dielectric_payload =
        surface_closure_uses_dielectric_payload(closure_type);
    const auto bssrdf_payload =
        surface_closure_uses_bssrdf_payload(closure_type);
    const auto general_film_payload =
        general_payload &
        cycles_closure::fresnel_uses_thin_film_payload(
            microfacet_fresnel);
    const auto glass_film_payload =
        cycles_closure::is_glass_microfacet(closure_type);

    // block_0 is the common tagged record. Glass uses the otherwise
    // unobservable Color lanes for evaluation_scale. Its post-setup Color is
    // unobservable: reflection/transmission tint are the physical inputs.
    const auto common_color = select(
        closure.color, closure.evaluation_scale, dielectric_payload);

    auto payload_0 = select(make_float4(0.0f), make_float4(
        closure.specular_tint,
        select(0.0f,
               closure.thin_film_thickness,
               general_film_payload)), general_payload);
    auto payload_1 = select(make_float4(0.0f), make_float4(
        closure.evaluation_scale,
        select(0.0f, closure.thin_film_ior, general_film_payload)),
        general_payload);
    auto payload_2 = select(make_float4(0.0f), make_float4(
        closure.sheen_transform_a,
        closure.sheen_transform_b,
        closure.ior,
        closure.microfacet_alpha_x), general_payload);
    Float4 payload_3 = select(make_float4(0.0f), make_float4(
        closure.microfacet_tangent, closure.microfacet_alpha_y),
        general_payload);

    payload_2 = select(
        payload_2,
        make_float4(
            closure.sheen_transform_a,
            0.0f,
            0.0f,
            closure.microfacet_alpha_x),
        hair_payload);
    payload_3 = select(
        payload_3,
        make_float4(
            closure.microfacet_tangent,
            closure.microfacet_alpha_y),
        hair_payload);

    payload_0 = select(
        payload_0,
        make_float4(closure.fresnel_f0, closure.ior),
        dielectric_payload);
    payload_1 = select(
        payload_1,
        make_float4(
            closure.fresnel_f90,
            select(0.0f,
                   closure.thin_film_thickness,
                   glass_film_payload)),
        dielectric_payload);
    payload_2 = select(
        payload_2,
        make_float4(
            closure.reflection_tint,
            select(0.0f, closure.thin_film_ior, glass_film_payload)),
        dielectric_payload);
    payload_3 = select(
        payload_3,
        make_float4(closure.transmission_tint, 0.0f),
        dielectric_payload);

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
            0.0f,
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
                closure_type,
                microfacet_fresnel,
                0u,
                0u)
                .bitcast<luisa::float4>(),
            make_float4(
                closure.weight,
                0.0f),
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
    return {
        .closure_type = identity.x,
        .microfacet_fresnel = identity.y,
        .weight = block_0[1u].xyz(),
        .sample_weight = block_0[2u].w,
        .color_or_evaluation_scale = block_0[2u].xyz(),
        .normal = block_0[3u].xyz(),
        .roughness = block_0[3u].w};
}

Bool surface_closure_uses_general_payload(
    UInt closure_type) noexcept {
    return cycles_closure::is_reflection_microfacet(closure_type) |
           (closure_type == cycles_closure::type_sheen) |
           (closure_type ==
            cycles_closure::type_thin_glass_transmission);
}

Bool surface_closure_uses_hair_payload(
    UInt closure_type) noexcept {
    return cycles_closure::is_hair(closure_type);
}

Bool surface_closure_uses_dielectric_payload(
    UInt closure_type) noexcept {
    return cycles_closure::is_refraction_microfacet(closure_type) |
           cycles_closure::is_glass_microfacet(closure_type);
}

Bool surface_closure_uses_bssrdf_payload(
    UInt closure_type) noexcept {
    return cycles_closure::is_bssrdf(closure_type);
}

SurfaceClosurePhysicalCommonRecord
project_surface_closure_physical_common(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    const UInt closure_type{closure.closure_type};
    return {
        .closure_type = closure_type,
        .microfacet_fresnel = closure.microfacet_fresnel,
        .weight = closure.weight,
        .sample_weight = closure.sample_weight,
        .color_or_evaluation_scale = select(
            closure.color,
            closure.evaluation_scale,
            surface_closure_uses_dielectric_payload(closure_type)),
        .normal = closure.normal,
        .roughness = closure.roughness};
}

SurfaceClosurePhysicalCommonOnlyRecord
project_surface_closure_physical_common_only(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {.common = project_surface_closure_physical_common(closure)};
}

SurfaceClosurePhysicalGeneralRecord
project_surface_closure_physical_general(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    const auto common = project_surface_closure_physical_common(closure);
    const auto film_payload =
        cycles_closure::fresnel_uses_thin_film_payload(
            common.microfacet_fresnel);
    return {
        .common = common,
        .payload = {
            .thin_film_thickness = select(
                0.0f, closure.thin_film_thickness, film_payload),
            .thin_film_ior = select(
                0.0f, closure.thin_film_ior, film_payload),
            .ior = closure.ior,
            .specular_tint = closure.specular_tint,
            .sheen_transform_a = closure.sheen_transform_a,
            .sheen_transform_b = closure.sheen_transform_b,
            .evaluation_scale = closure.evaluation_scale,
            .microfacet_tangent = closure.microfacet_tangent,
            .microfacet_alpha_x = closure.microfacet_alpha_x,
            .microfacet_alpha_y = closure.microfacet_alpha_y}};
}

SurfaceClosurePhysicalHairRecord
project_surface_closure_physical_hair(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .common = project_surface_closure_physical_common(closure),
        .payload = {
            .tangent = closure.microfacet_tangent,
            .roughness_u = closure.microfacet_alpha_x,
            .roughness_v = closure.microfacet_alpha_y,
            .offset = closure.sheen_transform_a}};
}

SurfaceClosurePhysicalDielectricRecord
project_surface_closure_physical_dielectric(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    const auto common = project_surface_closure_physical_common(closure);
    const auto film_payload =
        cycles_closure::is_glass_microfacet(common.closure_type);
    return {
        .common = common,
        .payload = {
            .ior = closure.ior,
            .thin_film_thickness = select(
                0.0f, closure.thin_film_thickness, film_payload),
            .thin_film_ior = select(
                0.0f, closure.thin_film_ior, film_payload),
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
            .thin_film_thickness = block_1[0u].w,
            .thin_film_ior = block_1[1u].w,
            .ior = block_1[2u].z,
            .specular_tint = block_1[0u].xyz(),
            .sheen_transform_a = block_1[2u].x,
            .sheen_transform_b = block_1[2u].y,
            .evaluation_scale = block_1[1u].xyz(),
            .microfacet_tangent = block_1[3u].xyz(),
            .microfacet_alpha_x = block_1[2u].w,
            .microfacet_alpha_y = block_1[3u].w}};
}

SurfaceClosurePhysicalHairRecord
unpack_surface_closure_physical_hair(
    const SurfaceClosurePhysicalCommonRecord &common,
    Expr<luisa::float4x4> block_1_expression) noexcept {
    const auto block_1 =
        luisa::compute::Float4x4{block_1_expression};
    return {
        .common = common,
        .payload = {
            .tangent = block_1[3u].xyz(),
            .roughness_u = block_1[2u].w,
            .roughness_v = block_1[3u].w,
            .offset = block_1[2u].x}};
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
            .ior = block_1[0u].w,
            .thin_film_thickness = block_1[1u].w,
            .thin_film_ior = block_1[2u].w,
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

}// namespace psycles::luisa_backend
