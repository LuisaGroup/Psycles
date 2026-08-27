#include "principled_metallic_component.h"

#include "principled_specular_state.h"
#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

PrincipledMetallicSetupParameters populate_principled_metallic(
    const PrincipledMetallicSetupInput &input) noexcept {
    const auto specular = populate_principled_specular_state(
        {.normal = input.normal,
         .incoming = input.incoming,
         .surface_shading_normal = input.surface_shading_normal,
         .surface_geometric_normal = input.surface_geometric_normal,
         .roughness = input.roughness,
         .specular_tint = input.specular_tint,
         .use_bump_map_correction = input.use_bump_map_correction});
    return {
        .lower_weight = input.lower_weight,
        .glossy_normal = specular.glossy_normal,
        .base_color = min(
            max(input.color, make_float3(0.0f)),
            make_float3(1.0f)),
        .specular_tint = specular.specular_tint,
        .incoming_cosine = specular.incoming_cosine,
        .roughness = specular.roughness,
        .metallic = clamp(input.metallic, 0.0f, 1.0f),
        .thin_film_thickness = input.thin_film_enabled
                                   ? input.thin_film_thickness
                                   : Float{0.0f},
        .thin_film_ior = input.thin_film_enabled
                             ? input.thin_film_ior
                             : Float{0.0f},
        .thin_film_enabled = input.thin_film_enabled,
        .preserve_ggx_energy = input.preserve_ggx_energy};
}

PrincipledMetallicSetupResult setup_principled_metallic(
    const ShaderServices &services,
    const PrincipledMetallicSetupParameters &parameters,
    Bool reflective_caustics) noexcept {
    const auto requested =
        parameters.metallic > cycles_closure::closure_weight_cutoff;
    const auto pre_weight = max(
        parameters.lower_weight * parameters.metallic,
        make_float3(0.0f));
    const auto allocation_weight = sample_weight(pre_weight);
    const auto allocated =
        requested & reflective_caustics &
        (allocation_weight >= cycles_closure::closure_weight_cutoff);
    const auto f0 = parameters.base_color;
    const auto fresnel_b = fresnel_f82_b(
        f0,
        min(parameters.specular_tint, make_float3(1.0f)));
    const auto fss =
        lerp(f0, make_float3(1.0f), 1.0f / 21.0f) -
        fresnel_b * (1.0f / 126.0f);
    const auto energy = ggx_energy(
        services,
        parameters.roughness,
        parameters.preserve_ggx_energy,
        parameters.incoming_cosine,
        fss);
    const auto interpolation = cycles_table_3d(
        services,
        parameters.roughness,
        parameters.incoming_cosine,
        0.5f,
        UInt{cycles45_tables::ggx_gen_schlick_s_offset},
        16u,
        16u,
        16u);
    auto albedo_estimate = lerp(
        f0, make_float3(1.0f), interpolation);
    if (parameters.thin_film_enabled) {
        const auto film = thin_film_f82_fresnel(
            services,
            parameters.thin_film_thickness,
            parameters.thin_film_ior,
            f0,
            fresnel_b,
            parameters.incoming_cosine);
        albedo_estimate = select(
            albedo_estimate,
            film,
            parameters.thin_film_thickness >
                thin_film_thickness_cutoff);
    }
    const auto weight = select(
        make_float3(0.0f),
        pre_weight * energy.darkening,
        allocated);
    const auto albedo = weight * albedo_estimate;
    return {
        .weight = weight,
        .allocation_weight = select(
            0.0f, allocation_weight, allocated),
        .sample_weight = select(
            0.0f,
            allocation_weight * sample_weight(albedo_estimate) *
                sample_weight(energy.darkening),
            allocated),
        .albedo = albedo,
        .normal = parameters.glossy_normal,
        .color = f0,
        .specular_tint = fresnel_b,
        .evaluation_scale = energy.energy_scale,
        .lower_weight = select(
            parameters.lower_weight,
            parameters.lower_weight * (1.0f - parameters.metallic),
            requested)};
}

PrincipledMetallicComponent::PrincipledMetallicComponent(
    const ShaderServices &services) noexcept
    : _services{services} {}

PrincipledMetallicSetupResult
PrincipledMetallicComponent::setup(
    const PrincipledMetallicSetupInput &input,
    const PrincipledMetallicSetupParameters &direct_parameters,
    Bool reflective_caustics) const noexcept {
    const auto *provider =
        _services.surface_closure_setup_provider();
    return provider != nullptr && !input.thin_film_enabled
               ? provider->principled_metallic(
                     input, reflective_caustics)
               : setup_principled_metallic(
                     _services,
                     direct_parameters,
                     reflective_caustics);
}

}// namespace psycles::luisa_backend::detail
