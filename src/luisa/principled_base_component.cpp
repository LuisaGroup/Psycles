#include "principled_base_component.h"
#include "principled_metallic_component.h"
#include "principled_specular_state.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 attenuate_lower_layer(Float3 lower_weight,
                                           Float3 layer_albedo,
                                           Bool layer_allocated) noexcept {
    const auto relative_albedo =
        safe_divide_components(layer_albedo, lower_weight);
    const auto remaining =
        clamp(1.0f - max_component(relative_albedo), 0.0f, 1.0f);
    return select(lower_weight, lower_weight * remaining, layer_allocated);
}

}// namespace

PrincipledBaseComponent::PrincipledBaseComponent(
    const ShaderServices &services, const SurfacePoint &point) noexcept
    : _services{services}, _point{point}, _glass{services, _point},
      _thin_glass{services, _point} {}

PrincipledDielectricSetupParameters populate_principled_dielectric(
    const PrincipledDielectricSetupInput &input) noexcept {
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
        .incoming_cosine = specular.incoming_cosine,
        .roughness = specular.roughness,
        .ior = input.ior,
        .specular_ior_level = input.specular_ior_level,
        .specular_tint = specular.specular_tint,
        .preserve_ggx_energy = input.preserve_ggx_energy};
}

PrincipledDielectricSetupResult setup_principled_dielectric(
    const ShaderServices &services,
    const PrincipledDielectricSetupParameters &parameters,
    Bool reflective_caustics) noexcept {
    const auto adjusted = adjusted_ior(
        parameters.ior, parameters.specular_ior_level);
    const auto dielectric_f0 = clamp(
        make_float3(adjusted.f0) * parameters.specular_tint,
        make_float3(0.0f),
        make_float3(1.0f));
    const auto real_f0 = f0_from_ior(adjusted.eta);
    const auto real_fss = fresnel_dielectric_fss(adjusted.eta);
    const auto fss_interpolation = clamp(
        (real_fss - real_f0) / (1.0f - real_f0),
        0.0f,
        1.0f);
    const auto dielectric_fss = lerp(
        dielectric_f0,
        make_float3(1.0f),
        fss_interpolation);
    const auto dielectric_energy = ggx_energy(
        services,
        parameters.roughness,
        parameters.preserve_ggx_energy,
        parameters.incoming_cosine,
        dielectric_fss);
    const auto dielectric_z = sqrt(abs(
        (adjusted.eta - 1.0f) /
        (adjusted.eta + 1.0f)));
    const auto dielectric_interpolation = cycles_table_3d(
        services,
        parameters.roughness,
        parameters.incoming_cosine,
        dielectric_z,
        UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset},
        16u,
        16u,
        16u);
    const auto dielectric_albedo_estimate = lerp(
        dielectric_f0,
        make_float3(1.0f),
        dielectric_interpolation);
    const auto dielectric_requested = adjusted.eta != 1.0f;
    const auto dielectric_pre_weight = select(
        make_float3(0.0f),
        parameters.lower_weight,
        dielectric_requested);
    const auto dielectric_allocated_weight = max(
        dielectric_pre_weight,
        make_float3(0.0f));
    const auto dielectric_allocation_weight =
        sample_weight(dielectric_allocated_weight);
    const auto dielectric_allocated =
        dielectric_requested & reflective_caustics &
        (dielectric_allocation_weight >=
         cycles_closure::closure_weight_cutoff);
    const auto weight = select(
        make_float3(0.0f),
        dielectric_allocated_weight * dielectric_energy.darkening,
        dielectric_allocated);
    const auto allocation_weight = select(
        0.0f,
        dielectric_allocation_weight,
        dielectric_allocated);
    const auto sample_weight_value = select(
        0.0f,
        dielectric_allocation_weight *
            sample_weight(dielectric_albedo_estimate) *
            sample_weight(dielectric_energy.darkening),
        dielectric_allocated);
    const auto albedo = weight * dielectric_albedo_estimate;
    return {
        .weight = weight,
        .allocation_weight = allocation_weight,
        .sample_weight = sample_weight_value,
        .albedo = albedo,
        .normal = parameters.glossy_normal,
        .color = dielectric_f0,
        .ior = adjusted.eta,
        .evaluation_scale = dielectric_energy.energy_scale,
        .lower_weight = attenuate_lower_layer(
            parameters.lower_weight,
            albedo,
            dielectric_allocated)};
}

PrincipledBaseResult PrincipledBaseComponent::evaluate(
    const TracedClosure &closure,
    compiler::PrincipledClosureFeatureMask features, Bool reflective_caustics,
    Bool refractive_caustics) const noexcept {
    const auto enabled =
        [features](compiler::PrincipledClosureFeature feature) noexcept {
            return (features & compiler::principled_closure_feature_bit(feature)) !=
                   0u;
        };
    const auto incoming = safe_normalize(_point.incoming, _point.shading_normal);
    const auto glossy_normal =
        maybe_ensure_valid_specular_reflection(_point, incoming, closure.normal);
    const auto base_color = max(closure.color, make_float3(0.0f));
    const auto clamped_base_color = min(base_color, make_float3(1.0f));
    const auto specular_tint = max(closure.specular_tint, make_float3(0.0f));
    const auto incoming_cosine = clamp(dot(glossy_normal, incoming), 0.0f, 1.0f);
    const auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    const PrincipledMetallicComponent metallic_component{_services};
    auto lower_weight = closure.weight;
    std::optional<TracedClosure> metallic;
    std::optional<TracedClosure> transmission;
    std::optional<TracedClosure> thin_glass_reflection;
    std::optional<TracedClosure> thin_glass_transmission;
    std::optional<TracedClosure> thin_glass_transparency;
    std::optional<TracedClosure> dielectric;

    // Metallic is the first base layer. Cycles attenuates every lower layer
    // whenever the authored socket crosses the closure cutoff, independently
    // of whether reflective caustics permit allocating the physical closure.
    if (enabled(compiler::PrincipledClosureFeature::metallic)) {
        const auto metallic_amount = clamp(
            closure.metallic, 0.0f, 1.0f);
        const auto setup = metallic_component.setup(
            {.lower_weight = lower_weight,
             .color = closure.color,
             .normal = closure.normal,
             .incoming = _point.incoming,
             .surface_shading_normal = _point.shading_normal,
             .surface_geometric_normal = _point.geometric_normal,
             .specular_tint = closure.specular_tint,
             .roughness = closure.roughness,
             .metallic = closure.metallic,
             .use_bump_map_correction =
                 _point.use_bump_map_correction,
             .preserve_ggx_energy = closure.preserve_ggx_energy},
            {.lower_weight = lower_weight,
             .glossy_normal = glossy_normal,
             .base_color = clamped_base_color,
             .specular_tint = specular_tint,
             .incoming_cosine = incoming_cosine,
             .roughness = roughness,
             .metallic = metallic_amount,
             .preserve_ggx_energy = closure.preserve_ggx_energy},
            reflective_caustics);

        auto physical = closure;
        physical.principled_lobe = PrincipledLobe::metallic;
        physical.weight = setup.weight;
        physical.allocation_weight = setup.allocation_weight;
        physical.sample_weight = setup.sample_weight;
        physical.setup_valid = true;
        physical.albedo = setup.albedo;
        physical.reflection_albedo = physical.albedo;
        physical.transmission_albedo = make_float3(0.0f);
        physical.color = setup.color;
        physical.normal = setup.normal;
        physical.ior = 1.0f;
        physical.specular_tint = setup.specular_tint;
        physical.evaluation_scale = setup.evaluation_scale;
        metallic.emplace(std::move(physical));
        lower_weight = setup.lower_weight;
    }

    // Thick Principled transmission is one generalized-Schlick glass
    // closure. Thin Wall instead expands to two constant-Fresnel GGX lobes;
    // both paths remain in the recorded AST because Thin Wall may be linked.
    const auto thick_transmission =
        enabled(compiler::PrincipledClosureFeature::thick_transmission);
    const auto thin_transmission =
        enabled(compiler::PrincipledClosureFeature::thin_transmission);
    if (thick_transmission || thin_transmission) {
        const auto transmission_amount =
            clamp(closure.transmission_weight, 0.0f, 1.0f);
        const auto transmission_requested =
            transmission_amount > cycles_closure::closure_weight_cutoff;
        const auto original_ior = max(closure.ior, 1.0e-5f);
        if (thick_transmission) {
            transmission.emplace(_glass.setup(
                {.prototype = closure,
                 .weight = lower_weight * transmission_amount,
                 .normal = glossy_normal,
                 .roughness = roughness,
                 .ior = original_ior,
                 .fresnel_f0 = make_float3(f0_from_ior(original_ior)) * specular_tint,
                 .fresnel_f90 = make_float3(1.0f),
                 .reflection_tint = select(make_float3(0.0f), make_float3(1.0f),
                                           reflective_caustics),
                 .transmission_tint =
                     select(make_float3(0.0f), sqrt(clamped_base_color),
                            refractive_caustics),
                 .enabled = transmission_requested & !closure.thin_wall &
                            (reflective_caustics | refractive_caustics),
                 .principled_lobe = PrincipledLobe::transmission,
                 .preserve_energy = closure.preserve_ggx_energy,
                 .beckmann = false}));
        }
        if (thin_transmission) {
            auto thin_glass = _thin_glass.setup(
                closure, lower_weight * transmission_amount, glossy_normal, roughness,
                original_ior, specular_tint, clamped_base_color,
                transmission_requested & closure.thin_wall &
                    (reflective_caustics | refractive_caustics),
                reflective_caustics, refractive_caustics);
            thin_glass_reflection.emplace(std::move(thin_glass.reflection));
            thin_glass_transmission.emplace(std::move(thin_glass.transmission));
            thin_glass_transparency.emplace(std::move(thin_glass.transparency));
        }
        lower_weight =
            select(lower_weight, lower_weight * (1.0f - transmission_amount),
                   transmission_requested);
    }

    // The remaining dielectric reflection uses the Specular IOR Level
    // adjusted eta. Its albedo attenuates diffuse only when the closure was
    // actually allocated; a disabled caustic branch cannot consume energy.
    if (enabled(compiler::PrincipledClosureFeature::dielectric)) {
        const auto input = PrincipledDielectricSetupInput{
            .lower_weight = lower_weight,
            .normal = closure.normal,
            .incoming = _point.incoming,
            .surface_shading_normal = _point.shading_normal,
            .surface_geometric_normal = _point.geometric_normal,
            .roughness = closure.roughness,
            .ior = closure.ior,
            .specular_ior_level = closure.specular_ior_level,
            .specular_tint = closure.specular_tint,
            .use_bump_map_correction =
                _point.use_bump_map_correction,
            .preserve_ggx_energy = closure.preserve_ggx_energy};
        const auto *provider =
            _services.surface_closure_setup_provider();
        const auto setup = provider != nullptr
                               ? provider->principled_dielectric(
                                     input,
                                     reflective_caustics)
                               : setup_principled_dielectric(
                                     _services,
                                     {.lower_weight = lower_weight,
                                      .glossy_normal = glossy_normal,
                                      .incoming_cosine = incoming_cosine,
                                      .roughness = roughness,
                                      .ior = closure.ior,
                                      .specular_ior_level =
                                          closure.specular_ior_level,
                                      .specular_tint = specular_tint,
                                      .preserve_ggx_energy =
                                          closure.preserve_ggx_energy},
                                     reflective_caustics);

        auto physical = closure;
        physical.principled_lobe = PrincipledLobe::dielectric;
        physical.weight = setup.weight;
        physical.allocation_weight = setup.allocation_weight;
        physical.sample_weight = setup.sample_weight;
        physical.setup_valid = true;
        physical.albedo = setup.albedo;
        physical.reflection_albedo = physical.albedo;
        physical.transmission_albedo = make_float3(0.0f);
        physical.color = setup.color;
        physical.normal = setup.normal;
        physical.ior = setup.ior;
        physical.evaluation_scale = setup.evaluation_scale;
        lower_weight = setup.lower_weight;
        dielectric.emplace(std::move(physical));
    }

    return {.metallic = std::move(metallic),
            .transmission = std::move(transmission),
            .thin_glass_reflection = std::move(thin_glass_reflection),
            .thin_glass_transmission = std::move(thin_glass_transmission),
            .thin_glass_transparency = std::move(thin_glass_transparency),
            .dielectric = std::move(dielectric),
            .base_weight = lower_weight};
}

}// namespace psycles::luisa_backend::detail
