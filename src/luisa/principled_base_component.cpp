#include "principled_base_component.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

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
    : _services{services}, _point{point}, _glass{services, point} {}

PrincipledBaseResult
PrincipledBaseComponent::evaluate(const TracedClosure &closure,
                                  Bool reflective_caustics,
                                  Bool refractive_caustics) const noexcept {
    const auto incoming = safe_normalize(_point.incoming, _point.shading_normal);
    const auto glossy_normal =
        maybe_ensure_valid_specular_reflection(_point, incoming, closure.normal);
    const auto base_color = max(closure.color, make_float3(0.0f));
    const auto clamped_base_color = min(base_color, make_float3(1.0f));
    const auto specular_tint = max(closure.specular_tint, make_float3(0.0f));
    const auto incoming_cosine = clamp(dot(glossy_normal, incoming), 0.0f, 1.0f);
    const auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    auto lower_weight = closure.weight;

    // Metallic is the first base layer. Cycles attenuates every lower layer
    // whenever the authored socket crosses the closure cutoff, independently
    // of whether reflective caustics permit allocating the physical closure.
    const auto metallic_amount = clamp(closure.metallic, 0.0f, 1.0f);
    const auto metallic_requested =
        metallic_amount > cycles_closure::closure_weight_cutoff;
    const auto metallic_pre_weight =
        max(lower_weight * metallic_amount, make_float3(0.0f));
    const auto metallic_allocation_weight = sample_weight(metallic_pre_weight);
    const auto metallic_allocated =
        metallic_requested & reflective_caustics &
        (metallic_allocation_weight >= cycles_closure::closure_weight_cutoff);
    const auto metallic_f0 = clamped_base_color;
    const auto metallic_b =
        fresnel_f82_b(metallic_f0, min(specular_tint, make_float3(1.0f)));
    const auto metallic_fss = lerp(metallic_f0, make_float3(1.0f), 1.0f / 21.0f) -
                              metallic_b * (1.0f / 126.0f);
    const auto metallic_energy =
        ggx_energy(_services, closure, incoming_cosine, metallic_fss);
    const auto metallic_interpolation = cycles_table_3d(
        _services, roughness, incoming_cosine, 0.5f,
        UInt{cycles45_tables::ggx_gen_schlick_s_offset}, 16u, 16u, 16u);
    const auto metallic_albedo_estimate =
        lerp(metallic_f0, make_float3(1.0f), metallic_interpolation);

    auto metallic = closure;
    metallic.principled_lobe = PrincipledLobe::metallic;
    metallic.weight =
        select(make_float3(0.0f), metallic_pre_weight * metallic_energy.darkening,
               metallic_allocated);
    metallic.allocation_weight =
        select(0.0f, metallic_allocation_weight, metallic_allocated);
    metallic.sample_weight = select(0.0f,
                                    metallic_allocation_weight *
                                        sample_weight(metallic_albedo_estimate) *
                                        sample_weight(metallic_energy.darkening),
                                    metallic_allocated);
    metallic.setup_valid = true;
    metallic.albedo = metallic.weight * metallic_albedo_estimate;
    metallic.reflection_albedo = metallic.albedo;
    metallic.transmission_albedo = make_float3(0.0f);
    metallic.color = metallic_f0;
    metallic.normal = glossy_normal;
    metallic.ior = 1.0f;
    metallic.specular_tint = metallic_b;
    metallic.evaluation_scale = metallic_energy.energy_scale;
    lower_weight = select(lower_weight, lower_weight * (1.0f - metallic_amount),
                          metallic_requested);

    // Thick Principled transmission is one generalized-Schlick glass
    // closure. Reflection and refraction caustic controls gate spectral
    // tints, not separate approximation closures, so their shared VNDF and
    // lobe-selection density remain coupled exactly as in Cycles.
    const auto transmission_amount =
        clamp(closure.transmission_weight, 0.0f, 1.0f);
    const auto transmission_requested =
        transmission_amount > cycles_closure::closure_weight_cutoff;
    const auto original_ior = max(closure.ior, 1.0e-5f);
    const auto transmission = _glass.setup(
        {.prototype = closure,
         .weight = lower_weight * transmission_amount,
         .normal = glossy_normal,
         .roughness = roughness,
         .ior = original_ior,
         .fresnel_f0 = make_float3(f0_from_ior(original_ior)) * specular_tint,
         .fresnel_f90 = make_float3(1.0f),
         .reflection_tint =
             select(make_float3(0.0f), make_float3(1.0f), reflective_caustics),
         .transmission_tint = select(make_float3(0.0f), sqrt(clamped_base_color),
                                     refractive_caustics),
         .enabled =
             transmission_requested & (reflective_caustics | refractive_caustics),
         .principled_lobe = PrincipledLobe::transmission,
         .preserve_energy = closure.preserve_ggx_energy,
         .beckmann = false});
    lower_weight =
        select(lower_weight, lower_weight * (1.0f - transmission_amount),
               transmission_requested);

    // The remaining dielectric reflection uses the Specular IOR Level
    // adjusted eta. Its albedo attenuates diffuse only when the closure was
    // actually allocated; a disabled caustic branch cannot consume energy.
    const auto adjusted = adjusted_ior(closure);
    const auto dielectric_f0 = clamp(make_float3(adjusted.f0) * specular_tint,
                                     make_float3(0.0f), make_float3(1.0f));
    const auto real_f0 = f0_from_ior(adjusted.eta);
    const auto real_fss = fresnel_dielectric_fss(adjusted.eta);
    const auto fss_interpolation =
        clamp((real_fss - real_f0) / (1.0f - real_f0), 0.0f, 1.0f);
    const auto dielectric_fss =
        lerp(dielectric_f0, make_float3(1.0f), fss_interpolation);
    const auto dielectric_energy =
        ggx_energy(_services, closure, incoming_cosine, dielectric_fss);
    const auto dielectric_z =
        sqrt(abs((adjusted.eta - 1.0f) / (adjusted.eta + 1.0f)));
    const auto dielectric_interpolation = cycles_table_3d(
        _services, roughness, incoming_cosine, dielectric_z,
        UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset}, 16u, 16u, 16u);
    const auto dielectric_albedo_estimate =
        lerp(dielectric_f0, make_float3(1.0f), dielectric_interpolation);
    const auto dielectric_requested = adjusted.eta != 1.0f;
    const auto dielectric_pre_weight =
        select(make_float3(0.0f), lower_weight, dielectric_requested);
    const auto dielectric_allocated_weight =
        max(dielectric_pre_weight, make_float3(0.0f));
    const auto dielectric_allocation_weight =
        sample_weight(dielectric_allocated_weight);
    const auto dielectric_allocated =
        dielectric_requested & reflective_caustics &
        (dielectric_allocation_weight >= cycles_closure::closure_weight_cutoff);

    auto dielectric = closure;
    dielectric.principled_lobe = PrincipledLobe::dielectric;
    dielectric.weight =
        select(make_float3(0.0f),
               dielectric_allocated_weight * dielectric_energy.darkening,
               dielectric_allocated);
    dielectric.allocation_weight =
        select(0.0f, dielectric_allocation_weight, dielectric_allocated);
    dielectric.sample_weight = select(
        0.0f,
        dielectric_allocation_weight * sample_weight(dielectric_albedo_estimate) *
            sample_weight(dielectric_energy.darkening),
        dielectric_allocated);
    dielectric.setup_valid = true;
    dielectric.albedo = dielectric.weight * dielectric_albedo_estimate;
    dielectric.reflection_albedo = dielectric.albedo;
    dielectric.transmission_albedo = make_float3(0.0f);
    dielectric.color = dielectric_f0;
    dielectric.normal = glossy_normal;
    dielectric.ior = adjusted.eta;
    dielectric.evaluation_scale = dielectric_energy.energy_scale;
    lower_weight = attenuate_lower_layer(lower_weight, dielectric.albedo,
                                         dielectric_allocated);

    return {.metallic = metallic,
            .transmission = transmission,
            .dielectric = dielectric,
            .diffuse_weight = max(lower_weight * base_color, make_float3(0.0f))};
}

}// namespace psycles::luisa_backend::detail
