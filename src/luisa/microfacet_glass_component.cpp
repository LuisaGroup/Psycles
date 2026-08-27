#include "microfacet_glass_component.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {
namespace {

struct GlassFresnel {
    Float3 reflection;
    Float3 transmission;
};

struct GlassGeometry {
    Float3 half_vector;
    Float inverse_half_length;
    Float cosine_incoming;
    Float cosine_outgoing;
    Float cosine_half_incoming;
    Float cosine_normal_half;
    Float cosine_half_outgoing;
    Bool transmission;
};

[[nodiscard]] Bool is_refraction(
    const SurfaceClosurePhysicalDielectricRecord &closure) noexcept {
    return closure.common.kind == static_cast<std::uint32_t>(
                                      SurfaceClosureKind::refraction);
}

[[nodiscard]] GlassFresnel glass_fresnel(
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float cosine_half_incoming) noexcept {
    // Cycles' negative generalized-Schlick exponent is a model tag: the
    // physical dielectric curve determines an interpolation coordinate
    // between the authored spectral F0 and F90 endpoints.
    const auto real_fresnel =
        fresnel_dielectric_cos(cosine_half_incoming, closure.payload.ior);
    const auto real_f0 = f0_from_ior(closure.payload.ior);
    const auto interpolation =
        clamp((real_fresnel - real_f0) / (1.0f - real_f0), 0.0f, 1.0f);
    const auto fresnel =
        lerp(closure.payload.fresnel_f0,
             closure.payload.fresnel_f90,
             interpolation);
    const auto refraction_only = is_refraction(closure);
    const auto pure_transmission = select(
        make_float3(1.0f), make_float3(0.0f), real_fresnel == 1.0f);
    return {
        .reflection = select(
            fresnel * closure.payload.reflection_tint,
            make_float3(0.0f),
            refraction_only),
        .transmission = select(
            (make_float3(1.0f) - fresnel) *
                closure.payload.transmission_tint,
            pure_transmission,
            refraction_only)};
}

[[nodiscard]] GlassFresnel masked_fresnel(
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float cosine_half_incoming,
    Bool reflection_allowed,
    Bool transmission_allowed) noexcept {
    auto result = glass_fresnel(closure, cosine_half_incoming);
    result.reflection =
        select(make_float3(0.0f), result.reflection, reflection_allowed);
    result.transmission =
        select(make_float3(0.0f), result.transmission, transmission_allowed);
    return result;
}

[[nodiscard]] Float
reflection_probability(const GlassFresnel &fresnel) noexcept {
    const auto reflection = sample_weight(fresnel.reflection);
    const auto transmission = sample_weight(fresnel.transmission);
    return reflection / max(reflection + transmission, 1.0e-20f);
}

[[nodiscard]] GlassGeometry glass_geometry(
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 normal) noexcept {
    const auto cosine_outgoing = dot(normal, outgoing);
    const auto transmission = cosine_outgoing < 0.0f;
    const auto unnormalized_half = select(
        incoming + outgoing,
        -(closure.payload.ior * outgoing + incoming),
        transmission);
    const auto half_length = sqrt(dot(unnormalized_half, unnormalized_half));
    // At eta == 1 every transmitted microfacet maps to -incoming. Cycles
    // classifies that event as singular; its finite-direction evaluation has
    // zero measure. Make the definition explicit so backend normalization
    // precision cannot turn the zero half-vector into an unbounded Jacobian.
    const auto unit_ior_transmission =
        transmission & (closure.payload.ior == 1.0f);
    const auto has_finite_half =
        (half_length != 0.0f) & !unit_ior_transmission;
    const auto safe_half_length =
        select(1.0f, half_length, has_finite_half);
    const auto inverse_half_length =
        select(0.0f, 1.0f / safe_half_length, has_finite_half);
    const auto half_vector = unnormalized_half * inverse_half_length;
    return {.half_vector = half_vector,
            .inverse_half_length = inverse_half_length,
            .cosine_incoming = dot(normal, incoming),
            .cosine_outgoing = cosine_outgoing,
            .cosine_half_incoming = dot(half_vector, incoming),
            .cosine_normal_half = dot(normal, half_vector),
            .cosine_half_outgoing = dot(half_vector, outgoing),
            .transmission = transmission};
}

[[nodiscard]] GgxEnergy glass_energy(const ShaderServices &services,
                                     const TracedClosure &closure,
                                     Float incoming_cosine) noexcept {
    if (!closure.preserve_ggx_energy) {
        return {.darkening = make_float3(1.0f), .energy_scale = make_float3(1.0f)};
    }

    const auto inverse_configuration = closure.ior < 1.0f;
    const auto lookup_ior =
        select(closure.ior, 1.0f / closure.ior, inverse_configuration);
    const auto energy_offset = select(
        UInt{cycles45_tables::ggx_glass_e_offset},
        UInt{cycles45_tables::ggx_glass_inv_e_offset}, inverse_configuration);
    const auto average_offset = select(
        UInt{cycles45_tables::ggx_glass_eavg_offset},
        UInt{cycles45_tables::ggx_glass_inv_eavg_offset}, inverse_configuration);
    const auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    const auto z = sqrt(abs((lookup_ior - 1.0f) / (lookup_ior + 1.0f)));
    const auto energy = cycles_table_3d(services, roughness, incoming_cosine, z,
                                        energy_offset, 16u, 16u, 16u);
    const auto average_energy =
        cycles_table_2d(services, roughness, z, average_offset, 16u, 16u);
    const auto missing_factor = (1.0f - energy) / energy;
    const auto energy_scale = 1.0f / energy;

    const auto real_f0 = f0_from_ior(closure.ior);
    const auto real_fss = fresnel_dielectric_fss(closure.ior);
    const auto interpolation =
        clamp((real_fss - real_f0) / (1.0f - real_f0), 0.0f, 1.0f);
    const auto reflection_fss =
        closure.reflection_tint *
        lerp(closure.fresnel_f0, closure.fresnel_f90, interpolation);
    // Cycles models transmissive multi-bounce Fresnel by the transmission
    // tint itself; only a strictly zero tint selects the reflection-only
    // integral above.
    const auto fss = select(reflection_fss, closure.transmission_tint,
                            any(closure.transmission_tint != make_float3(0.0f)));
    const auto fms = fss * average_energy /
                     (make_float3(1.0f) - fss * (1.0f - average_energy));
    const auto darkening =
        (make_float3(1.0f) + fms * missing_factor) / energy_scale;
    return {.darkening = darkening, .energy_scale = make_float3(energy_scale)};
}

[[nodiscard]] GlassFresnel
glass_albedo_estimate(const ShaderServices &services,
                      const TracedClosure &closure,
                      Float incoming_cosine) noexcept {
    if (closure.operation == compiler::ClosureOperation::refraction) {
        const auto real_fresnel =
            fresnel_dielectric_cos(incoming_cosine, closure.ior);
        return {
            .reflection = make_float3(0.0f),
            .transmission = select(
                make_float3(1.0f),
                make_float3(0.0f),
                real_fresnel == 1.0f)};
    }
    const auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    const auto z = sqrt(abs((closure.ior - 1.0f) / (closure.ior + 1.0f)));
    const auto interpolation = cycles_table_3d(
        services, roughness, incoming_cosine, z,
        UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset}, 16u, 16u, 16u);
    const auto fresnel =
        lerp(closure.fresnel_f0, closure.fresnel_f90, interpolation);
    return {.reflection = fresnel * closure.reflection_tint,
            .transmission =
                (make_float3(1.0f) - fresnel) * closure.transmission_tint};
}

}// namespace

MicrofacetGlassComponent::MicrofacetGlassComponent(
    const ShaderServices &services,
    const SurfaceClosurePoint &point) noexcept
    : _services{services}, _point{point} {}

TracedClosure MicrofacetGlassComponent::setup(
    const MicrofacetGlassSetup &parameters) const noexcept {
    auto closure = parameters.prototype;
    closure.operation = parameters.refraction_only
                            ? compiler::ClosureOperation::refraction
                            : compiler::ClosureOperation::glass;
    closure.principled_lobe = parameters.principled_lobe;
    closure.setup_valid = true;
    closure.normal = maybe_ensure_valid_specular_reflection(
        _point, safe_normalize(_point.incoming, _point.shading_normal),
        parameters.normal);
    // Standalone Cycles Glass/Refraction squares the linked socket value
    // before saturating microfacet alpha. Retain the equivalent perceptual
    // roughness here so later alpha and Roughness-pass paths share one value.
    closure.roughness = sqrt(clamp(
        parameters.roughness * parameters.roughness, 0.0f, 1.0f));
    const auto original_ior = max(parameters.ior, 1.0e-5f);
    closure.ior = select(original_ior, 1.0f / original_ior, _point.back_facing);
    closure.fresnel_f0 =
        clamp(parameters.fresnel_f0, make_float3(0.0f), make_float3(1.0f));
    closure.fresnel_f90 = parameters.fresnel_f90;
    closure.reflection_tint = parameters.reflection_tint;
    closure.transmission_tint = parameters.transmission_tint;
    closure.preserve_ggx_energy = parameters.preserve_energy;
    closure.beckmann = parameters.beckmann;

    const auto allocated_weight = max(parameters.weight, make_float3(0.0f));
    const auto allocation_weight = sample_weight(allocated_weight);
    const auto allocated =
        parameters.enabled &
        (allocation_weight >= cycles_closure::closure_weight_cutoff);
    closure.allocation_weight = select(0.0f, allocation_weight, allocated);

    const auto incoming_cosine = dot(
        closure.normal, safe_normalize(_point.incoming, _point.shading_normal));
    const auto estimated =
        glass_albedo_estimate(_services, closure, incoming_cosine);
    const auto energy = glass_energy(_services, closure, incoming_cosine);
    closure.weight =
        select(make_float3(0.0f), allocated_weight * energy.darkening, allocated);
    closure.reflection_albedo = select(
        make_float3(0.0f), closure.weight * estimated.reflection, allocated);
    closure.transmission_albedo = select(
        make_float3(0.0f), closure.weight * estimated.transmission, allocated);
    closure.albedo = closure.reflection_albedo + closure.transmission_albedo;
    // Cycles leaves a pure Refraction closure's allocation-derived sample
    // weight untouched. TIR zeros its evaluated/transmission albedo and makes
    // the conditional sample invalid, but must not change mixture selection
    // probabilities before a microfacet normal has been sampled.
    const auto estimated_sample_weight = parameters.refraction_only
                                             ? Float{1.0f}
                                             : sample_weight(estimated.reflection +
                                                             estimated.transmission);
    closure.sample_weight =
        select(0.0f,
               allocation_weight *
                   estimated_sample_weight *
                   sample_weight(energy.darkening),
               allocated);
    closure.evaluation_scale = energy.energy_scale;
    return closure;
}

MicrofacetEvaluation MicrofacetGlassComponent::evaluate(
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float3 incoming, Float3 outgoing,
    Float3 glossy_normal, Bool reflection_allowed, Bool transmission_allowed,
    Float glossy_filter_roughness) const noexcept {
    static_cast<void>(_services);
    static_cast<void>(_point);
    const auto setup_alpha =
        microfacet_alpha(closure.common, glossy_filter_roughness);
    const auto roughness_squared = setup_alpha * setup_alpha;
    const auto singular =
        roughness_squared <=
        cycles_closure::microfacet_singular_alpha_product;
    MicrofacetEvaluation result{
        .intensity = make_float3(0.0f),
        .pdf = 0.0f,
        .roughness_squared = select(
            roughness_squared, 0.0f, singular)};
    // Delta glass has no finite-direction evaluation. Keeping the complete
    // geometry, D, Lambda, Fresnel, and Jacobian calculation in the regular
    // branch is the evaluator analogue of the sampling partition below.
    $if(!singular) {
        const auto geometry =
            glass_geometry(closure, incoming, outgoing, glossy_normal);
        const auto alpha = max(setup_alpha, 1.0e-7f);
        const auto terms = microfacet_distribution_terms(
            closure.common,
            geometry.cosine_normal_half,
            geometry.cosine_incoming,
            geometry.cosine_outgoing,
            alpha);
        const auto fresnel = glass_fresnel(
            closure, geometry.cosine_half_incoming);
        const auto intensity_lobe = select(
            fresnel.reflection,
            fresnel.transmission,
            geometry.transmission);
        const auto transmission_jacobian =
            closure.payload.ior * closure.payload.ior *
            geometry.inverse_half_length *
            geometry.inverse_half_length *
            abs(geometry.cosine_half_incoming *
                geometry.cosine_half_outgoing);
        const auto common =
            terms.distribution / geometry.cosine_incoming *
            select(0.25f,
                   transmission_jacobian,
                   geometry.transmission);
        const auto value =
            intensity_lobe * common /
            (1.0f + terms.lambda_incoming + terms.lambda_outgoing) *
            closure.common.color_or_evaluation_scale;
        // Cycles orients the refractive half-vector by eta, not by N. For an
        // inverse-eta (backface) configuration both N.H and H.I may therefore
        // be negative. D, Fresnel, and the Jacobian are defined for that
        // orientation; the incoming hemisphere is the only directional
        // validity condition.
        // Keep the explicit ordered comparison even though !singular
        // dominates this block. For NaN roughness, !(x <= c) is true while
        // (x > c) is false; retaining the latter preserves the evaluator's
        // fail-closed input contract.
        const auto intensity_valid =
            (geometry.cosine_incoming > 0.0f) &
            (roughness_squared >
             cycles_closure::microfacet_singular_alpha_product);
        auto pdf_fresnel = fresnel;
        pdf_fresnel.reflection = select(
            make_float3(0.0f),
            pdf_fresnel.reflection,
            reflection_allowed);
        pdf_fresnel.transmission = select(
            make_float3(0.0f),
            pdf_fresnel.transmission,
            transmission_allowed);
        const auto probability =
            reflection_probability(pdf_fresnel);
        const auto lobe_probability = select(
            probability,
            1.0f - probability,
            geometry.transmission);
        // Both GGX and Beckmann use visible-normal sampling in Cycles, so
        // their PDFs share D / (N.I) / (1 + Lambda(I)).
        const auto directional_pdf =
            common / (1.0f + terms.lambda_incoming);
        const auto lobe_energy = select(
            sample_weight(pdf_fresnel.reflection),
            sample_weight(pdf_fresnel.transmission),
            geometry.transmission);
        const auto pdf_valid =
            intensity_valid & (lobe_energy > 0.0f);
        result.intensity = select(
            make_float3(0.0f), value, intensity_valid);
        result.pdf = select(
            0.0f,
            directional_pdf * lobe_probability,
            pdf_valid);
    };
    return result;
}

GlassSample MicrofacetGlassComponent::sample(
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float3 incoming, Float3 glossy_normal,
    Float2 random_direction, Float random_lobe, Bool reflection_allowed,
    Bool transmission_allowed, Float glossy_filter_roughness) const noexcept {
    static_cast<void>(_services);
    auto alpha = microfacet_alpha(
        closure.common, glossy_filter_roughness);
    auto singular =
        alpha * alpha <= cycles_closure::microfacet_singular_alpha_product;
    Float3 half_vector = glossy_normal;
    // Cycles' microfacet sampler partitions the domain before VNDF sampling:
    // H=N for a delta closure, otherwise exactly one distribution is sampled.
    // Preserve that partition as control flow so inactive distributions do not
    // execute eagerly in the Luisa expression graph.
    $if(!singular) {
        const auto sampling_alpha = max(alpha, 1.0e-7f);
        $if(closure.common.beckmann) {
            half_vector =
                cycles_sample_mapping::sample_beckmann_visible_normal(
                    glossy_normal,
                    incoming,
                    sampling_alpha,
                    random_direction);
        }
        $else {
            half_vector =
                cycles_sample_mapping::sample_ggx_visible_normal(
                    glossy_normal,
                    incoming,
                    sampling_alpha,
                    random_direction);
        };
    };
    const auto cosine_half_incoming = dot(half_vector, incoming);
    const auto fresnel = masked_fresnel(closure, cosine_half_incoming,
                                        reflection_allowed, transmission_allowed);
    const auto probability = reflection_probability(fresnel);
    const auto transmission = random_lobe >= probability;
    const auto reflected = 2.0f * cosine_half_incoming * half_vector - incoming;
    const auto eta_squared_cosine_transmitted =
        closure.payload.ior * closure.payload.ior -
        (1.0f - cosine_half_incoming * cosine_half_incoming);
    const auto cosine_half_outgoing =
        -sqrt(max(eta_squared_cosine_transmitted, 0.0f)) /
        closure.payload.ior;
    const auto inverse_eta = 1.0f / closure.payload.ior;
    const auto transmitted =
        (inverse_eta * cosine_half_incoming + cosine_half_outgoing) *
            half_vector -
        inverse_eta * incoming;
    const auto direction = select(reflected, transmitted, transmission);
    const auto selected_fresnel =
        select(fresnel.reflection, fresnel.transmission, transmission);
    const auto lobe_probability =
        select(probability, 1.0f - probability, transmission);
    const auto expected_negative = transmission;
    const auto shading_negative = dot(glossy_normal, direction) < 0.0f;
    const auto geometric_negative =
        dot(_point.geometric_normal, direction) < 0.0f;
    const auto shading_hemisphere_valid =
        select(!shading_negative, shading_negative, expected_negative);
    const auto geometric_hemisphere_valid =
        select(!geometric_negative, geometric_negative, expected_negative);
    const auto has_energy =
        sample_weight(fresnel.reflection + fresnel.transmission) > 0.0f;
    const auto valid = has_energy & (cosine_half_incoming > 0.0f) &
                       shading_hemisphere_valid & geometric_hemisphere_valid;
    singular = singular |
               (transmission &
                (abs(closure.payload.ior - 1.0f) < 1.0e-4f));
    return {.direction = direction,
            .singular_evaluation =
                closure.common.weight * selected_fresnel *
                closure.common.color_or_evaluation_scale * 1.0e6f,
            .singular_pdf = lobe_probability * 1.0e6f,
            .eta = select(1.0f, closure.payload.ior, transmission),
            .alpha = alpha,
            .transmission = transmission,
            .singular = singular,
            .valid = valid};
}

}// namespace psycles::luisa_backend::detail
