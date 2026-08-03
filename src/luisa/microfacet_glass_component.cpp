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

[[nodiscard]] GlassFresnel glass_fresnel(const TracedClosure &closure,
                                         Float cosine_half_incoming) noexcept {
    // Cycles' negative generalized-Schlick exponent is a model tag: the
    // physical dielectric curve determines an interpolation coordinate
    // between the authored spectral F0 and F90 endpoints.
    const auto real_fresnel =
        fresnel_dielectric_cos(cosine_half_incoming, closure.ior);
    const auto real_f0 = f0_from_ior(closure.ior);
    const auto interpolation =
        clamp((real_fresnel - real_f0) / (1.0f - real_f0), 0.0f, 1.0f);
    const auto fresnel =
        lerp(closure.fresnel_f0, closure.fresnel_f90, interpolation);
    return {.reflection = fresnel * closure.reflection_tint,
            .transmission =
                (make_float3(1.0f) - fresnel) * closure.transmission_tint};
}

[[nodiscard]] GlassFresnel masked_fresnel(const TracedClosure &closure,
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

[[nodiscard]] Float
glass_microfacet_alpha(const TracedClosure &closure,
                       Float glossy_filter_roughness) noexcept {
    auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
    alpha *= alpha;
    return max(alpha, glossy_filter_roughness);
}

[[nodiscard]] Float glass_microfacet_distribution(const TracedClosure &closure,
                                                  Float normal_half_cosine,
                                                  Float alpha) noexcept {
    if (!closure.beckmann) {
        return ggx_distribution(normal_half_cosine, alpha);
    }
    const auto cosine_squared =
        min(normal_half_cosine * normal_half_cosine, 1.0f);
    const auto alpha_squared = alpha * alpha;
    const auto exponent =
        (1.0f - cosine_squared) / (cosine_squared * alpha_squared);
    const auto denominator =
        exp(exponent) * pi * alpha_squared * cosine_squared * cosine_squared;
    return select(0.0f, 1.0f / denominator, normal_half_cosine > 0.0f);
}

[[nodiscard]] Float glass_microfacet_lambda(const TracedClosure &closure,
                                            Float normal_direction_cosine,
                                            Float alpha) noexcept {
    const auto cosine_squared = normal_direction_cosine * normal_direction_cosine;
    const auto squared_alpha_tangent =
        alpha * alpha * max(1.0f / cosine_squared - 1.0f, 0.0f);
    if (!closure.beckmann) {
        return 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
    }
    const auto a = rsqrt(squared_alpha_tangent);
    const auto approximation =
        ((0.396f * a - 1.259f) * a + 1.0f) / ((2.181f * a + 3.535f) * a);
    return select(approximation, 0.0f, squared_alpha_tangent < 0.39f);
}

[[nodiscard]] GlassGeometry glass_geometry(const TracedClosure &closure,
                                           Float3 incoming, Float3 outgoing,
                                           Float3 normal) noexcept {
    const auto cosine_outgoing = dot(normal, outgoing);
    const auto transmission = cosine_outgoing < 0.0f;
    const auto unnormalized_half = select(
        incoming + outgoing, -(closure.ior * outgoing + incoming), transmission);
    const auto inverse_half_length =
        1.0f / sqrt(dot(unnormalized_half, unnormalized_half));
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
    const ShaderServices &services, const SurfacePoint &point) noexcept
    : _services{services}, _point{point} {}

TracedClosure MicrofacetGlassComponent::setup(
    const MicrofacetGlassSetup &parameters) const noexcept {
    auto closure = parameters.prototype;
    closure.operation = compiler::ClosureOperation::glass;
    closure.principled_lobe = parameters.principled_lobe;
    closure.setup_valid = true;
    closure.normal = maybe_ensure_valid_specular_reflection(
        _point, safe_normalize(_point.incoming, _point.shading_normal),
        parameters.normal);
    closure.roughness = clamp(parameters.roughness, 0.0f, 1.0f);
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
    closure.sample_weight =
        select(0.0f,
               allocation_weight *
                   sample_weight(estimated.reflection + estimated.transmission) *
                   sample_weight(energy.darkening),
               allocated);
    closure.evaluation_scale = energy.energy_scale;
    return closure;
}

Float3 MicrofacetGlassComponent::intensity(
    const TracedClosure &closure, Float3 incoming, Float3 outgoing,
    Float3 glossy_normal, Float glossy_filter_roughness) const noexcept {
    static_cast<void>(_services);
    static_cast<void>(_point);
    const auto geometry =
        glass_geometry(closure, incoming, outgoing, glossy_normal);
    const auto setup_alpha =
        glass_microfacet_alpha(closure, glossy_filter_roughness);
    const auto alpha = max(setup_alpha, 1.0e-7f);
    const auto distribution = glass_microfacet_distribution(
        closure, geometry.cosine_normal_half, alpha);
    const auto lambda_incoming =
        glass_microfacet_lambda(closure, geometry.cosine_incoming, alpha);
    const auto lambda_outgoing =
        glass_microfacet_lambda(closure, geometry.cosine_outgoing, alpha);
    const auto fresnel = glass_fresnel(closure, geometry.cosine_half_incoming);
    const auto lobe =
        select(fresnel.reflection, fresnel.transmission, geometry.transmission);
    const auto transmission_jacobian =
        closure.ior * closure.ior * geometry.inverse_half_length *
        geometry.inverse_half_length *
        abs(geometry.cosine_half_incoming * geometry.cosine_half_outgoing);
    const auto common =
        distribution / geometry.cosine_incoming *
        select(0.25f, transmission_jacobian, geometry.transmission);
    const auto value = lobe * common /
                       (1.0f + lambda_incoming + lambda_outgoing) *
                       closure.evaluation_scale;
    const auto valid = (geometry.cosine_incoming > 0.0f) &
                       (geometry.cosine_normal_half > 0.0f) &
                       (geometry.cosine_half_incoming > 0.0f) &
                       (setup_alpha * setup_alpha >
                        cycles_closure::microfacet_singular_alpha_product);
    return select(make_float3(0.0f), value, valid);
}

Float MicrofacetGlassComponent::pdf(
    const TracedClosure &closure, Float3 incoming, Float3 outgoing,
    Float3 glossy_normal, Bool reflection_allowed, Bool transmission_allowed,
    Float glossy_filter_roughness) const noexcept {
    static_cast<void>(_services);
    static_cast<void>(_point);
    const auto geometry =
        glass_geometry(closure, incoming, outgoing, glossy_normal);
    const auto setup_alpha =
        glass_microfacet_alpha(closure, glossy_filter_roughness);
    const auto alpha = max(setup_alpha, 1.0e-7f);
    const auto distribution = glass_microfacet_distribution(
        closure, geometry.cosine_normal_half, alpha);
    const auto fresnel = masked_fresnel(closure, geometry.cosine_half_incoming,
                                        reflection_allowed, transmission_allowed);
    const auto probability = reflection_probability(fresnel);
    const auto lobe_probability =
        select(probability, 1.0f - probability, geometry.transmission);
    Float directional_pdf;
    if (closure.beckmann) {
        const auto reflection_jacobian =
            1.0f / (4.0f * geometry.cosine_half_incoming);
        const auto transmission_jacobian =
            abs(geometry.cosine_half_outgoing) * closure.ior * closure.ior *
            geometry.inverse_half_length * geometry.inverse_half_length;
        directional_pdf = distribution * max(geometry.cosine_normal_half, 0.0f) *
                          select(reflection_jacobian, transmission_jacobian,
                                 geometry.transmission);
    } else {
        const auto lambda_incoming =
            glass_microfacet_lambda(closure, geometry.cosine_incoming, alpha);
        const auto transmission_jacobian =
            closure.ior * closure.ior * geometry.inverse_half_length *
            geometry.inverse_half_length *
            abs(geometry.cosine_half_incoming * geometry.cosine_half_outgoing);
        const auto common =
            distribution / geometry.cosine_incoming *
            select(0.25f, transmission_jacobian, geometry.transmission);
        directional_pdf = common / (1.0f + lambda_incoming);
    }
    const auto lobe_energy =
        select(sample_weight(fresnel.reflection),
               sample_weight(fresnel.transmission), geometry.transmission);
    const auto valid =
        (geometry.cosine_incoming > 0.0f) & (geometry.cosine_normal_half > 0.0f) &
        (geometry.cosine_half_incoming > 0.0f) & (lobe_energy > 0.0f) &
        (setup_alpha * setup_alpha >
         cycles_closure::microfacet_singular_alpha_product);
    return select(0.0f, directional_pdf * lobe_probability, valid);
}

GlassSample MicrofacetGlassComponent::sample(
    const TracedClosure &closure, Float3 incoming, Float3 glossy_normal,
    Float2 random_direction, Float random_lobe, Bool reflection_allowed,
    Bool transmission_allowed, Float glossy_filter_roughness) const noexcept {
    static_cast<void>(_services);
    auto alpha = glass_microfacet_alpha(closure, glossy_filter_roughness);
    auto singular =
        alpha * alpha <= cycles_closure::microfacet_singular_alpha_product;
    const auto sampling_alpha = max(alpha, 1.0e-7f);
    Float3 sampled_half;
    if (closure.beckmann) {
        const auto tangent_squared = -sampling_alpha * sampling_alpha *
                                     log(max(1.0f - random_direction.x, 1.0e-7f));
        const auto cosine = rsqrt(1.0f + tangent_squared);
        const auto sine = sqrt(max(1.0f - cosine * cosine, 0.0f));
        const auto phi = two_pi * random_direction.y;
        const auto basis = cycles_sample_mapping::make_orthonormals(glossy_normal);
        sampled_half = basis.tangent * (sine * cos(phi)) +
                       basis.bitangent * (sine * sin(phi)) + glossy_normal * cosine;
    } else {
        sampled_half = cycles_sample_mapping::sample_ggx_visible_normal(
            glossy_normal, incoming, sampling_alpha, random_direction);
    }
    const auto half_vector = select(sampled_half, glossy_normal, singular);
    const auto cosine_half_incoming = dot(half_vector, incoming);
    const auto fresnel = masked_fresnel(closure, cosine_half_incoming,
                                        reflection_allowed, transmission_allowed);
    const auto probability = reflection_probability(fresnel);
    const auto transmission = random_lobe >= probability;
    const auto reflected = 2.0f * cosine_half_incoming * half_vector - incoming;
    const auto eta_squared_cosine_transmitted =
        closure.ior * closure.ior -
        (1.0f - cosine_half_incoming * cosine_half_incoming);
    const auto cosine_half_outgoing =
        -sqrt(max(eta_squared_cosine_transmitted, 0.0f)) / closure.ior;
    const auto inverse_eta = 1.0f / closure.ior;
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
    singular = singular | (transmission & (abs(closure.ior - 1.0f) < 1.0e-4f));
    return {.direction = direction,
            .singular_evaluation = closure.weight * selected_fresnel *
                                   closure.evaluation_scale * 1.0e6f,
            .singular_pdf = lobe_probability * 1.0e6f,
            .eta = select(1.0f, closure.ior, transmission),
            .alpha = alpha,
            .transmission = transmission,
            .singular = singular,
            .valid = valid};
}

}// namespace psycles::luisa_backend::detail
