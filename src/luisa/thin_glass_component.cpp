#include "thin_glass_component.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {
namespace {

struct ThinGlassFresnel {
    Float3 reflection;
    Float3 transmission;
};

[[nodiscard]] Float3 color_power(
    Float3 value, Float exponent) noexcept {
    return make_float3(
        pow(value.x, exponent),
        pow(value.y, exponent),
        pow(value.z, exponent));
}

[[nodiscard]] Float3 mirror(
    Float3 incident, Float3 normal) noexcept {
    return incident - 2.0f * normal * dot(incident, normal);
}

[[nodiscard]] ThinGlassFresnel thin_glass_fresnel(
    Float cosine_i,
    Float ior,
    Float3 reflection_tint,
    Float3 transmission_tint,
    Bool reflective_caustics,
    Bool refractive_caustics) noexcept {
    const auto f0 = make_float3(f0_from_ior(ior)) * reflection_tint;
    const auto front_fresnel = generalized_dielectric_fresnel(
        cosine_i, ior, f0);
    const auto front_reflection = select(
        make_float3(0.0f), front_fresnel, reflective_caustics);
    const auto front_transmission = select(
        make_float3(0.0f),
        make_float3(1.0f) - front_fresnel,
        refractive_caustics);

    // Cycles reports the transmitted cosine relative to the incident-side
    // normal, hence the negative sign. It makes the Beer-Lambert exponent
    // -1/cos(theta_t) positive without changing the authored tint.
    const auto eta_cosine_t_squared =
        ior * ior - (1.0f - cosine_i * cosine_i);
    const auto cosine_t =
        -sqrt(max(eta_cosine_t_squared, 0.0f)) / ior;
    const auto finite_cosine_t = cosine_t != 0.0f;
    const auto safe_cosine_t = select(-1.0f, cosine_t, finite_cosine_t);
    const auto absorption = select(
        make_float3(0.0f),
        color_power(transmission_tint, -1.0f / safe_cosine_t),
        finite_cosine_t);

    // With no thin-film layer the back-face coefficients equal the front
    // coefficients. Sum the infinite internal-reflection series per channel.
    const auto round_trip = front_reflection * absorption;
    const auto transmission = safe_divide_components(
        absorption * front_transmission * front_transmission,
        make_float3(1.0f) - round_trip * round_trip);
    return {
        .reflection = front_reflection +
                      transmission * front_reflection * absorption,
        .transmission = transmission};
}

[[nodiscard]] TracedClosure setup_microfacet_reflection(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    const TracedClosure &prototype,
    Float3 weight,
    Float3 normal,
    Float roughness,
    Float3 energy_tint,
    Bool enabled) noexcept {
    auto closure = prototype;
    closure.operation = compiler::ClosureOperation::glossy;
    closure.physical_kind = SurfaceClosureKind::none;
    closure.principled_lobe = PrincipledLobe::transmission;
    closure.normal = normal;
    closure.roughness = clamp(roughness, 0.0f, 1.0f);
    closure.ior = 1.0f;
    closure.color = make_float3(1.0f);
    closure.preserve_ggx_energy = true;
    closure.beckmann = false;
    closure.setup_valid = true;

    const auto allocated_weight = max(weight, make_float3(0.0f));
    const auto allocation_weight = sample_weight(allocated_weight);
    const auto allocated =
        enabled &
        (allocation_weight >= cycles_closure::closure_weight_cutoff);
    const auto incoming = safe_normalize(
        point.incoming, point.shading_normal);
    const auto energy = ggx_energy(
        services, closure, dot(normal, incoming), energy_tint);
    closure.weight = select(
        make_float3(0.0f),
        allocated_weight * energy.darkening,
        allocated);
    closure.allocation_weight = select(
        0.0f, allocation_weight, allocated);
    closure.sample_weight = select(
        0.0f,
        allocation_weight * sample_weight(energy.darkening),
        allocated);
    closure.albedo = closure.weight;
    closure.reflection_albedo = closure.weight;
    closure.transmission_albedo = make_float3(0.0f);
    closure.evaluation_scale = energy.energy_scale;
    return closure;
}

}// namespace

ThinGlassComponent::ThinGlassComponent(
    const ShaderServices &services,
    const SurfaceClosurePoint &point) noexcept
    : _services{services}, _point{point} {}

ThinGlassSetupResult ThinGlassComponent::setup(
    const TracedClosure &prototype,
    Float3 weight,
    Float3 normal,
    Float roughness,
    Float ior,
    Float3 reflection_tint,
    Float3 transmission_tint,
    Bool enabled,
    Bool reflective_caustics,
    Bool refractive_caustics) const noexcept {
    const auto incoming = safe_normalize(
        _point.incoming, _point.shading_normal);
    const auto surface_roughness = clamp(roughness, 0.0f, 1.0f);
    const auto microfacet_alpha = surface_roughness * surface_roughness;
    const auto original_ior = max(ior, 1.0e-5f);
    const auto fresnel = thin_glass_fresnel(
        dot(normal, incoming),
        original_ior,
        max(reflection_tint, make_float3(0.0f)),
        clamp(transmission_tint,
              make_float3(0.0f),
              make_float3(1.0f)),
        reflective_caustics,
        refractive_caustics);

    auto reflection = setup_microfacet_reflection(
        _services,
        _point,
        prototype,
        weight * fresnel.reflection,
        normal,
        surface_roughness,
        max(reflection_tint, make_float3(0.0f)),
        enabled);

    const auto transmission_alpha = clamp(
        microfacet_alpha *
            sqrt(max(
                3.4f * (original_ior - 1.0f) *
                    (original_ior - 0.5f) *
                    (original_ior - 0.5f) /
                    (original_ior * original_ior * original_ior),
                0.0f)),
        0.0f,
        1.0f);
    const auto singular =
        transmission_alpha * transmission_alpha <=
        cycles_closure::microfacet_singular_alpha_product;
    const auto non_camera =
        (_point.ray_visibility & camera_ray_visibility) == 0u;
    const auto becomes_transparent = enabled & non_camera & singular;

    auto transmission = prototype;
    transmission.operation = compiler::ClosureOperation::refraction;
    transmission.physical_kind =
        SurfaceClosureKind::thin_glass_transmission;
    transmission.principled_lobe = PrincipledLobe::transmission;
    transmission.normal = -normal;
    // SurfaceClosureRecord stores perceptual roughness. Squaring this value
    // in microfacet_alpha() recovers Cycles' already-derived transmission
    // alpha exactly.
    transmission.roughness = sqrt(transmission_alpha);
    transmission.ior = 1.0f;
    transmission.color = make_float3(1.0f);
    transmission.preserve_ggx_energy = true;
    transmission.beckmann = false;
    transmission.setup_valid = true;
    const auto transmission_weight =
        max(weight * fresnel.transmission, make_float3(0.0f));
    const auto transmission_allocation_weight =
        sample_weight(transmission_weight);
    const auto transmission_allocated =
        enabled & !becomes_transparent &
        (transmission_allocation_weight >=
         cycles_closure::closure_weight_cutoff);
    const auto transformed_incoming = mirror(incoming, transmission.normal);
    const auto transmission_energy = ggx_energy(
        _services,
        transmission,
        dot(transmission.normal, transformed_incoming),
        make_float3(1.0f));
    transmission.weight = select(
        make_float3(0.0f),
        transmission_weight * transmission_energy.darkening,
        transmission_allocated);
    transmission.allocation_weight = select(
        0.0f,
        transmission_allocation_weight,
        transmission_allocated);
    transmission.sample_weight = select(
        0.0f,
        transmission_allocation_weight *
            sample_weight(transmission_energy.darkening),
        transmission_allocated);
    transmission.albedo = transmission.weight;
    transmission.reflection_albedo = make_float3(0.0f);
    transmission.transmission_albedo = transmission.weight;
    transmission.evaluation_scale = transmission_energy.energy_scale;

    auto transparency = prototype;
    transparency.operation = compiler::ClosureOperation::transparent;
    transparency.physical_kind = SurfaceClosureKind::none;
    transparency.principled_lobe = PrincipledLobe::none;
    const auto transparent_state = transparent_closure_state(select(
        make_float3(0.0f), transmission_weight, becomes_transparent));
    transparency.weight = transparent_state.weight;
    transparency.allocation_weight = transparent_state.sample_weight;
    transparency.sample_weight = transparent_state.sample_weight;
    transparency.setup_valid = true;
    transparency.albedo = transparency.weight;
    transparency.reflection_albedo = make_float3(0.0f);
    transparency.transmission_albedo = make_float3(0.0f);
    transparency.color = make_float3(1.0f);
    transparency.normal = _point.shading_normal;
    transparency.roughness = 0.0f;
    transparency.ior = 1.0f;
    transparency.evaluation_scale = make_float3(1.0f);
    transparency.preserve_ggx_energy = false;
    transparency.beckmann = false;
    return {
        .reflection = reflection,
        .transmission = transmission,
        .transparency = transparency};
}

MicrofacetEvaluation ThinGlassComponent::evaluate(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float glossy_filter_roughness) const noexcept {
    const auto transformed_incoming =
        mirror(incoming, closure.common.normal);
    return microfacet_evaluate(
        _services,
        closure,
        transformed_incoming,
        outgoing,
        closure.common.normal,
        glossy_filter_roughness);
}

MicrofacetReflectionSample ThinGlassComponent::sample(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float2 random_direction,
    Float glossy_filter_roughness) const noexcept {
    const auto transformed_incoming =
        mirror(incoming, closure.common.normal);
    const auto alpha = microfacet_alpha(
        closure.common, glossy_filter_roughness);
    const auto singular =
        alpha * alpha <=
        cycles_closure::microfacet_singular_alpha_product;
    const auto sampling_alpha = max(alpha, 1.0e-10f);
    const auto half_vector =
        cycles_sample_mapping::sample_ggx_visible_normal(
            closure.common.normal,
            transformed_incoming,
            sampling_alpha,
            random_direction);
    const auto regular_direction =
        2.0f * dot(transformed_incoming, half_vector) * half_vector -
        transformed_incoming;
    const auto direction = select(
        regular_direction, -incoming, singular);
    const auto valid =
        (dot(closure.common.normal, transformed_incoming) > 0.0f) &
        (dot(closure.common.normal, direction) > 0.0f) &
        (dot(_point.geometric_normal, direction) < 0.0f);
    return {
        .direction = direction,
        .singular_evaluation = closure.common.weight * 1.0e6f,
        .singular_pdf = 1.0e6f,
        .alpha = select(alpha, 0.0f, singular),
        .singular = singular,
        .valid = valid};
}

}// namespace psycles::luisa_backend::detail
