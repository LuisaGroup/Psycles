#include "bssrdf_closure_component.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr float minimum_radius = 1.0e-8f;

[[nodiscard]] constexpr std::uint32_t bssrdf_closure_type(
    compiler::BssrdfMethod method) noexcept {
    switch (method) {
        case compiler::BssrdfMethod::burley:
            return cycles_closure::type_bssrdf_burley;
        case compiler::BssrdfMethod::random_walk:
            return cycles_closure::type_bssrdf_random_walk;
        case compiler::BssrdfMethod::random_walk_legacy:
            return cycles_closure::type_bssrdf_random_walk_legacy;
        case compiler::BssrdfMethod::random_walk_skin:
            return cycles_closure::type_bssrdf_random_walk_skin;
    }
    return cycles_closure::type_none;
}

[[nodiscard]] Float dipole_alpha_prime(
    Float rd, Float fourthird_a) noexcept {
    Float x0 = 0.0f;
    Float x1 = 1.0f;
    Float midpoint = 0.5f;
    for (auto iteration = 0u; iteration < 12u; ++iteration) {
        midpoint = 0.5f * (x0 + x1);
        const auto s = sqrt(max(3.0f * (1.0f - midpoint), 0.0f));
        const auto value = 0.5f * midpoint *
                           (1.0f + exp(-fourthird_a * s)) * exp(-s);
        const auto below = value < rd;
        x0 = select(x0, midpoint, below);
        x1 = select(midpoint, x1, below);
    }
    return select(
        select(midpoint, 0.999999f, rd >= 0.995f),
        0.0f,
        rd < 1.0e-4f);
}

[[nodiscard]] Float3 skin_radius_scale(
    Float3 albedo, Float ior) noexcept {
    const auto inverse_eta = 1.0f / ior;
    const auto diffuse_fresnel =
        inverse_eta * (-1.440f * inverse_eta + 0.710f) +
        0.668f + 0.0636f * ior;
    const auto fourthird_a =
        (4.0f / 3.0f) * (1.0f + diffuse_fresnel) /
        max(1.0f - diffuse_fresnel, 1.0e-20f);
    const auto alpha_prime = make_float3(
        dipole_alpha_prime(albedo.x, fourthird_a),
        dipole_alpha_prime(albedo.y, fourthird_a),
        dipole_alpha_prime(albedo.z, fourthird_a));
    return sqrt(max(
        3.0f * (make_float3(1.0f) - alpha_prime),
        make_float3(0.0f)));
}

}// namespace

BssrdfClosureComponent::BssrdfClosureComponent(
    const SurfacePoint &point) noexcept
    : _point{point} {}

BssrdfSetupResult BssrdfClosureComponent::setup(
    const TracedClosure &prototype) const noexcept {
    const auto allocation_weight = sample_weight(prototype.weight);
    const auto allocated =
        allocation_weight >= cycles_closure::closure_weight_cutoff;

    auto ior = clamp(prototype.subsurface_ior, 1.01f, 3.8f);
    auto anisotropy = prototype.subsurface_anisotropy;
    if (prototype.subsurface_method ==
        compiler::BssrdfMethod::random_walk) {
        anisotropy = clamp(anisotropy, -0.99f, 0.99f);
    } else {
        anisotropy = clamp(anisotropy, -0.99f, 0.9f);
    }

    auto radius = max(
        prototype.subsurface_radius * prototype.subsurface_scale,
        make_float3(0.0f));
    const auto diffuse_ancestor =
        _point.diffuse_depth > 0u;
    auto active_x = radius.x >= minimum_radius;
    auto active_y = radius.y >= minimum_radius;
    auto active_z = radius.z >= minimum_radius;
    if (prototype.subsurface_method == compiler::BssrdfMethod::burley) {
        active_x &= !diffuse_ancestor;
        active_y &= !diffuse_ancestor;
        active_z &= !diffuse_ancestor;
    }

    auto bssrdf_weight = make_float3(
        select(0.0f, prototype.weight.x, active_x),
        select(0.0f, prototype.weight.y, active_y),
        select(0.0f, prototype.weight.z, active_z));
    auto fallback_weight = make_float3(
        select(prototype.weight.x, 0.0f, active_x),
        select(prototype.weight.y, 0.0f, active_y),
        select(prototype.weight.z, 0.0f, active_z));
    radius = make_float3(
        select(0.0f, radius.x, active_x),
        select(0.0f, radius.y, active_y),
        select(0.0f, radius.z, active_z));

    const auto channel_count =
        select(0.0f, 1.0f, active_x) +
        select(0.0f, 1.0f, active_y) +
        select(0.0f, 1.0f, active_z);
    const auto setup_valid = allocated & (channel_count > 0.0f);

    if (prototype.subsurface_method ==
            compiler::BssrdfMethod::burley ||
        prototype.subsurface_method ==
            compiler::BssrdfMethod::random_walk_legacy) {
        radius *= 0.25f / pi;
    } else if (prototype.subsurface_method ==
               compiler::BssrdfMethod::random_walk_skin) {
        radius *= skin_radius_scale(
            clamp(prototype.color,
                make_float3(0.0f),
                make_float3(1.0f)),
            ior);
    }

    auto bssrdf = prototype;
    bssrdf.operation = compiler::ClosureOperation::subsurface;
    bssrdf.weight = select(
        make_float3(0.0f), bssrdf_weight, allocated);
    bssrdf.allocation_weight = select(
        0.0f, allocation_weight, allocated);
    bssrdf.sample_weight = select(
        0.0f,
        sample_weight(bssrdf_weight) * channel_count,
        setup_valid);
    bssrdf.setup_valid = setup_valid;
    bssrdf.albedo = bssrdf.weight;
    bssrdf.normal = prototype.normal;
    bssrdf.subsurface_radius = radius;
    bssrdf.subsurface_ior = ior;
    bssrdf.subsurface_anisotropy = anisotropy;
    if (prototype.subsurface_method ==
        compiler::BssrdfMethod::random_walk_skin) {
        bssrdf.roughness = 1.0f;
    } else {
        bssrdf.roughness = clamp(prototype.roughness, 0.0f, 1.0f);
    }
    set_cycles_closure_identity_after_setup(
        bssrdf, bssrdf_closure_type(prototype.subsurface_method));

    fallback_weight = bsdf_allocated_weight(fallback_weight);
    auto fallback = prototype;
    fallback.operation = compiler::ClosureOperation::diffuse;
    fallback.principled_lobe = PrincipledLobe::none;
    fallback.weight = fallback_weight;
    fallback.allocation_weight = sample_weight(fallback_weight);
    fallback.sample_weight = fallback.allocation_weight;
    fallback.setup_valid = true;
    fallback.albedo = fallback_weight;
    fallback.color = make_float3(1.0f);
    fallback.normal = prototype.normal;
    fallback.roughness = 0.0f;
    fallback.diffuse_roughness = 0.0f;
    fallback.evaluation_scale = make_float3(1.0f);
    set_cycles_closure_identity_after_setup(
        fallback, cycles_closure::type_diffuse);
    return {
        .bssrdf = bssrdf,
        .diffuse_fallback = fallback};
}

}// namespace psycles::luisa_backend::detail
