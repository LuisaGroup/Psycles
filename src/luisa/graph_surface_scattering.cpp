#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float fresnel_dielectric_cos(
    Float cosine, Float eta) noexcept {
    auto c = abs(cosine);
    auto g_squared = eta * eta - 1.0f + c * c;
    auto g = sqrt(max(g_squared, 0.0f));
    auto a = (g - c) / max(g + c, 1.0e-20f);
    auto b =
        (c * (g + c) - 1.0f) / select(1.0e-20f,
                                   c * (g - c) + 1.0f,
                                   abs(c * (g - c) + 1.0f) > 1.0e-20f);
    auto regular = 0.5f * a * a * (1.0f + b * b);
    return select(1.0f, regular, g_squared > 0.0f);
}
[[nodiscard]] Float f0_from_ior(Float ior) noexcept {
    auto ratio = (ior - 1.0f) / max(ior + 1.0f, 1.0e-20f);
    return ratio * ratio;
}

[[nodiscard]] Float ior_from_f0(Float f0) noexcept {
    auto root = sqrt(clamp(f0, 0.0f, 0.99f));
    return (1.0f + root) / max(1.0f - root, 1.0e-20f);
}

[[nodiscard]] Float fresnel_dielectric_fss(Float eta) noexcept {
    auto below_one =
        0.997118f +
        eta * (0.1014f - eta * (0.965241f + eta * 0.130607f));
    auto above_one =
        (eta - 1.0f) / max(4.08567f + 1.00071f * eta, 1.0e-20f);
    return select(above_one, below_one, eta < 1.0f);
}

[[nodiscard]] AdjustedIor adjusted_ior(
    const TracedClosure &closure) noexcept {
    auto original_eta = max(closure.ior, 1.0e-5f);
    auto original_f0 = f0_from_ior(original_eta);
    auto adjusted_f0 =
        original_f0 * (2.0f * max(closure.specular_ior_level, 0.0f));
    auto eta_from_adjusted = ior_from_f0(adjusted_f0);
    eta_from_adjusted = select(eta_from_adjusted,
        1.0f / max(eta_from_adjusted, 1.0e-20f),
        original_eta < 1.0f);
    auto should_adjust = closure.specular_ior_level != 0.5f;
    return {
        .eta = select(original_eta, eta_from_adjusted, should_adjust),
        .f0 = select(original_f0, adjusted_f0, should_adjust)};
}

[[nodiscard]] Float3 generalized_dielectric_fresnel(
    Float cosine, Float eta, Float3 f0) noexcept {
    auto real_fresnel = fresnel_dielectric_cos(cosine, eta);
    auto real_f0 = f0_from_ior(eta);
    auto interpolation =
        clamp((real_fresnel - real_f0) / max(1.0f - real_f0, 1.0e-20f),
            0.0f,
            1.0f);
    return lerp(f0, make_float3(1.0f), interpolation);
}

[[nodiscard]] Float3 fresnel_f82_b(Float3 f0, Float3 tint) noexcept {
    constexpr float f = 6.0f / 7.0f;
    constexpr float f5 = f * f * f * f * f;
    auto schlick = lerp(f0, make_float3(1.0f), f5);
    return schlick * (7.0f / (f5 * f)) * (make_float3(1.0f) - tint);
}

[[nodiscard]] Float3 fresnel_f82(
    Float cosine, Float3 f0, Float3 b) noexcept {
    auto mu = clamp(1.0f - cosine, 0.0f, 1.0f);
    auto mu2 = mu * mu;
    auto mu5 = mu2 * mu2 * mu;
    auto schlick = lerp(f0, make_float3(1.0f), mu5);
    return clamp(schlick - b * cosine * mu5 * mu,
        make_float3(0.0f),
        make_float3(1.0f));
}

[[nodiscard]] Float3 ensure_valid_specular_reflection(
    Float3 geometric_normal,
    Float3 incoming,
    Float3 shading_normal) noexcept {
    auto reflected =
        2.0f * dot(shading_normal, incoming) * shading_normal -
        incoming;
    auto incoming_geometric_cosine =
        max(dot(incoming, geometric_normal), 0.0f);
    auto threshold = min(0.9f * incoming_geometric_cosine, 0.01f);
    auto reflection_is_valid =
        dot(geometric_normal, reflected) >= threshold;

    auto tangent = safe_normalize(
        shading_normal -
            dot(shading_normal, geometric_normal) * geometric_normal,
        shading_normal);
    auto incoming_tangent = dot(incoming, tangent);
    auto a = incoming_tangent * incoming_tangent +
             incoming_geometric_cosine * incoming_geometric_cosine;
    auto b = 2.0f * (a + incoming_geometric_cosine * threshold);
    auto c = (threshold + incoming_geometric_cosine) *
             (threshold + incoming_geometric_cosine);
    auto discriminant = max(b * b - 4.0f * a * c, 0.0f);
    auto signed_root = select(b - sqrt(discriminant),
        b + sqrt(discriminant),
        incoming_tangent < 0.0f);
    auto normal_z_squared =
        clamp(0.25f * signed_root / max(a, 1.0e-20f), 0.0f, 1.0f);
    auto corrected = safe_normalize(
        sqrt(max(1.0f - normal_z_squared, 0.0f)) * tangent +
            sqrt(normal_z_squared) * geometric_normal,
        geometric_normal);
    return select(shading_normal,
        corrected,
        (!reflection_is_valid) & (a > 1.0e-20f));
}

[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    const TracedClosure &closure,
    Float incoming_cosine,
    Float3 fss) noexcept {
    if (!closure.preserve_ggx_energy) {
        return {.darkening = make_float3(1.0f),
            .energy_scale = make_float3(1.0f)};
    }

    auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    auto energy = max(cycles_table_2d(services,
                          roughness,
                          incoming_cosine,
                          UInt{cycles45_tables::ggx_e_offset},
                          32u,
                          32u),
        1.0e-20f);
    auto average_energy = cycles_table_1d(services,
        roughness,
        UInt{cycles45_tables::ggx_eavg_offset},
        32u);
    auto missing_factor = (1.0f - energy) / energy;
    auto energy_scale = 1.0f / energy;
    auto fms = fss * average_energy /
               max(make_float3(1.0f) - fss * (1.0f - average_energy),
                   make_float3(1.0e-20f));
    auto darkening =
        (make_float3(1.0f) + fms * missing_factor) / energy_scale;
    return {.darkening = darkening,
        .energy_scale = make_float3(energy_scale)};
}

[[nodiscard]] PrincipledState principled_state(
    const ShaderServices &services,
    const TracedClosure &closure,
    Float3 incoming,
    Float3 glossy_normal) noexcept {
    auto adjusted = adjusted_ior(closure);
    auto tint = max(closure.specular_tint, make_float3(0.0f));
    auto dielectric_f0 = clamp(make_float3(adjusted.f0) * tint,
        make_float3(0.0f),
        make_float3(1.0f));
    auto metallic_f0 =
        clamp(closure.color, make_float3(0.0f), make_float3(1.0f));
    auto metallic_tint = min(tint, make_float3(1.0f));
    auto metallic_b = fresnel_f82_b(metallic_f0, metallic_tint);

    auto real_f0 = f0_from_ior(adjusted.eta);
    auto real_fss = fresnel_dielectric_fss(adjusted.eta);
    auto fss_interpolation =
        clamp((real_fss - real_f0) / max(1.0f - real_f0, 1.0e-20f),
            0.0f,
            1.0f);
    auto dielectric_fss =
        lerp(dielectric_f0, make_float3(1.0f), fss_interpolation);
    auto metallic_fss =
        lerp(metallic_f0, make_float3(1.0f), 1.0f / 21.0f) -
        metallic_b * (1.0f / 126.0f);

    auto incoming_cosine =
        clamp(dot(glossy_normal, incoming), 0.0f, 1.0f);
    auto dielectric_energy =
        ggx_energy(services, closure, incoming_cosine, dielectric_fss);
    auto metallic_energy =
        ggx_energy(services, closure, incoming_cosine, metallic_fss);

    auto roughness = clamp(closure.roughness, 0.0f, 1.0f);
    auto dielectric_z = sqrt(abs(
        (adjusted.eta - 1.0f) / max(adjusted.eta + 1.0f, 1.0e-20f)));
    auto dielectric_s = cycles_table_3d(services,
        roughness,
        incoming_cosine,
        dielectric_z,
        UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset},
        16u,
        16u,
        16u);
    auto metallic_s = cycles_table_3d(services,
        roughness,
        incoming_cosine,
        0.5f,
        UInt{cycles45_tables::ggx_gen_schlick_s_offset},
        16u,
        16u,
        16u);
    auto dielectric_albedo_estimate =
        lerp(dielectric_f0, make_float3(1.0f), dielectric_s);
    auto metallic_albedo_estimate =
        lerp(metallic_f0, make_float3(1.0f), metallic_s);

    // Cycles allocates Principled layers in a defined order. Each
    // physical microfacet closure keeps its own weight and scalar
    // sample_weight; collapsing these into a single virtual closure
    // changes both closure selection and the rescaled third BSDF random
    // dimension.
    auto metallic = clamp(closure.metallic, 0.0f, 1.0f);
    auto metallic_active =
        metallic > cycles_closure::closure_weight_cutoff;
    auto metallic_factor = select(0.0f, metallic, metallic_active);
    auto lower_weight_factor =
        select(1.0f, 1.0f - metallic, metallic_active);
    auto metallic_pre_weight = closure.weight * metallic_factor;
    auto metallic_allocation_weight =
        sample_weight(max(metallic_pre_weight, make_float3(0.0f)));
    auto metallic_allocated = metallic_allocation_weight >=
                              cycles_closure::closure_weight_cutoff;
    auto metallic_weight = select(make_float3(0.0f),
        metallic_pre_weight * metallic_energy.darkening,
        metallic_allocated);
    auto metallic_albedo = metallic_weight * metallic_albedo_estimate;
    auto metallic_sample_weight = select(0.0f,
        metallic_allocation_weight *
            sample_weight(metallic_albedo_estimate) *
            sample_weight(metallic_energy.darkening),
        metallic_allocated);

    auto dielectric_requested = adjusted.eta != 1.0f;
    auto dielectric_pre_weight = select(make_float3(0.0f),
        closure.weight * lower_weight_factor,
        dielectric_requested);
    auto dielectric_allocation_weight =
        sample_weight(max(dielectric_pre_weight, make_float3(0.0f)));
    auto dielectric_allocated =
        dielectric_requested &
        (dielectric_allocation_weight >=
            cycles_closure::closure_weight_cutoff);
    auto dielectric_weight = select(make_float3(0.0f),
        dielectric_pre_weight * dielectric_energy.darkening,
        dielectric_allocated);
    auto dielectric_albedo =
        dielectric_weight * dielectric_albedo_estimate;
    auto dielectric_sample_weight = select(0.0f,
        dielectric_allocation_weight *
            sample_weight(dielectric_albedo_estimate) *
            sample_weight(dielectric_energy.darkening),
        dielectric_allocated);
    auto dielectric_layer_albedo_ratio = safe_divide_components(
        dielectric_albedo, dielectric_pre_weight);
    auto lower_layer_factor = select(1.0f,
        clamp(1.0f - max_component(dielectric_layer_albedo_ratio),
            0.0f,
            1.0f),
        dielectric_allocated);
    auto diffuse_weight = closure.weight * lower_weight_factor *
                          lower_layer_factor *
                          max(closure.color, make_float3(0.0f));
    return {.eta = adjusted.eta,
        .dielectric_f0 = dielectric_f0,
        .metallic_f0 = metallic_f0,
        .metallic_b = metallic_b,
        .dielectric_energy_scale = dielectric_energy.energy_scale,
        .metallic_energy_scale = metallic_energy.energy_scale,
        .dielectric_weight = dielectric_weight,
        .dielectric_allocation_weight = dielectric_allocation_weight,
        .dielectric_sample_weight = dielectric_sample_weight,
        .dielectric_albedo = dielectric_albedo,
        .metallic_weight = metallic_weight,
        .metallic_allocation_weight = metallic_allocation_weight,
        .metallic_sample_weight = metallic_sample_weight,
        .metallic_albedo = metallic_albedo,
        .diffuse_weight = diffuse_weight};
}

[[nodiscard]] bool is_scattering_operation(
    compiler::ClosureOperation operation) noexcept {
    switch (operation) {
    case compiler::ClosureOperation::diffuse:
    case compiler::ClosureOperation::translucent:
    case compiler::ClosureOperation::principled:
    case compiler::ClosureOperation::glossy:
    case compiler::ClosureOperation::transparent:
        return true;
    case compiler::ClosureOperation::null_closure:
    case compiler::ClosureOperation::add:
    case compiler::ClosureOperation::mix:
    case compiler::ClosureOperation::emission:
        return false;
    }
    return false;
}

[[nodiscard]] Float closure_sample_weight(
    const TracedClosure &closure) noexcept {
    return closure.sample_weight;
}

[[nodiscard]] Bool closure_allocated(
    const TracedClosure &closure) noexcept {
    if (!is_scattering_operation(closure.operation)) {
        return false;
    }
    return closure.allocation_weight >=
           cycles_closure::closure_weight_cutoff;
}

[[nodiscard]] UInt cycles_runtime_flags(const TracedClosure &closure,
    Float glossy_filter_roughness) noexcept {
    UInt flags = 0u;
    switch (closure.operation) {
    case compiler::ClosureOperation::diffuse:
        flags = cycles_closure::runtime_bsdf |
                cycles_closure::runtime_bsdf_has_eval;
        break;
    case compiler::ClosureOperation::translucent:
        flags = cycles_closure::runtime_bsdf |
                cycles_closure::runtime_bsdf_has_eval |
                cycles_closure::runtime_bsdf_has_transmission;
        break;
    case compiler::ClosureOperation::principled:
    case compiler::ClosureOperation::glossy: {
        auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
        alpha *= alpha;
        alpha = max(alpha, glossy_filter_roughness);
        flags =
            cycles_closure::runtime_bsdf |
            select(0u,
                cycles_closure::runtime_bsdf_has_eval,
                alpha * alpha >
                    cycles_closure::microfacet_singular_alpha_product);
        break;
    }
    case compiler::ClosureOperation::transparent:
        flags = cycles_closure::runtime_bsdf |
                cycles_closure::runtime_transparent;
        break;
    case compiler::ClosureOperation::emission:
        return cycles_closure::runtime_emission;
    case compiler::ClosureOperation::null_closure:
    case compiler::ClosureOperation::add:
    case compiler::ClosureOperation::mix:
        return 0u;
    }
    flags |= select(0u,
        cycles_closure::runtime_bsdf_has_eval,
        glossy_filter_roughness * glossy_filter_roughness >
            cycles_closure::microfacet_singular_alpha_product);
    return select(0u, flags, closure_allocated(closure));
}

[[nodiscard]] UInt cycles_closure_type(
    const TracedClosure &closure) noexcept {
    switch (closure.operation) {
    case compiler::ClosureOperation::diffuse:
        return select(cycles_closure::type_oren_nayar,
            cycles_closure::type_diffuse,
            closure.roughness < 1.0e-5f);
    case compiler::ClosureOperation::translucent:
        return cycles_closure::type_translucent;
    case compiler::ClosureOperation::glossy:
        return cycles_closure::type_microfacet_ggx;
    case compiler::ClosureOperation::transparent:
        return cycles_closure::type_transparent;
    case compiler::ClosureOperation::principled:
        return cycles_closure::type_microfacet_ggx;
    case compiler::ClosureOperation::null_closure:
    case compiler::ClosureOperation::add:
    case compiler::ClosureOperation::mix:
    case compiler::ClosureOperation::emission:
        return cycles_closure::type_none;
    }
    return cycles_closure::type_none;
}

[[nodiscard]] ClosureSelectionState closure_selection_state(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &closure,
    Float3 incoming,
    const SurfaceQuery &query) noexcept {
    static_cast<void>(services);
    const auto is_diffuse =
        closure.operation == compiler::ClosureOperation::diffuse;
    const auto is_translucent =
        closure.operation == compiler::ClosureOperation::translucent;
    const auto is_principled =
        closure.operation == compiler::ClosureOperation::principled;
    const auto is_glossy =
        closure.operation == compiler::ClosureOperation::glossy;
    const auto is_transparent =
        closure.operation == compiler::ClosureOperation::transparent;
    const auto diffuse_enabled =
        (query.lobe_mask & static_cast<std::uint32_t>(event_diffuse)) !=
        0u;
    const auto glossy_enabled =
        (query.lobe_mask & static_cast<std::uint32_t>(event_glossy)) !=
        0u;
    const auto transparent_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transparent)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transmission)) != 0u;
    auto eligible = is_transparent ? transparent_enabled
                    : is_translucent
                        ? (diffuse_enabled & transmission_enabled)
                    : is_diffuse    ? diffuse_enabled
                    : is_principled ? glossy_enabled
                    : is_glossy     ? glossy_enabled
                                    : Bool{false};
    eligible &= closure_allocated(closure);
    const auto glossy_normal = ensure_valid_specular_reflection(
        point.geometric_normal, incoming, closure.normal);
    return {.eligible = eligible,
        .weight =
            select(0.0f, closure_sample_weight(closure), eligible),
        .glossy_normal = glossy_normal};
}

[[nodiscard]] Float oren_nayar_g(Float cosine) noexcept {
    auto c = clamp(cosine, 0.0f, 1.0f);
    auto sine = sqrt(max(1.0f - c * c, 0.0f));
    auto theta = acos(c);
    auto safe_cosine = max(c, 1.0e-6f);
    auto regular = sine * (theta - 2.0f / 3.0f - sine * c) +
                   2.0f / 3.0f * (sine / safe_cosine) *
                       (1.0f - sine * sine * sine);
    auto taylor = (pi * 0.5f - 2.0f / 3.0f) - c;
    return select(regular, taylor, c < 1.0e-6f);
}

[[nodiscard]] Float3 diffuse_intensity(const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    auto nl = max(dot(closure.normal, outgoing), 0.0f);
    auto lambert = make_float3(nl * inverse_pi);

    auto sigma = clamp(
        closure.operation == compiler::ClosureOperation::principled
            ? closure.diffuse_roughness
            : closure.roughness,
        0.0f,
        1.0f);
    auto a = 1.0f / (pi + sigma * (pi * 0.5f - 2.0f / 3.0f));
    auto b = sigma * a;
    auto nv = max(dot(closure.normal, incoming), 0.0f);
    auto t = dot(outgoing, incoming) - nl * nv;
    auto positive_t = t > 0.0f;
    t = select(t, t / (max(nl, nv) + 1.17549435e-38f), positive_t);

    auto single_scatter = a + b * t;
    auto albedo =
        clamp(closure.color, make_float3(0.0f), make_float3(1.0f));
    auto e_average = a * pi + ((two_pi - 5.6f) / 3.0f) * b;
    auto albedo_squared = albedo * albedo;
    auto e_ms = inverse_pi * albedo_squared *
                (e_average / (1.0f - e_average)) /
                (make_float3(1.0f) - albedo * (1.0f - e_average));
    auto e_incoming = a * pi + b * oren_nayar_g(nv);
    auto multiscatter_term = e_ms * (1.0f - e_incoming);
    auto e_outgoing = a * pi + b * oren_nayar_g(nl);
    auto oren_nayar = nl * (make_float3(single_scatter) +
                               multiscatter_term * (1.0f - e_outgoing));

    auto use_lambert = sigma < 1.0e-5f;
    return select(oren_nayar, lambert, use_lambert);
}

[[nodiscard]] Float ggx_distribution(
    Float n_dot_h, Float alpha) noexcept {
    auto alpha2 = alpha * alpha;
    auto denominator = n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / max(pi * denominator * denominator, 1.0e-20f);
}

[[nodiscard]] Float microfacet_alpha(const TracedClosure &closure,
    Float glossy_filter_roughness) noexcept {
    // Cycles applies bsdf_microfacet_blur after closure setup. Keep the
    // original closure roughness for sample weights, layering, and
    // energy compensation; only evaluation and sampling see this
    // widened alpha.
    auto setup_alpha = clamp(closure.roughness, 0.0f, 1.0f);
    setup_alpha *= setup_alpha;
    return max(max(setup_alpha, glossy_filter_roughness), 1.0e-3f);
}

[[nodiscard]] Float smith_g1(Float n_dot_v, Float alpha) noexcept {
    auto cosine = max(n_dot_v, 1.0e-6f);
    auto tangent2 = max(1.0f / (cosine * cosine) - 1.0f, 0.0f);
    auto lambda = 0.5f * (sqrt(1.0f + alpha * alpha * tangent2) - 1.0f);
    return 1.0f / (1.0f + lambda);
}

[[nodiscard]] Float3 specular_f0(
    const TracedClosure &closure) noexcept {
    auto dielectric =
        (closure.ior - 1.0f) / max(closure.ior + 1.0f, 1.0e-20f);
    dielectric *= dielectric;
    return lerp(make_float3(dielectric),
        clamp(closure.color, make_float3(0.0f), make_float3(1.0f)),
        closure.metallic);
}

[[nodiscard]] Float3 microfacet_intensity(
    const ShaderServices &services,
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    static_cast<void>(services);
    auto n_dot_v = max(dot(glossy_normal, incoming), 0.0f);
    auto n_dot_l = max(dot(glossy_normal, outgoing), 0.0f);
    auto half_vector =
        safe_normalize(incoming + outgoing, glossy_normal);
    auto n_dot_h = max(dot(glossy_normal, half_vector), 0.0f);
    auto v_dot_h = max(dot(incoming, half_vector), 0.0f);
    auto alpha = microfacet_alpha(closure, glossy_filter_roughness);
    auto distribution = ggx_distribution(n_dot_h, alpha);
    auto lambda_v = 1.0f / smith_g1(n_dot_v, alpha) - 1.0f;
    auto lambda_l = 1.0f / smith_g1(n_dot_l, alpha) - 1.0f;
    auto geometry = 1.0f / (1.0f + lambda_v + lambda_l);
    Float3 fresnel;
    if (closure.operation == compiler::ClosureOperation::principled) {
        if (closure.principled_lobe == PrincipledLobe::metallic) {
            fresnel =
                fresnel_f82(
                    v_dot_h, closure.color, closure.specular_tint) *
                closure.evaluation_scale;
        } else {
            fresnel = generalized_dielectric_fresnel(
                          v_dot_h, closure.ior, closure.color) *
                      closure.evaluation_scale;
        }
    } else {
        auto f0 = specular_f0(closure);
        fresnel =
            f0 + (make_float3(1.0f) - f0) * pow(1.0f - v_dot_h, 5.0f);
    }
    auto intensity = fresnel * distribution * geometry /
                     max(4.0f * n_dot_v, 1.0e-20f);
    return select(make_float3(0.0f),
        intensity,
        (n_dot_v > 0.0f) & (n_dot_l > 0.0f) & (n_dot_h > 0.0f) &
            (v_dot_h > 0.0f));
}

[[nodiscard]] Float microfacet_pdf(const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    auto half_vector =
        safe_normalize(incoming + outgoing, glossy_normal);
    auto n_dot_h = max(dot(glossy_normal, half_vector), 0.0f);
    auto v_dot_h = max(dot(incoming, half_vector), 0.0f);
    auto alpha = microfacet_alpha(closure, glossy_filter_roughness);
    auto n_dot_v = max(dot(glossy_normal, incoming), 0.0f);
    auto n_dot_l = max(dot(glossy_normal, outgoing), 0.0f);
    auto pdf = ggx_distribution(n_dot_h, alpha) *
               smith_g1(n_dot_v, alpha) / max(4.0f * n_dot_v, 1.0e-20f);
    return select(0.0f,
        pdf,
        (n_dot_v > 0.0f) & (n_dot_l > 0.0f) & (n_dot_h > 0.0f) &
            (v_dot_h > 0.0f));
}

[[nodiscard]] Float3 sample_ggx(const TracedClosure &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    return cycles_sample_mapping::sample_ggx_visible_normal_reflection(
        glossy_normal,
        incoming,
        microfacet_alpha(closure, glossy_filter_roughness),
        random);
}

[[nodiscard]] Float3 sample_cosine_hemisphere(
    Float3 normal, Float2 random) noexcept {
    return cycles_sample_mapping::sample_cosine_hemisphere(
        normal, random)
        .direction;
}

[[nodiscard]] Float3 rotate_euler(
    Float3 value, Float3 rotation) noexcept {
    auto sx = sin(rotation.x);
    auto cx = cos(rotation.x);
    auto sy = sin(rotation.y);
    auto cy = cos(rotation.y);
    auto sz = sin(rotation.z);
    auto cz = cos(rotation.z);
    auto x_rotated = make_float3(value.x,
        cx * value.y - sx * value.z,
        sx * value.y + cx * value.z);
    auto y_rotated = make_float3(cy * x_rotated.x + sy * x_rotated.z,
        x_rotated.y,
        -sy * x_rotated.x + cy * x_rotated.z);
    return make_float3(cz * y_rotated.x - sz * y_rotated.y,
        sz * y_rotated.x + cz * y_rotated.y,
        y_rotated.z);
}

[[nodiscard]] Float3 rotate_euler_transposed(
    Float3 value, Float3 rotation) noexcept {
    auto sx = sin(rotation.x);
    auto cx = cos(rotation.x);
    auto sy = sin(rotation.y);
    auto cy = cos(rotation.y);
    auto sz = sin(rotation.z);
    auto cz = cos(rotation.z);
    return make_float3(
        cy * cz * value.x + cy * sz * value.y - sy * value.z,
        (sy * sx * cz - cx * sz) * value.x +
            (sy * sx * sz + cx * cz) * value.y + cy * sx * value.z,
        (sy * cx * cz + sx * sz) * value.x +
            (sy * cx * sz - sx * cz) * value.y + cy * cx * value.z);
}

[[nodiscard]] Float3 safe_divide_components(
    Float3 numerator, Float3 denominator) noexcept {
    return make_float3(
        select(
            0.0f, numerator.x / denominator.x, denominator.x != 0.0f),
        select(
            0.0f, numerator.y / denominator.y, denominator.y != 0.0f),
        select(
            0.0f, numerator.z / denominator.z, denominator.z != 0.0f));
}

} // namespace psycles::luisa_backend::detail
