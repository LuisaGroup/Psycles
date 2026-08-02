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

[[nodiscard]] Float3 maybe_ensure_valid_specular_reflection(
    const SurfacePoint &point,
    Float3 incoming,
    Float3 shading_normal) noexcept {
    const auto correction_enabled =
        point.use_bump_map_correction &
        !all(point.geometric_normal == shading_normal);
    return select(shading_normal,
        ensure_valid_specular_reflection(
            point.geometric_normal, incoming, shading_normal),
        correction_enabled);
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
    case compiler::ClosureOperation::glass:
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

[[nodiscard]] Float bump_shadowing_term(const SurfacePoint &point,
    Float3 smooth_normal,
    const TracedClosure &closure,
    Float3 direction,
    Bool is_evaluation) noexcept {
    // Cycles classifies Sheen, Lambert/Oren-Nayar, and Translucent in the
    // contiguous diffuse closure range. Express that semantic class here,
    // independently of the current host-side operation representation.
    const auto diffuse_closure =
        closure.operation == compiler::ClosureOperation::diffuse ||
        closure.operation == compiler::ClosureOperation::translucent ||
        (closure.operation == compiler::ClosureOperation::principled &&
            closure.principled_lobe == PrincipledLobe::sheen);

    const auto normals_equal = all(closure.normal == smooth_normal);
    const auto cosine_smooth_direction = dot(smooth_normal, direction);
    const auto cosine_smooth_closure =
        dot(smooth_normal, closure.normal);
    const auto cosine_closure_direction =
        dot(closure.normal, direction);

    // A closure-specific normal must not leak reflection or refraction
    // through the opposite side of the shader-wide smooth surface. Diffuse
    // closures use the same predicate during sample and eval; glossy
    // closures only use it during eval, exactly as Cycles does.
    const auto wrong_smooth_hemisphere =
        cosine_smooth_direction * cosine_smooth_closure *
            cosine_closure_direction <
        0.0f;
    const auto reject = wrong_smooth_hemisphere &
                        (is_evaluation | Bool{diffuse_closure});

    const auto cosine_i = abs(cosine_smooth_direction);
    const auto cosine_d = abs(cosine_smooth_closure);
    const auto tangent_d_squared =
        max(1.0f / max(cosine_d * cosine_d, 1.0e-20f) - 1.0f,
            0.0f);
    const auto bump_alpha_squared =
        clamp(0.125f * tangent_d_squared, 0.0f, 1.0f);
    const auto tangent_i_squared =
        max(1.0f / max(cosine_i * cosine_i, 1.0e-20f) - 1.0f,
            0.0f);
    const auto lambda = 0.5f *
                        (sqrt(1.0f +
                                  bump_alpha_squared *
                                      tangent_i_squared) -
                            1.0f);
    const auto smoothed = 1.0f / (1.0f + lambda);
    const auto smoothing_domain =
        Bool{diffuse_closure} & point.use_bump_map_correction &
        (cosine_d < 1.0f) & (cosine_i < 1.0f);
    const auto grazing_reject =
        smoothing_domain & (cosine_i < 1.0e-6f);
    auto result = select(1.0f,
        smoothed,
        smoothing_domain & !grazing_reject);
    result = select(result, 0.0f, reject | grazing_reject);
    // Cycles exits before all other predicates when the two normals compare
    // exactly equal. Retain that ordering even at degenerate dot products.
    return select(result, 1.0f, normals_equal);
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
        if (closure.operation ==
                compiler::ClosureOperation::principled &&
            closure.principled_lobe == PrincipledLobe::sheen) {
            flags = cycles_closure::runtime_bsdf |
                    cycles_closure::runtime_bsdf_has_eval;
            break;
        }
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
    case compiler::ClosureOperation::glass: {
        auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
        alpha *= alpha;
        alpha = max(alpha, glossy_filter_roughness);
        flags = cycles_closure::runtime_bsdf |
                cycles_closure::runtime_bsdf_has_transmission |
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
    return select(0u,
        flags,
        closure_allocated(closure) & closure.setup_valid);
}

[[nodiscard]] UInt cycles_closure_type(
    const TracedClosure &closure) noexcept {
    UInt type = cycles_closure::type_none;
    switch (closure.operation) {
    case compiler::ClosureOperation::diffuse:
        type = select(cycles_closure::type_oren_nayar,
            cycles_closure::type_diffuse,
            closure.roughness < 1.0e-5f);
        break;
    case compiler::ClosureOperation::translucent:
        type = cycles_closure::type_translucent;
        break;
    case compiler::ClosureOperation::glossy:
        type = cycles_closure::type_microfacet_ggx;
        break;
    case compiler::ClosureOperation::glass:
        if (closure.preserve_ggx_energy) {
            type = cycles_closure::type_microfacet_multi_ggx_glass;
            break;
        }
        type = closure.beckmann
                   ? UInt{cycles_closure::type_microfacet_beckmann_glass}
                   : UInt{cycles_closure::type_microfacet_ggx_glass};
        break;
    case compiler::ClosureOperation::transparent:
        type = cycles_closure::type_transparent;
        break;
    case compiler::ClosureOperation::principled:
        type = closure.principled_lobe == PrincipledLobe::sheen
                   ? cycles_closure::type_sheen
                   : cycles_closure::type_microfacet_ggx;
        break;
    case compiler::ClosureOperation::null_closure:
    case compiler::ClosureOperation::add:
    case compiler::ClosureOperation::mix:
    case compiler::ClosureOperation::emission:
        break;
    }
    return select(UInt{cycles_closure::type_none},
        type,
        closure.setup_valid);
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
    const auto is_sheen =
        is_principled &&
        closure.principled_lobe == PrincipledLobe::sheen;
    const auto is_glossy =
        closure.operation == compiler::ClosureOperation::glossy;
    const auto is_glass =
        closure.operation == compiler::ClosureOperation::glass;
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
                    : is_sheen      ? diffuse_enabled
                    : is_principled ? glossy_enabled
                    : is_glossy     ? glossy_enabled
                    : is_glass      ? (glossy_enabled | transmission_enabled)
                                    : Bool{false};
    eligible &= closure_allocated(closure) & closure.setup_valid;
    const auto glossy_normal = is_sheen
                                   ? closure.normal
                                   : maybe_ensure_valid_specular_reflection(
                                         point, incoming, closure.normal);
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

[[nodiscard]] Float sheen_intensity(const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    const auto basis =
        cycles_sample_mapping::make_orthonormals_safe_tangent(
            closure.normal, incoming);
    const auto local_outgoing = make_float3(
        dot(outgoing, basis.tangent),
        dot(outgoing, basis.bitangent),
        dot(outgoing, closure.normal));
    const auto a = closure.sheen_transform_a;
    const auto b = closure.sheen_transform_b;
    const auto length_squared =
        (a * local_outgoing.x + b * local_outgoing.z) *
            (a * local_outgoing.x + b * local_outgoing.z) +
        (a * local_outgoing.y) * (a * local_outgoing.y) +
        local_outgoing.z * local_outgoing.z;
    const auto transformed = a / length_squared;
    return inverse_pi * max(local_outgoing.z, 0.0f) *
           transformed * transformed;
}

[[nodiscard]] Float3 sample_sheen(const TracedClosure &closure,
    Float3 incoming,
    Float2 random) noexcept {
    const auto basis =
        cycles_sample_mapping::make_orthonormals_safe_tangent(
            closure.normal, incoming);
    const auto disk =
        cycles_sample_mapping::sample_uniform_disk(random);
    const auto disk_z = sqrt(max(1.0f - dot(disk, disk), 0.0f));
    const auto local_outgoing = normalize(make_float3(
        disk.x - disk_z * closure.sheen_transform_b,
        disk.y,
        disk_z * closure.sheen_transform_a));
    return basis.tangent * local_outgoing.x +
           basis.bitangent * local_outgoing.y +
           closure.normal * local_outgoing.z;
}

namespace {

[[nodiscard]] Float glass_microfacet_alpha(
    const TracedClosure &closure,
    Float glossy_filter_roughness) noexcept {
    auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
    alpha *= alpha;
    return max(alpha, glossy_filter_roughness);
}

[[nodiscard]] Float glass_microfacet_distribution(
    const TracedClosure &closure,
    Float n_dot_h,
    Float alpha) noexcept {
    if (!closure.beckmann) {
        return ggx_distribution(n_dot_h, alpha);
    }
    auto cosine_squared = min(n_dot_h * n_dot_h, 1.0f);
    auto alpha_squared = alpha * alpha;
    auto exponent = (1.0f - cosine_squared) /
                    max(cosine_squared * alpha_squared, 1.0e-20f);
    auto denominator = exp(min(exponent, 80.0f)) * pi *
                       max(alpha_squared, 1.0e-20f) *
                       cosine_squared * cosine_squared;
    return select(0.0f,
        1.0f / max(denominator, 1.0e-20f),
        n_dot_h > 0.0f);
}

[[nodiscard]] Float glass_microfacet_lambda(
    const TracedClosure &closure,
    Float n_dot_v,
    Float alpha) noexcept {
    auto cosine_squared = n_dot_v * n_dot_v;
    auto squared_alpha_tangent = alpha * alpha *
                                 max(1.0f /
                                             max(cosine_squared, 1.0e-20f) -
                                         1.0f,
                                     0.0f);
    if (!closure.beckmann) {
        return 0.5f *
               (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
    }
    auto a = rsqrt(max(squared_alpha_tangent, 1.0e-20f));
    auto approximation =
        ((0.396f * a - 1.259f) * a + 1.0f) /
        max((2.181f * a + 3.535f) * a, 1.0e-20f);
    return select(approximation,
        0.0f,
        squared_alpha_tangent < 0.39f);
}

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

[[nodiscard]] GlassGeometry glass_geometry(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 normal) noexcept {
    auto cosine_outgoing = dot(normal, outgoing);
    auto transmission = cosine_outgoing < 0.0f;
    auto unnormalized_half = select(incoming + outgoing,
        -(closure.ior * outgoing + incoming),
        transmission);
    auto inverse_half_length =
        rsqrt(max(dot(unnormalized_half, unnormalized_half), 1.0e-20f));
    auto half_vector = unnormalized_half * inverse_half_length;
    return {.half_vector = half_vector,
        .inverse_half_length = inverse_half_length,
        .cosine_incoming = dot(normal, incoming),
        .cosine_outgoing = cosine_outgoing,
        .cosine_half_incoming = dot(half_vector, incoming),
        .cosine_normal_half = dot(normal, half_vector),
        .cosine_half_outgoing = dot(half_vector, outgoing),
        .transmission = transmission};
}

[[nodiscard]] Float glass_reflection_probability(Float fresnel,
    Bool reflection_allowed,
    Bool transmission_allowed) noexcept {
    auto reflection_energy =
        select(0.0f, fresnel, reflection_allowed);
    auto transmission_energy =
        select(0.0f, 1.0f - fresnel, transmission_allowed);
    return reflection_energy /
           max(reflection_energy + transmission_energy, 1.0e-20f);
}

}// namespace

[[nodiscard]] Float3 glass_microfacet_intensity(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    auto geometry =
        glass_geometry(closure, incoming, outgoing, glossy_normal);
    auto setup_alpha = glass_microfacet_alpha(
        closure, glossy_filter_roughness);
    auto alpha = max(setup_alpha,
        1.0e-7f);
    auto distribution = glass_microfacet_distribution(
        closure, geometry.cosine_normal_half, alpha);
    auto lambda_incoming = glass_microfacet_lambda(
        closure, geometry.cosine_incoming, alpha);
    auto lambda_outgoing = glass_microfacet_lambda(
        closure, geometry.cosine_outgoing, alpha);
    auto reflection = fresnel_dielectric_cos(
        geometry.cosine_half_incoming, closure.ior);
    auto lobe = select(reflection,
        1.0f - reflection,
        geometry.transmission);
    auto transmission_jacobian =
        closure.ior * closure.ior *
        geometry.inverse_half_length * geometry.inverse_half_length *
        abs(geometry.cosine_half_incoming *
            geometry.cosine_half_outgoing);
    auto common = distribution /
                  max(geometry.cosine_incoming, 1.0e-20f) *
                  select(0.25f,
                      transmission_jacobian,
                      geometry.transmission);
    auto intensity = closure.color * lobe * common /
                     (1.0f + lambda_incoming + lambda_outgoing);
    auto valid = (geometry.cosine_incoming > 0.0f) &
                 (geometry.cosine_normal_half > 0.0f) &
                 (geometry.cosine_half_incoming > 0.0f) &
                 (setup_alpha * setup_alpha >
                     cycles_closure::microfacet_singular_alpha_product);
    return select(make_float3(0.0f), intensity, valid);
}

[[nodiscard]] Float glass_microfacet_pdf(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Bool reflection_allowed,
    Bool transmission_allowed,
    Float glossy_filter_roughness) noexcept {
    auto geometry =
        glass_geometry(closure, incoming, outgoing, glossy_normal);
    auto setup_alpha = glass_microfacet_alpha(
        closure, glossy_filter_roughness);
    auto alpha = max(setup_alpha,
        1.0e-7f);
    auto distribution = glass_microfacet_distribution(
        closure, geometry.cosine_normal_half, alpha);
    auto reflection = fresnel_dielectric_cos(
        geometry.cosine_half_incoming, closure.ior);
    auto reflection_probability = glass_reflection_probability(
        reflection, reflection_allowed, transmission_allowed);
    auto lobe_probability = select(reflection_probability,
        1.0f - reflection_probability,
        geometry.transmission);
    Float directional_pdf;
    if (closure.beckmann) {
        auto reflection_jacobian =
            1.0f /
            max(4.0f * geometry.cosine_half_incoming, 1.0e-20f);
        auto transmission_jacobian =
            abs(geometry.cosine_half_outgoing) * closure.ior *
            closure.ior * geometry.inverse_half_length *
            geometry.inverse_half_length;
        directional_pdf = distribution *
                          max(geometry.cosine_normal_half, 0.0f) *
                          select(reflection_jacobian,
                              transmission_jacobian,
                              geometry.transmission);
    } else {
        auto lambda_incoming = glass_microfacet_lambda(
            closure, geometry.cosine_incoming, alpha);
        auto transmission_jacobian =
            closure.ior * closure.ior *
            geometry.inverse_half_length *
            geometry.inverse_half_length *
            abs(geometry.cosine_half_incoming *
                geometry.cosine_half_outgoing);
        auto common = distribution /
                      max(geometry.cosine_incoming, 1.0e-20f) *
                      select(0.25f,
                          transmission_jacobian,
                          geometry.transmission);
        directional_pdf = common / (1.0f + lambda_incoming);
    }
    auto lobe_allowed = select(reflection_allowed,
        transmission_allowed,
        geometry.transmission);
    auto valid = (geometry.cosine_incoming > 0.0f) &
                 (geometry.cosine_normal_half > 0.0f) &
                 (geometry.cosine_half_incoming > 0.0f) & lobe_allowed;
    valid &= setup_alpha * setup_alpha >
             cycles_closure::microfacet_singular_alpha_product;
    return select(0.0f,
        directional_pdf * lobe_probability,
        valid);
}

[[nodiscard]] GlassSample sample_glass(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 geometric_normal,
    Float3 glossy_normal,
    Float2 random_direction,
    Float random_lobe,
    Bool reflection_allowed,
    Bool transmission_allowed,
    Float glossy_filter_roughness) noexcept {
    auto alpha = glass_microfacet_alpha(
        closure, glossy_filter_roughness);
    auto singular = alpha * alpha <=
                    cycles_closure::microfacet_singular_alpha_product;
    auto sampling_alpha = max(alpha, 1.0e-7f);
    Float3 sampled_half;
    if (closure.beckmann) {
        auto tangent_squared = -sampling_alpha * sampling_alpha *
                               log(max(1.0f - random_direction.x,
                                   1.0e-7f));
        auto cosine = rsqrt(1.0f + tangent_squared);
        auto sine = sqrt(max(1.0f - cosine * cosine, 0.0f));
        auto phi = two_pi * random_direction.y;
        auto basis = cycles_sample_mapping::make_orthonormals(
            glossy_normal);
        sampled_half = basis.tangent * (sine * cos(phi)) +
                       basis.bitangent * (sine * sin(phi)) +
                       glossy_normal * cosine;
    } else {
        sampled_half = cycles_sample_mapping::sample_ggx_visible_normal(
            glossy_normal,
            incoming,
            sampling_alpha,
            random_direction);
    }
    auto half_vector = select(sampled_half, glossy_normal, singular);
    auto cosine_half_incoming = dot(half_vector, incoming);
    auto eta_squared_cosine_transmitted =
        closure.ior * closure.ior -
        (1.0f - cosine_half_incoming * cosine_half_incoming);
    auto total_internal_reflection =
        eta_squared_cosine_transmitted <= 0.0f;
    auto reflection = fresnel_dielectric_cos(
        cosine_half_incoming, closure.ior);
    auto effective_transmission_allowed =
        transmission_allowed & (!total_internal_reflection);
    auto reflection_probability = glass_reflection_probability(
        reflection,
        reflection_allowed,
        effective_transmission_allowed);
    auto transmission =
        (random_lobe >= reflection_probability) &
        effective_transmission_allowed;
    auto reflected = 2.0f * cosine_half_incoming * half_vector -
                     incoming;
    auto cosine_half_outgoing =
        -sqrt(max(eta_squared_cosine_transmitted, 0.0f)) /
        max(closure.ior, 1.0e-20f);
    auto inverse_eta = 1.0f / max(closure.ior, 1.0e-20f);
    auto transmitted =
        (inverse_eta * cosine_half_incoming +
            cosine_half_outgoing) *
            half_vector -
        inverse_eta * incoming;
    auto direction = select(reflected, transmitted, transmission);
    auto lobe = select(reflection,
        1.0f - reflection,
        transmission);
    auto lobe_probability = select(reflection_probability,
        1.0f - reflection_probability,
        transmission);
    auto has_lobe = select(reflection_allowed,
        effective_transmission_allowed,
        transmission);
    auto expected_negative = transmission;
    auto shading_negative = dot(glossy_normal, direction) < 0.0f;
    auto geometric_negative = dot(geometric_normal, direction) < 0.0f;
    auto shading_hemisphere_valid = select(!shading_negative,
        shading_negative,
        expected_negative);
    auto geometric_hemisphere_valid = select(!geometric_negative,
        geometric_negative,
        expected_negative);
    auto valid = has_lobe & (cosine_half_incoming > 0.0f) &
                 shading_hemisphere_valid & geometric_hemisphere_valid;
    return {.direction = direction,
        .singular_evaluation =
            closure.weight * closure.color * lobe * 1.0e6f,
        .singular_pdf = lobe_probability * 1.0e6f,
        .eta = select(1.0f, closure.ior, transmission),
        .alpha = alpha,
        .transmission = transmission,
        .singular = singular,
        .valid = valid};
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
