#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {
namespace {

template<typename Closure>
[[nodiscard]] Bool has_kind(
    const Closure &closure,
    SurfaceClosureKind kind) noexcept {
    return closure.kind == static_cast<std::uint32_t>(kind);
}

template<typename Closure>
[[nodiscard]] Bool has_lobe(
    const Closure &closure,
    SurfaceClosureLobe lobe) noexcept {
    return closure.lobe == static_cast<std::uint32_t>(lobe);
}

[[nodiscard]] SurfaceClosureIdentityExpression closure_identity(
    const SurfaceClosureRecord &closure) noexcept {
    return {
        .kind = Expr<std::uint32_t>{closure.kind.expression()},
        .lobe = Expr<std::uint32_t>{closure.lobe.expression()},
        .bssrdf_method = Expr<std::uint32_t>{
            closure.bssrdf_method.expression()},
        .allocation_weight =
            Expr<float>{closure.allocation_weight.expression()},
        .setup_valid = Expr<bool>{closure.setup_valid.expression()},
        .roughness = Expr<float>{closure.roughness.expression()},
        .preserve_ggx_energy =
            Expr<bool>{closure.preserve_ggx_energy.expression()},
        .beckmann = Expr<bool>{closure.beckmann.expression()}};
}

}// namespace

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

[[nodiscard]] Float closure_sample_weight(
    const SurfaceClosureRecord &closure) noexcept {
    return closure.sample_weight;
}

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureRecord &closure) noexcept {
    return closure_allocated(closure_identity(closure));
}

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureIdentityExpression &closure) noexcept {
    return !has_kind(closure, SurfaceClosureKind::none) &
           (closure.allocation_weight >=
               cycles_closure::closure_weight_cutoff);
}

[[nodiscard]] Float bump_shadowing_term(const SurfacePoint &point,
    Float3 smooth_normal,
    const SurfaceClosureRecord &closure,
    Float3 direction,
    Bool is_evaluation) noexcept {
    // Cycles classifies Sheen, Lambert/Oren-Nayar, and Translucent in the
    // contiguous diffuse closure range. Express that semantic class here,
    // independently of the current host-side operation representation.
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto diffuse_closure =
        has_kind(closure, SurfaceClosureKind::diffuse) |
        has_kind(closure, SurfaceClosureKind::translucent) |
        (is_principled &
            has_lobe(closure, SurfaceClosureLobe::sheen));

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
                        (is_evaluation | diffuse_closure);

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
        diffuse_closure & point.use_bump_map_correction &
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

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept {
    return cycles_runtime_flags(
        closure_identity(closure),
        std::move(glossy_filter_roughness));
}

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureIdentityExpression &closure,
    Float glossy_filter_roughness) noexcept {
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto is_refraction = has_kind(
        closure, SurfaceClosureKind::refraction);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
    const auto is_bssrdf = has_kind(
        closure, SurfaceClosureKind::bssrdf);

    const auto bsdf = cycles_closure::runtime_bsdf;
    const auto has_eval =
        cycles_closure::runtime_bsdf_has_eval;
    UInt flags = 0u;
    flags = select(flags, bsdf | has_eval, is_diffuse);
    flags = select(flags,
        bsdf | has_eval |
            cycles_closure::runtime_bsdf_has_transmission,
        is_translucent);
    flags = select(flags, bsdf | has_eval, is_sheen);

    auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
    alpha *= alpha;
    alpha = max(alpha, glossy_filter_roughness);
    const auto regular_microfacet =
        alpha * alpha >
        cycles_closure::microfacet_singular_alpha_product;
    const auto microfacet_flags =
        bsdf | select(0u, has_eval, regular_microfacet);
    flags = select(flags,
        microfacet_flags,
        (is_principled & !is_sheen) | is_glossy);
    flags = select(flags,
        microfacet_flags |
            cycles_closure::runtime_bsdf_has_transmission,
        is_glass);
    flags = select(flags,
        microfacet_flags |
            cycles_closure::runtime_bsdf_has_transmission,
        is_refraction);
    flags = select(flags,
        bsdf | cycles_closure::runtime_transparent,
        is_transparent);
    flags = select(flags,
        UInt{cycles_closure::runtime_bssrdf},
        is_bssrdf);
    flags |= select(0u,
        has_eval,
        glossy_filter_roughness * glossy_filter_roughness >
            cycles_closure::microfacet_singular_alpha_product);
    return select(0u,
        flags,
        closure_allocated(closure) & closure.setup_valid);
}

[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosureRecord &closure) noexcept {
    return cycles_closure_type(closure_identity(closure));
}

[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosureIdentityExpression &closure) noexcept {
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto is_refraction = has_kind(
        closure, SurfaceClosureKind::refraction);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
    const auto is_bssrdf = has_kind(
        closure, SurfaceClosureKind::bssrdf);

    UInt type = cycles_closure::type_none;
    type = select(type,
        select(
            UInt{cycles_closure::type_oren_nayar},
            UInt{cycles_closure::type_diffuse},
            closure.roughness < 1.0e-5f),
        is_diffuse);
    type = select(type,
        UInt{cycles_closure::type_translucent},
        is_translucent);
    type = select(type,
        UInt{cycles_closure::type_microfacet_ggx},
        is_glossy | (is_principled & !is_sheen));
    const auto single_glass = select(
        UInt{cycles_closure::type_microfacet_ggx_glass},
        UInt{cycles_closure::type_microfacet_beckmann_glass},
        closure.beckmann);
    const auto glass = select(single_glass,
        UInt{cycles_closure::type_microfacet_multi_ggx_glass},
        closure.preserve_ggx_energy);
    type = select(type, glass, is_glass);
    const auto refraction = select(
        UInt{cycles_closure::type_microfacet_ggx_refraction},
        UInt{cycles_closure::type_microfacet_beckmann_refraction},
        closure.beckmann);
    type = select(type, refraction, is_refraction);
    type = select(type,
        UInt{cycles_closure::type_transparent},
        is_transparent);
    type = select(type,
        UInt{cycles_closure::type_sheen},
        is_sheen);
    auto bssrdf_type = UInt{cycles_closure::type_bssrdf_random_walk};
    bssrdf_type = select(
        bssrdf_type,
        UInt{cycles_closure::type_bssrdf_burley},
        closure.bssrdf_method == static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::burley));
    bssrdf_type = select(
        bssrdf_type,
        UInt{cycles_closure::type_bssrdf_random_walk_legacy},
        closure.bssrdf_method == static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk_legacy));
    bssrdf_type = select(
        bssrdf_type,
        UInt{cycles_closure::type_bssrdf_random_walk_skin},
        closure.bssrdf_method == static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk_skin));
    type = select(type, bssrdf_type, is_bssrdf);
    return select(UInt{cycles_closure::type_none},
        type,
        closure.setup_valid);
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

[[nodiscard]] Float3 diffuse_intensity(
    const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    auto nl = max(dot(closure.normal, outgoing), 0.0f);
    auto lambert = make_float3(nl * inverse_pi);

    auto sigma = clamp(closure.roughness, 0.0f, 1.0f);
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
    // Keep the Cycles form: at normal incidence `1 + (alpha2 - 1)`
    // catastrophically cancels when alpha2 is below float epsilon, while
    // `(1 - cos2) + alpha2 * cos2` preserves the finite GGX peak.
    auto cosine2 = min(n_dot_h * n_dot_h, 1.0f);
    auto denominator = (1.0f - cosine2) + alpha2 * cosine2;
    return alpha2 / max(pi * denominator * denominator, 1.0e-20f);
}

[[nodiscard]] Float microfacet_alpha(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept {
    // Cycles applies bsdf_microfacet_blur after closure setup. Keep the
    // original closure roughness for sample weights, layering, and
    // energy compensation; only evaluation and sampling see this
    // widened alpha.
    auto setup_alpha = clamp(closure.roughness, 0.0f, 1.0f);
    setup_alpha *= setup_alpha;
    return max(setup_alpha, glossy_filter_roughness);
}

[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept {
    const auto alpha = microfacet_alpha(
        closure, glossy_filter_roughness);
    return alpha * alpha <=
           cycles_closure::microfacet_singular_alpha_product;
}

[[nodiscard]] Float smith_g1(Float n_dot_v, Float alpha) noexcept {
    auto cosine = max(n_dot_v, 1.0e-6f);
    auto tangent2 = max(1.0f / (cosine * cosine) - 1.0f, 0.0f);
    auto lambda = 0.5f * (sqrt(1.0f + alpha * alpha * tangent2) - 1.0f);
    return 1.0f / (1.0f + lambda);
}

[[nodiscard]] Float3 specular_f0(
    const SurfaceClosureRecord &closure) noexcept {
    auto dielectric =
        (closure.ior - 1.0f) / max(closure.ior + 1.0f, 1.0e-20f);
    dielectric *= dielectric;
    return lerp(make_float3(dielectric),
        clamp(closure.color, make_float3(0.0f), make_float3(1.0f)),
        closure.metallic);
}

[[nodiscard]] Float3 microfacet_reflection_fresnel(
    const SurfaceClosureRecord &closure,
    Float cosine) noexcept {
    const auto f0 = specular_f0(closure);
    const auto generic =
        f0 + (make_float3(1.0f) - f0) *
                 pow(1.0f - cosine, 5.0f);
    const auto metallic =
        fresnel_f82(
            cosine, closure.color, closure.specular_tint) *
        closure.evaluation_scale;
    const auto dielectric =
        generalized_dielectric_fresnel(
            cosine, closure.ior, closure.color) *
        closure.evaluation_scale;
    const auto principled = select(
        dielectric,
        metallic,
        has_lobe(closure, SurfaceClosureLobe::metallic));
    return select(generic,
        principled,
        has_kind(closure, SurfaceClosureKind::principled));
}

[[nodiscard]] Float3 microfacet_intensity(
    const ShaderServices &services,
    const SurfaceClosureRecord &closure,
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
    const auto singular = microfacet_is_singular(
        closure, glossy_filter_roughness);
    auto alpha = max(
        microfacet_alpha(closure, glossy_filter_roughness), 1.0e-10f);
    auto distribution = ggx_distribution(n_dot_h, alpha);
    auto lambda_v = 1.0f / smith_g1(n_dot_v, alpha) - 1.0f;
    auto lambda_l = 1.0f / smith_g1(n_dot_l, alpha) - 1.0f;
    auto geometry = 1.0f / (1.0f + lambda_v + lambda_l);
    const auto fresnel = microfacet_reflection_fresnel(
        closure, v_dot_h);
    auto intensity = fresnel * distribution * geometry /
                     max(4.0f * n_dot_v, 1.0e-20f);
    return select(make_float3(0.0f),
        intensity,
        (n_dot_v > 0.0f) & (n_dot_l > 0.0f) & (n_dot_h > 0.0f) &
            (v_dot_h > 0.0f) & !singular);
}

[[nodiscard]] Float microfacet_pdf(
    const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    auto half_vector =
        safe_normalize(incoming + outgoing, glossy_normal);
    auto n_dot_h = max(dot(glossy_normal, half_vector), 0.0f);
    auto v_dot_h = max(dot(incoming, half_vector), 0.0f);
    const auto singular = microfacet_is_singular(
        closure, glossy_filter_roughness);
    auto alpha = max(
        microfacet_alpha(closure, glossy_filter_roughness), 1.0e-10f);
    auto n_dot_v = max(dot(glossy_normal, incoming), 0.0f);
    auto n_dot_l = max(dot(glossy_normal, outgoing), 0.0f);
    auto pdf = ggx_distribution(n_dot_h, alpha) *
               smith_g1(n_dot_v, alpha) / max(4.0f * n_dot_v, 1.0e-20f);
    return select(0.0f,
        pdf,
        (n_dot_v > 0.0f) & (n_dot_l > 0.0f) & (n_dot_h > 0.0f) &
            (v_dot_h > 0.0f) & !singular);
}

[[nodiscard]] MicrofacetReflectionSample sample_microfacet_reflection(
    const SurfacePoint &point,
    Float3 smooth_normal,
    const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept {
    const auto alpha = microfacet_alpha(
        closure, glossy_filter_roughness);
    const auto regular =
        cycles_sample_mapping::sample_ggx_visible_normal_reflection(
        glossy_normal,
        incoming,
        max(alpha, 1.0e-10f),
        random);
    const auto singular_direction =
        2.0f * dot(glossy_normal, incoming) * glossy_normal - incoming;
    const auto singular = alpha * alpha <=
                          cycles_closure::microfacet_singular_alpha_product;
    const auto direction = select(regular, singular_direction, singular);
    const auto fresnel = microfacet_reflection_fresnel(
        closure, max(dot(glossy_normal, incoming), 0.0f));
    const auto bump_shadowing = bump_shadowing_term(
        point, smooth_normal, closure, direction, false);
    const auto valid = (dot(glossy_normal, incoming) > 0.0f) &
                       (dot(glossy_normal, direction) > 0.0f) &
                       (dot(point.geometric_normal, direction) > 0.0f) &
                       (sample_weight(fresnel) > 0.0f);
    return {.direction = direction,
        .singular_evaluation =
            closure.weight * fresnel * bump_shadowing * 1.0e6f,
        .singular_pdf = 1.0e6f,
        .alpha = alpha,
        .singular = singular,
        .valid = valid};
}

[[nodiscard]] Float sheen_intensity(
    const SurfaceClosureRecord &closure,
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

[[nodiscard]] Float3 sample_sheen(
    const SurfaceClosureRecord &closure,
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
