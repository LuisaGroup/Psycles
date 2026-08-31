#include "graph_surface_internal.h"
#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {
[[nodiscard]] Float fresnel_dielectric_cos(
    Float cosine, Float eta) noexcept {
    auto c = abs(cosine);
    auto g_squared = eta * eta - 1.0f + c * c;
    Float result = 1.0f;
    $if(g_squared > 0.0f) {
        auto g = sqrt(g_squared);
        auto a = (g - c) / (g + c);
        auto b = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
        result = 0.5f * a * a * (1.0f + b * b);
    };
    return result;
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
    Float ior, Float specular_ior_level) noexcept {
    auto original_eta = max(ior, 1.0e-5f);
    auto original_f0 = f0_from_ior(original_eta);
    auto adjusted_f0 =
        original_f0 * (2.0f * max(specular_ior_level, 0.0f));
    auto eta_from_adjusted = ior_from_f0(adjusted_f0);
    eta_from_adjusted = select(eta_from_adjusted,
        1.0f / max(eta_from_adjusted, 1.0e-20f),
        original_eta < 1.0f);
    auto should_adjust = specular_ior_level != 0.5f;
    return {
        .eta = select(original_eta, eta_from_adjusted, should_adjust),
        .f0 = select(original_f0, adjusted_f0, should_adjust)};
}

[[nodiscard]] AdjustedIor adjusted_ior(
    const TracedClosure &closure) noexcept {
    return adjusted_ior(closure.ior, closure.specular_ior_level);
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

[[nodiscard]] Float3 fresnel_conductor(
    Float cosine, Float3 ior, Float3 extinction) noexcept {
    // Exact unpolarized Fresnel at an air/conductor interface. This is the
    // vector form of the two polarized rational equations used by Cycles;
    // n and k are already canonical non-negative physical parameters.
    const auto c = clamp(cosine, 0.0f, 1.0f);
    const auto n2 = ior * ior;
    const auto k2 = extinction * extinction;
    const auto two_nk = 2.0f * ior * extinction;
    const auto t1 = n2 - k2 - (1.0f - c * c);
    const auto t2 = sqrt(max(t1 * t1 + two_nk * two_nk,
                             make_float3(0.0f)));
    const auto u2 = max(0.5f * (t2 + t1), make_float3(0.0f));
    const auto v2 = max(0.5f * (t2 - t1), make_float3(0.0f));
    const auto u = sqrt(u2);
    const auto v = sqrt(v2);
    const auto rs_numerator =
        (c - u) * (c - u) + v2;
    const auto rs_denominator =
        (c + u) * (c + u) + v2;
    const auto t3 = (n2 - k2) * c;
    const auto t4 = two_nk * c;
    const auto rp_numerator =
        (t3 - u) * (t3 - u) + (t4 - v) * (t4 - v);
    const auto rp_denominator =
        (t3 + u) * (t3 + u) + (t4 + v) * (t4 + v);
    const auto rs = rs_numerator /
                    max(rs_denominator, make_float3(1.0e-20f));
    const auto rp = rp_numerator /
                    max(rp_denominator, make_float3(1.0e-20f));
    return clamp(0.5f * (rs + rp),
                 make_float3(0.0f), make_float3(1.0f));
}

[[nodiscard]] Float3 fresnel_conductor_fss(
    Float3 ior, Float3 extinction) noexcept {
    // Cycles fits F82 to the physical conductor at normal incidence and at
    // cos(theta)=1/7, then integrates that closed form. The fit is a pure
    // projection of the exact Fresnel function, not an authored tint model.
    constexpr float f = 6.0f / 7.0f;
    constexpr float f5 = f * f * f * f * f;
    const auto f0 = fresnel_conductor(1.0f, ior, extinction);
    const auto f82 = fresnel_conductor(1.0f / 7.0f, ior, extinction);
    const auto b = (7.0f / (f5 * f)) *
                   (lerp(f0, make_float3(1.0f), f5) - f82);
    return clamp(
        lerp(f0, make_float3(1.0f), 1.0f / 21.0f) - b * (1.0f / 126.0f),
        make_float3(0.0f), make_float3(1.0f));
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
    const SurfaceClosurePoint &point,
    Float3 incoming,
    Float3 shading_normal) noexcept {
    const auto correction_enabled =
        point.use_bump_map_correction &
        !point.is_curve &
        !all(point.geometric_normal == shading_normal);
    return select(shading_normal,
        ensure_valid_specular_reflection(
            point.geometric_normal, incoming, shading_normal),
        correction_enabled);
}

[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    Float roughness,
    bool preserve_ggx_energy,
    Float incoming_cosine,
    Float3 fss) noexcept {
    if (!preserve_ggx_energy) {
        return {.darkening = make_float3(1.0f),
            .energy_scale = make_float3(1.0f)};
    }

    roughness = clamp(roughness, 0.0f, 1.0f);
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

[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    const TracedClosure &closure,
    Float incoming_cosine,
    Float3 fss) noexcept {
    return ggx_energy(services,
        closure.roughness,
        closure.preserve_ggx_energy,
        incoming_cosine,
        fss);
}

[[nodiscard]] Float closure_sample_weight(
    const SurfaceClosurePhysicalCommonRecord &closure) noexcept {
    return closure.sample_weight;
}

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureRecord &closure) noexcept {
    // Allocation is a prefix property, independent of whether setup later
    // canonicalized the slot to ClosureType::NONE. This is Cycles'
    // num_closure topology: failed setup consumes an allocated slot.
    return closure.allocation_weight >=
           cycles_closure::closure_weight_cutoff;
}

[[nodiscard]] Float bump_shadowing_term(
    const SurfaceClosurePoint &point,
    Float3 smooth_normal,
    Float3 closure_normal,
    Bool diffuse_closure,
    Float3 direction,
    Bool is_evaluation) noexcept {
    // The family eliminator supplies Cycles' diffuse-type classification.
    // Recomputing it here from a universal record would reconstruct the tag
    // product after the dominating family dispatch.
    const auto normals_equal = all(closure_normal == smooth_normal);
    const auto cosine_smooth_direction = dot(smooth_normal, direction);
    const auto cosine_smooth_closure =
        dot(smooth_normal, closure_normal);
    const auto cosine_closure_direction =
        dot(closure_normal, direction);

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
    // exactly equal, then independently returns one for every curve. The
    // disjunction is observationally identical to those ordered returns even
    // at degenerate dot products because both branches have the same value.
    return select(result, 1.0f, normals_equal | point.is_curve);
}

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness,
    SurfaceClosureReachability reachability) noexcept {
    return select(
        UInt{0u},
        cycles_runtime_flags(
            closure.closure_type,
            closure.roughness,
            std::move(glossy_filter_roughness),
            reachability),
        closure_allocated(closure));
}

[[nodiscard]] UInt cycles_runtime_flags(
    UInt closure_type,
    Float roughness,
    Float glossy_filter_roughness,
    SurfaceClosureReachability reachability) noexcept {
    const auto bsdf = cycles_closure::runtime_bsdf;
    const auto has_eval = cycles_closure::runtime_bsdf_has_eval;
    const auto transmission =
        cycles_closure::runtime_bsdf_has_transmission;
    UInt flags = 0u;
    const auto select_type_flags =
        [&](Bool predicate, std::uint32_t value) noexcept {
            flags = select(flags, UInt{value}, predicate);
        };
    if (reachability.contains(SurfaceClosureKind::diffuse)) {
        select_type_flags(
            cycles_closure::is_diffuse_or_oren_nayar(closure_type),
            bsdf | has_eval);
    }
    if (reachability.contains(SurfaceClosureKind::translucent)) {
        select_type_flags(
            closure_type == cycles_closure::type_translucent,
            bsdf | has_eval | transmission);
    }
    if (reachability.contains(SurfaceClosureKind::rough_translucent)) {
        select_type_flags(
            closure_type == cycles_closure::type_rough_translucent,
            bsdf | has_eval | transmission);
    }
    if (reachability.contains_principled_lobe(
            SurfaceClosureLobe::sheen) ||
        reachability.contains(SurfaceClosureKind::sheen_microfiber)) {
        select_type_flags(
            closure_type == cycles_closure::type_sheen,
            bsdf | has_eval);
    }
    if (reachability.contains(SurfaceClosureKind::sheen_ashikhmin)) {
        select_type_flags(
            closure_type == cycles_closure::type_ashikhmin_velvet,
            bsdf | has_eval);
    }
    if (reachability.contains(SurfaceClosureKind::hair_reflection)) {
        select_type_flags(
            closure_type == cycles_closure::type_hair_reflection,
            bsdf | has_eval);
    }
    if (reachability.contains(SurfaceClosureKind::hair_transmission)) {
        select_type_flags(
            closure_type == cycles_closure::type_hair_transmission,
            bsdf | has_eval | transmission);
    }

    auto alpha = clamp(roughness, 0.0f, 1.0f);
    alpha *= alpha;
    alpha = max(alpha, glossy_filter_roughness);
    const auto regular_microfacet =
        alpha * alpha >
        cycles_closure::microfacet_singular_alpha_product;
    const auto microfacet_flags =
        bsdf | select(0u, has_eval, regular_microfacet);
    constexpr auto sheen_lobe_bit =
        surface_closure_lobe_bit(SurfaceClosureLobe::sheen);
    const auto reflection_microfacet_reachable =
        (reachability.contains(SurfaceClosureKind::principled) &&
         (reachability.principled_lobes & ~sheen_lobe_bit) != 0u) ||
        reachability.contains(SurfaceClosureKind::glossy) ||
        reachability.contains(SurfaceClosureKind::metallic_f82) ||
        reachability.contains(SurfaceClosureKind::metallic_conductor);
    if (reflection_microfacet_reachable) {
        flags = select(
            flags,
            microfacet_flags,
            cycles_closure::is_reflection_microfacet(closure_type));
    }
    if (reachability.contains(SurfaceClosureKind::glass)) {
        flags = select(
            flags,
            microfacet_flags | transmission,
            cycles_closure::is_glass_microfacet(closure_type));
    }
    if (reachability.contains(SurfaceClosureKind::refraction)) {
        flags = select(
            flags,
            microfacet_flags | transmission,
            cycles_closure::is_refraction_microfacet(closure_type));
    }
    if (reachability.contains(
            SurfaceClosureKind::thin_glass_transmission)) {
        flags = select(
            flags,
            UInt{bsdf} | select(0u, has_eval, regular_microfacet) |
                transmission,
            closure_type ==
                cycles_closure::type_thin_glass_transmission);
    }
    if (reachability.contains(SurfaceClosureKind::transparent)) {
        select_type_flags(
            closure_type == cycles_closure::type_transparent,
            bsdf | cycles_closure::runtime_transparent);
    }
    if (reachability.contains(SurfaceClosureKind::bssrdf)) {
        select_type_flags(
            cycles_closure::is_bssrdf(closure_type),
            cycles_closure::runtime_bssrdf);
    }
    flags |= select(
        0u,
        has_eval,
        glossy_filter_roughness * glossy_filter_roughness >
            cycles_closure::microfacet_singular_alpha_product);
    return select(
        UInt{0u},
        flags,
        closure_type != cycles_closure::type_none);
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
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float3 color,
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
    auto albedo = clamp(color, make_float3(0.0f), make_float3(1.0f));
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

namespace {

[[nodiscard]] Float microfacet_ggx_distribution(
    Float n_dot_h, Float alpha) noexcept {
    const auto alpha2 = alpha * alpha;
    // Keep the Cycles form: at normal incidence `1 + (alpha2 - 1)`
    // catastrophically cancels when alpha2 is below float epsilon, while
    // `(1 - cos2) + alpha2 * cos2` preserves the finite GGX peak.
    const auto cosine2 = min(n_dot_h * n_dot_h, 1.0f);
    const auto one_minus_cosine2 = 1.0f - cosine2;
    const auto ggx_denominator =
        one_minus_cosine2 + alpha2 * cosine2;
    return alpha2 /
           max(pi * ggx_denominator * ggx_denominator, 1.0e-20f);
}

[[nodiscard]] Float microfacet_beckmann_distribution(
    Float n_dot_h, Float alpha) noexcept {
    const auto alpha2 = alpha * alpha;
    const auto cosine2 = min(n_dot_h * n_dot_h, 1.0f);
    const auto one_minus_cosine2 = 1.0f - cosine2;
    const auto beckmann_exponent =
        one_minus_cosine2 / (cosine2 * alpha2);
    const auto beckmann_denominator =
        exp(beckmann_exponent) * pi * alpha2 * cosine2 * cosine2;
    return 1.0f / beckmann_denominator;
}

[[nodiscard]] Float microfacet_ggx_lambda(
    Float n_dot_v, Float alpha) noexcept {
    const auto cosine2 = n_dot_v * n_dot_v;
    const auto squared_alpha_tangent =
        alpha * alpha * max(1.0f / cosine2 - 1.0f, 0.0f);
    return 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
}

[[nodiscard]] Float microfacet_ggx_anisotropic_distribution(
    Float3 local_half, Float alpha_x, Float alpha_y) noexcept {
    const auto scaled_half = local_half /
                             make_float3(alpha_x, alpha_y, 1.0f);
    const auto alpha_product = alpha_x * alpha_y;
    const auto length_squared = dot(scaled_half, scaled_half);
    return inverse_pi /
           (alpha_product * length_squared * length_squared);
}

[[nodiscard]] Float microfacet_beckmann_anisotropic_distribution(
    Float3 local_half, Float alpha_x, Float alpha_y) noexcept {
    const auto scaled_half = local_half /
                             make_float3(alpha_x, alpha_y, 1.0f);
    const auto cosine_squared = scaled_half.z * scaled_half.z;
    const auto exponent = -(scaled_half.x * scaled_half.x +
                            scaled_half.y * scaled_half.y) /
                          cosine_squared;
    return exp(exponent) /
           (pi * alpha_x * alpha_y * cosine_squared * cosine_squared);
}

[[nodiscard]] Float microfacet_ggx_anisotropic_lambda(
    Float3 local_direction, Float alpha_x, Float alpha_y) noexcept {
    const auto scaled_x = alpha_x * local_direction.x;
    const auto scaled_y = alpha_y * local_direction.y;
    const auto squared_alpha_tangent =
        (scaled_x * scaled_x + scaled_y * scaled_y) /
        (local_direction.z * local_direction.z);
    return 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
}

[[nodiscard]] Float microfacet_beckmann_anisotropic_lambda(
    Float3 local_direction, Float alpha_x, Float alpha_y) noexcept {
    const auto scaled_x = alpha_x * local_direction.x;
    const auto scaled_y = alpha_y * local_direction.y;
    const auto squared_alpha_tangent =
        (scaled_x * scaled_x + scaled_y * scaled_y) /
        (local_direction.z * local_direction.z);
    const auto a = rsqrt(squared_alpha_tangent);
    const auto approximation =
        ((0.396f * a - 1.259f) * a + 1.0f) /
        ((2.181f * a + 3.535f) * a);
    return select(
        approximation, 0.0f, squared_alpha_tangent < 0.39f);
}

[[nodiscard]] Float microfacet_beckmann_lambda(
    Float n_dot_v, Float alpha) noexcept {
    const auto cosine2 = n_dot_v * n_dot_v;
    const auto squared_alpha_tangent =
        alpha * alpha * max(1.0f / cosine2 - 1.0f, 0.0f);
    const auto a = rsqrt(squared_alpha_tangent);
    const auto approximation =
        ((0.396f * a - 1.259f) * a + 1.0f) /
        ((2.181f * a + 3.535f) * a);
    return select(
        approximation, 0.0f, squared_alpha_tangent < 0.39f);
}

}// namespace

[[nodiscard]] MicrofacetDistributionTerms microfacet_distribution_terms(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float n_dot_h,
    Float n_dot_incoming,
    Float n_dot_outgoing,
    Float alpha) noexcept {
    Float distribution = 0.0f;
    Float lambda_incoming = 0.0f;
    Float lambda_outgoing = 0.0f;
    $if(cycles_closure::is_beckmann_microfacet(
        closure.closure_type)) {
        distribution = microfacet_beckmann_distribution(
            n_dot_h, alpha);
        lambda_incoming = microfacet_beckmann_lambda(
            n_dot_incoming, alpha);
        lambda_outgoing = microfacet_beckmann_lambda(
            n_dot_outgoing, alpha);
    }
    $else {
        distribution = microfacet_ggx_distribution(n_dot_h, alpha);
        lambda_incoming = microfacet_ggx_lambda(
            n_dot_incoming, alpha);
        lambda_outgoing = microfacet_ggx_lambda(
            n_dot_outgoing, alpha);
    };
    return {
        .distribution = distribution,
        .lambda_incoming = lambda_incoming,
        .lambda_outgoing = lambda_outgoing};
}

[[nodiscard]] Float microfacet_alpha(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept {
    // Cycles applies bsdf_microfacet_blur after closure setup. Keep the
    // original closure roughness for sample weights, layering, and
    // energy compensation; only evaluation and sampling see this
    // widened alpha.
    auto setup_alpha = clamp(closure.roughness, 0.0f, 1.0f);
    setup_alpha *= setup_alpha;
    return max(setup_alpha, glossy_filter_roughness);
}

[[nodiscard]] Float2 microfacet_alpha(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float glossy_filter_roughness) noexcept {
    return max(
        make_float2(
            closure.payload.microfacet_alpha_x,
            closure.payload.microfacet_alpha_y),
        make_float2(glossy_filter_roughness));
}

[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept {
    const auto alpha = microfacet_alpha(
        closure, glossy_filter_roughness);
    return alpha * alpha <=
           cycles_closure::microfacet_singular_alpha_product;
}

[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float glossy_filter_roughness) noexcept {
    const auto alpha = microfacet_alpha(
        closure, glossy_filter_roughness);
    return alpha.x * alpha.y <=
           cycles_closure::microfacet_singular_alpha_product;
}

[[nodiscard]] Float microfacet_specular_roughness_squared(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept {
    const auto alpha = microfacet_alpha(
        closure, glossy_filter_roughness);
    return select(
        alpha * alpha,
        0.0f,
        microfacet_is_singular(
            closure, glossy_filter_roughness));
}

[[nodiscard]] MicrofacetDistributionTerms
microfacet_reflection_distribution_terms(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 half_vector,
    Float3 incoming,
    Float3 outgoing,
    Float3 normal,
    Float2 alpha) noexcept {
    const auto n_dot_h = max(dot(normal, half_vector), 0.0f);
    const auto n_dot_incoming = max(dot(normal, incoming), 0.0f);
    const auto n_dot_outgoing = max(dot(normal, outgoing), 0.0f);
    Float distribution = 0.0f;
    Float lambda_incoming = 0.0f;
    Float lambda_outgoing = 0.0f;
    $if(alpha.x == alpha.y) {
        const auto terms = microfacet_distribution_terms(
            closure.common,
            n_dot_h,
            n_dot_incoming,
            n_dot_outgoing,
            alpha.x);
        distribution = terms.distribution;
        lambda_incoming = terms.lambda_incoming;
        lambda_outgoing = terms.lambda_outgoing;
    }
    $else {
        const auto basis =
            cycles_sample_mapping::make_orthonormals_tangent(
                normal, closure.payload.microfacet_tangent);
        const auto local_half = make_float3(
            dot(basis.tangent, half_vector),
            dot(basis.bitangent, half_vector),
            n_dot_h);
        const auto local_incoming = make_float3(
            dot(basis.tangent, incoming),
            dot(basis.bitangent, incoming),
            n_dot_incoming);
        const auto local_outgoing = make_float3(
            dot(basis.tangent, outgoing),
            dot(basis.bitangent, outgoing),
            n_dot_outgoing);
        $if(cycles_closure::is_beckmann_microfacet(
            closure.common.closure_type)) {
            distribution = microfacet_beckmann_anisotropic_distribution(
                local_half, alpha.x, alpha.y);
            lambda_incoming = microfacet_beckmann_anisotropic_lambda(
                local_incoming, alpha.x, alpha.y);
            lambda_outgoing = microfacet_beckmann_anisotropic_lambda(
                local_outgoing, alpha.x, alpha.y);
        }
        $else {
            distribution = microfacet_ggx_anisotropic_distribution(
                local_half, alpha.x, alpha.y);
            lambda_incoming = microfacet_ggx_anisotropic_lambda(
                local_incoming, alpha.x, alpha.y);
            lambda_outgoing = microfacet_ggx_anisotropic_lambda(
                local_outgoing, alpha.x, alpha.y);
        };
    };
    return {
        .distribution = distribution,
        .lambda_incoming = lambda_incoming,
        .lambda_outgoing = lambda_outgoing};
}

[[nodiscard]] Float3 microfacet_reflection_fresnel(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float cosine,
    const ShaderServices *services,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept {
    using Fresnel = cycles_closure::MicrofacetFresnel;
    const UInt fresnel_type{closure.common.microfacet_fresnel};
    // NONE is Cycles' ordinary Glossy behavior: color already lives in
    // ShaderClosure::weight and the remaining directional scale is constant.
    Float3 result = closure.payload.evaluation_scale;
    if (may_have_f82) {
        auto f82 =
            fresnel_f82(cosine,
                        closure.common.color_or_evaluation_scale,
                        closure.payload.specular_tint) *
            closure.payload.evaluation_scale;
        if (may_have_f82_thin_film) {
            LUISA_ASSERT(services != nullptr,
                         "Thin-film Fresnel requires Cycles table services.");
            const auto film = thin_film_f82_fresnel(
                *services,
                closure.payload.thin_film_thickness,
                closure.payload.thin_film_ior,
                closure.common.color_or_evaluation_scale,
                closure.payload.specular_tint,
                cosine) * closure.payload.evaluation_scale;
            f82 = select(
                f82,
                film,
                closure.payload.thin_film_thickness >
                    thin_film_thickness_cutoff);
        }
        result = select(
            result,
            f82,
            fresnel_type ==
                static_cast<std::uint32_t>(Fresnel::f82_tint));
    }
    if (may_have_dielectric) {
        const auto dielectric =
            make_float3(fresnel_dielectric_cos(
                cosine, closure.payload.ior)) *
            closure.payload.evaluation_scale;
        result = select(
            result,
            dielectric,
            fresnel_type == static_cast<std::uint32_t>(
                                Fresnel::dielectric));
    }
    if (may_have_generalized_schlick) {
        auto generalized = generalized_dielectric_fresnel(
                               cosine,
                               closure.payload.ior,
                               closure.common.color_or_evaluation_scale) *
            closure.payload.evaluation_scale;
        if (may_have_generalized_schlick_thin_film) {
            LUISA_ASSERT(services != nullptr,
                         "Thin-film Fresnel requires Cycles table services.");
            const auto film = thin_film_dielectric_fresnel(
                *services,
                closure.payload.thin_film_thickness,
                closure.payload.thin_film_ior,
                closure.payload.ior,
                closure.common.color_or_evaluation_scale,
                cosine).reflectance * closure.payload.evaluation_scale;
            generalized = select(
                generalized,
                film,
                closure.payload.thin_film_thickness >
                    thin_film_thickness_cutoff);
        }
        result = select(
            result,
            generalized,
            fresnel_type == static_cast<std::uint32_t>(
                                Fresnel::generalized_schlick));
    }
    if (may_have_conductor) {
        auto conductor = fresnel_conductor(
                             cosine,
                             closure.common.color_or_evaluation_scale,
                             closure.payload.specular_tint) *
                         closure.payload.evaluation_scale;
        if (may_have_conductor_thin_film) {
            LUISA_ASSERT(services != nullptr,
                         "Thin-film Fresnel requires Cycles table services.");
            const auto film = thin_film_conductor_fresnel(
                *services,
                closure.payload.thin_film_thickness,
                closure.payload.thin_film_ior,
                closure.common.color_or_evaluation_scale,
                closure.payload.specular_tint,
                cosine) * closure.payload.evaluation_scale;
            conductor = select(
                conductor,
                film,
                closure.payload.thin_film_thickness >
                    thin_film_thickness_cutoff);
        }
        result = select(
            result,
            conductor,
            fresnel_type ==
                static_cast<std::uint32_t>(Fresnel::conductor));
    }
    return result;
}

[[nodiscard]] MicrofacetEvaluation microfacet_evaluate(
    const ShaderServices &services,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness,
    bool may_be_anisotropic,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept {
    Float2 setup_alpha;
    if (may_be_anisotropic) {
        setup_alpha = microfacet_alpha(
            closure, glossy_filter_roughness);
    } else {
        setup_alpha = make_float2(microfacet_alpha(
            closure.common, glossy_filter_roughness));
    }
    const auto alpha_product = setup_alpha.x * setup_alpha.y;
    const auto singular =
        alpha_product <=
        cycles_closure::microfacet_singular_alpha_product;
    MicrofacetEvaluation result{
        .intensity = make_float3(0.0f),
        .pdf = 0.0f,
        .roughness_squared = select(alpha_product, 0.0f, singular)};
    // A delta closure has zero density in the finite-direction measure. The
    // complete regular evaluator is therefore unreachable in that domain,
    // matching Cycles' SD_BSDF_HAS_EVAL classification.
    $if(!singular) {
        const auto n_dot_v =
            max(dot(glossy_normal, incoming), 0.0f);
        const auto n_dot_l =
            max(dot(glossy_normal, outgoing), 0.0f);
        const auto half_vector =
            safe_normalize(incoming + outgoing, glossy_normal);
        const auto v_dot_h =
            max(dot(incoming, half_vector), 0.0f);
        MicrofacetDistributionTerms terms;
        if (may_be_anisotropic) {
            const auto alpha = max(
                setup_alpha,
                make_float2(1.0e-10f));
            terms = microfacet_reflection_distribution_terms(
                closure,
                half_vector,
                incoming,
                outgoing,
                glossy_normal,
                alpha);
        } else {
            const auto scalar_alpha =
                max(setup_alpha.x, 1.0e-10f);
            terms = microfacet_distribution_terms(
                closure.common,
                max(dot(glossy_normal, half_vector), 0.0f),
                n_dot_v,
                n_dot_l,
                scalar_alpha);
        }
        const auto geometry =
            1.0f / (1.0f + terms.lambda_incoming +
                    terms.lambda_outgoing);
        const auto fresnel = microfacet_reflection_fresnel(
            closure,
            v_dot_h,
            &services,
            may_have_f82,
            may_have_f82_thin_film,
            may_have_dielectric,
            may_have_generalized_schlick,
            may_have_generalized_schlick_thin_film,
            may_have_conductor,
            may_have_conductor_thin_film);
        const auto intensity =
            fresnel * terms.distribution * geometry /
            max(4.0f * n_dot_v, 1.0e-20f);
        const auto pdf =
            terms.distribution /
            max(4.0f * n_dot_v, 1.0e-20f) /
            (1.0f + terms.lambda_incoming);
        const auto valid =
            (n_dot_v > 0.0f) & (n_dot_l > 0.0f) &
            (dot(glossy_normal, half_vector) > 0.0f) &
            (v_dot_h > 0.0f);
        result.intensity = select(
            make_float3(0.0f), intensity, valid);
        result.pdf = select(0.0f, pdf, valid);
    };
    return result;
}

[[nodiscard]] MicrofacetReflectionSample sample_microfacet_reflection(
    const SurfaceClosurePoint &point,
    Float3 smooth_normal,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness,
    bool may_be_anisotropic,
    const ShaderServices *services,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept {
    Float2 alpha;
    if (may_be_anisotropic) {
        alpha = microfacet_alpha(
            closure, glossy_filter_roughness);
    } else {
        alpha = make_float2(microfacet_alpha(
            closure.common, glossy_filter_roughness));
    }
    const auto singular = alpha.x * alpha.y <=
                          cycles_closure::microfacet_singular_alpha_product;
    Float3 direction =
        2.0f * dot(glossy_normal, incoming) * glossy_normal - incoming;
    Float fresnel_cosine = max(dot(glossy_normal, incoming), 0.0f);
    // The delta and regular domains are disjoint by definition. In the delta
    // domain H=N, so evaluating either VNDF is dead work. Within the regular
    // domain the distribution tag selects exactly one VNDF; a value select
    // would eagerly evaluate both in the Luisa DSL.
    $if(!singular) {
        const auto sampling_alpha =
            max(alpha, make_float2(1.0e-10f));
        auto basis =
            cycles_sample_mapping::make_orthonormals(glossy_normal);
        if (may_be_anisotropic) {
            $if(sampling_alpha.x != sampling_alpha.y) {
                const auto anisotropic_basis =
                    cycles_sample_mapping::make_orthonormals_tangent(
                        glossy_normal,
                        closure.payload.microfacet_tangent);
                basis.tangent = anisotropic_basis.tangent;
                basis.bitangent = anisotropic_basis.bitangent;
            };
        }
        Float3 half_vector;
        $if(cycles_closure::is_beckmann_microfacet(
            closure.common.closure_type)) {
            half_vector =
                cycles_sample_mapping::sample_beckmann_visible_normal(
                    glossy_normal,
                    basis,
                    incoming,
                    sampling_alpha.x,
                    sampling_alpha.y,
                    random);
        }
        $else {
            half_vector =
                cycles_sample_mapping::sample_ggx_visible_normal(
                    glossy_normal,
                    basis,
                    incoming,
                    sampling_alpha.x,
                    sampling_alpha.y,
                    random);
        };
        direction =
            2.0f * dot(incoming, half_vector) * half_vector - incoming;
        // Cycles evaluates the sampled microfacet Fresnel at I.H. In the
        // singular domain H=N, so the initialized I.N value remains exact.
        fresnel_cosine = max(dot(half_vector, incoming), 0.0f);
    };
    const auto fresnel = microfacet_reflection_fresnel(
        closure,
        fresnel_cosine,
        services,
        may_have_f82,
        may_have_f82_thin_film,
        may_have_dielectric,
        may_have_generalized_schlick,
        may_have_generalized_schlick_thin_film,
        may_have_conductor,
        may_have_conductor_thin_film);
    const auto bump_shadowing = bump_shadowing_term(
        point,
        smooth_normal,
        closure.common.normal,
        false,
        direction,
        false);
    const auto valid = (dot(glossy_normal, incoming) > 0.0f) &
                       (dot(glossy_normal, direction) > 0.0f) &
                       (dot(point.geometric_normal, direction) > 0.0f) &
                       (sample_weight(fresnel) > 0.0f);
    return {.direction = direction,
        .singular_evaluation =
            closure.common.weight * fresnel * bump_shadowing * 1.0e6f,
        .singular_pdf = 1.0e6f,
        .roughness = alpha,
        .singular = singular,
        .valid = valid};
}

[[nodiscard]] Float sheen_intensity(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    const auto basis =
        cycles_sample_mapping::make_orthonormals_safe_tangent(
            closure.common.normal, incoming);
    const auto local_outgoing = make_float3(
        dot(outgoing, basis.tangent),
        dot(outgoing, basis.bitangent),
        dot(outgoing, closure.common.normal));
    const auto a = closure.payload.sheen_transform_a;
    const auto b = closure.payload.sheen_transform_b;
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
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float2 random) noexcept {
    const auto basis =
        cycles_sample_mapping::make_orthonormals_safe_tangent(
            closure.common.normal, incoming);
    const auto disk =
        cycles_sample_mapping::sample_uniform_disk(random);
    const auto disk_z = sqrt(max(1.0f - dot(disk, disk), 0.0f));
    const auto local_outgoing = normalize(make_float3(
        disk.x - disk_z * closure.payload.sheen_transform_b,
        disk.y,
        disk_z * closure.payload.sheen_transform_a));
    return basis.tangent * local_outgoing.x +
           basis.bitangent * local_outgoing.y +
           closure.common.normal * local_outgoing.z;
}

[[nodiscard]] AshikhminVelvetEvaluation evaluate_ashikhmin_velvet(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    bool sampling_domain) noexcept {
    const auto normal = closure.normal;
    const auto cos_ni = dot(normal, incoming);
    const auto cos_no = dot(normal, outgoing);
    const auto half_sum = incoming + outgoing;
    const auto half_length = sqrt(dot(half_sum, half_sum));
    // Totalize normalization and all later divisions outside the accepted
    // domain. These denominators equal the Cycles expressions wherever its
    // predicates accept the sample, while rejected lanes remain finite.
    const auto half_vector = half_sum /
                             max(half_length, 1.0e-20f);
    const auto cos_nh = dot(normal, half_vector);
    const auto cos_hi = abs(dot(incoming, half_vector));
    const auto valid =
        (cos_ni > (sampling_domain ? 1.0e-5f : 0.0f)) &
        (sampling_domain ? Bool{true} : cos_no > 0.0f) &
        (abs(cos_nh) < 1.0f - 1.0e-5f) &
        (cos_hi > 1.0e-5f);
    const auto cos_nh_over_hi = max(
        cos_nh / max(cos_hi, 1.0e-20f), 1.0e-5f);
    const auto fac1 = 2.0f * abs(cos_nh_over_hi * cos_ni);
    const auto fac2 = 2.0f * abs(cos_nh_over_hi * cos_no);
    const auto sin_nh_squared = max(
        1.0f - cos_nh * cos_nh, 1.0e-20f);
    const auto sin_nh_fourth = sin_nh_squared * sin_nh_squared;
    const auto cotangent_squared =
        (cos_nh * cos_nh) / sin_nh_squared;
    // setup stored Cycles' invsigma2 in this model-specific common scalar.
    const auto distribution =
        exp(-cotangent_squared * closure.roughness) *
        closure.roughness * inverse_pi / sin_nh_fourth;
    const auto masking = min(1.0f, min(fac1, fac2));
    const auto intensity =
        0.25f * distribution * masking / max(cos_ni, 1.0e-20f);
    return {
        .intensity = select(0.0f, intensity, valid),
        .pdf = select(
            0.0f, cycles_sample_mapping::inverse_two_pi, valid),
        .valid = valid};
}

[[nodiscard]] Float3 sample_uniform_hemisphere(
    Float3 normal,
    Float2 random) noexcept {
    const auto disk = cycles_sample_mapping::sample_uniform_disk(random);
    const auto z = 1.0f - dot(disk, disk);
    const auto xy = disk * sqrt(max(z + 1.0f, 0.0f));
    const auto basis = cycles_sample_mapping::make_orthonormals(normal);
    return basis.tangent * xy.x +
           basis.bitangent * xy.y +
           normal * z;
}

[[nodiscard]] Float3 sample_cosine_hemisphere(
    Float3 normal, Float2 random) noexcept {
    return cycles_sample_mapping::sample_cosine_hemisphere(
        normal, random)
        .direction;
}

} // namespace psycles::luisa_backend::detail
