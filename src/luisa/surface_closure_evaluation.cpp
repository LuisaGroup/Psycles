#include <psycles/luisa/surface_closure_evaluation.h>

#include "graph_surface_internal.h"
#include "hair_closure_scattering.h"
#include "microfacet_glass_component.h"
#include "thin_glass_component.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr auto evaluation_general_payload_reachability =
    SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::principled) |
                 surface_closure_kind_bit(SurfaceClosureKind::glossy) |
                 surface_closure_kind_bit(SurfaceClosureKind::metallic_f82) |
                 surface_closure_kind_bit(
                     SurfaceClosureKind::metallic_conductor) |
                 surface_closure_kind_bit(
                     SurfaceClosureKind::sheen_microfiber) |
                 surface_closure_kind_bit(
                     SurfaceClosureKind::thin_glass_transmission),
        .principled_lobes = all_surface_closure_lobes,
        .anisotropic_microfacet_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::principled) |
            surface_closure_kind_bit(SurfaceClosureKind::glossy) |
            surface_closure_kind_bit(SurfaceClosureKind::metallic_f82) |
            surface_closure_kind_bit(
                SurfaceClosureKind::metallic_conductor),
        .thin_film_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::metallic_f82) |
            surface_closure_kind_bit(
                SurfaceClosureKind::metallic_conductor),
        .thin_film_principled_lobes =
            all_thin_film_principled_lobes};

inline constexpr auto evaluation_dielectric_payload_reachability =
    SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::glass) |
                 surface_closure_kind_bit(SurfaceClosureKind::refraction),
        .principled_lobes = 0u,
        .thin_film_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::glass)};

inline constexpr auto evaluation_hair_payload_reachability =
    SurfaceClosureReachability{
        .kinds =
            surface_closure_kind_bit(SurfaceClosureKind::hair_reflection) |
            surface_closure_kind_bit(SurfaceClosureKind::hair_transmission),
        .principled_lobes = 0u};

inline constexpr auto evaluation_common_only_reachability =
    all_surface_closure_reachability &
    SurfaceClosureReachability{
        .kinds = all_surface_closure_kinds &
                 ~evaluation_general_payload_reachability.kinds &
                 ~evaluation_hair_payload_reachability.kinds &
                 ~evaluation_dielectric_payload_reachability.kinds,
        .principled_lobes = 0u};

[[nodiscard]] luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
zero_surface_closure_evaluation_contribution() noexcept {
    luisa::compute::Var<SurfaceClosureEvaluationContributionCall> result;
    result.f = make_float3(0.0f);
    result.diffuse_f = make_float3(0.0f);
    result.glossy_f = make_float3(0.0f);
    result.total_sample_weight = 0.0f;
    result.weighted_pdf = 0.0f;
    result.weighted_roughness_squared = 0.0f;
    result.events = static_cast<std::uint32_t>(event_none);
    return result;
}

}// namespace

SurfaceClosureEvaluationPolicy
make_surface_closure_evaluation_policy(
    bool sampled_light,
    Expr<std::uint32_t> light_shader_flags_expression) noexcept {
    if (!sampled_light) {
        return {
            .diffuse_included = true,
            .glossy_included = true,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
    }
    using namespace contract::cycles_abi;
    UInt light_shader_flags{light_shader_flags_expression};
    const auto excludes_diffuse =
        (light_shader_flags & shader_exclude_diffuse) != 0u;
    const auto excludes_glossy =
        (light_shader_flags & shader_exclude_glossy) != 0u;
    const auto excludes_transmit =
        (light_shader_flags & shader_exclude_transmit) != 0u;
    return {
        .diffuse_included = !excludes_diffuse,
        .glossy_included = !excludes_glossy,
        .glass_included =
            !(excludes_glossy & excludes_transmit),
        // CLOSURE_IS_BSDF_TRANSMISSION is a distinct Cycles type interval,
        // not the intersection of glossy labels and transmitted events.
        .transmission_included = !excludes_transmit,
        .preserve_pdf =
            (light_shader_flags & shader_use_mis) != 0u};
}

SurfaceClosureEvaluationDirections
make_surface_closure_evaluation_directions(
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> outgoing_expression) noexcept {
    const auto outgoing = detail::safe_normalize(
        Float3{outgoing_expression}, point.shading_normal);
    return {
        .incoming = detail::safe_normalize(
            point.incoming, -outgoing),
        .outgoing = outgoing};
}

namespace {

[[nodiscard]] luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
evaluate_common_closure(
    const SurfaceClosurePoint &point,
    Float3 shading_normal,
    const SurfaceClosurePhysicalCommonOnlyRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Bool selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    const auto &common = closure.common;
    const UInt type{common.closure_type};
    const auto is_transparent =
        reachability.contains(SurfaceClosureKind::transparent)
            ? type == cycles_closure::type_transparent
            : Bool{false};
    const auto is_diffuse =
        reachability.contains(SurfaceClosureKind::diffuse)
            ? cycles_closure::is_diffuse_or_oren_nayar(type)
            : Bool{false};
    const auto is_translucent =
        reachability.contains(SurfaceClosureKind::translucent)
            ? type == cycles_closure::type_translucent
            : Bool{false};
    const auto is_rough_translucent =
        reachability.contains(SurfaceClosureKind::rough_translucent)
            ? type == cycles_closure::type_rough_translucent
            : Bool{false};
    const auto is_bssrdf =
        reachability.contains(SurfaceClosureKind::bssrdf)
            ? cycles_closure::is_bssrdf(type)
            : Bool{false};
    const auto is_ashikhmin =
        reachability.contains(SurfaceClosureKind::sheen_ashikhmin)
            ? type == cycles_closure::type_ashikhmin_velvet
            : Bool{false};
    const auto diffuse_family =
        is_diffuse | is_translucent | is_rough_translucent | is_bssrdf |
        is_ashikhmin;
    const auto transparent_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_transparent)) != 0u;
    const auto diffuse_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_diffuse)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_transmission)) != 0u;
    auto result = zero_surface_closure_evaluation_contribution();

    $if(is_transparent) {
        if (reachability.contains(SurfaceClosureKind::transparent)) {
            result.total_sample_weight = select(
                0.0f,
                detail::closure_sample_weight(common),
                transparent_enabled);
        }
    }
    $elif(diffuse_family) {
        const auto diffuse_family_possible =
            reachability.contains(SurfaceClosureKind::diffuse) ||
            reachability.contains(SurfaceClosureKind::translucent) ||
            reachability.contains(SurfaceClosureKind::rough_translucent) ||
            reachability.contains(SurfaceClosureKind::bssrdf) ||
            reachability.contains(SurfaceClosureKind::sheen_ashikhmin);
        if (diffuse_family_possible) {
            const auto translucent_family =
                is_translucent | is_rough_translucent;
            const auto allowed =
                ((diffuse_enabled &
                  (is_diffuse | is_bssrdf | is_ashikhmin)) |
                 (diffuse_enabled & transmission_enabled &
                  translucent_family));
            $if(allowed) {
                Float pdf = 0.0f;
                Float3 value = make_float3(0.0f);
                const auto evaluated_bump_shadowing =
                    detail::bump_shadowing_term(
                        point,
                        shading_normal,
                        common.normal,
                        // This predicate is Cycles' ClosureType interval,
                        // not the sampling label or this implementation's
                        // common-family dispatch. Ashikhmin Velvet samples
                        // with LABEL_DIFFUSE but type 16 belongs to the
                        // glossy interval, so Cycles applies only the common
                        // wrong-hemisphere rejection and never the diffuse
                        // bump-terminator softening to it. Every other member
                        // of this branch has a diffuse ClosureType.
                        !is_ashikhmin,
                        outgoing,
                        !selected_sample);
                const auto bump_shadowing = select(
                    evaluated_bump_shadowing,
                    1.0f,
                    selected_sample & translucent_family);
                const auto bump_pdf_valid =
                    (bump_shadowing != 0.0f) | selected_sample;

                $if(is_ashikhmin) {
                    const auto velvet =
                        detail::evaluate_ashikhmin_velvet(
                            common, incoming, outgoing);
                    pdf = select(0.0f, velvet.pdf, bump_pdf_valid);
                    value = common.weight * velvet.intensity *
                            bump_shadowing;
                }
                $elif(is_diffuse) {
                    pdf = max(dot(common.normal, outgoing), 0.0f) *
                          detail::inverse_pi;
                    pdf = select(0.0f, pdf, bump_pdf_valid);
                    value = common.weight *
                            detail::diffuse_intensity(
                                common,
                                common.color_or_evaluation_scale,
                                incoming,
                                outgoing) *
                            bump_shadowing;
                }
                $elif(is_translucent) {
                    const auto glossy_normal =
                        detail::maybe_ensure_valid_specular_reflection(
                            point, incoming, common.normal);
                    pdf = max(dot(-glossy_normal, outgoing), 0.0f) *
                          detail::inverse_pi;
                    pdf = select(0.0f, pdf, bump_pdf_valid);
                    value = common.weight * pdf * bump_shadowing;
                }
                $elif(is_rough_translucent) {
                    pdf = max(dot(common.normal, outgoing), 0.0f) *
                          detail::inverse_pi;
                    pdf = select(0.0f, pdf, bump_pdf_valid);
                    const auto reflected_incoming =
                        incoming - 2.0f * common.normal *
                                       dot(incoming, common.normal);
                    value = common.weight *
                            detail::diffuse_intensity(
                                common,
                                common.color_or_evaluation_scale,
                                reflected_incoming,
                                outgoing) *
                            bump_shadowing;
                };

                // Ashikhmin's sample label is diffuse, but its Cycles
                // ClosureType belongs to the glossy interval. Light linking
                // and BsdfEval pass routing therefore use the glossy policy.
                const auto contributes = select(
                    policy.diffuse_included,
                    policy.glossy_included,
                    is_ashikhmin);
                const auto eligible_value = select(
                    make_float3(0.0f), value, contributes);
                const auto weight = detail::closure_sample_weight(common);
                const auto weighted_pdf = weight * pdf;
                const auto nonzero = detail::sample_weight(value) > 0.0f;
                UInt events = static_cast<std::uint32_t>(event_none);
                events |= select(
                    0u,
                    static_cast<std::uint32_t>(
                        event_diffuse | event_reflection),
                    contributes & !translucent_family & nonzero);
                events |= select(
                    0u,
                    static_cast<std::uint32_t>(
                        event_diffuse | event_transmission),
                    contributes & translucent_family & nonzero);
                result.f = eligible_value;
                result.diffuse_f = select(
                    eligible_value,
                    make_float3(0.0f),
                    is_ashikhmin);
                result.glossy_f = select(
                    make_float3(0.0f),
                    eligible_value,
                    is_ashikhmin);
                result.total_sample_weight = weight;
                result.weighted_pdf = weighted_pdf;
                result.weighted_roughness_squared = weighted_pdf;
                result.events = events;
            };
        }
    };
    return result;
}

[[nodiscard]] luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
evaluate_general_closure(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Float3 shading_normal,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Bool selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    const auto &common = closure.common;
    const UInt type{common.closure_type};
    const auto is_sheen =
        (reachability.contains_principled_lobe(
             SurfaceClosureLobe::sheen) ||
         reachability.contains(SurfaceClosureKind::sheen_microfiber))
            ? type == cycles_closure::type_sheen
            : Bool{false};
    const auto generic_glossy_possible =
        reachability.contains(SurfaceClosureKind::principled) ||
        reachability.contains(SurfaceClosureKind::glossy) ||
        reachability.contains(SurfaceClosureKind::metallic_f82) ||
        reachability.contains(SurfaceClosureKind::metallic_conductor);
    const auto generic_glossy =
        generic_glossy_possible
            ? cycles_closure::is_reflection_microfacet(type)
            : Bool{false};
    const auto is_thin =
        reachability.contains(
            SurfaceClosureKind::thin_glass_transmission)
            ? type == cycles_closure::type_thin_glass_transmission
            : Bool{false};
    const auto diffuse_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_diffuse)) != 0u;
    const auto glossy_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_glossy)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_transmission)) != 0u;
    auto result = zero_surface_closure_evaluation_contribution();

    $if(is_sheen) {
        if (reachability.contains_principled_lobe(
                SurfaceClosureLobe::sheen) ||
            reachability.contains(SurfaceClosureKind::sheen_microfiber)) {
            const auto allowed = diffuse_enabled;
            $if(allowed) {
                const auto bump_shadowing =
                    detail::bump_shadowing_term(
                        point,
                        shading_normal,
                        common.normal,
                        true,
                        outgoing,
                        !selected_sample);
                const auto bump_pdf_valid =
                    (bump_shadowing != 0.0f) | selected_sample;
                auto pdf = detail::sheen_intensity(
                    closure, incoming, outgoing);
                pdf = select(0.0f, pdf, bump_pdf_valid);
                const auto value =
                    common.weight * pdf * bump_shadowing;
                const auto contributes = policy.diffuse_included;
                const auto eligible_value = select(
                    make_float3(0.0f), value, contributes);
                const auto weight = detail::closure_sample_weight(common);
                const auto weighted_pdf = weight * pdf;
                result.f = eligible_value;
                result.diffuse_f = eligible_value;
                result.total_sample_weight = weight;
                result.weighted_pdf = weighted_pdf;
                result.weighted_roughness_squared = weighted_pdf;
                result.events = select(
                    0u,
                    static_cast<std::uint32_t>(
                        event_diffuse | event_reflection),
                    contributes &
                        (detail::sample_weight(value) > 0.0f));
            };
        }
    }
    $elif(generic_glossy) {
        const auto principled_glossy_possible =
            reachability.contains(SurfaceClosureKind::principled) &&
            (reachability.principled_lobes &
             (all_surface_closure_lobes &
              ~surface_closure_lobe_bit(SurfaceClosureLobe::sheen))) != 0u;
        if (reachability.contains(SurfaceClosureKind::glossy) ||
            reachability.contains(SurfaceClosureKind::metallic_f82) ||
            reachability.contains(SurfaceClosureKind::metallic_conductor) ||
            principled_glossy_possible) {
            const auto allowed = glossy_enabled;
            $if(allowed) {
                const auto glossy_normal =
                    detail::maybe_ensure_valid_specular_reflection(
                        point, incoming, common.normal);
                const auto bump_shadowing =
                    detail::bump_shadowing_term(
                        point,
                        shading_normal,
                        common.normal,
                        false,
                        outgoing,
                        !selected_sample);
                const auto bump_pdf_valid =
                    (bump_shadowing != 0.0f) | selected_sample;
                const auto may_have_f82 =
                    reachability.contains_principled_lobe(
                        SurfaceClosureLobe::metallic) ||
                    reachability.contains(
                        SurfaceClosureKind::metallic_f82);
                const auto may_have_f82_thin_film =
                    reachability.contains_thin_film_principled_lobe(
                        SurfaceClosureLobe::metallic) ||
                    reachability.contains_thin_film(
                        SurfaceClosureKind::metallic_f82);
                const auto may_have_dielectric =
                    reachability.contains_principled_lobe(
                        SurfaceClosureLobe::coat);
                const auto may_have_generalized_schlick =
                    reachability.contains_principled_lobe(
                        SurfaceClosureLobe::dielectric);
                const auto evaluation = detail::microfacet_evaluate(
                    services,
                    closure,
                    incoming,
                    outgoing,
                    glossy_normal,
                    query.glossy_filter_roughness,
                    reachability.contains_anisotropic_microfacet(
                        SurfaceClosureKind::principled) ||
                        reachability.contains_anisotropic_microfacet(
                            SurfaceClosureKind::glossy) ||
                        reachability.contains_anisotropic_microfacet(
                            SurfaceClosureKind::metallic_f82) ||
                        reachability.contains_anisotropic_microfacet(
                            SurfaceClosureKind::metallic_conductor),
                    may_have_f82,
                    may_have_f82_thin_film,
                    may_have_dielectric,
                    may_have_generalized_schlick,
                    reachability.contains_thin_film_principled_lobe(
                        SurfaceClosureLobe::dielectric),
                    reachability.contains(
                        SurfaceClosureKind::metallic_conductor),
                    reachability.contains_thin_film(
                        SurfaceClosureKind::metallic_conductor));
                const auto pdf = select(
                    0.0f, evaluation.pdf, bump_pdf_valid);
                const auto value =
                    common.weight * evaluation.intensity * bump_shadowing;
                const auto contributes = policy.glossy_included;
                const auto eligible_value = select(
                    make_float3(0.0f), value, contributes);
                const auto weight = detail::closure_sample_weight(common);
                const auto weighted_pdf = weight * pdf;
                result.f = eligible_value;
                result.glossy_f = eligible_value;
                result.total_sample_weight = weight;
                result.weighted_pdf = weighted_pdf;
                result.weighted_roughness_squared =
                    weighted_pdf * evaluation.roughness_squared;
                result.events = select(
                    0u,
                    static_cast<std::uint32_t>(
                        event_glossy | event_reflection),
                    contributes &
                        (detail::sample_weight(value) > 0.0f));
            };
        }
    }
    $elif(is_thin) {
        if (reachability.contains(
                SurfaceClosureKind::thin_glass_transmission)) {
            const auto allowed = glossy_enabled & transmission_enabled;
            $if(allowed) {
                const detail::ThinGlassComponent thin_glass{
                    services, point};
                const auto evaluated_bump_shadowing =
                    detail::bump_shadowing_term(
                        point,
                        shading_normal,
                        common.normal,
                        false,
                        outgoing,
                        !selected_sample);
                const auto bump_shadowing = select(
                    evaluated_bump_shadowing, 1.0f, selected_sample);
                const auto bump_pdf_valid =
                    (bump_shadowing != 0.0f) | selected_sample;
                const auto evaluation = thin_glass.evaluate(
                    closure,
                    incoming,
                    outgoing,
                    query.glossy_filter_roughness);
                const auto pdf = select(
                    0.0f, evaluation.pdf, bump_pdf_valid);
                const auto value =
                    common.weight * evaluation.intensity * bump_shadowing;
                const auto contributes = policy.transmission_included;
                const auto eligible_value = select(
                    make_float3(0.0f), value, contributes);
                const auto weight = detail::closure_sample_weight(common);
                const auto weighted_pdf = weight * pdf;
                result.f = eligible_value;
                result.total_sample_weight = weight;
                result.weighted_pdf = weighted_pdf;
                result.weighted_roughness_squared =
                    weighted_pdf * evaluation.roughness_squared;
                result.events = select(
                    0u,
                    static_cast<std::uint32_t>(
                        event_glossy | event_transmission),
                    contributes &
                        (detail::sample_weight(value) > 0.0f));
            };
        }
    };
    return result;
}

[[nodiscard]] luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
evaluate_dielectric_closure(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Float3 shading_normal,
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Bool selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    const auto &common = closure.common;
    const auto is_glass =
        reachability.contains(SurfaceClosureKind::glass)
            ? cycles_closure::is_glass_microfacet(common.closure_type)
            : Bool{false};
    const auto is_refraction =
        reachability.contains(SurfaceClosureKind::refraction)
            ? cycles_closure::is_refraction_microfacet(
                  common.closure_type)
            : Bool{false};
    const auto glossy_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_glossy)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_transmission)) != 0u;
    auto result = zero_surface_closure_evaluation_contribution();
    const detail::MicrofacetGlassComponent microfacet_glass{
        services, point};
    const auto glossy_normal =
        detail::maybe_ensure_valid_specular_reflection(
            point, incoming, common.normal);
    const auto glass_is_transmission =
        dot(glossy_normal, outgoing) < 0.0f;
    const auto glass_allowed =
        is_glass &
        select(glossy_enabled,
               transmission_enabled,
               glass_is_transmission);
    const auto refraction_allowed =
        is_refraction & glossy_enabled & transmission_enabled;
    const auto allowed = glass_allowed | refraction_allowed;
    $if(allowed) {
        const auto selected_unit_ior_glass_delta =
            selected_sample & glass_is_transmission &
            (abs(closure.payload.ior - 1.0f) < 1.0e-4f);
        const auto evaluated_bump_shadowing =
            detail::bump_shadowing_term(
                point,
                shading_normal,
                common.normal,
                false,
                outgoing,
                !selected_sample);
        const auto bump_shadowing = select(
            evaluated_bump_shadowing,
            1.0f,
            selected_sample & glass_is_transmission);
        const auto bump_pdf_valid =
            (bump_shadowing != 0.0f) | selected_sample;
        const auto evaluation = microfacet_glass.evaluate(
            closure,
            incoming,
            outgoing,
            glossy_normal,
            glossy_enabled,
            transmission_enabled,
            query.glossy_filter_roughness,
            reachability.contains_thin_film(
                SurfaceClosureKind::glass));
        auto pdf = select(0.0f, evaluation.pdf, bump_pdf_valid);
        pdf = select(pdf, 0.0f, selected_unit_ior_glass_delta);
        auto value =
            common.weight * evaluation.intensity * bump_shadowing;
        value = select(
            value, make_float3(0.0f), selected_unit_ior_glass_delta);
        const auto glass_contributes =
            glass_allowed & policy.glass_included;
        const auto refraction_contributes =
            refraction_allowed & policy.transmission_included;
        const auto contributes =
            glass_contributes | refraction_contributes;
        const auto eligible_value = select(
            make_float3(0.0f), value, contributes);
        const auto eligible_reflection = select(
            make_float3(0.0f),
            value,
            glass_contributes & !glass_is_transmission);
        const auto weight = detail::closure_sample_weight(common);
        const auto weighted_pdf = weight * pdf;
        const auto nonzero = detail::sample_weight(value) > 0.0f;
        UInt events = static_cast<std::uint32_t>(event_none);
        events |= select(
            0u,
            static_cast<std::uint32_t>(
                event_glossy | event_reflection),
            glass_contributes & !glass_is_transmission & nonzero);
        events |= select(
            0u,
            static_cast<std::uint32_t>(
                event_glossy | event_transmission),
            contributes & glass_is_transmission & nonzero);
        result.f = eligible_value;
        result.glossy_f = eligible_reflection;
        result.total_sample_weight = weight;
        result.weighted_pdf = weighted_pdf;
        result.weighted_roughness_squared =
            weighted_pdf * evaluation.roughness_squared;
        result.events = events;
    };
    return result;
}

[[nodiscard]] luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
evaluate_hair_closure(
    const SurfaceClosurePoint &point,
    Float3 shading_normal,
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Bool selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    const auto reflection =
        reachability.contains(SurfaceClosureKind::hair_reflection)
            ? closure.common.closure_type ==
                  cycles_closure::type_hair_reflection
            : Bool{false};
    const auto glossy_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_glossy)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
         static_cast<std::uint32_t>(event_transmission)) != 0u;
    const auto allowed = glossy_enabled &
                         select(transmission_enabled, Bool{true}, reflection);
    auto result = zero_surface_closure_evaluation_contribution();
    $if(allowed) {
        Float intensity = 0.0f;
        Float pdf = 0.0f;
        $if(reflection) {
            if (reachability.contains(SurfaceClosureKind::hair_reflection)) {
                const auto evaluation = detail::evaluate_hair_reflection(
                    closure, incoming, outgoing, selected_sample);
                intensity = evaluation.intensity;
                pdf = evaluation.pdf;
            }
        }
        $else {
            if (reachability.contains(SurfaceClosureKind::hair_transmission)) {
                const auto evaluation = detail::evaluate_hair_transmission(
                    closure, incoming, outgoing, selected_sample);
                intensity = evaluation.intensity;
                pdf = evaluation.pdf;
            }
        };
        const auto bump_shadowing = detail::bump_shadowing_term(
            point,
            shading_normal,
            closure.common.normal,
            false,
            outgoing,
            !selected_sample);
        const auto bump_pdf_valid =
            (bump_shadowing != 0.0f) | selected_sample;
        pdf = select(0.0f, pdf, bump_pdf_valid);
        const auto value = closure.common.weight * intensity * bump_shadowing;
        const auto contributes = select(
            policy.transmission_included,
            policy.glossy_included,
            reflection);
        const auto eligible_value = select(
            make_float3(0.0f), value, contributes);
        const auto weight = detail::closure_sample_weight(closure.common);
        const auto weighted_pdf = weight * pdf;
        const auto nonzero = detail::sample_weight(value) > 0.0f;
        result.f = eligible_value;
        result.glossy_f = select(
            make_float3(0.0f), eligible_value, reflection);
        result.total_sample_weight = weight;
        result.weighted_pdf = weighted_pdf;
        // Legacy Hair is neither singular nor microfacet; Cycles' generic
        // roughness classifier therefore returns exactly one.
        result.weighted_roughness_squared = weighted_pdf;
        result.events = select(
            0u,
            select(
                static_cast<std::uint32_t>(
                    event_glossy | event_transmission),
                static_cast<std::uint32_t>(
                    event_glossy | event_reflection),
                reflection),
            contributes & nonzero);
    };
    return result;
}

}// namespace

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
surface_closure_evaluation_contribution(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal_expression,
    const SurfaceClosurePhysicalRecord &closure,
    Expr<luisa::float3> incoming_expression,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Expr<bool> selected_sample_expression,
    SurfaceClosureReachability reachability) noexcept {
    const auto common = project_surface_closure_physical_common(closure);
    const auto is_general = surface_closure_uses_general_payload(
        common.closure_type);
    const auto is_dielectric = surface_closure_uses_dielectric_payload(
        common.closure_type);
    const auto is_hair = surface_closure_uses_hair_payload(
        common.closure_type);
    auto result = zero_surface_closure_evaluation_contribution();
    $if(is_dielectric) {
        result = evaluate_dielectric_closure(
            services,
            point,
            Float3{shading_normal_expression},
            project_surface_closure_physical_dielectric(closure),
            Float3{incoming_expression},
            Float3{outgoing_expression},
            query,
            policy,
            Bool{selected_sample_expression},
            reachability & evaluation_dielectric_payload_reachability);
    }
    $elif(is_hair) {
        result = evaluate_hair_closure(
            point,
            Float3{shading_normal_expression},
            project_surface_closure_physical_hair(closure),
            Float3{incoming_expression},
            Float3{outgoing_expression},
            query,
            policy,
            Bool{selected_sample_expression},
            reachability & evaluation_hair_payload_reachability);
    }
    $elif(is_general) {
        result = evaluate_general_closure(
            services,
            point,
            Float3{shading_normal_expression},
            project_surface_closure_physical_general(closure),
            Float3{incoming_expression},
            Float3{outgoing_expression},
            query,
            policy,
            Bool{selected_sample_expression},
            reachability & evaluation_general_payload_reachability);
    }
    $else {
        result = evaluate_common_closure(
            point,
            Float3{shading_normal_expression},
            project_surface_closure_physical_common_only(closure),
            Float3{incoming_expression},
            Float3{outgoing_expression},
            query,
            policy,
            Bool{selected_sample_expression},
            reachability & evaluation_common_only_reachability);
    };
    return result;
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
surface_closure_evaluation_contribution_from_physical_common(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    const SurfaceClosurePhysicalCommonRecord &common,
    const SurfaceClosurePhysicalPayloadLoader &load_payload,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Expr<bool> selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    Bool is_general_payload = false;
    if ((reachability.kinds &
         evaluation_general_payload_reachability.kinds) != 0u) {
        is_general_payload = surface_closure_uses_general_payload(
            common.closure_type);
    }
    Bool is_dielectric_payload = false;
    if ((reachability.kinds &
         evaluation_dielectric_payload_reachability.kinds) != 0u) {
        is_dielectric_payload = surface_closure_uses_dielectric_payload(
            common.closure_type);
    }
    Bool is_hair_payload = false;
    if ((reachability.kinds &
         evaluation_hair_payload_reachability.kinds) != 0u) {
        is_hair_payload = surface_closure_uses_hair_payload(
            common.closure_type);
    }

    auto result = zero_surface_closure_evaluation_contribution();
    // The three reachability meets are a proof obligation as well as a
    // specialization: the inner canonical consumer cannot re-introduce a tag
    // excluded by the dominating type predicate. Only `result` crosses the
    // branch merge, so mutually exclusive payload fields never coexist in the
    // caller's loop-carried state.
    $if(is_dielectric_payload) {
        const auto closure =
            unpack_surface_closure_physical_dielectric(common, load_payload());
        result = evaluate_dielectric_closure(
            services,
            point,
            Float3{shading_normal},
            closure,
            Float3{incoming},
            Float3{outgoing},
            query,
            policy,
            Bool{selected_sample},
            reachability & evaluation_dielectric_payload_reachability);
    }
    $elif(is_hair_payload) {
        const auto closure =
            unpack_surface_closure_physical_hair(common, load_payload());
        result = evaluate_hair_closure(
            point,
            Float3{shading_normal},
            closure,
            Float3{incoming},
            Float3{outgoing},
            query,
            policy,
            Bool{selected_sample},
            reachability & evaluation_hair_payload_reachability);
    }
    $elif(is_general_payload) {
        const auto closure =
            unpack_surface_closure_physical_general(common, load_payload());
        result = evaluate_general_closure(
            services,
            point,
            Float3{shading_normal},
            closure,
            Float3{incoming},
            Float3{outgoing},
            query,
            policy,
            Bool{selected_sample},
            reachability & evaluation_general_payload_reachability);
    }
    $else {
        const auto closure = unpack_surface_closure_physical_common_only(common);
        result = evaluate_common_closure(
            point,
            Float3{shading_normal},
            closure,
            Float3{incoming},
            Float3{outgoing},
            query,
            policy,
            Bool{selected_sample},
            reachability & evaluation_common_only_reachability);
    };
    return result;
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
surface_closure_evaluation_contribution_from_physical_blocks(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Expr<bool> selected_sample,
    SurfaceClosureReachability reachability) noexcept {
    const auto common = unpack_surface_closure_physical_common(block_0);
    return surface_closure_evaluation_contribution_from_physical_common(
        services, point, shading_normal, common,
        [block_1] { return luisa::compute::Float4x4{block_1}; },
        incoming, outgoing, query, policy, selected_sample, reachability);
}

SurfaceClosureEvaluationAccumulator::
    SurfaceClosureEvaluationAccumulator() noexcept
    : _result{SurfaceEvaluation::zero()} {}

void SurfaceClosureEvaluationAccumulator::add(
    const luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
        &contribution) noexcept {
    _result.f += contribution.f;
    _result.diffuse_f += contribution.diffuse_f;
    _result.glossy_f += contribution.glossy_f;
    _total_sample_weight += contribution.total_sample_weight;
    _weighted_pdf += contribution.weighted_pdf;
    _weighted_roughness_squared +=
        contribution.weighted_roughness_squared;
    _events |= contribution.events;
}

SurfaceEvaluation SurfaceClosureEvaluationAccumulator::finish(
    Expr<bool> preserve_pdf_expression) const noexcept {
    Bool preserve_pdf{preserve_pdf_expression};
    const auto has_pdf = _total_sample_weight > 0.0f;
    const auto pdf = select(0.0f,
        _weighted_pdf /
            max(_total_sample_weight, 1.0e-20f),
        has_pdf);
    auto result = SurfaceEvaluation::zero();
    result.f = _result.f;
    result.pdf = select(0.0f, pdf, preserve_pdf);
    result.diffuse_f = _result.diffuse_f;
    result.glossy_f = _result.glossy_f;
    result.diffuse_pdf = select(
        0.0f,
        pdf,
        (_events & static_cast<std::uint32_t>(
                       event_diffuse)) != 0u);
    result.average_roughness_squared = select(
        0.0f,
        _weighted_roughness_squared /
            max(_weighted_pdf, 1.0e-20f),
        _weighted_pdf > 0.0f);
    result.events = select(
        static_cast<std::uint32_t>(event_none),
        _events,
        _weighted_pdf > 0.0f);
    return result;
}

DirectSurfaceClosureEvaluationOperation::
    DirectSurfaceClosureEvaluationOperation(
        const ShaderServices &services,
        const SurfaceClosurePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy,
        SurfaceClosureReachability reachability) noexcept
    : _services{services},
      _point{point},
      _query{query},
      _policy{policy},
      _reachability{reachability} {}

void DirectSurfaceClosureEvaluationOperation::set_outgoing(
    Expr<luisa::float3> outgoing) noexcept {
    const auto directions =
        make_surface_closure_evaluation_directions(
            _point, outgoing);
    _incoming = directions.incoming;
    _outgoing = directions.outgoing;
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
DirectSurfaceClosureEvaluationOperation::evaluate(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosureExpression &closure,
    Expr<bool> selected_sample) const noexcept {
    return surface_closure_evaluation_contribution(
        _services,
        _point,
        shading_normal,
        closure.reference(),
        Expr<luisa::float3>{_incoming.expression()},
        Expr<luisa::float3>{_outgoing.expression()},
        _query,
        _policy,
        selected_sample, _reachability);
}

SurfaceClosureEvaluationVisitor::SurfaceClosureEvaluationVisitor(
    std::size_t capacity,
    const SurfaceClosureEvaluationOperation &operation,
    Expr<bool> preserve_pdf,
    Expr<std::uint32_t> selected_closure_index) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _operation{operation},
      _preserve_pdf{preserve_pdf},
      _selected_closure_index{selected_closure_index},
      _result{SurfaceEvaluation::zero()} {}

void SurfaceClosureEvaluationVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression>
        &closures) noexcept {
    SurfaceClosureEvaluationAccumulator accumulator;
    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retains(closure, allocated_count);
        $if(keep) {
            accumulator.add(_operation.evaluate(
                shading_normal,
                closure,
                allocated_count == _selected_closure_index));
        };
        allocated_count += select(0u, 1u, keep);
    }
    const auto result = accumulator.finish(_preserve_pdf);
    _result.f = result.f;
    _result.pdf = result.pdf;
    _result.diffuse_f = result.diffuse_f;
    _result.glossy_f = result.glossy_f;
    _result.diffuse_pdf = result.diffuse_pdf;
    _result.average_roughness_squared =
        result.average_roughness_squared;
    _result.events = result.events;
}

const SurfaceEvaluation &SurfaceClosureEvaluationVisitor::result()
    const noexcept {
    return _result;
}

}// namespace psycles::luisa_backend
