#include <psycles/luisa/surface_closure_evaluation.h>

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"
#include "thin_glass_component.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] Bool has_kind(
    const SurfaceClosureRecord &closure,
    SurfaceClosureKind kind) noexcept {
    return closure.kind == static_cast<std::uint32_t>(kind);
}

[[nodiscard]] Bool has_lobe(
    const SurfaceClosureRecord &closure,
    SurfaceClosureLobe lobe) noexcept {
    return closure.lobe == static_cast<std::uint32_t>(lobe);
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
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing_expression) noexcept {
    const auto outgoing = detail::safe_normalize(
        Float3{outgoing_expression}, point.shading_normal);
    return {
        .incoming = detail::safe_normalize(
            point.incoming, -outgoing),
        .outgoing = outgoing};
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
surface_closure_evaluation_contribution(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> shading_normal_expression,
    const SurfaceClosureRecord &closure,
    Expr<luisa::float3> incoming_expression,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Expr<bool> selected_sample_expression) noexcept {
    Float3 shading_normal{shading_normal_expression};
    Float3 incoming{incoming_expression};
    Float3 outgoing{outgoing_expression};
    Bool selected_sample{selected_sample_expression};
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_rough_translucent = has_kind(
        closure, SurfaceClosureKind::rough_translucent);
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
    const auto is_thin_glass_transmission = has_kind(
        closure, SurfaceClosureKind::thin_glass_transmission);
    const auto is_bssrdf = has_kind(
        closure, SurfaceClosureKind::bssrdf);
    const auto is_dielectric = is_glass | is_refraction;
    const auto generic_glossy =
        (is_principled & !is_sheen) | is_glossy;
    const auto diffuse_family =
        is_diffuse | is_translucent | is_rough_translucent |
        is_sheen | is_bssrdf;

    const auto diffuse_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_diffuse)) != 0u;
    const auto glossy_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_glossy)) != 0u;
    const auto transparent_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transparent)) != 0u;
    const auto transmission_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transmission)) != 0u;
    luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
        result;
    result.f = make_float3(0.0f);
    result.diffuse_f = make_float3(0.0f);
    result.glossy_f = make_float3(0.0f);
    result.total_sample_weight = 0.0f;
    result.weighted_pdf = 0.0f;
    result.weighted_roughness_squared = 0.0f;
    result.events = static_cast<std::uint32_t>(event_none);

    // A closure record has exactly one physical kind. Keep that algebraic
    // partition explicit in the generated device program: evaluating a
    // diffuse closure must not execute glass Jacobians or table lookups, and
    // evaluating glass must not execute Oren-Nayar or sheen transforms. The
    // previous select-only formulation produced the same mathematical sum,
    // but select is eager in the Luisa IR and therefore evaluated every BSDF
    // family for every record. Apart from undefined intermediate values for
    // inactive families, that also forced all family temporaries to coexist
    // in the HIP callable.
    $if(is_transparent) {
        result.total_sample_weight = select(
            0.0f,
            detail::closure_sample_weight(closure),
            transparent_enabled);
    }
    $elif(diffuse_family) {
        const auto translucent_family =
            is_translucent | is_rough_translucent;
        const auto translucent_allowed =
            diffuse_enabled & transmission_enabled &
            translucent_family;
        const auto allowed =
            ((diffuse_enabled &
                 (is_diffuse | is_sheen | is_bssrdf)) |
                translucent_allowed) &
            closure.setup_valid;

        $if(allowed) {
            Float pdf = 0.0f;
            Float3 value = make_float3(0.0f);

            // Cycles applies ordinary evaluation bump shadowing to every
            // closure. For the closure selected by bsdf_sample(), its common
            // post-sample correction is skipped exactly for transmitted
            // events. Translucent kinds are intrinsically transmissive.
            const auto evaluated_bump_shadowing =
                detail::bump_shadowing_term(
                    point,
                    shading_normal,
                    closure,
                    outgoing,
                    !selected_sample);
            const auto bump_shadowing = select(
                evaluated_bump_shadowing,
                1.0f,
                selected_sample & translucent_family);
            const auto bump_pdf_valid =
                (bump_shadowing != 0.0f) | selected_sample;

            $if(is_sheen) {
                pdf = detail::sheen_intensity(
                    closure, incoming, outgoing);
                pdf = select(0.0f, pdf, bump_pdf_valid);
                value = closure.weight * pdf * bump_shadowing;
            }
            $elif(is_diffuse) {
                pdf = max(dot(closure.normal, outgoing), 0.0f) *
                      detail::inverse_pi;
                pdf = select(0.0f, pdf, bump_pdf_valid);
                value = closure.weight *
                        detail::diffuse_intensity(
                            closure, incoming, outgoing) *
                        bump_shadowing;
            }
            $elif(is_translucent) {
                const auto glossy_normal =
                    detail::maybe_ensure_valid_specular_reflection(
                        point, incoming, closure.normal);
                pdf = max(dot(-glossy_normal, outgoing), 0.0f) *
                      detail::inverse_pi;
                pdf = select(0.0f, pdf, bump_pdf_valid);
                value = closure.weight * pdf * bump_shadowing;
            }
            $elif(is_rough_translucent) {
                pdf = max(dot(closure.normal, outgoing), 0.0f) *
                      detail::inverse_pi;
                pdf = select(0.0f, pdf, bump_pdf_valid);
                const auto reflected_incoming =
                    incoming - 2.0f * closure.normal *
                                   dot(incoming, closure.normal);
                value = closure.weight *
                        detail::diffuse_intensity(
                            closure,
                            reflected_incoming,
                            outgoing) *
                        bump_shadowing;
            };

            const auto contributes = policy.diffuse_included;
            const auto eligible_value = select(
                make_float3(0.0f), value, contributes);
            const auto weight =
                detail::closure_sample_weight(closure);
            const auto weighted_pdf = weight * pdf;
            const auto nonzero =
                detail::sample_weight(value) > 0.0f;
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
            result.diffuse_f = eligible_value;
            result.total_sample_weight = weight;
            result.weighted_pdf = weighted_pdf;
            result.weighted_roughness_squared =
                weighted_pdf *
                detail::cycles_bsdf_specular_roughness_squared(
                    closure, query.glossy_filter_roughness);
            result.events = events;
        };
    }
    $elif(generic_glossy) {
        const auto allowed = glossy_enabled & closure.setup_valid;
        $if(allowed) {
            const auto glossy_normal =
                detail::maybe_ensure_valid_specular_reflection(
                    point, incoming, closure.normal);
            const auto bump_shadowing =
                detail::bump_shadowing_term(
                    point,
                    shading_normal,
                    closure,
                    outgoing,
                    !selected_sample);
            const auto bump_pdf_valid =
                (bump_shadowing != 0.0f) | selected_sample;
            auto pdf = detail::microfacet_pdf(
                closure,
                incoming,
                outgoing,
                glossy_normal,
                query.glossy_filter_roughness);
            pdf = select(0.0f, pdf, bump_pdf_valid);
            const auto value =
                closure.weight *
                detail::microfacet_intensity(
                    services,
                    closure,
                    incoming,
                    outgoing,
                    glossy_normal,
                    query.glossy_filter_roughness) *
                bump_shadowing;
            const auto contributes = policy.glossy_included;
            const auto eligible_value = select(
                make_float3(0.0f), value, contributes);
            const auto weight =
                detail::closure_sample_weight(closure);
            const auto weighted_pdf = weight * pdf;

            result.f = eligible_value;
            result.glossy_f = eligible_value;
            result.total_sample_weight = weight;
            result.weighted_pdf = weighted_pdf;
            result.weighted_roughness_squared =
                weighted_pdf *
                detail::cycles_bsdf_specular_roughness_squared(
                    closure, query.glossy_filter_roughness);
            result.events = select(
                0u,
                static_cast<std::uint32_t>(
                    event_glossy | event_reflection),
                contributes &
                    (detail::sample_weight(value) > 0.0f));
        };
    }
    $elif(is_dielectric) {
        const detail::MicrofacetGlassComponent microfacet_glass{
            services, point};
        const auto glossy_normal =
            detail::maybe_ensure_valid_specular_reflection(
                point, incoming, closure.normal);
        const auto glass_is_transmission =
            dot(glossy_normal, outgoing) < 0.0f;
        const auto glass_allowed =
            is_glass &
            select(glossy_enabled,
                transmission_enabled,
                glass_is_transmission) &
            closure.setup_valid;
        const auto refraction_allowed =
            is_refraction & glossy_enabled & transmission_enabled &
            closure.setup_valid;
        const auto allowed = glass_allowed | refraction_allowed;

        $if(allowed) {
            const auto selected_unit_ior_glass_delta =
                selected_sample & glass_is_transmission &
                (abs(closure.ior - 1.0f) < 1.0e-4f);
            const auto evaluated_bump_shadowing =
                detail::bump_shadowing_term(
                    point,
                    shading_normal,
                    closure,
                    outgoing,
                    !selected_sample);
            const auto bump_shadowing = select(
                evaluated_bump_shadowing,
                1.0f,
                selected_sample & glass_is_transmission);
            const auto bump_pdf_valid =
                (bump_shadowing != 0.0f) | selected_sample;
            auto pdf = microfacet_glass.pdf(
                closure,
                incoming,
                outgoing,
                glossy_normal,
                glossy_enabled,
                transmission_enabled,
                query.glossy_filter_roughness);
            pdf = select(0.0f, pdf, bump_pdf_valid);
            pdf = select(
                pdf, 0.0f, selected_unit_ior_glass_delta);
            auto value =
                closure.weight *
                microfacet_glass.intensity(
                    closure,
                    incoming,
                    outgoing,
                    glossy_normal,
                    query.glossy_filter_roughness) *
                bump_shadowing;
            value = select(
                value,
                make_float3(0.0f),
                selected_unit_ior_glass_delta);

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
            const auto weight =
                detail::closure_sample_weight(closure);
            const auto weighted_pdf = weight * pdf;
            const auto nonzero =
                detail::sample_weight(value) > 0.0f;
            UInt events = static_cast<std::uint32_t>(event_none);
            events |= select(
                0u,
                static_cast<std::uint32_t>(
                    event_glossy | event_reflection),
                glass_contributes & !glass_is_transmission &
                    nonzero);
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
                weighted_pdf *
                detail::cycles_bsdf_specular_roughness_squared(
                    closure, query.glossy_filter_roughness);
            result.events = events;
        };
    }
    $elif(is_thin_glass_transmission) {
        const auto allowed =
            glossy_enabled & transmission_enabled &
            closure.setup_valid;
        $if(allowed) {
            const detail::ThinGlassComponent thin_glass{
                services, point};
            const auto evaluated_bump_shadowing =
                detail::bump_shadowing_term(
                    point,
                    shading_normal,
                    closure,
                    outgoing,
                    !selected_sample);
            const auto bump_shadowing = select(
                evaluated_bump_shadowing,
                1.0f,
                selected_sample);
            const auto bump_pdf_valid =
                (bump_shadowing != 0.0f) | selected_sample;
            auto pdf = thin_glass.pdf(
                closure,
                incoming,
                outgoing,
                query.glossy_filter_roughness);
            pdf = select(0.0f, pdf, bump_pdf_valid);
            const auto value =
                closure.weight *
                thin_glass.intensity(
                    closure,
                    incoming,
                    outgoing,
                    query.glossy_filter_roughness) *
                bump_shadowing;
            const auto contributes =
                policy.transmission_included;
            const auto eligible_value = select(
                make_float3(0.0f), value, contributes);
            const auto weight =
                detail::closure_sample_weight(closure);
            const auto weighted_pdf = weight * pdf;

            result.f = eligible_value;
            result.total_sample_weight = weight;
            result.weighted_pdf = weighted_pdf;
            result.weighted_roughness_squared =
                weighted_pdf *
                detail::cycles_bsdf_specular_roughness_squared(
                    closure, query.glossy_filter_roughness);
            result.events = select(
                0u,
                static_cast<std::uint32_t>(
                    event_glossy | event_transmission),
                contributes &
                    (detail::sample_weight(value) > 0.0f));
        };
    };
    return result;
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
        const SurfacePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy) noexcept
    : _services{services},
      _point{point},
      _query{query},
      _policy{policy} {}

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
        selected_sample);
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
