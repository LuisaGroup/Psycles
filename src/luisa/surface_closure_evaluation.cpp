#include <psycles/luisa/surface_closure_evaluation.h>

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"

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
    const detail::MicrofacetGlassComponent microfacet_glass{
        services, point};

    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
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
    const auto generic_glossy =
        (is_principled & !is_sheen) | is_glossy;

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

    const auto glossy_normal = select(
        detail::maybe_ensure_valid_specular_reflection(
            point, incoming, closure.normal),
        closure.normal,
        is_sheen);
    const auto glass_is_transmission =
        dot(glossy_normal, outgoing) < 0.0f;
    const auto selected_unit_ior_glass_delta =
        selected_sample & is_glass & glass_is_transmission &
        (abs(closure.ior - 1.0f) < 1.0e-4f);
    const auto bump_shadowing = detail::bump_shadowing_term(
        point,
        shading_normal,
        closure,
        outgoing,
        !selected_sample);
    const auto bump_direction_valid = bump_shadowing != 0.0f;
    const auto bump_pdf_valid =
        bump_direction_valid | selected_sample;
    const auto diffuse_normal = select(
        closure.normal, -glossy_normal, is_translucent);
    auto diffuse_pdf = select(
        max(dot(diffuse_normal, outgoing), 0.0f) *
            detail::inverse_pi,
        detail::sheen_intensity(closure, incoming, outgoing),
        is_sheen);
    diffuse_pdf = select(0.0f, diffuse_pdf, bump_pdf_valid);
    auto glossy_pdf = select(
        detail::microfacet_pdf(closure,
            incoming,
            outgoing,
            glossy_normal,
            query.glossy_filter_roughness),
        microfacet_glass.pdf(closure,
            incoming,
            outgoing,
            glossy_normal,
            glossy_enabled,
            transmission_enabled,
            query.glossy_filter_roughness),
        is_glass);
    glossy_pdf = select(
        0.0f, glossy_pdf, (!is_sheen) & bump_pdf_valid);
    glossy_pdf = select(
        glossy_pdf, 0.0f, selected_unit_ior_glass_delta);

    auto translucent_allowed =
        diffuse_enabled & transmission_enabled & is_translucent;
    auto diffuse_allowed =
        (diffuse_enabled & (is_diffuse | is_sheen)) |
        translucent_allowed;
    auto glossy_allowed = glossy_enabled & generic_glossy;
    auto glass_allowed =
        is_glass & select(glossy_enabled,
                       transmission_enabled,
                       glass_is_transmission);
    diffuse_allowed &= closure.setup_valid;
    glossy_allowed &= closure.setup_valid;
    glass_allowed &= closure.setup_valid;
    const auto diffuse_contributes =
        diffuse_allowed & policy.diffuse_included;
    const auto glossy_contributes =
        glossy_allowed & policy.glossy_included;
    const auto glass_contributes =
        glass_allowed & policy.glass_included;

    const auto diffuse_value =
        closure.weight *
        detail::diffuse_intensity(
            closure, incoming, outgoing) *
        bump_shadowing;
    const auto translucent_value =
        closure.weight *
        max(dot(-glossy_normal, outgoing), 0.0f) *
        detail::inverse_pi * bump_shadowing;
    const auto sheen_value =
        closure.weight * diffuse_pdf * bump_shadowing;
    const auto glossy_value =
        closure.weight *
        detail::microfacet_intensity(services,
            closure,
            incoming,
            outgoing,
            glossy_normal,
            query.glossy_filter_roughness) *
        bump_shadowing;
    auto glass_value =
        closure.weight *
        microfacet_glass.intensity(closure,
            incoming,
            outgoing,
            glossy_normal,
            query.glossy_filter_roughness) *
        bump_shadowing;
    glass_value = select(glass_value,
        make_float3(0.0f),
        selected_unit_ior_glass_delta);
    const auto diffuse_contribution =
        select(make_float3(0.0f), diffuse_value, is_diffuse) +
        select(make_float3(0.0f),
            translucent_value,
            is_translucent) +
        select(make_float3(0.0f), sheen_value, is_sheen);
    const auto glossy_contribution = select(
        make_float3(0.0f), glossy_value, generic_glossy);
    const auto glass_contribution = select(
        make_float3(0.0f), glass_value, is_glass);

    const auto pdf = select(
        select(diffuse_pdf, glossy_pdf, glossy_allowed),
        glossy_pdf,
        is_glass);
    const auto any_allowed =
        diffuse_allowed | glossy_allowed | glass_allowed;
    const auto enabled_pdf = select(0.0f, pdf, any_allowed);
    const auto eligible_diffuse = select(
        make_float3(0.0f),
        diffuse_contribution,
        diffuse_contributes);
    const auto eligible_glossy = select(
        make_float3(0.0f),
        glossy_contribution,
        glossy_contributes);
    const auto eligible_glass = select(
        make_float3(0.0f),
        glass_contribution,
        glass_contributes);
    const auto eligible_glass_reflection = select(
        make_float3(0.0f),
        glass_contribution,
        glass_contributes & !glass_is_transmission);

    auto weight = detail::closure_sample_weight(closure);
    weight = select(0.0f, weight, any_allowed);
    UInt events = static_cast<std::uint32_t>(event_none);
    events |= select(0u,
        static_cast<std::uint32_t>(
            event_diffuse | event_reflection),
        diffuse_contributes & !is_translucent &
            (detail::sample_weight(diffuse_contribution) > 0.0f));
    events |= select(0u,
        static_cast<std::uint32_t>(
            event_diffuse | event_transmission),
        translucent_allowed & policy.diffuse_included &
            (detail::sample_weight(diffuse_contribution) > 0.0f));
    events |= select(0u,
        static_cast<std::uint32_t>(
            event_glossy | event_reflection),
        glossy_contributes &
            (detail::sample_weight(glossy_contribution) > 0.0f));
    events |= select(0u,
        static_cast<std::uint32_t>(
            event_glossy | event_reflection),
        glass_contributes & !glass_is_transmission &
            (detail::sample_weight(glass_contribution) > 0.0f));
    events |= select(0u,
        static_cast<std::uint32_t>(
            event_glossy | event_transmission),
        glass_contributes & glass_is_transmission &
            (detail::sample_weight(glass_contribution) > 0.0f));

    luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
        result;
    result.f =
        eligible_diffuse + eligible_glossy + eligible_glass;
    result.diffuse_f = eligible_diffuse;
    result.glossy_f =
        eligible_glossy + eligible_glass_reflection;
    result.total_sample_weight =
        select(0.0f,
            detail::closure_sample_weight(closure),
            is_transparent & transparent_enabled) +
        weight;
    result.weighted_pdf = weight * enabled_pdf;
    result.events = events;
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
    result.events = select(
        static_cast<std::uint32_t>(event_none),
        _events,
        _weighted_pdf > 0.0f);
    return result;
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
    _result.events = result.events;
}

const SurfaceEvaluation &SurfaceClosureEvaluationVisitor::result()
    const noexcept {
    return _result;
}

}// namespace psycles::luisa_backend
