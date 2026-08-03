#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] Bool has_kind(
    const SurfaceClosureRecord &closure,
    SurfaceClosureKind kind) noexcept {
    return closure.kind ==
           static_cast<std::uint32_t>(kind);
}

[[nodiscard]] Bool has_lobe(
    const SurfaceClosureRecord &closure,
    SurfaceClosureLobe lobe) noexcept {
    return closure.lobe ==
           static_cast<std::uint32_t>(lobe);
}

}// namespace

SurfaceClosureEvaluator::SurfaceClosureEvaluator(
    const SurfacePoint &point,
    const SurfaceClosureSet &closures,
    Float3 shading_normal) noexcept
    : _point{point},
      _closures{closures},
      _shading_normal{shading_normal} {}

UInt SurfaceClosureEvaluator::runtime_flags(
    Float glossy_filter_roughness) const noexcept {
    UInt result = select(0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt index = 0u;
    $while(index < _closures.count()) {
        result |= detail::cycles_runtime_flags(
            _closures.entry(index),
            glossy_filter_roughness);
        index += 1u;
    };
    return result;
}

SurfaceClosureTrace SurfaceClosureEvaluator::closure_trace(
    UInt requested_index) const noexcept {
    const auto closure = _closures.entry(requested_index);
    const auto valid = requested_index < _closures.count();
    return {
        .count = _closures.count(),
        .runtime_flags = runtime_flags(),
        .index = requested_index,
        .type = detail::cycles_closure_type(closure),
        .sample_weight = closure.sample_weight,
        .weight = closure.weight,
        .normal = closure.normal,
        .valid = valid};
}

SurfaceAov SurfaceClosureEvaluator::aov() const noexcept {
    auto result = SurfaceAov{
        .albedo = make_float3(0.0f),
        .glossy_albedo = make_float3(0.0f),
        .transmission_albedo = make_float3(0.0f),
        .roughness = make_float2(0.0f),
        .normal = _point.shading_normal,
        .transparency = make_float3(0.0f)};
    Float total_weight = 0.0f;
    Float roughness_weight = 0.0f;
    Float roughness = 0.0f;
    Float3 normal = make_float3(0.0f);
    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
        const auto contribution =
            surface_closure_aov_contribution(
                _point, closure);
        result.albedo += contribution.albedo;
        result.glossy_albedo +=
            contribution.glossy_albedo;
        result.transmission_albedo +=
            contribution.transmission_albedo;
        result.transparency += contribution.transparency;
        total_weight += contribution.total_weight;
        roughness_weight +=
            contribution.roughness_weight;
        roughness += contribution.roughness;
        normal += contribution.normal;
        index += 1u;
    };
    const auto valid = total_weight > 0.0f;
    result.roughness = make_float2(select(
        1.0f,
        roughness /
            max(roughness_weight, 1.0e-20f),
        roughness_weight > 0.0f));
    result.normal = detail::safe_normalize(
        select(
            _point.shading_normal,
            normal,
            valid),
        _point.shading_normal);
    return result;
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate_impl(
    const ShaderServices &services,
    Float3 outgoing_expression,
    const SurfaceQuery &query,
    EvaluationMode mode,
    UInt light_shader_flags,
    UInt selected_closure_index) const noexcept {
    auto result = SurfaceEvaluation::zero();
    Float total_sample_weight = 0.0f;
    Float weighted_pdf = 0.0f;
    const auto outgoing = detail::safe_normalize(
        outgoing_expression, _point.shading_normal);
    const auto incoming = detail::safe_normalize(
        _point.incoming, -outgoing);
    const detail::MicrofacetGlassComponent microfacet_glass{
        services, _point};
    const auto sampled_light =
        mode == EvaluationMode::sampled_light;
    const auto sampled_bsdf =
        mode == EvaluationMode::sampled_bsdf;

    Bool light_diffuse_included = true;
    Bool light_glossy_included = true;
    Bool light_glass_included = true;
    if (sampled_light) {
        using namespace contract::cycles_abi;
        const auto excludes_diffuse =
            (light_shader_flags & shader_exclude_diffuse) != 0u;
        const auto excludes_glossy =
            (light_shader_flags & shader_exclude_glossy) != 0u;
        const auto excludes_transmit =
            (light_shader_flags & shader_exclude_transmit) != 0u;
        light_diffuse_included = !excludes_diffuse;
        light_glossy_included = !excludes_glossy;
        light_glass_included =
            !(excludes_glossy & excludes_transmit);
    }

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
    Bool has_diffuse = false;
    Bool has_translucent = false;
    Bool has_glossy = false;
    Bool has_glass_reflection = false;
    Bool has_glass_transmission = false;

    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
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
        const auto selected_sample = sampled_bsdf
                                         ? index == selected_closure_index
                                         : Bool{false};

        total_sample_weight += select(
            0.0f,
            detail::closure_sample_weight(closure),
            is_transparent & transparent_enabled);

        const auto glossy_normal = select(
            detail::maybe_ensure_valid_specular_reflection(
                _point, incoming, closure.normal),
            closure.normal,
            is_sheen);
        const auto glass_is_transmission =
            dot(glossy_normal, outgoing) < 0.0f;
        const auto selected_unit_ior_glass_delta =
            selected_sample & is_glass & glass_is_transmission &
            (abs(closure.ior - 1.0f) < 1.0e-4f);
        const auto bump_shadowing = detail::bump_shadowing_term(
            _point,
            _shading_normal,
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
        diffuse_pdf = select(
            0.0f, diffuse_pdf, bump_pdf_valid);
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
        auto glossy_allowed =
            glossy_enabled & generic_glossy;
        auto glass_allowed =
            is_glass & select(glossy_enabled,
                           transmission_enabled,
                           glass_is_transmission);
        diffuse_allowed &= closure.setup_valid;
        glossy_allowed &= closure.setup_valid;
        glass_allowed &= closure.setup_valid;
        const auto diffuse_contributes =
            diffuse_allowed & light_diffuse_included;
        const auto glossy_contributes =
            glossy_allowed & light_glossy_included;
        const auto glass_contributes =
            glass_allowed & light_glass_included;

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
        const auto enabled_pdf =
            select(0.0f, pdf, any_allowed);
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
        result.f +=
            eligible_diffuse + eligible_glossy + eligible_glass;
        result.diffuse_f += eligible_diffuse;
        result.glossy_f +=
            eligible_glossy + eligible_glass_reflection;

        auto weight = detail::closure_sample_weight(closure);
        weight = select(0.0f, weight, any_allowed);
        total_sample_weight += weight;
        weighted_pdf += weight * enabled_pdf;
        has_diffuse |=
            diffuse_contributes & !is_translucent &
            (detail::sample_weight(diffuse_contribution) > 0.0f);
        has_translucent |=
            translucent_allowed & light_diffuse_included &
            (detail::sample_weight(diffuse_contribution) > 0.0f);
        has_glossy |=
            glossy_contributes &
            (detail::sample_weight(glossy_contribution) > 0.0f);
        has_glass_reflection |=
            glass_contributes & !glass_is_transmission &
            (detail::sample_weight(glass_contribution) > 0.0f);
        has_glass_transmission |=
            glass_contributes & glass_is_transmission &
            (detail::sample_weight(glass_contribution) > 0.0f);
        index += 1u;
    };

    const auto has_pdf = total_sample_weight > 0.0f;
    result.pdf = select(0.0f,
        weighted_pdf / max(total_sample_weight, 1.0e-20f),
        has_pdf);
    result.diffuse_pdf = select(
        0.0f, result.pdf, has_diffuse | has_translucent);
    const auto has_diffuse_pdf = weighted_pdf > 0.0f;
    UInt events = static_cast<std::uint32_t>(event_none);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_diffuse | event_reflection),
        has_diffuse);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_glossy | event_reflection),
        has_glossy);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_diffuse | event_transmission),
        has_translucent);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_glossy | event_reflection),
        has_glass_reflection);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_glossy | event_transmission),
        has_glass_transmission);
    result.events = select(
        static_cast<std::uint32_t>(event_none),
        events,
        has_diffuse_pdf);
    if (sampled_light) {
        const auto use_mis =
            (light_shader_flags &
                contract::cycles_abi::shader_use_mis) != 0u;
        result.pdf = select(0.0f, result.pdf, use_mis);
    }
    return result;
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate(
    const ShaderServices &services,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query) const noexcept {
    return evaluate_impl(services,
        Float3{outgoing},
        query,
        EvaluationMode::regular,
        0u,
        ~std::uint32_t{0u});
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate_light(
    const ShaderServices &services,
    Expr<luisa::float3> outgoing,
    const SurfaceLightQuery &query) const noexcept {
    return evaluate_impl(services,
        Float3{outgoing},
        query.surface,
        EvaluationMode::sampled_light,
        query.shader_flags,
        ~std::uint32_t{0u});
}

}// namespace psycles::luisa_backend
