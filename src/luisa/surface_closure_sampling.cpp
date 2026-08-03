#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"

#include <psycles/luisa/cycles_closure.h>

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

SurfaceSampleTrace SurfaceClosureEvaluator::sample_impl(
    const ShaderServices &services,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query,
    bool trace_selection) const noexcept {
    auto trace = SurfaceSampleTrace::zero();
    auto &result = trace.sample;
    const auto incoming = detail::safe_normalize(
        _point.incoming, _point.shading_normal);
    const detail::MicrofacetGlassComponent microfacet_glass{
        services, _point};

    Float total_weight = 0.0f;
    UInt surface_runtime_flags = select(
        0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
        surface_runtime_flags |= detail::cycles_runtime_flags(
            closure, query.glossy_filter_roughness);
        total_weight += detail::closure_selection_state(
                            services,
                            _point,
                            closure,
                            incoming,
                            query)
                            .weight;
        index += 1u;
    };
    result.runtime_flags = surface_runtime_flags;

    const auto random_lobe = clamp(
        u_lobe, 0.0f, 0.99999994f);
    const auto target = random_lobe * total_weight;
    Float accumulated = 0.0f;
    Bool selected = false;
    UInt selected_closure_index = ~std::uint32_t{0u};
    Float selected_rescaled = 0.0f;

    // Invert the categorical closure distribution independently of the
    // conditional BSDF sampler. Keeping this loop restricted to selection
    // state is both the formal decomposition p(i, wi) = p(i) p(wi | i) and
    // prevents the complete sampler CFG from becoming loop-carried SSA.
    index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
        const auto selection = detail::closure_selection_state(
            services, _point, closure, incoming, query);
        const auto weight = selection.weight;
        const auto next = accumulated + weight;
        const auto choose =
            !selected & (weight > 0.0f) & (target < next);
        const auto rescaled = select(
            random_lobe,
            (target - accumulated) /
                max(weight, 1.0e-20f),
            _closures.count() > 1u);
        selected_closure_index = select(
            selected_closure_index, index, choose);
        selected_rescaled = select(
            selected_rescaled, rescaled, choose);
        selected |= choose;
        accumulated = next;
        index += 1u;
    };

    const auto closure = _closures.entry(selected_closure_index);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto sample_glossy =
        is_glossy | (is_principled & !is_sheen);
    const auto selection = detail::closure_selection_state(
        services, _point, closure, incoming, query);
    const auto selected_weight = selection.weight;

    Float3 candidate_direction =
        make_float3(0.0f, 0.0f, 1.0f);
    Float2 candidate_roughness = make_float2(1.0f);
    Bool candidate_valid = true;
    Bool candidate_glossy_singular = false;
    Float3 candidate_glossy_singular_evaluation =
        make_float3(0.0f);
    Float candidate_glossy_singular_pdf = 0.0f;
    Bool candidate_glass_transmission = false;
    Bool candidate_glass_singular = false;
    Float3 candidate_glass_singular_evaluation =
        make_float3(0.0f);
    Float candidate_glass_singular_pdf = 0.0f;
    Float candidate_glass_eta = 1.0f;

    $if(selected) {
        $if(is_transparent) {
            candidate_direction = -_point.incoming;
            candidate_roughness = make_float2(0.0f);
        }
        $elif(is_glass) {
            const auto glass = microfacet_glass.sample(
                closure,
                incoming,
                selection.glossy_normal,
                u_direction,
                selected_rescaled,
                (query.lobe_mask &
                    static_cast<std::uint32_t>(
                        event_glossy)) != 0u,
                (query.lobe_mask &
                    static_cast<std::uint32_t>(
                        event_transmission)) != 0u,
                query.glossy_filter_roughness);
            candidate_direction = glass.direction;
            candidate_roughness = make_float2(glass.alpha);
            candidate_valid = glass.valid;
            candidate_glass_transmission = glass.transmission;
            candidate_glass_singular = glass.singular;
            candidate_glass_singular_evaluation =
                glass.singular_evaluation;
            candidate_glass_singular_pdf = glass.singular_pdf;
            candidate_glass_eta = glass.eta;
        }
        $elif(is_sheen) {
            candidate_direction = detail::sample_sheen(
                closure, incoming, u_direction);
        }
        $elif(sample_glossy) {
            const auto glossy =
                detail::sample_microfacet_reflection(
                    _point,
                    _shading_normal,
                    closure,
                    incoming,
                    u_direction,
                    selection.glossy_normal,
                    query.glossy_filter_roughness);
            candidate_direction = glossy.direction;
            candidate_roughness = make_float2(glossy.alpha);
            candidate_valid = glossy.valid;
            candidate_glossy_singular = glossy.singular;
            candidate_glossy_singular_evaluation =
                glossy.singular_evaluation;
            candidate_glossy_singular_pdf =
                glossy.singular_pdf;
        }
        $else {
            const auto normal = select(
                closure.normal,
                -selection.glossy_normal,
                is_translucent);
            candidate_direction =
                detail::sample_cosine_hemisphere(
                    normal, u_direction);
        };

        result.wi = candidate_direction;
        result.roughness = candidate_roughness;
        if (trace_selection) {
            trace.closure_index = selected_closure_index;
            trace.closure_type =
                detail::cycles_closure_type(closure);
            trace.closure_sample_weight =
                detail::closure_sample_weight(closure);
            trace.selection_rescaled = selected_rescaled;
            trace.closure_weight = closure.weight;
            trace.closure_normal = closure.normal;
        }
    };

    const auto selected_transparent =
        selected & is_transparent;
    const auto selected_translucent =
        selected & is_translucent;
    const auto selected_glossy = selected & sample_glossy;
    const auto selected_glossy_singular =
        selected_glossy & candidate_glossy_singular;
    const auto selected_glass = selected & is_glass;
    const auto selected_glass_transmission =
        selected_glass & candidate_glass_transmission;
    const auto selected_glass_singular =
        selected_glass & candidate_glass_singular;
    const auto selected_candidate_valid =
        selected & candidate_valid;
    const auto transparent_weight = select(
        make_float3(0.0f),
        closure.weight,
        selected_transparent);
    const auto transparent_sample_weight = select(
        0.0f, selected_weight, selected_transparent);
    const auto glossy_singular_evaluation =
        candidate_glossy_singular_evaluation;
    const auto glossy_singular_pdf =
        candidate_glossy_singular_pdf;
    const auto glossy_sample_weight = select(
        0.0f, selected_weight, selected_glossy);
    const auto glass_singular_evaluation =
        candidate_glass_singular_evaluation;
    const auto glass_singular_pdf =
        candidate_glass_singular_pdf;
    const auto glass_sample_weight = select(
        0.0f, selected_weight, selected_glass);
    const auto glass_eta = candidate_glass_eta;

    const auto reflection_geometric_valid =
        dot(_point.geometric_normal, result.wi) > 0.0f;
    const auto transmission_geometric_valid =
        dot(_point.geometric_normal, result.wi) < 0.0f;
    const auto geometric_valid = select(
        reflection_geometric_valid,
        transmission_geometric_valid,
        selected_translucent | selected_glass_transmission);
    const auto sample_valid =
        selected & selected_candidate_valid &
        (selected_transparent | geometric_valid);
    auto diffuse_evaluation = evaluate_impl(
        services,
        result.wi,
        query,
        EvaluationMode::sampled_bsdf,
        0u,
        selected_closure_index);
    // Cycles skips the complete multi-closure MIS evaluation when the
    // selected BSDF returns a zero PDF. Keep invalid payloads observationally
    // zero instead of relying on downstream code to ignore result.valid.
    diffuse_evaluation.f = select(
        make_float3(0.0f), diffuse_evaluation.f, sample_valid);
    diffuse_evaluation.pdf = select(
        0.0f, diffuse_evaluation.pdf, sample_valid);
    diffuse_evaluation.diffuse_f = select(
        make_float3(0.0f),
        diffuse_evaluation.diffuse_f,
        sample_valid);
    diffuse_evaluation.glossy_f = select(
        make_float3(0.0f),
        diffuse_evaluation.glossy_f,
        sample_valid);
    diffuse_evaluation.diffuse_pdf = select(
        0.0f, diffuse_evaluation.diffuse_pdf, sample_valid);
    diffuse_evaluation.events = select(
        0u, diffuse_evaluation.events, sample_valid);
    result.valid = sample_valid;

    const auto transparent_singular =
        sample_valid & selected_transparent;
    const auto glossy_singular =
        sample_valid & selected_glossy &
        selected_glossy_singular;
    const auto glass_singular =
        sample_valid & selected_glass &
        selected_glass_singular;
    const auto transparent_delta = select(
        make_float3(0.0f),
        transparent_weight * 1.0e6f,
        transparent_singular);
    const auto glossy_delta = select(
        make_float3(0.0f),
        glossy_singular_evaluation,
        glossy_singular);
    const auto glass_delta = select(
        make_float3(0.0f),
        glass_singular_evaluation,
        glass_singular);
    const auto delta_pdf_numerator =
        select(0.0f,
            1.0e6f * transparent_sample_weight,
            transparent_singular) +
        select(0.0f,
            glossy_singular_pdf * glossy_sample_weight,
            glossy_singular) +
        select(0.0f,
            glass_singular_pdf * glass_sample_weight,
            glass_singular);
    result.evaluation.f =
        diffuse_evaluation.f + transparent_delta +
        glossy_delta + glass_delta;
    result.evaluation.pdf =
        diffuse_evaluation.pdf +
        delta_pdf_numerator /
            max(total_weight, 1.0e-20f);
    result.evaluation.diffuse_f =
        diffuse_evaluation.diffuse_f;
    result.evaluation.glossy_f =
        diffuse_evaluation.glossy_f + glossy_delta +
        select(glass_delta,
            make_float3(0.0f),
            selected_glass_transmission);
    result.evaluation.diffuse_pdf =
        diffuse_evaluation.diffuse_pdf;

    auto sampled_surface_events = select(
        static_cast<std::uint32_t>(
            event_diffuse | event_reflection),
        static_cast<std::uint32_t>(
            event_glossy | event_reflection),
        selected_glossy);
    sampled_surface_events = select(sampled_surface_events,
        static_cast<std::uint32_t>(
            event_diffuse | event_transmission),
        selected_translucent);
    sampled_surface_events = select(sampled_surface_events,
        static_cast<std::uint32_t>(
            event_singular | event_reflection),
        selected_glossy & selected_glossy_singular);
    auto glass_events = select(
        static_cast<std::uint32_t>(event_glossy),
        static_cast<std::uint32_t>(event_singular),
        selected_glass_singular);
    glass_events |= select(
        static_cast<std::uint32_t>(event_reflection),
        static_cast<std::uint32_t>(event_transmission),
        selected_glass_transmission);
    sampled_surface_events = select(
        sampled_surface_events,
        glass_events,
        selected_glass);
    result.evaluation.events = select(
        sampled_surface_events,
        static_cast<std::uint32_t>(
            event_transmission | event_transparent),
        selected_transparent);
    result.evaluation.events = select(
        0u, result.evaluation.events, sample_valid);
    result.eta = select(
        1.0f, glass_eta, selected_glass & sample_valid);
    result.roughness = select(
        make_float2(0.0f), result.roughness, sample_valid);
    if (trace_selection) {
        trace.closure_valid = selected;
    }
    return trace;
}

SurfaceSample SurfaceClosureEvaluator::sample(
    const ShaderServices &services,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return sample_impl(services,
        Float{u_lobe},
        Float2{u_direction},
        query,
        false)
        .sample;
}

SurfaceSampleTrace SurfaceClosureEvaluator::sample_trace(
    const ShaderServices &services,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return sample_impl(services,
        Float{u_lobe},
        Float2{u_direction},
        query,
        true);
}

}// namespace psycles::luisa_backend
