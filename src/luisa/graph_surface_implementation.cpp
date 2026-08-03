#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"
#include "principled_layer_component.h"

#include <utility>

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_voronoi.h>
#include <psycles/luisa/cycles_wave.h>
#include <psycles/luisa/cycles_volume.h>
#include <psycles/luisa/surface_closure_sampling.h>

namespace psycles::luisa_backend::detail {

GraphSurfaceImplementation::GraphSurfaceImplementation(
    std::shared_ptr<const compiler::SurfaceProgram> program) noexcept
    : _program{std::move(program)} {
    if (_program) {
        for (const auto &instruction : _program->value_instructions()) {
            _value_nodes.emplace_back(make_value_node(instruction));
            if (instruction.operation ==
                    compiler::ValueOperation::noise_factor ||
                instruction.operation ==
                    compiler::ValueOperation::noise_color) {
                const auto color_needed =
                    instruction.operation ==
                    compiler::ValueOperation::noise_color;
                const auto normalize =
                    (instruction.static_u1 & 1u) != 0u;
                const auto noise_type = static_cast<cycles_noise::Type>(
                    (instruction.static_u1 >> 8u) & 0xffu);
                cycles_noise::prepare_texture(
                    static_cast<std::uint32_t>(instruction.static_u0),
                    noise_type,
                    normalize,
                    color_needed);
            } else if (
                instruction.operation ==
                    compiler::ValueOperation::white_noise_value ||
                instruction.operation ==
                    compiler::ValueOperation::white_noise_color) {
                cycles_noise::prepare_white_texture(
                    static_cast<std::uint32_t>(instruction.static_u0),
                    instruction.operation ==
                        compiler::ValueOperation::white_noise_color);
            } else if (
                instruction.operation ==
                    compiler::ValueOperation::wave_color ||
                instruction.operation ==
                    compiler::ValueOperation::wave_factor) {
                cycles_wave::prepare_distortion_noise();
            } else if (cycles_voronoi::is_operation(
                           instruction.operation)) {
                cycles_voronoi::prepare(
                    cycles_voronoi::decode_configuration(instruction));
            }
        }
        for (const auto &closure : _program->closure_instructions()) {
            _capabilities.may_emit |=
                closure.operation ==
                    compiler::ClosureOperation::emission ||
                closure.operation ==
                    compiler::ClosureOperation::principled;
            _capabilities.may_be_transparent |=
                closure.operation ==
                    compiler::ClosureOperation::transparent ||
                closure.operation ==
                    compiler::ClosureOperation::principled;
            _capabilities.may_have_subsurface |=
                closure.operation ==
                compiler::ClosureOperation::principled;
        }
        _capabilities.emission_is_constant =
            _program->emission_evaluation() !=
            compiler::EmissionEvaluationMode::deferred;
        _capabilities.may_have_volume = _program->volume_root().valid();
    }
}

GraphSurfaceImplementation::~GraphSurfaceImplementation() noexcept =
    default;

[[nodiscard]] SurfaceCapabilities
GraphSurfaceImplementation::capabilities() const noexcept {
    return _capabilities;
}

[[nodiscard]] SurfaceEvaluation
GraphSurfaceImplementation::evaluate_traced(
    const ShaderServices &services,
    const TracedValues &values,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceQuery &query,
    const EvaluationContext &context) const noexcept {
    auto result = SurfaceEvaluation::zero();
    Float total_sample_weight = 0.0f;
    Float weighted_pdf = 0.0f;
    UInt closure_index = 0u;
    auto outgoing = safe_normalize(
        Float3{outgoing_expression}, point.shading_normal);
    auto incoming = safe_normalize(point.incoming, -outgoing);
    const MicrofacetGlassComponent microfacet_glass{services, point};
    auto light_shader_flags = UInt{context.light_shader_flags};
    auto selected_closure_index = UInt{context.selected_closure_index};
    const auto sampled_light =
        context.mode == EvaluationMode::sampled_light;
    const auto sampled_bsdf =
        context.mode == EvaluationMode::sampled_bsdf;
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
        // Cycles treats a glass closure as a hybrid rather than assigning
        // each sampled direction to one pure light-visibility class.
        light_glass_included =
            !(excludes_glossy & excludes_transmit);
    }
    auto diffuse_enabled =
        (query.lobe_mask & static_cast<std::uint32_t>(event_diffuse)) !=
        0u;
    auto glossy_enabled = (query.lobe_mask & static_cast<std::uint32_t>(
                                                 event_glossy)) != 0u;
    auto transparent_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transparent)) != 0u;
    auto transmission_enabled =
        (query.lobe_mask &
            static_cast<std::uint32_t>(event_transmission)) != 0u;
    Bool has_diffuse = false;
    Bool has_translucent = false;
    Bool has_glossy = false;
    Bool has_glass_reflection = false;
    Bool has_glass_transmission = false;

    for_each_physical_closure(services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            const auto physical =
                canonical_surface_closure(closure);
            const auto allocated = closure_allocated(physical);
            const auto current_closure_index = closure_index;
            closure_index += select(0u, 1u, allocated);
            Bool selected_sample = false;
            if (sampled_bsdf) {
                selected_sample =
                    allocated &
                    (current_closure_index == selected_closure_index);
            }
            if (closure.operation ==
                compiler::ClosureOperation::transparent) {
                auto weight = closure_sample_weight(physical);
                total_sample_weight +=
                    select(0.0f, weight, transparent_enabled);
                return;
            }
            const auto is_diffuse = closure.operation ==
                                    compiler::ClosureOperation::diffuse;
            const auto is_translucent =
                closure.operation ==
                compiler::ClosureOperation::translucent;
            const auto is_principled =
                closure.operation ==
                compiler::ClosureOperation::principled;
            const auto is_sheen =
                is_principled &&
                closure.principled_lobe == PrincipledLobe::sheen;
            const auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            const auto is_glass =
                closure.operation == compiler::ClosureOperation::glass;
            if (!is_diffuse && !is_translucent && !is_principled &&
                !is_glossy && !is_glass) {
                return;
            }
            auto glossy_normal = is_sheen
                                     ? closure.normal
                                     : maybe_ensure_valid_specular_reflection(
                                           point, incoming, closure.normal);
            auto glass_is_transmission =
                dot(glossy_normal, outgoing) < 0.0f;
            // Cycles promotes near-unit-IOR transmission samples to delta
            // even when their microfacet distribution is otherwise rough.
            // Such a selected closure is initialized from bsdf_sample() and
            // must not also be accumulated through its regular bsdf_eval().
            const auto selected_unit_ior_glass_delta =
                selected_sample & Bool{is_glass} &
                glass_is_transmission &
                (abs(closure.ior - 1.0f) < 1.0e-4f);
            const auto bump_shadowing = bump_shadowing_term(point,
                values.shading_normal,
                physical,
                outgoing,
                !selected_sample);
            const auto bump_direction_valid = bump_shadowing != 0.0f;
            // Cycles initializes the selected closure from bsdf_sample(),
            // whose bump term scales eval but never its sampling PDF. Every
            // competing closure is accumulated through bsdf_eval(), which
            // rejects both eval and PDF when its bump term is zero.
            const auto bump_pdf_valid =
                bump_direction_valid | selected_sample;
            auto diffuse_normal =
                is_translucent ? -glossy_normal : closure.normal;
            auto diffuse_pdf = is_sheen
                                   ? sheen_intensity(
                                         physical, incoming, outgoing)
                                   : max(dot(diffuse_normal, outgoing),
                                         0.0f) *
                                         inverse_pi;
            diffuse_pdf = select(
                0.0f, diffuse_pdf, bump_pdf_valid);
            Float glossy_pdf = 0.0f;
            if (!is_sheen) {
                glossy_pdf = is_glass
                                 ? microfacet_glass.pdf(physical,
                                       incoming,
                                       outgoing,
                                       glossy_normal,
                                       glossy_enabled,
                                       transmission_enabled,
                                       query.glossy_filter_roughness)
                                 : microfacet_pdf(physical,
                                       incoming,
                                       outgoing,
                                       glossy_normal,
                                       query.glossy_filter_roughness);
            }
            glossy_pdf = select(
                0.0f, glossy_pdf, bump_pdf_valid);
            glossy_pdf = select(
                glossy_pdf, 0.0f, selected_unit_ior_glass_delta);
            auto translucent_allowed =
                diffuse_enabled & transmission_enabled & is_translucent;
            auto diffuse_allowed =
                (diffuse_enabled & (is_diffuse || is_sheen)) |
                translucent_allowed;
            auto glossy_allowed =
                glossy_enabled &
                ((is_principled && !is_sheen) || is_glossy);
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
            Float3 diffuse_contribution;
            Float3 glossy_contribution;
            Float3 glass_contribution = make_float3(0.0f);
            if (is_glass) {
                diffuse_contribution = make_float3(0.0f);
                glossy_contribution = make_float3(0.0f);
                glass_contribution =
                    closure.weight *
                    microfacet_glass.intensity(physical,
                        incoming,
                        outgoing,
                        glossy_normal,
                        query.glossy_filter_roughness) *
                    bump_shadowing;
                glass_contribution = select(glass_contribution,
                    make_float3(0.0f),
                    selected_unit_ior_glass_delta);
            } else if (is_sheen) {
                diffuse_contribution =
                    closure.weight * diffuse_pdf * bump_shadowing;
                glossy_contribution = make_float3(0.0f);
            } else if (is_principled || is_glossy) {
                diffuse_contribution = make_float3(0.0f);
                glossy_contribution =
                    closure.weight * microfacet_intensity(services,
                                         physical,
                                         incoming,
                                         outgoing,
                                         glossy_normal,
                                         query.glossy_filter_roughness) *
                    bump_shadowing;
            } else if (is_translucent) {
                auto cosine = max(dot(-glossy_normal, outgoing), 0.0f) *
                              inverse_pi;
                diffuse_contribution =
                    closure.weight * cosine * bump_shadowing;
                glossy_contribution = make_float3(0.0f);
            } else {
                diffuse_contribution =
                    closure.weight *
                    diffuse_intensity(physical, incoming, outgoing) *
                    bump_shadowing;
                glossy_contribution = make_float3(0.0f);
            }
            auto pdf = is_glass
                           ? glossy_pdf
                           : select(diffuse_pdf,
                                 glossy_pdf,
                                 glossy_allowed);
            auto enabled_pdf =
                select(0.0f,
                    pdf,
                    diffuse_allowed | glossy_allowed | glass_allowed);
            const auto eligible_diffuse =
                select(make_float3(0.0f),
                    diffuse_contribution,
                    diffuse_contributes);
            const auto eligible_glossy =
                select(make_float3(0.0f),
                    glossy_contribution,
                    glossy_contributes);
            const auto eligible_glass =
                select(make_float3(0.0f),
                    glass_contribution,
                    glass_contributes);
            const auto eligible_glass_reflection =
                select(make_float3(0.0f),
                    glass_contribution,
                    glass_contributes & (!glass_is_transmission));
            // Light visibility is a linear projection of the evaluated
            // closure vector. Distributing it through the sum preserves the
            // projection exactly while closure weights and PDFs below remain
            // functions of the unprojected eligibility predicates.
            result.f +=
                eligible_diffuse +
                eligible_glossy +
                eligible_glass;
            result.diffuse_f += eligible_diffuse;
            result.glossy_f +=
                eligible_glossy +
                eligible_glass_reflection;
            auto weight = closure_sample_weight(physical);
            weight =
                select(0.0f,
                    weight,
                    diffuse_allowed | glossy_allowed | glass_allowed);
            total_sample_weight += weight;
            weighted_pdf += weight * enabled_pdf;
            has_diffuse =
                has_diffuse |
                ((diffuse_contributes & (!is_translucent)) &
                    (sample_weight(diffuse_contribution) > 0.0f));
            has_translucent =
                has_translucent |
                ((translucent_allowed & light_diffuse_included) &
                    (sample_weight(diffuse_contribution) > 0.0f));
            has_glossy =
                has_glossy |
                (glossy_contributes &
                    (sample_weight(glossy_contribution) > 0.0f));
            has_glass_reflection =
                has_glass_reflection |
                (glass_contributes & (!glass_is_transmission) &
                    (sample_weight(glass_contribution) > 0.0f));
            has_glass_transmission =
                has_glass_transmission |
                (glass_contributes & glass_is_transmission &
                    (sample_weight(glass_contribution) > 0.0f));
        });

    auto has_pdf = total_sample_weight > 0.0f;
    result.pdf = select(0.0f,
        weighted_pdf / max(total_sample_weight, 1.0e-20f),
        has_pdf);
    result.diffuse_pdf =
        select(0.0f, result.pdf, has_diffuse | has_translucent);
    auto has_diffuse_pdf = weighted_pdf > 0.0f;
    UInt events = static_cast<std::uint32_t>(event_none);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_diffuse | event_reflection),
        has_diffuse);
    events = select(events,
        events |
            static_cast<std::uint32_t>(event_glossy | event_reflection),
        has_glossy);
    events = select(events,
        events | static_cast<std::uint32_t>(
                     event_diffuse | event_transmission),
        has_translucent);
    events = select(events,
        events |
            static_cast<std::uint32_t>(event_glossy | event_reflection),
        has_glass_reflection);
    events = select(events,
        events |
            static_cast<std::uint32_t>(event_glossy | event_transmission),
        has_glass_transmission);
    result.events = select(static_cast<std::uint32_t>(event_none),
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

[[nodiscard]] SurfaceEvaluation GraphSurfaceImplementation::evaluate(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceQuery &query) const noexcept {
    if (!_program) {
        return SurfaceEvaluation::zero();
    }
    auto values = trace_values(services, point);
    return evaluate_traced(
        services,
        values,
        point,
        outgoing_expression,
        query,
        {.mode = EvaluationMode::regular,
            .light_shader_flags = 0u,
            .selected_closure_index = ~std::uint32_t{0u}});
}

[[nodiscard]] SurfaceEvaluation
GraphSurfaceImplementation::evaluate_light(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceLightQuery &query) const noexcept {
    if (!_program) {
        return SurfaceEvaluation::zero();
    }
    auto values = trace_values(services, point);
    return evaluate_traced(services,
        values,
        point,
        outgoing_expression,
        query.surface,
        {.mode = EvaluationMode::sampled_light,
            .light_shader_flags = query.shader_flags,
            .selected_closure_index = ~std::uint32_t{0u}});
}

[[nodiscard]] UInt GraphSurfaceImplementation::runtime_flags(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness_expression,
    Expr<bool> reflective_caustics_expression,
    Expr<bool> refractive_caustics_expression) const noexcept {
    if (!_program) {
        return 0u;
    }
    auto values = trace_values(services, point);
    auto glossy_filter_roughness =
        Float{glossy_filter_roughness_expression};
    auto reflective_caustics = Bool{reflective_caustics_expression};
    auto refractive_caustics = Bool{refractive_caustics_expression};
    UInt result = select(
        0u,
        cycles_closure::runtime_backfacing,
        point.back_facing);
    for_each_physical_closure(
        services,
        point,
        values,
        reflective_caustics,
        refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            result |= cycles_runtime_flags(
                canonical_surface_closure(closure),
                glossy_filter_roughness);
        });
    return result;
}

[[nodiscard]] SurfaceClosureTrace
GraphSurfaceImplementation::closure_trace(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<std::uint32_t> requested_index_expression,
    Expr<bool> reflective_caustics_expression,
    Expr<bool> refractive_caustics_expression) const noexcept {
    auto requested_index = UInt{requested_index_expression};
    auto reflective_caustics = Bool{reflective_caustics_expression};
    auto refractive_caustics = Bool{refractive_caustics_expression};
    auto result = SurfaceClosureTrace::zero(requested_index);
    if (!_program) {
        return result;
    }
    auto values = trace_values(services, point);
    UInt closure_count = 0u;
    UInt runtime_flags = select(
        0u, cycles_closure::runtime_backfacing, point.back_facing);
    for_each_physical_closure(services,
        point,
        values,
        reflective_caustics,
        refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            const auto physical =
                canonical_surface_closure(closure);
            runtime_flags |= cycles_runtime_flags(physical);
            auto allocated = closure_allocated(physical);
            auto match = allocated & (closure_count == requested_index);
            result.type = select(
                result.type, cycles_closure_type(physical), match);
            result.sample_weight = select(result.sample_weight,
                closure_sample_weight(physical),
                match);
            result.weight =
                select(result.weight, closure.weight, match);
            result.normal =
                select(result.normal, closure.normal, match);
            result.valid = result.valid | match;
            closure_count += select(0u, 1u, allocated);
        });
    result.count = closure_count;
    result.runtime_flags = runtime_flags;
    return result;
}

[[nodiscard]] SurfaceSampleTrace
GraphSurfaceImplementation::sample_with_trace(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> u_lobe_expression,
    Expr<luisa::float2> u_direction_expression,
    const SurfaceQuery &query,
    bool trace_selection) const noexcept {
    auto trace = SurfaceSampleTrace::zero();
    auto &result = trace.sample;
    if (!_program) {
        return trace;
    }

    auto values = trace_values(services, point);
    auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    const MicrofacetGlassComponent microfacet_glass{services, point};
    Float total_weight = 0.0f;
    UInt closure_count = 0u;
    UInt surface_runtime_flags = select(
        0u, cycles_closure::runtime_backfacing, point.back_facing);
    for_each_physical_closure(services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            const auto physical =
                canonical_surface_closure(closure);
            surface_runtime_flags |= cycles_runtime_flags(
                physical, query.glossy_filter_roughness);
            const auto selection = surface_closure_selection(
                point,
                physical,
                Expr<luisa::float3>{incoming.expression()},
                query);
            total_weight += selection.weight;
            closure_count += select(
                0u, 1u, closure_allocated(physical));
        });
    result.runtime_flags = surface_runtime_flags;

    auto random_lobe =
        clamp(Float{u_lobe_expression}, 0.0f, 0.99999994f);
    auto target = random_lobe * total_weight;
    auto random_direction = Float2{u_direction_expression};
    Float accumulated = 0.0f;
    Bool selected = false;
    Bool selected_transparent = false;
    Bool selected_translucent = false;
    Bool selected_glossy = false;
    Bool selected_glossy_singular = false;
    Bool selected_glass = false;
    Bool selected_glass_transmission = false;
    Bool selected_glass_singular = false;
    Bool selected_candidate_valid = true;
    UInt selected_closure_index = ~std::uint32_t{0u};
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    Float3 glossy_singular_evaluation = make_float3(0.0f);
    Float glossy_singular_pdf = 0.0f;
    Float glossy_sample_weight = 0.0f;
    Float3 glass_singular_evaluation = make_float3(0.0f);
    Float glass_singular_pdf = 0.0f;
    Float glass_sample_weight = 0.0f;
    Float glass_eta = 1.0f;
    UInt closure_index = 0u;

    for_each_physical_closure(services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            const auto physical =
                canonical_surface_closure(closure);
            auto is_translucent =
                closure.operation ==
                compiler::ClosureOperation::translucent;
            auto is_principled = closure.operation ==
                                 compiler::ClosureOperation::principled;
            auto is_sheen =
                is_principled &&
                closure.principled_lobe == PrincipledLobe::sheen;
            auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            auto is_transparent =
                closure.operation ==
                compiler::ClosureOperation::transparent;
            auto is_glass = closure.operation ==
                            compiler::ClosureOperation::glass;
            const auto allocated = closure_allocated(physical);
            const auto current_closure_index = closure_index;
            const auto selection = surface_closure_selection(
                point,
                physical,
                Expr<luisa::float3>{incoming.expression()},
                query);
            const auto glossy_normal = Float3{
                selection.glossy_normal.expression()};
            const auto weight = selection.weight;
            auto next = accumulated + weight;
            auto choose =
                (!selected) & (weight > 0.0f) & (target < next);
            const auto rescaled_selection = select(random_lobe,
                (target - accumulated) / max(weight, 1.0e-20f),
                closure_count > 1u);
            auto diffuse_direction = sample_cosine_hemisphere(
                is_translucent ? -glossy_normal : closure.normal,
                random_direction);
            auto glossy = sample_microfacet_reflection(point,
                values.shading_normal,
                physical,
                incoming,
                random_direction,
                glossy_normal,
                query.glossy_filter_roughness);
            auto sheen_direction = is_sheen
                                       ? sample_sheen(physical,
                                             incoming,
                                             random_direction)
                                       : make_float3(0.0f, 0.0f, 1.0f);
            auto transparent_direction = -point.incoming;
            const auto sample_glossy =
                is_glossy || (is_principled && !is_sheen);
            auto candidate_direction = is_transparent
                                           ? transparent_direction
                                       : is_sheen ? sheen_direction
                                                  : select(diffuse_direction,
                                                        glossy.direction,
                                                        sample_glossy);
            Bool candidate_valid = true;
            Bool glass_transmission = false;
            Bool glass_singular = false;
            Float3 candidate_glass_singular_evaluation =
                make_float3(0.0f);
            Float candidate_glass_singular_pdf = 0.0f;
            Float candidate_glass_eta = 1.0f;
            Float candidate_glass_alpha = 0.0f;
            if (sample_glossy) {
                candidate_valid = glossy.valid;
            }
            if (is_glass) {
                const auto glass = microfacet_glass.sample(physical,
                    incoming,
                    glossy_normal,
                    random_direction,
                    rescaled_selection,
                    (query.lobe_mask &
                        static_cast<std::uint32_t>(event_glossy)) != 0u,
                    (query.lobe_mask &
                        static_cast<std::uint32_t>(event_transmission)) !=
                        0u,
                    query.glossy_filter_roughness);
                candidate_direction = glass.direction;
                candidate_valid = glass.valid;
                glass_transmission = glass.transmission;
                glass_singular = glass.singular;
                candidate_glass_singular_evaluation =
                    glass.singular_evaluation;
                candidate_glass_singular_pdf = glass.singular_pdf;
                candidate_glass_eta = glass.eta;
                candidate_glass_alpha = glass.alpha;
            }
            result.wi = select(result.wi, candidate_direction, choose);
            selected_closure_index = select(selected_closure_index,
                current_closure_index,
                choose);
            auto nontransparent_roughness = select(make_float2(1.0f),
                make_float2(glossy.alpha),
                sample_glossy);
            if (is_sheen) {
                nontransparent_roughness = make_float2(1.0f);
            }
            if (is_glass) {
                nontransparent_roughness =
                    make_float2(candidate_glass_alpha);
            }
            result.roughness = select(result.roughness,
                is_transparent ? make_float2(0.0f)
                               : nontransparent_roughness,
                choose);
            if (trace_selection) {
                trace.closure_index = select(
                    trace.closure_index, current_closure_index, choose);
                trace.closure_type = select(trace.closure_type,
                    cycles_closure_type(physical),
                    choose);
                trace.closure_sample_weight =
                    select(trace.closure_sample_weight,
                        closure_sample_weight(physical),
                        choose);
                trace.selection_rescaled =
                    select(trace.selection_rescaled,
                        rescaled_selection,
                        choose);
                trace.closure_weight = select(
                    trace.closure_weight, closure.weight, choose);
                trace.closure_normal = select(
                    trace.closure_normal, closure.normal, choose);
            }
            transparent_weight = select(transparent_weight,
                closure.weight,
                is_transparent ? choose : Bool{false});
            transparent_sample_weight =
                select(transparent_sample_weight,
                    weight,
                    is_transparent ? choose : Bool{false});
            selected_transparent =
                selected_transparent |
                (is_transparent ? choose : Bool{false});
            selected_translucent =
                selected_translucent |
                (is_translucent ? choose : Bool{false});
            selected_glossy =
                selected_glossy |
                ((!is_transparent && !is_glass && sample_glossy)
                        ? choose
                        : Bool{false});
            selected_glossy_singular =
                selected_glossy_singular |
                ((!is_transparent && !is_glass && sample_glossy)
                        ? (choose & glossy.singular)
                        : Bool{false});
            selected_glass =
                selected_glass | (is_glass ? choose : Bool{false});
            selected_glass_transmission =
                selected_glass_transmission |
                (is_glass ? (choose & glass_transmission) : Bool{false});
            selected_glass_singular =
                selected_glass_singular |
                (is_glass ? (choose & glass_singular) : Bool{false});
            selected_candidate_valid = select(selected_candidate_valid,
                candidate_valid,
                choose);
            glossy_singular_evaluation = select(
                glossy_singular_evaluation,
                glossy.singular_evaluation,
                (!is_transparent && !is_glass && sample_glossy)
                    ? choose
                    : Bool{false});
            glossy_singular_pdf = select(glossy_singular_pdf,
                glossy.singular_pdf,
                (!is_transparent && !is_glass && sample_glossy)
                    ? choose
                    : Bool{false});
            glossy_sample_weight = select(glossy_sample_weight,
                weight,
                (!is_transparent && !is_glass && sample_glossy)
                    ? choose
                    : Bool{false});
            glass_singular_evaluation = select(
                glass_singular_evaluation,
                candidate_glass_singular_evaluation,
                is_glass ? choose : Bool{false});
            glass_singular_pdf = select(glass_singular_pdf,
                candidate_glass_singular_pdf,
                is_glass ? choose : Bool{false});
            glass_sample_weight = select(glass_sample_weight,
                weight,
                is_glass ? choose : Bool{false});
            glass_eta = select(glass_eta,
                candidate_glass_eta,
                is_glass ? choose : Bool{false});
            selected = selected | choose;
            accumulated = next;
            closure_index += select(0u, 1u, allocated);
        });

    auto reflection_geometric_valid =
        dot(point.geometric_normal, result.wi) > 0.0f;
    auto transmission_geometric_valid =
        dot(point.geometric_normal, result.wi) < 0.0f;
    auto geometric_valid = select(reflection_geometric_valid,
        transmission_geometric_valid,
        selected_translucent | selected_glass_transmission);
    auto sample_valid = selected & selected_candidate_valid &
                        (selected_transparent | geometric_valid);
    auto diffuse_evaluation =
        evaluate_traced(services,
            values,
            point,
            result.wi,
            query,
            {.mode = EvaluationMode::sampled_bsdf,
                .light_shader_flags = 0u,
                .selected_closure_index = selected_closure_index});
    // Cycles skips the complete multi-closure MIS evaluation when the
    // selected BSDF returns a zero PDF (for example a sampled reflection
    // below Ng). Keep invalid sample payloads observationally zero instead
    // of relying on downstream code to ignore result.valid.
    diffuse_evaluation.f = select(
        make_float3(0.0f), diffuse_evaluation.f, sample_valid);
    diffuse_evaluation.pdf = select(
        0.0f, diffuse_evaluation.pdf, sample_valid);
    diffuse_evaluation.diffuse_f = select(make_float3(0.0f),
        diffuse_evaluation.diffuse_f,
        sample_valid);
    diffuse_evaluation.glossy_f = select(make_float3(0.0f),
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
        sample_valid & selected_glossy & selected_glossy_singular;
    const auto glass_singular =
        sample_valid & selected_glass & selected_glass_singular;
    const auto transparent_delta = select(make_float3(0.0f),
        transparent_weight * 1.0e6f,
        transparent_singular);
    const auto glossy_delta = select(make_float3(0.0f),
        glossy_singular_evaluation,
        glossy_singular);
    const auto glass_delta = select(make_float3(0.0f),
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
    result.evaluation.f = diffuse_evaluation.f + transparent_delta +
                          glossy_delta + glass_delta;
    result.evaluation.pdf =
        diffuse_evaluation.pdf +
        delta_pdf_numerator / max(total_weight, 1.0e-20f);
    result.evaluation.diffuse_f = diffuse_evaluation.diffuse_f;
    result.evaluation.glossy_f =
        diffuse_evaluation.glossy_f + glossy_delta +
        select(glass_delta,
            make_float3(0.0f),
            selected_glass_transmission);
    result.evaluation.diffuse_pdf = diffuse_evaluation.diffuse_pdf;
    auto sampled_surface_events = select(
        static_cast<std::uint32_t>(event_diffuse | event_reflection),
        static_cast<std::uint32_t>(event_glossy | event_reflection),
        selected_glossy);
    sampled_surface_events = select(sampled_surface_events,
        static_cast<std::uint32_t>(event_diffuse | event_transmission),
        selected_translucent);
    sampled_surface_events = select(sampled_surface_events,
        static_cast<std::uint32_t>(event_singular | event_reflection),
        selected_glossy & selected_glossy_singular);
    auto glass_events = select(
        static_cast<std::uint32_t>(event_glossy),
        static_cast<std::uint32_t>(event_singular),
        selected_glass_singular);
    glass_events |= select(
        static_cast<std::uint32_t>(event_reflection),
        static_cast<std::uint32_t>(event_transmission),
        selected_glass_transmission);
    sampled_surface_events = select(sampled_surface_events,
        glass_events,
        selected_glass);
    result.evaluation.events = select(sampled_surface_events,
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

[[nodiscard]] SurfaceSample GraphSurfaceImplementation::sample(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> u_lobe_expression,
    Expr<luisa::float2> u_direction_expression,
    const SurfaceQuery &query) const noexcept {
    return sample_with_trace(services,
        point,
        u_lobe_expression,
        u_direction_expression,
        query,
        false)
        .sample;
}

[[nodiscard]] SurfaceSampleTrace
GraphSurfaceImplementation::sample_trace(const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> u_lobe_expression,
    Expr<luisa::float2> u_direction_expression,
    const SurfaceQuery &query) const noexcept {
    return sample_with_trace(services,
        point,
        u_lobe_expression,
        u_direction_expression,
        query,
        true);
}

[[nodiscard]] Float3 GraphSurfaceImplementation::transparent_extinction(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    if (!_program) {
        return make_float3(0.0f);
    }
    auto values = trace_values(services, point);
    Float3 result = make_float3(0.0f);
    for_each_closure(
        values, [&](const TracedClosure &closure) noexcept {
            if (closure.operation ==
                compiler::ClosureOperation::transparent) {
                result += transparent_closure_state(
                              closure.weight)
                              .weight;
            } else if (closure.operation ==
                       compiler::ClosureOperation::principled) {
                result += evaluate_principled_alpha_layer(
                              closure)
                              .transparency.weight;
            }
        });
    return result;
}

[[nodiscard]] VolumeCoefficients
GraphSurfaceImplementation::evaluate_volume(
    const ShaderServices &services,
    const SurfacePoint &point,
    const VolumeQuery &query,
    VolumePhaseCollector *collector) const noexcept {
    auto result = VolumeCoefficients::zero();
    if (!_program || !_program->volume_root().valid()) {
        return result;
    }
    const auto values = trace_values(services, point);
    for_each_volume(values,
        [&](const compiler::VolumeInstruction &volume,
            Float mix_weight) noexcept {
            const auto scalar_reader =
                [&](compiler::ValueExpressionId id) noexcept {
                    return scalar(id, values);
                };
            const auto vector_reader =
                [&](compiler::ValueExpressionId id) noexcept {
                    return vector(id, values);
                };
            const auto leaf = cycles_volume::evaluate_leaf(volume,
                mix_weight,
                services,
                point,
                query,
                scalar_reader,
                vector_reader);
            cycles_volume::accumulate_coefficients(leaf, result);
            if (collector != nullptr) {
                cycles_volume::emit_phase_closures(volume,
                    leaf,
                    scalar_reader,
                    [&](const cycles_volume_phase::Closure &phase,
                        Float3 weight) noexcept {
                        collector->add(phase, weight);
                    });
            }
        });
    return result;
}

[[nodiscard]] Float3 GraphSurfaceImplementation::shading_normal(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    if (!_program) {
        return point.shading_normal;
    }
    return trace_values(services, point).shading_normal;
}

[[nodiscard]] SurfaceAov GraphSurfaceImplementation::aov(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    auto result = SurfaceAov{.albedo = make_float3(0.0f),
        .glossy_albedo = make_float3(0.0f),
        .transmission_albedo = make_float3(0.0f),
        .roughness = make_float2(0.0f),
        .normal = point.shading_normal,
        .transparency = make_float3(0.0f)};
    if (!_program) {
        return result;
    }
    auto values = trace_values(services, point);
    Float total_weight = 0.0f;
    Float roughness_weight = 0.0f;
    Float roughness = 0.0f;
    Float3 normal = make_float3(0.0f);
    auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    for_each_physical_closure(services,
        point,
        values,
        true,
        true,
        [&](const TracedClosure &closure) noexcept {
            if (closure.operation ==
                compiler::ClosureOperation::transparent) {
                result.transparency += closure.weight;
                return;
            }
            const auto is_diffuse = closure.operation ==
                                    compiler::ClosureOperation::diffuse;
            const auto is_translucent =
                closure.operation ==
                compiler::ClosureOperation::translucent;
            const auto is_principled =
                closure.operation ==
                compiler::ClosureOperation::principled;
            const auto is_sheen =
                is_principled &&
                closure.principled_lobe == PrincipledLobe::sheen;
            const auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            const auto is_glass =
                closure.operation == compiler::ClosureOperation::glass;
            if (!is_diffuse && !is_translucent && !is_principled &&
                !is_glossy && !is_glass) {
                return;
            }
            auto glossy_normal = is_sheen
                                     ? closure.normal
                                     : maybe_ensure_valid_specular_reflection(
                                           point, incoming, closure.normal);
            Float3 diffuse_albedo = make_float3(0.0f);
            Float diffuse_weight = 0.0f;
            Float glossy_weight = 0.0f;
            if (is_glass) {
                result.glossy_albedo += closure.reflection_albedo;
                result.transmission_albedo += closure.transmission_albedo;
                glossy_weight = pass_weight(closure.weight);
            } else if (is_sheen) {
                diffuse_albedo = select(make_float3(0.0f),
                    closure.albedo,
                    closure.setup_valid);
                diffuse_weight = select(0.0f,
                    pass_weight(closure.weight),
                    closure.setup_valid);
            } else if (is_principled || is_glossy) {
                result.glossy_albedo += closure.albedo;
                glossy_weight = pass_weight(closure.weight);
            } else if (is_diffuse || is_translucent) {
                diffuse_albedo = closure.albedo;
                diffuse_weight = pass_weight(closure.weight);
            }
            const auto weight = diffuse_weight + glossy_weight;
            total_weight += weight;
            roughness_weight += glossy_weight;
            // Cycles Diffuse Color includes only diffuse/BSSRDF
            // closures. Glossy closure weights still contribute to the
            // Normal and Roughness passes, but never to Diffuse Color.
            result.albedo += diffuse_albedo;
            roughness += glossy_weight * closure.roughness;
            normal +=
                diffuse_weight *
                    (is_translucent ? glossy_normal : closure.normal) +
                glossy_weight * glossy_normal;
        });
    auto valid = total_weight > 0.0f;
    result.roughness = make_float2(
        select(1.0f,
            roughness / max(roughness_weight, 1.0e-20f),
            roughness_weight > 0.0f));
    result.normal =
        safe_normalize(select(point.shading_normal, normal, valid),
            point.shading_normal);
    return result;
}

} // namespace psycles::luisa_backend::detail
