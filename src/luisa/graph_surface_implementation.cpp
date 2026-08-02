#include "graph_surface_internal.h"

#include <utility>

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_volume.h>

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
            }
        }
        for (const auto &closure : _program->closure_instructions()) {
            _capabilities.may_emit |=
                closure.operation ==
                compiler::ClosureOperation::emission;
            _capabilities.may_be_transparent |=
                closure.operation ==
                compiler::ClosureOperation::transparent;
            _capabilities.may_have_subsurface |=
                closure.operation ==
                compiler::ClosureOperation::principled;
        }
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
    Expr<std::uint32_t> light_shader_flags_expression,
    bool sampled_light) const noexcept {
    auto result = SurfaceEvaluation::zero();
    Float total_sample_weight = 0.0f;
    Float weighted_pdf = 0.0f;
    auto outgoing = safe_normalize(
        Float3{outgoing_expression}, point.shading_normal);
    auto incoming = safe_normalize(point.incoming, -outgoing);
    auto light_shader_flags = UInt{light_shader_flags_expression};
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
        [&](const TracedClosure &closure) noexcept {
            if (closure.operation ==
                compiler::ClosureOperation::transparent) {
                auto weight = closure_sample_weight(closure);
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
            const auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            const auto is_glass =
                closure.operation == compiler::ClosureOperation::glass;
            if (!is_diffuse && !is_translucent && !is_principled &&
                !is_glossy && !is_glass) {
                return;
            }
            auto glossy_normal = ensure_valid_specular_reflection(
                point.geometric_normal, incoming, closure.normal);
            auto diffuse_normal =
                is_translucent ? -glossy_normal : closure.normal;
            auto diffuse_pdf =
                max(dot(diffuse_normal, outgoing), 0.0f) * inverse_pi;
            auto glossy_pdf = is_glass
                                  ? glass_microfacet_pdf(closure,
                                        incoming,
                                        outgoing,
                                        glossy_normal,
                                        glossy_enabled,
                                        transmission_enabled,
                                        query.glossy_filter_roughness)
                                  : microfacet_pdf(closure,
                                        incoming,
                                        outgoing,
                                        glossy_normal,
                                        query.glossy_filter_roughness);
            auto translucent_allowed =
                diffuse_enabled & transmission_enabled & is_translucent;
            auto diffuse_allowed =
                (diffuse_enabled & is_diffuse) | translucent_allowed;
            auto glossy_allowed =
                glossy_enabled & (is_principled || is_glossy);
            auto glass_is_transmission =
                dot(glossy_normal, outgoing) < 0.0f;
            auto glass_allowed =
                is_glass & select(glossy_enabled,
                               transmission_enabled,
                               glass_is_transmission);
            Bool contribution_excluded = false;
            if (sampled_light) {
                contribution_excluded =
                    sampled_light_excludes_closure(
                        closure, light_shader_flags);
            }
            const auto diffuse_contributes =
                diffuse_allowed & !contribution_excluded;
            const auto glossy_contributes =
                glossy_allowed & !contribution_excluded;
            const auto glass_contributes =
                glass_allowed & !contribution_excluded;
            Float3 diffuse_contribution;
            Float3 glossy_contribution;
            Float3 glass_contribution = make_float3(0.0f);
            if (is_glass) {
                diffuse_contribution = make_float3(0.0f);
                glossy_contribution = make_float3(0.0f);
                glass_contribution =
                    closure.weight *
                    glass_microfacet_intensity(closure,
                        incoming,
                        outgoing,
                        glossy_normal,
                        query.glossy_filter_roughness);
            } else if (is_principled || is_glossy) {
                diffuse_contribution = make_float3(0.0f);
                glossy_contribution =
                    closure.weight * microfacet_intensity(services,
                                         closure,
                                         incoming,
                                         outgoing,
                                         glossy_normal,
                                         query.glossy_filter_roughness);
            } else if (is_translucent) {
                auto cosine = max(dot(-glossy_normal, outgoing), 0.0f) *
                              inverse_pi;
                diffuse_contribution = closure.weight * cosine;
                glossy_contribution = make_float3(0.0f);
            } else {
                diffuse_contribution =
                    closure.weight *
                    diffuse_intensity(closure, incoming, outgoing);
                glossy_contribution = make_float3(0.0f);
            }
            auto pdf = is_glass
                           ? glossy_pdf
                           : select(diffuse_pdf,
                                 glossy_pdf,
                                 glossy_allowed);
            auto contribution = select(make_float3(0.0f),
                                    diffuse_contribution,
                                    diffuse_contributes) +
                                select(make_float3(0.0f),
                                    glossy_contribution,
                                    glossy_contributes) +
                                select(make_float3(0.0f),
                                    glass_contribution,
                                    glass_contributes);
            contribution = select(make_float3(0.0f),
                contribution,
                diffuse_contributes | glossy_contributes |
                    glass_contributes);
            auto enabled_pdf =
                select(0.0f,
                    pdf,
                    diffuse_allowed | glossy_allowed | glass_allowed);
            result.f += contribution;
            result.diffuse_f += select(make_float3(0.0f),
                diffuse_contribution,
                diffuse_contributes);
            result.glossy_f += select(make_float3(0.0f),
                                   glossy_contribution,
                                   glossy_contributes) +
                               select(make_float3(0.0f),
                                   glass_contribution,
                                   glass_contributes &
                                       (!glass_is_transmission));
            auto weight = closure_sample_weight(closure);
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
                ((translucent_allowed & !contribution_excluded) &
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
        services, values, point, outgoing_expression, query, 0u, false);
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
        query.shader_flags,
        true);
}

[[nodiscard]] SurfaceClosureTrace
GraphSurfaceImplementation::closure_trace(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<std::uint32_t> requested_index_expression) const noexcept {
    auto requested_index = UInt{requested_index_expression};
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
        [&](const TracedClosure &closure) noexcept {
            runtime_flags |= cycles_runtime_flags(closure);
            auto allocated = closure_allocated(closure);
            auto match = allocated & (closure_count == requested_index);
            result.type = select(
                result.type, cycles_closure_type(closure), match);
            result.sample_weight = select(result.sample_weight,
                closure_sample_weight(closure),
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
    Float total_weight = 0.0f;
    UInt closure_count = 0u;
    UInt surface_runtime_flags = select(
        0u, cycles_closure::runtime_backfacing, point.back_facing);
    for_each_physical_closure(services,
        point,
        values,
        [&](const TracedClosure &closure) noexcept {
            surface_runtime_flags |= cycles_runtime_flags(
                closure, query.glossy_filter_roughness);
            const auto selection = closure_selection_state(
                services, point, closure, incoming, query);
            total_weight += selection.weight;
            closure_count += select(0u, 1u, closure_allocated(closure));
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
    Bool selected_glass = false;
    Bool selected_glass_transmission = false;
    Bool selected_glass_singular = false;
    Bool selected_candidate_valid = true;
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    Float3 glass_singular_evaluation = make_float3(0.0f);
    Float glass_singular_pdf = 0.0f;
    Float glass_sample_weight = 0.0f;
    Float glass_eta = 1.0f;
    UInt closure_index = 0u;

    for_each_physical_closure(services,
        point,
        values,
        [&](const TracedClosure &closure) noexcept {
            auto is_translucent =
                closure.operation ==
                compiler::ClosureOperation::translucent;
            auto is_principled = closure.operation ==
                                 compiler::ClosureOperation::principled;
            auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            auto is_transparent =
                closure.operation ==
                compiler::ClosureOperation::transparent;
            auto is_glass = closure.operation ==
                            compiler::ClosureOperation::glass;
            const auto allocated = closure_allocated(closure);
            const auto current_closure_index = closure_index;
            const auto selection = closure_selection_state(
                services, point, closure, incoming, query);
            const auto glossy_normal = selection.glossy_normal;
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
            auto glossy_direction = sample_ggx(closure,
                incoming,
                random_direction,
                glossy_normal,
                query.glossy_filter_roughness);
            auto transparent_direction = -point.incoming;
            const auto sample_glossy = is_glossy || is_principled;
            auto candidate_direction = is_transparent
                                           ? transparent_direction
                                           : select(diffuse_direction,
                                                 glossy_direction,
                                                 sample_glossy);
            Bool candidate_valid = true;
            Bool glass_transmission = false;
            Bool glass_singular = false;
            Float3 candidate_glass_singular_evaluation =
                make_float3(0.0f);
            Float candidate_glass_singular_pdf = 0.0f;
            Float candidate_glass_eta = 1.0f;
            Float candidate_glass_alpha = 0.0f;
            if (is_glass) {
                const auto glass = sample_glass(closure,
                    incoming,
                    point.geometric_normal,
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
            auto sampled_glossy_roughness = microfacet_alpha(
                closure, query.glossy_filter_roughness);
            auto nontransparent_roughness = select(make_float2(1.0f),
                make_float2(sampled_glossy_roughness),
                sample_glossy);
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
                    cycles_closure_type(closure),
                    choose);
                trace.closure_sample_weight =
                    select(trace.closure_sample_weight,
                        closure_sample_weight(closure),
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

    auto diffuse_evaluation =
        evaluate_traced(
            services, values, point, result.wi, query, 0u, false);
    auto reflection_geometric_valid =
        dot(point.geometric_normal, result.wi) > 0.0f;
    auto transmission_geometric_valid =
        dot(point.geometric_normal, result.wi) < 0.0f;
    auto geometric_valid = select(reflection_geometric_valid,
        transmission_geometric_valid,
        selected_translucent | selected_glass_transmission);
    auto diffuse_valid =
        selected & (!selected_transparent) & geometric_valid &
        selected_candidate_valid;
    auto transparent_valid = selected & selected_transparent;
    result.valid = diffuse_valid | transparent_valid;
    result.evaluation.f = select(diffuse_evaluation.f,
        transparent_weight * 1.0e6f,
        selected_transparent);
    result.evaluation.f = select(result.evaluation.f,
        glass_singular_evaluation,
        selected_glass & selected_glass_singular);
    result.evaluation.pdf = select(diffuse_evaluation.pdf,
        1.0e6f * transparent_sample_weight /
            max(total_weight, 1.0e-20f),
        selected_transparent);
    result.evaluation.pdf = select(result.evaluation.pdf,
        glass_singular_pdf * glass_sample_weight /
            max(total_weight, 1.0e-20f),
        selected_glass & selected_glass_singular);
    result.evaluation.diffuse_f = select(diffuse_evaluation.diffuse_f,
        make_float3(0.0f),
        selected_transparent);
    result.evaluation.glossy_f = select(diffuse_evaluation.glossy_f,
        make_float3(0.0f),
        selected_transparent);
    result.evaluation.glossy_f = select(result.evaluation.glossy_f,
        select(glass_singular_evaluation,
            make_float3(0.0f),
            selected_glass_transmission),
        selected_glass & selected_glass_singular);
    result.evaluation.diffuse_pdf = select(
        diffuse_evaluation.diffuse_pdf, 0.0f, selected_transparent);
    auto sampled_surface_events = select(
        static_cast<std::uint32_t>(event_diffuse | event_reflection),
        static_cast<std::uint32_t>(event_glossy | event_reflection),
        selected_glossy);
    sampled_surface_events = select(sampled_surface_events,
        static_cast<std::uint32_t>(event_diffuse | event_transmission),
        selected_translucent);
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
    result.eta = select(1.0f, glass_eta, selected_glass);
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

[[nodiscard]] Float3 GraphSurfaceImplementation::emission(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3>) const noexcept {
    if (!_program) {
        return make_float3(0.0f);
    }
    auto values = trace_values(services, point);
    Float3 result = make_float3(0.0f);
    for_each_closure(
        values, [&](const TracedClosure &closure) noexcept {
            if (closure.operation ==
                compiler::ClosureOperation::emission) {
                result += closure.weight;
            }
        });
    return result;
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
                result += closure.weight;
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
    Float roughness = 0.0f;
    Float3 normal = make_float3(0.0f);
    auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    for_each_physical_closure(services,
        point,
        values,
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
            const auto is_glossy =
                closure.operation == compiler::ClosureOperation::glossy;
            const auto is_glass =
                closure.operation == compiler::ClosureOperation::glass;
            if (!is_diffuse && !is_translucent && !is_principled &&
                !is_glossy && !is_glass) {
                return;
            }
            auto glossy_normal = ensure_valid_specular_reflection(
                point.geometric_normal, incoming, closure.normal);
            Float3 diffuse_albedo = make_float3(0.0f);
            Float diffuse_weight = 0.0f;
            Float glossy_weight = 0.0f;
            if (is_glass) {
                auto fresnel = fresnel_dielectric_cos(
                    dot(glossy_normal, incoming), closure.ior);
                result.glossy_albedo += closure.albedo * fresnel;
                result.transmission_albedo +=
                    closure.albedo * (1.0f - fresnel);
                glossy_weight = pass_weight(closure.weight);
            } else if (is_principled || is_glossy) {
                result.glossy_albedo += closure.albedo;
                glossy_weight = pass_weight(closure.weight);
            } else if (is_diffuse || is_translucent) {
                diffuse_albedo = closure.albedo;
                diffuse_weight = pass_weight(closure.weight);
            }
            const auto weight = diffuse_weight + glossy_weight;
            total_weight += weight;
            // Cycles Diffuse Color includes only diffuse/BSSRDF
            // closures. Glossy closure weights still contribute to the
            // Normal and Roughness passes, but never to Diffuse Color.
            result.albedo += diffuse_albedo;
            roughness += weight * closure.roughness;
            normal +=
                diffuse_weight *
                    (is_translucent ? glossy_normal : closure.normal) +
                glossy_weight * glossy_normal;
        });
    auto valid = total_weight > 0.0f;
    result.roughness = make_float2(
        select(1.0f, roughness / max(total_weight, 1.0e-20f), valid));
    result.normal =
        safe_normalize(select(point.shading_normal, normal, valid),
            point.shading_normal);
    return result;
}

} // namespace psycles::luisa_backend::detail
