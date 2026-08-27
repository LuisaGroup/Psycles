#include "graph_surface_internal.h"

#include <utility>

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_magic.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_voronoi.h>
#include <psycles/luisa/cycles_wave.h>
#include <psycles/luisa/cycles_volume.h>
#include <psycles/luisa/surface_closure_operations.h>
#include <psycles/luisa/surface_closure_sampling.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceClosureReachability program_closure_reachability(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceClosurePlan &plan) noexcept {
    std::uint32_t operations = 0u;
    std::uint32_t principled_features = 0u;
    std::uint32_t anisotropic_operations = 0u;
    std::uint32_t anisotropic_principled_features = 0u;
    std::uint32_t thin_film_operations = 0u;
    std::uint32_t thin_film_principled_features = 0u;
    const auto &closures = program.closure_instructions();
    for (std::size_t index = 0u; index < closures.size(); ++index) {
        const auto id = compiler::ClosureExpressionId{
            static_cast<std::uint32_t>(index)};
        const auto &entry = plan.entry(id);
        if (!entry.reachable) { continue; }
        const auto operation = closures[index].operation;
        if (operation == compiler::ClosureOperation::null_closure ||
            operation == compiler::ClosureOperation::add ||
            operation == compiler::ClosureOperation::mix) {
            continue;
        }
        const auto operation_bit =
            std::uint32_t{1u} << static_cast<std::uint32_t>(operation);
        operations |= operation_bit;
        if (operation == compiler::ClosureOperation::principled) {
            principled_features |= entry.principled_features;
        }
        if (entry.microfacet_anisotropy) {
            anisotropic_operations |= operation_bit;
            if (operation == compiler::ClosureOperation::principled) {
                anisotropic_principled_features |=
                    entry.principled_features;
            }
        }
        if (entry.thin_film) {
            thin_film_operations |= operation_bit;
            if (operation == compiler::ClosureOperation::principled) {
                thin_film_principled_features |=
                    entry.principled_features;
            }
        }
    }
    return reachable_surface_closures(
        operations,
        principled_features,
        anisotropic_operations,
        anisotropic_principled_features,
        thin_film_operations,
        thin_film_principled_features);
}

}// namespace

GraphSurfaceImplementation::GraphSurfaceImplementation(
    std::shared_ptr<const compiler::SurfaceProgram> program) noexcept
    : GraphSurfaceImplementation{
          program,
          program
              ? compiler::conservative_surface_closure_plan(*program)
              : compiler::SurfaceClosurePlan{}} {}

GraphSurfaceImplementation::GraphSurfaceImplementation(
    std::shared_ptr<const compiler::SurfaceProgram> program,
    compiler::SurfaceClosurePlan closure_plan) noexcept
    : _program{std::move(program)},
      _closure_plan{std::move(closure_plan)} {
    if (_program) {
        if (!_closure_plan.compatible(*_program)) {
            _closure_plan =
                compiler::conservative_surface_closure_plan(*_program);
        }
        _value_dependency_plan =
            compiler::analyze_surface_value_dependencies(
                *_program, _closure_plan);
        _physical_closure_reachability =
            program_closure_reachability(*_program, _closure_plan);
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
                    compiler::ValueOperation::magic_color ||
                instruction.operation ==
                    compiler::ValueOperation::magic_factor) {
                cycles_magic::prepare();
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
        const auto has_principled_feature = [](
                                                const compiler::SurfaceClosurePlanEntry &entry,
                                                compiler::PrincipledClosureFeature feature) noexcept {
            return (entry.principled_features &
                    compiler::principled_closure_feature_bit(feature)) != 0u;
        };
        const auto &closures = _program->closure_instructions();
        for (std::size_t index = 0u; index < closures.size(); ++index) {
            const auto id = compiler::ClosureExpressionId{
                static_cast<std::uint32_t>(index)};
            const auto &entry = _closure_plan.entry(id);
            if (!entry.reachable) {
                continue;
            }
            const auto operation = closures[index].operation;
            const auto principled =
                operation == compiler::ClosureOperation::principled;
            _capabilities.may_emit |=
                operation == compiler::ClosureOperation::emission ||
                (principled && has_principled_feature(
                                   entry,
                                   compiler::PrincipledClosureFeature::emission));
            _capabilities.may_be_transparent |=
                operation == compiler::ClosureOperation::transparent ||
                (principled && has_principled_feature(
                                   entry,
                                   compiler::PrincipledClosureFeature::alpha));
            _capabilities.may_have_subsurface |=
                operation == compiler::ClosureOperation::subsurface ||
                (principled &&
                 (has_principled_feature(
                      entry,
                      compiler::PrincipledClosureFeature::thick_subsurface) ||
                  has_principled_feature(
                      entry,
                      compiler::PrincipledClosureFeature::thin_subsurface)));
        }
        _capabilities.emission_is_constant =
            !_capabilities.may_emit ||
            _program->emission_evaluation() !=
                compiler::EmissionEvaluationMode::deferred;
        _capabilities.may_have_volume = _program->volume_root().valid();
        if (_program->displacement_root().valid()) {
            _displacement_dependency_mask = value_dependency_mask(
                _program->displacement_root());
        }
        if (_program->surface_normal_root().valid()) {
            _surface_normal_dependency_mask = value_dependency_mask(
                _program->surface_normal_root());
            const auto &instructions =
                _program->value_instructions();
            for (std::size_t index = 0u;
                 index < instructions.size();
                 ++index) {
                _automatic_bump_uses_undisplaced_geometry |=
                    _surface_normal_dependency_mask[index] &&
                    instructions[index].operation ==
                        compiler::ValueOperation::bump &&
                    (instructions[index].static_u0 & 4u) != 0u;
            }
        }
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
    const SurfaceClosurePoint closure_point{point};
    const auto directions =
        make_surface_closure_evaluation_directions(
            closure_point, outgoing_expression);
    const auto sampled_light =
        context.mode == EvaluationMode::sampled_light;
    const auto policy = make_surface_closure_evaluation_policy(
        sampled_light, context.light_shader_flags);
    const auto sampled_bsdf =
        context.mode == EvaluationMode::sampled_bsdf;
    const auto selected_closure_index =
        UInt{context.selected_closure_index};

    SurfaceClosureEvaluationAccumulator accumulator;
    UInt closure_index = 0u;
    for_each_physical_closure(
        services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const TracedClosure &closure) noexcept {
            const auto physical =
                canonical_surface_closure(closure);
            const auto allocated = closure_allocated(physical);
            const auto current_closure_index = closure_index;
            const auto selected_sample =
                Bool{sampled_bsdf} & allocated &
                (current_closure_index ==
                 selected_closure_index);
            $if(allocated) {
                accumulator.add(
                    surface_closure_evaluation_contribution(
                        services,
                        closure_point,
                        values.shading_normal,
                        physical,
                        directions.incoming,
                        directions.outgoing,
                        query,
                        policy,
                        selected_sample,
                        _physical_closure_reachability));
            };
            closure_index += select(0u, 1u, allocated);
        });
    return accumulator.finish(policy.preserve_pdf);
}
[[nodiscard]] SurfaceEvaluation GraphSurfaceImplementation::evaluate(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing_expression,
    const SurfaceQuery &query) const noexcept {
    if (!_program) {
        return SurfaceEvaluation::zero();
    }
    auto values = trace_surface_values(
        services, point, &_value_dependency_plan.physical);
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
    auto values = trace_surface_values(
        services, point, &_value_dependency_plan.physical);
    return evaluate_traced(services,
        values,
        point,
        outgoing_expression,
        query.surface,
        {.mode = EvaluationMode::sampled_light,
            .light_shader_flags = query.shader_flags,
            .selected_closure_index = ~std::uint32_t{0u}});
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
    auto values = trace_surface_values(
        services, point, &_value_dependency_plan.physical);
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
                physical.sample_weight,
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
    if (!_program) {
        return SurfaceSampleTrace::zero();
    }
    const SurfaceClosurePoint closure_point{point};
    const auto policy =
        make_surface_closure_evaluation_policy(
            false, Expr<std::uint32_t>{0u});
    DirectSurfaceClosureEvaluationOperation evaluation{
        services,
        closure_point,
        query,
        policy,
        _physical_closure_reachability};
    DirectSurfaceClosureSamplingOperation sampling{
        services,
        closure_point,
        query,
        _physical_closure_reachability};
    SurfaceClosureSamplingVisitor visitor{
        maximum_surface_closure_capacity,
        closure_point,
        sampling,
        evaluation,
        u_lobe_expression,
        u_direction_expression,
        trace_selection};
    static_cast<void>(collect_closures(
        services,
        point,
        query.reflective_caustics,
        query.refractive_caustics,
        visitor));
    return visitor.result();
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
    auto values = trace_surface_values(
        services, point, &_value_dependency_plan.physical);
    Float3 result = make_float3(0.0f);
    for_each_physical_closure(
        services,
        point,
        values,
        true,
        true,
        [&](const TracedClosure &closure) noexcept {
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
    if (!_program->surface_normal_root().valid()) {
        return point.shading_normal;
    }
    const auto values = trace_value_stage(
        services,
        automatic_bump_point(point),
        &_surface_normal_dependency_mask);
    return values.values[
        _program->surface_normal_root().value].vector();
}

[[nodiscard]] Float3 GraphSurfaceImplementation::displacement(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    if (!_program || !_program->displacement_root().valid()) {
        return make_float3(0.0f);
    }
    const auto values = trace_values(
        services, point, &_displacement_dependency_mask);
    return values.values[
        _program->displacement_root().value].vector();
}

} // namespace psycles::luisa_backend::detail
