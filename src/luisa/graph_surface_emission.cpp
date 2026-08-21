#include "graph_surface_internal.h"
#include "principled_layer_component.h"

#include <psycles/luisa/surface_closure_operations.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] SurfacePopulation GraphSurfaceImplementation::populate(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfacePopulationQuery &query,
    SurfaceClosureCollector &collector) const noexcept {
    if (!_program) {
        collector.begin(point.shading_normal);
        collector.finish();
        return {
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal};
    }

    // One topology-closed schedule dominates both consumers. The dependency
    // plan is the exact union of emission and physical-closure endpoints, so a
    // shared DAG input is evaluated once and every original closure parameter
    // remains a typed Luisa expression until canonical closure construction.
    const auto values = trace_surface_values(
        services, point, &_value_dependency_plan.preparation);
    const auto emitted = emission_traced(
        services,
        point,
        values,
        query.emission_reflective_caustics);
    const auto collection = collect_traced_closures(
        services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        collector);
    return {
        .emission = emitted,
        .shading_normal = collection.shading_normal};
}

[[nodiscard]] Float3 GraphSurfaceImplementation::emission(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3>,
    Expr<bool> reflective_caustics) const noexcept {
    if (!_program) {
        return make_float3(0.0f);
    }
    const auto values = trace_surface_values(
        services, point, &_value_dependency_plan.emission);
    return emission_traced(
        services, point, values, reflective_caustics);
}

[[nodiscard]] Float3 GraphSurfaceImplementation::emission_traced(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedValues &values,
    Expr<bool> reflective_caustics) const noexcept {
    const PrincipledLayerComponent principled_layers{services, point};
    Float3 result = make_float3(0.0f);
    for_each_closure(
        values,
        _value_dependency_plan.emission_closures,
        _value_dependency_plan.emission,
        [&](const TracedClosure &closure) noexcept {
            if (closure.operation ==
                compiler::ClosureOperation::emission) {
                result += closure.weight;
            } else if (closure.operation ==
                           compiler::ClosureOperation::principled &&
                       (closure.principled_features &
                        compiler::principled_closure_feature_bit(
                            compiler::PrincipledClosureFeature::emission)) != 0u) {
                result += principled_layers
                              .evaluate_emission(
                                  closure,
                                  closure.principled_features,
                                  reflective_caustics)
                              .radiance;
            }
        });
    return result;
}

[[nodiscard]] SurfacePreparation GraphSurfaceImplementation::prepare(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfacePreparationQuery &query) const noexcept {
    auto result = SurfacePreparation::zero(point);
    if (!_program) {
        return result;
    }

    // This is the production graph-fusion point: one strongly typed value
    // schedule feeds both the authored emission root and the raw physical
    // closure reductions. No closure record is baked or stored in a weakly
    // typed device array.
    const auto values = trace_surface_values(
        services, point, &_value_dependency_plan.preparation);
    return prepare_traced_values(
        services, point, values, query);
}

[[nodiscard]] SurfacePreparation
GraphSurfaceImplementation::prepare_traced_values(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedValues &values,
    const SurfacePreparationQuery &query) const noexcept {
    auto result = SurfacePreparation::zero(point);
    if (!_program) {
        return result;
    }
    result.emission = emission_traced(
        services,
        point,
        values,
        query.emission_reflective_caustics);
    const auto identity = make_surface_closure_identity_callable();
    const auto aov_operation = make_surface_closure_aov_callable();
    SurfacePreparationVisitor visitor{
        point,
        query.glossy_filter_roughness,
        query.include_runtime_flags,
        query.include_aov,
        maximum_surface_closure_capacity,
        identity,
        aov_operation};
    static_cast<void>(collect_traced_closures(
        services,
        point,
        values,
        query.reflective_caustics,
        query.refractive_caustics,
        visitor));
    result.runtime_flags = visitor.runtime_flags();
    result.shading_normal = visitor.shading_normal();
    result.aov = visitor.aov();
    return result;
}

[[nodiscard]] Float3 GraphSurfaceImplementation::constant_emission(
    const SurfaceParameterServices &services,
    Expr<std::uint32_t> parameter_block) const noexcept {
    if (!_program ||
        _program->emission_evaluation() !=
            compiler::EmissionEvaluationMode::constant) {
        return make_float3(0.0f);
    }
    const auto &values = _program->value_instructions();
    const auto parameter = [&](compiler::ValueExpressionId expression)
        -> const compiler::ValueInstruction & {
        return values[expression.value];
    };
    Float3 result = make_float3(0.0f);
    const auto visit =
        [&](auto &&self,
            compiler::ClosureExpressionId id,
            Float weight) noexcept -> void {
        const auto &closure =
            _program->closure_instructions()[id.value];
        switch (closure.operation) {
            case compiler::ClosureOperation::emission: {
                const auto &color = parameter(closure.color);
                const auto &strength = parameter(closure.strength);
                result +=
                    services.parameter_float3(
                        parameter_block,
                        color.parameter.value) *
                    services.parameter_float(
                        parameter_block,
                        strength.parameter.value) *
                    weight;
                return;
            }
            case compiler::ClosureOperation::add:
                self(self, closure.a, weight);
                self(self, closure.b, weight);
                return;
            case compiler::ClosureOperation::mix: {
                const auto &factor = parameter(closure.factor);
                const auto mix = clamp(
                    services.parameter_float(
                        parameter_block,
                        factor.parameter.value),
                    0.0f,
                    1.0f);
                self(self, closure.a, weight * (1.0f - mix));
                self(self, closure.b, weight * mix);
                return;
            }
            case compiler::ClosureOperation::null_closure:
            case compiler::ClosureOperation::diffuse:
            case compiler::ClosureOperation::translucent:
            case compiler::ClosureOperation::principled:
            case compiler::ClosureOperation::glossy:
            case compiler::ClosureOperation::glass:
            case compiler::ClosureOperation::refraction:
            case compiler::ClosureOperation::transparent:
                return;
        }
    };
    visit(visit, _program->root(), 1.0f);
    return result;
}

}// namespace psycles::luisa_backend::detail
