#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

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
            } else if (closure.operation ==
                       compiler::ClosureOperation::principled) {
                result += closure.emission;
            }
        });
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
            case compiler::ClosureOperation::transparent:
                return;
        }
    };
    visit(visit, _program->root(), 1.0f);
    return result;
}

}// namespace psycles::luisa_backend::detail
