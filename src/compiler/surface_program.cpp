#include "surface_program_builder.h"

#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

struct EmissionProof {
    bool may_emit{};
    bool cycles_constant{true};
};

[[nodiscard]] bool is_unconnected_parameter(
    const std::vector<ValueInstruction> &values,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    return expression.valid() &&
           expression.value < values.size() &&
           values[expression.value].operation ==
               ValueOperation::parameter &&
           values[expression.value].parameter.valid() &&
           values[expression.value].source_node == owner;
}

// This transfer function is the closed-form counterpart of Cycles'
// output_estimate_emission relation. Non-emitting closure leaves are the
// additive identity and do not force shader evaluation. Emission inputs and
// Mix weights are provably constant only when they are direct, unconnected
// parameters; a linked value remains deferred even if a later optimizer could
// prove it numerically uniform. That preserves Cycles' conservative scheduling
// without evaluating or baking a single material value on the host.
[[nodiscard]] EmissionEvaluationMode analyze_emission_evaluation(
    const std::vector<ValueInstruction> &values,
    const std::vector<ClosureInstruction> &closures,
    ClosureExpressionId root) noexcept {
    if (!root.valid() || root.value >= closures.size()) {
        return EmissionEvaluationMode::none;
    }
    std::vector<EmissionProof> proofs;
    proofs.reserve(closures.size());
    const auto dependency = [&](ClosureExpressionId id) noexcept {
        return id.valid() && id.value < proofs.size()
                   ? proofs[id.value]
                   : EmissionProof{.may_emit = true,
                         .cycles_constant = false};
    };
    for (const auto &closure : closures) {
        EmissionProof proof;
        switch (closure.operation) {
            case ClosureOperation::null_closure:
            case ClosureOperation::diffuse:
            case ClosureOperation::translucent:
            case ClosureOperation::glossy:
            case ClosureOperation::glass:
            case ClosureOperation::refraction:
            case ClosureOperation::transparent:
                break;
            case ClosureOperation::principled:
                // Cycles never exposes Principled through the constant
                // emission path: alpha, sheen and coat can all affect its
                // emitted radiance. Material-specific zero emission is
                // refined separately by estimate_surface_emission().
                proof.may_emit = true;
                proof.cycles_constant = false;
                break;
            case ClosureOperation::emission:
                proof.may_emit = true;
                proof.cycles_constant =
                    is_unconnected_parameter(
                        values, closure.color, closure.source_node) &&
                    is_unconnected_parameter(
                        values, closure.strength, closure.source_node);
                break;
            case ClosureOperation::add: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                proof.may_emit = a.may_emit || b.may_emit;
                proof.cycles_constant =
                    a.cycles_constant && b.cycles_constant;
                break;
            }
            case ClosureOperation::mix: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                proof.may_emit = a.may_emit || b.may_emit;
                proof.cycles_constant =
                    a.cycles_constant && b.cycles_constant &&
                    is_unconnected_parameter(
                        values, closure.factor, closure.source_node);
                break;
            }
        }
        proofs.emplace_back(proof);
    }
    const auto result = proofs[root.value];
    if (!result.may_emit) {
        return EmissionEvaluationMode::none;
    }
    return result.cycles_constant
               ? EmissionEvaluationMode::constant
               : EmissionEvaluationMode::deferred;
}

}// namespace

SurfaceProgram::SurfaceProgram(
    std::uint64_t structure_signature,
    std::vector<ParameterDesc> parameters,
    std::vector<ValueInstruction> value_instructions,
    std::vector<ClosureInstruction> closure_instructions,
    ClosureExpressionId root,
    std::vector<VolumeInstruction> volume_instructions,
    VolumeExpressionId volume_root) noexcept
    : _structure_signature{structure_signature},
      _parameters{std::move(parameters)},
      _value_instructions{std::move(value_instructions)},
      _closure_instructions{std::move(closure_instructions)},
      _volume_instructions{std::move(volume_instructions)},
      _root{root},
      _volume_root{volume_root},
      _emission_evaluation{analyze_emission_evaluation(
          _value_instructions,
          _closure_instructions,
          _root)} {}

SurfaceParameterBlock::SurfaceParameterBlock(
    const SurfaceProgram &program) {
    _values.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        _values.emplace_back(parameter.default_value);
    }
}

const contract::SocketValue *SurfaceParameterBlock::find(
    ParameterId id) const noexcept {
    if (!id.valid() ||
        static_cast<std::size_t>(id.value) >= _values.size()) {
        return nullptr;
    }
    return &_values[id.value];
}

bool SurfaceParameterBlock::set(
    const SurfaceProgram &program,
    ParameterId id,
    contract::SocketValue value) {
    if (!id.valid() ||
        static_cast<std::size_t>(id.value) >= _values.size() ||
        static_cast<std::size_t>(id.value) >=
            program.parameters().size() ||
        value.type != program.parameters()[id.value].type ||
        !value.well_typed()) {
        return false;
    }
    _values[id.value] = std::move(value);
    return true;
}

namespace {

[[nodiscard]] const contract::SocketValue *direct_parameter_value(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    if (!expression.valid() ||
        expression.value >= program.value_instructions().size()) {
        return nullptr;
    }
    const auto &instruction =
        program.value_instructions()[expression.value];
    if (instruction.operation != ValueOperation::parameter ||
        instruction.source_node != owner ||
        !instruction.parameter.valid()) {
        return nullptr;
    }
    return parameters.find(instruction.parameter);
}

[[nodiscard]] Vec3f add(Vec3f a, Vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f value, float scale) noexcept {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale};
}

}// namespace

Vec3f estimate_surface_emission(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) {
    if (!program.root().valid()) {
        return {};
    }
    std::vector<Vec3f> estimates;
    estimates.reserve(program.closure_instructions().size());
    const auto dependency = [&](ClosureExpressionId id) noexcept {
        // A malformed dependency must not silently remove an emitter from
        // the scene. Valid compiled programs always take the first branch.
        return id.valid() && id.value < estimates.size()
                   ? estimates[id.value]
                   : Vec3f{1.0f, 1.0f, 1.0f};
    };
    for (const auto &closure : program.closure_instructions()) {
        Vec3f estimate{};
        switch (closure.operation) {
            case ClosureOperation::principled:
            case ClosureOperation::emission: {
                const auto color_expression =
                    closure.operation == ClosureOperation::principled
                        ? closure.emission_color
                        : closure.color;
                const auto strength_expression =
                    closure.operation == ClosureOperation::principled
                        ? closure.emission_strength
                        : closure.strength;
                auto color = Vec3f{1.0f, 1.0f, 1.0f};
                if (const auto *value = direct_parameter_value(
                        program,
                        parameters,
                        color_expression,
                        closure.source_node)) {
                    if (const auto *literal =
                            std::get_if<Vec3f>(&value->value)) {
                        color = *literal;
                    }
                }
                auto strength = 1.0f;
                if (const auto *value = direct_parameter_value(
                        program,
                        parameters,
                        strength_expression,
                        closure.source_node)) {
                    if (const auto *literal =
                            std::get_if<float>(&value->value)) {
                        strength = *literal;
                    }
                }
                estimate = multiply(color, strength);
                break;
            }
            case ClosureOperation::add:
                estimate = add(
                    dependency(closure.a),
                    dependency(closure.b));
                break;
            case ClosureOperation::mix: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                const auto *value = direct_parameter_value(
                    program,
                    parameters,
                    closure.factor,
                    closure.source_node);
                const auto *factor =
                    value != nullptr
                        ? std::get_if<float>(&value->value)
                        : nullptr;
                // Cycles intentionally does not try to estimate a linked
                // factor: both branches remain possible and are summed.
                estimate = factor != nullptr
                               ? add(
                                     multiply(a, 1.0f - *factor),
                                     multiply(b, *factor))
                               : add(a, b);
                break;
            }
            case ClosureOperation::null_closure:
            case ClosureOperation::diffuse:
            case ClosureOperation::translucent:
            case ClosureOperation::glossy:
            case ClosureOperation::glass:
            case ClosureOperation::refraction:
            case ClosureOperation::transparent:
                break;
        }
        estimates.emplace_back(estimate);
    }
    return program.root().value < estimates.size()
               ? estimates[program.root().value]
               : Vec3f{};
}

SurfaceProgramCompilation compile_surface_program(
    const ShaderProgram &shader) {
    return detail::SurfaceProgramBuilder{shader}.build();
}

SurfaceParameterBinding bind_surface_parameters(
    const SurfaceProgram &program,
    const ShaderProgram &shader) {
    SurfaceParameterBinding result;
    if (program.structure_signature() !=
        shader.analysis().structure_signature) {
        result.diagnostics.emplace_back(
            SurfaceProgramDiagnostic{
                .code =
                    SurfaceProgramDiagnosticCode::structure_mismatch,
                .message =
                    "shader structure does not match the reusable "
                    "surface program",
                .node = std::nullopt,
                .socket = {}});
        return result;
    }

    SurfaceParameterBlock block{program};
    for (const auto &parameter : program.parameters()) {
        const auto *node =
            shader.graph().find(parameter.node);
        if (node == nullptr) {
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::missing_input,
                    .message =
                        "parameter binding references a missing node",
                    .node = parameter.node,
                    .socket = parameter.socket});
            continue;
        }
        auto input = node->inputs.find(parameter.socket);
        if (input == node->inputs.end() ||
            input->second.source ||
            !input->second.value) {
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::missing_input,
                    .message =
                        detail::node_prefix(parameter.node) +
                        "runtime parameter input '" +
                        parameter.socket +
                        "' is missing or connected",
                    .node = parameter.node,
                    .socket = parameter.socket});
            continue;
        }
        if (!block.set(
                program,
                parameter.id,
                *input->second.value)) {
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::type_mismatch,
                    .message =
                        detail::node_prefix(parameter.node) +
                        "runtime parameter input '" +
                        parameter.socket +
                        "' changed type",
                    .node = parameter.node,
                    .socket = parameter.socket});
        }
    }

    if (result.diagnostics.empty()) {
        result.parameters = std::move(block);
    }
    return result;
}

}// namespace psycles::compiler
