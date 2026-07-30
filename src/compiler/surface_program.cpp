#include "surface_program_builder.h"

#include <utility>

namespace psycles::compiler {

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
      _volume_root{volume_root} {}

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
