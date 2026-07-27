#include <psycles/compiler/shader_program.h>

namespace psycles::compiler {

ShaderCompilation ShaderCompiler::compile(const contract::ShaderGraph &graph) const {
    auto normalized = graph;

    for (const auto &source_node : graph.nodes()) {
        auto *node = normalized.find(source_node.id);
        const auto *schema = _registry.find(source_node.type);
        if (node == nullptr || schema == nullptr) {
            continue;
        }

        for (const auto &input : schema->inputs) {
            if (!node->inputs.contains(input.name) && input.default_value) {
                static_cast<void>(normalized.set_input(
                    node->id, input.name, *input.default_value));
            }
        }
        for (const auto &property : schema->properties) {
            if (!node->properties.contains(property.name) && property.default_value) {
                static_cast<void>(normalized.set_property(
                    node->id, property.name, *property.default_value));
            }
        }
    }

    auto validation = validate_shader_graph(normalized, _registry);
    if (!validation.ok()) {
        return {
            .program = nullptr,
            .diagnostics = std::move(validation.diagnostics)};
    }

    return {
        .program = std::make_shared<const ShaderProgram>(
            std::move(normalized),
            std::move(*validation.analysis)),
        .diagnostics = std::move(validation.diagnostics)};
}

}// namespace psycles::compiler

