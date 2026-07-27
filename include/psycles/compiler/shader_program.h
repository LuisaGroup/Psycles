#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <psycles/contract/shader_graph.h>

namespace psycles::compiler {

class ShaderProgram {

private:
    contract::ShaderGraph _graph;
    contract::GraphAnalysis _analysis;

public:
    ShaderProgram(contract::ShaderGraph graph, contract::GraphAnalysis analysis) noexcept
        : _graph{std::move(graph)}, _analysis{std::move(analysis)} {}

    [[nodiscard]] const contract::ShaderGraph &graph() const noexcept { return _graph; }
    [[nodiscard]] const contract::GraphAnalysis &analysis() const noexcept { return _analysis; }
};

struct ShaderCompilation {
    std::shared_ptr<const ShaderProgram> program;
    std::vector<contract::GraphDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept { return program != nullptr; }
};

class ShaderCompiler {

private:
    contract::NodeRegistry _registry;

public:
    explicit ShaderCompiler(contract::NodeRegistry registry) noexcept
        : _registry{std::move(registry)} {}

    [[nodiscard]] const contract::NodeRegistry &registry() const noexcept {
        return _registry;
    }

    [[nodiscard]] ShaderCompilation compile(const contract::ShaderGraph &graph) const;
};

}// namespace psycles::compiler

