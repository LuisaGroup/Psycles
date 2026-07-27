#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/shader_graph.h>

namespace psycles::adapter {

using CyclesNodeId = std::uint64_t;

enum class CyclesGraphStage : std::uint8_t {
    normalized,
    svm_lowered
};

struct CyclesOutputRef {
    CyclesNodeId node{};
    std::string socket;
};

struct CyclesInput {
    std::string socket;
    std::optional<CyclesOutputRef> source;
    std::optional<contract::SocketValue> value;
};

struct CyclesNode {
    CyclesNodeId id{};
    std::string type;
    // Canonical discriminator for operation- or data-type-dependent Cycles
    // nodes, for example type="math", variant="add".
    std::string variant;
    std::string label;
    std::vector<CyclesInput> inputs;
    std::map<std::string, contract::SocketValue, std::less<>> properties;
};

struct CyclesNormalizedShaderGraph {
    CyclesGraphStage stage{CyclesGraphStage::normalized};
    std::vector<CyclesNode> nodes;
    std::array<std::optional<CyclesOutputRef>,
               static_cast<std::size_t>(contract::ShaderDomain::count)>
        roots;

    void set_root(
        contract::ShaderDomain domain,
        std::optional<CyclesOutputRef> root);
    [[nodiscard]] const std::optional<CyclesOutputRef> &root(
        contract::ShaderDomain domain) const noexcept;
};

struct CyclesNodeMapping {
    std::string cycles_type;
    std::string cycles_variant;
    std::string psycles_type;
    std::map<std::string, std::string, std::less<>> inputs;
    std::map<std::string, std::string, std::less<>> outputs;
    std::map<std::string, std::string, std::less<>> properties;
};

class CyclesNodeMappingRegistry {

private:
    std::map<std::string, CyclesNodeMapping, std::less<>> _mappings;

public:
    [[nodiscard]] bool register_mapping(CyclesNodeMapping mapping);
    [[nodiscard]] const CyclesNodeMapping *find(
        std::string_view cycles_type,
        std::string_view cycles_variant = {}) const;
    [[nodiscard]] std::size_t size() const noexcept {
        return _mappings.size();
    }
};

enum class CyclesAdapterDiagnosticCode : std::uint8_t {
    graph_already_svm_lowered,
    duplicate_node_id,
    unknown_node_type,
    unknown_input,
    unknown_output,
    unknown_property,
    invalid_node_reference,
    invalid_binding
};

struct CyclesAdapterDiagnostic {
    CyclesAdapterDiagnosticCode code{};
    std::string message;
    std::optional<CyclesNodeId> node;
    std::string socket;
};

struct CyclesShaderGraphAdaptation {
    std::optional<contract::ShaderGraph> graph;
    std::map<CyclesNodeId, contract::NodeId> node_map;
    std::vector<CyclesAdapterDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept {
        return graph.has_value() && diagnostics.empty();
    }
};

[[nodiscard]] CyclesNodeMappingRegistry make_core_cycles_node_mappings();

[[nodiscard]] CyclesShaderGraphAdaptation adapt_cycles_shader_graph(
    const CyclesNormalizedShaderGraph &source,
    const CyclesNodeMappingRegistry &mappings);

}// namespace psycles::adapter
