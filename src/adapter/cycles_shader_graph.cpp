#include <psycles/adapter/cycles_shader_graph.h>

#include <set>
#include <utility>

namespace psycles::adapter {

namespace {

template<typename Mapping>
[[nodiscard]] const std::string *mapped_name(
    const Mapping &mapping,
    std::string_view name) noexcept {
    auto iter = mapping.find(name);
    return iter == mapping.end() ? nullptr : &iter->second;
}

[[nodiscard]] std::string node_prefix(CyclesNodeId node) {
    return "Cycles node " + std::to_string(node) + ": ";
}

[[nodiscard]] std::string mapping_key(
    std::string_view type,
    std::string_view variant) {
    std::string result;
    result.reserve(type.size() + variant.size() + 1u);
    result.append(type);
    result.push_back('\0');
    result.append(variant);
    return result;
}

}// namespace

void CyclesNormalizedShaderGraph::set_root(
    contract::ShaderDomain domain,
    std::optional<CyclesOutputRef> root_value) {
    roots[static_cast<std::size_t>(domain)] = std::move(root_value);
}

const std::optional<CyclesOutputRef> &
CyclesNormalizedShaderGraph::root(
    contract::ShaderDomain domain) const noexcept {
    return roots[static_cast<std::size_t>(domain)];
}

bool CyclesNodeMappingRegistry::register_mapping(
    CyclesNodeMapping mapping) {
    if (mapping.cycles_type.empty() ||
        mapping.psycles_type.empty() ||
        _mappings.contains(mapping_key(
            mapping.cycles_type, mapping.cycles_variant))) {
        return false;
    }
    auto key = mapping_key(
        mapping.cycles_type, mapping.cycles_variant);
    _mappings.emplace(std::move(key), std::move(mapping));
    return true;
}

const CyclesNodeMapping *CyclesNodeMappingRegistry::find(
    std::string_view cycles_type,
    std::string_view cycles_variant) const {
    auto iter = _mappings.find(
        mapping_key(cycles_type, cycles_variant));
    return iter == _mappings.end() ? nullptr : &iter->second;
}

CyclesNodeMappingRegistry make_core_cycles_node_mappings() {
    using namespace compiler;

    CyclesNodeMappingRegistry registry;
    auto add = [&](CyclesNodeMapping mapping) {
        static_cast<void>(registry.register_mapping(std::move(mapping)));
    };

    add({
        .cycles_type = "value",
        .cycles_variant = {},
        .psycles_type = node_type::constant_float,
        .inputs = {{"Value", "Value"}},
        .outputs = {{"Value", "Value"}},
        .properties = {}});
    add({
        .cycles_type = "rgb",
        .cycles_variant = {},
        .psycles_type = node_type::constant_color,
        .inputs = {{"Color", "Color"}},
        .outputs = {{"Color", "Color"}},
        .properties = {}});
    add({
        .cycles_type = "geometry",
        .cycles_variant = {},
        .psycles_type = node_type::geometry,
        .inputs = {},
        .outputs = {
            {"Position", "Position"},
            {"Normal", "Normal"},
            {"True Normal", "GeometricNormal"},
            {"Incoming", "Incoming"},
            {"Tangent", "Tangent"}},
        .properties = {}});
    add({
        .cycles_type = "math",
        .cycles_variant = "add",
        .psycles_type = node_type::add_float,
        .inputs = {{"Value1", "A"}, {"Value2", "B"}},
        .outputs = {{"Value", "Value"}},
        .properties = {}});
    add({
        .cycles_type = "math",
        .cycles_variant = "multiply",
        .psycles_type = node_type::multiply_float,
        .inputs = {{"Value1", "A"}, {"Value2", "B"}},
        .outputs = {{"Value", "Value"}},
        .properties = {}});
    add({
        .cycles_type = "mix",
        .cycles_variant = "color",
        .psycles_type = node_type::mix_color,
        .inputs = {{"Factor", "Factor"}, {"A", "A"}, {"B", "B"}},
        .outputs = {{"Color", "Color"}},
        .properties = {}});
    add({
        .cycles_type = "diffuse_bsdf",
        .cycles_variant = {},
        .psycles_type = node_type::diffuse_bsdf,
        .inputs = {
            {"Color", "Color"},
            {"Normal", "Normal"},
            {"Roughness", "Roughness"}},
        .outputs = {{"BSDF", "Closure"}},
        .properties = {}});
    add({
        .cycles_type = "principled_bsdf",
        .cycles_variant = {},
        .psycles_type = node_type::principled_bsdf,
        .inputs = {
            {"Base Color", "BaseColor"},
            {"Metallic", "Metallic"},
            {"Roughness", "Roughness"},
            {"Diffuse Roughness", "DiffuseRoughness"},
            {"Subsurface Weight", "SubsurfaceWeight"},
            {"Subsurface Radius", "SubsurfaceRadius"},
            {"Subsurface Scale", "SubsurfaceScale"},
            {"IOR", "IOR"},
            {"Specular IOR Level", "SpecularIORLevel"},
            {"Specular Tint", "SpecularTint"},
            {"Emission Color", "EmissionColor"},
            {"Emission Strength", "EmissionStrength"},
            {"Normal", "Normal"}},
        .outputs = {{"BSDF", "Closure"}},
        .properties = {{"distribution", "Distribution"}}});
    add({
        .cycles_type = "glass_bsdf",
        .cycles_variant = {},
        .psycles_type = node_type::glass_bsdf,
        .inputs = {
            {"Color", "Color"},
            {"Roughness", "Roughness"},
            {"IOR", "IOR"},
            {"Normal", "Normal"}},
        .outputs = {{"BSDF", "Closure"}},
        .properties = {{"distribution", "Distribution"}}});
    add({
        .cycles_type = "emission",
        .cycles_variant = {},
        .psycles_type = node_type::emission,
        .inputs = {{"Color", "Color"}, {"Strength", "Strength"}},
        .outputs = {{"Emission", "Closure"}},
        .properties = {}});
    add({
        .cycles_type = "transparent_bsdf",
        .cycles_variant = {},
        .psycles_type = node_type::transparent_bsdf,
        .inputs = {{"Color", "Color"}},
        .outputs = {{"BSDF", "Closure"}},
        .properties = {}});
    add({
        .cycles_type = "add_closure",
        .cycles_variant = {},
        .psycles_type = node_type::add_closure,
        .inputs = {{"Closure1", "A"}, {"Closure2", "B"}},
        .outputs = {{"Closure", "Closure"}},
        .properties = {}});
    add({
        .cycles_type = "mix_closure",
        .cycles_variant = {},
        .psycles_type = node_type::mix_closure,
        .inputs = {
            {"Fac", "Factor"},
            {"Closure1", "A"},
            {"Closure2", "B"}},
        .outputs = {{"Closure", "Closure"}},
        .properties = {}});

    return registry;
}

CyclesShaderGraphAdaptation adapt_cycles_shader_graph(
    const CyclesNormalizedShaderGraph &source,
    const CyclesNodeMappingRegistry &mappings) {

    CyclesShaderGraphAdaptation result;
    auto diagnose = [&](CyclesAdapterDiagnosticCode code,
                        std::string message,
                        std::optional<CyclesNodeId> node = std::nullopt,
                        std::string socket = {}) {
        result.diagnostics.emplace_back(CyclesAdapterDiagnostic{
            .code = code,
            .message = std::move(message),
            .node = node,
            .socket = std::move(socket)});
    };

    if (source.stage != CyclesGraphStage::normalized) {
        diagnose(
            CyclesAdapterDiagnosticCode::graph_already_svm_lowered,
            "the Cycles graph has already undergone SVM closure lowering; "
            "Psycles requires the normalized closure tree");
        return result;
    }

    contract::ShaderGraph graph;
    std::map<CyclesNodeId, const CyclesNode *> source_nodes;
    std::map<CyclesNodeId, const CyclesNodeMapping *> node_mappings;

    for (const auto &node : source.nodes) {
        if (!source_nodes.emplace(node.id, &node).second) {
            diagnose(
                CyclesAdapterDiagnosticCode::duplicate_node_id,
                node_prefix(node.id) + "duplicate identifier",
                node.id);
            continue;
        }
        const auto *mapping = mappings.find(
            node.type, node.variant);
        if (mapping == nullptr) {
            diagnose(
                CyclesAdapterDiagnosticCode::unknown_node_type,
                node_prefix(node.id) + "unsupported normalized node type '" +
                    node.type +
                    (node.variant.empty()
                         ? std::string{}
                         : "' variant '" + node.variant) +
                    "'",
                node.id);
            continue;
        }
        node_mappings.emplace(node.id, mapping);
        result.node_map.emplace(
            node.id,
            graph.add_node(mapping->psycles_type, node.label));
    }

    for (const auto &node : source.nodes) {
        auto destination_iter = result.node_map.find(node.id);
        auto mapping_iter = node_mappings.find(node.id);
        if (destination_iter == result.node_map.end() ||
            mapping_iter == node_mappings.end()) {
            continue;
        }
        const auto destination = destination_iter->second;
        const auto &mapping = *mapping_iter->second;

        std::set<std::string, std::less<>> seen_inputs;
        for (const auto &input : node.inputs) {
            const auto *destination_socket =
                mapped_name(mapping.inputs, input.socket);
            if (destination_socket == nullptr) {
                diagnose(
                    CyclesAdapterDiagnosticCode::unknown_input,
                    node_prefix(node.id) + "unmapped input '" +
                        input.socket + "'",
                    node.id,
                    input.socket);
                continue;
            }
            if (!seen_inputs.emplace(*destination_socket).second ||
                input.source.has_value() == input.value.has_value()) {
                diagnose(
                    CyclesAdapterDiagnosticCode::invalid_binding,
                    node_prefix(node.id) + "input '" + input.socket +
                        "' must have exactly one source or literal value",
                    node.id,
                    input.socket);
                continue;
            }

            if (input.value) {
                if (!graph.set_input(
                        destination,
                        *destination_socket,
                        *input.value)) {
                    diagnose(
                        CyclesAdapterDiagnosticCode::invalid_node_reference,
                        node_prefix(node.id) +
                            "failed to set destination input",
                        node.id,
                        input.socket);
                }
                continue;
            }

            const auto &source_ref = *input.source;
            auto source_node_iter = result.node_map.find(source_ref.node);
            auto source_mapping_iter = node_mappings.find(source_ref.node);
            if (source_node_iter == result.node_map.end() ||
                source_mapping_iter == node_mappings.end()) {
                diagnose(
                    CyclesAdapterDiagnosticCode::invalid_node_reference,
                    node_prefix(node.id) + "input '" + input.socket +
                        "' references a missing source node",
                    node.id,
                    input.socket);
                continue;
            }
            const auto *source_socket = mapped_name(
                source_mapping_iter->second->outputs,
                source_ref.socket);
            if (source_socket == nullptr) {
                diagnose(
                    CyclesAdapterDiagnosticCode::unknown_output,
                    node_prefix(source_ref.node) + "unmapped output '" +
                        source_ref.socket + "'",
                    source_ref.node,
                    source_ref.socket);
                continue;
            }
            if (!graph.connect(
                    {.node = source_node_iter->second,
                     .socket = *source_socket},
                    destination,
                    *destination_socket)) {
                diagnose(
                    CyclesAdapterDiagnosticCode::invalid_node_reference,
                    node_prefix(node.id) +
                        "failed to connect destination input",
                    node.id,
                    input.socket);
            }
        }

        for (const auto &[name, value] : node.properties) {
            const auto *destination_property =
                mapped_name(mapping.properties, name);
            if (destination_property == nullptr) {
                diagnose(
                    CyclesAdapterDiagnosticCode::unknown_property,
                    node_prefix(node.id) + "unmapped static property '" +
                        name + "'",
                    node.id,
                    name);
                continue;
            }
            if (!graph.set_property(
                    destination, *destination_property, value)) {
                diagnose(
                    CyclesAdapterDiagnosticCode::invalid_binding,
                    node_prefix(node.id) +
                        "failed to set mapped static property '" +
                        name + "'",
                    node.id,
                    name);
            }
        }
    }

    for (std::size_t i = 0u;
         i < static_cast<std::size_t>(contract::ShaderDomain::count);
         ++i) {
        const auto domain = static_cast<contract::ShaderDomain>(i);
        const auto &root = source.root(domain);
        if (!root) {
            continue;
        }
        auto node_iter = result.node_map.find(root->node);
        auto mapping_iter = node_mappings.find(root->node);
        if (node_iter == result.node_map.end() ||
            mapping_iter == node_mappings.end()) {
            diagnose(
                CyclesAdapterDiagnosticCode::invalid_node_reference,
                "Cycles shader root references a missing node",
                root->node,
                root->socket);
            continue;
        }
        const auto *socket = mapped_name(
            mapping_iter->second->outputs, root->socket);
        if (socket == nullptr) {
            diagnose(
                CyclesAdapterDiagnosticCode::unknown_output,
                node_prefix(root->node) + "unmapped root output '" +
                    root->socket + "'",
                root->node,
                root->socket);
            continue;
        }
        graph.set_root(
            domain,
            contract::OutputRef{
                .node = node_iter->second,
                .socket = *socket});
    }

    if (result.diagnostics.empty()) {
        result.graph = std::move(graph);
    }
    return result;
}

}// namespace psycles::adapter
