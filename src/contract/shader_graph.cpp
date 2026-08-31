#include <psycles/contract/shader_graph.h>

#include <algorithm>
#include <bit>
#include <functional>
#include <queue>
#include <set>
#include <sstream>
#include <type_traits>

namespace psycles::contract {

namespace {

[[nodiscard]] bool literal_matches(SocketType type, const SocketLiteral &literal) noexcept {
    switch (type) {
        case SocketType::boolean:
            return std::holds_alternative<bool>(literal);
        case SocketType::integer:
            return std::holds_alternative<std::int64_t>(literal);
        case SocketType::unsigned_integer:
            return std::holds_alternative<std::uint64_t>(literal);
        case SocketType::floating:
            return std::holds_alternative<float>(literal);
        case SocketType::float2:
            return std::holds_alternative<Vec2f>(literal);
        case SocketType::float3:
        case SocketType::color:
        case SocketType::spectrum:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal:
            return std::holds_alternative<Vec3f>(literal);
        case SocketType::transform:
            return std::holds_alternative<Mat4f>(literal);
        case SocketType::string:
            return std::holds_alternative<std::string>(literal);
        case SocketType::closure:
        case SocketType::volume_closure:
            return false;
    }
    return false;
}

template<typename Schema>
[[nodiscard]] const Schema *find_schema(
    const std::vector<Schema> &schemas,
    std::string_view name) noexcept {
    auto iter = std::find_if(
        schemas.begin(), schemas.end(),
        [&](const auto &schema) { return schema.name == name; });
    return iter == schemas.end() ? nullptr : &*iter;
}

[[nodiscard]] const SocketSchema *find_output(
    const NodeSchema &schema,
    std::string_view name) noexcept {
    return find_schema(schema.outputs, name);
}

[[nodiscard]] bool root_type_matches(
    ShaderDomain domain,
    SocketType type) noexcept {
    switch (domain) {
        case ShaderDomain::surface:
            return type == SocketType::closure;
        case ShaderDomain::volume:
            // Cycles has one Shader socket type. Emission therefore serves
            // both the Surface and Volume inputs of ShaderNodeOutputMaterial.
            return type == SocketType::closure ||
                   type == SocketType::volume_closure;
        case ShaderDomain::surface_normal:
            return type == SocketType::normal ||
                   type == SocketType::vector ||
                   type == SocketType::float3;
        case ShaderDomain::displacement:
            return type == SocketType::vector || type == SocketType::float3;
        case ShaderDomain::count:
            return false;
    }
    return false;
}

class StableHash {

private:
    static constexpr std::uint64_t offset = 14695981039346656037ull;
    static constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t _value{offset};

public:
    void bytes(const void *data, std::size_t size) noexcept {
        const auto *p = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0u; i < size; ++i) {
            _value ^= static_cast<std::uint64_t>(p[i]);
            _value *= prime;
        }
    }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    void scalar(const T &value) noexcept {
        bytes(&value, sizeof(value));
    }

    void string(std::string_view value) noexcept {
        scalar(value.size());
        bytes(value.data(), value.size());
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return _value; }
};

void hash_socket_value(StableHash &hash, const SocketValue &value) noexcept {
    hash.scalar(value.type);
    std::visit(
        [&](const auto &literal) noexcept {
            using T = std::decay_t<decltype(literal)>;
            if constexpr (std::is_same_v<T, std::string>) {
                hash.string(literal);
            } else if constexpr (std::is_same_v<T, Vec2f>) {
                hash.scalar(std::bit_cast<std::uint32_t>(literal.x));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.y));
            } else if constexpr (std::is_same_v<T, Vec3f>) {
                hash.scalar(std::bit_cast<std::uint32_t>(literal.x));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.y));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.z));
            } else if constexpr (std::is_same_v<T, Vec4f>) {
                hash.scalar(std::bit_cast<std::uint32_t>(literal.x));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.y));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.z));
                hash.scalar(std::bit_cast<std::uint32_t>(literal.w));
            } else if constexpr (std::is_same_v<T, Mat4f>) {
                for (auto element : literal.elements) {
                    hash.scalar(std::bit_cast<std::uint32_t>(element));
                }
            } else if constexpr (std::is_same_v<T, float>) {
                hash.scalar(std::bit_cast<std::uint32_t>(literal));
            } else {
                hash.scalar(literal);
            }
        },
        value.value);
}

[[nodiscard]] std::string node_prefix(NodeId node) {
    return "node " + std::to_string(node.value) + ": ";
}

}// namespace

bool SocketValue::well_typed() const noexcept {
    return literal_matches(type, value);
}

SocketValue SocketValue::boolean(bool value) {
    return {SocketType::boolean, value};
}
SocketValue SocketValue::integer(std::int64_t value) {
    return {SocketType::integer, value};
}
SocketValue SocketValue::unsigned_integer(std::uint64_t value) {
    return {SocketType::unsigned_integer, value};
}
SocketValue SocketValue::floating(float value) {
    return {SocketType::floating, value};
}
SocketValue SocketValue::float2(Vec2f value) {
    return {SocketType::float2, value};
}
SocketValue SocketValue::float3(Vec3f value) {
    return {SocketType::float3, value};
}
SocketValue SocketValue::color(Vec3f value) {
    return {SocketType::color, value};
}
SocketValue SocketValue::spectrum(Vec3f value) {
    return {SocketType::spectrum, value};
}
SocketValue SocketValue::point(Vec3f value) {
    return {SocketType::point, value};
}
SocketValue SocketValue::vector(Vec3f value) {
    return {SocketType::vector, value};
}
SocketValue SocketValue::normal(Vec3f value) {
    return {SocketType::normal, value};
}
SocketValue SocketValue::transform(Mat4f value) {
    return {SocketType::transform, value};
}
SocketValue SocketValue::string(std::string value) {
    return {SocketType::string, std::move(value)};
}

bool NodeRegistry::register_schema(NodeSchema schema) {
    if (schema.type.empty() || _schemas.contains(schema.type)) {
        return false;
    }

    auto unique_names = [](const auto &items) {
        std::set<std::string, std::less<>> names;
        return std::all_of(items.begin(), items.end(), [&](const auto &item) {
            return !item.name.empty() && names.emplace(item.name).second;
        });
    };
    if (!unique_names(schema.inputs) ||
        !unique_names(schema.outputs) ||
        !unique_names(schema.properties)) {
        return false;
    }
    if (std::any_of(
            schema.inputs.begin(), schema.inputs.end(),
            [](const auto &item) {
                return item.default_value &&
                       (item.default_value->type != item.type ||
                        !item.default_value->well_typed());
            }) ||
        std::any_of(
            schema.properties.begin(), schema.properties.end(),
            [](const auto &item) {
                return item.default_value &&
                       (item.default_value->type != item.type ||
                        !item.default_value->well_typed());
            })) {
        return false;
    }

    _schemas.emplace(schema.type, std::move(schema));
    return true;
}

const NodeSchema *NodeRegistry::find(std::string_view type) const noexcept {
    auto iter = _schemas.find(type);
    return iter == _schemas.end() ? nullptr : &iter->second;
}

NodeId ShaderGraph::add_node(std::string type, std::string label) {
    auto id = NodeId{_next_node_id++};
    _nodes.emplace_back(ShaderNode{
        .id = id,
        .type = std::move(type),
        .label = std::move(label),
        .inputs = {},
        .properties = {}});
    return id;
}

bool ShaderGraph::set_input(
    NodeId node,
    std::string socket,
    SocketValue value) {
    auto *n = find(node);
    if (n == nullptr) {
        return false;
    }
    n->inputs[std::move(socket)] = InputBinding{
        .source = std::nullopt,
        .value = std::move(value)};
    return true;
}

bool ShaderGraph::set_property(
    NodeId node,
    std::string property,
    SocketValue value) {
    auto *n = find(node);
    if (n == nullptr) {
        return false;
    }
    n->properties.insert_or_assign(std::move(property), std::move(value));
    return true;
}

bool ShaderGraph::connect(
    OutputRef source,
    NodeId destination,
    std::string input_socket) {
    auto *n = find(destination);
    if (n == nullptr) {
        return false;
    }
    // Cycles ShaderInput stores its socket value independently of its link.
    // Preserve an authored fallback so graph rewrites which disconnect the
    // link observe the same value as Cycles' constant folder.
    auto &binding = n->inputs[std::move(input_socket)];
    binding.source = std::move(source);
    return true;
}

void ShaderGraph::set_root(
    ShaderDomain domain,
    std::optional<OutputRef> root) {
    _roots[static_cast<std::size_t>(domain)] = std::move(root);
}

ShaderNode *ShaderGraph::find(NodeId id) noexcept {
    auto iter = std::find_if(
        _nodes.begin(), _nodes.end(),
        [&](const auto &node) { return node.id == id; });
    return iter == _nodes.end() ? nullptr : &*iter;
}

const ShaderNode *ShaderGraph::find(NodeId id) const noexcept {
    auto iter = std::find_if(
        _nodes.begin(), _nodes.end(),
        [&](const auto &node) { return node.id == id; });
    return iter == _nodes.end() ? nullptr : &*iter;
}

const std::optional<OutputRef> &ShaderGraph::root(ShaderDomain domain) const noexcept {
    return _roots[static_cast<std::size_t>(domain)];
}

bool GraphValidation::ok() const noexcept {
    return analysis.has_value() &&
           std::none_of(
               diagnostics.begin(), diagnostics.end(),
               [](const auto &diagnostic) {
                   return diagnostic.severity == DiagnosticSeverity::error;
               });
}

GraphValidation validate_shader_graph(
    const ShaderGraph &graph,
    const NodeRegistry &registry) {

    GraphValidation result;
    auto diagnose = [&](GraphDiagnosticCode code,
                        std::string message,
                        std::optional<NodeId> node = std::nullopt,
                        std::string socket = {}) {
        result.diagnostics.emplace_back(GraphDiagnostic{
            .severity = DiagnosticSeverity::error,
            .code = code,
            .message = std::move(message),
            .node = node,
            .socket = std::move(socket)});
    };

    std::map<NodeId, const ShaderNode *> nodes;
    for (const auto &node : graph.nodes()) {
        if (!nodes.emplace(node.id, &node).second) {
            diagnose(
                GraphDiagnosticCode::duplicate_node_id,
                node_prefix(node.id) + "duplicate node identifier",
                node.id);
        }
    }

    bool has_root = false;
    for (std::size_t i = 0u;
         i < static_cast<std::size_t>(ShaderDomain::count);
         ++i) {
        has_root |= graph.root(static_cast<ShaderDomain>(i)).has_value();
    }
    if (!has_root) {
        diagnose(
            GraphDiagnosticCode::no_output_root,
            "shader graph has no surface, volume, or displacement root");
    }

    for (const auto &[id, node] : nodes) {
        const auto *schema = registry.find(node->type);
        if (schema == nullptr) {
            diagnose(
                GraphDiagnosticCode::unknown_node_type,
                node_prefix(id) + "unregistered node type '" + node->type + "'",
                id);
            continue;
        }

        for (const auto &[name, binding] : node->inputs) {
            const auto *input_schema = find_schema(schema->inputs, name);
            if (input_schema == nullptr) {
                diagnose(
                    GraphDiagnosticCode::unknown_input,
                    node_prefix(id) + "unknown input '" + name + "'",
                    id,
                    name);
                continue;
            }

            if (binding.value) {
                if (!binding.value->well_typed()) {
                    diagnose(
                        GraphDiagnosticCode::ill_typed_literal,
                        node_prefix(id) + "input '" + name +
                            "' contains an ill-typed literal",
                        id,
                        name);
                } else if (binding.value->type != input_schema->type) {
                    diagnose(
                        GraphDiagnosticCode::socket_type_mismatch,
                        node_prefix(id) + "input '" + name +
                            "' literal type does not match the schema",
                        id,
                        name);
                }
            }

            if (binding.source) {
                auto source_iter = nodes.find(binding.source->node);
                if (source_iter == nodes.end()) {
                    diagnose(
                        GraphDiagnosticCode::invalid_node_reference,
                        node_prefix(id) + "input '" + name +
                            "' references a missing source node",
                        id,
                        name);
                    continue;
                }
                const auto *source_schema = registry.find(source_iter->second->type);
                if (source_schema == nullptr) {
                    continue;
                }
                const auto *source_output =
                    find_output(*source_schema, binding.source->socket);
                if (source_output == nullptr) {
                    diagnose(
                        GraphDiagnosticCode::unknown_output,
                        node_prefix(binding.source->node) + "unknown output '" +
                            binding.source->socket + "'",
                        binding.source->node,
                        binding.source->socket);
                } else if (source_output->type != input_schema->type) {
                    diagnose(
                        GraphDiagnosticCode::socket_type_mismatch,
                        node_prefix(id) + "input '" + name +
                            "' is connected to an incompatible output",
                        id,
                        name);
                }
            }
        }

        for (const auto &input_schema : schema->inputs) {
            if (input_schema.required &&
                !input_schema.default_value &&
                !node->inputs.contains(input_schema.name)) {
                diagnose(
                    GraphDiagnosticCode::missing_required_input,
                    node_prefix(id) + "missing required input '" +
                        input_schema.name + "'",
                    id,
                    input_schema.name);
            }
        }

        for (const auto &[name, value] : node->properties) {
            const auto *property_schema = find_schema(schema->properties, name);
            if (property_schema == nullptr) {
                diagnose(
                    GraphDiagnosticCode::unknown_property,
                    node_prefix(id) + "unknown static property '" + name + "'",
                    id,
                    name);
            } else if (!value.well_typed()) {
                diagnose(
                    GraphDiagnosticCode::ill_typed_literal,
                    node_prefix(id) + "property '" + name +
                        "' contains an ill-typed literal",
                    id,
                    name);
            } else if (value.type != property_schema->type) {
                diagnose(
                    GraphDiagnosticCode::socket_type_mismatch,
                    node_prefix(id) + "property '" + name +
                        "' type does not match the schema",
                    id,
                    name);
            }
        }

        for (const auto &property_schema : schema->properties) {
            if (property_schema.required &&
                !property_schema.default_value &&
                !node->properties.contains(property_schema.name)) {
                diagnose(
                    GraphDiagnosticCode::missing_required_property,
                    node_prefix(id) + "missing required property '" +
                        property_schema.name + "'",
                    id,
                    property_schema.name);
            }
        }
    }

    std::set<NodeId> reachable;
    std::function<void(NodeId)> mark_reachable = [&](NodeId id) {
        if (!reachable.emplace(id).second) {
            return;
        }
        auto iter = nodes.find(id);
        if (iter == nodes.end()) {
            return;
        }
        for (const auto &[_, binding] : iter->second->inputs) {
            if (binding.source) {
                mark_reachable(binding.source->node);
            }
        }
    };

    for (std::size_t i = 0u;
         i < static_cast<std::size_t>(ShaderDomain::count);
         ++i) {
        auto domain = static_cast<ShaderDomain>(i);
        const auto &root = graph.root(domain);
        if (!root) {
            continue;
        }
        auto node_iter = nodes.find(root->node);
        if (node_iter == nodes.end()) {
            diagnose(
                GraphDiagnosticCode::invalid_node_reference,
                "shader root references a missing node",
                root->node,
                root->socket);
            continue;
        }
        const auto *schema = registry.find(node_iter->second->type);
        if (schema == nullptr) {
            continue;
        }
        const auto *root_output = find_output(*schema, root->socket);
        if (root_output == nullptr) {
            diagnose(
                GraphDiagnosticCode::unknown_output,
                node_prefix(root->node) + "root references unknown output '" +
                    root->socket + "'",
                root->node,
                root->socket);
            continue;
        }
        if (!root_type_matches(domain, root_output->type)) {
            diagnose(
                GraphDiagnosticCode::invalid_root_type,
                "shader root output has an incompatible type",
                root->node,
                root->socket);
        }
        mark_reachable(root->node);
    }

    std::map<NodeId, std::size_t> indegree;
    std::map<NodeId, std::vector<NodeId>> outgoing;
    for (auto id : reachable) {
        indegree.emplace(id, 0u);
    }
    for (auto destination : reachable) {
        const auto *node = nodes.at(destination);
        for (const auto &[_, binding] : node->inputs) {
            if (binding.source && reachable.contains(binding.source->node)) {
                outgoing[binding.source->node].emplace_back(destination);
                ++indegree[destination];
            }
        }
    }

    std::priority_queue<
        NodeId,
        std::vector<NodeId>,
        std::greater<>>
        ready;
    for (const auto &[id, degree] : indegree) {
        if (degree == 0u) {
            ready.push(id);
        }
    }

    std::vector<NodeId> order;
    while (!ready.empty()) {
        auto id = ready.top();
        ready.pop();
        order.emplace_back(id);
        for (auto dependent : outgoing[id]) {
            auto &degree = indegree[dependent];
            --degree;
            if (degree == 0u) {
                ready.push(dependent);
            }
        }
    }

    if (order.size() != reachable.size()) {
        diagnose(
            GraphDiagnosticCode::cycle,
            "reachable shader graph contains a cycle");
    }

    const auto has_error = std::any_of(
        result.diagnostics.begin(), result.diagnostics.end(),
        [](const auto &diagnostic) {
            return diagnostic.severity == DiagnosticSeverity::error;
        });
    if (has_error) {
        return result;
    }

    std::map<NodeId, std::size_t> ordinal;
    for (std::size_t i = 0u; i < order.size(); ++i) {
        ordinal.emplace(order[i], i);
    }

    StableHash structure_hash;
    StableHash parameter_hash;
    std::vector<RuntimePropertyRef> runtime_properties;
    std::uint64_t features{};

    for (auto id : order) {
        const auto *node = nodes.at(id);
        const auto *schema = registry.find(node->type);
        structure_hash.string(node->type);
        features |= schema->required_features;

        for (const auto &property : schema->properties) {
            structure_hash.string(property.name);
            structure_hash.scalar(property.type);
            structure_hash.scalar(property.role);
            auto value_iter = node->properties.find(property.name);
            const auto *value =
                value_iter != node->properties.end()
                    ? &value_iter->second
                    : property.default_value
                          ? &*property.default_value
                          : nullptr;
            if (property.role == PropertyRole::runtime_parameter) {
                runtime_properties.emplace_back(RuntimePropertyRef{
                    .node = id,
                    .property = property.name,
                    .type = property.type});
                parameter_hash.string(node->type);
                parameter_hash.string(property.name);
                if (value != nullptr) {
                    hash_socket_value(parameter_hash, *value);
                }
            } else if (value != nullptr) {
                hash_socket_value(structure_hash, *value);
            }
        }

        for (const auto &input : schema->inputs) {
            structure_hash.string(input.name);
            structure_hash.scalar(input.type);
            auto binding_iter = node->inputs.find(input.name);
            if (binding_iter != node->inputs.end() &&
                binding_iter->second.source) {
                const auto &source = *binding_iter->second.source;
                structure_hash.scalar(ordinal.at(source.node));
                structure_hash.string(source.socket);
            } else {
                constexpr std::uint8_t unlinked = 0xffu;
                structure_hash.scalar(unlinked);
                parameter_hash.string(node->type);
                parameter_hash.string(input.name);
                if (binding_iter != node->inputs.end() &&
                    binding_iter->second.value) {
                    hash_socket_value(
                        parameter_hash, *binding_iter->second.value);
                } else if (input.default_value) {
                    hash_socket_value(parameter_hash, *input.default_value);
                }
            }
        }
    }

    for (std::size_t i = 0u;
         i < static_cast<std::size_t>(ShaderDomain::count);
         ++i) {
        auto domain = static_cast<ShaderDomain>(i);
        structure_hash.scalar(domain);
        const auto &root = graph.root(domain);
        if (root) {
            structure_hash.scalar(ordinal.at(root->node));
            structure_hash.string(root->socket);
        } else {
            constexpr std::uint8_t absent = 0u;
            structure_hash.scalar(absent);
        }
    }

    result.analysis = GraphAnalysis{
        .evaluation_order = std::move(order),
        .runtime_properties = std::move(runtime_properties),
        .required_features = features,
        .structure_signature = structure_hash.value(),
        .parameter_signature = parameter_hash.value()};
    return result;
}

}// namespace psycles::contract
