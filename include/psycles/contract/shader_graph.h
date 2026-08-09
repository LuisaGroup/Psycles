#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <psycles/core/math.h>

namespace psycles::contract {

enum class SocketType : std::uint8_t {
    boolean,
    integer,
    unsigned_integer,
    floating,
    float2,
    float3,
    color,
    spectrum,
    point,
    vector,
    normal,
    transform,
    string,
    closure,
    volume_closure
};

using SocketLiteral = std::variant<
    bool,
    std::int64_t,
    std::uint64_t,
    float,
    Vec2f,
    Vec3f,
    Vec4f,
    Mat4f,
    std::string>;

struct SocketValue {
    SocketType type;
    SocketLiteral value;

    [[nodiscard]] bool well_typed() const noexcept;

    [[nodiscard]] static SocketValue boolean(bool value);
    [[nodiscard]] static SocketValue integer(std::int64_t value);
    [[nodiscard]] static SocketValue unsigned_integer(std::uint64_t value);
    [[nodiscard]] static SocketValue floating(float value);
    [[nodiscard]] static SocketValue float2(Vec2f value);
    [[nodiscard]] static SocketValue float3(Vec3f value);
    [[nodiscard]] static SocketValue color(Vec3f value);
    [[nodiscard]] static SocketValue spectrum(Vec3f value);
    [[nodiscard]] static SocketValue point(Vec3f value);
    [[nodiscard]] static SocketValue vector(Vec3f value);
    [[nodiscard]] static SocketValue normal(Vec3f value);
    [[nodiscard]] static SocketValue transform(Mat4f value);
    [[nodiscard]] static SocketValue string(std::string value);

    bool operator==(const SocketValue &) const noexcept = default;
};

struct SocketSchema {
    std::string name;
    SocketType type;
    bool required{false};
    std::optional<SocketValue> default_value;
};

// A node property is either part of the program's generated code shape or
// material data consumed by that program. This distinction is a compiler
// contract: runtime parameters are excluded from the structure signature
// only when surface lowering binds them through the typed parameter block.
enum class PropertyRole : std::uint8_t {
    code_shape,
    runtime_parameter
};

struct PropertySchema {
    std::string name;
    SocketType type;
    bool required{false};
    std::optional<SocketValue> default_value;
    PropertyRole role{PropertyRole::code_shape};
};

enum class ShaderFeature : std::uint64_t {
    none = 0u,
    surface = 1ull << 0u,
    emission = 1ull << 1u,
    transparency = 1ull << 2u,
    volume = 1ull << 3u,
    displacement = 1ull << 4u,
    derivatives = 1ull << 5u,
    attributes = 1ull << 6u,
    ray_state = 1ull << 7u,
    subsurface = 1ull << 8u
};

[[nodiscard]] constexpr std::uint64_t feature_bit(ShaderFeature feature) noexcept {
    return static_cast<std::uint64_t>(feature);
}

struct NodeSchema {
    std::string type;
    std::vector<SocketSchema> inputs;
    std::vector<SocketSchema> outputs;
    std::vector<PropertySchema> properties;
    std::uint64_t required_features{};
};

class NodeRegistry {

private:
    std::map<std::string, NodeSchema, std::less<>> _schemas;

public:
    [[nodiscard]] bool register_schema(NodeSchema schema);
    [[nodiscard]] const NodeSchema *find(std::string_view type) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return _schemas.size(); }
};

struct NodeId {
    static constexpr auto invalid_value = ~std::uint32_t{0u};

    std::uint32_t value{invalid_value};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return value != invalid_value;
    }

    auto operator<=>(const NodeId &) const noexcept = default;
};

struct OutputRef {
    NodeId node;
    std::string socket;

    bool operator==(const OutputRef &) const noexcept = default;
};

struct InputBinding {
    std::optional<OutputRef> source;
    std::optional<SocketValue> value;
};

struct ShaderNode {
    NodeId id;
    std::string type;
    std::string label;
    std::map<std::string, InputBinding, std::less<>> inputs;
    std::map<std::string, SocketValue, std::less<>> properties;
};

enum class ShaderDomain : std::uint8_t {
    surface,
    volume,
    // Cycles evaluates an unconnected automatic-bump normal as a shader
    // side effect before closure setup. Keep that observable value separate
    // from the geometric displacement vector: DISPLACEMENT uses only the
    // latter, while BOTH deliberately retains both roots.
    surface_normal,
    displacement,
    count
};

class ShaderGraph {

private:
    std::vector<ShaderNode> _nodes;
    std::array<std::optional<OutputRef>,
               static_cast<std::size_t>(ShaderDomain::count)>
        _roots;
    std::uint32_t _next_node_id{};

public:
    [[nodiscard]] NodeId add_node(std::string type, std::string label = {});
    [[nodiscard]] bool set_input(NodeId node, std::string socket, SocketValue value);
    [[nodiscard]] bool set_property(NodeId node, std::string property, SocketValue value);
    [[nodiscard]] bool connect(OutputRef source, NodeId destination, std::string input_socket);
    void set_root(ShaderDomain domain, std::optional<OutputRef> root);

    [[nodiscard]] ShaderNode *find(NodeId id) noexcept;
    [[nodiscard]] const ShaderNode *find(NodeId id) const noexcept;
    [[nodiscard]] const std::vector<ShaderNode> &nodes() const noexcept { return _nodes; }
    [[nodiscard]] const std::optional<OutputRef> &root(ShaderDomain domain) const noexcept;
};

enum class DiagnosticSeverity : std::uint8_t {
    warning,
    error
};

enum class GraphDiagnosticCode : std::uint8_t {
    no_output_root,
    duplicate_node_id,
    unknown_node_type,
    unknown_input,
    unknown_output,
    unknown_property,
    missing_required_input,
    missing_required_property,
    ill_typed_literal,
    socket_type_mismatch,
    invalid_node_reference,
    invalid_root_type,
    cycle
};

struct GraphDiagnostic {
    DiagnosticSeverity severity{DiagnosticSeverity::error};
    GraphDiagnosticCode code{};
    std::string message;
    std::optional<NodeId> node;
    std::string socket;
};

struct RuntimePropertyRef {
    NodeId node;
    std::string property;
    SocketType type{};
};

struct GraphAnalysis {
    std::vector<NodeId> evaluation_order;
    std::vector<RuntimePropertyRef> runtime_properties;
    std::uint64_t required_features{};
    std::uint64_t structure_signature{};
    std::uint64_t parameter_signature{};
};

struct GraphValidation {
    std::vector<GraphDiagnostic> diagnostics;
    std::optional<GraphAnalysis> analysis;

    [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] GraphValidation validate_shader_graph(
    const ShaderGraph &graph,
    const NodeRegistry &registry);

}// namespace psycles::contract
