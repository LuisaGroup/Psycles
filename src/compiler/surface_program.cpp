#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/surface_program.h>

#include <cstdlib>
#include <map>
#include <string_view>
#include <utility>
#include <variant>

namespace psycles::compiler {

namespace {

struct OutputKey {
    contract::NodeId node;
    std::string socket;

    auto operator<=>(const OutputKey &) const noexcept = default;
};

using LoweredOutput =
    std::variant<ValueExpressionId, ClosureExpressionId>;

[[nodiscard]] const contract::InputBinding *find_input(
    const contract::ShaderNode &node,
    std::string_view name) noexcept {
    auto iter = node.inputs.find(name);
    return iter == node.inputs.end() ? nullptr : &iter->second;
}

[[nodiscard]] const contract::SocketValue *find_property(
    const contract::ShaderNode &node,
    std::string_view name) noexcept {
    auto iter = node.properties.find(name);
    return iter == node.properties.end() ? nullptr : &iter->second;
}

[[nodiscard]] std::string property_string(
    const contract::ShaderNode &node,
    std::string_view name,
    std::string fallback = {}) {
    const auto *value = find_property(node, name);
    if (value == nullptr ||
        value->type != contract::SocketType::string) {
        return fallback;
    }
    return std::get<std::string>(value->value);
}

[[nodiscard]] std::uint64_t property_uint(
    const contract::ShaderNode &node,
    std::string_view name,
    std::uint64_t fallback = 0u) noexcept {
    const auto *value = find_property(node, name);
    if (value == nullptr ||
        value->type != contract::SocketType::unsigned_integer) {
        return fallback;
    }
    return std::get<std::uint64_t>(value->value);
}

[[nodiscard]] bool property_bool(
    const contract::ShaderNode &node,
    std::string_view name,
    bool fallback = false) noexcept {
    const auto *value = find_property(node, name);
    if (value == nullptr ||
        value->type != contract::SocketType::boolean) {
        return fallback;
    }
    return std::get<bool>(value->value);
}

[[nodiscard]] float property_float(
    const contract::ShaderNode &node,
    std::string_view name,
    float fallback = 0.0f) noexcept {
    const auto *value = find_property(node, name);
    if (value == nullptr ||
        value->type != contract::SocketType::floating) {
        return fallback;
    }
    return std::get<float>(value->value);
}

[[nodiscard]] std::vector<float> parse_float_table(
    const std::string &encoded) {
    std::vector<float> result;
    const char *cursor = encoded.c_str();
    const char *end = cursor + encoded.size();
    while (cursor < end) {
        while (cursor < end &&
               (*cursor == ',' || *cursor == ';' ||
                *cursor == ' ' || *cursor == '\t' ||
                *cursor == '\n' || *cursor == '\r')) {
            ++cursor;
        }
        if (cursor == end) {
            break;
        }
        char *next = nullptr;
        const auto value = std::strtof(cursor, &next);
        if (next == cursor) {
            break;
        }
        result.emplace_back(value);
        cursor = next;
    }
    return result;
}

[[nodiscard]] std::string node_prefix(contract::NodeId node) {
    return "node " + std::to_string(node.value) + ": ";
}

class SurfaceProgramBuilder {

private:
    const ShaderProgram &_shader;
    std::vector<SurfaceProgramDiagnostic> _diagnostics;
    std::vector<ParameterDesc> _parameters;
    std::vector<ValueInstruction> _value_instructions;
    std::vector<ClosureInstruction> _closure_instructions;
    std::map<OutputKey, LoweredOutput> _outputs;

private:
    void diagnose(
        SurfaceProgramDiagnosticCode code,
        std::string message,
        std::optional<contract::NodeId> node = std::nullopt,
        std::string socket = {}) {
        _diagnostics.emplace_back(SurfaceProgramDiagnostic{
            .code = code,
            .message = std::move(message),
            .node = node,
            .socket = std::move(socket)});
    }

    [[nodiscard]] ParameterId add_parameter(
        const contract::ShaderNode &node,
        std::string_view socket,
        const contract::SocketValue &value) {
        auto id =
            ParameterId{static_cast<std::uint32_t>(_parameters.size())};
        _parameters.emplace_back(ParameterDesc{
            .id = id,
            .node = node.id,
            .socket = std::string{socket},
            .type = value.type,
            .default_value = value});
        return id;
    }

    [[nodiscard]] ValueExpressionId append(
        ValueInstruction instruction) {
        auto id = ValueExpressionId{
            static_cast<std::uint32_t>(_value_instructions.size())};
        _value_instructions.emplace_back(std::move(instruction));
        return id;
    }

    [[nodiscard]] ClosureExpressionId append(
        ClosureInstruction instruction) {
        auto id = ClosureExpressionId{
            static_cast<std::uint32_t>(_closure_instructions.size())};
        _closure_instructions.emplace_back(std::move(instruction));
        return id;
    }

    template<typename Id>
    [[nodiscard]] std::optional<Id> source_output(
        const contract::ShaderNode &node,
        std::string_view socket,
        const contract::InputBinding &binding) {
        if (!binding.source) {
            return std::nullopt;
        }
        auto iter = _outputs.find(
            {.node = binding.source->node,
             .socket = binding.source->socket});
        if (iter == _outputs.end()) {
            diagnose(
                SurfaceProgramDiagnosticCode::missing_output,
                node_prefix(node.id) + "input '" + std::string{socket} +
                    "' references an output that was not lowered",
                node.id,
                std::string{socket});
            return std::nullopt;
        }
        if (const auto *id = std::get_if<Id>(&iter->second)) {
            return *id;
        }
        diagnose(
            SurfaceProgramDiagnosticCode::type_mismatch,
            node_prefix(node.id) + "input '" + std::string{socket} +
                "' has an incompatible lowered type",
            node.id,
            std::string{socket});
        return std::nullopt;
    }

    [[nodiscard]] std::optional<ValueExpressionId> lower_value_input(
        const contract::ShaderNode &node,
        std::string_view socket) {
        const auto *binding = find_input(node, socket);
        if (binding == nullptr) {
            diagnose(
                SurfaceProgramDiagnosticCode::missing_input,
                node_prefix(node.id) + "missing normalized input '" +
                    std::string{socket} + "'",
                node.id,
                std::string{socket});
            return std::nullopt;
        }
        if (binding->source) {
            return source_output<ValueExpressionId>(
                node, socket, *binding);
        }
        if (!binding->value) {
            diagnose(
                SurfaceProgramDiagnosticCode::missing_input,
                node_prefix(node.id) + "input '" + std::string{socket} +
                    "' has no value",
                node.id,
                std::string{socket});
            return std::nullopt;
        }
        auto parameter = add_parameter(node, socket, *binding->value);
        return append(ValueInstruction{
            .operation = ValueOperation::parameter,
            .source_node = node.id,
            .result_type = binding->value->type,
            .parameter = parameter});
    }

    [[nodiscard]] std::optional<ClosureExpressionId>
    lower_closure_input(
        const contract::ShaderNode &node,
        std::string_view socket) {
        const auto *binding = find_input(node, socket);
        if (binding == nullptr || !binding->source) {
            diagnose(
                SurfaceProgramDiagnosticCode::missing_input,
                node_prefix(node.id) + "closure input '" +
                    std::string{socket} + "' is not connected",
                node.id,
                std::string{socket});
            return std::nullopt;
        }
        return source_output<ClosureExpressionId>(
            node, socket, *binding);
    }

    void publish(
        contract::NodeId node,
        std::string socket,
        LoweredOutput output) {
        _outputs.insert_or_assign(
            OutputKey{.node = node, .socket = std::move(socket)},
            std::move(output));
    }

    void publish_unary_value(
        const contract::ShaderNode &node,
        std::string_view input,
        std::string output,
        contract::SocketType result_type,
        ValueOperation operation) {
        if (auto value = lower_value_input(node, input)) {
            publish(
                node.id,
                std::move(output),
                append(ValueInstruction{
                    .operation = operation,
                    .source_node = node.id,
                    .result_type = result_type,
                    .a = *value}));
        }
    }

    void publish_binary_value(
        const contract::ShaderNode &node,
        std::string_view input_a,
        std::string_view input_b,
        std::string output,
        contract::SocketType result_type,
        ValueOperation operation) {
        auto a = lower_value_input(node, input_a);
        auto b = lower_value_input(node, input_b);
        if (a && b) {
            publish(
                node.id,
                std::move(output),
                append(ValueInstruction{
                    .operation = operation,
                    .source_node = node.id,
                    .result_type = result_type,
                    .a = *a,
                    .b = *b}));
        }
    }

    void lower_node(const contract::ShaderNode &node) {
        using contract::SocketType;

        if (node.type == node_type::constant_float) {
            if (auto value = lower_value_input(node, "Value")) {
                publish(node.id, "Value", *value);
            }
            return;
        }
        if (node.type == node_type::constant_color) {
            if (auto value = lower_value_input(node, "Color")) {
                publish(node.id, "Color", *value);
            }
            return;
        }
        if (node.type == node_type::geometry) {
            publish(
                node.id,
                "Position",
                append(ValueInstruction{
                    .operation = ValueOperation::surface_position,
                    .source_node = node.id,
                    .result_type = SocketType::point}));
            publish(
                node.id,
                "Normal",
                append(ValueInstruction{
                    .operation = ValueOperation::shading_normal,
                    .source_node = node.id,
                    .result_type = SocketType::normal}));
            publish(
                node.id,
                "GeometricNormal",
                append(ValueInstruction{
                    .operation = ValueOperation::geometric_normal,
                    .source_node = node.id,
                    .result_type = SocketType::normal}));
            publish(
                node.id,
                "Incoming",
                append(ValueInstruction{
                    .operation = ValueOperation::incoming,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Tangent",
                append(ValueInstruction{
                    .operation = ValueOperation::tangent,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Backfacing",
                append(ValueInstruction{
                    .operation = ValueOperation::back_facing,
                    .source_node = node.id,
                    .result_type = SocketType::floating}));
            publish(
                node.id,
                "RandomPerIsland",
                append(ValueInstruction{
                    .operation = ValueOperation::random_per_island,
                    .source_node = node.id,
                    .result_type = SocketType::floating}));
            return;
        }
        if (node.type == node_type::texture_coordinate) {
            publish(
                node.id,
                "UV",
                append(ValueInstruction{
                    .operation = ValueOperation::uv,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Normal",
                append(ValueInstruction{
                    .operation = ValueOperation::shading_normal,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Generated",
                append(ValueInstruction{
                    .operation = ValueOperation::generated,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Object",
                append(ValueInstruction{
                    .operation = ValueOperation::object_position,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            return;
        }
        if (node.type == node_type::object_info) {
            publish(
                node.id,
                "Location",
                append(ValueInstruction{
                    .operation = ValueOperation::object_location,
                    .source_node = node.id,
                    .result_type = SocketType::vector}));
            publish(
                node.id,
                "Random",
                append(ValueInstruction{
                    .operation = ValueOperation::object_random,
                    .source_node = node.id,
                    .result_type = SocketType::floating}));
            return;
        }
        if (node.type == node_type::light_path) {
            for (const auto &[output_name, operation] : {
                     std::pair{
                         "IsCameraRay",
                         ValueOperation::path_is_camera},
                     std::pair{
                         "IsShadowRay",
                         ValueOperation::path_is_shadow},
                     std::pair{
                         "IsDiffuseRay",
                         ValueOperation::path_is_diffuse},
                     std::pair{
                         "IsGlossyRay",
                         ValueOperation::path_is_glossy},
                     std::pair{
                         "IsSingularRay",
                         ValueOperation::path_is_singular},
                     std::pair{
                         "IsReflectionRay",
                         ValueOperation::path_is_reflection},
                     std::pair{
                         "IsTransmissionRay",
                         ValueOperation::path_is_transmission},
                     std::pair{
                         "IsVolumeScatterRay",
                         ValueOperation::path_is_volume_scatter},
                     std::pair{
                         "RayLength",
                         ValueOperation::path_ray_length},
                     std::pair{
                         "RayDepth",
                         ValueOperation::path_ray_depth},
                     std::pair{
                         "DiffuseDepth",
                         ValueOperation::path_diffuse_depth},
                     std::pair{
                         "GlossyDepth",
                         ValueOperation::path_glossy_depth},
                     std::pair{
                         "TransparentDepth",
                         ValueOperation::path_transparent_depth},
                     std::pair{
                         "TransmissionDepth",
                         ValueOperation::path_transmission_depth}}) {
                publish(
                    node.id,
                    output_name,
                    append(ValueInstruction{
                        .operation = operation,
                        .source_node = node.id,
                        .result_type = SocketType::floating}));
            }
            return;
        }
        if (node.type == node_type::layer_weight) {
            auto blend = lower_value_input(node, "Blend");
            auto normal = lower_value_input(node, "Normal");
            if (blend && normal) {
                publish(
                    node.id,
                    "Fresnel",
                    append(ValueInstruction{
                        .operation =
                            ValueOperation::layer_weight_fresnel,
                        .source_node = node.id,
                        .result_type = SocketType::floating,
                        .a = *blend,
                        .b = *normal}));
                publish(
                    node.id,
                    "Facing",
                    append(ValueInstruction{
                        .operation =
                            ValueOperation::layer_weight_facing,
                        .source_node = node.id,
                        .result_type = SocketType::floating,
                        .a = *blend,
                        .b = *normal}));
            }
            return;
        }
        if (node.type == node_type::mapping) {
            auto vector = lower_value_input(node, "Vector");
            auto location = lower_value_input(node, "Location");
            auto rotation = lower_value_input(node, "Rotation");
            auto scale = lower_value_input(node, "Scale");
            if (vector && location && rotation && scale) {
                const auto vector_type =
                    property_string(node, "VectorType", "POINT");
                const auto mode =
                    vector_type == "TEXTURE"
                        ? 1u
                        : vector_type == "VECTOR"
                              ? 2u
                              : vector_type == "NORMAL" ? 3u : 0u;
                publish(
                    node.id,
                    "Vector",
                    append(ValueInstruction{
                        .operation = ValueOperation::mapping,
                        .source_node = node.id,
                        .result_type = SocketType::vector,
                        .a = *vector,
                        .b = *location,
                        .c = *rotation,
                        .d = *scale,
                        .static_u0 = mode}));
            }
            return;
        }
        if (node.type == node_type::image_texture) {
            if (auto vector = lower_value_input(node, "Vector")) {
                const auto image = property_uint(node, "Image");
                const auto extension =
                    property_string(node, "Extension", "REPEAT");
                const auto color_space =
                    property_string(node, "ColorSpace", "sRGB");
                const auto address =
                    extension == "CLIP"
                        ? 1u
                        : extension == "EXTEND"
                              ? 2u
                              : extension == "MIRROR" ? 3u : 0u;
                const auto srgb =
                    color_space == "sRGB" ? 1u : 0u;
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::image_color,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *vector,
                        .static_u0 = image,
                        .static_u1 = address | (srgb << 8u)}));
                publish(
                    node.id,
                    "Alpha",
                    append(ValueInstruction{
                        .operation = ValueOperation::image_alpha,
                        .source_node = node.id,
                        .result_type = SocketType::floating,
                        .a = *vector,
                        .static_u0 = image,
                        .static_u1 = address}));
            }
            return;
        }
        if (node.type == node_type::add_float ||
            node.type == node_type::subtract_float ||
            node.type == node_type::multiply_float ||
            node.type == node_type::divide_float ||
            node.type == node_type::minimum_float ||
            node.type == node_type::maximum_float ||
            node.type == node_type::power_float) {
            auto operation =
                node.type == node_type::add_float
                    ? ValueOperation::add
                    : node.type == node_type::subtract_float
                          ? ValueOperation::subtract
                          : node.type == node_type::multiply_float
                                ? ValueOperation::multiply
                                : node.type == node_type::divide_float
                                      ? ValueOperation::divide
                                      : node.type ==
                                                node_type::minimum_float
                                            ? ValueOperation::minimum
                                            : node.type ==
                                                      node_type::maximum_float
                                                  ? ValueOperation::maximum
                                                  : ValueOperation::power;
            publish_binary_value(
                node,
                "A",
                "B",
                "Value",
                SocketType::floating,
                operation);
            return;
        }
        if (node.type == node_type::absolute_float ||
            node.type == node_type::clamp_float) {
            publish_unary_value(
                node,
                "Value",
                "Value",
                SocketType::floating,
                node.type == node_type::absolute_float
                    ? ValueOperation::absolute
                    : ValueOperation::clamp01);
            return;
        }
        if (node.type == node_type::scalar_to_color) {
            publish_unary_value(
                node,
                "Value",
                "Color",
                SocketType::color,
                ValueOperation::scalar_to_color);
            return;
        }
        if (node.type == node_type::color_to_scalar) {
            publish_unary_value(
                node,
                "Color",
                "Value",
                SocketType::floating,
                ValueOperation::color_to_scalar);
            return;
        }
        if (node.type == node_type::vector_to_scalar) {
            publish_unary_value(
                node,
                "Vector",
                "Value",
                SocketType::floating,
                ValueOperation::color_to_scalar);
            return;
        }
        if (node.type == node_type::vector_to_color ||
            node.type == node_type::color_to_vector ||
            node.type == node_type::vector_to_normal ||
            node.type == node_type::normal_to_vector) {
            const auto input =
                node.type == node_type::vector_to_color ||
                        node.type == node_type::vector_to_normal
                    ? "Vector"
                    : node.type == node_type::normal_to_vector
                          ? "Normal"
                          : "Color";
            const auto output =
                node.type == node_type::vector_to_color
                    ? "Color"
                    : node.type == node_type::vector_to_normal
                          ? "Normal"
                          : "Vector";
            const auto result_type =
                node.type == node_type::vector_to_color
                    ? SocketType::color
                    : node.type == node_type::vector_to_normal
                          ? SocketType::normal
                          : SocketType::vector;
            publish_unary_value(
                node,
                input,
                output,
                result_type,
                ValueOperation::passthrough);
            return;
        }
        if (node.type == node_type::mix_color) {
            auto factor = lower_value_input(node, "Factor");
            auto a = lower_value_input(node, "A");
            auto b = lower_value_input(node, "B");
            if (factor && a && b) {
                const auto mode = property_string(
                    node, "BlendMode", "MIX");
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::mix,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *a,
                        .b = *b,
                        .c = *factor,
                        .static_u0 =
                            mode == "VALUE"
                                ? 1u
                                : mode == "COLOR" ? 2u : 0u}));
            }
            return;
        }
        if (node.type == node_type::multiply_color) {
            auto a = lower_value_input(node, "A");
            auto b = lower_value_input(node, "B");
            auto factor = lower_value_input(node, "Factor");
            if (a && b && factor) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::multiply_color,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *a,
                        .b = *b,
                        .c = *factor}));
            }
            return;
        }
        if (node.type == node_type::hue_saturation) {
            auto hue = lower_value_input(node, "Hue");
            auto saturation =
                lower_value_input(node, "Saturation");
            auto value = lower_value_input(node, "Value");
            auto factor = lower_value_input(node, "Factor");
            auto color = lower_value_input(node, "Color");
            if (hue && saturation && value && factor && color) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::hue_saturation,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *color,
                        .b = *hue,
                        .c = *saturation,
                        .d = *value,
                        .e = *factor}));
            }
            return;
        }
        if (node.type == node_type::invert_color) {
            auto factor = lower_value_input(node, "Factor");
            auto color = lower_value_input(node, "Color");
            if (factor && color) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::invert,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *color,
                        .b = *factor}));
            }
            return;
        }
        if (node.type == node_type::normal_map) {
            auto strength = lower_value_input(node, "Strength");
            auto color = lower_value_input(node, "Color");
            if (strength && color) {
                publish(
                    node.id,
                    "Normal",
                    append(ValueInstruction{
                        .operation = ValueOperation::normal_map,
                        .source_node = node.id,
                        .result_type = SocketType::normal,
                        .a = *color,
                        .b = *strength}));
            }
            return;
        }
        if (node.type == node_type::bump) {
            auto height = lower_value_input(node, "Height");
            auto strength = lower_value_input(node, "Strength");
            auto distance = lower_value_input(node, "Distance");
            auto filter_width =
                lower_value_input(node, "FilterWidth");
            auto normal = lower_value_input(node, "Normal");
            if (height && strength && distance &&
                filter_width && normal) {
                publish(
                    node.id,
                    "Normal",
                    append(ValueInstruction{
                        .operation = ValueOperation::bump,
                        .source_node = node.id,
                        .result_type = SocketType::normal,
                        .a = *height,
                        .b = *strength,
                        .c = *distance,
                        .d = *filter_width,
                        .e = *normal,
                        .static_u0 =
                            property_bool(node, "Invert")
                                ? 1u
                                : 0u}));
            }
            return;
        }
        if (node.type == node_type::vertex_color) {
            auto instruction = ValueInstruction{
                .operation = ValueOperation::attribute_color,
                .source_node = node.id,
                .result_type = SocketType::color,
                .static_u0 =
                    property_uint(node, "AttributeId")};
            publish(
                node.id,
                "Color",
                append(instruction));
            instruction.operation =
                ValueOperation::attribute_alpha;
            instruction.result_type =
                SocketType::floating;
            publish(
                node.id,
                "Alpha",
                append(std::move(instruction)));
            return;
        }
        if (node.type == node_type::noise_texture) {
            auto vector = lower_value_input(node, "Vector");
            auto scale = lower_value_input(node, "Scale");
            auto detail = lower_value_input(node, "Detail");
            auto roughness =
                lower_value_input(node, "Roughness");
            if (vector && scale && detail && roughness) {
                auto instruction = ValueInstruction{
                    .operation = ValueOperation::noise_factor,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .a = *vector,
                    .b = *scale,
                    .c = *detail,
                    .d = *roughness,
                    .static_u0 =
                        property_uint(node, "Dimensions", 3u),
                    .static_u1 =
                        property_bool(node, "Normalize") ? 1u : 0u};
                publish(
                    node.id,
                    "Factor",
                    append(instruction));
                instruction.operation = ValueOperation::noise_color;
                instruction.result_type = SocketType::color;
                publish(
                    node.id,
                    "Color",
                    append(std::move(instruction)));
            }
            return;
        }
        if (node.type == node_type::brick_texture) {
            auto vector = lower_value_input(node, "Vector");
            auto color1 = lower_value_input(node, "Color1");
            auto color2 = lower_value_input(node, "Color2");
            auto mortar = lower_value_input(node, "Mortar");
            auto scale = lower_value_input(node, "Scale");
            auto mortar_size =
                lower_value_input(node, "MortarSize");
            auto mortar_smooth =
                lower_value_input(node, "MortarSmooth");
            auto bias = lower_value_input(node, "Bias");
            auto brick_width =
                lower_value_input(node, "BrickWidth");
            auto row_height =
                lower_value_input(node, "RowHeight");
            if (vector && color1 && color2 && mortar && scale &&
                mortar_size && mortar_smooth && bias &&
                brick_width && row_height) {
                auto instruction = ValueInstruction{
                    .operation = ValueOperation::brick_color,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .a = *vector,
                    .b = *color1,
                    .c = *color2,
                    .d = *mortar,
                    .e = *scale,
                    .f = *mortar_size,
                    .g = *mortar_smooth,
                    .h = *bias,
                    .i = *brick_width,
                    .j = *row_height,
                    .static_u0 = property_uint(
                        node, "OffsetFrequency", 2u),
                    .static_u1 = property_uint(
                        node, "SquashFrequency", 2u),
                    .static_f0 = property_float(
                        node, "OffsetAmount", 0.5f),
                    .static_f1 = property_float(
                        node, "SquashAmount", 1.0f)};
                publish(
                    node.id,
                    "Color",
                    append(instruction));
                instruction.operation =
                    ValueOperation::brick_factor;
                instruction.result_type =
                    SocketType::floating;
                publish(
                    node.id,
                    "Factor",
                    append(std::move(instruction)));
            }
            return;
        }
        if (node.type == node_type::gradient_texture) {
            if (auto vector = lower_value_input(node, "Vector")) {
                const auto kind = property_string(
                    node, "GradientType", "LINEAR");
                const auto mode =
                    kind == "QUADRATIC"
                        ? 1u
                        : kind == "EASING"
                              ? 2u
                              : kind == "DIAGONAL"
                                    ? 3u
                                    : kind == "RADIAL"
                                          ? 4u
                                          : kind == "SPHERICAL"
                                                ? 5u
                                                : kind == "QUADRATIC_SPHERE"
                                                      ? 6u
                                                      : 0u;
                publish(
                    node.id,
                    "Factor",
                    append(ValueInstruction{
                        .operation = ValueOperation::gradient,
                        .source_node = node.id,
                        .result_type = SocketType::floating,
                        .a = *vector,
                        .static_u0 = mode}));
            }
            return;
        }
        if (node.type == node_type::color_ramp) {
            if (auto factor = lower_value_input(node, "Factor")) {
                auto instruction = ValueInstruction{
                    .operation = ValueOperation::color_ramp,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .a = *factor,
                    .static_u0 =
                        property_string(
                            node, "Interpolation", "LINEAR") ==
                                "CONSTANT"
                            ? 1u
                            : 0u,
                    .static_table = parse_float_table(
                        property_string(node, "Table"))};
                publish(
                    node.id,
                    "Color",
                    append(instruction));
                instruction.result_type = SocketType::floating;
                instruction.static_u1 = 1u;
                publish(
                    node.id,
                    "Alpha",
                    append(std::move(instruction)));
            }
            return;
        }
        if (node.type == node_type::rgb_curve) {
            auto factor = lower_value_input(node, "Factor");
            auto color = lower_value_input(node, "Color");
            if (factor && color) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::rgb_curve,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *color,
                        .b = *factor,
                        .static_table = parse_float_table(
                            property_string(node, "Table"))}));
            }
            return;
        }
        if (node.type == node_type::separate_color) {
            if (auto color = lower_value_input(node, "Color")) {
                for (const auto [name, operation] :
                     {std::pair{"R", ValueOperation::separate_r},
                      std::pair{"G", ValueOperation::separate_g},
                      std::pair{"B", ValueOperation::separate_b}}) {
                    publish(
                        node.id,
                        name,
                        append(ValueInstruction{
                            .operation = operation,
                            .source_node = node.id,
                            .result_type = SocketType::floating,
                            .a = *color}));
                }
            }
            return;
        }
        if (node.type == node_type::combine_color) {
            auto r = lower_value_input(node, "R");
            auto g = lower_value_input(node, "G");
            auto b = lower_value_input(node, "B");
            if (r && g && b) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::combine_color,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *r,
                        .b = *g,
                        .c = *b}));
            }
            return;
        }
        if (node.type == node_type::nishita_sky) {
            auto elevation =
                lower_value_input(node, "SunElevation");
            auto rotation = lower_value_input(node, "SunRotation");
            auto size = lower_value_input(node, "SunSize");
            auto intensity =
                lower_value_input(node, "SunIntensity");
            auto altitude = lower_value_input(node, "Altitude");
            auto air = lower_value_input(node, "AirDensity");
            auto dust = lower_value_input(node, "DustDensity");
            auto ozone = lower_value_input(node, "OzoneDensity");
            if (elevation && rotation && size && intensity &&
                altitude && air && dust && ozone) {
                publish(
                    node.id,
                    "Color",
                    append(ValueInstruction{
                        .operation = ValueOperation::nishita_sky,
                        .source_node = node.id,
                        .result_type = SocketType::color,
                        .a = *elevation,
                        .b = *rotation,
                        .c = *size,
                        .d = *intensity,
                        .e = *altitude,
                        .f = *air,
                        .g = *dust,
                        .h = *ozone}));
            }
            return;
        }
        if (node.type == node_type::diffuse_bsdf ||
            node.type == node_type::principled_bsdf ||
            node.type == node_type::glossy_bsdf) {
            const auto color_name =
                node.type == node_type::principled_bsdf
                    ? "BaseColor"
                    : "Color";
            auto color = lower_value_input(node, color_name);
            auto roughness =
                lower_value_input(node, "Roughness");
            auto normal = lower_value_input(node, "Normal");
            std::optional<ValueExpressionId> metallic;
            std::optional<ValueExpressionId> ior;
            if (node.type == node_type::principled_bsdf) {
                metallic = lower_value_input(node, "Metallic");
                ior = lower_value_input(node, "IOR");
            }
            if (color && roughness && normal &&
                (node.type != node_type::principled_bsdf ||
                 (metallic && ior))) {
                publish(
                    node.id,
                    "Closure",
                    append(ClosureInstruction{
                        .operation =
                            node.type == node_type::principled_bsdf
                                ? ClosureOperation::principled
                                : node.type ==
                                          node_type::glossy_bsdf
                                      ? ClosureOperation::glossy
                                      : ClosureOperation::diffuse,
                        .source_node = node.id,
                        .color = *color,
                        .normal = *normal,
                        .roughness = *roughness,
                        .metallic =
                            metallic.value_or(ValueExpressionId{}),
                        .ior = ior.value_or(ValueExpressionId{})}));
            }
            return;
        }
        if (node.type == node_type::emission) {
            auto color = lower_value_input(node, "Color");
            auto strength = lower_value_input(node, "Strength");
            if (color && strength) {
                publish(
                    node.id,
                    "Closure",
                    append(ClosureInstruction{
                        .operation = ClosureOperation::emission,
                        .source_node = node.id,
                        .color = *color,
                        .strength = *strength}));
            }
            return;
        }
        if (node.type == node_type::transparent_bsdf) {
            if (auto color = lower_value_input(node, "Color")) {
                publish(
                    node.id,
                    "Closure",
                    append(ClosureInstruction{
                        .operation = ClosureOperation::transparent,
                        .source_node = node.id,
                        .color = *color}));
            }
            return;
        }
        if (node.type == node_type::add_closure ||
            node.type == node_type::mix_closure) {
            auto a = lower_closure_input(node, "A");
            auto b = lower_closure_input(node, "B");
            std::optional<ValueExpressionId> factor;
            if (node.type == node_type::mix_closure) {
                factor = lower_value_input(node, "Factor");
            }
            if (a && b &&
                (node.type == node_type::add_closure || factor)) {
                publish(
                    node.id,
                    "Closure",
                    append(ClosureInstruction{
                        .operation =
                            node.type == node_type::add_closure
                                ? ClosureOperation::add
                                : ClosureOperation::mix,
                        .source_node = node.id,
                        .factor =
                            factor.value_or(ValueExpressionId{}),
                        .a = *a,
                        .b = *b}));
            }
            return;
        }

        diagnose(
            SurfaceProgramDiagnosticCode::unsupported_node,
            node_prefix(node.id) +
                "surface lowering is not implemented for node type '" +
                node.type + "'",
            node.id);
    }

public:
    explicit SurfaceProgramBuilder(
        const ShaderProgram &shader) noexcept
        : _shader{shader} {}

    [[nodiscard]] SurfaceProgramCompilation build() {
        for (auto node_id :
             _shader.analysis().evaluation_order) {
            const auto *node = _shader.graph().find(node_id);
            if (node == nullptr) {
                diagnose(
                    SurfaceProgramDiagnosticCode::missing_output,
                    "validated shader program references a missing node",
                    node_id);
                continue;
            }
            lower_node(*node);
        }

        const auto &root =
            _shader.graph().root(contract::ShaderDomain::surface);
        if (!root) {
            diagnose(
                SurfaceProgramDiagnosticCode::missing_surface_root,
                "shader has no surface root");
        }

        ClosureExpressionId lowered_root;
        if (root) {
            auto iter = _outputs.find(
                {.node = root->node, .socket = root->socket});
            if (iter == _outputs.end()) {
                diagnose(
                    SurfaceProgramDiagnosticCode::missing_output,
                    "surface root was not lowered",
                    root->node,
                    root->socket);
            } else if (
                const auto *closure =
                    std::get_if<ClosureExpressionId>(
                        &iter->second)) {
                lowered_root = *closure;
            } else {
                diagnose(
                    SurfaceProgramDiagnosticCode::type_mismatch,
                    "surface root did not lower to a closure",
                    root->node,
                    root->socket);
            }
        }

        if (!_diagnostics.empty()) {
            return {
                .program = nullptr,
                .diagnostics = std::move(_diagnostics)};
        }

        return {
            .program = std::make_shared<const SurfaceProgram>(
                _shader.analysis().structure_signature,
                std::move(_parameters),
                std::move(_value_instructions),
                std::move(_closure_instructions),
                lowered_root),
            .diagnostics = {}};
    }
};

}// namespace

SurfaceProgram::SurfaceProgram(
    std::uint64_t structure_signature,
    std::vector<ParameterDesc> parameters,
    std::vector<ValueInstruction> value_instructions,
    std::vector<ClosureInstruction> closure_instructions,
    ClosureExpressionId root) noexcept
    : _structure_signature{structure_signature},
      _parameters{std::move(parameters)},
      _value_instructions{std::move(value_instructions)},
      _closure_instructions{std::move(closure_instructions)},
      _root{root} {}

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
    return SurfaceProgramBuilder{shader}.build();
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
                        node_prefix(parameter.node) +
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
                        node_prefix(parameter.node) +
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
