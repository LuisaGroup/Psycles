#include "surface_program_builder.h"

#include <string_view>
#include <utility>

namespace psycles::compiler::detail {
namespace {

namespace operand = value_operand;

[[nodiscard]] std::uint64_t mapping_axis(
    std::string_view axis) noexcept {
    return axis == "X"
               ? 1u
               : axis == "Y" ? 2u : axis == "Z" ? 3u : 0u;
}

[[nodiscard]] std::uint64_t mapping_axes(
    const contract::ShaderNode &node) {
    const auto x = mapping_axis(
        property_string(node, "XMapping", "X"));
    const auto y = mapping_axis(
        property_string(node, "YMapping", "Y"));
    const auto z = mapping_axis(
        property_string(node, "ZMapping", "Z"));
    return x == 1u && y == 2u && z == 3u
               ? 0u
               : x | (y << 2u) | (z << 4u);
}

}// namespace

// Lowers typed math, conversion, color, and normal nodes. A true result means
// the node family was recognized, even when input diagnostics prevented an
// instruction from being emitted.
[[nodiscard]] bool SurfaceProgramBuilder::lower_value_node(
    const contract::ShaderNode &node) {
    using contract::SocketType;

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
                    .operands = make_value_operands<operand::mapping>({
                        {operand::mapping::vector, *vector},
                        {operand::mapping::location, *location},
                        {operand::mapping::rotation, *rotation},
                        {operand::mapping::scale, *scale}}),
                    .static_u0 = mode,
                    .static_u1 = mapping_axes(node)}));
        }
        return true;
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
        return true;
    }
    if (node.type == node_type::math) {
        auto a = lower_value_input(node, "A");
        auto b = lower_value_input(node, "B");
        auto c = lower_value_input(node, "C");
        if (a && b && c) {
            publish(
                node.id,
                "Value",
                append(ValueInstruction{
                    .operation = ValueOperation::math,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands = make_value_operands<operand::ternary>({
                        {operand::ternary::a, *a},
                        {operand::ternary::b, *b},
                        {operand::ternary::c, *c}}),
                    .static_u0 = static_cast<std::uint64_t>(
                        math_operation(node))}));
        }
        return true;
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
        return true;
    }
    if (node.type == node_type::clamp_range) {
        auto value = lower_value_input(node, "Value");
        auto minimum = lower_value_input(node, "Min");
        auto maximum = lower_value_input(node, "Max");
        if (value && minimum && maximum) {
            publish(
                node.id,
                "Result",
                append(ValueInstruction{
                    .operation = ValueOperation::clamp_range,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands = make_value_operands<operand::clamp_range>({
                        {operand::clamp_range::value, *value},
                        {operand::clamp_range::minimum, *minimum},
                        {operand::clamp_range::maximum, *maximum}}),
                    .static_u0 =
                        property_string(
                            node, "Mode", "MINMAX") == "RANGE"
                            ? 1u
                            : 0u}));
        }
        return true;
    }
    if (node.type == node_type::map_range) {
        const auto vector_mode =
            property_string(node, "DataType", "FLOAT") ==
            "FLOAT_VECTOR";
        const auto interpolation =
            map_range_interpolation(node);
        const auto clamp =
            property_bool(node, "Clamp", true) ? 1u : 0u;
        if (vector_mode) {
            auto value = lower_value_input(node, "Vector");
            auto from_min =
                lower_value_input(node, "FromMinVector");
            auto from_max =
                lower_value_input(node, "FromMaxVector");
            auto to_min =
                lower_value_input(node, "ToMinVector");
            auto to_max =
                lower_value_input(node, "ToMaxVector");
            auto steps =
                lower_value_input(node, "StepsVector");
            if (value && from_min && from_max && to_min &&
                to_max && steps) {
                publish(
                    node.id,
                    "Vector",
                    append(ValueInstruction{
                        .operation =
                            ValueOperation::map_range_vector,
                        .source_node = node.id,
                        .result_type = SocketType::vector,
                        .operands = make_value_operands<operand::map_range>({
                            {operand::map_range::value, *value},
                            {operand::map_range::from_min, *from_min},
                            {operand::map_range::from_max, *from_max},
                            {operand::map_range::to_min, *to_min},
                            {operand::map_range::to_max, *to_max},
                            {operand::map_range::steps, *steps}}),
                        .static_u0 = interpolation,
                        .static_u1 = clamp}));
            }
        } else {
            auto value = lower_value_input(node, "Value");
            auto from_min =
                lower_value_input(node, "FromMin");
            auto from_max =
                lower_value_input(node, "FromMax");
            auto to_min = lower_value_input(node, "ToMin");
            auto to_max = lower_value_input(node, "ToMax");
            auto steps = lower_value_input(node, "Steps");
            if (value && from_min && from_max && to_min &&
                to_max && steps) {
                publish(
                    node.id,
                    "Result",
                    append(ValueInstruction{
                        .operation =
                            ValueOperation::map_range_float,
                        .source_node = node.id,
                        .result_type = SocketType::floating,
                        .operands = make_value_operands<operand::map_range>({
                            {operand::map_range::value, *value},
                            {operand::map_range::from_min, *from_min},
                            {operand::map_range::from_max, *from_max},
                            {operand::map_range::to_min, *to_min},
                            {operand::map_range::to_max, *to_max},
                            {operand::map_range::steps, *steps}}),
                        .static_u0 = interpolation,
                        .static_u1 = clamp}));
            }
        }
        return true;
    }
    if (node.type == node_type::vector_math) {
        auto a = lower_value_input(node, "A");
        auto b = lower_value_input(node, "B");
        auto c = lower_value_input(node, "C");
        auto scale = lower_value_input(node, "Scale");
        if (a && b && c && scale) {
            const auto operation =
                static_cast<std::uint64_t>(
                    vector_math_operation(node));
            publish(
                node.id,
                "Vector",
                append(ValueInstruction{
                    .operation =
                        ValueOperation::vector_math_vector,
                    .source_node = node.id,
                    .result_type = SocketType::vector,
                    .operands = make_value_operands<operand::vector_math>({
                        {operand::vector_math::a, *a},
                        {operand::vector_math::b, *b},
                        {operand::vector_math::c, *c},
                        {operand::vector_math::scale, *scale}}),
                    .static_u0 = operation}));
            publish(
                node.id,
                "Value",
                append(ValueInstruction{
                    .operation =
                        ValueOperation::vector_math_value,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands = make_value_operands<operand::vector_math>({
                        {operand::vector_math::a, *a},
                        {operand::vector_math::b, *b},
                        {operand::vector_math::c, *c},
                        {operand::vector_math::scale, *scale}}),
                    .static_u0 = operation}));
        }
        return true;
    }
    if (node.type == node_type::scalar_to_color) {
        publish_unary_value(
            node,
            "Value",
            "Color",
            SocketType::color,
            ValueOperation::scalar_to_color);
        return true;
    }
    if (node.type == node_type::scalar_to_boolean) {
        publish_unary_value(
            node,
            "Value",
            "Boolean",
            SocketType::boolean,
            ValueOperation::scalar_to_boolean);
        return true;
    }
    if (node.type == node_type::color_to_scalar) {
        publish_unary_value(
            node,
            "Color",
            "Value",
            SocketType::floating,
            ValueOperation::color_to_scalar);
        return true;
    }
    if (node.type == node_type::vector_to_scalar) {
        publish_unary_value(
            node,
            "Vector",
            "Value",
            SocketType::floating,
            ValueOperation::vector_to_scalar);
        return true;
    }
    if (node.type == node_type::vector_to_color ||
        node.type == node_type::color_to_vector ||
        node.type == node_type::point_to_vector ||
        node.type == node_type::float3_to_vector ||
        node.type == node_type::vector_to_normal ||
        node.type == node_type::normal_to_vector) {
        const auto input =
            node.type == node_type::vector_to_color ||
                    node.type == node_type::vector_to_normal
                ? "Vector"
                : node.type == node_type::normal_to_vector
                      ? "Normal"
                      : node.type == node_type::point_to_vector
                            ? "Point"
                            : node.type == node_type::float3_to_vector
                                  ? "Value"
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
        return true;
    }
    if (node.type == node_type::mix_float) {
        auto factor = lower_value_input(node, "Factor");
        auto a = lower_value_input(node, "A");
        auto b = lower_value_input(node, "B");
        if (factor && a && b) {
            publish(
                node.id,
                "Value",
                append(ValueInstruction{
                    .operation = ValueOperation::mix_float,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands = make_value_operands<operand::mix>({
                        {operand::mix::a, *a},
                        {operand::mix::b, *b},
                        {operand::mix::factor, *factor}}),
                    .static_u0 = property_bool(
                                     node,
                                     "ClampFactor",
                                     true)
                                     ? 1u
                                     : 0u}));
        }
        return true;
    }
    if (node.type == node_type::mix_vector ||
        node.type == node_type::mix_vector_nonuniform) {
        auto factor = lower_value_input(node, "Factor");
        auto a = lower_value_input(node, "A");
        auto b = lower_value_input(node, "B");
        if (factor && a && b) {
            publish(
                node.id,
                "Vector",
                append(ValueInstruction{
                    .operation = ValueOperation::mix_vector,
                    .source_node = node.id,
                    .result_type = SocketType::vector,
                    .operands = make_value_operands<operand::mix>({
                        {operand::mix::a, *a},
                        {operand::mix::b, *b},
                        {operand::mix::factor, *factor}}),
                    .static_u0 =
                        node.type ==
                                node_type::mix_vector_nonuniform
                            ? 1u
                            : 0u,
                    .static_u1 = property_bool(
                                     node,
                                     "ClampFactor",
                                     true)
                                     ? 1u
                                     : 0u}));
        }
        return true;
    }
    if (node.type == node_type::mix_color) {
        auto factor = lower_value_input(node, "Factor");
        auto a = lower_value_input(node, "A");
        auto b = lower_value_input(node, "B");
        if (factor && a && b) {
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::mix,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .operands = make_value_operands<operand::mix>({
                        {operand::mix::a, *a},
                        {operand::mix::b, *b},
                        {operand::mix::factor, *factor}}),
                    .static_u0 = static_cast<std::uint64_t>(
                        blend_operation(node)),
                    .static_u1 =
                        (property_bool(
                             node,
                             "ClampFactor",
                             true)
                             ? 1u
                             : 0u) |
                        (property_bool(
                             node,
                             "ClampResult",
                             false)
                             ? 2u
                             : 0u)}));
        }
        return true;
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
                    .operands = make_value_operands<operand::mix>({
                        {operand::mix::a, *a},
                        {operand::mix::b, *b},
                        {operand::mix::factor, *factor}})}));
        }
        return true;
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
                    .operands =
                        make_value_operands<operand::hue_saturation>({
                            {operand::hue_saturation::color, *color},
                            {operand::hue_saturation::hue, *hue},
                            {operand::hue_saturation::saturation, *saturation},
                            {operand::hue_saturation::value, *value},
                            {operand::hue_saturation::factor, *factor}})}));
        }
        return true;
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
                    .operands = make_value_operands<operand::color_factor>({
                        {operand::color_factor::color, *color},
                        {operand::color_factor::factor, *factor}})}));
        }
        return true;
    }
    if (node.type == node_type::gamma_color) {
        auto color = lower_value_input(node, "Color");
        auto gamma = lower_value_input(node, "Gamma");
        if (color && gamma) {
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::gamma,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .operands = make_value_operands<operand::gamma>({
                        {operand::gamma::color, *color},
                        {operand::gamma::exponent, *gamma}})}));
        }
        return true;
    }
    if (node.type == node_type::brightness_contrast) {
        auto color = lower_value_input(node, "Color");
        auto bright = lower_value_input(node, "Bright");
        auto contrast = lower_value_input(node, "Contrast");
        if (color && bright && contrast) {
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation =
                        ValueOperation::brightness_contrast,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .operands =
                        make_value_operands<operand::brightness_contrast>({
                            {operand::brightness_contrast::color, *color},
                            {operand::brightness_contrast::brightness,
                             *bright},
                            {operand::brightness_contrast::contrast,
                             *contrast}})}));
        }
        return true;
    }
    if (node.type == node_type::blackbody) {
        publish_unary_value(
            node,
            "Temperature",
            "Color",
            SocketType::color,
            ValueOperation::blackbody);
        return true;
    }
    if (node.type == node_type::wavelength) {
        publish_unary_value(
            node,
            "Wavelength",
            "Color",
            SocketType::color,
            ValueOperation::wavelength);
        return true;
    }
    if (node.type == node_type::normal_map) {
        auto strength = lower_value_input(node, "Strength");
        auto color = lower_value_input(node, "Color");
        auto uv_map = lower_property_parameter(node, "UvMapId");
        if (strength && color && uv_map) {
            const auto space =
                property_string(node, "Space", "TANGENT");
            const auto normal_map_space =
                space == "OBJECT"
                    ? NormalMapSpace::object
                    : space == "WORLD"
                          ? NormalMapSpace::world
                          : space == "BLENDER_OBJECT"
                                ? NormalMapSpace::
                                      blender_object
                                : space == "BLENDER_WORLD"
                                      ? NormalMapSpace::
                                            blender_world
                                      : NormalMapSpace::
                                            tangent;
            const auto normal_map_base =
                property_string(
                    node, "Base", "ORIGINAL") == "DISPLACED"
                    ? NormalMapBase::displaced
                    : NormalMapBase::original;
            const auto normal_map_convention =
                property_string(
                    node, "Convention", "OPENGL") == "DIRECTX"
                    ? NormalMapConvention::direct_x
                    : NormalMapConvention::open_gl;
            publish(
                node.id,
                "Normal",
                append(ValueInstruction{
                    .operation = ValueOperation::normal_map,
                    .source_node = node.id,
                    .result_type = SocketType::normal,
                    .operands = make_value_operands<operand::normal_map>({
                        {operand::normal_map::color, *color},
                        {operand::normal_map::strength, *strength},
                        {operand::normal_map::uv_map, *uv_map}}),
                    .static_u0 = encode_normal_map_configuration(
                        normal_map_space,
                        property_bool(node, "UvMapNamed"),
                        normal_map_base,
                        normal_map_convention)}));
        }
        return true;
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
                    .operands = make_value_operands<operand::bump>({
                        {operand::bump::height, *height},
                        {operand::bump::strength, *strength},
                        {operand::bump::distance, *distance},
                        {operand::bump::filter_width, *filter_width},
                        {operand::bump::normal, *normal}}),
                    .static_u0 =
                        (property_bool(node, "Invert")
                             ? 1u
                             : 0u) |
                        (property_bool(
                             node, "NormalLinked")
                             ? 2u
                             : 0u) |
                        (property_bool(
                             node, "UseObjectSpace")
                             ? 4u
                             : 0u)}));
        }
        return true;
    }
    return false;
}

}// namespace psycles::compiler::detail
