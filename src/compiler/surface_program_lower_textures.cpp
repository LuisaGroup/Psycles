#include "surface_program_builder.h"
#include "hosek_sky.h"

#include <utility>

namespace psycles::compiler::detail {

// Lowers image, attribute, procedural, ramp, and sky nodes. A true result
// means the node family was recognized, even when input diagnostics
// prevented an instruction from being emitted.
[[nodiscard]] bool SurfaceProgramBuilder::lower_texture_node(
    const contract::ShaderNode &node) {
    using contract::SocketType;

    if (node.type == node_type::image_texture) {
        if (auto vector = lower_value_input(node, "Vector")) {
            const auto image = property_uint(node, "Image");
            const auto extension =
                property_string(node, "Extension", "REPEAT");
            const auto color_space =
                property_string(node, "ColorSpace", "sRGB");
            const auto interpolation_name =
                property_string(
                    node, "Interpolation", "Linear");
            const auto projection_name =
                property_string(node, "Projection", "FLAT");
            const auto address =
                extension == "CLIP"
                    ? 1u
                    : extension == "EXTEND"
                          ? 2u
                          : extension == "MIRROR" ? 3u : 0u;
            const auto srgb =
                color_space == "sRGB" ? 1u : 0u;
            const auto unassociate_alpha =
                property_bool(node, "UnassociateAlpha") ? 1u : 0u;
            const auto interpolation =
                interpolation_name == "Closest"
                    ? 0u
                    : interpolation_name == "Linear"
                          ? 1u
                          : interpolation_name == "Cubic"
                                ? 2u
                                : 3u;
            const auto projection =
                projection_name == "BOX"
                    ? 1u
                    : projection_name == "SPHERE"
                          ? 2u
                          : projection_name == "TUBE"
                                ? 3u
                                : 0u;
            const auto flags =
                address |
                (srgb << 8u) |
                (unassociate_alpha << 9u) |
                (interpolation << 10u) |
                (projection << 12u);
            const auto projection_blend =
                property_float(
                    node, "ProjectionBlend", 0.0f);
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::image_color,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .a = *vector,
                    .static_u0 = image,
                    .static_u1 = flags,
                    .static_f0 = projection_blend}));
            publish(
                node.id,
                "Alpha",
                append(ValueInstruction{
                    .operation = ValueOperation::image_alpha,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .a = *vector,
                    .static_u0 = image,
                    .static_u1 = flags,
                    .static_f0 = projection_blend}));
        }
        return true;
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
        return true;
    }
    if (node.type == node_type::noise_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto w = lower_value_input(node, "W");
        auto scale = lower_value_input(node, "Scale");
        auto detail = lower_value_input(node, "Detail");
        auto roughness =
            lower_value_input(node, "Roughness");
        auto lacunarity =
            lower_value_input(node, "Lacunarity");
        auto offset = lower_value_input(node, "Offset");
        auto gain = lower_value_input(node, "Gain");
        auto distortion =
            lower_value_input(node, "Distortion");
        if (vector && w && scale && detail && roughness &&
            lacunarity && offset && gain && distortion) {
            const auto type_name =
                property_string(node, "NoiseType", "FBM");
            const auto noise_type =
                type_name == "MULTIFRACTAL"
                    ? NoiseType::multifractal
                    : type_name == "HYBRID_MULTIFRACTAL"
                          ? NoiseType::hybrid_multifractal
                          : type_name == "RIDGED_MULTIFRACTAL"
                                ? NoiseType::
                                      ridged_multifractal
                                : type_name == "HETERO_TERRAIN"
                                      ? NoiseType::
                                            hetero_terrain
                                      : NoiseType::fbm;
            const auto needs_color =
                property_bool(node, "NeedsColor");
            auto instruction = ValueInstruction{
                .operation =
                    needs_color
                        ? ValueOperation::noise_color
                        : ValueOperation::noise_factor,
                .source_node = node.id,
                .result_type =
                    needs_color
                        ? SocketType::color
                        : SocketType::floating,
                .a = *vector,
                .b = *scale,
                .c = *detail,
                .d = *roughness,
                .e = *lacunarity,
                .f = *distortion,
                .g = *w,
                .h = *offset,
                .i = *gain,
                .static_u0 =
                    property_uint(node, "Dimensions", 3u),
                .static_u1 =
                    (property_bool(node, "Normalize")
                         ? 1u
                         : 0u) |
                    (static_cast<std::uint64_t>(noise_type)
                     << 8u)};
            publish(
                node.id,
                needs_color ? "Color" : "Factor",
                append(std::move(instruction)));
        }
        return true;
    }
    if (node.type == node_type::white_noise_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto w = lower_value_input(node, "W");
        if (vector && w) {
            const auto needs_color =
                property_bool(node, "NeedsColor");
            auto instruction = ValueInstruction{
                .operation =
                    needs_color
                        ? ValueOperation::white_noise_color
                        : ValueOperation::white_noise_value,
                .source_node = node.id,
                .result_type =
                    needs_color
                        ? SocketType::color
                        : SocketType::floating,
                .a = *vector,
                .b = *w,
                .static_u0 =
                    property_uint(node, "Dimensions", 3u)};
            publish(
                node.id,
                needs_color ? "Color" : "Value",
                append(std::move(instruction)));
        }
        return true;
    }
    if (node.type == node_type::checker_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto color1 = lower_value_input(node, "Color1");
        auto color2 = lower_value_input(node, "Color2");
        auto scale = lower_value_input(node, "Scale");
        if (vector && color1 && color2 && scale) {
            auto instruction = ValueInstruction{
                .operation = ValueOperation::checker_color,
                .source_node = node.id,
                .result_type = SocketType::color,
                .a = *vector,
                .b = *color1,
                .c = *color2,
                .d = *scale};
            publish(
                node.id,
                "Color",
                append(instruction));
            instruction.operation =
                ValueOperation::checker_factor;
            instruction.result_type =
                SocketType::floating;
            publish(
                node.id,
                "Factor",
                append(std::move(instruction)));
        }
        return true;
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
        return true;
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
        return true;
    }
    if (node.type == node_type::color_ramp) {
        if (auto factor = lower_value_input(node, "Factor")) {
            auto instruction = ValueInstruction{
                .operation = ValueOperation::color_ramp,
                .source_node = node.id,
                .result_type = SocketType::color,
                .a = *factor,
                .static_u0 =
                    (property_string(
                         node, "Interpolation", "LINEAR") ==
                             "CONSTANT"
                         ? 1u
                         : 0u) |
                    (property_bool(node, "Sampled")
                         ? 2u
                         : 0u),
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
        return true;
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
                    .static_u0 =
                        (property_bool(node, "Sampled")
                             ? 1u
                             : 0u) |
                        (property_bool(
                             node, "Extrapolate", true)
                             ? 2u
                             : 0u),
                    .static_f0 =
                        property_float(node, "MinX", 0.0f),
                    .static_f1 =
                        property_float(node, "MaxX", 1.0f),
                    .static_table = parse_float_table(
                        property_string(node, "Table"))}));
        }
        return true;
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
                        .a = *color,
                        .static_u0 = color_mode(node)}));
            }
        }
        return true;
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
                    .c = *b,
                    .static_u0 = color_mode(node)}));
        }
        return true;
    }
    if (node.type == node_type::hosek_wilkie_sky) {
        if (auto direction = lower_value_input(node, "Vector")) {
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation =
                        ValueOperation::hosek_wilkie_sky,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .a = *direction,
                    .static_table = cook_hosek_wilkie_sky(
                        property_float(
                            node, "SunDirectionX", 0.0f),
                        property_float(
                            node, "SunDirectionY", 0.0f),
                        property_float(
                            node, "SunDirectionZ", 1.0f),
                        property_float(node, "Turbidity", 2.2f),
                        property_float(
                            node, "GroundAlbedo", 0.3f))}));
        }
        return true;
    }
    if (node.type == node_type::nishita_sky) {
        auto direction = lower_value_input(node, "Vector");
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
        if (direction && elevation && rotation && size &&
            intensity && altitude && air && dust && ozone) {
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
                    .h = *ozone,
                    .i = *direction,
                    .static_u0 = _nishita_count++}));
        }
        return true;
    }
    return false;
}

}// namespace psycles::compiler::detail
