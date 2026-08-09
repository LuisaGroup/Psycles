#include "surface_program_builder.h"
#include "hosek_sky.h"

#include <cstdint>
#include <string_view>
#include <utility>

namespace psycles::compiler::detail {
namespace {

namespace operand = value_operand;

[[nodiscard]] std::uint32_t texture_interpolation(
    std::string_view name) noexcept {
    return name == "Closest"
               ? 0u
               : name == "Linear"
                     ? 1u
                     : name == "Cubic" ? 2u : 3u;
}

[[nodiscard]] std::uint32_t texture_color_flags(
    std::string_view color_space,
    std::string_view interpolation,
    bool unassociate_alpha = false) noexcept {
    const auto srgb = color_space == "sRGB" ? 1u : 0u;
    return (srgb << 8u) |
           (static_cast<std::uint32_t>(unassociate_alpha) << 9u) |
           (texture_interpolation(interpolation) << 10u);
}

}// namespace

// Lowers image, attribute, procedural, ramp, and sky nodes. A true result
// means the node family was recognized, even when input diagnostics
// prevented an instruction from being emitted.
[[nodiscard]] bool SurfaceProgramBuilder::lower_texture_node(
    const contract::ShaderNode &node) {
    using contract::SocketType;

    if (node.type == node_type::image_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto image = lower_property_parameter(node, "Image");
        auto projection_blend =
            lower_property_parameter(node, "ProjectionBlend");
        if (vector && image && projection_blend) {
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
            const auto unassociate_alpha =
                property_bool(node, "UnassociateAlpha");
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
                texture_color_flags(
                    color_space,
                    interpolation_name,
                    unassociate_alpha) |
                (projection << 12u);
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::image_color,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .operands = make_value_operands<operand::image_texture>({
                        {operand::image_texture::vector, *vector},
                        {operand::image_texture::image, *image},
                        {operand::image_texture::projection_blend,
                         *projection_blend}}),
                    .static_u1 = flags}));
            publish(
                node.id,
                "Alpha",
                append(ValueInstruction{
                    .operation = ValueOperation::image_alpha,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands = make_value_operands<operand::image_texture>({
                        {operand::image_texture::vector, *vector},
                        {operand::image_texture::image, *image},
                        {operand::image_texture::projection_blend,
                         *projection_blend}}),
                    .static_u1 = flags}));
        }
        return true;
    }
    if (node.type == node_type::environment_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto image = lower_property_parameter(node, "Image");
        if (vector && image) {
            const auto projection =
                property_string(
                    node, "Projection", "EQUIRECTANGULAR") ==
                        "MIRROR_BALL"
                    ? 1u
                    : 0u;
            const auto flags =
                texture_color_flags(
                    property_string(node, "ColorSpace", "Non-Color"),
                    property_string(node, "Interpolation", "Linear")) |
                (projection << 12u);
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::environment_color,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .operands =
                        make_value_operands<operand::environment_texture>({
                            {operand::environment_texture::vector, *vector},
                            {operand::environment_texture::image, *image}}),
                    .static_u1 = flags}));
            publish(
                node.id,
                "Alpha",
                append(ValueInstruction{
                    .operation = ValueOperation::environment_alpha,
                    .source_node = node.id,
                    .result_type = SocketType::floating,
                    .operands =
                        make_value_operands<operand::environment_texture>({
                            {operand::environment_texture::vector, *vector},
                            {operand::environment_texture::image, *image}}),
                    .static_u1 = flags}));
        }
        return true;
    }
    if (node.type == node_type::vertex_color) {
        auto attribute = lower_property_parameter(node, "AttributeId");
        if (!attribute) {
            return true;
        }
        auto instruction = ValueInstruction{
            .operation = ValueOperation::attribute_color,
            .source_node = node.id,
            .result_type = SocketType::color,
            .operands = make_value_operands<operand::attribute>({
                {operand::attribute::id, *attribute}})};
        publish(
            node.id,
            "Color",
            append(instruction));
        instruction.result_type = SocketType::vector;
        publish(
            node.id,
            "Vector",
            append(instruction));
        instruction.operation =
            ValueOperation::attribute_factor;
        instruction.result_type = SocketType::floating;
        publish(
            node.id,
            "Factor",
            append(instruction));
        instruction.operation =
            ValueOperation::attribute_alpha;
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
                .operands = make_value_operands<operand::noise>({
                    {operand::noise::vector, *vector},
                    {operand::noise::scale, *scale},
                    {operand::noise::detail, *detail},
                    {operand::noise::roughness, *roughness},
                    {operand::noise::lacunarity, *lacunarity},
                    {operand::noise::distortion, *distortion},
                    {operand::noise::w, *w},
                    {operand::noise::offset, *offset},
                    {operand::noise::gain, *gain}}),
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
                .operands = make_value_operands<operand::white_noise>({
                    {operand::white_noise::vector, *vector},
                    {operand::white_noise::w, *w}}),
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
                .operands = make_value_operands<operand::checker>({
                    {operand::checker::vector, *vector},
                    {operand::checker::color1, *color1},
                    {operand::checker::color2, *color2},
                    {operand::checker::scale, *scale}})};
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
        auto offset_amount =
            lower_property_parameter(node, "OffsetAmount");
        auto offset_frequency =
            lower_property_parameter(node, "OffsetFrequency");
        auto squash_amount =
            lower_property_parameter(node, "SquashAmount");
        auto squash_frequency =
            lower_property_parameter(node, "SquashFrequency");
        if (vector && color1 && color2 && mortar && scale &&
            mortar_size && mortar_smooth && bias &&
            brick_width && row_height && offset_amount &&
            offset_frequency && squash_amount && squash_frequency) {
            auto instruction = ValueInstruction{
                .operation = ValueOperation::brick_color,
                .source_node = node.id,
                .result_type = SocketType::color,
                .operands = make_value_operands<operand::brick>({
                    {operand::brick::vector, *vector},
                    {operand::brick::color1, *color1},
                    {operand::brick::color2, *color2},
                    {operand::brick::mortar, *mortar},
                    {operand::brick::scale, *scale},
                    {operand::brick::mortar_size, *mortar_size},
                    {operand::brick::mortar_smooth, *mortar_smooth},
                    {operand::brick::bias, *bias},
                    {operand::brick::brick_width, *brick_width},
                    {operand::brick::row_height, *row_height},
                    {operand::brick::offset_amount, *offset_amount},
                    {operand::brick::offset_frequency, *offset_frequency},
                    {operand::brick::squash_amount, *squash_amount},
                    {operand::brick::squash_frequency, *squash_frequency}})};
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
    if (node.type == node_type::magic_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto scale = lower_value_input(node, "Scale");
        auto distortion = lower_value_input(node, "Distortion");
        auto depth = lower_property_parameter(node, "Depth");
        if (vector && scale && distortion && depth) {
            const auto needs_color =
                property_bool(node, "NeedsColor");
            publish(
                node.id,
                needs_color ? "Color" : "Factor",
                append(ValueInstruction{
                    .operation = needs_color
                                     ? ValueOperation::magic_color
                                     : ValueOperation::magic_factor,
                    .source_node = node.id,
                    .result_type = needs_color
                                       ? SocketType::color
                                       : SocketType::floating,
                    .operands = make_value_operands<operand::magic>({
                        {operand::magic::vector, *vector},
                        {operand::magic::scale, *scale},
                        {operand::magic::distortion, *distortion},
                        {operand::magic::depth, *depth}})}));
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
                    .operands = make_value_operands<operand::gradient>({
                        {operand::gradient::vector, *vector}}),
                    .static_u0 = mode}));
        }
        return true;
    }
    if (node.type == node_type::wave_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto scale = lower_value_input(node, "Scale");
        auto distortion = lower_value_input(node, "Distortion");
        auto detail = lower_value_input(node, "Detail");
        auto detail_scale = lower_value_input(node, "DetailScale");
        auto detail_roughness =
            lower_value_input(node, "DetailRoughness");
        auto phase = lower_value_input(node, "PhaseOffset");
        if (vector && scale && distortion && detail && detail_scale &&
            detail_roughness && phase) {
            const auto wave_type =
                property_string(node, "WaveType", "BANDS") == "RINGS"
                    ? 1u
                    : 0u;
            const auto direction_code = [](std::string_view name,
                                            std::string_view fourth) noexcept {
                return name == "Y"
                           ? 1u
                           : name == "Z"
                                 ? 2u
                                 : name == fourth ? 3u : 0u;
            };
            const auto bands_direction = direction_code(
                property_string(node, "BandsDirection", "X"),
                "DIAGONAL");
            const auto rings_direction = direction_code(
                property_string(node, "RingsDirection", "X"),
                "SPHERICAL");
            const auto profile_name =
                property_string(node, "Profile", "SIN");
            const auto profile =
                profile_name == "SAW"
                    ? 1u
                    : profile_name == "TRI" ? 2u : 0u;
            const auto configuration =
                wave_type |
                (bands_direction << 8u) |
                (rings_direction << 16u) |
                (profile << 24u);
            const auto needs_color =
                property_bool(node, "NeedsColor");
            auto instruction = ValueInstruction{
                .operation = needs_color
                                 ? ValueOperation::wave_color
                                 : ValueOperation::wave_factor,
                .source_node = node.id,
                .result_type = needs_color
                                   ? SocketType::color
                                   : SocketType::floating,
                .operands = make_value_operands<operand::wave>({
                    {operand::wave::vector, *vector},
                    {operand::wave::scale, *scale},
                    {operand::wave::distortion, *distortion},
                    {operand::wave::detail, *detail},
                    {operand::wave::detail_scale, *detail_scale},
                    {operand::wave::detail_roughness, *detail_roughness},
                    {operand::wave::phase, *phase}}),
                .static_u0 = configuration};
            publish(
                node.id,
                needs_color ? "Color" : "Factor",
                append(std::move(instruction)));
        }
        return true;
    }
    if (node.type == node_type::voronoi_texture) {
        auto vector = lower_value_input(node, "Vector");
        auto w = lower_value_input(node, "W");
        auto scale = lower_value_input(node, "Scale");
        auto detail = lower_value_input(node, "Detail");
        auto roughness = lower_value_input(node, "Roughness");
        auto lacunarity = lower_value_input(node, "Lacunarity");
        auto smoothness = lower_value_input(node, "Smoothness");
        auto exponent = lower_value_input(node, "Exponent");
        auto randomness = lower_value_input(node, "Randomness");
        if (vector && w && scale && detail && roughness &&
            lacunarity && smoothness && exponent && randomness) {
            const auto feature_name =
                property_string(node, "Feature", "F1");
            const auto feature =
                feature_name == "F2"
                    ? VoronoiFeature::f2
                    : feature_name == "SMOOTH_F1"
                          ? VoronoiFeature::smooth_f1
                          : feature_name == "DISTANCE_TO_EDGE"
                                ? VoronoiFeature::distance_to_edge
                                : feature_name == "N_SPHERE_RADIUS"
                                      ? VoronoiFeature::n_sphere_radius
                                      : VoronoiFeature::f1;
            const auto metric_name =
                property_string(
                    node, "DistanceMetric", "EUCLIDEAN");
            const auto metric =
                metric_name == "MANHATTAN"
                    ? VoronoiDistanceMetric::manhattan
                    : metric_name == "CHEBYCHEV"
                          ? VoronoiDistanceMetric::chebychev
                          : metric_name == "MINKOWSKI"
                                ? VoronoiDistanceMetric::minkowski
                                : VoronoiDistanceMetric::euclidean;
            const auto output_name =
                property_string(node, "Output", "Distance");
            auto operation = ValueOperation::voronoi_distance;
            auto result_type = SocketType::floating;
            if (output_name == "Color") {
                operation = ValueOperation::voronoi_color;
                result_type = SocketType::color;
            } else if (output_name == "Position") {
                operation = ValueOperation::voronoi_position;
                result_type = SocketType::vector;
            } else if (output_name == "W") {
                operation = ValueOperation::voronoi_w;
            } else if (output_name == "Radius") {
                operation = ValueOperation::voronoi_radius;
            }
            const auto configuration =
                property_uint(node, "Dimensions", 3u) |
                (static_cast<std::uint64_t>(feature) << 8u) |
                (static_cast<std::uint64_t>(metric) << 16u) |
                (property_bool(node, "Normalize") ? 1ull << 24u : 0u);
            publish(
                node.id,
                output_name,
                append(ValueInstruction{
                    .operation = operation,
                    .source_node = node.id,
                    .result_type = result_type,
                    .operands = make_value_operands<operand::voronoi>({
                        {operand::voronoi::vector, *vector},
                        {operand::voronoi::w, *w},
                        {operand::voronoi::scale, *scale},
                        {operand::voronoi::detail, *detail},
                        {operand::voronoi::roughness, *roughness},
                        {operand::voronoi::lacunarity, *lacunarity},
                        {operand::voronoi::smoothness, *smoothness},
                        {operand::voronoi::exponent, *exponent},
                        {operand::voronoi::randomness, *randomness}}),
                    .static_u0 = configuration}));
        }
        return true;
    }
    if (node.type == node_type::color_ramp) {
        auto factor = lower_value_input(node, "Factor");
        auto table = lower_property_data(node, "Table");
        if (factor && table) {
            auto instruction = ValueInstruction{
                .operation = ValueOperation::color_ramp,
                .source_node = node.id,
                .result_type = SocketType::color,
                .parameter = *table,
                .operands = make_value_operands<operand::color_ramp>({
                    {operand::color_ramp::factor, *factor}}),
                .static_u0 =
                    (property_string(
                         node, "Interpolation", "LINEAR") ==
                             "CONSTANT"
                         ? 1u
                         : 0u) |
                    (property_bool(node, "Sampled")
                         ? 2u
                         : 0u)};
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
        auto min_x = lower_property_parameter(node, "MinX");
        auto max_x = lower_property_parameter(node, "MaxX");
        auto extrapolate =
            lower_property_parameter(node, "Extrapolate");
        auto table = lower_property_data(node, "Table");
        if (factor && color && min_x && max_x && extrapolate && table) {
            publish(
                node.id,
                "Color",
                append(ValueInstruction{
                    .operation = ValueOperation::rgb_curve,
                    .source_node = node.id,
                    .result_type = SocketType::color,
                    .parameter = *table,
                    .operands = make_value_operands<operand::rgb_curve>({
                        {operand::rgb_curve::color, *color},
                        {operand::rgb_curve::factor, *factor},
                        {operand::rgb_curve::min_x, *min_x},
                        {operand::rgb_curve::max_x, *max_x},
                        {operand::rgb_curve::extrapolate, *extrapolate}}),
                    .static_u0 =
                        property_bool(node, "Sampled")
                            ? 1u
                            : 0u}));
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
                        .operands =
                            make_value_operands<operand::separate_color>({
                                {operand::separate_color::color, *color}}),
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
                    .operands = make_value_operands<operand::combine_color>({
                        {operand::combine_color::r, *r},
                        {operand::combine_color::g, *g},
                        {operand::combine_color::b, *b}}),
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
                    .operands = make_value_operands<operand::sky>({
                        {operand::sky::direction, *direction}}),
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
                    .operands = make_value_operands<operand::nishita_sky>({
                        {operand::nishita_sky::elevation, *elevation},
                        {operand::nishita_sky::rotation, *rotation},
                        {operand::nishita_sky::size, *size},
                        {operand::nishita_sky::intensity, *intensity},
                        {operand::nishita_sky::altitude, *altitude},
                        {operand::nishita_sky::air, *air},
                        {operand::nishita_sky::dust, *dust},
                        {operand::nishita_sky::ozone, *ozone},
                        {operand::nishita_sky::direction, *direction}}),
                    .static_u0 = _nishita_count++}));
        }
        return true;
    }
    return false;
}

}// namespace psycles::compiler::detail
