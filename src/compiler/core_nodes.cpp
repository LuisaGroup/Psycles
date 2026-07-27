#include <psycles/compiler/core_nodes.h>

#include <utility>

namespace psycles::compiler {

using contract::NodeRegistry;
using contract::NodeSchema;
using contract::PropertySchema;
using contract::ShaderFeature;
using contract::SocketSchema;
using contract::SocketType;
using contract::SocketValue;
using contract::feature_bit;

namespace {

[[nodiscard]] SocketSchema input(
    std::string name,
    SocketType type,
    SocketValue default_value) {
    return {
        .name = std::move(name),
        .type = type,
        .required = false,
        .default_value = std::move(default_value)};
}

[[nodiscard]] SocketSchema required_input(std::string name, SocketType type) {
    return {
        .name = std::move(name),
        .type = type,
        .required = true,
        .default_value = std::nullopt};
}

[[nodiscard]] SocketSchema output(std::string name, SocketType type) {
    return {
        .name = std::move(name),
        .type = type,
        .required = false,
        .default_value = std::nullopt};
}

[[nodiscard]] PropertySchema property(
    std::string name,
    SocketType type,
    SocketValue default_value) {
    return {
        .name = std::move(name),
        .type = type,
        .required = false,
        .default_value = std::move(default_value)};
}

}// namespace

NodeRegistry make_core_node_registry() {
    NodeRegistry registry;

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::constant_float,
        .inputs = {input("Value", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Value", SocketType::floating)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::constant_color,
        .inputs = {input("Color", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::geometry,
        .inputs = {},
        .outputs = {
            output("Position", SocketType::point),
            output("Normal", SocketType::normal),
            output("GeometricNormal", SocketType::normal),
            output("Incoming", SocketType::vector),
            output("Tangent", SocketType::vector),
            output("Backfacing", SocketType::floating),
            output("RandomPerIsland", SocketType::floating)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::ray_state)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::texture_coordinate,
        .inputs = {},
        .outputs = {
            output("UV", SocketType::vector),
            output("Normal", SocketType::vector),
            output("Generated", SocketType::vector),
            output("Object", SocketType::vector)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::attributes)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::object_info,
        .inputs = {},
        .outputs = {
            output("Location", SocketType::vector),
            output("Random", SocketType::floating)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::attributes)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::particle_info,
        .inputs = {},
        .outputs = {
            output("Index", SocketType::floating),
            output("Random", SocketType::floating)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::attributes)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::light_path,
        .inputs = {},
        .outputs = {
            output("IsCameraRay", SocketType::floating),
            output("IsShadowRay", SocketType::floating),
            output("IsDiffuseRay", SocketType::floating),
            output("IsGlossyRay", SocketType::floating),
            output("IsSingularRay", SocketType::floating),
            output("IsReflectionRay", SocketType::floating),
            output("IsTransmissionRay", SocketType::floating),
            output("IsVolumeScatterRay", SocketType::floating),
            output("RayLength", SocketType::floating),
            output("RayDepth", SocketType::floating),
            output("DiffuseDepth", SocketType::floating),
            output("GlossyDepth", SocketType::floating),
            output("TransparentDepth", SocketType::floating),
            output("TransmissionDepth", SocketType::floating)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::ray_state)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::layer_weight,
        .inputs = {
            input(
                "Blend",
                SocketType::floating,
                SocketValue::floating(0.5f)),
            input(
                "Normal",
                SocketType::normal,
                SocketValue::normal({0.0f, 0.0f, 0.0f}))},
        .outputs = {
            output("Fresnel", SocketType::floating),
            output("Facing", SocketType::floating)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::ray_state)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::mapping,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input("Location", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input("Rotation", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input("Scale", SocketType::vector, SocketValue::vector({1.0f, 1.0f, 1.0f}))},
        .outputs = {output("Vector", SocketType::vector)},
        .properties = {
            property(
                "VectorType",
                SocketType::string,
                SocketValue::string("POINT"))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::image_texture,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {
            output("Color", SocketType::color),
            output("Alpha", SocketType::floating)},
        .properties = {
            property(
                "Image",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(0u)),
            property(
                "Extension",
                SocketType::string,
                SocketValue::string("REPEAT")),
            property(
                "ColorSpace",
                SocketType::string,
                SocketValue::string("sRGB")),
            property(
                "UnassociateAlpha",
                SocketType::boolean,
                SocketValue::boolean(false))},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::derivatives)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::noise_texture,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input("W", SocketType::floating, SocketValue::floating(0.0f)),
            input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
            input("Detail", SocketType::floating, SocketValue::floating(2.0f)),
            input("Roughness", SocketType::floating, SocketValue::floating(0.5f)),
            input("Lacunarity", SocketType::floating, SocketValue::floating(2.0f)),
            input("Offset", SocketType::floating, SocketValue::floating(0.0f)),
            input("Gain", SocketType::floating, SocketValue::floating(1.0f)),
            input("Distortion", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {
            output("Factor", SocketType::floating),
            output("Color", SocketType::color)},
        .properties = {
            property(
                "Dimensions",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(3u)),
            property(
                "Normalize",
                SocketType::boolean,
                SocketValue::boolean(false)),
            property(
                "NoiseType",
                SocketType::string,
                SocketValue::string("FBM")),
            property(
                "NeedsColor",
                SocketType::boolean,
                SocketValue::boolean(false))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::white_noise_texture,
        .inputs = {
            input(
                "Vector",
                SocketType::vector,
                SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input(
                "W",
                SocketType::floating,
                SocketValue::floating(0.0f))},
        .outputs = {
            output("Value", SocketType::floating),
            output("Color", SocketType::color)},
        .properties = {
            property(
                "Dimensions",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(3u)),
            property(
                "NeedsColor",
                SocketType::boolean,
                SocketValue::boolean(false))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::brick_texture,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f})),
            input("Color1", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f})),
            input("Color2", SocketType::color, SocketValue::color({0.2f, 0.2f, 0.2f})),
            input("Mortar", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f})),
            input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
            input("MortarSize", SocketType::floating, SocketValue::floating(0.02f)),
            input("MortarSmooth", SocketType::floating, SocketValue::floating(0.0f)),
            input("Bias", SocketType::floating, SocketValue::floating(0.0f)),
            input("BrickWidth", SocketType::floating, SocketValue::floating(0.5f)),
            input("RowHeight", SocketType::floating, SocketValue::floating(0.25f))},
        .outputs = {
            output("Color", SocketType::color),
            output("Factor", SocketType::floating)},
        .properties = {
            property(
                "OffsetAmount",
                SocketType::floating,
                SocketValue::floating(0.5f)),
            property(
                "OffsetFrequency",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(2u)),
            property(
                "SquashAmount",
                SocketType::floating,
                SocketValue::floating(1.0f)),
            property(
                "SquashFrequency",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(2u))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::gradient_texture,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Factor", SocketType::floating)},
        .properties = {
            property(
                "GradientType",
                SocketType::string,
                SocketValue::string("LINEAR"))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::nishita_sky,
        .inputs = {
            input("SunElevation", SocketType::floating, SocketValue::floating(0.7853982f)),
            input("SunRotation", SocketType::floating, SocketValue::floating(0.0f)),
            input("SunSize", SocketType::floating, SocketValue::floating(0.00918043f)),
            input("SunIntensity", SocketType::floating, SocketValue::floating(1.0f)),
            input("Altitude", SocketType::floating, SocketValue::floating(0.0f)),
            input("AirDensity", SocketType::floating, SocketValue::floating(1.0f)),
            input("DustDensity", SocketType::floating, SocketValue::floating(1.0f)),
            input("OzoneDensity", SocketType::floating, SocketValue::floating(1.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::ray_state)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::add_float,
        .inputs = {
            input("A", SocketType::floating, SocketValue::floating(0.0f)),
            input("B", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Value", SocketType::floating)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::multiply_float,
        .inputs = {
            input("A", SocketType::floating, SocketValue::floating(0.0f)),
            input("B", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Value", SocketType::floating)},
        .properties = {},
        .required_features = {}}));

    for (const auto *type : {
             node_type::subtract_float,
             node_type::divide_float,
             node_type::minimum_float,
             node_type::maximum_float,
             node_type::power_float}) {
        static_cast<void>(registry.register_schema(NodeSchema{
            .type = type,
            .inputs = {
                input("A", SocketType::floating, SocketValue::floating(0.0f)),
                input("B", SocketType::floating, SocketValue::floating(0.0f))},
            .outputs = {output("Value", SocketType::floating)},
            .properties = {},
            .required_features = {}}));
    }

    for (const auto *type : {
             node_type::absolute_float,
             node_type::clamp_float}) {
        static_cast<void>(registry.register_schema(NodeSchema{
            .type = type,
            .inputs = {
                input("Value", SocketType::floating, SocketValue::floating(0.0f))},
            .outputs = {output("Value", SocketType::floating)},
            .properties = {},
            .required_features = {}}));
    }

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::clamp_range,
        .inputs = {
            input("Value", SocketType::floating, SocketValue::floating(1.0f)),
            input("Min", SocketType::floating, SocketValue::floating(0.0f)),
            input("Max", SocketType::floating, SocketValue::floating(1.0f))},
        .outputs = {output("Result", SocketType::floating)},
        .properties = {
            property(
                "Mode",
                SocketType::string,
                SocketValue::string("MINMAX"))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::scalar_to_color,
        .inputs = {
            input("Value", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::color_to_scalar,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Value", SocketType::floating)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::vector_to_scalar,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Value", SocketType::floating)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::vector_to_color,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::color_to_vector,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Vector", SocketType::vector)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::vector_to_normal,
        .inputs = {
            input("Vector", SocketType::vector, SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Normal", SocketType::normal)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::normal_to_vector,
        .inputs = {
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 1.0f}))},
        .outputs = {output("Vector", SocketType::vector)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::mix_color,
        .inputs = {
            input("Factor", SocketType::floating, SocketValue::floating(0.5f)),
            input("A", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f})),
            input("B", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {
            property(
                "BlendMode",
                SocketType::string,
                SocketValue::string("MIX"))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::multiply_color,
        .inputs = {
            input("A", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f})),
            input("B", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f})),
            input("Factor", SocketType::floating, SocketValue::floating(1.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::hue_saturation,
        .inputs = {
            input("Hue", SocketType::floating, SocketValue::floating(0.5f)),
            input("Saturation", SocketType::floating, SocketValue::floating(1.0f)),
            input("Value", SocketType::floating, SocketValue::floating(1.0f)),
            input("Factor", SocketType::floating, SocketValue::floating(1.0f)),
            input("Color", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::invert_color,
        .inputs = {
            input("Factor", SocketType::floating, SocketValue::floating(1.0f)),
            input("Color", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::gamma_color,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f})),
            input("Gamma", SocketType::floating, SocketValue::floating(1.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::brightness_contrast,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f})),
            input("Bright", SocketType::floating, SocketValue::floating(0.0f)),
            input("Contrast", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::color_ramp,
        .inputs = {
            input("Factor", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {
            output("Color", SocketType::color),
            output("Alpha", SocketType::floating)},
        .properties = {
            property(
                "Interpolation",
                SocketType::string,
                SocketValue::string("LINEAR")),
            property(
                "Table",
                SocketType::string,
                SocketValue::string("0,0,0,0,1;1,1,1,1,1"))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::rgb_curve,
        .inputs = {
            input("Factor", SocketType::floating, SocketValue::floating(1.0f)),
            input("Color", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {
            property(
                "Table",
                SocketType::string,
                SocketValue::string(""))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::separate_color,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.0f, 0.0f, 0.0f}))},
        .outputs = {
            output("R", SocketType::floating),
            output("G", SocketType::floating),
            output("B", SocketType::floating)},
        .properties = {
            property(
                "Mode",
                SocketType::string,
                SocketValue::string("RGB"))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::combine_color,
        .inputs = {
            input("R", SocketType::floating, SocketValue::floating(0.0f)),
            input("G", SocketType::floating, SocketValue::floating(0.0f)),
            input("B", SocketType::floating, SocketValue::floating(0.0f))},
        .outputs = {output("Color", SocketType::color)},
        .properties = {
            property(
                "Mode",
                SocketType::string,
                SocketValue::string("RGB"))},
        .required_features = {}}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::normal_map,
        .inputs = {
            input("Strength", SocketType::floating, SocketValue::floating(1.0f)),
            input("Color", SocketType::color, SocketValue::color({0.5f, 0.5f, 1.0f}))},
        .outputs = {output("Normal", SocketType::normal)},
        .properties = {
            property(
                "Space",
                SocketType::string,
                SocketValue::string("TANGENT"))},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::derivatives)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::bump,
        .inputs = {
            input("Height", SocketType::floating, SocketValue::floating(1.0f)),
            input("Strength", SocketType::floating, SocketValue::floating(1.0f)),
            input("Distance", SocketType::floating, SocketValue::floating(1.0f)),
            input("FilterWidth", SocketType::floating, SocketValue::floating(1.0f)),
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 1.0f}))},
        .outputs = {output("Normal", SocketType::normal)},
        .properties = {
            property(
                "Invert",
                SocketType::boolean,
                SocketValue::boolean(false))},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::derivatives)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::vertex_color,
        .inputs = {},
        .outputs = {
            output("Color", SocketType::color),
            output("Alpha", SocketType::floating)},
        .properties = {
            property(
                "AttributeId",
                SocketType::unsigned_integer,
                SocketValue::unsigned_integer(0u))},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::attributes)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::diffuse_bsdf,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f})),
            input("Roughness", SocketType::floating, SocketValue::floating(0.0f)),
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::translucent_bsdf,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f})),
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::principled_bsdf,
        .inputs = {
            input("BaseColor", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f})),
            input("Metallic", SocketType::floating, SocketValue::floating(0.0f)),
            input("Roughness", SocketType::floating, SocketValue::floating(0.5f)),
            input("DiffuseRoughness", SocketType::floating, SocketValue::floating(0.0f)),
            input("IOR", SocketType::floating, SocketValue::floating(1.5f)),
            input("SpecularIORLevel", SocketType::floating, SocketValue::floating(0.5f)),
            input("SpecularTint", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f})),
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {
            property("Distribution", SocketType::string, SocketValue::string("GGX"))},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::glossy_bsdf,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({0.8f, 0.8f, 0.8f})),
            input("Roughness", SocketType::floating, SocketValue::floating(0.2f)),
            input("Normal", SocketType::normal, SocketValue::normal({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::emission,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f})),
            input("Strength", SocketType::floating, SocketValue::floating(1.0f))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::emission)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::transparent_bsdf,
        .inputs = {
            input("Color", SocketType::color, SocketValue::color({1.0f, 1.0f, 1.0f}))},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features =
            feature_bit(ShaderFeature::surface) |
            feature_bit(ShaderFeature::transparency)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::add_closure,
        .inputs = {
            required_input("A", SocketType::closure),
            required_input("B", SocketType::closure)},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features = feature_bit(ShaderFeature::surface)}));

    static_cast<void>(registry.register_schema(NodeSchema{
        .type = node_type::mix_closure,
        .inputs = {
            input("Factor", SocketType::floating, SocketValue::floating(0.5f)),
            required_input("A", SocketType::closure),
            required_input("B", SocketType::closure)},
        .outputs = {output("Closure", SocketType::closure)},
        .properties = {},
        .required_features = feature_bit(ShaderFeature::surface)}));

    return registry;
}

}// namespace psycles::compiler
