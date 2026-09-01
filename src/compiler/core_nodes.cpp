#include <psycles/compiler/core_nodes.h>

#include <utility>

namespace psycles::compiler {

using contract::feature_bit;
using contract::NodeRegistry;
using contract::NodeSchema;
using contract::PropertySchema;
using contract::ShaderFeature;
using contract::SocketSchema;
using contract::SocketType;
using contract::SocketValue;

namespace {

[[nodiscard]] SocketSchema input(std::string name, SocketType type,
                                 SocketValue default_value) {
  return {.name = std::move(name),
          .type = type,
          .required = false,
          .default_value = std::move(default_value)};
}

[[nodiscard]] SocketSchema required_input(std::string name, SocketType type) {
  return {.name = std::move(name),
          .type = type,
          .required = true,
          .default_value = std::nullopt};
}

[[nodiscard]] SocketSchema output(std::string name, SocketType type) {
  return {.name = std::move(name),
          .type = type,
          .required = false,
          .default_value = std::nullopt};
}

[[nodiscard]] PropertySchema property(std::string name, SocketType type,
                                      SocketValue default_value) {
  return {.name = std::move(name),
          .type = type,
          .required = false,
          .default_value = std::move(default_value)};
}

[[nodiscard]] PropertySchema runtime_property(
    std::string name,
    SocketType type,
    SocketValue default_value) {
  auto schema = property(
      std::move(name), type, std::move(default_value));
  schema.role = contract::PropertyRole::runtime_parameter;
  return schema;
}

} // namespace

NodeRegistry make_core_node_registry() {
  NodeRegistry registry;

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::constant_float,
                 .inputs = {input("Value", SocketType::floating,
                                  SocketValue::floating(0.0f))},
                 .outputs = {output("Value", SocketType::floating)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::constant_color,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::geometry,
                 .inputs = {},
                 .outputs = {output("Position", SocketType::point),
                             output("Normal", SocketType::normal),
                             output("GeometricNormal", SocketType::normal),
                             output("Incoming", SocketType::vector),
                             output("Tangent", SocketType::vector),
                             output("Parametric", SocketType::point),
                             output("Backfacing", SocketType::floating),
                             output("Pointiness", SocketType::floating),
                             output("RandomPerIsland", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::camera_data,
                 .inputs = {},
                 .outputs = {output("View Vector", SocketType::vector),
                             output("View Z Depth", SocketType::floating),
                             output("View Distance", SocketType::floating)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::texture_coordinate,
      .inputs = {},
      .outputs = {output("Generated", SocketType::point),
                  output("Normal", SocketType::normal),
                  output("UV", SocketType::point),
                  output("Object", SocketType::point),
                  output("Camera", SocketType::point),
                  output("Window", SocketType::point),
                  output("Reflection", SocketType::normal)},
      .properties = {property("FromDupli", SocketType::boolean,
                              SocketValue::boolean(false)),
                     property("UseTransform", SocketType::boolean,
                              SocketValue::boolean(false)),
                     property("ObjectTransform", SocketType::transform,
                              SocketValue::transform(Mat4f{})),
                     property("UvMapNamed", SocketType::boolean,
                              SocketValue::boolean(false)),
                     runtime_property(
                         "UvMapId", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(0u))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::uv_map,
      .inputs = {},
      .outputs = {output("UV", SocketType::point)},
      .properties = {property("FromDupli", SocketType::boolean,
                              SocketValue::boolean(false)),
                     property("Attribute", SocketType::string,
                              SocketValue::string("")),
                     // Compatibility binding for the retiring surface-value
                     // program. The Cycles SVM compiler consumes Attribute.
                     runtime_property("AttributeId",
                                      SocketType::unsigned_integer,
                                      SocketValue::unsigned_integer(0u))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::object_info,
                 .inputs = {},
                 .outputs = {output("Location", SocketType::vector),
                             output("Color", SocketType::color),
                             output("Alpha", SocketType::floating),
                             output("ObjectIndex", SocketType::floating),
                             output("MaterialIndex", SocketType::floating),
                             output("Random", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::particle_info,
                 .inputs = {},
                 .outputs = {output("Index", SocketType::floating),
                             output("Random", SocketType::floating),
                             output("Age", SocketType::floating),
                             output("Lifetime", SocketType::floating),
                             output("Location", SocketType::point),
                             output("Size", SocketType::floating),
                             output("Velocity", SocketType::vector),
                             output("AngularVelocity", SocketType::vector)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::point_info,
                 .inputs = {},
                 .outputs = {output("Position", SocketType::point),
                             output("Radius", SocketType::floating),
                             output("Random", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::hair_info,
                 .inputs = {},
                 .outputs = {output("IsStrand", SocketType::floating),
                             output("Intercept", SocketType::floating),
                             output("Length", SocketType::floating),
                             output("Thickness", SocketType::floating),
                             output("TangentNormal", SocketType::normal),
                             output("Random", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::light_path,
                 .inputs = {},
                 .outputs = {output("IsCameraRay", SocketType::floating),
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
                             output("TransmissionDepth", SocketType::floating),
                             output("PortalDepth", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::light_falloff,
      .inputs =
          {input("Strength", SocketType::floating,
                 SocketValue::floating(100.0f)),
           input("Smooth", SocketType::floating,
                 SocketValue::floating(0.0f))},
      .outputs = {output("Quadratic", SocketType::floating),
                  output("Linear", SocketType::floating),
                  output("Constant", SocketType::floating)},
      .properties = {},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::layer_weight,
                 .inputs = {input("Blend", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Fresnel", SocketType::floating),
                             output("Facing", SocketType::floating)},
                 .properties = {property("NormalLinked", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::fresnel,
                 .inputs = {input("IOR", SocketType::floating,
                                  SocketValue::floating(1.5f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Factor", SocketType::floating)},
                 .properties = {property("NormalLinked", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::ambient_occlusion,
      .inputs = {input("Distance", SocketType::floating,
                       SocketValue::floating(1.0f)),
                 input("Normal", SocketType::normal,
                       SocketValue::normal({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("AO", SocketType::floating)},
      // Samples is material data, not shader shape. Blender import narrows it
      // to the exact uint8 value stored by Cycles' SVM instruction.
      .properties = {
          runtime_property("Samples", SocketType::unsigned_integer,
                           SocketValue::unsigned_integer(16u)),
          property("NormalLinked", SocketType::boolean,
                   SocketValue::boolean(false)),
          property("Inside", SocketType::boolean,
                   SocketValue::boolean(false)),
          property("OnlyLocal", SocketType::boolean,
                   SocketValue::boolean(false)),
          property("GlobalRadius", SocketType::boolean,
                   SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::ray_state) |
                           feature_bit(ShaderFeature::ambient_occlusion)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::mapping,
      .inputs = {input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f})),
                 input("Location", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f})),
                 input("Rotation", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f})),
                 input("Scale", SocketType::vector,
                       SocketValue::vector({1.0f, 1.0f, 1.0f}))},
      .outputs = {output("Vector", SocketType::vector)},
      .properties =
          {property("VectorType", SocketType::string,
                    SocketValue::string("POINT")),
           property("XMapping", SocketType::string, SocketValue::string("X")),
           property("YMapping", SocketType::string, SocketValue::string("Y")),
           property("ZMapping", SocketType::string, SocketValue::string("Z")),
           // The Blender adapter temporarily represents TextureNode's
           // embedded TextureMapping as a graph node at the serialization
           // boundary. Cycles SVM projection consumes this marker and folds
           // the state back into the texture node before graph optimization;
           // an authored Mapping node always keeps the default false value.
           property("LegacyTextureMapping", SocketType::boolean,
                    SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::image_texture,
      .inputs = {input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Color", SocketType::color),
                  output("Alpha", SocketType::floating)},
      .properties = {runtime_property(
                         "Image", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(0u)),
                     property("Extension", SocketType::string,
                              SocketValue::string("REPEAT")),
                     property("Interpolation", SocketType::string,
                              SocketValue::string("Linear")),
                     property("Projection", SocketType::string,
                              SocketValue::string("FLAT")),
                     runtime_property(
                         "ProjectionBlend", SocketType::floating,
                         SocketValue::floating(0.0f)),
                     property("ColorSpace", SocketType::string,
                              SocketValue::string("sRGB")),
                     property("UnassociateAlpha", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::derivatives)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::environment_texture,
      .inputs = {input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Color", SocketType::color),
                  output("Alpha", SocketType::floating)},
      .properties = {runtime_property(
                         "Image", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(0u)),
                     property("Interpolation", SocketType::string,
                              SocketValue::string("Linear")),
                     property("Projection", SocketType::string,
                              SocketValue::string("EQUIRECTANGULAR")),
                     property("ColorSpace", SocketType::string,
                              SocketValue::string("Non-Color"))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::derivatives)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::noise_texture,
      .inputs =
          {input("Vector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("W", SocketType::floating, SocketValue::floating(0.0f)),
           input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
           input("Detail", SocketType::floating, SocketValue::floating(2.0f)),
           input("Roughness", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("Lacunarity", SocketType::floating,
                 SocketValue::floating(2.0f)),
           input("Offset", SocketType::floating, SocketValue::floating(0.0f)),
           input("Gain", SocketType::floating, SocketValue::floating(1.0f)),
           input("Distortion", SocketType::floating,
                 SocketValue::floating(0.0f))},
      .outputs = {output("Factor", SocketType::floating),
                  output("Color", SocketType::color)},
      .properties = {property("Dimensions", SocketType::unsigned_integer,
                              SocketValue::unsigned_integer(3u)),
                     property("Normalize", SocketType::boolean,
                              SocketValue::boolean(true)),
                     property("NoiseType", SocketType::string,
                              SocketValue::string("FBM")),
                     property("NeedsColor", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::white_noise_texture,
      .inputs = {input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f})),
                 input("W", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Value", SocketType::floating),
                  output("Color", SocketType::color)},
      .properties = {property("Dimensions", SocketType::unsigned_integer,
                              SocketValue::unsigned_integer(3u)),
                     property("NeedsColor", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::checker_texture,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Color1", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Color2", SocketType::color,
                                  SocketValue::color({0.2f, 0.2f, 0.2f})),
                            input("Scale", SocketType::floating,
                                  SocketValue::floating(5.0f))},
                 .outputs = {output("Color", SocketType::color),
                             output("Factor", SocketType::floating)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::brick_texture,
      .inputs =
          {input("Vector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("Color1", SocketType::color,
                 SocketValue::color({0.8f, 0.8f, 0.8f})),
           input("Color2", SocketType::color,
                 SocketValue::color({0.2f, 0.2f, 0.2f})),
           input("Mortar", SocketType::color,
                 SocketValue::color({0.0f, 0.0f, 0.0f})),
           input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
           input("MortarSize", SocketType::floating,
                 SocketValue::floating(0.02f)),
           input("MortarSmooth", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("Bias", SocketType::floating, SocketValue::floating(0.0f)),
           input("BrickWidth", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("RowHeight", SocketType::floating,
                 SocketValue::floating(0.25f))},
      .outputs = {output("Color", SocketType::color),
                  output("Factor", SocketType::floating)},
      .properties = {runtime_property(
                         "OffsetAmount", SocketType::floating,
                         SocketValue::floating(0.5f)),
                     runtime_property(
                         "OffsetFrequency", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(2u)),
                     runtime_property(
                         "SquashAmount", SocketType::floating,
                         SocketValue::floating(1.0f)),
                     runtime_property(
                         "SquashFrequency", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(2u))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::magic_texture,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Scale", SocketType::floating,
                                  SocketValue::floating(5.0f)),
                            input("Distortion", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Color", SocketType::color),
                             output("Factor", SocketType::floating)},
                 .properties = {runtime_property(
                                    "Depth", SocketType::unsigned_integer,
                                    SocketValue::unsigned_integer(2u)),
                                property("NeedsColor", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::wave_texture,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Scale", SocketType::floating,
                                  SocketValue::floating(5.0f)),
                            input("Distortion", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Detail", SocketType::floating,
                                  SocketValue::floating(2.0f)),
                            input("DetailScale", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("DetailRoughness", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            input("PhaseOffset", SocketType::floating,
                                  SocketValue::floating(0.0f))},
                 .outputs = {output("Color", SocketType::color),
                             output("Factor", SocketType::floating)},
                 .properties = {property("WaveType", SocketType::string,
                                         SocketValue::string("BANDS")),
                                property("BandsDirection", SocketType::string,
                                         SocketValue::string("X")),
                                property("RingsDirection", SocketType::string,
                                         SocketValue::string("X")),
                                property("Profile", SocketType::string,
                                         SocketValue::string("SIN")),
                                property("NeedsColor", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::gabor_texture,
      .inputs =
          {input("Vector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
           input("Frequency", SocketType::floating,
                 SocketValue::floating(2.0f)),
           input("Anisotropy", SocketType::floating,
                 SocketValue::floating(1.0f)),
           input("Orientation 2D", SocketType::floating,
                 SocketValue::floating(0.7853981633974483f)),
           input("Orientation 3D", SocketType::vector,
                 SocketValue::vector(
                     {1.4142135623730951f, 1.4142135623730951f, 0.0f}))},
      .outputs = {output("Value", SocketType::floating),
                  output("Phase", SocketType::floating),
                  output("Intensity", SocketType::floating)},
      .properties = {property("Type", SocketType::string,
                              SocketValue::string("2D"))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::voronoi_texture,
      .inputs =
          {input("Vector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("W", SocketType::floating, SocketValue::floating(0.0f)),
           input("Scale", SocketType::floating, SocketValue::floating(5.0f)),
           input("Detail", SocketType::floating, SocketValue::floating(0.0f)),
           input("Roughness", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("Lacunarity", SocketType::floating,
                 SocketValue::floating(2.0f)),
           input("Smoothness", SocketType::floating,
                 SocketValue::floating(1.0f)),
           input("Exponent", SocketType::floating, SocketValue::floating(0.5f)),
           input("Randomness", SocketType::floating,
                 SocketValue::floating(1.0f))},
      .outputs = {output("Distance", SocketType::floating),
                  output("Color", SocketType::color),
                  output("Position", SocketType::vector),
                  output("W", SocketType::floating),
                  output("Radius", SocketType::floating)},
      .properties = {property("Dimensions", SocketType::unsigned_integer,
                              SocketValue::unsigned_integer(3u)),
                     property("Feature", SocketType::string,
                              SocketValue::string("F1")),
                     property("DistanceMetric", SocketType::string,
                              SocketValue::string("EUCLIDEAN")),
                     property("Normalize", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::gradient_texture,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color),
                             output("Factor", SocketType::floating)},
                 .properties = {property("GradientType", SocketType::string,
                                         SocketValue::string("LINEAR"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::ies_light,
                 .inputs = {input("Strength", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Factor", SocketType::floating)},
                 .properties = {property("IES", SocketType::string,
                                         SocketValue::string(""))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::hosek_wilkie_sky,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {property("SunDirectionX", SocketType::floating,
                                         SocketValue::floating(0.0f)),
                                property("SunDirectionY", SocketType::floating,
                                         SocketValue::floating(0.0f)),
                                property("SunDirectionZ", SocketType::floating,
                                         SocketValue::floating(1.0f)),
                                property("Turbidity", SocketType::floating,
                                         SocketValue::floating(2.2f)),
                                property("GroundAlbedo", SocketType::floating,
                                         SocketValue::floating(0.3f))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::nishita_sky,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("SunElevation", SocketType::floating,
                                  SocketValue::floating(0.7853982f)),
                            input("SunRotation", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("SunSize", SocketType::floating,
                                  SocketValue::floating(0.00918043f)),
                            input("SunIntensity", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Altitude", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("AirDensity", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("DustDensity", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("OzoneDensity", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::ray_state)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::math,
      .inputs = {input("A", SocketType::floating, SocketValue::floating(0.5f)),
                 input("B", SocketType::floating, SocketValue::floating(0.5f)),
                 input("C", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Value", SocketType::floating)},
      .properties = {property("Operation", SocketType::string,
                              SocketValue::string("ADD"))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::add_float,
      .inputs = {input("A", SocketType::floating, SocketValue::floating(0.0f)),
                 input("B", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Value", SocketType::floating)},
      .properties = {},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::multiply_float,
      .inputs = {input("A", SocketType::floating, SocketValue::floating(0.0f)),
                 input("B", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Value", SocketType::floating)},
      .properties = {},
      .required_features = {}}));

  for (const auto *type : {node_type::subtract_float, node_type::divide_float,
                           node_type::minimum_float, node_type::maximum_float,
                           node_type::power_float}) {
    static_cast<void>(registry.register_schema(
        NodeSchema{.type = type,
                   .inputs = {input("A", SocketType::floating,
                                    SocketValue::floating(0.0f)),
                              input("B", SocketType::floating,
                                    SocketValue::floating(0.0f))},
                   .outputs = {output("Value", SocketType::floating)},
                   .properties = {},
                   .required_features = {}}));
  }

  for (const auto *type : {node_type::absolute_float, node_type::clamp_float}) {
    static_cast<void>(registry.register_schema(
        NodeSchema{.type = type,
                   .inputs = {input("Value", SocketType::floating,
                                    SocketValue::floating(0.0f))},
                   .outputs = {output("Value", SocketType::floating)},
                   .properties = {},
                   .required_features = {}}));
  }

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::clamp_range,
      .inputs =
          {input("Value", SocketType::floating, SocketValue::floating(1.0f)),
           input("Min", SocketType::floating, SocketValue::floating(0.0f)),
           input("Max", SocketType::floating, SocketValue::floating(1.0f))},
      .outputs = {output("Result", SocketType::floating)},
      .properties = {property("Mode", SocketType::string,
                              SocketValue::string("MINMAX"))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::map_range,
      .inputs =
          {input("Value", SocketType::floating, SocketValue::floating(1.0f)),
           input("FromMin", SocketType::floating, SocketValue::floating(0.0f)),
           input("FromMax", SocketType::floating, SocketValue::floating(1.0f)),
           input("ToMin", SocketType::floating, SocketValue::floating(0.0f)),
           input("ToMax", SocketType::floating, SocketValue::floating(1.0f)),
           input("Steps", SocketType::floating, SocketValue::floating(4.0f)),
           input("Vector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("FromMinVector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("FromMaxVector", SocketType::vector,
                 SocketValue::vector({1.0f, 1.0f, 1.0f})),
           input("ToMinVector", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("ToMaxVector", SocketType::vector,
                 SocketValue::vector({1.0f, 1.0f, 1.0f})),
           input("StepsVector", SocketType::vector,
                 SocketValue::vector({4.0f, 4.0f, 4.0f}))},
      .outputs = {output("Result", SocketType::floating),
                  output("Vector", SocketType::vector)},
      .properties = {property("DataType", SocketType::string,
                              SocketValue::string("FLOAT")),
                     property("Interpolation", SocketType::string,
                              SocketValue::string("LINEAR")),
                     property("Clamp", SocketType::boolean,
                              SocketValue::boolean(true))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::vector_math,
                 .inputs = {input("A", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("B", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("C", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Scale", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Vector", SocketType::vector),
                             output("Value", SocketType::floating)},
                 .properties = {property("Operation", SocketType::string,
                                         SocketValue::string("ADD"))},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::vector_rotate,
      .inputs = {input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f})),
                 input("Rotation", SocketType::point,
                       SocketValue::point({0.0f, 0.0f, 0.0f})),
                 input("Center", SocketType::point,
                       SocketValue::point({0.0f, 0.0f, 0.0f})),
                 input("Axis", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 1.0f})),
                 input("Angle", SocketType::floating,
                       SocketValue::floating(0.0f))},
      .outputs = {output("Vector", SocketType::vector)},
      .properties = {property("Type", SocketType::string,
                              SocketValue::string("AXIS_ANGLE")),
                     property("Invert", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::normal,
      .inputs = {input("Normal", SocketType::normal,
                       SocketValue::normal({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Normal", SocketType::normal),
                  output("Dot", SocketType::floating)},
      .properties = {runtime_property(
          "Direction", SocketType::vector,
          SocketValue::vector({0.0f, 0.0f, 0.0f}))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::vector_transform,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Vector", SocketType::vector)},
                 .properties = {property("Type", SocketType::string,
                                         SocketValue::string("VECTOR")),
                                property("Convert From", SocketType::string,
                                         SocketValue::string("WORLD")),
                                property("Convert To", SocketType::string,
                                         SocketValue::string("OBJECT"))},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::wireframe,
      .inputs = {input("Size", SocketType::floating,
                       SocketValue::floating(0.01f))},
      .outputs = {output("Fac", SocketType::floating)},
      .properties = {property("Use Pixel Size", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::derivatives)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::scalar_to_color,
                 .inputs = {input("Value", SocketType::floating,
                                  SocketValue::floating(0.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::scalar_to_boolean,
                 .inputs = {input("Value", SocketType::floating,
                                  SocketValue::floating(0.0f))},
                 .outputs = {output("Boolean", SocketType::boolean)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::color_to_scalar,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Value", SocketType::floating)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::vector_to_scalar,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Value", SocketType::floating)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::vector_to_color,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::color_to_vector,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Vector", SocketType::vector)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::point_to_vector,
                 .inputs = {input("Point", SocketType::point,
                                  SocketValue::point({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Vector", SocketType::vector)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::float3_to_vector,
                 .inputs = {input("Value", SocketType::float3,
                                  SocketValue::float3({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Vector", SocketType::vector)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::vector_to_normal,
                 .inputs = {input("Vector", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Normal", SocketType::normal)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::normal_to_vector,
                 .inputs = {input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 1.0f}))},
                 .outputs = {output("Vector", SocketType::vector)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::mix_float,
      .inputs = {input("Factor", SocketType::floating,
                       SocketValue::floating(0.5f)),
                 input("A", SocketType::floating, SocketValue::floating(0.0f)),
                 input("B", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Value", SocketType::floating)},
      .properties = {property("ClampFactor", SocketType::boolean,
                              SocketValue::boolean(true))},
      .required_features = {}}));

  for (const auto *type :
       {node_type::mix_vector, node_type::mix_vector_nonuniform}) {
    static_cast<void>(registry.register_schema(NodeSchema{
        .type = type,
        .inputs = {input("Factor",
                         type == node_type::mix_vector ? SocketType::floating
                                                       : SocketType::vector,
                         type == node_type::mix_vector
                             ? SocketValue::floating(0.5f)
                             : SocketValue::vector({0.5f, 0.5f, 0.5f})),
                   input("A", SocketType::vector,
                         SocketValue::vector({0.0f, 0.0f, 0.0f})),
                   input("B", SocketType::vector,
                         SocketValue::vector({0.0f, 0.0f, 0.0f}))},
        .outputs = {output("Vector", SocketType::vector)},
        .properties = {property("ClampFactor", SocketType::boolean,
                                SocketValue::boolean(true))},
        .required_features = {}}));
  }

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::legacy_mix_color,
                 .inputs = {input("Factor", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            input("A", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f})),
                            input("B", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {property("BlendMode", SocketType::string,
                                         SocketValue::string("MIX")),
                                property("ClampResult", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::mix_color,
                 .inputs = {input("Factor", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            input("A", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f})),
                            input("B", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {property("BlendMode", SocketType::string,
                                         SocketValue::string("MIX")),
                                property("ClampFactor", SocketType::boolean,
                                         SocketValue::boolean(true)),
                                property("ClampResult", SocketType::boolean,
                                         SocketValue::boolean(false))},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::multiply_color,
                 .inputs = {input("A", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f})),
                            input("B", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Factor", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::hue_saturation,
      .inputs =
          {input("Hue", SocketType::floating, SocketValue::floating(0.5f)),
           input("Saturation", SocketType::floating,
                 SocketValue::floating(1.0f)),
           input("Value", SocketType::floating, SocketValue::floating(1.0f)),
           input("Factor", SocketType::floating, SocketValue::floating(1.0f)),
           input("Color", SocketType::color,
                 SocketValue::color({0.8f, 0.8f, 0.8f}))},
      .outputs = {output("Color", SocketType::color)},
      .properties = {},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::invert_color,
                 .inputs = {input("Factor", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f}))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::gamma_color,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Gamma", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::brightness_contrast,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Bright", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Contrast", SocketType::floating,
                                  SocketValue::floating(0.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::blackbody,
                 .inputs = {input("Temperature", SocketType::floating,
                                  SocketValue::floating(1200.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::wavelength,
                 .inputs = {input("Wavelength", SocketType::floating,
                                  SocketValue::floating(500.0f))},
                 .outputs = {output("Color", SocketType::color)},
                 .properties = {},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::color_ramp,
      .inputs = {input("Factor", SocketType::floating,
                       SocketValue::floating(0.0f))},
      .outputs = {output("Color", SocketType::color),
                  output("Alpha", SocketType::floating)},
      .properties = {property("Interpolation", SocketType::string,
                              SocketValue::string("LINEAR")),
                     property("Sampled", SocketType::boolean,
                              SocketValue::boolean(false)),
                     runtime_property(
                         "Table", SocketType::string,
                         SocketValue::string("0,0,0,0,1;1,1,1,1,1"))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::rgb_curve,
      .inputs = {input("Factor", SocketType::floating,
                       SocketValue::floating(1.0f)),
                 input("Color", SocketType::color,
                       SocketValue::color({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Color", SocketType::color)},
      .properties =
          {property("Sampled", SocketType::boolean,
                    SocketValue::boolean(false)),
           runtime_property(
               "MinX", SocketType::floating,
               SocketValue::floating(0.0f)),
           runtime_property(
               "MaxX", SocketType::floating,
               SocketValue::floating(1.0f)),
           runtime_property(
               "Extrapolate", SocketType::boolean,
               SocketValue::boolean(true)),
           runtime_property(
               "Table", SocketType::string,
               SocketValue::string(""))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::vector_curve,
      .inputs = {input("Factor", SocketType::floating,
                       SocketValue::floating(1.0f)),
                 input("Vector", SocketType::vector,
                       SocketValue::vector({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Vector", SocketType::vector)},
      .properties =
          {property("Sampled", SocketType::boolean,
                    SocketValue::boolean(false)),
           runtime_property(
               "MinX", SocketType::floating,
               SocketValue::floating(0.0f)),
           runtime_property(
               "MaxX", SocketType::floating,
               SocketValue::floating(1.0f)),
           runtime_property(
               "Extrapolate", SocketType::boolean,
               SocketValue::boolean(true)),
           runtime_property(
               "Table", SocketType::string,
               SocketValue::string(""))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::float_curve,
      .inputs = {input("Factor", SocketType::floating,
                       SocketValue::floating(1.0f)),
                 input("Value", SocketType::floating,
                       SocketValue::floating(1.0f))},
      .outputs = {output("Value", SocketType::floating)},
      .properties =
          {property("Sampled", SocketType::boolean,
                    SocketValue::boolean(false)),
           runtime_property(
               "MinX", SocketType::floating,
               SocketValue::floating(0.0f)),
           runtime_property(
               "MaxX", SocketType::floating,
               SocketValue::floating(1.0f)),
           runtime_property(
               "Extrapolate", SocketType::boolean,
               SocketValue::boolean(true)),
           runtime_property(
               "Table", SocketType::string,
               SocketValue::string(""))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::separate_color,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("R", SocketType::floating),
                             output("G", SocketType::floating),
                             output("B", SocketType::floating)},
                 .properties = {property("Mode", SocketType::string,
                                         SocketValue::string("RGB"))},
                 .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::combine_color,
      .inputs = {input("R", SocketType::floating, SocketValue::floating(0.0f)),
                 input("G", SocketType::floating, SocketValue::floating(0.0f)),
                 input("B", SocketType::floating, SocketValue::floating(0.0f))},
      .outputs = {output("Color", SocketType::color)},
      .properties = {property("Mode", SocketType::string,
                              SocketValue::string("RGB"))},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::separate_xyz,
      // Cycles 5.2.1 SeparateXYZNode declares this as SOCKET_IN_COLOR even
      // though Blender presents a vector socket.
      .inputs = {input("Vector", SocketType::color,
                       SocketValue::color({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("X", SocketType::floating),
                  output("Y", SocketType::floating),
                  output("Z", SocketType::floating)},
      .properties = {},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::combine_xyz,
      .inputs = {input("X", SocketType::floating,
                       SocketValue::floating(0.0f)),
                 input("Y", SocketType::floating,
                       SocketValue::floating(0.0f)),
                 input("Z", SocketType::floating,
                       SocketValue::floating(0.0f))},
      .outputs = {output("Vector", SocketType::vector)},
      .properties = {},
      .required_features = {}}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::normal_map,
      .inputs = {input("Strength", SocketType::floating,
                       SocketValue::floating(1.0f)),
                 input("Color", SocketType::color,
                       SocketValue::color({0.5f, 0.5f, 1.0f}))},
      .outputs = {output("Normal", SocketType::normal)},
      .properties = {property("Space", SocketType::string,
                              SocketValue::string("TANGENT")),
                     property("Convention", SocketType::string,
                              SocketValue::string("OPENGL")),
                     property("Base", SocketType::string,
                              SocketValue::string("ORIGINAL")),
                     property("Attribute", SocketType::string,
                              SocketValue::string("")),
                     property("UvMapNamed", SocketType::boolean,
                              SocketValue::boolean(false)),
                     runtime_property(
                         "UvMapId", SocketType::unsigned_integer,
                         SocketValue::unsigned_integer(0u))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::derivatives)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::tangent,
      .inputs = {},
      .outputs = {output("Tangent", SocketType::normal)},
      .properties = {property("Direction Type", SocketType::string,
                              SocketValue::string("RADIAL")),
                     property("Axis", SocketType::string,
                              SocketValue::string("X")),
                     property("Attribute", SocketType::string,
                              SocketValue::string(""))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::bump,
      .inputs =
          {input("Height", SocketType::floating, SocketValue::floating(1.0f)),
           input("Strength", SocketType::floating, SocketValue::floating(1.0f)),
           input("Distance", SocketType::floating, SocketValue::floating(0.1f)),
           input("FilterWidth", SocketType::floating,
                 SocketValue::floating(0.1f)),
           input("Normal", SocketType::normal,
                 SocketValue::normal({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Normal", SocketType::normal)},
      .properties = {property("Invert", SocketType::boolean,
                              SocketValue::boolean(false)),
                     property("NormalLinked", SocketType::boolean,
                              SocketValue::boolean(false)),
                     // Cycles enables this only for the automatic bump
                     // stage of a BOTH displacement material. It changes
                     // the differential frame in which the surface
                     // gradient is constructed.
                     property("UseObjectSpace", SocketType::boolean,
                              SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::derivatives)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::displacement,
      .inputs =
          {input("Height", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("Midlevel", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("Scale", SocketType::floating,
                 SocketValue::floating(1.0f)),
           input("Normal", SocketType::normal,
                 SocketValue::normal({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Displacement", SocketType::vector)},
      .properties =
          {property("Space", SocketType::string,
                    SocketValue::string("OBJECT")),
           property("NormalLinked", SocketType::boolean,
                    SocketValue::boolean(false))},
      .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::vertex_color,
      .inputs = {},
      .outputs = {output("Color", SocketType::color),
                  output("Alpha", SocketType::floating)},
      .properties = {
          property("Layer Name", SocketType::string,
                   SocketValue::string("")),
          runtime_property("AttributeId", SocketType::unsigned_integer,
                           SocketValue::unsigned_integer(0u))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::attribute,
      .inputs = {},
      .outputs = {output("Color", SocketType::color),
                  output("Vector", SocketType::vector),
                  output("Fac", SocketType::floating),
                  output("Alpha", SocketType::floating)},
      .properties = {
          property("Attribute", SocketType::string,
                   SocketValue::string("")),
          property("Stochastic", SocketType::boolean,
                   SocketValue::boolean(true)),
          runtime_property("AttributeId", SocketType::unsigned_integer,
                           SocketValue::unsigned_integer(0u))},
      .required_features = feature_bit(ShaderFeature::attributes)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::diffuse_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::translucent_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::principled_bsdf,
      .inputs =
          {input("BaseColor", SocketType::color,
                 SocketValue::color({0.8f, 0.8f, 0.8f})),
           input("Metallic", SocketType::floating, SocketValue::floating(0.0f)),
           input("Roughness", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("DiffuseRoughness", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("SubsurfaceWeight", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("SubsurfaceRadius", SocketType::vector,
                 SocketValue::vector({1.0f, 0.2f, 0.1f})),
           input("SubsurfaceScale", SocketType::floating,
                 SocketValue::floating(0.005f)),
           input("SubsurfaceIOR", SocketType::floating,
                 SocketValue::floating(1.4f)),
           input("SubsurfaceAnisotropy", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("TransmissionWeight", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("IOR", SocketType::floating, SocketValue::floating(1.5f)),
           input("SpecularIORLevel", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("SpecularTint", SocketType::color,
                 SocketValue::color({1.0f, 1.0f, 1.0f})),
           input("Anisotropic", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("AnisotropicRotation", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("Tangent", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f})),
           input("Alpha", SocketType::floating, SocketValue::floating(1.0f)),
           input("ThinWall", SocketType::boolean,
                 SocketValue::boolean(false)),
           input("SheenWeight", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("SheenRoughness", SocketType::floating,
                 SocketValue::floating(0.5f)),
           input("SheenTint", SocketType::color,
                 SocketValue::color({1.0f, 1.0f, 1.0f})),
           input("CoatWeight", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("CoatRoughness", SocketType::floating,
                 SocketValue::floating(0.03f)),
           input("CoatIOR", SocketType::floating, SocketValue::floating(1.5f)),
           input("CoatTint", SocketType::color,
                 SocketValue::color({1.0f, 1.0f, 1.0f})),
           input("CoatNormal", SocketType::normal,
                 SocketValue::normal({0.0f, 0.0f, 0.0f})),
           input("EmissionColor", SocketType::color,
                 SocketValue::color({1.0f, 1.0f, 1.0f})),
           input("EmissionStrength", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("ThinFilmThickness", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("ThinFilmIOR", SocketType::floating,
                 SocketValue::floating(1.33f)),
           input("Normal", SocketType::normal,
                 SocketValue::normal({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Closure", SocketType::closure)},
      .properties = {property("Distribution", SocketType::string,
                              SocketValue::string("MULTI_GGX")),
                     property("SubsurfaceMethod", SocketType::string,
                              SocketValue::string("RANDOM_WALK"))},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::emission)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::subsurface_scattering,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Scale", SocketType::floating,
                                  SocketValue::floating(0.005f)),
                            input("Radius", SocketType::vector,
                                  SocketValue::vector({1.0f, 0.2f, 0.1f})),
                            input("IOR", SocketType::floating,
                                  SocketValue::floating(1.4f)),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Anisotropy", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property("Method", SocketType::string,
                                         SocketValue::string("RANDOM_WALK"))},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::subsurface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::glossy_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(0.2f)),
                            input("Anisotropy", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Rotation", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Tangent", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property("Distribution", SocketType::string,
                                         SocketValue::string("GGX"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::metallic_bsdf,
                 .inputs = {
                     input("BaseColor", SocketType::color,
                           SocketValue::color({0.617f, 0.577f, 0.540f})),
                     input("EdgeTint", SocketType::color,
                           SocketValue::color({0.695f, 0.726f, 0.770f})),
                     input("IOR", SocketType::vector,
                           SocketValue::vector({2.757f, 2.513f, 2.231f})),
                     input("Extinction", SocketType::vector,
                           SocketValue::vector({3.867f, 3.404f, 3.009f})),
                     input("Roughness", SocketType::floating,
                           SocketValue::floating(0.5f)),
                     input("Anisotropy", SocketType::floating,
                           SocketValue::floating(0.0f)),
                     input("Rotation", SocketType::floating,
                           SocketValue::floating(0.0f)),
                     input("Tangent", SocketType::vector,
                           SocketValue::vector({0.0f, 0.0f, 0.0f})),
                     input("ThinFilmThickness", SocketType::floating,
                           SocketValue::floating(0.0f)),
                     input("ThinFilmIOR", SocketType::floating,
                           SocketValue::floating(1.33f)),
                     input("Normal", SocketType::normal,
                           SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {
                     property("Distribution", SocketType::string,
                              SocketValue::string("MULTI_GGX")),
                     property("FresnelType", SocketType::string,
                              SocketValue::string("F82"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::sheen_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property(
                     "Distribution", SocketType::string,
                     SocketValue::string("MICROFIBER"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::toon_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Size", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            input("Smooth", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property(
                     "Component", SocketType::string,
                     SocketValue::string("DIFFUSE"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::ray_portal_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Position", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f})),
                            input("Direction", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::hair_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Offset", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("RoughnessU", SocketType::floating,
                                  SocketValue::floating(0.1f)),
                            input("RoughnessV", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Tangent", SocketType::vector,
                                  SocketValue::vector({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property(
                     "Component", SocketType::string,
                     SocketValue::string("REFLECTION"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::glass_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("IOR", SocketType::floating,
                                  SocketValue::floating(1.5f)),
                            input("ThinFilmThickness", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("ThinFilmIOR", SocketType::floating,
                                  SocketValue::floating(1.33f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property("Distribution", SocketType::string,
                                         SocketValue::string("GGX"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::refraction_bsdf,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Roughness", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("IOR", SocketType::floating,
                                  SocketValue::floating(1.5f)),
                            input("Normal", SocketType::normal,
                                  SocketValue::normal({0.0f, 0.0f, 0.0f}))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {property("Distribution", SocketType::string,
                                         SocketValue::string("GGX"))},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::emission,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Strength", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::emission)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::background,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Strength", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface) |
                                      feature_bit(ShaderFeature::emission)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::transparent_bsdf,
      .inputs = {input("Color", SocketType::color,
                       SocketValue::color({1.0f, 1.0f, 1.0f}))},
      .outputs = {output("Closure", SocketType::closure)},
      .properties = {},
      .required_features = feature_bit(ShaderFeature::surface) |
                           feature_bit(ShaderFeature::transparency)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::null_closure,
                 .inputs = {},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::add_closure,
                 .inputs = {required_input("A", SocketType::closure),
                            required_input("B", SocketType::closure)},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::mix_closure,
                 .inputs = {input("Factor", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            required_input("A", SocketType::closure),
                            required_input("B", SocketType::closure)},
                 .outputs = {output("Closure", SocketType::closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::surface)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::volume_absorption,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Density", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::volume_scatter,
      .inputs =
          {input("Color", SocketType::color,
                 SocketValue::color({0.8f, 0.8f, 0.8f})),
           input("Density", SocketType::floating, SocketValue::floating(1.0f)),
           input("Anisotropy", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("IOR", SocketType::floating, SocketValue::floating(1.33f)),
           input("Backscatter", SocketType::floating,
                 SocketValue::floating(0.1f)),
           input("Alpha", SocketType::floating, SocketValue::floating(0.5f)),
           input("Diameter", SocketType::floating,
                 SocketValue::floating(20.0f))},
      .outputs = {output("Volume", SocketType::volume_closure)},
      .properties = {property("Phase", SocketType::string,
                              SocketValue::string("HENYEY_GREENSTEIN"))},
      .required_features = feature_bit(ShaderFeature::volume)}));

  static_cast<void>(registry.register_schema(NodeSchema{
      .type = node_type::volume_coefficients,
      .inputs =
          {input("ScatterCoefficients", SocketType::vector,
                 SocketValue::vector({1.0f, 1.0f, 1.0f})),
           input("AbsorptionCoefficients", SocketType::vector,
                 SocketValue::vector({1.0f, 1.0f, 1.0f})),
           input("Anisotropy", SocketType::floating,
                 SocketValue::floating(0.0f)),
           input("IOR", SocketType::floating, SocketValue::floating(1.33f)),
           input("Backscatter", SocketType::floating,
                 SocketValue::floating(0.1f)),
           input("Alpha", SocketType::floating, SocketValue::floating(0.5f)),
           input("Diameter", SocketType::floating,
                 SocketValue::floating(20.0f)),
           input("EmissionCoefficients", SocketType::vector,
                 SocketValue::vector({0.0f, 0.0f, 0.0f}))},
      .outputs = {output("Volume", SocketType::volume_closure)},
      .properties = {property("Phase", SocketType::string,
                              SocketValue::string("HENYEY_GREENSTEIN"))},
      .required_features = feature_bit(ShaderFeature::volume)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::volume_emission,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.8f, 0.8f, 0.8f})),
                            input("Strength", SocketType::floating,
                                  SocketValue::floating(1.0f))},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume) |
                                      feature_bit(ShaderFeature::emission)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::principled_volume,
                 .inputs = {input("Color", SocketType::color,
                                  SocketValue::color({0.5f, 0.5f, 0.5f})),
                            input("Density", SocketType::floating,
                                  SocketValue::floating(1.0f)),
                            input("Anisotropy", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("AbsorptionColor", SocketType::color,
                                  SocketValue::color({0.0f, 0.0f, 0.0f})),
                            input("EmissionStrength", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("EmissionColor", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("BlackbodyIntensity", SocketType::floating,
                                  SocketValue::floating(0.0f)),
                            input("BlackbodyTint", SocketType::color,
                                  SocketValue::color({1.0f, 1.0f, 1.0f})),
                            input("Temperature", SocketType::floating,
                                  SocketValue::floating(1000.0f))},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume) |
                                      feature_bit(ShaderFeature::emission)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::null_volume,
                 .inputs = {},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::add_volume,
                 .inputs = {required_input("A", SocketType::volume_closure),
                            required_input("B", SocketType::volume_closure)},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume)}));

  static_cast<void>(registry.register_schema(
      NodeSchema{.type = node_type::mix_volume,
                 .inputs = {input("Factor", SocketType::floating,
                                  SocketValue::floating(0.5f)),
                            required_input("A", SocketType::volume_closure),
                            required_input("B", SocketType::volume_closure)},
                 .outputs = {output("Volume", SocketType::volume_closure)},
                 .properties = {},
                 .required_features = feature_bit(ShaderFeature::volume)}));

  return registry;
}

} // namespace psycles::compiler
