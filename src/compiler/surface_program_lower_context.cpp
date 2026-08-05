#include "surface_program_builder.h"

#include <tuple>
#include <utility>

namespace psycles::compiler::detail {

// Lowers graph context and path-state inputs. A true result means the node
// family was recognized, even when input diagnostics prevented an
// instruction from being emitted.
[[nodiscard]] bool
SurfaceProgramBuilder::lower_context_node(const contract::ShaderNode &node) {
  using contract::SocketType;

  if (node.type == node_type::constant_float) {
    if (auto value = lower_value_input(node, "Value")) {
      publish(node.id, "Value", *value);
    }
    return true;
  }
  if (node.type == node_type::constant_color) {
    if (auto value = lower_value_input(node, "Color")) {
      publish(node.id, "Color", *value);
    }
    return true;
  }
  if (node.type == node_type::geometry) {
    publish(
        node.id, "Position",
        append(ValueInstruction{.operation = ValueOperation::surface_position,
                                .source_node = node.id,
                                .result_type = SocketType::point}));
    publish(node.id, "Normal",
            append(ValueInstruction{.operation = ValueOperation::shading_normal,
                                    .source_node = node.id,
                                    .result_type = SocketType::normal}));
    publish(
        node.id, "GeometricNormal",
        append(ValueInstruction{.operation = ValueOperation::geometric_normal,
                                .source_node = node.id,
                                .result_type = SocketType::normal}));
    publish(node.id, "Incoming",
            append(ValueInstruction{.operation = ValueOperation::incoming,
                                    .source_node = node.id,
                                    .result_type = SocketType::vector}));
    publish(node.id, "Tangent",
            append(ValueInstruction{.operation = ValueOperation::tangent,
                                    .source_node = node.id,
                                    .result_type = SocketType::vector}));
    publish(node.id, "Backfacing",
            append(ValueInstruction{.operation = ValueOperation::back_facing,
                                    .source_node = node.id,
                                    .result_type = SocketType::floating}));
    publish(node.id, "Pointiness",
            append(ValueInstruction{.operation = ValueOperation::pointiness,
                                    .source_node = node.id,
                                    .result_type = SocketType::floating}));
    publish(
        node.id, "RandomPerIsland",
        append(ValueInstruction{.operation = ValueOperation::random_per_island,
                                .source_node = node.id,
                                .result_type = SocketType::floating}));
    return true;
  }
  if (node.type == node_type::texture_coordinate) {
    const auto object_use_transform = property_bool(node, "ObjectUseTransform");
    std::vector<float> object_world_to_object;
    if (object_use_transform) {
      const auto transform = property_transform(node, "ObjectWorldToObject");
      object_world_to_object.assign(transform.elements.begin(),
                                    transform.elements.end());
    }
    publish(node.id, "UV",
            append(ValueInstruction{
                .operation = ValueOperation::uv,
                .source_node = node.id,
                .result_type = SocketType::vector,
                .static_u0 = property_bool(node, "UvMapNamed") ? 1u : 0u,
                .static_u1 = property_uint(node, "UvMapId")}));
    publish(node.id, "Normal",
            append(ValueInstruction{.operation = ValueOperation::shading_normal,
                                    .source_node = node.id,
                                    .result_type = SocketType::vector}));
    publish(node.id, "Generated",
            append(ValueInstruction{.operation = ValueOperation::generated,
                                    .source_node = node.id,
                                    .result_type = SocketType::vector}));
    publish(
        node.id, "Object",
        append(ValueInstruction{
            .operation = object_use_transform
                             ? ValueOperation::object_position_with_transform
                             : ValueOperation::object_position,
            .source_node = node.id,
            .result_type = SocketType::vector,
            .static_table = std::move(object_world_to_object)}));
    return true;
  }
  if (node.type == node_type::object_info) {
    publish(
        node.id, "Location",
        append(ValueInstruction{.operation = ValueOperation::object_location,
                                .source_node = node.id,
                                .result_type = SocketType::vector}));
    publish(node.id, "Random",
            append(ValueInstruction{.operation = ValueOperation::object_random,
                                    .source_node = node.id,
                                    .result_type = SocketType::floating}));
    return true;
  }
  if (node.type == node_type::particle_info) {
    publish(node.id, "Index",
            append(ValueInstruction{.operation = ValueOperation::particle_index,
                                    .source_node = node.id,
                                    .result_type = SocketType::floating}));
    publish(
        node.id, "Random",
        append(ValueInstruction{.operation = ValueOperation::particle_random,
                                .source_node = node.id,
                                .result_type = SocketType::floating}));
    return true;
  }
  if (node.type == node_type::hair_info) {
    for (const auto &[output_name, operation, type] :
         {std::tuple{"IsStrand", ValueOperation::curve_is_strand,
                     SocketType::floating},
          std::tuple{"Intercept", ValueOperation::curve_intercept,
                     SocketType::floating},
          std::tuple{"Length", ValueOperation::curve_length,
                     SocketType::floating},
          std::tuple{"Thickness", ValueOperation::curve_thickness,
                     SocketType::floating},
          std::tuple{"TangentNormal", ValueOperation::curve_tangent_normal,
                     SocketType::normal},
          std::tuple{"Random", ValueOperation::curve_random,
                     SocketType::floating}}) {
      publish(node.id, output_name,
              append(ValueInstruction{.operation = operation,
                                      .source_node = node.id,
                                      .result_type = type}));
    }
    return true;
  }
  if (node.type == node_type::light_path) {
    for (const auto &[output_name, operation] :
         {std::pair{"IsCameraRay", ValueOperation::path_is_camera},
          std::pair{"IsShadowRay", ValueOperation::path_is_shadow},
          std::pair{"IsDiffuseRay", ValueOperation::path_is_diffuse},
          std::pair{"IsGlossyRay", ValueOperation::path_is_glossy},
          std::pair{"IsSingularRay", ValueOperation::path_is_singular},
          std::pair{"IsReflectionRay", ValueOperation::path_is_reflection},
          std::pair{"IsTransmissionRay", ValueOperation::path_is_transmission},
          std::pair{"IsVolumeScatterRay",
                    ValueOperation::path_is_volume_scatter},
          std::pair{"RayLength", ValueOperation::path_ray_length},
          std::pair{"RayDepth", ValueOperation::path_ray_depth},
          std::pair{"DiffuseDepth", ValueOperation::path_diffuse_depth},
          std::pair{"GlossyDepth", ValueOperation::path_glossy_depth},
          std::pair{"TransparentDepth", ValueOperation::path_transparent_depth},
          std::pair{"TransmissionDepth",
                    ValueOperation::path_transmission_depth}}) {
      publish(node.id, output_name,
              append(ValueInstruction{.operation = operation,
                                      .source_node = node.id,
                                      .result_type = SocketType::floating}));
    }
    return true;
  }
  if (node.type == node_type::layer_weight) {
    auto blend = lower_value_input(node, "Blend");
    auto normal = lower_value_input(node, "Normal");
    if (blend && normal) {
      publish(node.id, "Fresnel",
              append(ValueInstruction{
                  .operation = ValueOperation::layer_weight_fresnel,
                  .source_node = node.id,
                  .result_type = SocketType::floating,
                  .a = *blend,
                  .b = *normal,
                  .static_u0 = property_bool(node, "NormalLinked") ? 1u : 0u}));
      publish(node.id, "Facing",
              append(ValueInstruction{
                  .operation = ValueOperation::layer_weight_facing,
                  .source_node = node.id,
                  .result_type = SocketType::floating,
                  .a = *blend,
                  .b = *normal,
                  .static_u0 = property_bool(node, "NormalLinked") ? 1u : 0u}));
    }
    return true;
  }
  if (node.type == node_type::fresnel) {
    auto ior = lower_value_input(node, "IOR");
    auto normal = lower_value_input(node, "Normal");
    if (ior && normal) {
      publish(node.id, "Factor",
              append(ValueInstruction{.operation = ValueOperation::fresnel,
                                      .source_node = node.id,
                                      .result_type = SocketType::floating,
                                      .a = *ior,
                                      .b = *normal,
                                      .static_u0 =
                                          property_bool(node, "NormalLinked")
                                              ? 1u
                                              : 0u}));
    }
    return true;
  }
  return false;
}

} // namespace psycles::compiler::detail
