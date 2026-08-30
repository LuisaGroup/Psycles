#include "compact_surface_program_test_support.h"

#include <psycles/compiler/core_nodes.h>

#include "path_tracer_internal.h"
#include "path_tracer_surface_value_family.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

static_assert(
    surface_value_family_has_direct_evaluator(SurfaceSvmValueOpcode::geometry));
static_assert(surface_value_family_has_direct_evaluator(
    SurfaceSvmValueOpcode::geometry_derivative));
static_assert(surface_value_family_has_direct_evaluator(
    SurfaceSvmValueOpcode::tex_coord));
static_assert(surface_value_family_has_direct_evaluator(
    SurfaceSvmValueOpcode::tex_coord_derivative));
static_assert(surface_value_family_has_direct_evaluator(
    SurfaceSvmValueOpcode::bump_support));
static_assert(surface_value_family_has_direct_evaluator(
    SurfaceSvmValueOpcode::displacement));

[[nodiscard]] bool is_transform_operation(ValueOperation operation) noexcept {
  return operation == ValueOperation::object_position_with_transform ||
         operation == ValueOperation::sampled_object_position_with_transform;
}

} // namespace

ShaderGraph make_direct_state_bump_graph() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Direct state Geometry");
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Direct state coordinates");
  const auto transformed_coordinates = graph.add_node(
      node_type::texture_coordinate, "Direct transformed coordinates");
  const auto position_to_vector = graph.add_node(
      node_type::point_to_vector, "Direct state position vector");
  const auto geometric_normal_to_vector = graph.add_node(
      node_type::normal_to_vector, "Direct state geometric normal vector");
  const auto add_position_normal = graph.add_node(
      node_type::vector_math, "Direct state position plus normal");
  const auto add_incoming =
      graph.add_node(node_type::vector_math, "Direct state plus incoming");
  const auto add_uv =
      graph.add_node(node_type::vector_math, "Direct state plus UV");
  const auto add_generated =
      graph.add_node(node_type::vector_math, "Direct state plus generated");
  const auto add_object =
      graph.add_node(node_type::vector_math, "Direct state plus object");
  const auto add_transformed_object = graph.add_node(
      node_type::vector_math, "Direct state plus transformed object");
  const auto height =
      graph.add_node(node_type::vector_to_scalar, "Direct state bump height");
  const auto displacement = graph.add_node(
      node_type::displacement, "Direct state object displacement");
  const auto displacement_height = graph.add_node(
      node_type::vector_to_scalar,
      "Direct state object displacement height");
  const auto bump =
      graph.add_node(node_type::bump, "Direct state derivative Bump");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Direct state diffuse");

  Mat4f world_to_object;
  world_to_object.elements = {1.7f,  0.0f,   0.0f,  0.0f, 0.0f, 0.8f,
                              0.0f,  0.0f,   0.0f,  0.0f, 1.2f, 0.0f,
                              0.13f, -0.27f, 0.41f, 1.0f};
  auto configured =
      graph.set_property(transformed_coordinates, "ObjectUseTransform",
                         SocketValue::boolean(true)) &&
      graph.set_property(transformed_coordinates, "ObjectWorldToObject",
                         SocketValue::transform(world_to_object)) &&
      graph.connect({.node = geometry, .socket = "Position"},
                    position_to_vector, "Point") &&
      graph.connect({.node = geometry, .socket = "GeometricNormal"},
                    geometric_normal_to_vector, "Normal") &&
      graph.connect({.node = position_to_vector, .socket = "Vector"},
                    add_position_normal, "A") &&
      graph.connect({.node = geometric_normal_to_vector, .socket = "Vector"},
                    add_position_normal, "B") &&
      graph.connect({.node = add_position_normal, .socket = "Vector"},
                    add_incoming, "A") &&
      graph.connect({.node = geometry, .socket = "Incoming"}, add_incoming,
                    "B") &&
      graph.connect({.node = add_incoming, .socket = "Vector"}, add_uv, "A") &&
      graph.connect({.node = coordinates, .socket = "UV"}, add_uv, "B") &&
      graph.connect({.node = add_uv, .socket = "Vector"}, add_generated, "A") &&
      graph.connect({.node = coordinates, .socket = "Generated"}, add_generated,
                    "B") &&
      graph.connect({.node = add_generated, .socket = "Vector"}, add_object,
                    "A") &&
      graph.connect({.node = coordinates, .socket = "Object"}, add_object,
                    "B") &&
      graph.connect({.node = add_object, .socket = "Vector"},
                    add_transformed_object, "A") &&
      graph.connect({.node = transformed_coordinates, .socket = "Object"},
                    add_transformed_object, "B") &&
      graph.connect({.node = add_transformed_object, .socket = "Vector"},
                    height, "Vector") &&
      graph.connect({.node = height, .socket = "Value"}, displacement,
                    "Height") &&
      graph.connect({.node = geometry, .socket = "Normal"}, displacement,
                    "Normal") &&
      graph.connect({.node = displacement, .socket = "Displacement"},
                    displacement_height, "Vector") &&
      graph.connect({.node = displacement_height, .socket = "Value"}, bump,
                    "Height") &&
      graph.connect({.node = geometry, .socket = "Normal"}, bump, "Normal") &&
      graph.connect({.node = bump, .socket = "Normal"}, diffuse, "Normal") &&
      graph.set_input(bump, "Strength", SocketValue::floating(0.67f)) &&
      graph.set_input(bump, "Distance", SocketValue::floating(0.23f)) &&
      graph.set_input(bump, "FilterWidth", SocketValue::floating(0.19f)) &&
      graph.set_input(displacement, "Midlevel",
                      SocketValue::floating(-0.21f)) &&
      graph.set_input(displacement, "Scale",
                      SocketValue::floating(0.43f)) &&
      graph.set_property(displacement, "Space",
                         SocketValue::string("OBJECT")) &&
      graph.set_property(displacement, "NormalLinked",
                         SocketValue::boolean(true)) &&
      graph.set_property(bump, "Invert", SocketValue::boolean(true)) &&
      graph.set_property(bump, "NormalLinked", SocketValue::boolean(true)) &&
      graph.set_property(bump, "UseObjectSpace", SocketValue::boolean(false)) &&
      graph.set_input(diffuse, "Color",
                      SocketValue::color({0.29f, 0.53f, 0.71f})) &&
      graph.set_input(diffuse, "Roughness", SocketValue::floating(0.37f));
  for (const auto node : {add_position_normal, add_incoming, add_uv,
                          add_generated, add_object, add_transformed_object}) {
    configured = configured && graph.set_property(node, "Operation",
                                                  SocketValue::string("ADD"));
  }
  if (!configured) {
    throw std::runtime_error{"failed to configure direct state/Bump SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  return graph;
}

std::string
validate_direct_state_surface_runtime(const SurfaceValueRuntime &runtime) {
  constexpr std::array operations{
      ValueOperation::surface_position,
      ValueOperation::shading_normal,
      ValueOperation::geometric_normal,
      ValueOperation::incoming,
      ValueOperation::uv,
      ValueOperation::generated,
      ValueOperation::object_position,
      ValueOperation::object_position_with_transform,
      ValueOperation::bump_offset_zero,
      ValueOperation::bump_filter_width,
      ValueOperation::bump_samples,
      ValueOperation::displacement,
      ValueOperation::sampled_surface_position,
      ValueOperation::sampled_uv,
      ValueOperation::sampled_generated,
      ValueOperation::sampled_object_position,
      ValueOperation::sampled_object_position_with_transform};
  for (const auto operation : operations) {
    if (std::none_of(runtime.value_variants.begin(),
                     runtime.value_variants.end(),
                     [operation](const auto &variant) noexcept {
                       return variant.instruction.operation == operation;
                     })) {
      return "direct state/Bump fixture lost operation " +
             std::to_string(static_cast<std::uint32_t>(operation));
    }
  }

  const auto count_variant = [&](ValueOperation operation) noexcept {
    return std::count_if(runtime.value_variants.begin(),
                         runtime.value_variants.end(),
                         [operation](const auto &variant) noexcept {
                           return variant.instruction.operation == operation;
                         });
  };
  const auto ordinary_variants =
      count_variant(ValueOperation::object_position_with_transform);
  const auto derivative_variants =
      count_variant(ValueOperation::sampled_object_position_with_transform);
  const auto displacement_variants =
      count_variant(ValueOperation::displacement);
  const auto displacement_immediate_exact = std::all_of(
      runtime.value_variants.begin(), runtime.value_variants.end(),
      [](const auto &variant) noexcept {
        return variant.instruction.operation != ValueOperation::displacement ||
               (variant.svm_immediates.size() == 1u &&
                variant.svm_immediates.front() ==
                    static_cast<std::uint16_t>(
                        displacement_object_space |
                        displacement_normal_linked));
      });

  const auto &scene = runtime.svm_scene;
  std::vector<bool> transform_metadata_seen(scene.value_metadata.size(), false);
  auto transform_record_count = std::size_t{};
  auto exact_relation = true;
  for (const auto &record : scene.instructions) {
    if (surface_svm_bytecode_kind(record) != SurfaceSvmBytecodeKind::value) {
      continue;
    }
    const auto value = surface_svm_value_instruction(record);
    if (!is_transform_operation(surface_value_operation(value))) {
      continue;
    }
    ++transform_record_count;
    const auto metadata_in_domain =
        value.metadata_index < scene.value_metadata.size();
    exact_relation &= metadata_in_domain;
    if (!metadata_in_domain) {
      continue;
    }
    // Injectivity is the executable ownership relation: two distinct
    // transform records may not alias a mutable metadata identity even
    // when their 16 authored floats happen to be equal.
    exact_relation &= !transform_metadata_seen[value.metadata_index];
    transform_metadata_seen[value.metadata_index] = true;
    exact_relation &=
        scene.value_metadata[value.metadata_index].static_table_count == 16u;
  }
  const auto owned_payload_count = std::count(
      transform_metadata_seen.begin(), transform_metadata_seen.end(), true);
  exact_relation &= transform_record_count == owned_payload_count;
  if (ordinary_variants != 1u || derivative_variants != 1u ||
      displacement_variants != 1u || !displacement_immediate_exact ||
      !exact_relation) {
    std::ostringstream message;
    message << "compact runtime violated the transform evaluator/metadata "
               "injection (variants="
            << ordinary_variants
            << ", derivative variants=" << derivative_variants
            << ", displacement variants=" << displacement_variants
            << ", records=" << transform_record_count
            << ", owned payloads=" << owned_payload_count << ')';
    return message.str();
  }
  return {};
}

} // namespace psycles::test_support
