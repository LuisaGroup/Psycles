/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_graph.h"
#include "cycles_svm_constant_fold.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] bool check_node_inputs_has_links(
    const GraphNode *node) noexcept {
  return std::any_of(node->inputs.begin(), node->inputs.end(),
                     [](const auto &input) { return input.link != nullptr; });
}

[[nodiscard]] bool check_node_inputs_traversed(
    const GraphNode *node, const GraphNodeSet &done) noexcept {
  return std::all_of(node->inputs.begin(), node->inputs.end(),
                     [&](const auto &input) {
                       return input.link == nullptr ||
                              done.contains(input.link->parent);
                     });
}

enum class ConstantFoldStage : std::uint8_t {
  blender_inline,
  cycles_graph,
};

void run_constant_fold_stage(CyclesGraph &graph, ConstantFoldStage stage) {
  GraphNodeSet done;
  GraphNodeSet scheduled;
  std::queue<GraphNode *> traverse_queue;

  for (const auto &node : graph.nodes()) {
    if (!check_node_inputs_has_links(node.get())) {
      traverse_queue.push(node.get());
      scheduled.insert(node.get());
    }
  }

  while (!traverse_queue.empty()) {
    auto *node = traverse_queue.front();
    traverse_queue.pop();
    done.insert(node);
    for (auto &output_socket : node->outputs) {
      if (output_socket.links.empty()) {
        continue;
      }
      for (auto *input : output_socket.links) {
        if (!scheduled.contains(input->parent) &&
            check_node_inputs_traversed(input->parent, done)) {
          traverse_queue.push(input->parent);
          scheduled.insert(input->parent);
        }
      }
      const ConstantFolder folder{&graph, node, &output_socket};
      if (stage == ConstantFoldStage::blender_inline) {
        node->inline_blender_constant_fold(folder);
      } else {
        node->constant_fold(folder);
      }
    }
  }
}

[[nodiscard]] bool is_surface_closure(std::string_view type) noexcept {
  return type == node_type::diffuse_bsdf ||
         type == node_type::translucent_bsdf ||
         type == node_type::principled_bsdf ||
         type == node_type::subsurface_scattering ||
         type == node_type::glossy_bsdf ||
         type == node_type::metallic_bsdf ||
         type == node_type::sheen_bsdf || type == node_type::hair_bsdf ||
         type == node_type::glass_bsdf ||
         type == node_type::refraction_bsdf || type == node_type::emission ||
         type == node_type::background ||
         type == node_type::transparent_bsdf;
}

[[nodiscard]] bool is_volume_closure(std::string_view type) noexcept {
  return type == node_type::volume_absorption ||
         type == node_type::volume_scatter ||
         type == node_type::volume_coefficients ||
         type == node_type::volume_emission ||
         type == node_type::principled_volume;
}

[[nodiscard]] bool is_combine_closure(std::string_view type) noexcept {
  return type == node_type::add_closure || type == node_type::mix_closure ||
         type == node_type::add_volume || type == node_type::mix_volume;
}

[[nodiscard]] bool is_null_closure(std::string_view type) noexcept {
  return type == node_type::null_closure || type == node_type::null_volume;
}

[[nodiscard]] std::string_view
projected_node_type(std::string_view type) noexcept {
  // Psycles' canonical Multiply Color helper is exactly Cycles Mix Color in
  // MULTIPLY mode with clamped factor and unclamped result. Normalize it at
  // the graph boundary so the SVM graph, constant folding, and emitted node
  // stream remain Cycles' model rather than introducing another node family.
  if (type == node_type::multiply_color) {
    return node_type::mix_color;
  }
  if (type == node_type::add_float || type == node_type::subtract_float ||
      type == node_type::multiply_float || type == node_type::divide_float ||
      type == node_type::minimum_float || type == node_type::maximum_float ||
      type == node_type::power_float) {
    return node_type::math;
  }
  return type;
}

[[nodiscard]] std::string_view
projected_binary_math_operation(std::string_view type) noexcept {
  return type == node_type::add_float        ? "ADD"
         : type == node_type::subtract_float ? "SUBTRACT"
         : type == node_type::multiply_float ? "MULTIPLY"
         : type == node_type::divide_float   ? "DIVIDE"
         : type == node_type::minimum_float  ? "MINIMUM"
         : type == node_type::maximum_float  ? "MAXIMUM"
         : type == node_type::power_float    ? "POWER"
                                             : std::string_view{};
}

[[nodiscard]] std::string_view projected_input_name(
    std::string_view node, std::string_view input) noexcept {
  if (node == node_type::math) {
    return input == "A"   ? "Value1"
           : input == "B" ? "Value2"
           : input == "C" ? "Value3"
                          : input;
  }
  if (node == node_type::vector_math) {
    return input == "A"   ? "Vector1"
           : input == "B" ? "Vector2"
           : input == "C" ? "Vector3"
                          : input;
  }
  if (node == node_type::mix_closure || node == node_type::mix_volume) {
    return input == "Factor" ? "Fac"
           : input == "A"    ? "Closure1"
           : input == "B"    ? "Closure2"
                             : input;
  }
  if (node == node_type::add_closure || node == node_type::add_volume) {
    return input == "A"   ? "Closure1"
           : input == "B" ? "Closure2"
                          : input;
  }
  if ((node == node_type::invert_color ||
       node == node_type::hue_saturation) &&
      input == "Factor") {
    return "Fac";
  }
  if (node == node_type::legacy_mix_color) {
    return input == "Factor" ? "Fac"
           : input == "A"    ? "Color1"
           : input == "B"    ? "Color2"
                             : input;
  }
  if (node == node_type::combine_color) {
    return input == "R"   ? "Red"
           : input == "G" ? "Green"
           : input == "B" ? "Blue"
                          : input;
  }
  if (node == node_type::bump && input == "FilterWidth") {
    return "Filter Width";
  }
  return input;
}

[[nodiscard]] std::string_view projected_output_name(
    std::string_view node, std::string_view output) noexcept {
  if (node == node_type::mix_float && output == "Value") {
    return "Result";
  }
  if ((node == node_type::mix_vector ||
       node == node_type::mix_vector_nonuniform) &&
      output == "Vector") {
    return "Result";
  }
  if (node == node_type::mix_color && output == "Color") {
    return "Result";
  }
  if ((node == node_type::diffuse_bsdf ||
       node == node_type::translucent_bsdf ||
       node == node_type::transparent_bsdf) &&
      output == "Closure") {
    return "BSDF";
  }
  if (node == node_type::emission && output == "Closure") {
    return "Emission";
  }
  if (node == node_type::geometry) {
    return output == "GeometricNormal"
               ? "True Normal"
           : output == "RandomPerIsland" ? "Random Per Island"
                                          : output;
  }
  if (node == node_type::object_info) {
    return output == "ObjectIndex"     ? "Object Index"
           : output == "MaterialIndex" ? "Material Index"
                                        : output;
  }
  if (node == node_type::particle_info &&
      output == "AngularVelocity") {
    return "Angular Velocity";
  }
  if (node == node_type::hair_info) {
    return output == "IsStrand"       ? "Is Strand"
           : output == "TangentNormal" ? "Tangent Normal"
                                        : output;
  }
  if (node == node_type::light_path) {
    static constexpr auto outputs = std::array{
        std::pair{std::string_view{"IsCameraRay"},
                  std::string_view{"Is Camera Ray"}},
        std::pair{std::string_view{"IsShadowRay"},
                  std::string_view{"Is Shadow Ray"}},
        std::pair{std::string_view{"IsDiffuseRay"},
                  std::string_view{"Is Diffuse Ray"}},
        std::pair{std::string_view{"IsGlossyRay"},
                  std::string_view{"Is Glossy Ray"}},
        std::pair{std::string_view{"IsSingularRay"},
                  std::string_view{"Is Singular Ray"}},
        std::pair{std::string_view{"IsReflectionRay"},
                  std::string_view{"Is Reflection Ray"}},
        std::pair{std::string_view{"IsTransmissionRay"},
                  std::string_view{"Is Transmission Ray"}},
        std::pair{std::string_view{"IsVolumeScatterRay"},
                  std::string_view{"Is Volume Scatter Ray"}},
        std::pair{std::string_view{"RayLength"},
                  std::string_view{"Ray Length"}},
        std::pair{std::string_view{"RayDepth"},
                  std::string_view{"Ray Depth"}},
        std::pair{std::string_view{"DiffuseDepth"},
                  std::string_view{"Diffuse Depth"}},
        std::pair{std::string_view{"GlossyDepth"},
                  std::string_view{"Glossy Depth"}},
        std::pair{std::string_view{"TransparentDepth"},
                  std::string_view{"Transparent Depth"}},
        std::pair{std::string_view{"TransmissionDepth"},
                  std::string_view{"Transmission Depth"}},
        std::pair{std::string_view{"PortalDepth"},
                  std::string_view{"Portal Depth"}}};
    if (const auto mapped = std::find_if(
            outputs.begin(), outputs.end(),
            [&](const auto &entry) noexcept { return entry.first == output; });
        mapped != outputs.end()) {
      return mapped->second;
    }
  }
  if (node == node_type::separate_color) {
    return output == "R"   ? "Red"
           : output == "G" ? "Green"
           : output == "B" ? "Blue"
                           : output;
  }
  return output;
}

[[nodiscard]] bool is_float3_socket(GraphSocketType type) noexcept {
  return type == GraphSocketType::color || type == GraphSocketType::vector ||
         type == GraphSocketType::normal || type == GraphSocketType::point;
}

[[nodiscard]] std::string_view float3_socket_name(
    GraphSocketType type) noexcept {
  switch (type) {
    case GraphSocketType::color:
      return "color";
    case GraphSocketType::vector:
      return "vector";
    case GraphSocketType::normal:
      return "normal";
    case GraphSocketType::point:
      return "point";
    case GraphSocketType::floating:
    case GraphSocketType::integer:
    case GraphSocketType::closure:
      return {};
  }
  return {};
}

[[nodiscard]] contract::SocketValue zero_float3_value(
    GraphSocketType type) {
  switch (type) {
    case GraphSocketType::color:
      return contract::SocketValue::color({0.0f, 0.0f, 0.0f});
    case GraphSocketType::vector:
      return contract::SocketValue::vector({0.0f, 0.0f, 0.0f});
    case GraphSocketType::normal:
      return contract::SocketValue::normal({0.0f, 0.0f, 0.0f});
    case GraphSocketType::point:
      return contract::SocketValue::point({0.0f, 0.0f, 0.0f});
    case GraphSocketType::floating:
    case GraphSocketType::integer:
    case GraphSocketType::closure:
      break;
  }
  std::abort();
}

// The contract graph uses a canonical VECTOR for Blender's generic Vector
// sockets. Restore the precise Cycles ShaderInput declaration here, before
// ShaderGraph::connect applies Cycles' automatic conversion rules.
[[nodiscard]] GraphSocketType projected_input_type(
    std::string_view node, std::string_view input,
    GraphSocketType contract_type) noexcept {
  if (input == "Vector" &&
      (node == node_type::ies_light || node == node_type::wave_texture ||
       node == node_type::magic_texture ||
       node == node_type::checker_texture ||
       node == node_type::brick_texture)) {
    return GraphSocketType::point;
  }
  return contract_type;
}

[[nodiscard]] std::optional<contract::SocketValue> projected_input_value(
    std::optional<contract::SocketValue> value, GraphSocketType type) {
  if (!value || value->type == contract::SocketType::point ||
      type != GraphSocketType::point) {
    return value;
  }
  if (const auto *vector = std::get_if<Vec3f>(&value->value)) {
    return contract::SocketValue::point(*vector);
  }
  return value;
}

[[nodiscard]] std::uint16_t socket_flags(std::string_view node,
                                         std::string_view input) noexcept {
  // Direct projection of the SocketType::LINK_* flags declared by the
  // corresponding Cycles 5.2.1 NODE_DEFINE blocks in shader_nodes.cpp.
  if (input == "Normal" &&
      (node == node_type::layer_weight || node == node_type::fresnel ||
       node == node_type::ambient_occlusion || node == node_type::bump ||
       node == node_type::displacement || is_surface_closure(node))) {
    auto flags = static_cast<std::uint16_t>(graph_socket_link_normal);
    // Fresnel and Layer Weight are the only Cycles 5.2 shader nodes whose
    // Normal sockets combine LINK_NORMAL with OSL_INTERNAL. In SVM mode the
    // latter suppresses ShaderGraph::default_inputs' Geometry.Normal edge;
    // their handlers use sd->N when the explicit link remains absent.
    if (node == node_type::layer_weight || node == node_type::fresnel) {
      flags |= graph_socket_osl_internal;
    }
    return flags;
  }
  if (input == "CoatNormal" && node == node_type::principled_bsdf) {
    return graph_socket_link_normal;
  }
  if (input == "Tangent" &&
      (node == node_type::principled_bsdf || node == node_type::glossy_bsdf ||
       node == node_type::metallic_bsdf || node == node_type::hair_bsdf)) {
    return graph_socket_link_tangent;
  }
  if (input == "Vector") {
    if (node == node_type::ies_light) {
      return graph_socket_link_texture_incoming;
    }
    if (node == node_type::image_texture) {
      return graph_socket_link_texture_uv;
    }
    if (node == node_type::environment_texture) {
      return graph_socket_link_position;
    }
    if (node == node_type::gradient_texture || node == node_type::noise_texture ||
        node == node_type::gabor_texture ||
        node == node_type::voronoi_texture || node == node_type::wave_texture ||
        node == node_type::magic_texture || node == node_type::checker_texture ||
        node == node_type::brick_texture) {
      return graph_socket_link_texture_generated;
    }
  }
  return graph_socket_none;
}

[[nodiscard]] GraphNodeSpecialType special_type(
    std::string_view type) noexcept {
  if (type == node_type::geometry) {
    return GraphNodeSpecialType::geometry;
  }
  if (type == node_type::bump) {
    return GraphNodeSpecialType::bump;
  }
  if (is_combine_closure(type)) {
    return GraphNodeSpecialType::combine_closure;
  }
  if (is_surface_closure(type) || is_volume_closure(type)) {
    return GraphNodeSpecialType::closure;
  }
  return GraphNodeSpecialType::none;
}

[[nodiscard]] std::optional<float> float_value(
    const GraphInput *input) noexcept {
  if (input == nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<GraphInput> geometry_inputs() { return {}; }

[[nodiscard]] std::vector<GraphOutput> geometry_outputs() {
  return {
      {.name = "Position", .type = GraphSocketType::point, .links = {}},
      {.name = "Normal", .type = GraphSocketType::normal, .links = {}},
      {.name = "Tangent", .type = GraphSocketType::normal, .links = {}},
      {.name = "True Normal", .type = GraphSocketType::normal, .links = {}},
      {.name = "Incoming", .type = GraphSocketType::vector, .links = {}},
      {.name = "Parametric", .type = GraphSocketType::point, .links = {}},
      {.name = "Backfacing", .type = GraphSocketType::floating, .links = {}},
      {.name = "Pointiness", .type = GraphSocketType::floating, .links = {}},
      {.name = "Random Per Island",
       .type = GraphSocketType::floating,
       .links = {}},
  };
}

[[nodiscard]] std::vector<GraphInput> incoming_transform_inputs() {
  return {{.name = "Vector",
           .type = GraphSocketType::vector,
           .value = contract::SocketValue::vector({0.0f, 0.0f, 0.0f})}};
}

[[nodiscard]] std::vector<GraphOutput> incoming_transform_outputs() {
  return {{.name = "Vector",
           .type = GraphSocketType::vector,
           .links = {}}};
}

[[nodiscard]] auto incoming_transform_properties() {
  std::map<std::string, contract::SocketValue, std::less<>> properties;
  properties.emplace("Type", contract::SocketValue::string("NORMAL"));
  properties.emplace("Convert From", contract::SocketValue::string("WORLD"));
  properties.emplace("Convert To", contract::SocketValue::string("OBJECT"));
  return properties;
}

[[nodiscard]] std::vector<GraphInput> texture_coordinate_inputs() {
  return {};
}

[[nodiscard]] std::vector<GraphOutput> texture_coordinate_outputs() {
  return {
      {.name = "Generated", .type = GraphSocketType::point, .links = {}},
      {.name = "Normal", .type = GraphSocketType::normal, .links = {}},
      {.name = "UV", .type = GraphSocketType::point, .links = {}},
      {.name = "Object", .type = GraphSocketType::point, .links = {}},
      {.name = "Camera", .type = GraphSocketType::point, .links = {}},
      {.name = "Window", .type = GraphSocketType::point, .links = {}},
      {.name = "Reflection", .type = GraphSocketType::normal, .links = {}},
  };
}

[[nodiscard]] auto texture_coordinate_properties() {
  std::map<std::string, contract::SocketValue, std::less<>> properties;
  properties.emplace("FromDupli", contract::SocketValue::boolean(false));
  properties.emplace("UseTransform", contract::SocketValue::boolean(false));
  properties.emplace("ObjectTransform",
                     contract::SocketValue::transform(Mat4f{}));
  return properties;
}

[[nodiscard]] std::vector<GraphInput> mix_weight_inputs() {
  return {
      {.name = "Weight",
       .type = GraphSocketType::floating,
       .value = contract::SocketValue::floating(1.0f)},
      {.name = "Fac",
       .type = GraphSocketType::floating,
       .value = contract::SocketValue::floating(1.0f)},
  };
}

[[nodiscard]] std::vector<GraphOutput> mix_weight_outputs() {
  return {
      {.name = "Weight1", .type = GraphSocketType::floating, .links = {}},
      {.name = "Weight2", .type = GraphSocketType::floating, .links = {}},
  };
}

[[nodiscard]] std::vector<GraphInput> multiply_inputs() {
  return {
      {.name = "Value1",
       .type = GraphSocketType::floating,
       .value = contract::SocketValue::floating(0.5f)},
      {.name = "Value2",
       .type = GraphSocketType::floating,
       .value = contract::SocketValue::floating(0.5f)},
      {.name = "Value3",
       .type = GraphSocketType::floating,
       .value = contract::SocketValue::floating(0.0f)},
  };
}

[[nodiscard]] std::vector<GraphOutput> multiply_outputs() {
  return {{.name = "Value", .type = GraphSocketType::floating, .links = {}}};
}

[[nodiscard]] std::vector<GraphInput> output_inputs() {
  return {
      {.name = "Surface",
       .type = GraphSocketType::closure,
       .value = std::nullopt},
      {.name = "Volume",
       .type = GraphSocketType::closure,
       .value = std::nullopt},
      {.name = "Displacement",
       .type = GraphSocketType::vector,
       .value = contract::SocketValue::vector({0.0f, 0.0f, 0.0f})},
      {.name = "Normal",
       .type = GraphSocketType::normal,
       .value = contract::SocketValue::normal({0.0f, 0.0f, 0.0f})},
  };
}

} // namespace

GraphInput *GraphNode::input(std::string_view name) noexcept {
  const auto iter = std::find_if(inputs.begin(), inputs.end(), [&](auto &item) {
    return item.name == name;
  });
  return iter == inputs.end() ? nullptr : &*iter;
}

const GraphInput *GraphNode::input(std::string_view name) const noexcept {
  const auto iter =
      std::find_if(inputs.begin(), inputs.end(), [&](const auto &item) {
        return item.name == name;
      });
  return iter == inputs.end() ? nullptr : &*iter;
}

GraphOutput *GraphNode::output(std::string_view name) noexcept {
  const auto iter =
      std::find_if(outputs.begin(), outputs.end(), [&](auto &item) {
        return item.name == name;
      });
  return iter == outputs.end() ? nullptr : &*iter;
}

const GraphOutput *GraphNode::output(std::string_view name) const noexcept {
  const auto iter =
      std::find_if(outputs.begin(), outputs.end(), [&](const auto &item) {
        return item.name == name;
      });
  return iter == outputs.end() ? nullptr : &*iter;
}

std::optional<GraphSocketType>
graph_socket_type(contract::SocketType type) noexcept {
  switch (type) {
    case contract::SocketType::boolean:
    case contract::SocketType::integer:
    case contract::SocketType::unsigned_integer:
      return GraphSocketType::integer;
    case contract::SocketType::floating:
      return GraphSocketType::floating;
    case contract::SocketType::float3:
    case contract::SocketType::vector:
      return GraphSocketType::vector;
    case contract::SocketType::color:
    case contract::SocketType::spectrum:
      return GraphSocketType::color;
    case contract::SocketType::point:
      return GraphSocketType::point;
    case contract::SocketType::normal:
      return GraphSocketType::normal;
    case contract::SocketType::closure:
    case contract::SocketType::volume_closure:
      return GraphSocketType::closure;
    case contract::SocketType::float2:
    case contract::SocketType::transform:
    case contract::SocketType::string:
      return std::nullopt;
  }
  return std::nullopt;
}

GraphOutput *CyclesGraph::root(GraphDomain domain) const noexcept {
  const auto *node = output_node();
  if (node == nullptr) {
    return nullptr;
  }
  const auto name =
      domain == GraphDomain::surface        ? "Surface"
      : domain == GraphDomain::volume       ? "Volume"
      : domain == GraphDomain::displacement ? "Displacement"
      : domain == GraphDomain::bump          ? "Normal"
                                             : nullptr;
  if (name == nullptr) {
    return nullptr;
  }
  const auto *input = node->input(name);
  return input != nullptr ? input->link : nullptr;
}

GraphNode *CyclesGraph::add_node(
    std::string type, std::string label, std::vector<GraphInput> inputs,
    std::vector<GraphOutput> outputs, GraphNodeSpecialType node_special_type,
    std::map<std::string, contract::SocketValue, std::less<>> properties) {
  auto node = make_graph_node(type);
  node->id = _next_node_id++;
  node->type = std::move(type);
  node->label = std::move(label);
  node->inputs = std::move(inputs);
  node->outputs = std::move(outputs);
  node->properties = std::move(properties);
  node->special_type = node_special_type;
  auto *result = node.get();
  for (auto &input : result->inputs) {
    input.parent = result;
  }
  for (auto &output : result->outputs) {
    output.parent = result;
  }
  _nodes.emplace_back(std::move(node));
  return result;
}

bool CyclesGraph::connect(GraphOutput *output, GraphInput *input) noexcept {
  if (output == nullptr || input == nullptr || output->type != input->type ||
      input->link != nullptr) {
    return false;
  }
  input->link = output;
  output->links.emplace_back(input);
  return true;
}

bool CyclesGraph::connect_with_autoconvert(GraphOutput *output,
                                           GraphInput *input) {
  if (output == nullptr || input == nullptr || input->link != nullptr) {
    return false;
  }
  if (output->type == input->type) {
    return connect(output, input);
  }
  // ShaderGraph::connect inserts a ConvertNode for every non-closure type
  // mismatch. Only use that source-level operation where projection has
  // restored exact Cycles socket types; canonical contract sockets elsewhere
  // must first be migrated explicitly instead of changing them by accident.
  // The currently reachable mismatch is the float3 family, whose Cycles SVM
  // conversion is a pure stack alias and emits no NODE_CONVERT.
  if (!is_float3_socket(output->type) || !is_float3_socket(input->type)) {
    return false;
  }
  const auto from_name = float3_socket_name(output->type);
  const auto to_name = float3_socket_name(input->type);
  auto *convert = add_node(
      cycles_synthetic_float3_autoconvert,
      "Convert " + std::string{from_name} + " to " + std::string{to_name},
      {{.name = "value_" + std::string{from_name},
        .type = output->type,
        .value = zero_float3_value(output->type)}},
      {{.name = "value_" + std::string{to_name},
        .type = input->type,
        .links = {}}},
      GraphNodeSpecialType::autoconvert);
  return connect(output, &convert->inputs.front()) &&
         connect(&convert->outputs.front(), input);
}

void CyclesGraph::compose_float3_autoconverts() {
  // The contract boundary canonicalizes Blender float3-family links through
  // VECTOR. Once a precise Cycles target type is restored, a path
  //
  //   A -> VECTOR -> B
  //
  // consists only of identity maps on the three stored components. Its normal
  // form is A -> B (or no Convert when A == B). Restrict composition to the
  // AUTOCONVERT introduced by precise projection, so authored contract nodes
  // keep their own source-level identity.
  for (const auto &node_owner : _nodes) {
    auto *node = node_owner.get();
    if (node->special_type != GraphNodeSpecialType::autoconvert ||
        node->inputs.size() != 1u || node->outputs.size() != 1u) {
      continue;
    }
    auto *input = &node->inputs.front();
    auto *output = &node->outputs.front();
    while (input->link != nullptr) {
      auto *previous = input->link->parent;
      if (previous == nullptr ||
          previous->shader_node_type() != NODE_CONVERT ||
          previous->inputs.size() != 1u || previous->outputs.size() != 1u ||
          previous->inputs.front().link == nullptr ||
          !is_float3_socket(previous->inputs.front().type) ||
          !is_float3_socket(previous->outputs.front().type) ||
          previous->outputs.front().type != input->type) {
        break;
      }

      auto *source = previous->inputs.front().link;
      const auto source_type = previous->inputs.front().type;
      disconnect(input);
      if (source_type == output->type) {
        relink(node, output, source);
        break;
      }

      const auto source_name = float3_socket_name(source_type);
      const auto target_name = float3_socket_name(output->type);
      input->name = "value_" + std::string{source_name};
      input->type = source_type;
      input->value = zero_float3_value(source_type);
      node->label = "Convert " + std::string{source_name} + " to " +
                    std::string{target_name};
      if (!connect(source, input)) {
        reject("Cycles float3 autoconvert composition failed");
        return;
      }
    }
  }
}

void CyclesGraph::disconnect(GraphInput *input) noexcept {
  if (input == nullptr || input->link == nullptr) {
    return;
  }
  auto &links = input->link->links;
  links.erase(std::remove(links.begin(), links.end(), input), links.end());
  input->link = nullptr;
}

void CyclesGraph::disconnect(GraphOutput *output) noexcept {
  if (output == nullptr) {
    return;
  }
  for (auto *input : output->links) {
    input->link = nullptr;
  }
  output->links.clear();
}

void CyclesGraph::reject(std::string diagnostic) {
  if (_diagnostic.empty()) {
    _diagnostic = std::move(diagnostic);
  }
}

CyclesGraph CyclesGraph::project(
    const ShaderProgram &shader,
    const contract::ShaderColorSpace &color_space) {
  CyclesGraph graph;
  graph._color_space = color_space;
  const auto registry = make_core_node_registry();
  std::unordered_map<std::uint32_t, GraphNode *> nodes;

  // ShaderGraph constructs OutputNode first, before every authored node.
  auto *output_node = graph.add_node("cycles.synthetic.output", "Output",
                                     output_inputs(), {},
                                     GraphNodeSpecialType::output);

  for (const auto &source : shader.graph().nodes()) {
    if (source.id.value + 1u != graph._next_node_id) {
      graph.reject("Cycles SVM requires source nodes in creation-id order");
      return graph;
    }
    const auto *schema = registry.find(source.type);
    if (schema == nullptr) {
      graph.reject("Cycles SVM graph has no schema for node: " + source.type);
      return graph;
    }
    const auto target_type = projected_node_type(source.type);

    std::vector<GraphInput> inputs;
    inputs.reserve(schema->inputs.size() + 2u);
    for (const auto &socket : schema->inputs) {
      const auto type = graph_socket_type(socket.type);
      if (!type) {
        // Cycles properties do not occupy ShaderInput stack lanes. A source
        // schema using such a type as an input cannot be projected silently.
        graph.reject("Cycles SVM input type is not stack representable: " +
                     source.type + "." + socket.name);
        return graph;
      }
      const auto binding = source.inputs.find(socket.name);
      const auto projected_type =
          projected_input_type(target_type, socket.name, *type);
      auto value = binding == source.inputs.end() ? std::nullopt
                                                  : binding->second.value;
      inputs.emplace_back(GraphInput{
          .name = std::string{projected_input_name(target_type, socket.name)},
          .type = projected_type,
          .flags = socket_flags(target_type, socket.name),
          .value = projected_input_value(std::move(value), projected_type),
      });
    }
    if (!projected_binary_math_operation(source.type).empty()) {
      inputs.emplace_back(GraphInput{
          .name = "Value3",
          .type = GraphSocketType::floating,
          .value = contract::SocketValue::floating(0.0f),
      });
    }
    if (source.type == node_type::bump) {
      auto authored_inputs = std::move(inputs);
      inputs.clear();
      inputs.reserve(8u);
      const auto append_authored = [&](std::string_view name) {
        const auto iter = std::find_if(
            authored_inputs.begin(), authored_inputs.end(),
            [name](const GraphInput &input) { return input.name == name; });
        if (iter == authored_inputs.end()) {
          return false;
        }
        inputs.emplace_back(std::move(*iter));
        authored_inputs.erase(iter);
        return true;
      };
      if (!append_authored("Height")) {
        graph.reject("Cycles Bump Height input is absent");
        return graph;
      }
      for (const auto name : {"SampleCenter", "SampleX", "SampleY"}) {
        inputs.emplace_back(GraphInput{
            .name = name,
            .type = GraphSocketType::floating,
            .value = contract::SocketValue::floating(0.0f),
        });
      }
      if (!append_authored("Normal") || !append_authored("Strength") ||
          !append_authored("Distance") || !append_authored("Filter Width") ||
          !authored_inputs.empty()) {
        graph.reject("Cycles Bump input projection is not isomorphic");
        return graph;
      }
    }
    if (is_surface_closure(target_type)) {
      inputs.emplace_back(GraphInput{
          .name = "SurfaceMixWeight",
          .type = GraphSocketType::floating,
          .value = contract::SocketValue::floating(0.0f),
      });
    }
    if (is_volume_closure(target_type) || target_type == node_type::emission) {
      inputs.emplace_back(GraphInput{
          .name = "VolumeMixWeight",
          .type = GraphSocketType::floating,
          .value = contract::SocketValue::floating(0.0f),
      });
    }

    std::vector<GraphOutput> outputs;
    outputs.reserve(schema->outputs.size());
    for (const auto &socket : schema->outputs) {
      const auto type = graph_socket_type(socket.type);
      if (!type) {
        graph.reject("Cycles SVM output type is not stack representable: " +
                     source.type + "." + socket.name);
        return graph;
      }
      outputs.emplace_back(
          GraphOutput{.name =
                          std::string{
                              projected_output_name(target_type, socket.name)},
                      .type = *type,
                      .links = {}});
    }

    auto properties = source.properties;
    if (source.type == node_type::multiply_color) {
      properties = {
          {"BlendMode", contract::SocketValue::string("MULTIPLY")},
          {"ClampFactor", contract::SocketValue::boolean(true)},
          {"ClampResult", contract::SocketValue::boolean(false)},
      };
    }
    if (const auto operation = projected_binary_math_operation(source.type);
        !operation.empty()) {
      properties = {
          {"Operation", contract::SocketValue::string(std::string{operation})},
      };
    }
    // NormalLinked belongs to the legacy surface-program representation.
    // Cycles encodes this fact solely as the ShaderInput edge itself.
    properties.erase("NormalLinked");
    auto *node = graph.add_node(std::string{target_type}, source.label,
                                std::move(inputs), std::move(outputs),
                                special_type(target_type),
                                std::move(properties));
    nodes.emplace(source.id.value, node);
  }

  for (const auto &source : shader.graph().nodes()) {
    auto *destination = nodes.at(source.id.value);
    for (const auto &[name, binding] : source.inputs) {
      if (!binding.source) {
        continue;
      }
      const auto producer_iter = nodes.find(binding.source->node.value);
      if (producer_iter == nodes.end() ||
          is_null_closure(producer_iter->second->type)) {
        continue;
      }
      auto *input = destination->input(
          projected_input_name(destination->type, name));
      auto *output = producer_iter->second->output(projected_output_name(
          producer_iter->second->type, binding.source->socket));
      if (!graph.connect_with_autoconvert(output, input)) {
        graph.reject("Cycles SVM could not project graph link into " +
                     destination->type + "." + name);
        return graph;
      }
    }
  }

  for (auto index = std::size_t{};
       index < static_cast<std::size_t>(contract::ShaderDomain::count);
       ++index) {
    const auto domain = static_cast<contract::ShaderDomain>(index);
    const auto &source_root = shader.graph().root(domain);
    if (!source_root) {
      continue;
    }
    const auto node_iter = nodes.find(source_root->node.value);
    if (node_iter == nodes.end() || is_null_closure(node_iter->second->type)) {
      continue;
    }
    auto *output = node_iter->second->output(projected_output_name(
        node_iter->second->type, source_root->socket));
    if (output == nullptr) {
      graph.reject("Cycles SVM root refers to an absent output socket");
      return graph;
    }
    const auto output_input_name =
        domain == contract::ShaderDomain::surface          ? "Surface"
        : domain == contract::ShaderDomain::volume         ? "Volume"
        : domain == contract::ShaderDomain::displacement   ? "Displacement"
                                                           : "Normal";
    if (!graph.connect(output, output_node->input(output_input_name))) {
      graph.reject("Cycles SVM could not connect graph output root");
      return graph;
    }
  }

  graph.inline_blender_functions();
  graph.project_texture_mappings();
  if (!graph.valid()) {
    return graph;
  }
  graph.expand();
  graph.default_inputs();
  graph.compose_float3_autoconverts();
  graph.clean();
  graph.refine_bump_nodes();
  if (!graph.valid()) {
    return graph;
  }
  if (auto *surface = graph.root(GraphDomain::surface)) {
    graph.transform_multi_closure(surface->parent, nullptr, false);
  }
  if (auto *volume = graph.root(GraphDomain::volume)) {
    graph.transform_multi_closure(volume->parent, nullptr, true);
  }
  return graph;
}

Vec3f CyclesGraph::rec709_to_scene_linear(Vec3f value) const noexcept {
  return {
      _color_space.rec709_to_r.x * value.x +
          _color_space.rec709_to_r.y * value.y +
          _color_space.rec709_to_r.z * value.z,
      _color_space.rec709_to_g.x * value.x +
          _color_space.rec709_to_g.y * value.y +
          _color_space.rec709_to_g.z * value.z,
      _color_space.rec709_to_b.x * value.x +
          _color_space.rec709_to_b.y * value.y +
          _color_space.rec709_to_b.z * value.z,
  };
}

void CyclesGraph::find_dependencies(GraphNodeSet &dependencies,
                                    GraphInput *input) {
  auto *node = input != nullptr && input->link != nullptr ? input->link->parent
                                                          : nullptr;
  if (node != nullptr && !dependencies.contains(node)) {
    for (auto &dependency : node->inputs) {
      find_dependencies(dependencies, &dependency);
    }
    dependencies.insert(node);
  }
}

void CyclesGraph::copy_nodes(
    GraphNodeSet &nodes,
    std::map<GraphNode *, GraphNode *, GraphNodeIdComparator> &node_map) {
  for (auto *node : nodes) {
    std::vector<GraphInput> inputs;
    inputs.reserve(node->inputs.size());
    for (const auto &input : node->inputs) {
      inputs.emplace_back(GraphInput{.name = input.name,
                                     .type = input.type,
                                     .flags = input.flags,
                                     .value = input.value});
    }
    std::vector<GraphOutput> outputs;
    outputs.reserve(node->outputs.size());
    for (const auto &output : node->outputs) {
      outputs.emplace_back(
          GraphOutput{.name = output.name, .type = output.type, .links = {}});
    }
    auto *copy =
        add_node(node->type, node->label, std::move(inputs), std::move(outputs),
                 node->special_type, node->properties);
    copy->bump = node->bump;
    copy->copy_runtime_state_from(*node);
    node_map.emplace(node, copy);
  }

  for (auto *node : nodes) {
    for (const auto &input : node->inputs) {
      if (input.link == nullptr) {
        continue;
      }
      auto *source = node_map.at(input.link->parent);
      auto *destination = node_map.at(input.parent);
      if (!connect(source->output(input.link->name),
                   destination->input(input.name))) {
        reject("Cycles bump dependency copy could not recreate a link");
        return;
      }
    }
  }
}

void CyclesGraph::project_texture_mappings() {
  for (const auto &node_owner : _nodes) {
    auto *mapping_node = node_owner.get();
    if (mapping_node->type != node_type::mapping) {
      continue;
    }
    const auto marker_iter =
        mapping_node->properties.find("LegacyTextureMapping");
    if (marker_iter == mapping_node->properties.end() ||
        marker_iter->second.type != contract::SocketType::boolean) {
      reject("Cycles Mapping legacy marker is absent or ill typed");
      return;
    }
    const auto *marked = std::get_if<bool>(&marker_iter->second.value);
    if (marked == nullptr) {
      reject("Cycles Mapping legacy marker is ill typed");
      return;
    }
    if (!*marked) {
      continue;
    }

    auto *mapping_input = mapping_node->input("Vector");
    auto *mapping_output = mapping_node->output("Vector");
    if (mapping_input == nullptr || mapping_output == nullptr ||
        mapping_output->links.size() != 1u) {
      reject("Cycles legacy TextureMapping must have one texture consumer");
      return;
    }
    auto *texture_input = mapping_output->links.front();
    auto *texture_node = texture_input->parent;
    auto *texture_mapping = texture_node->texture_mapping();
    if (texture_input->name != "Vector" || texture_mapping == nullptr) {
      reject("Cycles legacy TextureMapping consumer is not a TextureNode");
      return;
    }
    std::string diagnostic;
    if (!configure_texture_mapping_from_legacy_node(
            *texture_mapping, *mapping_node, diagnostic)) {
      reject(std::move(diagnostic));
      return;
    }

    auto *source = mapping_input->link;
    const auto value = mapping_input->value;
    disconnect(texture_input);
    if (source != nullptr) {
      if (!connect(source, texture_input)) {
        reject("Cycles legacy TextureMapping input bypass failed");
        return;
      }
    } else {
      texture_input->value = value;
    }
  }
}

void CyclesGraph::refine_bump_nodes() {
  for (auto index = std::size_t{}; index < _nodes.size(); ++index) {
    auto *node = _nodes[index].get();
    auto *height = node->input("Height");
    if (node->special_type != GraphNodeSpecialType::bump || height == nullptr ||
        height->link == nullptr) {
      continue;
    }
    auto *filter_width_input = node->input("Filter Width");
    const auto filter_width = float_value(filter_width_input);
    if (!filter_width) {
      reject("Cycles Bump Filter Width is not a float");
      return;
    }

    GraphNodeSet dependencies;
    find_dependencies(dependencies, height);
    std::map<GraphNode *, GraphNode *, GraphNodeIdComparator> nodes_dx;
    std::map<GraphNode *, GraphNode *, GraphNodeIdComparator> nodes_dy;
    copy_nodes(dependencies, nodes_dx);
    copy_nodes(dependencies, nodes_dy);
    if (!valid()) {
      return;
    }

    for (auto *dependency : dependencies) {
      dependency->bump = SHADER_BUMP_CENTER;
      dependency->bump_filter_width = *filter_width;
    }
    for (const auto &[source, copy] : nodes_dx) {
      static_cast<void>(source);
      copy->bump = SHADER_BUMP_DX;
      copy->bump_filter_width = *filter_width;
    }
    for (const auto &[source, copy] : nodes_dy) {
      static_cast<void>(source);
      copy->bump = SHADER_BUMP_DY;
      copy->bump_filter_width = *filter_width;
    }

    auto *output = height->link;
    auto *output_dx = nodes_dx.at(output->parent)->output(output->name);
    auto *output_dy = nodes_dy.at(output->parent)->output(output->name);
    if (!connect(output_dx, node->input("SampleX")) ||
        !connect(output_dy, node->input("SampleY")) ||
        !connect(output, node->input("SampleCenter"))) {
      reject("Cycles Bump sample graph could not be connected");
      return;
    }
    disconnect(height);
  }
}

void CyclesGraph::default_inputs() {
  GraphNode *geometry = nullptr;
  GraphNode *texture_coordinate = nullptr;
  GraphNode *incoming_transform = nullptr;

  // The loop intentionally observes appended nodes, matching Cycles'
  // index-based ShaderGraph::default_inputs traversal.
  for (auto index = std::size_t{}; index < _nodes.size(); ++index) {
    auto *node = _nodes[index].get();
    for (auto &input : node->inputs) {
      // This graph is the SVM graph (do_osl=false). Match Cycles' exact
      // `!OSL_INTERNAL || do_osl` predicate before considering LINK_* flags.
      if (input.link != nullptr ||
          (input.flags & graph_socket_osl_internal) != 0u) {
        continue;
      }
      if ((input.flags & (graph_socket_link_texture_generated |
                          graph_socket_link_texture_normal |
                          graph_socket_link_texture_uv)) != 0u) {
        if (texture_coordinate == nullptr) {
          texture_coordinate = add_node(
              cycles_synthetic_texture_coordinate, "Texture Coordinate",
              texture_coordinate_inputs(), texture_coordinate_outputs(),
              GraphNodeSpecialType::none,
              texture_coordinate_properties());
        }
        auto *output = texture_coordinate->output(
            (input.flags & graph_socket_link_texture_generated) != 0u
                ? "Generated"
            : (input.flags & graph_socket_link_texture_normal) != 0u ? "Normal"
                                                                     : "UV");
        static_cast<void>(connect(output, &input));
      } else if ((input.flags & graph_socket_link_texture_incoming) != 0u) {
        if (geometry == nullptr) {
          geometry = add_node(node_type::geometry, "Geometry",
                              geometry_inputs(), geometry_outputs(),
                              GraphNodeSpecialType::geometry);
        }
        if (incoming_transform == nullptr) {
          incoming_transform = add_node(
              node_type::vector_transform, "Vector Transform",
              incoming_transform_inputs(), incoming_transform_outputs(),
              GraphNodeSpecialType::none, incoming_transform_properties());
          if (!connect(geometry->output("Incoming"),
                       incoming_transform->input("Vector"))) {
            reject("Cycles texture Incoming transform could not be connected");
            return;
          }
        }
        if (!connect_with_autoconvert(
                incoming_transform->output("Vector"), &input)) {
          reject("Cycles transformed texture Incoming could not be connected");
          return;
        }
      } else if ((input.flags & (graph_socket_link_incoming |
                                 graph_socket_link_normal |
                                 graph_socket_link_position |
                                 graph_socket_link_tangent)) != 0u) {
        if (geometry == nullptr) {
          geometry = add_node(node_type::geometry, "Geometry",
                              geometry_inputs(), geometry_outputs(),
                              GraphNodeSpecialType::geometry);
        }
        const auto output_name =
            (input.flags & graph_socket_link_normal) != 0u      ? "Normal"
            : (input.flags & graph_socket_link_position) != 0u ? "Position"
            : (input.flags & graph_socket_link_tangent) != 0u  ? "Tangent"
                                                               : "Incoming";
        static_cast<void>(connect(geometry->output(output_name), &input));
      }
    }
  }
}

void CyclesGraph::expand() {
  for (auto index = std::size_t{}; index < _nodes.size(); ++index) {
    _nodes[index]->expand(*this);
  }
}

void CyclesGraph::inline_blender_functions() {
  // Blender's shader-node inliner evaluates a node only when it has a
  // multi-function and every available input is primitive. Individual node
  // implementations encode socket availability and field-valued defaults;
  // this topological pass supplies the same one-way primitive propagation.
  run_constant_fold_stage(*this, ConstantFoldStage::blender_inline);
}

void CyclesGraph::constant_fold() {
  const auto has_displacement =
      root(GraphDomain::displacement) != nullptr;
  run_constant_fold_stage(*this, ConstantFoldStage::cycles_graph);

  // Cycles restores a ColorNode when folding removes the displacement root.
  // Until ColorNode plus the exact automatic Color-to-Vector ConvertNode are
  // migrated, rejecting this boundary is the only non-divergent behavior.
  if (has_displacement && root(GraphDomain::displacement) == nullptr) {
    reject("Cycles displacement constant restoration is not migrated");
  }
}

void CyclesGraph::simplify_settings() {
  for (const auto &node : _nodes) {
    node->simplify_settings();
  }
}

void CyclesGraph::deduplicate_nodes() {
  GraphNodeSet scheduled;
  GraphNodeSet done;
  std::map<std::string, GraphNodeSet, std::less<>> candidates;
  std::queue<GraphNode *> traverse_queue;

  for (const auto &node : _nodes) {
    if (!check_node_inputs_has_links(node.get())) {
      traverse_queue.push(node.get());
      scheduled.insert(node.get());
    }
  }

  while (!traverse_queue.empty()) {
    auto *node = traverse_queue.front();
    traverse_queue.pop();
    done.insert(node);
    auto has_output_links = false;
    for (auto &output_socket : node->outputs) {
      for (auto *input : output_socket.links) {
        has_output_links = true;
        if (!scheduled.contains(input->parent) &&
            check_node_inputs_traversed(input->parent, done)) {
          traverse_queue.push(input->parent);
          scheduled.insert(input->parent);
        }
      }
    }
    if (!has_output_links) {
      continue;
    }

    GraphNode *merge_with = nullptr;
    for (auto *candidate : candidates[node->type]) {
      if (candidate != node && node->equals(*candidate)) {
        merge_with = candidate;
        break;
      }
    }
    if (merge_with == nullptr) {
      candidates[node->type].insert(node);
      continue;
    }
    if (node->outputs.size() != merge_with->outputs.size()) {
      std::abort();
    }
    for (auto index = std::size_t{}; index < node->outputs.size(); ++index) {
      relink(node, &node->outputs[index], &merge_with->outputs[index]);
    }
  }
}

void CyclesGraph::optimize_volume_output() {
  auto *node = output_node();
  auto *volume = node != nullptr ? node->input("Volume") : nullptr;
  if (volume == nullptr || volume->link == nullptr) {
    return;
  }

  struct NodeAndNonLinearComparator {
    [[nodiscard]] bool operator()(
        const std::pair<GraphNode *, bool> &lhs,
        const std::pair<GraphNode *, bool> &rhs) const noexcept {
      return lhs.first->id < rhs.first->id ||
             (lhs.first->id == rhs.first->id && lhs.second < rhs.second);
    }
  };

  auto has_valid_volume = false;
  std::set<std::pair<GraphNode *, bool>, NodeAndNonLinearComparator> scheduled;
  std::queue<std::pair<GraphNode *, bool>> traverse_queue;
  traverse_queue.emplace(volume->link->parent, false);
  scheduled.emplace(volume->link->parent, false);

  while (!traverse_queue.empty()) {
    auto [current, nonlinear] = traverse_queue.front();
    traverse_queue.pop();
    if (nonlinear && current->type == node_type::attribute) {
      current->properties["Stochastic"] =
          contract::SocketValue::boolean(false);
    }
    nonlinear = nonlinear || !current->is_linear_operation();
    has_valid_volume = has_valid_volume || current->has_volume_support();
    for (auto &input : current->inputs) {
      if (input.link == nullptr) {
        continue;
      }
      const auto state = std::pair{input.link->parent, nonlinear};
      if (scheduled.insert(state).second) {
        traverse_queue.push(state);
      }
    }
  }

  if (!has_valid_volume) {
    disconnect(volume->link);
  }
}

void CyclesGraph::relink(GraphNode *node, GraphOutput *from,
                         GraphOutput *to) {
  if (node == nullptr || from == nullptr || to == nullptr) {
    std::abort();
  }
  for (auto &input : node->inputs) {
    if (input.link != nullptr) {
      disconnect(&input);
    }
  }
  const auto links = from->links;
  for (auto *input : links) {
    disconnect(input);
    if (!connect(to, input)) {
      std::abort();
    }
  }
}

void CyclesGraph::break_cycles(GraphNode *node, std::vector<bool> &visited,
                               std::vector<bool> &on_stack) {
  visited[node->id] = true;
  on_stack[node->id] = true;
  for (auto &input : node->inputs) {
    if (input.link == nullptr) {
      continue;
    }
    auto *dependency = input.link->parent;
    if (on_stack[dependency->id]) {
      disconnect(&input);
    } else if (!visited[dependency->id]) {
      break_cycles(dependency, visited, on_stack);
    }
  }
  on_stack[node->id] = false;
}

void CyclesGraph::clean() {
  constant_fold();
  simplify_settings();
  deduplicate_nodes();
  optimize_volume_output();

  std::vector<bool> visited(_next_node_id, false);
  std::vector<bool> on_stack(_next_node_id, false);
  if (auto *node = output_node()) {
    break_cycles(node, visited, on_stack);
  }
  for (const auto &node : _nodes) {
    if (node->special_type == GraphNodeSpecialType::output_aov) {
      break_cycles(node.get(), visited, on_stack);
    }
  }

  for (const auto &node : _nodes) {
    if (visited[node->id]) {
      continue;
    }
    for (auto &input : node->inputs) {
      if (input.link != nullptr) {
        disconnect(&input);
      }
    }
  }

  std::vector<std::unique_ptr<GraphNode>> new_nodes;
  new_nodes.reserve(_nodes.size());
  for (auto &node : _nodes) {
    if (visited[node->id]) {
      new_nodes.emplace_back(std::move(node));
    }
  }
  _nodes = std::move(new_nodes);
}

void CyclesGraph::transform_multi_closure(GraphNode *node,
                                          GraphOutput *weight_output,
                                          bool volume) {
  if (node == nullptr) {
    return;
  }
  if (node->special_type == GraphNodeSpecialType::combine_closure) {
    auto *factor = node->input("Fac");
    auto *closure1 = node->input("Closure1");
    auto *closure2 = node->input("Closure2");
    if (closure1 == nullptr || closure2 == nullptr) {
      reject("Cycles combine-closure sockets are incomplete");
      return;
    }

    GraphOutput *weight1 = weight_output;
    GraphOutput *weight2 = weight_output;
    if (factor != nullptr) {
      auto *mix = add_node(cycles_synthetic_mix_closure_weight,
                           "Mix Closure Weight", mix_weight_inputs(),
                           mix_weight_outputs());
      if (factor->link != nullptr) {
        static_cast<void>(connect(factor->link, mix->input("Fac")));
      } else if (const auto value = float_value(factor)) {
        mix->input("Fac")->value = contract::SocketValue::floating(*value);
      } else {
        reject("Cycles mix-closure factor is not a float");
        return;
      }
      if (weight_output != nullptr) {
        static_cast<void>(connect(weight_output, mix->input("Weight")));
      }
      weight1 = mix->output("Weight1");
      weight2 = mix->output("Weight2");
    }

    if (closure1->link != nullptr) {
      transform_multi_closure(closure1->link->parent, weight1, volume);
    }
    if (closure2->link != nullptr) {
      transform_multi_closure(closure2->link->parent, weight2, volume);
    }
    return;
  }

  auto *weight = node->input(volume ? "VolumeMixWeight" : "SurfaceMixWeight");
  if (weight == nullptr) {
    return;
  }
  const auto literal = float_value(weight);
  if (!literal) {
    reject("Cycles closure mix weight is not a float");
    return;
  }
  if (weight->link != nullptr || *literal != 0.0f) {
    auto *multiply = add_node(
        cycles_synthetic_math, "Closure Weight Multiply", multiply_inputs(),
        multiply_outputs(), GraphNodeSpecialType::none,
        {{"Operation", contract::SocketValue::string("MULTIPLY")}});
    if (weight->link != nullptr) {
      static_cast<void>(connect(weight->link, multiply->input("Value1")));
      disconnect(weight);
    } else {
      multiply->input("Value1")->value =
          contract::SocketValue::floating(*literal);
    }
    if (weight_output != nullptr) {
      static_cast<void>(connect(weight_output, multiply->input("Value2")));
    } else {
      multiply->input("Value2")->value =
          contract::SocketValue::floating(1.0f);
    }
    weight_output = multiply->output("Value");
  }

  if (weight_output != nullptr) {
    static_cast<void>(connect(weight_output, weight));
  } else {
    weight->value = contract::SocketValue::floating(*literal + 1.0f);
  }
}

} // namespace psycles::compiler::cycles_svm
