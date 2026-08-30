/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_graph.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
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

[[nodiscard]] std::string_view projected_input_name(
    std::string_view node, std::string_view input) noexcept {
  if (node == node_type::math) {
    return input == "A"   ? "Value1"
           : input == "B" ? "Value2"
           : input == "C" ? "Value3"
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
  if (node == node_type::separate_color) {
    return output == "R"   ? "Red"
           : output == "G" ? "Green"
           : output == "B" ? "Blue"
                           : output;
  }
  return output;
}

[[nodiscard]] std::uint16_t socket_flags(std::string_view node,
                                         std::string_view input) noexcept {
  // Direct projection of the SocketType::LINK_* flags declared by the
  // corresponding Cycles 5.2.1 NODE_DEFINE blocks in shader_nodes.cpp.
  if (input == "Normal" &&
      (node == node_type::layer_weight || node == node_type::fresnel ||
       node == node_type::ambient_occlusion || node == node_type::bump ||
       node == node_type::displacement || is_surface_closure(node))) {
    return graph_socket_link_normal;
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
    if (node == node_type::image_texture) {
      return graph_socket_link_texture_uv;
    }
    if (node == node_type::environment_texture) {
      return graph_socket_link_position;
    }
    if (node == node_type::gradient_texture || node == node_type::noise_texture ||
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

CyclesGraph CyclesGraph::project(const ShaderProgram &shader) {
  CyclesGraph graph;
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
      inputs.emplace_back(GraphInput{
          .name = std::string{projected_input_name(source.type, socket.name)},
          .type = *type,
          .flags = socket_flags(source.type, socket.name),
          .value = binding == source.inputs.end() ? std::nullopt
                                                  : binding->second.value,
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
    if (is_surface_closure(source.type)) {
      inputs.emplace_back(GraphInput{
          .name = "SurfaceMixWeight",
          .type = GraphSocketType::floating,
          .value = contract::SocketValue::floating(0.0f),
      });
    }
    if (is_volume_closure(source.type) || source.type == node_type::emission) {
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
                              projected_output_name(source.type, socket.name)},
                      .type = *type,
                      .links = {}});
    }

    auto properties = source.properties;
    if (source.type == node_type::bump) {
      properties.erase("NormalLinked");
    }
    auto *node = graph.add_node(source.type, source.label, std::move(inputs),
                                std::move(outputs), special_type(source.type),
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
          projected_input_name(source.type, name));
      auto *output = producer_iter->second->output(projected_output_name(
          producer_iter->second->type, binding.source->socket));
      if (!graph.connect(output, input)) {
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

  graph.expand();
  graph.default_inputs();
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

  // The loop intentionally observes appended nodes, matching Cycles'
  // index-based ShaderGraph::default_inputs traversal.
  for (auto index = std::size_t{}; index < _nodes.size(); ++index) {
    auto *node = _nodes[index].get();
    for (auto &input : node->inputs) {
      if (input.link != nullptr) {
        continue;
      }
      if ((input.flags & (graph_socket_link_texture_generated |
                          graph_socket_link_texture_normal |
                          graph_socket_link_texture_uv)) != 0u) {
        if (texture_coordinate == nullptr) {
          texture_coordinate = add_node(
              cycles_synthetic_texture_coordinate, "Texture Coordinate",
              texture_coordinate_inputs(), texture_coordinate_outputs());
        }
        auto *output = texture_coordinate->output(
            (input.flags & graph_socket_link_texture_generated) != 0u
                ? "Generated"
            : (input.flags & graph_socket_link_texture_normal) != 0u ? "Normal"
                                                                     : "UV");
        static_cast<void>(connect(output, &input));
      } else if ((input.flags & (graph_socket_link_texture_incoming |
                                 graph_socket_link_incoming |
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

void CyclesGraph::constant_fold() {
  GraphNodeSet done;
  GraphNodeSet scheduled;
  std::queue<GraphNode *> traverse_queue;
  const auto has_displacement =
      root(GraphDomain::displacement) != nullptr;

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
      const ConstantFolder folder{this, node, &output_socket};
      node->constant_fold(folder);
    }
  }

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
