/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_graph.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

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
  if (type == node_type::geometry || type == cycles_synthetic_geometry) {
    return GraphNodeSpecialType::geometry;
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
      {.name = "GeometricNormal", .type = GraphSocketType::normal, .links = {}},
      {.name = "Incoming", .type = GraphSocketType::vector, .links = {}},
      {.name = "Parametric", .type = GraphSocketType::point, .links = {}},
      {.name = "Backfacing", .type = GraphSocketType::floating, .links = {}},
      {.name = "Pointiness", .type = GraphSocketType::floating, .links = {}},
      {.name = "RandomPerIsland", .type = GraphSocketType::floating, .links = {}},
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

[[nodiscard]] GraphDomain graph_domain(contract::ShaderDomain domain) noexcept {
  switch (domain) {
    case contract::ShaderDomain::surface:
      return GraphDomain::surface;
    case contract::ShaderDomain::volume:
      return GraphDomain::volume;
    case contract::ShaderDomain::surface_normal:
      return GraphDomain::bump;
    case contract::ShaderDomain::displacement:
      return GraphDomain::displacement;
    case contract::ShaderDomain::count:
      break;
  }
  return GraphDomain::count;
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
  const auto index = static_cast<std::size_t>(domain);
  return index < _roots.size() ? _roots[index] : nullptr;
}

GraphNode *CyclesGraph::add_node(
    std::string type, std::string label, std::vector<GraphInput> inputs,
    std::vector<GraphOutput> outputs, GraphNodeSpecialType node_special_type,
    std::map<std::string, contract::SocketValue, std::less<>> properties) {
  auto node = std::make_unique<GraphNode>();
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
  if (output == nullptr || input == nullptr || output->type != input->type) {
    return false;
  }
  disconnect(input);
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

void CyclesGraph::set_root(GraphDomain domain, GraphOutput *output) noexcept {
  const auto index = static_cast<std::size_t>(domain);
  if (index < _roots.size()) {
    _roots[index] = output;
  }
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
          .name = socket.name,
          .type = *type,
          .flags = socket_flags(source.type, socket.name),
          .value = binding == source.inputs.end() ? std::nullopt
                                                  : binding->second.value,
      });
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
          GraphOutput{.name = socket.name, .type = *type, .links = {}});
    }

    auto *node = graph.add_node(source.type, source.label, std::move(inputs),
                                std::move(outputs), special_type(source.type),
                                source.properties);
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
      auto *input = destination->input(name);
      auto *output = producer_iter->second->output(binding.source->socket);
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
    auto *output = node_iter->second->output(source_root->socket);
    if (output == nullptr) {
      graph.reject("Cycles SVM root refers to an absent output socket");
      return graph;
    }
    graph.set_root(graph_domain(domain), output);
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

  graph.default_inputs();
  if (auto *surface = graph.root(GraphDomain::surface)) {
    graph.transform_multi_closure(surface->parent, nullptr, false);
  }
  if (auto *volume = graph.root(GraphDomain::volume)) {
    graph.transform_multi_closure(volume->parent, nullptr, true);
  }
  return graph;
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
          geometry = add_node(cycles_synthetic_geometry, "Geometry",
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

void CyclesGraph::transform_multi_closure(GraphNode *node,
                                          GraphOutput *weight_output,
                                          bool volume) {
  if (node == nullptr) {
    return;
  }
  if (node->special_type == GraphNodeSpecialType::combine_closure) {
    auto *factor = node->input("Factor");
    auto *closure1 = node->input("A");
    auto *closure2 = node->input("B");
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
