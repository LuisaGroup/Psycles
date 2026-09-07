/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_range_nodes.h"
#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<NodeMapRangeType>
range_type(const GraphNode &node) {
  const auto it = node.properties.find("Interpolation");
  if (it == node.properties.end()) {
    return std::nullopt;
  }
  const auto *name = std::get_if<std::string>(&it->second.value);
  if (!name) {
    return std::nullopt;
  }
  if (*name == "LINEAR") {
    return NODE_MAP_RANGE_LINEAR;
  }
  if (*name == "STEPPED") {
    return NODE_MAP_RANGE_STEPPED;
  }
  if (*name == "SMOOTHSTEP") {
    return NODE_MAP_RANGE_SMOOTHSTEP;
  }
  if (*name == "SMOOTHERSTEP") {
    return NODE_MAP_RANGE_SMOOTHERSTEP;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<bool> use_clamp(const GraphNode &node) {
  const auto it = node.properties.find("Clamp");
  if (it == node.properties.end()) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<bool>(&it->second.value)) {
    return *value;
  }
  return std::nullopt;
}

class MapRangeNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_MAP_RANGE;
  }

  void expand(CyclesGraph &graph) override {
    const auto clamp = use_clamp(*this);
    if (!clamp) {
      graph.reject("Cycles Map Range Clamp property is invalid");
      return;
    }
    auto *result = output("Result");
    if (!*clamp || result->links.empty()) {
      return;
    }
    auto *limit = graph.add_node(
        node_type::clamp_range, "Clamp",
        {{.name = "Value",
          .type = GraphSocketType::floating,
          .value = contract::SocketValue::floating(1.0f)},
         {.name = "Min",
          .type = GraphSocketType::floating,
          .value = input("To Min")->value},
         {.name = "Max",
          .type = GraphSocketType::floating,
          .value = input("To Max")->value}},
        {{.name = "Result", .type = GraphSocketType::floating, .links = {}}},
        GraphNodeSpecialType::none,
        {{"Mode", contract::SocketValue::string("RANGE")}});
    // Both nodes survive. CyclesGraph::relink also disconnects the replaced
    // node's inputs, so it is not the ShaderGraph output-only relink needed
    // here.
    const auto consumers = result->links;
    graph.disconnect(result);
    for (auto *consumer : consumers) {
      if (!graph.connect(limit->output("Result"), consumer)) {
        graph.reject("Cycles Map Range Clamp output relink failed");
        return;
      }
    }
    if (!graph.connect(result, limit->input("Value"))) {
      graph.reject("Cycles Map Range Clamp input link failed");
      return;
    }
    for (const auto &[from, to] :
         {std::pair{"To Min", "Min"}, std::pair{"To Max", "Max"}}) {
      auto *source = input(from)->link;
      if (source && !graph.connect(source, limit->input(to))) {
        graph.reject("Cycles Map Range Clamp bound link failed");
        return;
      }
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    // Cycles' scalar node checks only To Min/Max; its vector node checks all
    // four bounds. Preserve that graph/bump optimization distinction.
    return range_type(*this) == NODE_MAP_RANGE_LINEAR &&
           input("To Min")->link == nullptr && input("To Max")->link == nullptr;
  }

  void compile(SVMCompiler &compiler) override {
    const auto mode = range_type(*this);
    if (!mode) {
      compiler.fail("Cycles Map Range interpolation is invalid");
      return;
    }
    compiler.add_node(
        this, NODE_MAP_RANGE,
        SVMNodeMapRange{.range_type = *mode,
                        .value = compiler.input_float("Value"),
                        .from_min = compiler.input_float("From Min"),
                        .from_max = compiler.input_float("From Max"),
                        .to_min = compiler.input_float("To Min"),
                        .to_max = compiler.input_float("To Max"),
                        .steps = compiler.input_float("Steps"),
                        .result_offset = compiler.output("Result"),
                        ._pad = {0u, 0u, 0u}});
  }
};

class VectorMapRangeNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_VECTOR_MAP_RANGE;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return range_type(*this) == NODE_MAP_RANGE_LINEAR &&
           input("From_Min_FLOAT3")->link == nullptr &&
           input("From_Max_FLOAT3")->link == nullptr &&
           input("To_Min_FLOAT3")->link == nullptr &&
           input("To_Max_FLOAT3")->link == nullptr;
  }

  void compile(SVMCompiler &compiler) override {
    const auto mode = range_type(*this);
    const auto clamp = use_clamp(*this);
    if (!mode || !clamp) {
      compiler.fail("Cycles Vector Map Range properties are invalid");
      return;
    }
    compiler.add_node(this, NODE_VECTOR_MAP_RANGE,
                      SVMNodeVectorMapRange{
                          .range_type = *mode,
                          .use_clamp = static_cast<std::uint8_t>(*clamp),
                          ._pad = {0u, 0u, 0u},
                          .value = compiler.input_float3("Vector"),
                          .from_min = compiler.input_float3("From_Min_FLOAT3"),
                          .from_max = compiler.input_float3("From_Max_FLOAT3"),
                          .to_min = compiler.input_float3("To_Min_FLOAT3"),
                          .to_max = compiler.input_float3("To_Max_FLOAT3"),
                          .steps = compiler.input_float3("Steps_FLOAT3"),
                          .result_offset = compiler.output("Vector"),
                          ._pad2 = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_range_graph_node(std::string_view type) {
  if (type == node_type::map_range) {
    return std::make_unique<MapRangeNode>();
  }
  if (type == cycles_synthetic_vector_map_range) {
    return std::make_unique<VectorMapRangeNode>();
  }
  return nullptr;
}
} // namespace psycles::compiler::cycles_svm
