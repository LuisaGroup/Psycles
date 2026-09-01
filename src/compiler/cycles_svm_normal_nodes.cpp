/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_normal_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<Vec3f>
vector_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::vector) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<Vec3f>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

class NormalNode final : public GraphNode {
public:
  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_NORMAL;
  }

  void compile(SVMCompiler &compiler) override {
    const auto direction = vector_property(this, "Direction").value_or(Vec3f{});
    compiler.add_node(
        this, NODE_NORMAL,
        SVMNodeNormal{.in_normal = compiler.input_float3("Normal"),
                      .out_normal_offset = compiler.output("Normal"),
                      .out_dot_offset = compiler.output("Dot"),
                      ._pad = {0u, 0u},
                      .direction_x = direction.x,
                      .direction_y = direction.y,
                      .direction_z = direction.z});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_normal_graph_node(std::string_view type) {
  if (type == node_type::normal) {
    return std::make_unique<NormalNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
