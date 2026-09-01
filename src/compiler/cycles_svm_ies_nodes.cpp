/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_ies_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<std::string_view>
string_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::string) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<std::string>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

class IESLightNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_IES;
  }

  void compile(SVMCompiler &compiler) override {
    const auto content = string_property(this, "IES");
    auto *vector_input = input("Vector");
    if (!content || vector_input == nullptr) {
      compiler.fail("Cycles IES Light node state is ill typed");
      return;
    }

    // Exact IESLightNode::compile order: embedded TextureMapping first,
    // NODE_IES second, then release the temporary mapped coordinate.
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_input, nullptr);
    compiler.add_node(
        this, NODE_IES,
        SVMNodeIES{.strength = compiler.input_float("Strength"),
                   .slot = compiler.ies(*content),
                   .vector_offset = vector_offset,
                   .fac_offset = compiler.output("Factor"),
                   ._pad = {0u, 0u}});
    tex_mapping.compile_end(compiler, vector_input, vector_offset);
  }
};

} // namespace

std::unique_ptr<GraphNode> make_ies_graph_node(std::string_view type) {
  if (type == node_type::ies_light) {
    return std::make_unique<IESLightNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
