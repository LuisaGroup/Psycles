/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_light_falloff_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <array>

namespace psycles::compiler::cycles_svm {
namespace {

struct LightFalloffOutput {
  std::string_view name;
  NodeLightFalloff falloff_type;
};

// Cycles 5.2.1 LightFalloffNode::compile emission order. Each live output is
// one SVM record because its falloff selector is part of the instruction.
constexpr auto light_falloff_outputs = std::array{
    LightFalloffOutput{"Quadratic", NODE_LIGHT_FALLOFF_QUADRATIC},
    LightFalloffOutput{"Linear", NODE_LIGHT_FALLOFF_LINEAR},
    LightFalloffOutput{"Constant", NODE_LIGHT_FALLOFF_CONSTANT},
};

class LightFalloffNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_LIGHT_FALLOFF;
  }

  void compile(SVMCompiler &compiler) override {
    for (const auto &[name, falloff_type] : light_falloff_outputs) {
      const auto *shader_output = output(name);
      if (shader_output == nullptr || shader_output->links.empty()) {
        continue;
      }
      compiler.add_node(
          this, NODE_LIGHT_FALLOFF,
          SVMNodeLightFalloff{.falloff_type = falloff_type,
                              .strength = compiler.input_float("Strength"),
                              .smooth = compiler.input_float("Smooth"),
                              .out_offset = compiler.output(name),
                              ._pad = {0u, 0u, 0u}});
    }
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_light_falloff_graph_node(std::string_view type) {
  if (type == node_type::light_falloff) {
    return std::make_unique<LightFalloffNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
