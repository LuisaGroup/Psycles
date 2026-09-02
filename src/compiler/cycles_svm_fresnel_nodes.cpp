/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_fresnel_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

namespace psycles::compiler::cycles_svm {
namespace {

class FresnelNode final : public GraphNode {
public:
  [[nodiscard]] bool has_spatial_varying() const noexcept override {
    return true;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_FRESNEL;
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_FRESNEL,
        SVMNodeFresnel{.ior = compiler.input_float("IOR"),
                       .normal_offset = compiler.input_link("Normal"),
                       .out_offset = compiler.output("Factor"),
                       ._pad = {0u, 0u}});
  }
};

class LayerWeightNode final : public GraphNode {
public:
  [[nodiscard]] bool has_spatial_varying() const noexcept override {
    return true;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_LAYER_WEIGHT;
  }

  void compile(SVMCompiler &compiler) override {
    const auto emit = [&](NodeBlendWeightType weight_type,
                          std::string_view output_name) {
      auto *shader_output = output(output_name);
      if (shader_output != nullptr && !shader_output->links.empty()) {
        compiler.add_node(
            this, NODE_LAYER_WEIGHT,
            SVMNodeLayerWeight{.weight_type = weight_type,
                               .blend = compiler.input_float("Blend"),
                               .normal_offset = compiler.input_link("Normal"),
                               .out_offset = compiler.output(output_name),
                               ._pad = {0u, 0u}});
      }
    };
    emit(NODE_LAYER_WEIGHT_FRESNEL, "Fresnel");
    emit(NODE_LAYER_WEIGHT_FACING, "Facing");
  }
};

} // namespace

std::unique_ptr<GraphNode> make_fresnel_graph_node(std::string_view type) {
  if (type == node_type::fresnel) {
    return std::make_unique<FresnelNode>();
  }
  if (type == node_type::layer_weight) {
    return std::make_unique<LayerWeightNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
