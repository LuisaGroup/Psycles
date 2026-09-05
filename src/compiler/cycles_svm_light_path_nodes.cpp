/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_light_path_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <array>

namespace psycles::compiler::cycles_svm {
namespace {

struct LightPathOutput {
  std::string_view name;
  NodeLightPath path_type;
};

// Cycles 5.2.1 LightPathNode::compile emission order. This is host graph
// metaprogramming only: each live socket still emits one runtime SVM record.
constexpr auto light_path_outputs = std::array{
    LightPathOutput{"Is Camera Ray", NODE_LP_camera},
    LightPathOutput{"Is Shadow Ray", NODE_LP_shadow},
    LightPathOutput{"Is Diffuse Ray", NODE_LP_diffuse},
    LightPathOutput{"Is Glossy Ray", NODE_LP_glossy},
    LightPathOutput{"Is Singular Ray", NODE_LP_singular},
    LightPathOutput{"Is Reflection Ray", NODE_LP_reflection},
    LightPathOutput{"Is Transmission Ray", NODE_LP_transmission},
    LightPathOutput{"Is Volume Scatter Ray", NODE_LP_volume_scatter},
    LightPathOutput{"Ray Length", NODE_LP_ray_length},
    LightPathOutput{"Ray Depth", NODE_LP_ray_depth},
    LightPathOutput{"Diffuse Depth", NODE_LP_ray_diffuse},
    LightPathOutput{"Glossy Depth", NODE_LP_ray_glossy},
    LightPathOutput{"Transparent Depth", NODE_LP_ray_transparent},
    LightPathOutput{"Transmission Depth", NODE_LP_ray_transmission},
    LightPathOutput{"Portal Depth", NODE_LP_ray_portal},
};

class LightPathNode final : public GraphNode {
public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_light_path;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_LIGHT_PATH;
  }

  void compile(SVMCompiler &compiler) override {
    for (const auto &[name, path_type] : light_path_outputs) {
      auto *shader_output = output(name);
      if (shader_output != nullptr && !shader_output->links.empty()) {
        compiler.add_node(this, NODE_LIGHT_PATH,
                          SVMNodeLightPath{.path_type = path_type,
                                           .out_offset = compiler.output(name),
                                           ._pad = {0u, 0u, 0u}});
      }
    }
  }
};

} // namespace

std::unique_ptr<GraphNode> make_light_path_graph_node(std::string_view type) {
  if (type == node_type::light_path) {
    return std::make_unique<LightPathNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
