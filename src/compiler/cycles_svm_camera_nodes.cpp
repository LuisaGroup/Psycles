/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_camera_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

namespace psycles::compiler::cycles_svm {
namespace {

class CameraNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CAMERA;
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_CAMERA,
        SVMNodeCamera{.vector_offset = compiler.output("View Vector"),
                      .zdepth_offset = compiler.output("View Z Depth"),
                      .distance_offset = compiler.output("View Distance"),
                      ._pad = {0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_camera_graph_node(std::string_view type) {
  if (type == node_type::camera_data) {
    return std::make_unique<CameraNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
