/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_geometry_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<bool>
boolean_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::boolean) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<bool>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<float>
float_socket_value(const GraphInput *input) noexcept {
  if (input == nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] NodeBumpOffset node_bump_offset(ShaderBump bump) noexcept {
  switch (bump) {
  case SHADER_BUMP_DX:
    return NODE_BUMP_OFFSET_DX;
  case SHADER_BUMP_DY:
    return NODE_BUMP_OFFSET_DY;
  default:
    return NODE_BUMP_OFFSET_CENTER;
  }
}

class BumpNode final : public GraphNode {
public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return kernel_feature_node_bump;
  }

  void compile(SVMCompiler &compiler) override {
    const auto invert = boolean_property(this, "Invert");
    const auto use_object_space = boolean_property(this, "UseObjectSpace");
    const auto filter_width = float_socket_value(input("Filter Width"));
    if (!invert || !use_object_space || !filter_width) {
      compiler.fail("Cycles Bump properties are invalid");
      return;
    }
    compiler.add_node(
        this, NODE_SET_BUMP,
        SVMNodeSetBump{.scale = compiler.input_float("Distance"),
                       .strength = compiler.input_float("Strength"),
                       .bump_filter_width = *filter_width,
                       .normal_offset = compiler.input_link("Normal"),
                       .invert = static_cast<std::uint8_t>(*invert),
                       .use_object_space =
                           static_cast<std::uint8_t>(*use_object_space),
                       .center_offset = compiler.input_link("SampleCenter"),
                       .dx_offset = compiler.input_link("SampleX"),
                       .dy_offset = compiler.input_link("SampleY"),
                       .out_offset = compiler.output("Normal"),
                       .bump_state_offset = compiler.get_bump_state_offset()});
  }

  void constant_fold(const ConstantFolder &folder) override {
    // Cycles BumpNode::constant_fold bypasses an unconnected Height to the
    // Normal input. CyclesGraph::default_inputs has already materialized the
    // LINK_NORMAL Geometry node before clean(), so every reachable projected
    // graph has a linked Normal at this point.
    auto *height = input("Height");
    auto *normal = input("Normal");
    if (height != nullptr && height->link == nullptr && normal != nullptr &&
        normal->link != nullptr) {
      folder.bypass(normal->link);
    }
  }
};

class WireframeNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    const auto use_pixel_size = boolean_property(this, "Use Pixel Size");
    if (!use_pixel_size) {
      compiler.fail("Cycles Wireframe Use Pixel Size property is invalid");
      return;
    }
    compiler.add_node(
        this, NODE_WIREFRAME,
        SVMNodeWireframe{.in_size = compiler.input_float("Size"),
                         .bump_filter_width = bump_filter_width,
                         .use_pixel_size =
                             static_cast<std::uint8_t>(*use_pixel_size),
                         .bump_offset = node_bump_offset(bump),
                         .out_fac_offset = compiler.output("Fac"),
                         ._pad = {0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_geometry_graph_node(std::string_view type) {
  if (type == node_type::bump) {
    return std::make_unique<BumpNode>();
  }
  if (type == node_type::wireframe) {
    return std::make_unique<WireframeNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
