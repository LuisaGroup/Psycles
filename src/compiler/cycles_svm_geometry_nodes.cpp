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

class GeometryNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_GEOMETRY;
  }

  void compile(SVMCompiler &compiler) override {
    const auto bump_offset = node_bump_offset(bump);
    const auto use_derivative =
        need_derivatives || bump != SHADER_BUMP_NONE;
    const auto store_derivatives =
        static_cast<std::uint8_t>(need_derivatives);

    auto *out = output("Position");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_P,
                          .bump_offset = bump_offset,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("Position"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    }

    // Cycles does not support a finite-difference bump offset for Normal,
    // Tangent, True Normal, or Incoming.
    out = output("Normal");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_N,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("Normal"),
                          .bump_filter_width = bump_filter_width});
    }

    out = output("Tangent");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_T,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("Tangent"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    }

    out = output("True Normal");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_Ng,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("True Normal"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    }

    out = output("Incoming");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_I,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("Incoming"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    }

    out = output("Parametric");
    if (out != nullptr && !out->links.empty()) {
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = NODE_GEOM_uv,
                          .bump_offset = bump_offset,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("Parametric"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    }

    if (auto *socket = output("Backfacing");
        socket != nullptr && !socket->links.empty()) {
      compiler.add_node(
          this, NODE_LIGHT_PATH,
          SVMNodeLightPath{.path_type = NODE_LP_backfacing,
                           .out_offset = compiler.output("Backfacing"),
                           ._pad = {0u, 0u, 0u}});
    }
    for (const auto name : {"Pointiness", "Random Per Island"}) {
      if (const auto *socket = output(name);
          socket != nullptr && !socket->links.empty()) {
        compiler.fail("Cycles Geometry output is not migrated: " +
                      std::string{name});
        return;
      }
    }
  }
};

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
  if (type == node_type::geometry) {
    return std::make_unique<GeometryNode>();
  }
  if (type == node_type::bump) {
    return std::make_unique<BumpNode>();
  }
  if (type == node_type::wireframe) {
    return std::make_unique<WireframeNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
