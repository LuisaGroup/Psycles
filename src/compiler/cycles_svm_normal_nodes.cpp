/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_normal_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <string>
#include <string_view>
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

[[nodiscard]] std::optional<NodeNormalMapSpace>
normal_map_space(std::string_view value) noexcept {
  if (value == "TANGENT") { return NODE_NORMAL_MAP_TANGENT; }
  if (value == "OBJECT") { return NODE_NORMAL_MAP_OBJECT; }
  if (value == "WORLD") { return NODE_NORMAL_MAP_WORLD; }
  if (value == "BLENDER_OBJECT") { return NODE_NORMAL_MAP_BLENDER_OBJECT; }
  if (value == "BLENDER_WORLD") { return NODE_NORMAL_MAP_BLENDER_WORLD; }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeTangentDirectionType>
tangent_direction_type(std::string_view value) noexcept {
  if (value == "RADIAL") { return NODE_TANGENT_RADIAL; }
  if (value == "UV_MAP") { return NODE_TANGENT_UVMAP; }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeTangentAxis>
tangent_axis(std::string_view value) noexcept {
  if (value == "X") { return NODE_TANGENT_AXIS_X; }
  if (value == "Y") { return NODE_TANGENT_AXIS_Y; }
  if (value == "Z") { return NODE_TANGENT_AXIS_Z; }
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

class NormalMapNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_NORMAL_MAP;
  }

  void compile(SVMCompiler &compiler) override {
    const auto space = normal_map_space(
        string_property(this, "Space").value_or(std::string_view{}));
    const auto convention = string_property(this, "Convention");
    const auto base = string_property(this, "Base");
    const auto attribute = string_property(this, "Attribute");
    if (!space || !convention || !base || !attribute ||
        (*convention != "OPENGL" && *convention != "DIRECTX") ||
        (*base != "ORIGINAL" && *base != "DISPLACED")) {
      compiler.fail("Cycles Normal Map properties are invalid");
      return;
    }

    auto attr = std::uint32_t{};
    auto attr_sign = std::uint32_t{};
    if (*space == NODE_NORMAL_MAP_TANGENT) {
      const auto original = *base == "ORIGINAL";
      if (attribute->empty()) {
        attr = compiler.attribute(
            original ? ATTR_STD_UV_TANGENT_UNDISPLACED
                     : ATTR_STD_UV_TANGENT);
        attr_sign = compiler.attribute(
            original ? ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED
                     : ATTR_STD_UV_TANGENT_SIGN);
      } else {
        const auto suffix = original ? ".undisplaced_tangent" : ".tangent";
        const auto sign_suffix = original ? ".undisplaced_tangent_sign"
                                          : ".tangent_sign";
        attr = compiler.attribute(std::string{*attribute} + suffix);
        attr_sign = compiler.attribute(std::string{*attribute} + sign_suffix);
      }
    }

    compiler.add_node(
        this, NODE_NORMAL_MAP,
        SVMNodeNormalMap{
            .space = *space,
            .invert_green = *convention == "DIRECTX" ? 1 : 0,
            .use_original_base = *base == "ORIGINAL" ? 1 : 0,
            .attr = static_cast<int>(attr),
            .attr_sign = static_cast<int>(attr_sign),
            .color = compiler.input_float3("Color"),
            .strength = compiler.input_float("Strength"),
            .normal_offset = compiler.output("Normal"),
            ._pad = {0u, 0u, 0u}});
  }
};

class TangentNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TANGENT;
  }

  void compile(SVMCompiler &compiler) override {
    const auto direction_type = tangent_direction_type(
        string_property(this, "Direction Type").value_or(std::string_view{}));
    const auto axis = tangent_axis(
        string_property(this, "Axis").value_or(std::string_view{}));
    const auto attribute = string_property(this, "Attribute");
    if (!direction_type || !axis || !attribute) {
      compiler.fail("Cycles Tangent properties are invalid");
      return;
    }

    const auto attr =
        *direction_type == NODE_TANGENT_UVMAP
            ? (attribute->empty()
                   ? compiler.attribute(ATTR_STD_UV_TANGENT)
                   : compiler.attribute(std::string{*attribute} + ".tangent"))
            : compiler.attribute(ATTR_STD_GENERATED);
    compiler.add_node(
        this, NODE_TANGENT,
        SVMNodeTangent{.direction_type = *direction_type,
                       .axis = *axis,
                       .attr = static_cast<int>(attr),
                       .tangent_offset = compiler.output("Tangent"),
                       ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_normal_graph_node(std::string_view type) {
  if (type == node_type::normal) {
    return std::make_unique<NormalNode>();
  }
  if (type == node_type::normal_map) {
    return std::make_unique<NormalMapNode>();
  }
  if (type == node_type::tangent) {
    return std::make_unique<TangentNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
