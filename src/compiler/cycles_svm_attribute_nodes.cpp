/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_attribute_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <array>
#include <bit>
#include <optional>
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

[[nodiscard]] NodeBumpOffset
node_bump_offset(ShaderBump bump) noexcept {
  switch (bump) {
  case SHADER_BUMP_DX:
    return NODE_BUMP_OFFSET_DX;
  case SHADER_BUMP_DY:
    return NODE_BUMP_OFFSET_DY;
  default:
    return NODE_BUMP_OFFSET_CENTER;
  }
}

[[nodiscard]] bool output_is_live(const GraphNode *node,
                                  std::string_view name) noexcept {
  const auto *out = node->output(name);
  return out != nullptr && !out->links.empty();
}

void add_named_attribute_request(AttributeRequestSet &requests,
                                 std::string_view attribute) {
  requests.add_standard(attribute);

  const auto standard = attribute_standard_from_name(attribute);
  if (standard == ATTR_STD_UV_TANGENT ||
      standard == ATTR_STD_UV_TANGENT_SIGN ||
      standard == ATTR_STD_UV_TANGENT_UNDISPLACED ||
      standard == ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED) {
    requests.add(ATTR_STD_UV);
    return;
  }

  static constexpr auto suffixes = std::array{
      std::string_view{".tangent_sign"}, std::string_view{".tangent"},
      std::string_view{".undisplaced_tangent"},
      std::string_view{".undisplaced_tangent_sign"},
  };
  for (const auto suffix : suffixes) {
    if (attribute.ends_with(suffix)) {
      requests.add(attribute.substr(0u, attribute.size() - suffix.size()));
    }
  }
}

class VertexColorNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_VERTEX_COLOR;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    if (output_is_live(this, "Color") || output_is_live(this, "Alpha")) {
      const auto layer_name = string_property(this, "Layer Name");
      if (layer_name && !layer_name->empty()) {
        requests.add_standard(*layer_name);
      } else if (layer_name) {
        requests.add(ATTR_STD_VERTEX_COLOR);
      }
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    const auto layer_name = string_property(this, "Layer Name");
    if (!layer_name) {
      compiler.fail("Cycles Vertex Color Layer Name property is invalid");
      return;
    }
    const auto layer_id =
        !layer_name->empty()
            ? compiler.attribute(*layer_name)
            : compiler.attribute(ATTR_STD_VERTEX_COLOR);
    compiler.add_node(
        this, NODE_VERTEX_COLOR,
        SVMNodeVertexColor{
            .layer_id = static_cast<std::uint8_t>(layer_id),
            .color_offset = compiler.output("Color"),
            .alpha_offset = compiler.output("Alpha"),
            .bump_offset = node_bump_offset(bump),
            .bump_filter_width = bump_filter_width,
        });
  }
};

class AttributeNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_ATTR;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    const auto attribute_name = string_property(this, "Attribute");
    if (attribute_name &&
        (output_is_live(this, "Color") || output_is_live(this, "Vector") ||
         output_is_live(this, "Fac") || output_is_live(this, "Alpha"))) {
      add_named_attribute_request(requests, *attribute_name);
    }
    if (context.has_volume) {
      requests.add(ATTR_STD_GENERATED_TRANSFORM);
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    const auto attribute_name = string_property(this, "Attribute");
    const auto stochastic_sample = boolean_property(this, "Stochastic");
    if (!attribute_name || !stochastic_sample) {
      compiler.fail("Cycles Attribute properties are invalid");
      return;
    }

    const auto bump_offset = node_bump_offset(bump);
    const auto use_derivative =
        need_derivatives || bump != SHADER_BUMP_NONE;
    const auto store_derivatives =
        static_cast<std::uint8_t>(need_derivatives);
    const auto attr = compiler.attribute_standard(*attribute_name);
    const auto bump_filter_or_stochastic =
        compiler.output_type() == SHADER_TYPE_VOLUME
            ? std::bit_cast<float>(
                  static_cast<std::uint32_t>(*stochastic_sample))
            : bump_filter_width;

    auto *color_out = output("Color");
    auto *vector_out = output("Vector");
    auto *fac_out = output("Fac");
    auto *alpha_out = output("Alpha");

    if (color_out != nullptr && !color_out->links.empty()) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(attr),
              .out_offset = compiler.output("Color"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT3,
              .bump_offset = bump_offset,
              .store_derivatives = store_derivatives,
              .bump_filter_width = bump_filter_or_stochastic,
          },
          use_derivative);
    }
    if (vector_out != nullptr && !vector_out->links.empty()) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(attr),
              .out_offset = compiler.output("Vector"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT3,
              .bump_offset = bump_offset,
              .store_derivatives = store_derivatives,
              .bump_filter_width = bump_filter_or_stochastic,
          },
          use_derivative);
    }
    if (fac_out != nullptr && !fac_out->links.empty()) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(attr),
              .out_offset = compiler.output("Fac"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT,
              .bump_offset = bump_offset,
              .store_derivatives = store_derivatives,
              .bump_filter_width = bump_filter_or_stochastic,
          },
          use_derivative);
    }
    if (alpha_out != nullptr && !alpha_out->links.empty()) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(attr),
              .out_offset = compiler.output("Alpha"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT_ALPHA,
              .bump_offset = bump_offset,
              .store_derivatives = store_derivatives,
              .bump_filter_width = bump_filter_or_stochastic,
          },
          use_derivative);
    }
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_attribute_graph_node(std::string_view type) {
  if (type == node_type::vertex_color) {
    return std::make_unique<VertexColorNode>();
  }
  if (type == node_type::attribute) {
    return std::make_unique<AttributeNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
