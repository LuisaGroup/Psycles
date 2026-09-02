/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_texture_coordinate_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_transform.h>

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

[[nodiscard]] std::optional<Mat4f>
transform_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::transform) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<Mat4f>(&iter->second.value)) {
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

[[nodiscard]] PackedTransform make_transform(const Mat4f &matrix) noexcept {
  const auto &e = matrix.elements;
  return {
      .x = {e[0u], e[4u], e[8u], e[12u]},
      .y = {e[1u], e[5u], e[9u], e[13u]},
      .z = {e[2u], e[6u], e[10u], e[14u]},
  };
}

[[nodiscard]] bool output_is_live(const GraphNode *node,
                                  std::string_view name) noexcept {
  const auto *out = node->output(name);
  return out != nullptr && !out->links.empty();
}

class TextureCoordinateNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_COORD;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    const auto from_dupli = boolean_property(this, "FromDupli");
    if (from_dupli && !*from_dupli) {
      if (context.has_surface_link()) {
        if (output_is_live(this, "Generated")) {
          requests.add(ATTR_STD_GENERATED);
        }
        if (output_is_live(this, "UV")) {
          requests.add(ATTR_STD_UV);
        }
      }
      if (context.has_volume && output_is_live(this, "Generated")) {
        requests.add(ATTR_STD_GENERATED_TRANSFORM);
      }
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    const auto from_dupli = boolean_property(this, "FromDupli");
    const auto use_transform = boolean_property(this, "UseTransform");
    const auto object_transform = transform_property(this, "ObjectTransform");
    if (!from_dupli || !use_transform || !object_transform) {
      compiler.fail("Cycles Texture Coordinate properties are invalid");
      return;
    }

    const auto bump_offset = node_bump_offset(bump);
    const auto use_derivative = need_derivatives || bump != SHADER_BUMP_NONE;
    const auto store_derivatives = static_cast<std::uint8_t>(need_derivatives);
    auto emit_tex_coord = [&](std::string_view output_name,
                              NodeTexCoord texco_type,
                              NodeBumpOffset output_bump_offset) {
      compiler.add_node(
          this, NODE_TEX_COORD,
          SVMNodeTexCoord{.texco_type = texco_type,
                          .bump_offset = output_bump_offset,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output(output_name),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
    };

    auto *out = output("Generated");
    if (out != nullptr && !out->links.empty()) {
      if (compiler.background()) {
        compiler.add_node(
            this, NODE_GEOMETRY,
            SVMNodeGeometry{.geom_type = NODE_GEOM_P,
                            .bump_offset = bump_offset,
                            .store_derivatives = store_derivatives,
                            .out_offset = compiler.output("Generated"),
                            .bump_filter_width = bump_filter_width},
            use_derivative);
      } else if (*from_dupli) {
        emit_tex_coord("Generated", NODE_TEXCO_DUPLI_GENERATED,
                       NODE_BUMP_OFFSET_CENTER);
      } else if (compiler.output_type() == SHADER_TYPE_VOLUME) {
        emit_tex_coord("Generated", NODE_TEXCO_VOLUME_GENERATED,
                       bump_offset);
      } else {
        compiler.add_node(
            this, NODE_ATTR,
            SVMNodeAttr{
                .attr = static_cast<std::int32_t>(
                    compiler.attribute(ATTR_STD_GENERATED)),
                .out_offset = compiler.output("Generated"),
                .output_type = NODE_ATTR_OUTPUT_FLOAT3,
                .bump_offset = bump_offset,
                .store_derivatives = store_derivatives,
                .bump_filter_width = bump_filter_width},
            use_derivative);
      }
    }

    out = output("Normal");
    if (out != nullptr && !out->links.empty()) {
      emit_tex_coord("Normal", NODE_TEXCO_NORMAL, bump_offset);
    }

    out = output("UV");
    if (out != nullptr && !out->links.empty()) {
      if (*from_dupli) {
        emit_tex_coord("UV", NODE_TEXCO_DUPLI_UV,
                       NODE_BUMP_OFFSET_CENTER);
      } else {
        compiler.add_node(
            this, NODE_ATTR,
            SVMNodeAttr{
                .attr = static_cast<std::int32_t>(
                    compiler.attribute(ATTR_STD_UV)),
                .out_offset = compiler.output("UV"),
                .output_type = NODE_ATTR_OUTPUT_FLOAT3,
                .bump_offset = bump_offset,
                .store_derivatives = store_derivatives,
                .bump_filter_width = bump_filter_width},
            use_derivative);
      }
    }

    out = output("Object");
    if (out != nullptr && !out->links.empty()) {
      emit_tex_coord("Object",
                     *use_transform ? NODE_TEXCO_OBJECT_WITH_TRANSFORM
                                    : NODE_TEXCO_OBJECT,
                     bump_offset);
      if (*use_transform) {
        compiler.add_node_data(
            make_transform(cycles_inverse_affine_transform(*object_transform)));
      }
    }

    out = output("Camera");
    if (out != nullptr && !out->links.empty()) {
      emit_tex_coord("Camera", NODE_TEXCO_CAMERA, bump_offset);
    }

    out = output("Window");
    if (out != nullptr && !out->links.empty()) {
      emit_tex_coord("Window", NODE_TEXCO_WINDOW, bump_offset);
    }

    out = output("Reflection");
    if (out != nullptr && !out->links.empty()) {
      if (compiler.background()) {
        compiler.add_node(
            this, NODE_GEOMETRY,
            SVMNodeGeometry{.geom_type = NODE_GEOM_I,
                            .bump_offset = NODE_BUMP_OFFSET_CENTER,
                            .store_derivatives = store_derivatives,
                            .out_offset = compiler.output("Reflection"),
                            .bump_filter_width = bump_filter_width},
            use_derivative);
      } else {
        emit_tex_coord("Reflection", NODE_TEXCO_REFLECTION,
                       NODE_BUMP_OFFSET_CENTER);
      }
    }
  }
};

class UVMapNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_COORD;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    const auto from_dupli = boolean_property(this, "FromDupli");
    const auto attribute = string_property(this, "Attribute");
    if (context.has_surface && from_dupli && !*from_dupli && attribute &&
        output_is_live(this, "UV")) {
      if (attribute->empty()) {
        requests.add(ATTR_STD_UV);
      } else {
        requests.add(*attribute);
      }
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    const auto from_dupli = boolean_property(this, "FromDupli");
    const auto attribute = string_property(this, "Attribute");
    if (!from_dupli || !attribute) {
      compiler.fail("Cycles UV Map properties are invalid");
      return;
    }

    auto *out = output("UV");
    if (out == nullptr || out->links.empty()) {
      return;
    }

    const auto bump_offset = node_bump_offset(bump);
    const auto use_derivative = need_derivatives || bump != SHADER_BUMP_NONE;
    const auto store_derivatives = static_cast<std::uint8_t>(need_derivatives);
    if (*from_dupli) {
      compiler.add_node(
          this, NODE_TEX_COORD,
          SVMNodeTexCoord{.texco_type = NODE_TEXCO_DUPLI_UV,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives = store_derivatives,
                          .out_offset = compiler.output("UV"),
                          .bump_filter_width = bump_filter_width},
          use_derivative);
      return;
    }

    const auto attr = attribute->empty() ? compiler.attribute(ATTR_STD_UV)
                                         : compiler.attribute(*attribute);
    compiler.add_node(
        this, NODE_ATTR,
        SVMNodeAttr{.attr = static_cast<std::int32_t>(attr),
                    .out_offset = compiler.output("UV"),
                    .output_type = NODE_ATTR_OUTPUT_FLOAT3,
                    .bump_offset = bump_offset,
                    .store_derivatives = store_derivatives,
                    .bump_filter_width = bump_filter_width},
        use_derivative);
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_texture_coordinate_graph_node(std::string_view type) {
  if (type == node_type::texture_coordinate ||
      type == cycles_synthetic_texture_coordinate) {
    return std::make_unique<TextureCoordinateNode>();
  }
  if (type == node_type::uv_map) {
    return std::make_unique<UVMapNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
