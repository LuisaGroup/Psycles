/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_vector_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <array>
#include <optional>
#include <string>
#include <utility>
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

[[nodiscard]] std::optional<NodeVectorRotateType>
vector_rotate_type(const GraphNode *node) noexcept {
  const auto type = string_property(node, "Type");
  if (type == "AXIS_ANGLE") {
    return NODE_VECTOR_ROTATE_TYPE_AXIS;
  }
  if (type == "X_AXIS") {
    return NODE_VECTOR_ROTATE_TYPE_AXIS_X;
  }
  if (type == "Y_AXIS") {
    return NODE_VECTOR_ROTATE_TYPE_AXIS_Y;
  }
  if (type == "Z_AXIS") {
    return NODE_VECTOR_ROTATE_TYPE_AXIS_Z;
  }
  if (type == "EULER_XYZ") {
    return NODE_VECTOR_ROTATE_TYPE_EULER_XYZ;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeVectorMathType>
vector_math_type(const GraphNode *node) noexcept {
  // This table is the Blender operation spelling -> Cycles 5.2.1
  // NodeVectorMathType ABI. Keep it complete and in the exact enum order so a
  // newly introduced operation cannot silently acquire different semantics.
  static constexpr std::array<std::pair<std::string_view, NodeVectorMathType>,
                              30u>
      operations{{
          {"ADD", NODE_VECTOR_MATH_ADD},
          {"SUBTRACT", NODE_VECTOR_MATH_SUBTRACT},
          {"MULTIPLY", NODE_VECTOR_MATH_MULTIPLY},
          {"DIVIDE", NODE_VECTOR_MATH_DIVIDE},
          {"CROSS_PRODUCT", NODE_VECTOR_MATH_CROSS_PRODUCT},
          {"PROJECT", NODE_VECTOR_MATH_PROJECT},
          {"REFLECT", NODE_VECTOR_MATH_REFLECT},
          {"DOT_PRODUCT", NODE_VECTOR_MATH_DOT_PRODUCT},
          {"DISTANCE", NODE_VECTOR_MATH_DISTANCE},
          {"LENGTH", NODE_VECTOR_MATH_LENGTH},
          {"SCALE", NODE_VECTOR_MATH_SCALE},
          {"NORMALIZE", NODE_VECTOR_MATH_NORMALIZE},
          {"SNAP", NODE_VECTOR_MATH_SNAP},
          {"FLOOR", NODE_VECTOR_MATH_FLOOR},
          {"CEIL", NODE_VECTOR_MATH_CEIL},
          {"MODULO", NODE_VECTOR_MATH_MODULO},
          {"FRACTION", NODE_VECTOR_MATH_FRACTION},
          {"ABSOLUTE", NODE_VECTOR_MATH_ABSOLUTE},
          {"MINIMUM", NODE_VECTOR_MATH_MINIMUM},
          {"MAXIMUM", NODE_VECTOR_MATH_MAXIMUM},
          {"WRAP", NODE_VECTOR_MATH_WRAP},
          {"SINE", NODE_VECTOR_MATH_SINE},
          {"COSINE", NODE_VECTOR_MATH_COSINE},
          {"TANGENT", NODE_VECTOR_MATH_TANGENT},
          {"REFRACT", NODE_VECTOR_MATH_REFRACT},
          {"FACEFORWARD", NODE_VECTOR_MATH_FACEFORWARD},
          {"MULTIPLY_ADD", NODE_VECTOR_MATH_MULTIPLY_ADD},
          {"POWER", NODE_VECTOR_MATH_POWER},
          {"SIGN", NODE_VECTOR_MATH_SIGN},
          {"ROUND", NODE_VECTOR_MATH_ROUND},
      }};
  static_assert(operations.front().second == NODE_VECTOR_MATH_ADD);
  static_assert(operations.back().second == NODE_VECTOR_MATH_ROUND);

  const auto operation = string_property(node, "Operation");
  if (!operation) {
    return std::nullopt;
  }
  for (const auto &[name, type] : operations) {
    if (*operation == name) {
      return type;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeVectorTransformType>
vector_transform_type(const GraphNode *node) noexcept {
  const auto type = string_property(node, "Type");
  if (type == "VECTOR") {
    return NODE_VECTOR_TRANSFORM_TYPE_VECTOR;
  }
  if (type == "POINT") {
    return NODE_VECTOR_TRANSFORM_TYPE_POINT;
  }
  if (type == "NORMAL") {
    return NODE_VECTOR_TRANSFORM_TYPE_NORMAL;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeVectorTransformConvertSpace>
vector_transform_space(const GraphNode *node,
                       std::string_view property) noexcept {
  const auto space = string_property(node, property);
  if (space == "WORLD") {
    return NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD;
  }
  if (space == "OBJECT") {
    return NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT;
  }
  if (space == "CAMERA") {
    return NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA;
  }
  return std::nullopt;
}

class VectorRotateNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    const auto rotate_type = vector_rotate_type(this);
    const auto invert = boolean_property(this, "Invert");
    if (!rotate_type || !invert) {
      compiler.fail("Cycles Vector Rotate properties are invalid");
      return;
    }
    compiler.add_node(
        this, NODE_VECTOR_ROTATE,
        SVMNodeVectorRotate{.rotate_type = *rotate_type,
                            .vector = compiler.input_float3("Vector"),
                            .center = compiler.input_float3("Center"),
                            .axis = compiler.input_float3("Axis"),
                            .rotation = compiler.input_float3("Rotation"),
                            .angle = compiler.input_float("Angle"),
                            .invert = static_cast<std::uint8_t>(*invert),
                            .result_offset = compiler.output("Vector"),
                            ._pad = {0u, 0u}});
  }
};

class VectorMathNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    const auto operation = vector_math_type(this);
    if (!operation) {
      compiler.fail("Cycles Vector Math Operation is invalid");
      return;
    }
    compiler.add_node(
        this, NODE_VECTOR_MATH,
        SVMNodeVectorMath{.math_type = *operation,
                          .a = compiler.input_float3("Vector1"),
                          .b = compiler.input_float3("Vector2"),
                          .c = compiler.input_float3("Vector3"),
                          .param1 = compiler.input_float("Scale"),
                          .value_offset = compiler.output("Value"),
                          .vector_offset = compiler.output("Vector"),
                          ._pad = {0u, 0u}});
  }
};

class VectorTransformNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    const auto transform_type = vector_transform_type(this);
    const auto convert_from = vector_transform_space(this, "Convert From");
    const auto convert_to = vector_transform_space(this, "Convert To");
    if (!transform_type || !convert_from || !convert_to) {
      compiler.fail("Cycles Vector Transform properties are invalid");
      return;
    }
    compiler.add_node(
        this, NODE_VECTOR_TRANSFORM,
        SVMNodeVectorTransform{.transform_type = *transform_type,
                               .convert_from = *convert_from,
                               .convert_to = *convert_to,
                               .vector_in = compiler.input_float3("Vector"),
                               .vector_out_offset = compiler.output("Vector"),
                               ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_vector_graph_node(std::string_view type) {
  if (type == node_type::vector_math) {
    return std::make_unique<VectorMathNode>();
  }
  if (type == node_type::vector_rotate) {
    return std::make_unique<VectorRotateNode>();
  }
  if (type == node_type::vector_transform) {
    return std::make_unique<VectorTransformNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
