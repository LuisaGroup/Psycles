/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_vector_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <array>
#include <cmath>
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
private:
  // Host-only constant evaluation, ported from Cycles svm/math_util.h.
  // This is graph compilation, not a CPU shader/rendering oracle. Both result
  // slots start at zero, as in VectorMathNode::constant_fold (the inactive
  // output of a vector/scalar operation is therefore also well-defined).
  void fold_constants(const ConstantFolder &folder, NodeVectorMathType operation,
                      bool blender_inline = false) const {
    if (!folder.all_inputs_constant()) {
      return;
    }
    auto vector_input = [&](std::string_view name) {
      const auto *socket = input(name);
      return socket && socket->value ? std::get_if<Vec3f>(&socket->value->value)
                                     : nullptr;
    };
    const auto *ap = vector_input("Vector1");
    const auto *bp = vector_input("Vector2");
    const auto *cp = vector_input("Vector3");
    const auto *scale_input = input("Scale");
    const auto *sp = scale_input && scale_input->value
                         ? std::get_if<float>(&scale_input->value->value)
                         : nullptr;
    if (!ap || !bp || !cp || !sp) {
      return;
    }
    const auto a = *ap, b = *bp, c = *cp;
    const auto scale = *sp;
    const auto zip = [](Vec3f x, Vec3f y, auto op) -> Vec3f {
      return {op(x.x, y.x), op(x.y, y.y), op(x.z, y.z)};
    };
    const auto mul = [](Vec3f x, float y) -> Vec3f {
      return {x.x * y, x.y * y, x.z * y};
    };
    const auto dot = [](Vec3f x, Vec3f y) { return x.x * y.x + x.y * y.y + x.z * y.z; };
    const auto sub = [&](Vec3f x, Vec3f y) {
      return zip(x, y, [](float u, float v) { return u - v; });
    };
    const auto normalize = [&](Vec3f x) {
      const auto squared = dot(x, x);
      // Blender's multi-function has a real near-zero/NaN domain guard;
      // Cycles' later host fold uses safe_normalize's exact-zero guard.
      // Do not collapse those stages or apply this host-only rule in SVM.
      if (blender_inline && !(squared > 1.0e-35f)) {
        return Vec3f{};
      }
      const auto length = std::sqrt(squared);
      return length != 0.0f ? mul(x, 1.0f / length) : x;
    };
    const auto divide = [](float x, float y) { return y != 0.0f ? x / y : 0.0f; };
    float value = 0.0f;
    Vec3f vector{};
    switch (operation) {
    case NODE_VECTOR_MATH_ADD:
      vector = zip(a, b, [](float x, float y) { return x + y; });
      break;
    case NODE_VECTOR_MATH_SUBTRACT:
      vector = sub(a, b);
      break;
    case NODE_VECTOR_MATH_MULTIPLY:
      vector = zip(a, b, [](float x, float y) { return x * y; });
      break;
    case NODE_VECTOR_MATH_DIVIDE:
      vector = zip(a, b, divide);
      break;
    case NODE_VECTOR_MATH_CROSS_PRODUCT: {
      const auto cross_component = [&](float x, float y, float z, float w) {
        // Blender's host multi-function uses cross_high_precision. Exact
        // cancellation can otherwise remove a nonzero emission closure.
        // Device SVM and Cycles' later host fold retain ordinary float math.
        return blender_inline ? static_cast<float>(static_cast<double>(x) * y -
                                                   static_cast<double>(z) * w)
                              : x * y - z * w;
      };
      vector = {cross_component(a.y, b.z, a.z, b.y),
                cross_component(a.z, b.x, a.x, b.z),
                cross_component(a.x, b.y, a.y, b.x)};
      break;
    }
    case NODE_VECTOR_MATH_PROJECT: {
      // Blender tests the vector itself for zero, not its squared length.
      // Preserve the original non-finite result when that length underflows.
      const auto zero = blender_inline ? b == Vec3f{} : dot(b, b) == 0.0f;
      vector = zero ? Vec3f{} : mul(b, dot(a, b) / dot(b, b));
      break;
    }
    case NODE_VECTOR_MATH_REFLECT: {
      const auto n = normalize(b);
      vector = sub(a, mul(mul(n, 2.0f), dot(a, n)));
      break;
    }
    case NODE_VECTOR_MATH_REFRACT: {
      const auto n = normalize(b);
      const auto d = dot(n, a);
      const auto k = 1.0f - scale * scale * (1.0f - d * d);
      vector =
          k < 0.0f ? Vec3f{} : sub(mul(a, scale), mul(n, scale * d + std::sqrt(k)));
      break;
    }
    case NODE_VECTOR_MATH_FACEFORWARD:
      vector = dot(c, b) < 0.0f ? a : mul(a, -1.0f);
      break;
    case NODE_VECTOR_MATH_MULTIPLY_ADD:
      vector = {a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z};
      break;
    case NODE_VECTOR_MATH_DOT_PRODUCT:
      value = dot(a, b);
      break;
    case NODE_VECTOR_MATH_DISTANCE:
      value = std::sqrt(dot(sub(a, b), sub(a, b)));
      break;
    case NODE_VECTOR_MATH_LENGTH:
      value = std::sqrt(dot(a, a));
      break;
    case NODE_VECTOR_MATH_SCALE:
      vector = mul(a, scale);
      break;
    case NODE_VECTOR_MATH_NORMALIZE:
      vector = normalize(a);
      break;
    case NODE_VECTOR_MATH_SNAP:
      vector =
          zip(a, b, [&](float x, float y) { return std::floor(divide(x, y)) * y; });
      break;
    case NODE_VECTOR_MATH_ROUND:
      vector = zip(a, b, [](float x, float) { return std::floor(x + 0.5f); });
      break;
    case NODE_VECTOR_MATH_FLOOR:
      vector = zip(a, b, [](float x, float) { return std::floor(x); });
      break;
    case NODE_VECTOR_MATH_CEIL:
      vector = zip(a, b, [](float x, float) { return std::ceil(x); });
      break;
    case NODE_VECTOR_MATH_MODULO:
      vector = zip(a, b,
                   [](float x, float y) { return y != 0.0f ? std::fmod(x, y) : 0.0f; });
      break;
    case NODE_VECTOR_MATH_WRAP: {
      const auto wrap = [](float x, float hi, float lo) {
        const auto range = hi - lo;
        return range != 0.0f ? x - range * std::floor((x - lo) / range) : lo;
      };
      vector = {wrap(a.x, b.x, c.x), wrap(a.y, b.y, c.y), wrap(a.z, b.z, c.z)};
      break;
    }
    case NODE_VECTOR_MATH_FRACTION:
      vector = zip(a, b, [](float x, float) { return x - std::floor(x); });
      break;
    case NODE_VECTOR_MATH_ABSOLUTE:
      vector = zip(a, b, [](float x, float) { return std::fabs(x); });
      break;
    case NODE_VECTOR_MATH_POWER:
      vector = zip(
          a, b, [](float x, float y) { return svm_math(NODE_MATH_POWER, x, y, 0.0f); });
      break;
    case NODE_VECTOR_MATH_SIGN:
      vector = zip(a, b, [](float x, float) {
        return x > 0.0f ? 1.0f : x < 0.0f ? -1.0f : 0.0f;
      });
      break;
    case NODE_VECTOR_MATH_MINIMUM:
      vector = zip(a, b, [](float x, float y) { return std::fmin(x, y); });
      break;
    case NODE_VECTOR_MATH_MAXIMUM:
      vector = zip(a, b, [](float x, float y) { return std::fmax(x, y); });
      break;
    case NODE_VECTOR_MATH_SINE:
      vector = zip(a, b, [](float x, float) { return std::sin(x); });
      break;
    case NODE_VECTOR_MATH_COSINE:
      vector = zip(a, b, [](float x, float) { return std::cos(x); });
      break;
    case NODE_VECTOR_MATH_TANGENT:
      vector = zip(a, b, [](float x, float) { return std::tan(x); });
      break;
    }
    if (folder.output->name == "Value") {
      folder.make_constant(value);
    } else if (folder.output->name == "Vector") {
      folder.make_constant(vector);
    }
  }

public:
  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    if (const auto operation = vector_math_type(this)) {
      fold_constants(folder, *operation, true);
    }
  }

  void constant_fold(const ConstantFolder &folder) override {
    const auto operation = vector_math_type(this);
    if (!operation) {
      return;
    }
    if (folder.all_inputs_constant()) {
      fold_constants(folder, *operation);
    } else {
      folder.fold_vector_math(*operation);
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    // Cycles VectorMathNode::is_linear_operation: a constant divisor, or
    // at most one variable input for ADD/SUBTRACT/MULTIPLY/MULTIPLY_ADD.
    switch (vector_math_type(this).value_or(NODE_VECTOR_MATH_NORMALIZE)) {
    case NODE_VECTOR_MATH_ADD:
    case NODE_VECTOR_MATH_SUBTRACT:
    case NODE_VECTOR_MATH_MULTIPLY:
    case NODE_VECTOR_MATH_MULTIPLY_ADD:
      break;
    case NODE_VECTOR_MATH_DIVIDE:
      return input("Vector2")->link == nullptr;
    default:
      return false;
    }
    unsigned variable_inputs = 0;
    for (const auto &socket : inputs) {
      variable_inputs += socket.link != nullptr;
    }
    return variable_inputs <= 1;
  }

  void compile(SVMCompiler &compiler) override {
    const auto operation = vector_math_type(this);
    if (!operation) {
      compiler.fail("Cycles Vector Math Operation is invalid");
      return;
    }
    compiler.add_node(this, NODE_VECTOR_MATH,
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
