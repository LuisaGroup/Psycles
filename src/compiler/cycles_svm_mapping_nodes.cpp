/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_mapping_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

using Transform = PackedTransform;

[[nodiscard]] Vec3f add(Vec3f lhs, Vec3f rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] Vec3f subtract(Vec3f lhs, Vec3f rhs) noexcept {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

[[nodiscard]] Vec3f multiply(Vec3f lhs, Vec3f rhs) noexcept {
  return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

[[nodiscard]] Vec3f divide(Vec3f value, float divisor) noexcept {
  const auto inverse = 1.0f / divisor;
  return {value.x * inverse, value.y * inverse, value.z * inverse};
}

[[nodiscard]] Vec3f safe_divide(Vec3f lhs, Vec3f rhs) noexcept {
  return {rhs.x != 0.0f ? lhs.x / rhs.x : 0.0f,
          rhs.y != 0.0f ? lhs.y / rhs.y : 0.0f,
          rhs.z != 0.0f ? lhs.z / rhs.z : 0.0f};
}

[[nodiscard]] float dot(Vec3f lhs, Vec3f rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] Vec3f cross(Vec3f lhs, Vec3f rhs) noexcept {
  return {lhs.y * rhs.z - lhs.z * rhs.y,
          lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}

[[nodiscard]] Vec3f safe_normalize(Vec3f value) noexcept {
  const auto length_squared = dot(value, value);
  return length_squared == 0.0f
             ? Vec3f{}
             : divide(value, std::sqrt(length_squared));
}

[[nodiscard]] Transform make_transform(float a, float b, float c, float d,
                                       float e, float f, float g, float h,
                                       float i, float j, float k,
                                       float l) noexcept {
  return {.x = {a, b, c, d},
          .y = {e, f, g, h},
          .z = {i, j, k, l}};
}

[[nodiscard]] float dot4(packed_float4 lhs, packed_float4 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

[[nodiscard]] Transform multiply(Transform lhs, Transform rhs) noexcept {
  const auto column_x =
      packed_float4{rhs.x.x, rhs.y.x, rhs.z.x, 0.0f};
  const auto column_y =
      packed_float4{rhs.x.y, rhs.y.y, rhs.z.y, 0.0f};
  const auto column_z =
      packed_float4{rhs.x.z, rhs.y.z, rhs.z.z, 0.0f};
  const auto column_w =
      packed_float4{rhs.x.w, rhs.y.w, rhs.z.w, 1.0f};
  return {.x = {dot4(lhs.x, column_x), dot4(lhs.x, column_y),
                dot4(lhs.x, column_z), dot4(lhs.x, column_w)},
          .y = {dot4(lhs.y, column_x), dot4(lhs.y, column_y),
                dot4(lhs.y, column_z), dot4(lhs.y, column_w)},
          .z = {dot4(lhs.z, column_x), dot4(lhs.z, column_y),
                dot4(lhs.z, column_z), dot4(lhs.z, column_w)}};
}

[[nodiscard]] Transform transform_scale(Vec3f scale) noexcept {
  return make_transform(scale.x, 0.0f, 0.0f, 0.0f, 0.0f, scale.y, 0.0f,
                        0.0f, 0.0f, 0.0f, scale.z, 0.0f);
}

[[nodiscard]] Transform transform_translate(Vec3f translation) noexcept {
  return make_transform(1.0f, 0.0f, 0.0f, translation.x, 0.0f, 1.0f, 0.0f,
                        translation.y, 0.0f, 0.0f, 1.0f, translation.z);
}

[[nodiscard]] Transform transform_rotate(float angle, Vec3f axis) noexcept {
  const auto sine = std::sin(angle);
  const auto cosine = std::cos(angle);
  const auto tangent = 1.0f - cosine;
  axis = safe_normalize(axis);
  return make_transform(
      axis.x * axis.x * tangent + cosine,
      axis.x * axis.y * tangent - sine * axis.z,
      axis.x * axis.z * tangent + sine * axis.y, 0.0f,
      axis.y * axis.x * tangent + sine * axis.z,
      axis.y * axis.y * tangent + cosine,
      axis.y * axis.z * tangent - sine * axis.x, 0.0f,
      axis.z * axis.x * tangent - sine * axis.y,
      axis.z * axis.y * tangent + sine * axis.x,
      axis.z * axis.z * tangent + cosine, 0.0f);
}

[[nodiscard]] Transform transform_euler(Vec3f rotation) noexcept {
  return multiply(
      multiply(transform_rotate(rotation.z, {0.0f, 0.0f, 1.0f}),
               transform_rotate(rotation.y, {0.0f, 1.0f, 0.0f})),
      transform_rotate(rotation.x, {1.0f, 0.0f, 0.0f}));
}

[[nodiscard]] Transform transform_inverse(Transform transform) noexcept {
  auto x = Vec3f{transform.x.x, transform.y.x, transform.z.x};
  auto y = Vec3f{transform.x.y, transform.y.y, transform.z.y};
  auto z = Vec3f{transform.x.z, transform.y.z, transform.z.z};
  const auto w = Vec3f{transform.x.w, transform.y.w, transform.z.w};

  auto determinant = dot(x, cross(y, z));
  if (determinant == 0.0f) {
    x.x += 1.0e-8f;
    y.y += 1.0e-8f;
    z.z += 1.0e-8f;
    determinant = dot(x, cross(y, z));
    if (determinant == 0.0f) {
      determinant = std::numeric_limits<float>::max();
    }
  }

  const auto inverse_x = divide(cross(y, z), determinant);
  const auto inverse_y = divide(cross(z, x), determinant);
  const auto inverse_z = divide(cross(x, y), determinant);
  return {.x = {inverse_x.x, inverse_x.y, inverse_x.z,
                -dot(inverse_x, w)},
          .y = {inverse_y.x, inverse_y.y, inverse_y.z,
                -dot(inverse_y, w)},
          .z = {inverse_z.x, inverse_z.y, inverse_z.z,
                -dot(inverse_z, w)}};
}

[[nodiscard]] Transform transform_transposed_inverse(
    Transform transform) noexcept {
  const auto inverse = transform_inverse(transform);
  return make_transform(inverse.x.x, inverse.y.x, inverse.z.x, 0.0f,
                        inverse.x.y, inverse.y.y, inverse.z.y, 0.0f,
                        inverse.x.z, inverse.y.z, inverse.z.z, 0.0f);
}

[[nodiscard]] Vec3f transform_direction(Transform transform,
                                        Vec3f vector) noexcept {
  return {vector.x * transform.x.x + vector.y * transform.x.y +
              vector.z * transform.x.z,
          vector.x * transform.y.x + vector.y * transform.y.y +
              vector.z * transform.y.z,
          vector.x * transform.z.x + vector.y * transform.z.y +
              vector.z * transform.z.z};
}

[[nodiscard]] Vec3f transform_direction_transposed(
    Transform transform, Vec3f vector) noexcept {
  return {vector.x * transform.x.x + vector.y * transform.y.x +
              vector.z * transform.z.x,
          vector.x * transform.x.y + vector.y * transform.y.y +
              vector.z * transform.z.y,
          vector.x * transform.x.z + vector.y * transform.y.z +
              vector.z * transform.z.z};
}

[[nodiscard]] Vec3f svm_mapping(NodeMappingType type, Vec3f vector,
                                Vec3f location, Vec3f rotation,
                                Vec3f scale) noexcept {
  const auto rotation_transform = transform_euler(rotation);
  switch (type) {
    case NODE_MAPPING_TYPE_POINT:
      return add(transform_direction(rotation_transform,
                                     multiply(vector, scale)),
                 location);
    case NODE_MAPPING_TYPE_TEXTURE:
      return safe_divide(transform_direction_transposed(
                             rotation_transform, subtract(vector, location)),
                         scale);
    case NODE_MAPPING_TYPE_VECTOR:
      return transform_direction(rotation_transform, multiply(vector, scale));
    case NODE_MAPPING_TYPE_NORMAL:
      return safe_normalize(transform_direction(
          rotation_transform, safe_divide(vector, scale)));
  }
  return {};
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

[[nodiscard]] std::optional<Vec3f>
float3_value(const GraphInput *input, bool require_unlinked = true) noexcept {
  if (input == nullptr || (require_unlinked && input->link != nullptr) ||
      !input->value) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<Vec3f>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeMappingType>
mapping_type(std::string_view name) noexcept {
  if (name == "POINT") {
    return NODE_MAPPING_TYPE_POINT;
  }
  if (name == "TEXTURE") {
    return NODE_MAPPING_TYPE_TEXTURE;
  }
  if (name == "VECTOR") {
    return NODE_MAPPING_TYPE_VECTOR;
  }
  if (name == "NORMAL") {
    return NODE_MAPPING_TYPE_NORMAL;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<TextureMappingAxis>
mapping_axis(std::string_view name) noexcept {
  if (name == "NONE") {
    return TextureMappingAxis::none;
  }
  if (name == "X") {
    return TextureMappingAxis::x;
  }
  if (name == "Y") {
    return TextureMappingAxis::y;
  }
  if (name == "Z") {
    return TextureMappingAxis::z;
  }
  return std::nullopt;
}

[[nodiscard]] Transform compute_transform(
    const TextureMapping &mapping) noexcept {
  auto axis_transform = transform_scale({0.0f, 0.0f, 0.0f});
  const auto set_axis = [](packed_float4 &row, TextureMappingAxis axis) {
    switch (axis) {
      case TextureMappingAxis::none:
        break;
      case TextureMappingAxis::x:
        row.x = 1.0f;
        break;
      case TextureMappingAxis::y:
        row.y = 1.0f;
        break;
      case TextureMappingAxis::z:
        row.z = 1.0f;
        break;
    }
  };
  set_axis(axis_transform.x, mapping.x_mapping);
  set_axis(axis_transform.y, mapping.y_mapping);
  set_axis(axis_transform.z, mapping.z_mapping);

  auto scale = mapping.scale;
  if (mapping.type == NODE_MAPPING_TYPE_TEXTURE ||
      mapping.type == NODE_MAPPING_TYPE_NORMAL) {
    const auto clamp_scale = [](float value) {
      return std::abs(value) < 1.0e-5f
                 ? (value < 0.0f ? -1.0e-5f : 1.0e-5f)
                 : value;
    };
    scale = {clamp_scale(scale.x), clamp_scale(scale.y),
             clamp_scale(scale.z)};
  }

  const auto scale_transform = transform_scale(scale);
  const auto rotation_transform = transform_euler(mapping.rotation);
  const auto translation_transform = transform_translate(mapping.translation);
  Transform result{};
  switch (mapping.type) {
    case NODE_MAPPING_TYPE_TEXTURE:
      result = transform_inverse(multiply(
          multiply(translation_transform, rotation_transform),
          scale_transform));
      break;
    case NODE_MAPPING_TYPE_POINT:
      result = multiply(multiply(translation_transform, rotation_transform),
                        scale_transform);
      break;
    case NODE_MAPPING_TYPE_VECTOR:
      result = multiply(rotation_transform, scale_transform);
      break;
    case NODE_MAPPING_TYPE_NORMAL:
      result = transform_transposed_inverse(
          multiply(rotation_transform, scale_transform));
      break;
  }
  return multiply(result, axis_transform);
}

class MappingNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_MAPPING;
  }

  void constant_fold(const ConstantFolder &folder) override {
    const auto type_name = string_property(this, "VectorType");
    const auto type = type_name ? mapping_type(*type_name) : std::nullopt;
    if (!type) {
      return;
    }
    if (folder.all_inputs_constant()) {
      const auto vector = float3_value(input("Vector"));
      const auto location = float3_value(input("Location"));
      const auto rotation = float3_value(input("Rotation"));
      const auto scale = float3_value(input("Scale"));
      if (vector && location && rotation && scale) {
        folder.make_constant(
            svm_mapping(*type, *vector, *location, *rotation, *scale));
      }
    } else {
      folder.fold_mapping(*type);
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto type_name = string_property(this, "VectorType");
    const auto type = type_name ? mapping_type(*type_name) : std::nullopt;
    if (!type) {
      compiler.fail("Cycles Mapping Type is invalid");
      return;
    }
    compiler.add_node(
        this, NODE_MAPPING,
        SVMNodeMapping{.mapping_type = *type,
                       .vector = compiler.input_float3("Vector"),
                       .location = compiler.input_float3("Location"),
                       .rotation = compiler.input_float3("Rotation"),
                       .scale = compiler.input_float3("Scale"),
                       .result_offset = compiler.output("Vector"),
                       ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

bool TextureMapping::skip() const noexcept {
  return translation == Vec3f{} && rotation == Vec3f{} &&
         scale == Vec3f{1.0f, 1.0f, 1.0f} &&
         x_mapping == TextureMappingAxis::x &&
         y_mapping == TextureMappingAxis::y &&
         z_mapping == TextureMappingAxis::z && !use_minmax;
}

void TextureMapping::compile(SVMCompiler &compiler,
                             SVMStackOffset offset_in,
                             SVMStackOffset offset_out,
                             GraphNode *node) const {
  compiler.add_node(
      node, NODE_TEXTURE_MAPPING,
      SVMNodeTextureMapping{.vec_offset = offset_in,
                            .out_offset = offset_out,
                            ._pad = {0u, 0u},
                            .tfm = compute_transform(*this)});
  if (use_minmax) {
    compiler.add_node(
        nullptr, NODE_MIN_MAX,
        SVMNodeMinMax{.vec_offset = offset_out,
                      .out_offset = offset_out,
                      ._pad = {0u, 0u},
                      .mn = {minimum.x, minimum.y, minimum.z},
                      .mx = {maximum.x, maximum.y, maximum.z}});
  }
  if (type == NODE_MAPPING_TYPE_NORMAL) {
    compiler.add_node(
        node, NODE_VECTOR_MATH,
        SVMNodeVectorMath{
            .math_type = NODE_VECTOR_MATH_NORMALIZE,
            .a = compiler.input_float3_from_offset(offset_out),
            .b = {},
            .c = {},
            .param1 = {},
            .value_offset = SVM_STACK_INVALID,
            .vector_offset = offset_out,
            ._pad = {0u, 0u}});
  }
}

SVMStackOffset TextureMapping::compile_begin(SVMCompiler &compiler,
                                             GraphInput *vector_in,
                                             GraphNode *node) const {
  if (!skip()) {
    const auto offset_in = compiler.stack_assign(vector_in);
    const auto offset_out = compiler.stack_find_offset(vector_in);
    compile(compiler, offset_in, offset_out, node);
    return offset_out;
  }
  return compiler.stack_assign(vector_in);
}

void TextureMapping::compile_end(SVMCompiler &compiler,
                                 GraphInput *vector_in,
                                 SVMStackOffset vector_offset) const {
  if (!skip()) {
    compiler.stack_clear_offset(vector_in, vector_offset);
  }
}

void TextureNode::copy_runtime_state_from(const GraphNode &other) {
  const auto *mapping = other.texture_mapping();
  if (mapping == nullptr) {
    std::abort();
  }
  tex_mapping = *mapping;
}

TextureMapping *TextureNode::texture_mapping() noexcept {
  return &tex_mapping;
}

const TextureMapping *TextureNode::texture_mapping() const noexcept {
  return &tex_mapping;
}

bool TextureNode::equals(const GraphNode &other) const noexcept {
  const auto *mapping = other.texture_mapping();
  return mapping != nullptr && tex_mapping == *mapping &&
         GraphNode::equals(other);
}

std::unique_ptr<GraphNode> make_mapping_graph_node(std::string_view type) {
  if (type == node_type::mapping) {
    return std::make_unique<MappingNode>();
  }
  return nullptr;
}

bool configure_texture_mapping_from_legacy_node(
    TextureMapping &mapping, const GraphNode &legacy_mapping,
    std::string &diagnostic) {
  const auto is_legacy =
      boolean_property(&legacy_mapping, "LegacyTextureMapping");
  const auto type_name = string_property(&legacy_mapping, "VectorType");
  const auto x_name = string_property(&legacy_mapping, "XMapping");
  const auto y_name = string_property(&legacy_mapping, "YMapping");
  const auto z_name = string_property(&legacy_mapping, "ZMapping");
  const auto type = type_name ? mapping_type(*type_name) : std::nullopt;
  const auto x = x_name ? mapping_axis(*x_name) : std::nullopt;
  const auto y = y_name ? mapping_axis(*y_name) : std::nullopt;
  const auto z = z_name ? mapping_axis(*z_name) : std::nullopt;
  const auto translation = float3_value(legacy_mapping.input("Location"));
  const auto rotation = float3_value(legacy_mapping.input("Rotation"));
  const auto scale = float3_value(legacy_mapping.input("Scale"));
  if (!is_legacy || !*is_legacy || !type || !x || !y || !z ||
      !translation || !rotation || !scale) {
    diagnostic = "Cycles legacy TextureMapping state is invalid";
    return false;
  }
  mapping.translation = *translation;
  mapping.rotation = *rotation;
  mapping.scale = *scale;
  mapping.type = *type;
  mapping.x_mapping = *x;
  mapping.y_mapping = *y;
  mapping.z_mapping = *z;
  return true;
}

} // namespace psycles::compiler::cycles_svm
