/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_ramp_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

struct RGBRampTable {
  std::vector<packed_float4> values;
  bool interpolate{};
};

struct Curve3Table {
  std::vector<packed_float4> values;
  float min_x{};
  float max_x{1.0f};
  bool extrapolate{true};
};

struct FloatCurveTable {
  std::vector<float> values;
  float min_x{};
  float max_x{1.0f};
  bool extrapolate{true};
};

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

[[nodiscard]] std::optional<float>
floating_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view trim_ascii(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t' ||
                            value.front() == '\r' || value.front() == '\n')) {
    value.remove_prefix(1u);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                            value.back() == '\r' || value.back() == '\n')) {
    value.remove_suffix(1u);
  }
  return value;
}

[[nodiscard]] bool parse_float(std::string_view text, float &value) noexcept {
  text = trim_ascii(text);
  if (text.empty()) {
    return false;
  }
  const auto *begin = text.data();
  const auto *end = begin + text.size();
  const auto result =
      std::from_chars(begin, end, value, std::chars_format::general);
  return result.ec == std::errc{} && result.ptr == end;
}

template<std::size_t N>
[[nodiscard]] bool parse_row(
    std::string_view row, std::array<float, N> &components) noexcept {
  for (auto index = std::size_t{}; index < N; ++index) {
    const auto delimiter = row.find(',');
    const auto token =
        delimiter == std::string_view::npos ? row : row.substr(0u, delimiter);
    if (!parse_float(token, components[index])) {
      return false;
    }
    if (index + 1u == N) {
      if (delimiter != std::string_view::npos) {
        return false;
      }
    } else {
      if (delimiter == std::string_view::npos) {
        return false;
      }
      row.remove_prefix(delimiter + 1u);
    }
  }
  return true;
}

[[nodiscard]] std::optional<RGBRampTable>
rgb_ramp_table(const GraphNode *node) {
  const auto sampled = boolean_property(node, "Sampled");
  const auto interpolation = string_property(node, "Interpolation");
  const auto table_text = string_property(node, "Table");
  if (!sampled || !*sampled || !interpolation || !table_text) {
    return std::nullopt;
  }
  const auto interpolation_valid =
      *interpolation == "LINEAR" || *interpolation == "CONSTANT" ||
      *interpolation == "EASE" || *interpolation == "CARDINAL" ||
      *interpolation == "B_SPLINE";
  if (!interpolation_valid) {
    return std::nullopt;
  }

  RGBRampTable table;
  table.interpolate = *interpolation != "CONSTANT";
  auto remaining = *table_text;
  while (!remaining.empty()) {
    const auto delimiter = remaining.find(';');
    const auto row = delimiter == std::string_view::npos
                         ? remaining
                         : remaining.substr(0u, delimiter);
    std::array<float, 5u> components{};
    if (!parse_row(row, components)) {
      return std::nullopt;
    }
    table.values.emplace_back(packed_float4{
        components[1u], components[2u], components[3u], components[4u]});
    if (delimiter == std::string_view::npos) {
      remaining = {};
    } else {
      remaining.remove_prefix(delimiter + 1u);
      if (remaining.empty()) {
        return std::nullopt;
      }
    }
  }
  if (table.values.empty() ||
      table.values.size() > static_cast<std::size_t>(
                                std::numeric_limits<std::int32_t>::max() / 4)) {
    return std::nullopt;
  }
  return table;
}

[[nodiscard]] std::optional<Curve3Table> curve3_table(const GraphNode *node) {
  const auto sampled = boolean_property(node, "Sampled");
  const auto min_x = floating_property(node, "MinX");
  const auto max_x = floating_property(node, "MaxX");
  const auto extrapolate = boolean_property(node, "Extrapolate");
  const auto table_text = string_property(node, "Table");
  if (!sampled || !*sampled || !min_x || !max_x || !extrapolate ||
      !table_text) {
    return std::nullopt;
  }

  Curve3Table table{.values = {},
                    .min_x = *min_x,
                    .max_x = *max_x,
                    .extrapolate = *extrapolate};
  auto remaining = *table_text;
  while (!remaining.empty()) {
    const auto delimiter = remaining.find(';');
    const auto row = delimiter == std::string_view::npos
                         ? remaining
                         : remaining.substr(0u, delimiter);
    std::array<float, 4u> components{};
    if (!parse_row(row, components)) {
      return std::nullopt;
    }
    // Cycles stores packed_float3 curve values as float4 table entries.
    // make_float4(packed_float3) supplies the otherwise unused w = 1.
    table.values.emplace_back(
        packed_float4{components[1u], components[2u], components[3u], 1.0f});
    if (delimiter == std::string_view::npos) {
      remaining = {};
    } else {
      remaining.remove_prefix(delimiter + 1u);
      if (remaining.empty()) {
        return std::nullopt;
      }
    }
  }
  if (table.values.size() < 2u ||
      table.values.size() > static_cast<std::size_t>(
                                std::numeric_limits<std::int32_t>::max() / 4)) {
    return std::nullopt;
  }
  return table;
}

[[nodiscard]] std::optional<FloatCurveTable>
float_curve_table(const GraphNode *node) {
  const auto sampled = boolean_property(node, "Sampled");
  const auto min_x = floating_property(node, "MinX");
  const auto max_x = floating_property(node, "MaxX");
  const auto extrapolate = boolean_property(node, "Extrapolate");
  const auto table_text = string_property(node, "Table");
  if (!sampled || !*sampled || !min_x || !max_x || !extrapolate ||
      !table_text) {
    return std::nullopt;
  }

  FloatCurveTable table{.values = {},
                        .min_x = *min_x,
                        .max_x = *max_x,
                        .extrapolate = *extrapolate};
  auto remaining = *table_text;
  while (!remaining.empty()) {
    const auto delimiter = remaining.find(';');
    const auto row = delimiter == std::string_view::npos
                         ? remaining
                         : remaining.substr(0u, delimiter);
    std::array<float, 2u> components{};
    if (!parse_row(row, components)) {
      return std::nullopt;
    }
    table.values.emplace_back(components[1u]);
    if (delimiter == std::string_view::npos) {
      remaining = {};
    } else {
      remaining.remove_prefix(delimiter + 1u);
      if (remaining.empty()) {
        return std::nullopt;
      }
    }
  }
  if (table.values.size() < 2u ||
      table.values.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  return table;
}

[[nodiscard]] std::optional<float>
float_literal(const GraphInput *input) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Vec3f>
float3_literal(const GraphInput *input, contract::SocketType type) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != type) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<Vec3f>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] packed_float4
rgb_ramp_lookup(const std::vector<packed_float4> &values, float factor,
                bool interpolate, bool extrapolate = false) noexcept {
  const auto last = values.size() - 1u;
  if ((factor < 0.0f || factor > 1.0f) && extrapolate) {
    packed_float4 value{};
    packed_float4 delta{};
    if (factor < 0.0f) {
      value = values.front();
      const auto &next = values[1u];
      delta = {value.x - next.x, value.y - next.y, value.z - next.z,
               value.w - next.w};
      factor = -factor;
    } else {
      value = values.back();
      const auto &previous = values[last - 1u];
      delta = {value.x - previous.x, value.y - previous.y,
               value.z - previous.z, value.w - previous.w};
      factor -= 1.0f;
    }
    const auto scale = factor * static_cast<float>(last);
    value.x += delta.x * scale;
    value.y += delta.y * scale;
    value.z += delta.z * scale;
    value.w += delta.w * scale;
    return value;
  }
  const auto scaled = std::clamp(factor, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index =
      std::clamp(static_cast<int>(scaled), 0, static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto result = values[static_cast<std::size_t>(index)];
  if (interpolate && t > 0.0f) {
    const auto &next = values[static_cast<std::size_t>(index) + 1u];
    result.x = (1.0f - t) * result.x + t * next.x;
    result.y = (1.0f - t) * result.y + t * next.y;
    result.z = (1.0f - t) * result.z + t * next.z;
    result.w = (1.0f - t) * result.w + t * next.w;
  }
  return result;
}

[[nodiscard]] float float_ramp_lookup(const std::vector<float> &values,
                                      float factor, bool interpolate,
                                      bool extrapolate) noexcept {
  const auto last = values.size() - 1u;
  if ((factor < 0.0f || factor > 1.0f) && extrapolate) {
    float value{};
    float delta{};
    if (factor < 0.0f) {
      value = values.front();
      delta = value - values[1u];
      factor = -factor;
    } else {
      value = values.back();
      delta = value - values[last - 1u];
      factor -= 1.0f;
    }
    return value + delta * factor * static_cast<float>(last);
  }
  const auto scaled = std::clamp(factor, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index =
      std::clamp(static_cast<int>(scaled), 0, static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto result = values[static_cast<std::size_t>(index)];
  if (interpolate && t > 0.0f) {
    result =
        (1.0f - t) * result + t * values[static_cast<std::size_t>(index) + 1u];
  }
  return result;
}

[[nodiscard]] std::optional<Vec3f>
fold_curve3(const GraphNode *node, std::string_view value_name,
            contract::SocketType value_type) noexcept {
  const auto table = curve3_table(node);
  const auto factor = float_literal(node->input("Factor"));
  const auto value = float3_literal(node->input(value_name), value_type);
  if (!table || !factor || !value) {
    return std::nullopt;
  }
  const auto range = table->max_x - table->min_x;
  const Vec3f relative{(value->x - table->min_x) / range,
                       (value->y - table->min_x) / range,
                       (value->z - table->min_x) / range};
  const auto red =
      rgb_ramp_lookup(table->values, relative.x, true, table->extrapolate).x;
  const auto green =
      rgb_ramp_lookup(table->values, relative.y, true, table->extrapolate).y;
  const auto blue =
      rgb_ramp_lookup(table->values, relative.z, true, table->extrapolate).z;
  return Vec3f{(1.0f - *factor) * value->x + *factor * red,
               (1.0f - *factor) * value->y + *factor * green,
               (1.0f - *factor) * value->z + *factor * blue};
}

[[nodiscard]] std::optional<float>
fold_float_curve(const GraphNode *node) noexcept {
  const auto table = float_curve_table(node);
  const auto factor = float_literal(node->input("Factor"));
  const auto value = float_literal(node->input("Value"));
  if (!table || !factor || !value) {
    return std::nullopt;
  }
  const auto relative = (*value - table->min_x) / (table->max_x - table->min_x);
  const auto result =
      float_ramp_lookup(table->values, relative, true, table->extrapolate);
  return *value + *factor * (result - *value);
}

class RGBRampNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_RGB_RAMP;
  }

  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto table = rgb_ramp_table(this);
    const auto factor = float_literal(input("Factor"));
    if (!table || !factor) {
      return;
    }
    const auto value = rgb_ramp_lookup(table->values, *factor,
                                       table->interpolate);
    if (folder.output->name == "Color") {
      folder.make_constant(Vec3f{value.x, value.y, value.z});
    } else if (folder.output->name == "Alpha") {
      folder.make_constant(value.w);
    }
  }

  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    // ShaderNodeValToRGB is a Blender multi-function. The serialized adapter
    // carries the same 257-entry table Cycles constructs immediately after
    // that stage, so its lookup preserves the fold topology while tolerating
    // only the tiny sampling error already permitted by the numeric oracle.
    constant_fold(folder);
  }

  void compile(SVMCompiler &compiler) override {
    const auto table = rgb_ramp_table(this);
    if (!table) {
      compiler.fail("Cycles RGB Ramp requires a valid sampled Blender table");
      return;
    }
    compiler.add_node(
        this, NODE_RGB_RAMP,
        SVMNodeRGBRamp{
            .table_size = static_cast<std::uint32_t>(table->values.size()),
            .fac = compiler.input_float("Factor"),
            .interpolate = static_cast<std::uint8_t>(table->interpolate),
            .color_offset = compiler.output("Color"),
            .alpha_offset = compiler.output("Alpha"),
            ._pad = {0u}});
    for (const auto &value : table->values) {
      compiler.add_node_data(value);
    }
  }
};

class CurvesNode : public GraphNode {
protected:
  void constant_fold_curve3(const ConstantFolder &folder,
                            std::string_view value_name,
                            contract::SocketType value_type) {
    if (folder.all_inputs_constant()) {
      if (const auto value = fold_curve3(this, value_name, value_type)) {
        folder.make_constant(*value);
      }
      return;
    }
    const auto factor = float_literal(input("Factor"));
    auto *value = input(value_name);
    if (factor && *factor == 0.0f && value != nullptr &&
        value->link != nullptr) {
      folder.bypass(value->link);
    }
  }

  void inline_blender_constant_fold_curve3(const ConstantFolder &folder,
                                           std::string_view value_name,
                                           contract::SocketType value_type) {
    // Blender's curve multi-functions fold only when both inputs are
    // primitive. Cycles' later zero-Factor bypass must not feed back into
    // this earlier graph stage.
    if (folder.all_inputs_constant()) {
      if (const auto value = fold_curve3(this, value_name, value_type)) {
        folder.make_constant(*value);
      }
    }
  }

  void compile_curve3(SVMCompiler &compiler, std::string_view value_name,
                      std::string_view label) {
    const auto table = curve3_table(this);
    if (!table) {
      compiler.fail(std::string{label} +
                    " requires a valid sampled Blender table");
      return;
    }
    compiler.add_node(
        this, NODE_CURVES,
        SVMNodeCurves{
            .color = compiler.input_float3(value_name),
            .fac = compiler.input_float("Factor"),
            .min_x = table->min_x,
            .max_x = table->max_x,
            .table_size = static_cast<std::uint32_t>(table->values.size()),
            .extrapolate = static_cast<std::uint8_t>(table->extrapolate),
            .out_offset = compiler.output(value_name),
            ._pad = {0u, 0u}});
    for (const auto &value : table->values) {
      compiler.add_node_data(value);
    }
  }
};

class RGBCurveNode final : public CurvesNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CURVES;
  }

  void constant_fold(const ConstantFolder &folder) override {
    constant_fold_curve3(folder, "Color", contract::SocketType::color);
  }

  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    inline_blender_constant_fold_curve3(folder, "Color",
                                        contract::SocketType::color);
  }

  void compile(SVMCompiler &compiler) override {
    compile_curve3(compiler, "Color", "Cycles RGB Curves");
  }
};

class VectorCurveNode final : public CurvesNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CURVES;
  }

  void constant_fold(const ConstantFolder &folder) override {
    constant_fold_curve3(folder, "Vector", contract::SocketType::vector);
  }

  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    inline_blender_constant_fold_curve3(folder, "Vector",
                                        contract::SocketType::vector);
  }

  void compile(SVMCompiler &compiler) override {
    compile_curve3(compiler, "Vector", "Cycles Vector Curves");
  }
};

class FloatCurveNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_FLOAT_CURVE;
  }

  void constant_fold(const ConstantFolder &folder) override {
    if (folder.all_inputs_constant()) {
      if (const auto value = fold_float_curve(this)) {
        folder.make_constant(*value);
      }
      return;
    }
    const auto factor = float_literal(input("Factor"));
    auto *value = input("Value");
    if (factor && *factor == 0.0f && value != nullptr &&
        value->link != nullptr) {
      folder.bypass(value->link);
    }
  }

  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    if (folder.all_inputs_constant()) {
      if (const auto value = fold_float_curve(this)) {
        folder.make_constant(*value);
      }
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto table = float_curve_table(this);
    if (!table) {
      compiler.fail(
          "Cycles Float Curve requires a valid sampled Blender table");
      return;
    }
    compiler.add_node(
        this, NODE_FLOAT_CURVE,
        SVMNodeFloatCurve{
            .fac = compiler.input_float("Factor"),
            .value_in = compiler.input_float("Value"),
            .min_x = table->min_x,
            .max_x = table->max_x,
            .table_size = static_cast<std::uint32_t>(table->values.size()),
            .extrapolate = static_cast<std::uint8_t>(table->extrapolate),
            .out_offset = compiler.output("Value"),
            ._pad = {0u, 0u}});
    for (const auto value : table->values) {
      compiler.add_node_data_float(value);
    }
  }
};

} // namespace

std::unique_ptr<GraphNode> make_ramp_graph_node(std::string_view type) {
  if (type == node_type::color_ramp) {
    return std::make_unique<RGBRampNode>();
  }
  if (type == node_type::rgb_curve) {
    return std::make_unique<RGBCurveNode>();
  }
  if (type == node_type::vector_curve) {
    return std::make_unique<VectorCurveNode>();
  }
  if (type == node_type::float_curve) {
    return std::make_unique<FloatCurveNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
