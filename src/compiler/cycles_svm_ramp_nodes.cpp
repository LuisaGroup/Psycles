/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_ramp_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
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

[[nodiscard]] bool parse_row(std::string_view row,
                             packed_float4 &value) noexcept {
  float components[5u]{};
  for (auto index = std::size_t{}; index < 5u; ++index) {
    const auto delimiter = row.find(',');
    const auto token =
        delimiter == std::string_view::npos ? row : row.substr(0u, delimiter);
    if (!parse_float(token, components[index])) {
      return false;
    }
    if (index == 4u) {
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
  value = {components[1u], components[2u], components[3u], components[4u]};
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
    packed_float4 value{};
    if (!parse_row(row, value)) {
      return std::nullopt;
    }
    table.values.emplace_back(value);
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

[[nodiscard]] std::optional<float>
factor_literal(const GraphInput *input) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] packed_float4 rgb_ramp_lookup(const RGBRampTable &table,
                                            float factor) noexcept {
  const auto last = table.values.size() - 1u;
  const auto scaled = std::clamp(factor, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index =
      std::clamp(static_cast<int>(scaled), 0, static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto result = table.values[static_cast<std::size_t>(index)];
  if (table.interpolate && t > 0.0f) {
    const auto &next = table.values[static_cast<std::size_t>(index) + 1u];
    result.x = (1.0f - t) * result.x + t * next.x;
    result.y = (1.0f - t) * result.y + t * next.y;
    result.z = (1.0f - t) * result.z + t * next.z;
    result.w = (1.0f - t) * result.w + t * next.w;
  }
  return result;
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
    const auto factor = factor_literal(input("Factor"));
    if (!table || !factor) {
      return;
    }
    const auto value = rgb_ramp_lookup(*table, *factor);
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

} // namespace

std::unique_ptr<GraphNode> make_ramp_graph_node(std::string_view type) {
  if (type == node_type::color_ramp) {
    return std::make_unique<RGBRampNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
