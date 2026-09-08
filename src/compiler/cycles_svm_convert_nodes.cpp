/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_convert_nodes.h"
#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <type_traits>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

template<typename T>
[[nodiscard]] std::optional<T> literal(const GraphInput *input,
                                       contract::SocketType type) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != type) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<T>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

class ConvertNode final : public GraphNode {
private:
  [[nodiscard]] static bool is_float3(GraphSocketType type) noexcept {
    return type == GraphSocketType::color ||
           type == GraphSocketType::vector ||
           type == GraphSocketType::normal || type == GraphSocketType::point;
  }

  [[nodiscard]] static std::optional<NodeConvert>
  convert_type(GraphSocketType from, GraphSocketType to) noexcept {
    if (from == GraphSocketType::floating) {
      return to == GraphSocketType::integer ? NODE_CONVERT_FI
                                             : NODE_CONVERT_FV;
    }
    if (from == GraphSocketType::integer) {
      return to == GraphSocketType::floating ? NODE_CONVERT_IF
                                              : NODE_CONVERT_IV;
    }
    if (to == GraphSocketType::floating) {
      return from == GraphSocketType::color ? NODE_CONVERT_CF
                                             : NODE_CONVERT_VF;
    }
    if (to == GraphSocketType::integer) {
      return from == GraphSocketType::color ? NODE_CONVERT_CI
                                             : NODE_CONVERT_VI;
    }
    if (is_float3(from) && is_float3(to)) {
      return NODE_CONVERT_NONE;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool inverse(const GraphNode *previous) const noexcept {
    if (previous == nullptr || previous->shader_node_type() != NODE_CONVERT ||
        inputs.size() != 1u || outputs.size() != 1u ||
        previous->inputs.size() != 1u || previous->outputs.size() != 1u) {
      return false;
    }
    return is_float3(inputs.front().type) &&
           (outputs.front().type == GraphSocketType::floating ||
            is_float3(outputs.front().type)) &&
           previous->inputs.front().type == outputs.front().type &&
           previous->outputs.front().type == inputs.front().type;
  }

  [[nodiscard]] static std::optional<std::int32_t>
  integer_literal(const GraphInput *input) noexcept {
    if (input == nullptr || !input->value) {
      return std::nullopt;
    }
    return std::visit(
        [](const auto &item) -> std::optional<std::int32_t> {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, bool>) {
            return item ? 1 : 0;
          } else if constexpr (std::is_same_v<T, std::int64_t> ||
                               std::is_same_v<T, std::uint64_t>) {
            return static_cast<std::int32_t>(item);
          }
          return std::nullopt;
        },
        input->value->value);
  }

public:
  [[nodiscard]] bool equals(const GraphNode &other) const noexcept override {
    // Cycles registers a distinct NodeType for every ConvertNode(from, to).
    // Projection uses one factory tag, so the complete socket pair must still
    // participate in identity even when the SVM operations happen to match.
    return GraphNode::equals(other) && outputs.size() == 1u &&
           other.outputs.size() == 1u &&
           outputs.front().type == other.outputs.front().type;
  }

  void compile(SVMCompiler &compiler) override {
    auto *in = inputs.empty() ? nullptr : &inputs.front();
    auto *out = outputs.empty() ? nullptr : &outputs.front();
    if (in == nullptr || out == nullptr) {
      compiler.fail("Cycles Convert node sockets are absent");
      return;
    }
    const auto operation = convert_type(in->type, out->type);
    if (!operation) {
      compiler.fail("Cycles Convert node type pair is unsupported");
      return;
    }
    if (*operation != NODE_CONVERT_NONE) {
      compiler.add_node(
          this, NODE_CONVERT,
          SVMNodeConvert{.convert_type = *operation,
                         .from_offset = compiler.input_link(in->name),
                         .to_offset = compiler.output(out->name),
                         ._pad = {0u, 0u}});
      return;
    }
    if (in->link != nullptr) {
      compiler.stack_link(in, out);
      return;
    }
    const auto *value =
        in->value ? std::get_if<Vec3f>(&in->value->value) : nullptr;
    if (value == nullptr) {
      compiler.fail("Cycles float3 Convert node value is ill typed");
      return;
    }
    if (const auto offset = compiler.output(out->name);
        offset != SVM_STACK_INVALID) {
      compiler.add_value_node(this, *value, offset);
    }
  }

  void constant_fold(const ConstantFolder &folder) override {
    auto *in = inputs.empty() ? nullptr : &inputs.front();
    auto *out = outputs.empty() ? nullptr : &outputs.front();
    if (in == nullptr || out == nullptr) {
      return;
    }
    if (folder.all_inputs_constant()) {
      if (in->type == GraphSocketType::floating) {
        const auto value =
            literal<float>(in, contract::SocketType::floating);
        if (!value) {
          return;
        }
        if (out->type == GraphSocketType::integer) {
          folder.make_constant(static_cast<std::int32_t>(*value));
        } else if (is_float3(out->type)) {
          folder.make_constant(Vec3f{*value, *value, *value});
        }
      } else if (in->type == GraphSocketType::integer) {
        const auto value = integer_literal(in);
        if (!value) {
          return;
        }
        if (out->type == GraphSocketType::floating) {
          folder.make_constant(static_cast<float>(*value));
        } else if (is_float3(out->type)) {
          const auto scalar = static_cast<float>(*value);
          folder.make_constant(Vec3f{scalar, scalar, scalar});
        }
      } else if (is_float3(in->type)) {
        const auto *value =
            in->value ? std::get_if<Vec3f>(&in->value->value) : nullptr;
        if (value == nullptr) {
          return;
        }
        if (out->type == GraphSocketType::floating ||
            out->type == GraphSocketType::integer) {
          const auto scalar =
              in->type == GraphSocketType::color
                  ? folder.graph->linear_rgb_to_gray(*value)
                  : (value->x + value->y + value->z) / 3.0f;
          if (out->type == GraphSocketType::integer) {
            folder.make_constant(static_cast<std::int32_t>(scalar));
          } else {
            folder.make_constant(scalar);
          }
        } else if (is_float3(out->type)) {
          folder.make_constant(*value);
        }
      }
    } else if (in->link != nullptr && inverse(in->link->parent)) {
      auto *previous = in->link->parent;
      auto *previous_input = previous->inputs.empty()
                                 ? nullptr
                                 : &previous->inputs.front();
      if (previous_input != nullptr && previous_input->link != nullptr) {
        folder.bypass(previous_input->link);
      }
    }
  }

  void inline_blender_constant_fold(const ConstantFolder &folder) override {
    if (origin != contract::ShaderNodeOrigin::blender_implicit_conversion ||
        !folder.all_inputs_constant() || inputs.size() != 1u || outputs.size() != 1u) {
      return;
    }
    const auto &in = inputs.front();
    const auto &out = outputs.front();
    if (in.type == GraphSocketType::color &&
        (out.type == GraphSocketType::floating || out.type == GraphSocketType::integer)) {
      const auto value = literal<Vec3f>(&in, contract::SocketType::color);
      if (!value) { return; }
      const auto gray = folder.graph->blender_rgb_to_gray(*value);
      if (out.type == GraphSocketType::integer) {
        folder.make_constant(static_cast<std::int32_t>(gray));
      } else {
        folder.make_constant(gray);
      }
    } else {
      // All inputs are primitive: no inverse/identity rewrite on live links
      // is allowed to run in the earlier Blender phase.
      constant_fold(folder);
    }
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CONVERT;
  }
};

} // namespace

std::unique_ptr<GraphNode> make_convert_graph_node(std::string_view type) {
  if (type == cycles_synthetic_float3_autoconvert ||
      type == node_type::vector_to_color ||
      type == node_type::color_to_vector ||
      type == node_type::point_to_vector ||
      type == node_type::float3_to_vector ||
      type == node_type::vector_to_normal ||
      type == node_type::normal_to_vector ||
      type == node_type::scalar_to_color ||
      type == node_type::scalar_to_boolean ||
      type == node_type::color_to_scalar ||
      type == node_type::vector_to_scalar) {
    return std::make_unique<ConvertNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
