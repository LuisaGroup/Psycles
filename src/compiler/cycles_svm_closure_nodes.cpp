/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_closure_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
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

class BackgroundNode final : public GraphNode {
public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_emission;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  void constant_fold(const ConstantFolder &folder) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    const auto color_value =
        literal<Vec3f>(color, contract::SocketType::color);
    const auto strength_value =
        literal<float>(strength, contract::SocketType::floating);
    if ((color_value && *color_value == Vec3f{}) ||
        (strength_value && *strength_value == 0.0f)) {
      folder.discard();
    }
  }

  void compile(SVMCompiler &compiler) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    if (color == nullptr || strength == nullptr) {
      compiler.fail("Cycles Background inputs are absent");
      return;
    }
    if (color->link != nullptr || strength->link != nullptr) {
      compiler.add_node(
          this, NODE_EMISSION_WEIGHT,
          SVMNodeEmissionWeight{.color = compiler.input_float3("Color"),
                                .strength =
                                    compiler.input_float("Strength")});
    } else {
      const auto c = literal<Vec3f>(color, contract::SocketType::color);
      const auto s =
          literal<float>(strength, contract::SocketType::floating);
      if (!c || !s) {
        compiler.fail("Cycles Background inputs are ill typed");
        return;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{.rgb = packed_float3{
                                      c->x * *s, c->y * *s, c->z * *s}});
    }
    compiler.add_node(
        this, NODE_CLOSURE_BACKGROUND,
        SVMNodeClosureBackground{
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_closure_graph_node(std::string_view type) {
  if (type == node_type::background) {
    return std::make_unique<BackgroundNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
