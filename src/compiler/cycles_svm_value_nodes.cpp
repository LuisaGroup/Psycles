/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_value_nodes.h"

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

class ValueNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (const auto value =
            literal<float>(input("Value"), contract::SocketType::floating)) {
      folder.make_constant(*value);
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto value =
        literal<float>(input("Value"), contract::SocketType::floating);
    if (!value) {
      compiler.fail("Cycles Value node input is ill typed or linked");
      return;
    }
    const auto offset = compiler.output("Value");
    if (offset != SVM_STACK_INVALID) {
      compiler.add_value_node(this, *value, offset);
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }
};

class ColorNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (const auto value =
            literal<Vec3f>(input("Color"), contract::SocketType::color)) {
      folder.make_constant(*value);
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto value =
        literal<Vec3f>(input("Color"), contract::SocketType::color);
    if (!value) {
      compiler.fail("Cycles Color node input is ill typed or linked");
      return;
    }
    const auto offset = compiler.output("Color");
    if (offset != SVM_STACK_INVALID) {
      compiler.add_value_node(this, *value, offset);
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }
};

} // namespace

std::unique_ptr<GraphNode> make_value_graph_node(std::string_view type) {
  if (type == node_type::constant_float) {
    return std::make_unique<ValueNode>();
  }
  if (type == node_type::constant_color) {
    return std::make_unique<ColorNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
