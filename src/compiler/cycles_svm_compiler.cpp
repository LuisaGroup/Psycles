/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

constexpr auto kernel_feature_node_bsdf = 1u << 0u;
constexpr auto kernel_feature_node_emission = 1u << 1u;
constexpr auto kernel_feature_node_volume = 1u << 2u;
constexpr auto kernel_feature_node_bump = 1u << 3u;
constexpr auto kernel_feature_node_bump_state = 1u << 4u;
constexpr auto kernel_feature_node_voronoi_extra = 1u << 5u;
constexpr auto kernel_feature_node_raytrace = 1u << 6u;
constexpr auto kernel_feature_node_aov = 1u << 7u;
constexpr auto kernel_feature_node_light_path = 1u << 8u;
constexpr auto kernel_feature_node_principled_hair = 1u << 9u;
constexpr auto kernel_feature_node_portal = 1u << 10u;

constexpr auto kernel_feature_node_mask_surface =
    kernel_feature_node_bsdf | kernel_feature_node_emission |
    kernel_feature_node_bump | kernel_feature_node_bump_state |
    kernel_feature_node_voronoi_extra | kernel_feature_node_raytrace |
    kernel_feature_node_aov | kernel_feature_node_light_path |
    kernel_feature_node_principled_hair | kernel_feature_node_portal;
constexpr auto kernel_feature_node_mask_volume =
    kernel_feature_node_emission | kernel_feature_node_volume |
    kernel_feature_node_voronoi_extra | kernel_feature_node_light_path |
    kernel_feature_node_portal;
constexpr auto kernel_feature_node_mask_displacement =
    kernel_feature_node_voronoi_extra | kernel_feature_node_bump |
    kernel_feature_node_bump_state | kernel_feature_node_portal;

enum class ShaderType : std::uint8_t {
  surface,
  volume,
  displacement,
  bump,
};

class Stack {
private:
  std::array<int, SVM_STACK_SIZE> _users{};
  std::uint32_t _peak{};
  bool _failed{};

public:
  void clear() noexcept { _users.fill(0); }

  [[nodiscard]] SVMStackOffset assign(std::uint32_t size) noexcept {
    auto offset = -1;
    for (auto index = std::uint32_t{}, unused = std::uint32_t{};
         index < SVM_STACK_SIZE; ++index) {
      if (_users[index] != 0) {
        unused = 0u;
      } else {
        ++unused;
      }
      if (unused == size) {
        offset = static_cast<int>(index + 1u - size);
        _peak = std::max(_peak, index + 1u);
        while (static_cast<int>(index) >= offset) {
          _users[index] = 1;
          if (index == 0u) {
            break;
          }
          --index;
        }
        return static_cast<SVMStackOffset>(offset);
      }
    }
    _failed = true;
    return 0u;
  }

  void release(SVMStackOffset offset, std::uint32_t size) noexcept {
    const auto begin = static_cast<std::uint32_t>(offset);
    if (begin + size > SVM_STACK_SIZE) {
      std::abort();
    }
    for (auto index = std::uint32_t{}; index < size; ++index) {
      auto &users = _users[begin + index];
      if (users <= 0) {
        std::abort();
      }
      --users;
    }
  }

  [[nodiscard]] bool failed() const noexcept { return _failed; }
  [[nodiscard]] std::uint32_t peak() const noexcept { return _peak; }
};

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

[[nodiscard]] std::uint32_t stack_base_size(GraphSocketType type) noexcept {
  switch (type) {
    case GraphSocketType::floating:
    case GraphSocketType::integer:
      return 1u;
    case GraphSocketType::color:
    case GraphSocketType::vector:
    case GraphSocketType::normal:
    case GraphSocketType::point:
      return 3u;
    case GraphSocketType::closure:
      return 0u;
  }
  return 0u;
}

[[nodiscard]] ShaderNodeType node_type_with_derivatives(
    ShaderNodeType type) noexcept {
  switch (type) {
#define SHADER_NODE_TYPE_DERIVATIVE(name) \
  case name:                               \
    return name##_DERIVATIVE;
#include <psycles/compiler/cycles_svm_node_types_template.h>
    default:
      return type;
  }
}

[[nodiscard]] std::optional<NodeMathType> math_type(
    const GraphNode *node) noexcept {
  if (node->type == cycles_synthetic_math) {
    return NODE_MATH_MULTIPLY;
  }
  return std::nullopt;
}

class Compiler {
private:
  struct CompilerState {
    explicit CompilerState(std::size_t node_capacity)
        : nodes_done_flag(node_capacity, false) {}

    GraphNodeSet nodes_done;
    GraphNodeSet closure_done;
    GraphNodeSet aov_nodes;
    std::vector<bool> nodes_done_flag;
    std::uint32_t node_feature_mask{};
  };

  CyclesGraph _graph;
  BytecodeBuilder _stream;
  Stack _stack;
  GraphNode *_current_node{};
  ShaderType _current_type{ShaderType::surface};
  SVMStackOffset _mix_weight_offset{SVM_STACK_INVALID};
  std::string _diagnostic;

private:
  [[nodiscard]] bool reject(std::string diagnostic) {
    if (_diagnostic.empty()) {
      _diagnostic = std::move(diagnostic);
    }
    return false;
  }

  [[nodiscard]] std::uint32_t stack_size(const GraphInput *input) const
      noexcept {
    const auto size = stack_base_size(input->type);
    return input->parent->need_derivatives ? size * 3u : size;
  }

  [[nodiscard]] std::uint32_t stack_size(const GraphOutput *output) const
      noexcept {
    const auto size = stack_base_size(output->type);
    return output->parent->need_derivatives ? size * 3u : size;
  }

  [[nodiscard]] SVMStackOffset stack_find_offset(std::uint32_t size) {
    const auto offset = _stack.assign(size);
    if (_stack.failed()) {
      static_cast<void>(reject("Shader graph: out of SVM stack space"));
    }
    return offset;
  }

  void add_value_node(GraphNode *node, float value,
                      SVMStackOffset stack_offset) {
    static_cast<void>(add_node(
        node, NODE_VALUE_F,
        SVMNodeValueF{.value = value,
                      .out_offset = stack_offset,
                      ._pad = {0u, 0u, 0u}}));
  }

  void add_value_node(GraphNode *node, Vec3f value,
                      SVMStackOffset stack_offset) {
    static_cast<void>(add_node(
        node, NODE_VALUE_V,
        SVMNodeValueV{.out_offset = stack_offset,
                      ._pad = {0u, 0u, 0u},
                      .value = packed_float3{value.x, value.y, value.z}}));
  }

  [[nodiscard]] SVMStackOffset stack_assign(GraphInput *input) {
    if (input->stack_offset == SVM_STACK_INVALID) {
      if (input->link != nullptr) {
        if (input->link->stack_offset == SVM_STACK_INVALID) {
          static_cast<void>(reject("Cycles SVM input producer was not scheduled"));
          return 0u;
        }
        input->stack_offset = input->link->stack_offset;
      } else {
        input->stack_offset = stack_find_offset(stack_size(input));
        if (!_diagnostic.empty()) {
          return 0u;
        }
        if (input->type == GraphSocketType::floating) {
          const auto value =
              literal<float>(input, contract::SocketType::floating);
          if (!value) {
            static_cast<void>(reject("Cycles SVM float input is ill typed"));
            return 0u;
          }
          add_value_node(input->parent, *value, input->stack_offset);
        } else if (input->type == GraphSocketType::integer) {
          std::optional<std::int32_t> value;
          if (input->value) {
            if (const auto *v = std::get_if<std::int64_t>(&input->value->value)) {
              value = static_cast<std::int32_t>(*v);
            } else if (const auto *v =
                           std::get_if<std::uint64_t>(&input->value->value)) {
              value = static_cast<std::int32_t>(*v);
            } else if (const auto *v = std::get_if<bool>(&input->value->value)) {
              value = *v ? 1 : 0;
            }
          }
          if (!value) {
            static_cast<void>(reject("Cycles SVM integer input is ill typed"));
            return 0u;
          }
          add_value_node(input->parent,
                         std::bit_cast<float>(static_cast<std::uint32_t>(*value)),
                         input->stack_offset);
        } else {
          std::optional<Vec3f> value;
          if (input->value) {
            value = std::visit(
                [](const auto &item) -> std::optional<Vec3f> {
                  using T = std::decay_t<decltype(item)>;
                  if constexpr (std::is_same_v<T, Vec3f>) {
                    return item;
                  }
                  return std::nullopt;
                },
                input->value->value);
          }
          if (!value) {
            static_cast<void>(reject("Cycles SVM vector input is ill typed"));
            return 0u;
          }
          add_value_node(input->parent, *value, input->stack_offset);
        }
      }
    }
    return input->stack_offset;
  }

  [[nodiscard]] SVMStackOffset stack_assign(GraphOutput *output_socket) {
    if (output_socket->stack_offset == SVM_STACK_INVALID) {
      output_socket->stack_offset = stack_find_offset(stack_size(output_socket));
    }
    return output_socket->stack_offset;
  }

  [[nodiscard]] SVMInputFloat input_float(std::string_view name) {
    auto *input = _current_node->input(name);
    if (input == nullptr) {
      static_cast<void>(reject("Cycles SVM float input is absent: " +
                               std::string{name}));
      return {};
    }
    if (input->link != nullptr) {
      return cycles_svm::input_float(stack_assign(input));
    }
    const auto value = literal<float>(input, contract::SocketType::floating);
    if (!value) {
      static_cast<void>(reject("Cycles SVM float input is ill typed: " +
                               std::string{name}));
      return {};
    }
    return cycles_svm::input_float(*value);
  }

  [[nodiscard]] SVMInputFloat3 input_float3(std::string_view name) {
    auto *input = _current_node->input(name);
    if (input == nullptr) {
      static_cast<void>(reject("Cycles SVM float3 input is absent: " +
                               std::string{name}));
      return {};
    }
    if (input->link != nullptr) {
      return cycles_svm::input_float3(stack_assign(input));
    }
    std::optional<Vec3f> value;
    if (input->value) {
      value = std::visit(
          [](const auto &item) -> std::optional<Vec3f> {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, Vec3f>) {
              return item;
            }
            return std::nullopt;
          },
          input->value->value);
    }
    if (!value) {
      static_cast<void>(reject("Cycles SVM float3 input is ill typed: " +
                               std::string{name}));
      return {};
    }
    return cycles_svm::input_float3(value->x, value->y, value->z);
  }

  [[nodiscard]] SVMStackOffset input_link(std::string_view name) {
    auto *input = _current_node->input(name);
    if (input == nullptr) {
      static_cast<void>(reject("Cycles SVM linked input is absent: " +
                               std::string{name}));
      return SVM_STACK_INVALID;
    }
    return (input->link != nullptr || input->constant_folded_in)
               ? stack_assign(input)
               : SVM_STACK_INVALID;
  }

  [[nodiscard]] SVMStackOffset output(std::string_view name) {
    auto *shader_output = _current_node->output(name);
    if (shader_output == nullptr) {
      static_cast<void>(reject("Cycles SVM output is absent: " +
                               std::string{name}));
      return SVM_STACK_INVALID;
    }
    return !shader_output->links.empty() ? stack_assign(shader_output)
                                         : SVM_STACK_INVALID;
  }

  template<SvmPayload T>
  [[nodiscard]] std::size_t add_node(GraphNode *node, ShaderNodeType type,
                                     const T &payload,
                                     bool use_derivatives = false) {
    const auto resolved =
        ((use_derivatives || (node != nullptr && node->need_derivatives)) &&
         _current_type != ShaderType::volume)
            ? node_type_with_derivatives(type)
            : type;
    const auto offset = _stream.add_node(resolved, payload);
    if (node != nullptr) {
      node->added_to_svm = true;
    }
    return offset;
  }

  [[nodiscard]] std::size_t add_node(ShaderNodeType type) {
    return _stream.add_node(type);
  }

  [[nodiscard]] bool compile_geometry(GraphNode *node) {
    struct Output {
      std::string_view name;
      NodeGeometry geometry;
    };
    static constexpr Output outputs[] = {
        {"Position", NODE_GEOM_P},
        {"Normal", NODE_GEOM_N},
        {"Tangent", NODE_GEOM_T},
        {"GeometricNormal", NODE_GEOM_Ng},
        {"Incoming", NODE_GEOM_I},
        {"Parametric", NODE_GEOM_uv},
    };
    for (const auto &item : outputs) {
      auto *socket = node->output(item.name);
      if (socket == nullptr || socket->links.empty()) {
        continue;
      }
      static_cast<void>(add_node(
          node, NODE_GEOMETRY,
          SVMNodeGeometry{.geom_type = item.geometry,
                          .bump_offset = NODE_BUMP_OFFSET_CENTER,
                          .store_derivatives =
                              static_cast<std::uint8_t>(node->need_derivatives),
                          .out_offset = output(item.name),
                          .bump_filter_width = 0.0f},
          item.geometry != NODE_GEOM_N && item.geometry != NODE_GEOM_Ng));
    }
    if (auto *socket = node->output("Backfacing");
        socket != nullptr && !socket->links.empty()) {
      static_cast<void>(add_node(
          node, NODE_LIGHT_PATH,
          SVMNodeLightPath{.path_type = NODE_LP_backfacing,
                           .out_offset = output("Backfacing"),
                           ._pad = {0u, 0u, 0u}}));
    }
    return _diagnostic.empty();
  }

  [[nodiscard]] bool compile_math(GraphNode *node) {
    const auto operation = math_type(node);
    if (!operation) {
      return reject("Cycles Math operation is not migrated exactly");
    }
    const auto value1_name = node->input("Value1") != nullptr ? "Value1" : "A";
    const auto value2_name = node->input("Value2") != nullptr ? "Value2" : "B";
    const auto value3_name = node->input("Value3") != nullptr ? "Value3" : "C";
    const auto result_name = node->output("Result") != nullptr ? "Result" : "Value";
    static_cast<void>(add_node(
        node, NODE_MATH,
        SVMNodeMath{.math_type = *operation,
                    .value1 = input_float(value1_name),
                    .value2 = input_float(value2_name),
                    .value3 = input_float(value3_name),
                    .result_offset = output(result_name),
                    ._pad = {0u, 0u, 0u}}));
    return _diagnostic.empty();
  }

  [[nodiscard]] bool compile_mix_closure_weight(GraphNode *node) {
    static_cast<void>(add_node(
        node, NODE_MIX_CLOSURE,
        SVMNodeMixClosure{.fac = input_float("Fac"),
                          .in_weight_offset = input_link("Weight"),
                          .weight1_offset = output("Weight1"),
                          .weight2_offset = output("Weight2"),
                          ._pad = {0u}}));
    return _diagnostic.empty();
  }

  template<SvmPayload T>
  [[nodiscard]] bool compile_bsdf(GraphNode *node, ClosureType closure,
                                  const T &data) {
    auto *color = node->input("Color");
    if (color == nullptr) {
      return reject("Cycles BSDF Color input is absent");
    }
    if (color->link != nullptr) {
      static_cast<void>(add_node(
          node, NODE_CLOSURE_WEIGHT,
          SVMNodeClosureWeight{.weight_offset = input_link("Color"),
                               ._pad = {0u, 0u, 0u}}));
    } else {
      const auto value = literal<Vec3f>(color, contract::SocketType::color);
      if (!value) {
        return reject("Cycles BSDF Color input is ill typed");
      }
      static_cast<void>(add_node(
          node, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{
              .rgb = packed_float3{value->x, value->y, value->z}}));
    }
    static_cast<void>(add_node(
        node, NODE_CLOSURE_BSDF,
        SVMNodeClosureBsdf{.closure_type = closure,
                           .mix_weight_offset = _mix_weight_offset,
                           ._pad = {0u, 0u, 0u}}));
    _stream.add_node_data(data);
    return _diagnostic.empty();
  }

  [[nodiscard]] bool compile_diffuse(GraphNode *node) {
    return compile_bsdf(
        node, CLOSURE_BSDF_DIFFUSE_ID,
        SVMNodeDiffuseBsdfData{.color = input_float3("Color"),
                               .roughness = input_float("Roughness"),
                               .normal_offset = input_link("Normal"),
                               ._pad = {0u, 0u, 0u}});
  }

  [[nodiscard]] bool compile_translucent(GraphNode *node) {
    return compile_bsdf(
        node, CLOSURE_BSDF_TRANSLUCENT_ID,
        SVMNodeSimpleBsdfData{.param1 = {},
                              .normal_offset = input_link("Normal"),
                              ._pad = {0u, 0u, 0u}});
  }

  [[nodiscard]] bool compile_transparent(GraphNode *node) {
    return compile_bsdf(node, CLOSURE_BSDF_TRANSPARENT_ID,
                        SVMNodeSimpleBsdfData{});
  }

  [[nodiscard]] bool compile_emission(GraphNode *node) {
    auto *color = node->input("Color");
    auto *strength = node->input("Strength");
    if (color == nullptr || strength == nullptr) {
      return reject("Cycles Emission inputs are absent");
    }
    if (color->link != nullptr || strength->link != nullptr) {
      static_cast<void>(add_node(
          node, NODE_EMISSION_WEIGHT,
          SVMNodeEmissionWeight{.color = input_float3("Color"),
                                .strength = input_float("Strength")}));
    } else {
      const auto c = literal<Vec3f>(color, contract::SocketType::color);
      const auto s = literal<float>(strength, contract::SocketType::floating);
      if (!c || !s) {
        return reject("Cycles Emission inputs are ill typed");
      }
      static_cast<void>(add_node(
          node, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{.rgb = packed_float3{
                                      c->x * *s, c->y * *s, c->z * *s}}));
    }
    static_cast<void>(add_node(
        node, NODE_CLOSURE_EMISSION,
        SVMNodeClosureEmission{.mix_weight_offset = _mix_weight_offset,
                               ._pad = {0u, 0u, 0u}}));
    return _diagnostic.empty();
  }

  [[nodiscard]] bool compile_node(GraphNode *node) {
    if (node->type == cycles_synthetic_geometry ||
        node->type == node_type::geometry) {
      return compile_geometry(node);
    }
    if (node->type == cycles_synthetic_mix_closure_weight) {
      return compile_mix_closure_weight(node);
    }
    if (node->type == cycles_synthetic_math) {
      return compile_math(node);
    }
    if (node->type == node_type::diffuse_bsdf) {
      return compile_diffuse(node);
    }
    if (node->type == node_type::translucent_bsdf) {
      return compile_translucent(node);
    }
    if (node->type == node_type::transparent_bsdf) {
      return compile_transparent(node);
    }
    if (node->type == node_type::emission) {
      return compile_emission(node);
    }
    if (node->type == node_type::null_closure ||
        node->type == node_type::null_volume ||
        node->special_type == GraphNodeSpecialType::combine_closure) {
      return true;
    }
    return reject("Cycles SVM node family is not migrated: " + node->type);
  }

  [[nodiscard]] std::uint32_t node_feature(const GraphNode *node) const
      noexcept {
    if (node->type == node_type::emission ||
        node->type == node_type::volume_emission) {
      return kernel_feature_node_emission;
    }
    if (node->type == node_type::volume_absorption ||
        node->type == node_type::volume_scatter ||
        node->type == node_type::volume_coefficients ||
        node->type == node_type::principled_volume) {
      return kernel_feature_node_volume;
    }
    if (node->special_type == GraphNodeSpecialType::closure) {
      return kernel_feature_node_bsdf;
    }
    return 0u;
  }

  [[nodiscard]] bool is_sole_user(const GraphNode *node,
                                  const GraphOutput *output_socket,
                                  const GraphNodeSet &done) const {
    for (const auto *input : output_socket->links) {
      if (input->parent != node && !done.contains(input->parent)) {
        return false;
      }
    }
    return true;
  }

  void stack_clear_users(GraphNode *node, const GraphNodeSet &done) {
    for (auto &input : node->inputs) {
      auto *producer = input.link;
      if (producer != nullptr && producer->stack_offset != SVM_STACK_INVALID &&
          is_sole_user(node, producer, done)) {
        _stack.release(producer->stack_offset, stack_size(producer));
        producer->stack_offset = SVM_STACK_INVALID;
        for (auto *consumer : producer->links) {
          consumer->stack_offset = SVM_STACK_INVALID;
        }
      }
    }
  }

  void stack_clear_temporary(GraphNode *node) {
    for (auto &input : node->inputs) {
      if (input.link == nullptr && input.stack_offset != SVM_STACK_INVALID) {
        _stack.release(input.stack_offset, stack_size(&input));
        input.stack_offset = SVM_STACK_INVALID;
      }
    }
  }

  [[nodiscard]] bool generate_node(GraphNode *node, GraphNodeSet &done) {
    _current_node = node;
    const auto ok = compile_node(node);
    _current_node = nullptr;
    if (!ok) {
      return false;
    }
    stack_clear_users(node, done);
    stack_clear_temporary(node);
    return true;
  }

  [[nodiscard]] std::uint32_t stack_node_output_size(
      const GraphNode *node) const noexcept {
    auto size = std::uint32_t{};
    for (const auto &output_socket : node->outputs) {
      if (!output_socket.links.empty() &&
          output_socket.stack_offset == SVM_STACK_INVALID) {
        size += stack_size(&output_socket);
      }
    }
    return size;
  }

  void find_dependencies(GraphNodeSet &dependencies, const GraphNodeSet &done,
                         GraphInput *input, GraphNode *skip_node = nullptr) {
    auto *node = input->link != nullptr ? input->link->parent : nullptr;
    if (node != nullptr && !done.contains(node) && node != skip_node &&
        !dependencies.contains(node)) {
      for (auto &dependency_input : node->inputs) {
        find_dependencies(dependencies, done, &dependency_input, skip_node);
      }
      dependencies.insert(node);
    }
  }

  [[nodiscard]] bool generate_svm_nodes(const GraphNodeSet &nodes,
                                        CompilerState *state) {
    auto &done = state->nodes_done;
    auto &done_flag = state->nodes_done_flag;

    auto get_producers = [&](const GraphNode *node,
                             std::vector<GraphNode *> &producers) {
      producers.clear();
      for (const auto &input : node->inputs) {
        if (input.link != nullptr) {
          auto *producer = input.link->parent;
          if (!done_flag[producer->id] &&
              std::find(producers.begin(), producers.end(), producer) ==
                  producers.end()) {
            producers.emplace_back(producer);
          }
        }
      }
    };

    std::unordered_map<const GraphNode *, int> num_consumers;
    std::vector<GraphNode *> consumers;
    for (auto *node : nodes) {
      if (done_flag[node->id]) {
        continue;
      }
      consumers.clear();
      for (const auto &output_socket : node->outputs) {
        for (const auto *input : output_socket.links) {
          auto *consumer = input->parent;
          if (!done_flag[consumer->id] && nodes.contains(consumer) &&
              std::find(consumers.begin(), consumers.end(), consumer) ==
                  consumers.end()) {
            consumers.emplace_back(consumer);
          }
        }
      }
      num_consumers[node] = static_cast<int>(consumers.size());
    }

    std::unordered_map<const GraphNode *, int> sethi_ullman_number;
    auto current_sethi_ullman_number = [&](GraphNode *node) {
      return num_consumers[node] > 1
                 ? static_cast<int>(stack_node_output_size(node))
                 : sethi_ullman_number[node];
    };
    auto node_order_key = [&](GraphNode *node) {
      return current_sethi_ullman_number(node) -
             static_cast<int>(stack_node_output_size(node));
    };
    auto node_order_compare = [&](GraphNode *lhs, GraphNode *rhs) {
      return node_order_key(lhs) > node_order_key(rhs) ||
             (node_order_key(lhs) == node_order_key(rhs) && lhs->id < rhs->id);
    };

    std::function<int(GraphNode *)> compute_sethi_ullman_number =
        [&](GraphNode *node) {
          const auto cached = sethi_ullman_number.find(node);
          if (cached != sethi_ullman_number.end()) {
            return cached->second;
          }
          std::vector<GraphNode *> producers;
          get_producers(node, producers);
          for (auto *producer : producers) {
            static_cast<void>(compute_sethi_ullman_number(producer));
          }
          std::sort(producers.begin(), producers.end(), node_order_compare);
          auto output_size = 0;
          auto peak_size = 0;
          for (auto *producer : producers) {
            peak_size = std::max(
                peak_size,
                output_size + current_sethi_ullman_number(producer));
            output_size += static_cast<int>(stack_node_output_size(producer));
          }
          peak_size =
              std::max(peak_size,
                       output_size +
                           static_cast<int>(stack_node_output_size(node)));
          sethi_ullman_number[node] = peak_size;
          return peak_size;
        };

    std::vector<GraphNode *> sinks;
    for (auto *node : nodes) {
      if (done_flag[node->id]) {
        continue;
      }
      static_cast<void>(compute_sethi_ullman_number(node));
      auto is_sink = true;
      for (const auto &output_socket : node->outputs) {
        for (const auto *input : output_socket.links) {
          if (!done_flag[input->parent->id] && nodes.contains(input->parent)) {
            is_sink = false;
            break;
          }
        }
        if (!is_sink) {
          break;
        }
      }
      if (is_sink) {
        sinks.emplace_back(node);
      }
    }
    std::sort(sinks.begin(), sinks.end(), node_order_compare);

    std::function<bool(GraphNode *)> generate = [&](GraphNode *node) {
      if (done_flag[node->id]) {
        return true;
      }
      std::vector<GraphNode *> producers;
      get_producers(node, producers);
      std::sort(producers.begin(), producers.end(), node_order_compare);
      for (auto *producer : producers) {
        if (!generate(producer)) {
          return false;
        }
      }
      if (!generate_node(node, done)) {
        return false;
      }
      done.insert(node);
      done_flag[node->id] = true;
      return true;
    };

    for (auto *node : sinks) {
      if (!generate(node)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool generate_closure_node(GraphNode *node,
                                           CompilerState *state) {
    const auto feature = node_feature(node);
    if ((state->node_feature_mask & feature) != feature) {
      return true;
    }
    for (auto &input : node->inputs) {
      if (input.link != nullptr) {
        GraphNodeSet dependencies;
        find_dependencies(dependencies, state->nodes_done, &input);
        if (!generate_svm_nodes(dependencies, state)) {
          return false;
        }
      }
    }
    auto *weight = node->input(_current_type == ShaderType::volume
                                   ? "VolumeMixWeight"
                                   : "SurfaceMixWeight");
    if (weight != nullptr) {
      const auto value = literal<float>(weight, contract::SocketType::floating);
      if (weight->link != nullptr || (value && *value != 1.0f)) {
        _mix_weight_offset = stack_assign(weight);
      } else {
        _mix_weight_offset = SVM_STACK_INVALID;
      }
    }
    if (!generate_node(node, state->nodes_done)) {
      return false;
    }
    _mix_weight_offset = SVM_STACK_INVALID;
    return true;
  }

  [[nodiscard]] bool generated_shared_closure_nodes(
      GraphNode *root_node, GraphNode *node, CompilerState *state,
      const GraphNodeSet &shared) {
    if (shared.contains(node)) {
      return generate_multi_closure(root_node, node, state);
    }
    for (auto &input : node->inputs) {
      if (input.type == GraphSocketType::closure && input.link != nullptr &&
          !generated_shared_closure_nodes(root_node, input.link->parent, state,
                                          shared)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool generate_multi_closure(GraphNode *root_node,
                                            GraphNode *node,
                                            CompilerState *state) {
    if (state->closure_done.contains(node)) {
      return true;
    }
    state->closure_done.insert(node);

    if (node->special_type == GraphNodeSpecialType::combine_closure) {
      auto *closure1 = node->input("A");
      auto *closure2 = node->input("B");
      auto *factor = node->input("Factor");
      if (closure1 == nullptr || closure2 == nullptr) {
        return reject("Cycles combine-closure sockets are incomplete");
      }
      if (closure1->link == nullptr && closure2->link == nullptr) {
        return true;
      }
      if (factor != nullptr && factor->link != nullptr) {
        GraphNodeSet dependencies;
        find_dependencies(dependencies, state->nodes_done, factor);
        if (!generate_svm_nodes(dependencies, state)) {
          return false;
        }

        GraphNodeSet closure1_dependencies;
        GraphNodeSet closure2_dependencies;
        GraphNodeSet shared_dependencies;
        find_dependencies(closure1_dependencies, state->nodes_done, closure1);
        find_dependencies(closure2_dependencies, state->nodes_done, closure2);
        std::set_intersection(
            closure1_dependencies.begin(), closure1_dependencies.end(),
            closure2_dependencies.begin(), closure2_dependencies.end(),
            std::inserter(shared_dependencies, shared_dependencies.begin()),
            GraphNodeIdComparator{});

        if (root_node != node) {
          for (auto &input : root_node->inputs) {
            GraphNodeSet root_dependencies;
            find_dependencies(root_dependencies, state->nodes_done, &input,
                              node);
            std::set_intersection(
                root_dependencies.begin(), root_dependencies.end(),
                closure1_dependencies.begin(), closure1_dependencies.end(),
                std::inserter(shared_dependencies,
                              shared_dependencies.begin()),
                GraphNodeIdComparator{});
            std::set_intersection(
                root_dependencies.begin(), root_dependencies.end(),
                closure2_dependencies.begin(), closure2_dependencies.end(),
                std::inserter(shared_dependencies,
                              shared_dependencies.begin()),
                GraphNodeIdComparator{});
          }
        }
        if (!shared_dependencies.empty()) {
          if (closure1->link != nullptr &&
              !generated_shared_closure_nodes(
                  root_node, closure1->link->parent, state,
                  shared_dependencies)) {
            return false;
          }
          if (closure2->link != nullptr &&
              !generated_shared_closure_nodes(
                  root_node, closure2->link->parent, state,
                  shared_dependencies)) {
            return false;
          }
          if (!generate_svm_nodes(shared_dependencies, state)) {
            return false;
          }
        }

        if (closure1->link != nullptr) {
          const auto start = _stream.size();
          static_cast<void>(add_node(
              nullptr, NODE_JUMP_IF_ONE,
              SVMNodeJumpIfOne{.jump_offset = 0,
                               .stack_offset = stack_assign(factor),
                               ._pad = {0u, 0u, 0u}}));
          if (!generate_multi_closure(root_node, closure1->link->parent,
                                      state)) {
            return false;
          }
          constexpr auto node_size =
              1u + sizeof(SVMNodeJumpIfOne) / sizeof(std::uint32_t);
          _stream.set_word(start + 1u,
                           static_cast<std::uint32_t>(_stream.size() -
                                                      (start + node_size)));
        }
        if (closure2->link != nullptr) {
          const auto start = _stream.size();
          static_cast<void>(add_node(
              nullptr, NODE_JUMP_IF_ZERO,
              SVMNodeJumpIfZero{.jump_offset = 0,
                                .stack_offset = stack_assign(factor),
                                ._pad = {0u, 0u, 0u}}));
          if (!generate_multi_closure(root_node, closure2->link->parent,
                                      state)) {
            return false;
          }
          constexpr auto node_size =
              1u + sizeof(SVMNodeJumpIfZero) / sizeof(std::uint32_t);
          _stream.set_word(start + 1u,
                           static_cast<std::uint32_t>(_stream.size() -
                                                      (start + node_size)));
        }
        factor->stack_offset = SVM_STACK_INVALID;
      } else {
        if (closure1->link != nullptr &&
            !generate_multi_closure(root_node, closure1->link->parent, state)) {
          return false;
        }
        if (closure2->link != nullptr &&
            !generate_multi_closure(root_node, closure2->link->parent, state)) {
          return false;
        }
      }
    } else if (!generate_closure_node(node, state)) {
      return false;
    }

    state->nodes_done.insert(node);
    state->nodes_done_flag[node->id] = true;
    return true;
  }

  void reset_type_state() {
    _stack.clear();
    for (const auto &node : _graph.nodes()) {
      node->added_to_svm = false;
      for (auto &input : node->inputs) {
        input.stack_offset = SVM_STACK_INVALID;
      }
      for (auto &output_socket : node->outputs) {
        output_socket.stack_offset = SVM_STACK_INVALID;
      }
    }
  }

  [[nodiscard]] bool compile_type(ShaderType type) {
    _current_type = type;
    reset_type_state();
    CompilerState state{_graph.node_id_capacity()};
    GraphInput *root_input = nullptr;
    auto *output_node = _graph.output_node();
    if (output_node == nullptr) {
      return reject("Cycles SVM graph has no OutputNode");
    }
    switch (type) {
      case ShaderType::surface:
        root_input = output_node->input("Surface");
        state.node_feature_mask = kernel_feature_node_mask_surface;
        break;
      case ShaderType::volume:
        root_input = output_node->input("Volume");
        state.node_feature_mask = kernel_feature_node_mask_volume;
        break;
      case ShaderType::displacement:
        root_input = output_node->input("Displacement");
        state.node_feature_mask = kernel_feature_node_mask_displacement;
        break;
      case ShaderType::bump:
        root_input = output_node->input("Normal");
        state.node_feature_mask = kernel_feature_node_mask_displacement;
        break;
    }
    if (root_input != nullptr && root_input->link != nullptr &&
        !generate_multi_closure(root_input->link->parent,
                                root_input->link->parent, &state)) {
      return false;
    }
    if (type == ShaderType::displacement && root_input != nullptr &&
        root_input->link != nullptr) {
      _current_node = output_node;
      static_cast<void>(add_node(
          output_node, NODE_SET_DISPLACEMENT,
          SVMNodeSetDisplacement{.fac_offset = stack_assign(root_input),
                                 ._pad = {0u, 0u, 0u}}));
      _current_node = nullptr;
    }
    if (type != ShaderType::bump) {
      static_cast<void>(add_node(NODE_END));
    }
    return _diagnostic.empty();
  }

  [[nodiscard]] ShaderImage finish(bool valid) {
    ShaderImage result;
    result.valid = valid && _diagnostic.empty();
    result.diagnostic = std::move(_diagnostic);
    result.words.assign(_stream.words().begin(), _stream.words().end());
    result.node_types_used = _stream.node_types_used();
    result.peak_stack_usage = _stack.peak();
    return result;
  }

public:
  explicit Compiler(const ShaderProgram &shader)
      : _graph{CyclesGraph::project(shader)} {}

  [[nodiscard]] ShaderImage compile() {
    if (!_graph.valid()) {
      static_cast<void>(reject(_graph.diagnostic()));
      return finish(false);
    }
    const auto jump_index = _stream.add_node(
        NODE_SHADER_JUMP, SVMNodeShaderJump{0, 0, 0});
    if (jump_index != 0u) {
      std::abort();
    }

    const auto has_bump = _graph.root(GraphDomain::bump) != nullptr;
    const auto surface_offset = _stream.size();
    if (has_bump && !compile_type(ShaderType::bump)) {
      return finish(false);
    }
    if (!compile_type(ShaderType::surface)) {
      return finish(false);
    }
    const auto volume_offset = _stream.size();
    if (!compile_type(ShaderType::volume)) {
      return finish(false);
    }
    const auto displacement_offset = _stream.size();
    if (!compile_type(ShaderType::displacement)) {
      return finish(false);
    }

    constexpr auto maximum_offset = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    if (surface_offset > maximum_offset || volume_offset > maximum_offset ||
        displacement_offset > maximum_offset) {
      static_cast<void>(reject("Cycles SVM shader offsets overflow int32"));
      return finish(false);
    }
    _stream.set_word(1u, static_cast<std::uint32_t>(surface_offset));
    _stream.set_word(2u, static_cast<std::uint32_t>(volume_offset));
    _stream.set_word(3u, static_cast<std::uint32_t>(displacement_offset));
    return finish(true);
  }
};

} // namespace

ShaderImage compile_shader(const ShaderProgram &shader) {
  return Compiler{shader}.compile();
}

} // namespace psycles::compiler::cycles_svm
