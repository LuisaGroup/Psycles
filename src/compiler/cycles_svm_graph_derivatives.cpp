/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_graph.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <unordered_set>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] bool is_image_derivative_sink(const GraphNode *node) noexcept {
  if (node == nullptr) {
    return false;
  }
  switch (node->shader_node_type()) {
    case NODE_TEX_IMAGE:
    case NODE_TEX_IMAGE_BOX:
    case NODE_TEX_ENVIRONMENT:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool is_float3_socket(GraphSocketType type) noexcept {
  return type == GraphSocketType::color ||
         type == GraphSocketType::vector ||
         type == GraphSocketType::normal ||
         type == GraphSocketType::point;
}

[[nodiscard]] bool is_differentiable_socket(GraphSocketType type) noexcept {
  return type == GraphSocketType::floating || is_float3_socket(type);
}

enum class DerivativeRole : std::uint8_t {
  none = 0u,
  leaf = 1u,
  interior = 2u,
};

struct DerivativeRule {
  DerivativeRole role{DerivativeRole::none};
  std::vector<const GraphInput *> dual_inputs;
};

[[nodiscard]] DerivativeRule classify_derivative_node(
    const GraphNode &node) {
  if (is_image_derivative_sink(&node)) {
    return {};
  }
  const auto op = node.shader_node_type();
  // Explicitly exclude NODE_VERTEX_COLOR: node_vertex_color ignores
  // store_derivatives and does not write Dual3 outputs onto the stack.
  if (op == NODE_TEX_COORD || op == NODE_GEOMETRY || op == NODE_ATTR ||
      op == NODE_TANGENT || node.type == node_type::constant_float ||
      node.type == node_type::constant_color) {
    return {.role = DerivativeRole::leaf, .dual_inputs = {}};
  }
  DerivativeRule rule{};
  if (op == NODE_MAPPING || op == NODE_TEXTURE_MAPPING) {
    rule.role = DerivativeRole::interior;
    if (const auto *vec = node.input("Vector"); vec != nullptr) {
      rule.dual_inputs.push_back(vec);
    }
    return rule;
  }
  if (node.type == node_type::separate_xyz) {
    rule.role = DerivativeRole::interior;
    if (const auto *vec = node.input("Vector"); vec != nullptr) {
      rule.dual_inputs.push_back(vec);
    }
    return rule;
  }
  if (node.type == node_type::combine_xyz) {
    rule.role = DerivativeRole::interior;
    for (const auto *name : {"X", "Y", "Z"}) {
      if (const auto *in = node.input(name); in != nullptr) {
        rule.dual_inputs.push_back(in);
      }
    }
    return rule;
  }
  if (node.type == node_type::vector_math) {
    rule.role = DerivativeRole::interior;
    for (const auto &in : node.inputs) {
      if (is_differentiable_socket(in.type)) {
        rule.dual_inputs.push_back(&in);
      }
    }
    return rule;
  }
  if (op == NODE_CONVERT && node.inputs.size() == 1u &&
      node.outputs.size() == 1u &&
      is_differentiable_socket(node.inputs.front().type) &&
      is_differentiable_socket(node.outputs.front().type)) {
    rule.role = DerivativeRole::interior;
    rule.dual_inputs.push_back(&node.inputs.front());
    return rule;
  }
  return {};
}

[[nodiscard]] std::unordered_set<const GraphNode *>
collect_reachable_nodes(std::initializer_list<const GraphOutput *> roots) {
  std::unordered_set<const GraphNode *> reachable;
  std::vector<const GraphNode *> worklist;
  for (const GraphOutput *root_output : roots) {
    if (root_output != nullptr && root_output->parent != nullptr &&
        reachable.insert(root_output->parent).second) {
      worklist.push_back(root_output->parent);
    }
  }
  while (!worklist.empty()) {
    const GraphNode *node = worklist.back();
    worklist.pop_back();
    for (const auto &in : node->inputs) {
      if (in.link != nullptr && in.link->parent != nullptr &&
          reachable.insert(in.link->parent).second) {
        worklist.push_back(in.link->parent);
      }
    }
  }
  return reachable;
}

[[nodiscard]] std::unordered_set<const GraphNode *>
collect_non_volume_nodes(const CyclesGraph &graph) {
  auto reachable = collect_reachable_nodes({
      graph.root(GraphDomain::surface),
      graph.root(GraphDomain::bump),
      graph.root(GraphDomain::displacement),
  });
  if (const auto *volume_root = graph.root(GraphDomain::volume);
      volume_root != nullptr) {
    for (const GraphNode *vol_node : collect_reachable_nodes({volume_root})) {
      reachable.erase(vol_node);
    }
  }
  return reachable;
}

} // namespace

bool GraphNode::outputs_have_derivatives() const noexcept {
  return need_derivatives && !is_image_derivative_sink(this);
}

void CyclesGraph::propagate_derivatives(bool enable_texture_derivatives) {
  if (!enable_texture_derivatives || _nodes.empty()) {
    return;
  }

  const auto non_volume = collect_non_volume_nodes(*this);
  if (non_volume.empty()) {
    return;
  }

  const std::size_t id_cap = std::max<std::size_t>(
      _next_node_id, _nodes.size());
  std::vector<DerivativeRule> rules(id_cap);
  std::vector<std::uint8_t> is_non_volume(id_cap, 0u);
  for (const auto &node_ptr : _nodes) {
    if (!node_ptr || node_ptr->id >= id_cap) {
      continue;
    }
    if (non_volume.contains(node_ptr.get())) {
      is_non_volume[node_ptr->id] = 1u;
      rules[node_ptr->id] = classify_derivative_node(*node_ptr);
    }
  }

  // Pass 1: bottom-up capability fixed-point so an interior _DERIVATIVE op is
  // never enabled unless every linked dual input actually writes Dual outputs.
  std::vector<std::uint8_t> can_produce(id_cap, 0u);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &node_ptr : _nodes) {
      if (!node_ptr || node_ptr->id >= id_cap ||
          !is_non_volume[node_ptr->id] || can_produce[node_ptr->id] != 0u) {
        continue;
      }
      const auto &rule = rules[node_ptr->id];
      if (rule.role == DerivativeRole::none) {
        continue;
      }
      if (rule.role == DerivativeRole::leaf) {
        can_produce[node_ptr->id] = 1u;
        changed = true;
        continue;
      }
      bool all_ready = true;
      for (const GraphInput *in : rule.dual_inputs) {
        if (in != nullptr && in->link != nullptr) {
          const GraphNode *src = in->link->parent;
          if (src == nullptr || src->id >= id_cap ||
              can_produce[src->id] == 0u) {
            all_ready = false;
            break;
          }
        }
      }
      if (all_ready) {
        can_produce[node_ptr->id] = 1u;
        changed = true;
      }
    }
  }

  // Pass 2: backward demand propagation from image sinks in non-volume domains.
  std::vector<GraphNode *> demand_worklist;
  for (const auto &node_ptr : _nodes) {
    if (!node_ptr || node_ptr->id >= id_cap ||
        !is_non_volume[node_ptr->id] || !is_image_derivative_sink(node_ptr.get())) {
      continue;
    }
    const GraphInput *vec_in = node_ptr->input("Vector");
    if (vec_in == nullptr || vec_in->link == nullptr) {
      continue;
    }
    GraphNode *src = vec_in->link->parent;
    if (src == nullptr || src->id >= id_cap || can_produce[src->id] == 0u) {
      continue;
    }
    node_ptr->need_derivatives = true;
    if (!src->need_derivatives) {
      src->need_derivatives = true;
      demand_worklist.push_back(src);
    }
  }

  while (!demand_worklist.empty()) {
    GraphNode *node = demand_worklist.back();
    demand_worklist.pop_back();
    if (node->id >= id_cap) {
      continue;
    }
    const auto &rule = rules[node->id];
    for (const GraphInput *in : rule.dual_inputs) {
      if (in == nullptr || in->link == nullptr) {
        continue;
      }
      GraphNode *src = in->link->parent;
      if (src != nullptr && src->id < id_cap &&
          can_produce[src->id] != 0u && !src->need_derivatives) {
        src->need_derivatives = true;
        demand_worklist.push_back(src);
      }
    }
  }

  // Pass 3: forward alias closure over NODE_CONVERT_NONE (float3 <-> float3),
  // whose compile() uses stack_link() requiring identical producer/consumer sizes.
  changed = true;
  while (changed) {
    changed = false;
    for (const auto &node_ptr : _nodes) {
      if (!node_ptr || node_ptr->id >= id_cap ||
          !is_non_volume[node_ptr->id] || node_ptr->need_derivatives) {
        continue;
      }
      if (node_ptr->shader_node_type() != NODE_CONVERT ||
          node_ptr->inputs.size() != 1u || node_ptr->outputs.size() != 1u) {
        continue;
      }
      const auto &in = node_ptr->inputs.front();
      const auto &out = node_ptr->outputs.front();
      if (!is_float3_socket(in.type) || !is_float3_socket(out.type) ||
          in.link == nullptr || in.link->parent == nullptr) {
        continue;
      }
      if (in.link->parent->outputs_have_derivatives()) {
        node_ptr->need_derivatives = true;
        changed = true;
      }
    }
  }
}

} // namespace psycles::compiler::cycles_svm
