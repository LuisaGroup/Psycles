#include "cycles_svm_blender_values.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

namespace psycles::compiler::cycles_svm {

bool set_blender_value_origin(GraphNode &node, contract::ShaderNodeOrigin origin) noexcept {
  using Origin = contract::ShaderNodeOrigin;
  switch (origin) {
    case Origin::authored:
      break;
    case Origin::blender_input_value:
      if (node.type != node_type::constant_color && node.type != node_type::constant_float) {
        return false;
      }
      break;
    case Origin::blender_implicit_conversion:
      if (node.shader_node_type() != NODE_CONVERT ||
          node.inputs.size() != 1u || node.outputs.size() != 1u) {
        return false;
      }
      break;
    default:
      return false;
  }
  node.origin = origin;
  return true;
}

void inline_blender_socket_value(const ConstantFolder &folder) {
  auto *node = folder.node;
  using Origin = contract::ShaderNodeOrigin;
  if (node->origin == Origin::blender_input_value) {
    node->constant_fold(folder);
  } else if (node->origin == Origin::blender_implicit_conversion &&
             node->inputs.front().link != nullptr) {
    // Original ShaderNodesInliner::handle_implicit_conversion forwards the
    // actual LinkedSocketValue even through differently typed group sockets.
    // Retain typed projection edges during primitive propagation: eagerly
    // rebuilding a Cycles Convert here would reintroduce the erased boundary.
    auto *source = node->inputs.front().link;
    while (source->blender_source != nullptr) { source = source->blender_source; }
    folder.output->blender_source = source;
  } else {
    node->inline_blender_constant_fold(folder);
  }
}

void finish_blender_socket_values(CyclesGraph &graph) noexcept {
  // All links now have native Cycles types. Source-only metadata must not
  // enter deduplication, bump clones, node compilation or the interpreter.
  for (const auto &node : graph.nodes()) {
    node->origin = contract::ShaderNodeOrigin::authored;
    for (auto &output : node->outputs) { output.blender_source = nullptr; }
  }
}

} // namespace psycles::compiler::cycles_svm
