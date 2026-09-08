#pragma once

#include "cycles_svm_graph.h"

namespace psycles::compiler::cycles_svm {

[[nodiscard]] bool set_blender_value_origin(
    GraphNode &node, contract::ShaderNodeOrigin origin) noexcept;
void inline_blender_socket_value(const ConstantFolder &folder);
void finish_blender_socket_values(CyclesGraph &graph) noexcept;

} // namespace psycles::compiler::cycles_svm
