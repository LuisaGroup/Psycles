#pragma once

#include "cycles_svm_graph.h"

namespace psycles::compiler::cycles_svm {
[[nodiscard]] std::unique_ptr<GraphNode>
make_range_graph_node(std::string_view type);
} // namespace psycles::compiler::cycles_svm
