/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_graph.h"

#include <memory>
#include <string_view>

namespace psycles::compiler::cycles_svm {

[[nodiscard]] std::unique_ptr<GraphNode>
make_geometry_graph_node(std::string_view type);

} // namespace psycles::compiler::cycles_svm
