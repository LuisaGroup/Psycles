/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <memory>
#include <string_view>

namespace psycles::compiler::cycles_svm {

struct GraphNode;

[[nodiscard]] std::unique_ptr<GraphNode>
make_light_path_graph_node(std::string_view type);

} // namespace psycles::compiler::cycles_svm
