/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_graph.h"

#include <memory>
#include <string_view>

namespace psycles::compiler::cycles_svm {

enum class TextureMappingAxis : std::uint8_t {
  none,
  x,
  y,
  z,
};

enum class TextureMappingProjection : std::uint8_t {
  flat,
  cube,
  tube,
  sphere,
};

// Isomorphic host representation of Cycles 5.2.1 TextureMapping. It is
// embedded in TextureNode rather than represented by a device graph node.
class TextureMapping {
public:
  Vec3f translation{};
  Vec3f rotation{};
  Vec3f scale{1.0f, 1.0f, 1.0f};
  Vec3f minimum{};
  Vec3f maximum{};
  bool use_minmax{};
  NodeMappingType type{NODE_MAPPING_TYPE_TEXTURE};
  TextureMappingAxis x_mapping{TextureMappingAxis::x};
  TextureMappingAxis y_mapping{TextureMappingAxis::y};
  TextureMappingAxis z_mapping{TextureMappingAxis::z};
  TextureMappingProjection projection{TextureMappingProjection::flat};

  [[nodiscard]] bool operator==(const TextureMapping &) const noexcept =
      default;
  [[nodiscard]] bool skip() const noexcept;
  void compile(SVMCompiler &compiler, SVMStackOffset offset_in,
               SVMStackOffset offset_out, GraphNode *node) const;
  [[nodiscard]] SVMStackOffset compile_begin(SVMCompiler &compiler,
                                             GraphInput *vector_in,
                                             GraphNode *node) const;
  void compile_end(SVMCompiler &compiler, GraphInput *vector_in,
                   SVMStackOffset vector_offset) const;
};

class TextureNode : public GraphNode {
public:
  TextureMapping tex_mapping;

  void copy_runtime_state_from(const GraphNode &other) override;
  [[nodiscard]] TextureMapping *texture_mapping() noexcept override;
  [[nodiscard]] const TextureMapping *texture_mapping() const noexcept override;
  [[nodiscard]] bool equals(const GraphNode &other) const noexcept override;
};

[[nodiscard]] std::unique_ptr<GraphNode>
make_mapping_graph_node(std::string_view type);

[[nodiscard]] bool configure_texture_mapping_from_legacy_node(
    TextureMapping &mapping, const GraphNode &legacy_mapping,
    std::string &diagnostic);

} // namespace psycles::compiler::cycles_svm
