/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_graph.h"

#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace psycles::compiler::cycles_svm {

inline constexpr auto kernel_feature_node_bsdf = 1u << 0u;
inline constexpr auto kernel_feature_node_emission = 1u << 1u;
inline constexpr auto kernel_feature_node_volume = 1u << 2u;
inline constexpr auto kernel_feature_node_bump = 1u << 3u;
inline constexpr auto kernel_feature_node_bump_state = 1u << 4u;
inline constexpr auto kernel_feature_node_voronoi_extra = 1u << 5u;
inline constexpr auto kernel_feature_node_raytrace = 1u << 6u;
inline constexpr auto kernel_feature_node_aov = 1u << 7u;
inline constexpr auto kernel_feature_node_light_path = 1u << 8u;
inline constexpr auto kernel_feature_node_principled_hair = 1u << 9u;
inline constexpr auto kernel_feature_node_portal = 1u << 10u;

// Host graph compiler corresponding to Cycles 5.2.1 scene/SVMCompiler. Node
// emission stays on GraphNode::compile(SVMCompiler &), so virtual dispatch is
// resolved while constructing the bytecode and never reaches a device kernel.
class SVMCompiler {
protected:
  virtual void add_node_payload(GraphNode *node, ShaderNodeType type,
                                const void *payload,
                                std::size_t payload_size,
                                bool use_derivatives) = 0;
  virtual void add_node_data(const void *payload, std::size_t payload_size) = 0;
  virtual void add_bsdf_node_payload(const SVMNodeClosureBsdf &node,
                                     const void *payload,
                                     std::size_t payload_size) = 0;

public:
  virtual ~SVMCompiler() noexcept = default;

  virtual void fail(std::string diagnostic) = 0;
  [[nodiscard]] virtual SVMInputFloat input_float(std::string_view name) = 0;
  [[nodiscard]] virtual SVMInputFloat3 input_float3(std::string_view name) = 0;
  [[nodiscard]] virtual SVMStackOffset input_link(std::string_view name) = 0;
  [[nodiscard]] virtual SVMStackOffset output(std::string_view name) = 0;

  virtual void add_value_node(GraphNode *node, float value,
                              SVMStackOffset stack_offset) = 0;
  virtual void add_value_node(GraphNode *node, Vec3f value,
                              SVMStackOffset stack_offset) = 0;

  template<SvmPayload T>
  void add_node(GraphNode *node, ShaderNodeType type, const T &payload,
                bool use_derivatives = false) {
    add_node_payload(node, type, &payload, sizeof(T), use_derivatives);
  }

  [[nodiscard]] virtual std::size_t add_node(ShaderNodeType type) = 0;

  template<SvmPayload T>
  void add_bsdf_node(const SVMNodeClosureBsdf &node, const T &data) {
    add_bsdf_node_payload(node, &data, sizeof(data));
  }

  [[nodiscard]] virtual SVMStackOffset
  closure_mix_weight_offset() const noexcept = 0;
  [[nodiscard]] virtual ShaderType output_type() const noexcept = 0;
};

} // namespace psycles::compiler::cycles_svm
