/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/compiler/cycles_svm_types.h>
#include <psycles/compiler/shader_program.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace psycles::compiler::cycles_svm {

class SVMCompiler;
class ConstantFolder;
class CyclesGraph;
class TextureMapping;

inline constexpr auto cycles_synthetic_texture_coordinate =
    "cycles.synthetic.texture_coordinate";
inline constexpr auto cycles_synthetic_mix_closure_weight =
    "cycles.synthetic.mix_closure_weight";
inline constexpr auto cycles_synthetic_math = "cycles.synthetic.math";

// This is the host graph consumed by the Cycles 5.2 SVM compiler. Its
// topology and mutable compiler fields correspond to ShaderInput,
// ShaderOutput, ShaderNode, and ShaderGraph in scene/shader_graph.h. The
// contract graph remains the serialization boundary; no Psycles execution IR
// is introduced between it and this Cycles graph.

enum class GraphSocketType : std::uint8_t {
  floating,
  integer,
  color,
  vector,
  normal,
  point,
  closure,
};

enum class GraphNodeSpecialType : std::uint8_t {
  none,
  geometry,
  bump,
  closure,
  combine_closure,
  output,
  output_aov,
};

enum ShaderBump : std::uint8_t {
  SHADER_BUMP_NONE,
  SHADER_BUMP_CENTER,
  SHADER_BUMP_DX,
  SHADER_BUMP_DY,
};

enum class GraphDomain : std::uint8_t {
  surface,
  volume,
  displacement,
  bump,
  count,
};

enum GraphSocketFlag : std::uint16_t {
  graph_socket_none = 0u,
  graph_socket_link_texture_generated = 1u << 0u,
  graph_socket_link_texture_normal = 1u << 1u,
  graph_socket_link_texture_uv = 1u << 2u,
  graph_socket_link_texture_incoming = 1u << 3u,
  graph_socket_link_incoming = 1u << 4u,
  graph_socket_link_normal = 1u << 5u,
  graph_socket_link_position = 1u << 6u,
  graph_socket_link_tangent = 1u << 7u,
};

struct GraphNode;
struct GraphOutput;

struct GraphInput {
  GraphNode *parent{};
  std::string name;
  GraphSocketType type{};
  std::uint16_t flags{};
  std::optional<contract::SocketValue> value;
  GraphOutput *link{};
  SVMStackOffset stack_offset{SVM_STACK_INVALID};
  bool constant_folded_in{};
};

struct GraphOutput {
  GraphNode *parent{};
  std::string name;
  GraphSocketType type{};
  std::vector<GraphInput *> links;
  SVMStackOffset stack_offset{SVM_STACK_INVALID};
};

struct GraphNode {
  virtual ~GraphNode() noexcept = default;

  std::uint32_t id{};
  std::string type;
  std::string label;
  std::vector<GraphInput> inputs;
  std::vector<GraphOutput> outputs;
  std::map<std::string, contract::SocketValue, std::less<>> properties;
  GraphNodeSpecialType special_type{GraphNodeSpecialType::none};
  ShaderBump bump{SHADER_BUMP_NONE};
  float bump_filter_width{};
  bool added_to_svm{};
  bool need_derivatives{};

  virtual void compile(SVMCompiler &compiler) = 0;
  virtual void expand(CyclesGraph &graph);
  virtual void constant_fold(const ConstantFolder &folder);
  virtual void simplify_settings();
  // Subclass state which does not belong to the serialized socket/property
  // maps is copied explicitly when ShaderGraph duplicates bump dependencies.
  virtual void copy_runtime_state_from(const GraphNode &other);
  [[nodiscard]] virtual TextureMapping *texture_mapping() noexcept;
  [[nodiscard]] virtual const TextureMapping *texture_mapping() const noexcept;
  [[nodiscard]] virtual std::uint32_t get_feature() const noexcept;
  [[nodiscard]] virtual ShaderNodeType shader_node_type() const noexcept;
  [[nodiscard]] virtual bool equals(const GraphNode &other) const noexcept;
  [[nodiscard]] virtual bool has_volume_support() const noexcept;
  [[nodiscard]] virtual bool is_linear_operation() const noexcept;

  [[nodiscard]] GraphInput *input(std::string_view name) noexcept;
  [[nodiscard]] const GraphInput *input(std::string_view name) const noexcept;
  [[nodiscard]] GraphOutput *output(std::string_view name) noexcept;
  [[nodiscard]] const GraphOutput *output(std::string_view name) const noexcept;
};

[[nodiscard]] std::unique_ptr<GraphNode>
make_graph_node(std::string_view type);

struct GraphNodeIdComparator {
  [[nodiscard]] bool operator()(const GraphNode *lhs,
                                const GraphNode *rhs) const noexcept {
    return lhs->id < rhs->id;
  }
};

using GraphNodeSet = std::set<GraphNode *, GraphNodeIdComparator>;

class CyclesGraph {
private:
  std::vector<std::unique_ptr<GraphNode>> _nodes;
  std::uint32_t _next_node_id{};
  std::string _diagnostic;

public:
  [[nodiscard]] static CyclesGraph project(const ShaderProgram &shader);

  [[nodiscard]] bool valid() const noexcept { return _diagnostic.empty(); }
  [[nodiscard]] const std::string &diagnostic() const noexcept {
    return _diagnostic;
  }
  [[nodiscard]] const std::vector<std::unique_ptr<GraphNode>> &nodes() const
      noexcept {
    return _nodes;
  }
  [[nodiscard]] std::size_t node_id_capacity() const noexcept {
    return _next_node_id;
  }
  [[nodiscard]] GraphOutput *root(GraphDomain domain) const noexcept;
  [[nodiscard]] GraphNode *output_node() const noexcept {
    return _nodes.empty() ? nullptr : _nodes.front().get();
  }

  [[nodiscard]] GraphNode *add_node(
      std::string type, std::string label,
      std::vector<GraphInput> inputs,
      std::vector<GraphOutput> outputs,
      GraphNodeSpecialType special_type = GraphNodeSpecialType::none,
      std::map<std::string, contract::SocketValue, std::less<>> properties = {});
  [[nodiscard]] bool connect(GraphOutput *output, GraphInput *input) noexcept;
  void disconnect(GraphInput *input) noexcept;
  void disconnect(GraphOutput *output) noexcept;

  // Exact ShaderGraph::default_inputs and transform_multi_closure stages for
  // the SVM-visible graph. Unsupported source metadata fails projection.
  void default_inputs();
  void project_texture_mappings();
  void transform_multi_closure(GraphNode *node, GraphOutput *weight_output,
                               bool volume);
  void expand();
  void clean();
  void refine_bump_nodes();

private:
  void constant_fold();
  void simplify_settings();
  void deduplicate_nodes();
  void optimize_volume_output();
  void relink(GraphNode *node, GraphOutput *from, GraphOutput *to);
  void break_cycles(GraphNode *node, std::vector<bool> &visited,
                    std::vector<bool> &on_stack);
  void find_dependencies(GraphNodeSet &dependencies, GraphInput *input);
  void copy_nodes(
      GraphNodeSet &nodes,
      std::map<GraphNode *, GraphNode *, GraphNodeIdComparator> &node_map);
  void reject(std::string diagnostic);
};

[[nodiscard]] std::optional<GraphSocketType>
graph_socket_type(contract::SocketType type) noexcept;

} // namespace psycles::compiler::cycles_svm
