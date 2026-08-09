#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/surface_program.h>

namespace psycles::compiler::detail {

struct OutputKey {
  contract::NodeId node;
  std::string socket;

  auto operator<=>(const OutputKey &) const noexcept = default;
};

using LoweredOutput =
    std::variant<ValueExpressionId, ClosureExpressionId, VolumeExpressionId>;

[[nodiscard]] std::string property_string(const contract::ShaderNode &node,
                                          std::string_view name,
                                          std::string fallback = {});

[[nodiscard]] std::uint64_t property_uint(const contract::ShaderNode &node,
                                          std::string_view name,
                                          std::uint64_t fallback = 0u) noexcept;

[[nodiscard]] bool property_bool(const contract::ShaderNode &node,
                                 std::string_view name,
                                 bool fallback = false) noexcept;

[[nodiscard]] float property_float(const contract::ShaderNode &node,
                                   std::string_view name,
                                   float fallback = 0.0f) noexcept;

[[nodiscard]] Mat4f property_transform(const contract::ShaderNode &node,
                                       std::string_view name,
                                       Mat4f fallback = {}) noexcept;

[[nodiscard]] VolumePhase volume_phase(const contract::ShaderNode &node);

[[nodiscard]] std::vector<float> parse_float_table(const std::string &encoded);

[[nodiscard]] std::string node_prefix(contract::NodeId node);

[[nodiscard]] std::uint64_t color_mode(const contract::ShaderNode &node);

[[nodiscard]] MathOperation math_operation(const contract::ShaderNode &node);

[[nodiscard]] std::uint64_t
map_range_interpolation(const contract::ShaderNode &node);

[[nodiscard]] VectorMathOperation
vector_math_operation(const contract::ShaderNode &node);

[[nodiscard]] BlendOperation blend_operation(const contract::ShaderNode &node);

class SurfaceProgramBuilder {

private:
  const ShaderProgram &_shader;
  std::vector<SurfaceProgramDiagnostic> _diagnostics;
  std::vector<ParameterDesc> _parameters;
  std::vector<ValueInstruction> _value_instructions;
  std::vector<ClosureInstruction> _closure_instructions;
  std::vector<VolumeInstruction> _volume_instructions;
  std::map<OutputKey, LoweredOutput> _outputs;
  std::uint32_t _nishita_count{};

private:
  void diagnose(SurfaceProgramDiagnosticCode code, std::string message,
                std::optional<contract::NodeId> node = std::nullopt,
                std::string socket = {});

  [[nodiscard]] ParameterId add_parameter(const contract::ShaderNode &node,
                                          std::string_view socket,
                                          const contract::SocketValue &value,
                                          ParameterSource source);

  [[nodiscard]] std::optional<ValueExpressionId>
  lower_property_parameter(const contract::ShaderNode &node,
                           std::string_view property);

  [[nodiscard]] ValueExpressionId append(ValueInstruction instruction);

  [[nodiscard]] ClosureExpressionId append(ClosureInstruction instruction);

  [[nodiscard]] VolumeExpressionId append(VolumeInstruction instruction);

  template <typename Id>
  [[nodiscard]] std::optional<Id>
  source_output(const contract::ShaderNode &node, std::string_view socket,
                const contract::InputBinding &binding);

  [[nodiscard]] std::optional<ValueExpressionId>
  lower_value_input(const contract::ShaderNode &node, std::string_view socket);

  [[nodiscard]] std::optional<ClosureExpressionId>
  lower_closure_input(const contract::ShaderNode &node,
                      std::string_view socket);

  [[nodiscard]] std::optional<VolumeExpressionId>
  lower_volume_input(const contract::ShaderNode &node, std::string_view socket);

  void publish(contract::NodeId node, std::string socket, LoweredOutput output);

  void publish_unary_value(const contract::ShaderNode &node,
                           std::string_view input, std::string output,
                           contract::SocketType result_type,
                           ValueOperation operation);

  void publish_binary_value(const contract::ShaderNode &node,
                            std::string_view input_a, std::string_view input_b,
                            std::string output,
                            contract::SocketType result_type,
                            ValueOperation operation);

  [[nodiscard]] bool lower_context_node(const contract::ShaderNode &node);
  [[nodiscard]] bool lower_value_node(const contract::ShaderNode &node);
  [[nodiscard]] bool lower_texture_node(const contract::ShaderNode &node);
  [[nodiscard]] bool lower_closure_node(const contract::ShaderNode &node);
  void lower_node(const contract::ShaderNode &node);

public:
  explicit SurfaceProgramBuilder(const ShaderProgram &shader) noexcept;

  [[nodiscard]] SurfaceProgramCompilation build();
};

} // namespace psycles::compiler::detail
