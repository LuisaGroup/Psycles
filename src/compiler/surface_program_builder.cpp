#include "surface_program_builder.h"
#include "surface_program_compaction.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace psycles::compiler::detail {

[[nodiscard]] const contract::InputBinding *
find_input(const contract::ShaderNode &node, std::string_view name) noexcept {
  auto iter = node.inputs.find(name);
  return iter == node.inputs.end() ? nullptr : &iter->second;
}

[[nodiscard]] const contract::SocketValue *
find_property(const contract::ShaderNode &node,
              std::string_view name) noexcept {
  auto iter = node.properties.find(name);
  return iter == node.properties.end() ? nullptr : &iter->second;
}

[[nodiscard]] std::string property_string(const contract::ShaderNode &node,
                                          std::string_view name,
                                          std::string fallback) {
  const auto *value = find_property(node, name);
  if (value == nullptr || value->type != contract::SocketType::string) {
    return fallback;
  }
  return std::get<std::string>(value->value);
}

[[nodiscard]] std::uint64_t property_uint(const contract::ShaderNode &node,
                                          std::string_view name,
                                          std::uint64_t fallback) noexcept {
  const auto *value = find_property(node, name);
  if (value == nullptr ||
      value->type != contract::SocketType::unsigned_integer) {
    return fallback;
  }
  return std::get<std::uint64_t>(value->value);
}

[[nodiscard]] bool property_bool(const contract::ShaderNode &node,
                                 std::string_view name,
                                 bool fallback) noexcept {
  const auto *value = find_property(node, name);
  if (value == nullptr || value->type != contract::SocketType::boolean) {
    return fallback;
  }
  return std::get<bool>(value->value);
}

[[nodiscard]] VolumePhase volume_phase(const contract::ShaderNode &node) {
  const auto phase = property_string(node, "Phase", "HENYEY_GREENSTEIN");
  if (phase == "FOURNIER_FORAND") {
    return VolumePhase::fournier_forand;
  }
  if (phase == "DRAINE") {
    return VolumePhase::draine;
  }
  if (phase == "RAYLEIGH") {
    return VolumePhase::rayleigh;
  }
  if (phase == "MIE") {
    return VolumePhase::mie;
  }
  return VolumePhase::henyey_greenstein;
}

[[nodiscard]] float property_float(const contract::ShaderNode &node,
                                   std::string_view name,
                                   float fallback) noexcept {
  const auto *value = find_property(node, name);
  if (value == nullptr || value->type != contract::SocketType::floating) {
    return fallback;
  }
  return std::get<float>(value->value);
}

[[nodiscard]] Mat4f property_transform(const contract::ShaderNode &node,
                                       std::string_view name,
                                       Mat4f fallback) noexcept {
  const auto *value = find_property(node, name);
  if (value == nullptr || value->type != contract::SocketType::transform) {
    return fallback;
  }
  return std::get<Mat4f>(value->value);
}

[[nodiscard]] std::vector<float> parse_float_table(const std::string &encoded) {
  std::vector<float> result;
  const char *cursor = encoded.c_str();
  const char *end = cursor + encoded.size();
  while (cursor < end) {
    while (cursor < end &&
           (*cursor == ',' || *cursor == ';' || *cursor == ' ' ||
            *cursor == '\t' || *cursor == '\n' || *cursor == '\r')) {
      ++cursor;
    }
    if (cursor == end) {
      break;
    }
    char *next = nullptr;
    const auto value = std::strtof(cursor, &next);
    if (next == cursor) {
      break;
    }
    result.emplace_back(value);
    cursor = next;
  }
  return result;
}

[[nodiscard]] std::string node_prefix(contract::NodeId node) {
  return "node " + std::to_string(node.value) + ": ";
}

[[nodiscard]] std::uint64_t color_mode(const contract::ShaderNode &node) {
  const auto mode = property_string(node, "Mode", "RGB");
  return mode == "HSV" ? 1u : mode == "HSL" ? 2u : 0u;
}

[[nodiscard]] MathOperation math_operation(const contract::ShaderNode &node) {
  const auto operation = property_string(node, "Operation", "ADD");
  if (operation == "SUBTRACT") {
    return MathOperation::subtract;
  }
  if (operation == "MULTIPLY") {
    return MathOperation::multiply;
  }
  if (operation == "DIVIDE") {
    return MathOperation::divide;
  }
  if (operation == "MULTIPLY_ADD") {
    return MathOperation::multiply_add;
  }
  if (operation == "POWER") {
    return MathOperation::power;
  }
  if (operation == "LOGARITHM") {
    return MathOperation::logarithm;
  }
  if (operation == "SQRT") {
    return MathOperation::square_root;
  }
  if (operation == "INVERSE_SQRT") {
    return MathOperation::inverse_square_root;
  }
  if (operation == "ABSOLUTE") {
    return MathOperation::absolute;
  }
  if (operation == "EXPONENT") {
    return MathOperation::exponent;
  }
  if (operation == "MINIMUM") {
    return MathOperation::minimum;
  }
  if (operation == "MAXIMUM") {
    return MathOperation::maximum;
  }
  if (operation == "LESS_THAN") {
    return MathOperation::less_than;
  }
  if (operation == "GREATER_THAN") {
    return MathOperation::greater_than;
  }
  if (operation == "SIGN") {
    return MathOperation::sign;
  }
  if (operation == "COMPARE") {
    return MathOperation::compare;
  }
  if (operation == "SMOOTH_MIN") {
    return MathOperation::smooth_minimum;
  }
  if (operation == "SMOOTH_MAX") {
    return MathOperation::smooth_maximum;
  }
  if (operation == "ROUND") {
    return MathOperation::round;
  }
  if (operation == "FLOOR") {
    return MathOperation::floor;
  }
  if (operation == "CEIL") {
    return MathOperation::ceil;
  }
  if (operation == "TRUNC") {
    return MathOperation::trunc;
  }
  if (operation == "FRACT") {
    return MathOperation::fraction;
  }
  if (operation == "MODULO") {
    return MathOperation::modulo;
  }
  if (operation == "FLOORED_MODULO") {
    return MathOperation::floored_modulo;
  }
  if (operation == "WRAP") {
    return MathOperation::wrap;
  }
  if (operation == "SNAP") {
    return MathOperation::snap;
  }
  if (operation == "PINGPONG") {
    return MathOperation::ping_pong;
  }
  if (operation == "SINE") {
    return MathOperation::sine;
  }
  if (operation == "COSINE") {
    return MathOperation::cosine;
  }
  if (operation == "TANGENT") {
    return MathOperation::tangent;
  }
  if (operation == "ARCSINE") {
    return MathOperation::arcsine;
  }
  if (operation == "ARCCOSINE") {
    return MathOperation::arccosine;
  }
  if (operation == "ARCTANGENT") {
    return MathOperation::arctangent;
  }
  if (operation == "ARCTAN2") {
    return MathOperation::arctangent2;
  }
  if (operation == "SINH") {
    return MathOperation::hyperbolic_sine;
  }
  if (operation == "COSH") {
    return MathOperation::hyperbolic_cosine;
  }
  if (operation == "TANH") {
    return MathOperation::hyperbolic_tangent;
  }
  if (operation == "RADIANS") {
    return MathOperation::radians;
  }
  if (operation == "DEGREES") {
    return MathOperation::degrees;
  }
  return MathOperation::add;
}

[[nodiscard]] std::uint64_t
map_range_interpolation(const contract::ShaderNode &node) {
  const auto interpolation = property_string(node, "Interpolation", "LINEAR");
  return interpolation == "STEPPED"        ? 1u
         : interpolation == "SMOOTHSTEP"   ? 2u
         : interpolation == "SMOOTHERSTEP" ? 3u
                                           : 0u;
}

[[nodiscard]] VectorMathOperation
vector_math_operation(const contract::ShaderNode &node) {
  const auto operation = property_string(node, "Operation", "ADD");
  if (operation == "SUBTRACT") {
    return VectorMathOperation::subtract;
  }
  if (operation == "MULTIPLY") {
    return VectorMathOperation::multiply;
  }
  if (operation == "DIVIDE") {
    return VectorMathOperation::divide;
  }
  if (operation == "MULTIPLY_ADD") {
    return VectorMathOperation::multiply_add;
  }
  if (operation == "CROSS_PRODUCT") {
    return VectorMathOperation::cross_product;
  }
  if (operation == "PROJECT") {
    return VectorMathOperation::project;
  }
  if (operation == "REFLECT") {
    return VectorMathOperation::reflect;
  }
  if (operation == "REFRACT") {
    return VectorMathOperation::refract;
  }
  if (operation == "FACEFORWARD") {
    return VectorMathOperation::faceforward;
  }
  if (operation == "DOT_PRODUCT") {
    return VectorMathOperation::dot_product;
  }
  if (operation == "DISTANCE") {
    return VectorMathOperation::distance;
  }
  if (operation == "LENGTH") {
    return VectorMathOperation::length;
  }
  if (operation == "SCALE") {
    return VectorMathOperation::scale;
  }
  if (operation == "NORMALIZE") {
    return VectorMathOperation::normalize;
  }
  if (operation == "ABSOLUTE") {
    return VectorMathOperation::absolute;
  }
  if (operation == "POWER") {
    return VectorMathOperation::power;
  }
  if (operation == "SIGN") {
    return VectorMathOperation::sign;
  }
  if (operation == "MINIMUM") {
    return VectorMathOperation::minimum;
  }
  if (operation == "MAXIMUM") {
    return VectorMathOperation::maximum;
  }
  if (operation == "FLOOR") {
    return VectorMathOperation::floor;
  }
  if (operation == "CEIL") {
    return VectorMathOperation::ceil;
  }
  if (operation == "FRACTION") {
    return VectorMathOperation::fraction;
  }
  if (operation == "MODULO") {
    return VectorMathOperation::modulo;
  }
  if (operation == "WRAP") {
    return VectorMathOperation::wrap;
  }
  if (operation == "SNAP") {
    return VectorMathOperation::snap;
  }
  if (operation == "SINE") {
    return VectorMathOperation::sine;
  }
  if (operation == "COSINE") {
    return VectorMathOperation::cosine;
  }
  if (operation == "TANGENT") {
    return VectorMathOperation::tangent;
  }
  return VectorMathOperation::add;
}

[[nodiscard]] BlendOperation blend_operation(const contract::ShaderNode &node) {
  const auto mode = property_string(node, "BlendMode", "MIX");
  if (mode == "DARKEN") {
    return BlendOperation::darken;
  }
  if (mode == "MULTIPLY") {
    return BlendOperation::multiply;
  }
  if (mode == "BURN") {
    return BlendOperation::burn;
  }
  if (mode == "LIGHTEN") {
    return BlendOperation::lighten;
  }
  if (mode == "SCREEN") {
    return BlendOperation::screen;
  }
  if (mode == "DODGE") {
    return BlendOperation::dodge;
  }
  if (mode == "ADD") {
    return BlendOperation::add;
  }
  if (mode == "OVERLAY") {
    return BlendOperation::overlay;
  }
  if (mode == "SOFT_LIGHT") {
    return BlendOperation::soft_light;
  }
  if (mode == "LINEAR_LIGHT") {
    return BlendOperation::linear_light;
  }
  if (mode == "DIFFERENCE") {
    return BlendOperation::difference;
  }
  if (mode == "EXCLUSION") {
    return BlendOperation::exclusion;
  }
  if (mode == "SUBTRACT") {
    return BlendOperation::subtract;
  }
  if (mode == "DIVIDE") {
    return BlendOperation::divide;
  }
  if (mode == "HUE") {
    return BlendOperation::hue;
  }
  if (mode == "SATURATION") {
    return BlendOperation::saturation;
  }
  if (mode == "COLOR") {
    return BlendOperation::color;
  }
  if (mode == "VALUE") {
    return BlendOperation::value;
  }
  return BlendOperation::mix;
}

void SurfaceProgramBuilder::diagnose(SurfaceProgramDiagnosticCode code,
                                     std::string message,
                                     std::optional<contract::NodeId> node,
                                     std::string socket) {
  _diagnostics.emplace_back(
      SurfaceProgramDiagnostic{.code = code,
                               .message = std::move(message),
                               .node = node,
                               .socket = std::move(socket)});
}

[[nodiscard]] ParameterId
SurfaceProgramBuilder::add_parameter(const contract::ShaderNode &node,
                                     std::string_view socket,
                                     const contract::SocketValue &value,
                                     ParameterSource source) {
  auto id = ParameterId{static_cast<std::uint32_t>(_parameters.size())};
  _parameters.emplace_back(ParameterDesc{.id = id,
                                         .node = node.id,
                                         .socket = std::string{socket},
                                         .type = value.type,
                                         .default_value = value,
                                         .source = source});
  return id;
}

[[nodiscard]] std::optional<ValueExpressionId>
SurfaceProgramBuilder::lower_property_parameter(
    const contract::ShaderNode &node,
    std::string_view property) {
  const auto *value = find_property(node, property);
  if (value == nullptr) {
    diagnose(SurfaceProgramDiagnosticCode::missing_input,
             node_prefix(node.id) + "runtime property '" +
                 std::string{property} + "' is missing",
             node.id, std::string{property});
    return std::nullopt;
  }
  const auto parameter = add_parameter(
      node, property, *value, ParameterSource::property);
  return append(ValueInstruction{.operation = ValueOperation::parameter,
                                 .source_node = node.id,
                                 .result_type = value->type,
                                 .parameter = parameter});
}

[[nodiscard]] ValueExpressionId
SurfaceProgramBuilder::append(ValueInstruction instruction) {
  auto id =
      ValueExpressionId{static_cast<std::uint32_t>(_value_instructions.size())};
  _value_instructions.emplace_back(std::move(instruction));
  return id;
}

[[nodiscard]] ClosureExpressionId
SurfaceProgramBuilder::append(ClosureInstruction instruction) {
  auto id = ClosureExpressionId{
      static_cast<std::uint32_t>(_closure_instructions.size())};
  _closure_instructions.emplace_back(std::move(instruction));
  return id;
}

[[nodiscard]] VolumeExpressionId
SurfaceProgramBuilder::append(VolumeInstruction instruction) {
  auto id = VolumeExpressionId{
      static_cast<std::uint32_t>(_volume_instructions.size())};
  _volume_instructions.emplace_back(std::move(instruction));
  return id;
}

template <typename Id>
[[nodiscard]] std::optional<Id>
SurfaceProgramBuilder::source_output(const contract::ShaderNode &node,
                                     std::string_view socket,
                                     const contract::InputBinding &binding) {
  if (!binding.source) {
    return std::nullopt;
  }
  auto iter = _outputs.find(
      {.node = binding.source->node, .socket = binding.source->socket});
  if (iter == _outputs.end()) {
    diagnose(SurfaceProgramDiagnosticCode::missing_output,
             node_prefix(node.id) + "input '" + std::string{socket} +
                 "' references an output that was not lowered",
             node.id, std::string{socket});
    return std::nullopt;
  }
  if (const auto *id = std::get_if<Id>(&iter->second)) {
    return *id;
  }
  diagnose(SurfaceProgramDiagnosticCode::type_mismatch,
           node_prefix(node.id) + "input '" + std::string{socket} +
               "' has an incompatible lowered type",
           node.id, std::string{socket});
  return std::nullopt;
}

[[nodiscard]] std::optional<ValueExpressionId>
SurfaceProgramBuilder::lower_value_input(const contract::ShaderNode &node,
                                         std::string_view socket) {
  const auto *binding = find_input(node, socket);
  if (binding == nullptr) {
    diagnose(SurfaceProgramDiagnosticCode::missing_input,
             node_prefix(node.id) + "missing normalized input '" +
                 std::string{socket} + "'",
             node.id, std::string{socket});
    return std::nullopt;
  }
  if (binding->source) {
    return source_output<ValueExpressionId>(node, socket, *binding);
  }
  if (!binding->value) {
    diagnose(SurfaceProgramDiagnosticCode::missing_input,
             node_prefix(node.id) + "input '" + std::string{socket} +
                 "' has no value",
             node.id, std::string{socket});
    return std::nullopt;
  }
  auto parameter = add_parameter(
      node, socket, *binding->value, ParameterSource::input);
  return append(ValueInstruction{.operation = ValueOperation::parameter,
                                 .source_node = node.id,
                                 .result_type = binding->value->type,
                                 .parameter = parameter});
}

[[nodiscard]] std::optional<ClosureExpressionId>
SurfaceProgramBuilder::lower_closure_input(const contract::ShaderNode &node,
                                           std::string_view socket) {
  const auto *binding = find_input(node, socket);
  if (binding == nullptr || !binding->source) {
    diagnose(SurfaceProgramDiagnosticCode::missing_input,
             node_prefix(node.id) + "closure input '" + std::string{socket} +
                 "' is not connected",
             node.id, std::string{socket});
    return std::nullopt;
  }
  return source_output<ClosureExpressionId>(node, socket, *binding);
}

[[nodiscard]] std::optional<VolumeExpressionId>
SurfaceProgramBuilder::lower_volume_input(const contract::ShaderNode &node,
                                          std::string_view socket) {
  const auto *binding = find_input(node, socket);
  if (binding == nullptr || !binding->source) {
    diagnose(SurfaceProgramDiagnosticCode::missing_input,
             node_prefix(node.id) + "volume input '" + std::string{socket} +
                 "' is not connected",
             node.id, std::string{socket});
    return std::nullopt;
  }
  return source_output<VolumeExpressionId>(node, socket, *binding);
}

void SurfaceProgramBuilder::publish(contract::NodeId node, std::string socket,
                                    LoweredOutput output) {
  _outputs.insert_or_assign(
      OutputKey{.node = node, .socket = std::move(socket)}, std::move(output));
}

void SurfaceProgramBuilder::publish_unary_value(
    const contract::ShaderNode &node, std::string_view input,
    std::string output, contract::SocketType result_type,
    ValueOperation operation) {
  if (auto value = lower_value_input(node, input)) {
    publish(node.id, std::move(output),
            append(ValueInstruction{.operation = operation,
                                    .source_node = node.id,
                                    .result_type = result_type,
                                    .a = *value}));
  }
}

void SurfaceProgramBuilder::publish_binary_value(
    const contract::ShaderNode &node, std::string_view input_a,
    std::string_view input_b, std::string output,
    contract::SocketType result_type, ValueOperation operation) {
  auto a = lower_value_input(node, input_a);
  auto b = lower_value_input(node, input_b);
  if (a && b) {
    publish(node.id, std::move(output),
            append(ValueInstruction{.operation = operation,
                                    .source_node = node.id,
                                    .result_type = result_type,
                                    .a = *a,
                                    .b = *b}));
  }
}

void SurfaceProgramBuilder::lower_node(const contract::ShaderNode &node) {
  if (lower_context_node(node) || lower_value_node(node) ||
      lower_texture_node(node) || lower_closure_node(node)) {
    return;
  }
  diagnose(SurfaceProgramDiagnosticCode::unsupported_node,
           node_prefix(node.id) +
               "surface lowering is not implemented for node type '" +
               node.type + "'",
           node.id);
}

SurfaceProgramBuilder::SurfaceProgramBuilder(
    const ShaderProgram &shader) noexcept
    : _shader{shader} {}

[[nodiscard]] SurfaceProgramCompilation SurfaceProgramBuilder::build() {
  for (auto node_id : _shader.analysis().evaluation_order) {
    const auto *node = _shader.graph().find(node_id);
    if (node == nullptr) {
      diagnose(SurfaceProgramDiagnosticCode::missing_output,
               "validated shader program references a missing node", node_id);
      continue;
    }
    lower_node(*node);
  }

  const auto &root = _shader.graph().root(contract::ShaderDomain::surface);
  if (!root) {
    diagnose(SurfaceProgramDiagnosticCode::missing_surface_root,
             "shader has no surface root");
  }

  ClosureExpressionId lowered_root;
  if (root) {
    auto iter = _outputs.find({.node = root->node, .socket = root->socket});
    if (iter == _outputs.end()) {
      diagnose(SurfaceProgramDiagnosticCode::missing_output,
               "surface root was not lowered", root->node, root->socket);
    } else if (const auto *closure =
                   std::get_if<ClosureExpressionId>(&iter->second)) {
      lowered_root = *closure;
    } else {
      diagnose(SurfaceProgramDiagnosticCode::type_mismatch,
               "surface root did not lower to a closure", root->node,
               root->socket);
    }
  }

  VolumeExpressionId lowered_volume_root;
  const auto &volume_root =
      _shader.graph().root(contract::ShaderDomain::volume);
  if (volume_root) {
    auto iter = _outputs.find(
        {.node = volume_root->node, .socket = volume_root->socket});
    if (iter == _outputs.end()) {
      diagnose(SurfaceProgramDiagnosticCode::missing_output,
               "volume root was not lowered", volume_root->node,
               volume_root->socket);
    } else if (const auto *volume =
                   std::get_if<VolumeExpressionId>(&iter->second)) {
      lowered_volume_root = *volume;
    } else {
      diagnose(SurfaceProgramDiagnosticCode::type_mismatch,
               "volume root did not lower to a volume closure",
               volume_root->node, volume_root->socket);
    }
  }

  ValueExpressionId lowered_displacement_root;
  ValueExpressionId lowered_surface_normal_root;
  const auto &surface_normal_root =
      _shader.graph().root(contract::ShaderDomain::surface_normal);
  if (surface_normal_root) {
    auto iter = _outputs.find(
        {.node = surface_normal_root->node,
         .socket = surface_normal_root->socket});
    if (iter == _outputs.end()) {
      diagnose(SurfaceProgramDiagnosticCode::missing_output,
               "surface normal root was not lowered",
               surface_normal_root->node,
               surface_normal_root->socket);
    } else if (const auto *value =
                   std::get_if<ValueExpressionId>(&iter->second)) {
      lowered_surface_normal_root = *value;
    } else {
      diagnose(SurfaceProgramDiagnosticCode::type_mismatch,
               "surface normal root did not lower to a value",
               surface_normal_root->node,
               surface_normal_root->socket);
    }
  }

  const auto &displacement_root =
      _shader.graph().root(contract::ShaderDomain::displacement);
  if (displacement_root) {
    auto iter = _outputs.find(
        {.node = displacement_root->node, .socket = displacement_root->socket});
    if (iter == _outputs.end()) {
      diagnose(SurfaceProgramDiagnosticCode::missing_output,
               "displacement root was not lowered", displacement_root->node,
               displacement_root->socket);
    } else if (const auto *value =
                   std::get_if<ValueExpressionId>(&iter->second)) {
      lowered_displacement_root = *value;
    } else {
      diagnose(SurfaceProgramDiagnosticCode::type_mismatch,
               "displacement root did not lower to a value",
               displacement_root->node, displacement_root->socket);
    }
  }

  for (const auto &property :
       _shader.analysis().runtime_properties) {
    const auto lowered = std::ranges::any_of(
        _parameters,
        [&](const ParameterDesc &parameter) noexcept {
          return parameter.source == ParameterSource::property &&
                 parameter.node == property.node &&
                 parameter.socket == property.property &&
                 parameter.type == property.type;
        });
    if (!lowered) {
      diagnose(
          SurfaceProgramDiagnosticCode::unsupported_node,
          node_prefix(property.node) + "runtime property '" +
              property.property +
              "' was not lowered into the typed parameter block",
          property.node,
          property.property);
    }
  }

  if (!_diagnostics.empty()) {
    return {.program = nullptr, .diagnostics = std::move(_diagnostics)};
  }

  auto storage = compact_surface_program(
      SurfaceProgramStorage{.parameters = std::move(_parameters),
                            .values = std::move(_value_instructions),
                            .closures = std::move(_closure_instructions),
                            .root = lowered_root,
                            .volumes = std::move(_volume_instructions),
                            .volume_root = lowered_volume_root,
                            .surface_normal_root =
                                lowered_surface_normal_root,
                            .displacement_root = lowered_displacement_root});
  return {.program = std::make_shared<const SurfaceProgram>(
              _shader.analysis().structure_signature,
              std::move(storage.parameters), std::move(storage.values),
              std::move(storage.closures), storage.root,
              std::move(storage.volumes), storage.volume_root,
              storage.surface_normal_root,
              storage.displacement_root),
          .diagnostics = {}};
}

} // namespace psycles::compiler::detail
