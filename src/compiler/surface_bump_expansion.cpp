#include <psycles/compiler/surface_bump_expansion.h>

#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

constexpr auto volume_value_dependencies = std::array{
    &VolumeInstruction::color,
    &VolumeInstruction::density,
    &VolumeInstruction::anisotropy,
    &VolumeInstruction::ior,
    &VolumeInstruction::backscatter,
    &VolumeInstruction::alpha,
    &VolumeInstruction::diameter,
    &VolumeInstruction::scatter_coefficients,
    &VolumeInstruction::absorption_coefficients,
    &VolumeInstruction::absorption_color,
    &VolumeInstruction::emission_coefficients,
    &VolumeInstruction::emission_strength,
    &VolumeInstruction::emission_color,
    &VolumeInstruction::blackbody_intensity,
    &VolumeInstruction::blackbody_tint,
    &VolumeInstruction::temperature,
    &VolumeInstruction::factor};

struct InstructionKey {
  ValueOperation operation{};
  std::uint32_t source_node{};
  contract::SocketType result_type{};
  std::uint32_t parameter{};
  std::vector<std::uint32_t> operands;
  std::uint64_t static_u0{};
  std::uint64_t static_u1{};
  std::uint32_t static_f0{};
  std::uint32_t static_f1{};
  std::vector<std::uint32_t> static_table;

  auto operator<=>(const InstructionKey &) const = default;
};

[[nodiscard]] InstructionKey
make_instruction_key(const ValueInstruction &instruction) {
  InstructionKey key{
      .operation = instruction.operation,
      .source_node = instruction.source_node.value,
      .result_type = instruction.result_type,
      .parameter = instruction.parameter.value,
      .operands = {},
      .static_u0 = instruction.static_u0,
      .static_u1 = instruction.static_u1,
      .static_f0 = std::bit_cast<std::uint32_t>(instruction.static_f0),
      .static_f1 = std::bit_cast<std::uint32_t>(instruction.static_f1),
      .static_table = {}};
  key.operands.reserve(instruction.operands.size());
  for (const auto operand : instruction.operands) {
    key.operands.emplace_back(operand.value);
  }
  key.static_table.reserve(instruction.static_table.size());
  for (const auto value : instruction.static_table) {
    key.static_table.emplace_back(std::bit_cast<std::uint32_t>(value));
  }
  return key;
}

struct SampleContext {
  ValueExpressionId dx;
  ValueExpressionId dy;

  [[nodiscard]] bool root() const noexcept {
    return !dx.valid() && !dy.valid();
  }
};

struct ContextKey {
  std::uint32_t source{};
  std::uint32_t dx{};
  std::uint32_t dy{};

  auto operator<=>(const ContextKey &) const noexcept = default;
};

[[nodiscard]] ContextKey make_context_key(ValueExpressionId source,
                                          SampleContext context) noexcept {
  return {.source = source.value,
          .dx = context.dx.value,
          .dy = context.dy.value};
}

[[nodiscard]] bool is_internal_bump_operation(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::bump_offset_zero:
  case ValueOperation::bump_filter_width:
  case ValueOperation::bump_samples:
  case ValueOperation::sampled_surface_position:
  case ValueOperation::sampled_uv:
  case ValueOperation::sampled_generated:
  case ValueOperation::sampled_object_position:
  case ValueOperation::sampled_object_position_with_transform:
  case ValueOperation::sampled_pointiness:
  case ValueOperation::sampled_attribute_color:
  case ValueOperation::sampled_attribute_factor:
  case ValueOperation::sampled_attribute_alpha:
  case ValueOperation::sampled_normal_map:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] ValueOperation
sampled_operation(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::surface_position:
    return ValueOperation::sampled_surface_position;
  case ValueOperation::uv:
    return ValueOperation::sampled_uv;
  case ValueOperation::generated:
    return ValueOperation::sampled_generated;
  case ValueOperation::object_position:
    return ValueOperation::sampled_object_position;
  case ValueOperation::object_position_with_transform:
    return ValueOperation::sampled_object_position_with_transform;
  case ValueOperation::pointiness:
    return ValueOperation::sampled_pointiness;
  case ValueOperation::attribute_color:
    return ValueOperation::sampled_attribute_color;
  case ValueOperation::attribute_factor:
    return ValueOperation::sampled_attribute_factor;
  case ValueOperation::attribute_alpha:
    return ValueOperation::sampled_attribute_alpha;
  case ValueOperation::normal_map:
    return ValueOperation::sampled_normal_map;
  default:
    return operation;
  }
}

class ExpansionBuilder {

private:
  const SurfaceProgram &_source;
  SurfaceBumpExpansion _result;
  std::vector<ValueInstruction> _values;
  std::map<InstructionKey, ValueExpressionId> _value_numbers;
  std::map<ContextKey, ValueExpressionId> _memo;
  ValueExpressionId _zero;

private:
  [[nodiscard]] ValueExpressionId fail(std::string diagnostic) {
    if (_result.diagnostic.empty()) {
      _result.diagnostic = std::move(diagnostic);
    }
    return {};
  }

  [[nodiscard]] ValueExpressionId emit(ValueInstruction instruction) {
    if (!_result.diagnostic.empty()) {
      return {};
    }
    for (const auto operand : instruction.operands) {
      if (!operand.valid() || operand.value >= _values.size()) {
        return fail("expanded Bump graph is not in strict topological order");
      }
    }
    auto key = make_instruction_key(instruction);
    if (const auto iter = _value_numbers.find(key);
        iter != _value_numbers.end()) {
      return iter->second;
    }
    if (_values.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return fail("expanded Bump graph exceeds 32-bit value ids");
    }
    const auto id = ValueExpressionId{
        static_cast<std::uint32_t>(_values.size())};
    _values.emplace_back(std::move(instruction));
    _value_numbers.emplace(std::move(key), id);
    return id;
  }

  [[nodiscard]] ValueExpressionId zero() {
    if (!_zero.valid()) {
      _zero = emit(ValueInstruction{
          .operation = ValueOperation::bump_offset_zero,
          .source_node = {},
          .result_type = contract::SocketType::floating,
          .parameter = {},
          .operands = {},
          .static_table = {}});
    }
    return _zero;
  }

  [[nodiscard]] ValueExpressionId add_offset(ValueExpressionId offset,
                                             ValueExpressionId width) {
    if (!offset.valid() || offset == zero()) {
      return width;
    }
    return emit(ValueInstruction{
        .operation = ValueOperation::add,
        .source_node = {},
        .result_type = contract::SocketType::floating,
        .parameter = {},
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, offset},
            {value_operand::binary::b, width}}),
        .static_table = {}});
  }

  [[nodiscard]] bool source_operand_valid(ValueExpressionId owner,
                                          ValueExpressionId operand) {
    if (!operand.valid() || operand.value >= owner.value ||
        operand.value >= _source.value_instructions().size()) {
      static_cast<void>(
          fail("source SurfaceProgram is not a strict topological value graph"));
      return false;
    }
    return true;
  }

  [[nodiscard]] ValueExpressionId transform(ValueExpressionId source,
                                            SampleContext context) {
    if (!_result.diagnostic.empty()) {
      return {};
    }
    if (!source.valid() ||
        source.value >= _source.value_instructions().size()) {
      return fail("source SurfaceProgram contains an invalid value reference");
    }
    const auto memo_key = make_context_key(source, context);
    if (const auto iter = _memo.find(memo_key); iter != _memo.end()) {
      return iter->second;
    }

    const auto &source_instruction =
        _source.value_instructions()[source.value];
    if (is_internal_bump_operation(source_instruction.operation)) {
      return fail("Bump expansion received an already-expanded internal opcode");
    }
    for (const auto operand : source_instruction.operands) {
      if (!source_operand_valid(source, operand)) {
        return {};
      }
    }

    ValueExpressionId result;
    if (source_instruction.operation == ValueOperation::bump) {
      const auto height_source =
          source_instruction.operand(value_operand::bump::height);
      const auto center = transform(height_source, context);
      const auto strength = transform(
          source_instruction.operand(value_operand::bump::strength), context);
      const auto distance = transform(
          source_instruction.operand(value_operand::bump::distance), context);
      const auto raw_width = transform(
          source_instruction.operand(value_operand::bump::filter_width),
          context);
      const auto width = emit(ValueInstruction{
          .operation = ValueOperation::bump_filter_width,
          .source_node = source_instruction.source_node,
          .result_type = contract::SocketType::floating,
          .parameter = {},
          .operands = make_value_operands<value_operand::unary>({
              {value_operand::unary::input, raw_width}}),
          .static_table = {}});
      const auto base_zero = zero();
      const auto context_dx = context.root() ? base_zero : context.dx;
      const auto context_dy = context.root() ? base_zero : context.dy;
      const auto sample_x = SampleContext{
          .dx = add_offset(context_dx, width), .dy = context_dy};
      const auto sample_y = SampleContext{
          .dx = context_dx, .dy = add_offset(context_dy, width)};
      const auto height_x = transform(height_source, sample_x);
      const auto height_y = transform(height_source, sample_y);
      const auto normal = transform(
          source_instruction.operand(value_operand::bump::normal), context);
      auto expanded = source_instruction;
      expanded.operation = ValueOperation::bump_samples;
      expanded.operands = make_value_operands<value_operand::bump_samples>({
          {value_operand::bump_samples::height_center, center},
          {value_operand::bump_samples::height_x, height_x},
          {value_operand::bump_samples::height_y, height_y},
          {value_operand::bump_samples::strength, strength},
          {value_operand::bump_samples::distance, distance},
          {value_operand::bump_samples::filter_width, width},
          {value_operand::bump_samples::normal, normal}});
      result = emit(std::move(expanded));
      if (result.valid()) {
        ++_result.bump_count;
      }
    } else {
      auto expanded = source_instruction;
      expanded.operands.clear();
      expanded.operands.reserve(source_instruction.operands.size() +
                                (context.root() ? 0u : 2u));
      const auto contextual_operation =
          context.root() ? source_instruction.operation
                         : sampled_operation(source_instruction.operation);
      if (contextual_operation != source_instruction.operation) {
        if (!context.dx.valid() || !context.dy.valid()) {
          return fail("a differential sample context is only partially defined");
        }
        expanded.operation = contextual_operation;
        expanded.operands.emplace_back(context.dx);
        expanded.operands.emplace_back(context.dy);
      }
      for (const auto operand : source_instruction.operands) {
        expanded.operands.emplace_back(transform(operand, context));
      }
      result = emit(std::move(expanded));
      if (result.valid() &&
          contextual_operation != source_instruction.operation) {
        ++_result.sampled_instruction_count;
      }
    }
    if (result.valid()) {
      _memo.emplace(memo_key, result);
    }
    return result;
  }

  [[nodiscard]] ValueExpressionId remap_root(ValueExpressionId source) {
    if (!source.valid()) {
      return {};
    }
    if (source.value >= _result.root_values.size()) {
      return fail("a SurfaceProgram endpoint references an invalid value");
    }
    return _result.root_values[source.value];
  }

public:
  explicit ExpansionBuilder(const SurfaceProgram &source) : _source{source} {}

  [[nodiscard]] SurfaceBumpExpansion build() {
    _result.root_values.reserve(_source.value_instructions().size());
    for (auto index = std::size_t{0u};
         index < _source.value_instructions().size(); ++index) {
      _result.root_values.emplace_back(transform(
          ValueExpressionId{static_cast<std::uint32_t>(index)}, {}));
      if (!_result.diagnostic.empty()) {
        return std::move(_result);
      }
    }

    auto closures = _source.closure_instructions();
    for (auto &closure : closures) {
      for (const auto field : closure_value_dependency_members) {
        closure.*field = remap_root(closure.*field);
      }
    }
    auto volumes = _source.volume_instructions();
    for (auto &volume : volumes) {
      for (const auto field : volume_value_dependencies) {
        volume.*field = remap_root(volume.*field);
      }
    }
    const auto surface_normal_root =
        remap_root(_source.surface_normal_root());
    const auto displacement_root = remap_root(_source.displacement_root());
    if (!_result.diagnostic.empty()) {
      return std::move(_result);
    }
    _result.program = std::make_shared<const SurfaceProgram>(
        _source.structure_signature(), _source.parameters(), std::move(_values),
        std::move(closures), _source.root(), std::move(volumes),
        _source.volume_root(), surface_normal_root, displacement_root);
    _result.valid = true;
    return std::move(_result);
  }
};

} // namespace

SurfaceBumpExpansion expand_surface_bump_program(
    const SurfaceProgram &program) {
  return ExpansionBuilder{program}.build();
}

} // namespace psycles::compiler
