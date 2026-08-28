#include "surface_value_variant_interner.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace psycles::compiler::detail {
namespace {

[[nodiscard]] std::uint64_t type_key(SurfaceValueBank bank) noexcept {
  return static_cast<std::uint64_t>(bank);
}

[[nodiscard]] std::vector<std::uint64_t>
make_static_variant_key(const SurfaceProgram &program,
                        const ValueInstruction &instruction) {
  std::vector<std::uint64_t> key;
  key.reserve(6u + instruction.operands.size());
  auto result_bank = SurfaceValueBank::scalar;
  [[maybe_unused]] const auto result_supported =
      classify_surface_value_type(instruction.result_type, result_bank);
  key.emplace_back(static_cast<std::uint64_t>(instruction.operation));
  key.emplace_back(type_key(result_bank));
  key.emplace_back(surface_value_svm_evaluator_static_u0(
      instruction.operation, instruction.static_u0));
  key.emplace_back(surface_value_svm_evaluator_static_u1(
      instruction.operation, instruction.static_u1));
  key.emplace_back(instruction.operands.size());
  for (const auto operand : instruction.operands) {
    auto bank = SurfaceValueBank::scalar;
    [[maybe_unused]] const auto operand_supported = classify_surface_value_type(
        program.value_instructions()[operand.value].result_type, bank);
    key.emplace_back(type_key(bank));
  }
  key.emplace_back(
      surface_value_handler_static_table_size(instruction.operation));
  return key;
}

template <typename DecodeInstruction>
[[nodiscard]] std::string
populate_routes(std::size_t instruction_count,
                std::span<const std::uint32_t> instruction_variants,
                std::vector<SurfaceValueStaticVariant> &variants,
                DecodeInstruction &&decode_instruction) {
  constexpr auto local_route_bit = std::uint8_t{1u};
  constexpr auto parameter_route_bit = std::uint8_t{2u};
  constexpr auto dynamic_route_bits =
      static_cast<std::uint8_t>(local_route_bit | parameter_route_bit);

  if (instruction_variants.size() != instruction_count) {
    return "the immutable-variant stream is not parallel to the instructions";
  }

  // Coordinate-wise join in P({local, parameter}). A singleton permits one
  // direct load. The two-element top is precisely the dynamic route; bottom
  // is rejected because every published evaluator operand must be observed.
  std::vector<std::vector<std::uint8_t>> domains;
  domains.reserve(variants.size());
  for (const auto &variant : variants) {
    domains.emplace_back(variant.operand_types.size(), std::uint8_t{});
  }

  for (auto instruction_index = std::size_t{};
       instruction_index < instruction_count; ++instruction_index) {
    const auto decoded = decode_instruction(instruction_index);
    if (!decoded.has_value()) {
      if (instruction_variants[instruction_index] !=
          SurfaceValueAddress::invalid_value) {
        return "a non-value instruction has an evaluator variant";
      }
      continue;
    }

    const auto variant_index = instruction_variants[instruction_index];
    if (variant_index >= variants.size()) {
      return "a value instruction has an invalid evaluator variant";
    }
    const auto operand_count = surface_value_operand_count(*decoded);
    auto &variant_domains = domains[variant_index];
    if (operand_count != variant_domains.size()) {
      return "an evaluator variant changed operand arity";
    }
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      const auto compact = decode_instruction.operand(instruction_index,
                                                      *decoded, operand_index);
      if (!compact.has_value() || !compact->valid()) {
        return "an evaluator variant observes an invalid operand";
      }
      const auto atom =
          compact->parameter() ? parameter_route_bit : local_route_bit;
      variant_domains[operand_index] =
          static_cast<std::uint8_t>(variant_domains[operand_index] | atom);
    }
  }

  for (auto variant_index = std::size_t{}; variant_index < variants.size();
       ++variant_index) {
    auto &variant = variants[variant_index];
    const auto &variant_domains = domains[variant_index];
    variant.operand_routes.clear();
    variant.operand_routes.reserve(variant_domains.size());
    for (const auto domain : variant_domains) {
      if (domain == local_route_bit) {
        variant.operand_routes.emplace_back(SurfaceValueOperandRoute::local);
      } else if (domain == parameter_route_bit) {
        variant.operand_routes.emplace_back(
            SurfaceValueOperandRoute::parameter);
      } else if (domain == dynamic_route_bits) {
        variant.operand_routes.emplace_back(SurfaceValueOperandRoute::dynamic);
      } else {
        return "an evaluator variant has an unobserved operand position";
      }
    }
  }
  return {};
}

struct LegacyInstructionDecoder {
  const SurfaceValueSceneImage &image;

  [[nodiscard]] std::optional<SurfaceValueBytecodeInstruction>
  operator()(std::size_t index) const noexcept {
    const auto instruction = image.instructions[index];
    return is_surface_value_surface_normal_transition(instruction)
               ? std::nullopt
               : std::optional{instruction};
  }

  [[nodiscard]] std::optional<SurfaceValueOperandAddress>
  operand(std::size_t, const SurfaceValueBytecodeInstruction &instruction,
          std::size_t operand_index) const noexcept {
    const auto count = surface_value_operand_count(instruction);
    auto word = instruction.operand_payload;
    if (count > surface_value_inline_operand_capacity) {
      const auto word_index = operand_index / surface_value_operands_per_word;
      if (instruction.operand_payload >= image.operands.size() ||
          word_index >= image.operands.size() - instruction.operand_payload) {
        return std::nullopt;
      }
      word = image.operands[instruction.operand_payload + word_index];
    }
    return surface_value_operand_from_word(
        word, operand_index % surface_value_operands_per_word);
  }
};

struct UnifiedInstructionDecoder {
  const SurfaceSvmSceneImage &image;

  [[nodiscard]] std::optional<SurfaceValueBytecodeInstruction>
  operator()(std::size_t index) const noexcept {
    const auto instruction = image.instructions[index];
    return surface_svm_bytecode_kind(instruction) ==
                   SurfaceSvmBytecodeKind::value
               ? std::optional{surface_svm_value_instruction(instruction)}
               : std::nullopt;
  }

  [[nodiscard]] std::optional<SurfaceValueOperandAddress>
  operand(std::size_t, const SurfaceValueBytecodeInstruction &instruction,
          std::size_t operand_index) const noexcept {
    const auto count = surface_value_operand_count(instruction);
    auto word = instruction.operand_payload;
    if (count > surface_value_inline_operand_capacity) {
      const auto word_index = operand_index / surface_value_operands_per_word;
      if (instruction.operand_payload >= image.value_operands.size() ||
          word_index >=
              image.value_operands.size() - instruction.operand_payload) {
        return std::nullopt;
      }
      word = image.value_operands[instruction.operand_payload + word_index];
    }
    return surface_value_operand_from_word(
        word, operand_index % surface_value_operands_per_word);
  }
};

} // namespace

bool SurfaceValueVariantInterner::intern(const SurfaceProgram &program,
                                         const ValueInstruction &instruction,
                                         std::uint32_t &variant_index,
                                         std::string &diagnostic) {
  if (_finished) {
    diagnostic = "the value evaluator interner was already finalized";
    return false;
  }
  if (instruction.operation == ValueOperation::parameter ||
      instruction.operation == ValueOperation::passthrough) {
    diagnostic = "a non-executable value operation reached the evaluator "
                 "interner";
    return false;
  }
  if (instruction.operands.size() !=
      value_operation_operand_count(instruction.operation)) {
    diagnostic = "a value evaluator has an invalid operand arity";
    return false;
  }
  if (!surface_value_svm_static_fields_valid(instruction.operation,
                                             instruction.static_u0,
                                             instruction.static_u1)) {
    diagnostic =
        "a value evaluator has immutable fields outside its SVM contract";
    return false;
  }
  if (!surface_value_static_table_shape_valid(
          instruction.operation, instruction.static_table.size())) {
    diagnostic =
        "a value evaluator has an invalid statically indexed table shape";
    return false;
  }
  auto result_bank = SurfaceValueBank::scalar;
  if (!classify_surface_value_type(instruction.result_type, result_bank)) {
    diagnostic = "a value evaluator has an unsupported result execution type";
    return false;
  }
  for (const auto operand : instruction.operands) {
    if (!operand.valid() ||
        operand.value >= program.value_instructions().size()) {
      diagnostic = "a value evaluator has an invalid source operand";
      return false;
    }
    auto operand_bank = SurfaceValueBank::scalar;
    if (!classify_surface_value_type(
            program.value_instructions()[operand.value].result_type,
            operand_bank)) {
      diagnostic =
          "a value evaluator has an unsupported operand execution type";
      return false;
    }
  }

  auto key = make_static_variant_key(program, instruction);
  const auto [iter, inserted] = _indices.try_emplace(
      std::move(key), static_cast<std::uint32_t>(_variants.size()));
  variant_index = iter->second;
  const auto immediate =
      static_cast<std::uint16_t>(make_surface_value_svm_immediate(
          instruction.operation, instruction.static_u0, instruction.static_u1));
  const auto device_key = make_surface_value_handler_key(
      instruction.operation, result_bank, immediate);
  const auto [handler, handler_inserted] =
      _handler_indices.try_emplace(device_key, variant_index);
  if (handler->second != variant_index) {
    diagnostic =
        "two distinct typed evaluator shapes have the same device handler "
        "key";
    if (inserted) {
      _indices.erase(iter);
    }
    return false;
  }
  if (!inserted) {
    auto &immediates = _variants[variant_index].svm_immediates;
    if (std::find(immediates.begin(), immediates.end(), immediate) ==
        immediates.end()) {
      immediates.emplace_back(immediate);
    }
    return true;
  }
  if (_variants.size() >= std::numeric_limits<std::uint32_t>::max()) {
    diagnostic = "the scene has too many immutable value variants";
    _indices.erase(iter);
    if (handler_inserted) {
      _handler_indices.erase(handler);
    }
    return false;
  }

  auto normalized = instruction;
  normalized.result_type = canonical_surface_value_type(result_bank);
  // Every executable ParameterId and float/static-table payload is bytecode
  // data. None of these fields changes the handler AST. The only statically
  // indexed tables retain a zero-filled shape witness so ValueNode records the
  // exact fixed number of accesses without capturing material contents.
  normalized.parameter = {};
  normalized.static_f0 = 0.0f;
  normalized.static_f1 = 0.0f;
  normalized.static_u0 = surface_value_svm_evaluator_static_u0(
      normalized.operation, normalized.static_u0);
  normalized.static_u1 = surface_value_svm_evaluator_static_u1(
      normalized.operation, normalized.static_u1);
  normalized.static_table.assign(
      surface_value_handler_static_table_size(normalized.operation), 0.0f);

  std::vector<contract::SocketType> operand_types;
  operand_types.reserve(normalized.operands.size());
  for (auto operand_index = std::size_t{};
       operand_index < normalized.operands.size(); ++operand_index) {
    const auto source = normalized.operands[operand_index];
    auto operand_bank = SurfaceValueBank::scalar;
    [[maybe_unused]] const auto supported = classify_surface_value_type(
        program.value_instructions()[source.value].result_type, operand_bank);
    operand_types.emplace_back(canonical_surface_value_type(operand_bank));
    normalized.operands[operand_index] =
        ValueExpressionId{static_cast<std::uint32_t>(operand_index)};
  }
  normalized.source_node = {};
  _variants.emplace_back(
      SurfaceValueStaticVariant{.instruction = std::move(normalized),
                                .operand_types = std::move(operand_types),
                                .svm_immediates = {immediate},
                                .operand_routes = {}});
  return true;
}

std::vector<SurfaceValueStaticVariant> SurfaceValueVariantInterner::finish() {
  _finished = true;
  for (auto &variant : _variants) {
    std::sort(variant.svm_immediates.begin(), variant.svm_immediates.end());
    variant.svm_immediates.erase(std::unique(variant.svm_immediates.begin(),
                                             variant.svm_immediates.end()),
                                 variant.svm_immediates.end());
  }
  return std::move(_variants);
}

std::string populate_surface_value_operand_routes(
    const SurfaceValueSceneImage &image,
    std::span<const std::uint32_t> instruction_variants,
    std::vector<SurfaceValueStaticVariant> &variants) {
  return populate_routes(image.instructions.size(), instruction_variants,
                         variants, LegacyInstructionDecoder{image});
}

std::string populate_surface_value_operand_routes(
    const SurfaceSvmSceneImage &image,
    std::span<const std::uint32_t> instruction_variants,
    std::vector<SurfaceValueStaticVariant> &variants) {
  return populate_routes(image.instructions.size(), instruction_variants,
                         variants, UnifiedInstructionDecoder{image});
}

} // namespace psycles::compiler::detail
