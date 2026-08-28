#include <psycles/compiler/surface_svm_program.h>

#include "surface_value_variant_interner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

inline constexpr auto invalid_source = SurfaceValueAddress::invalid_value;

[[nodiscard]] SurfaceSvmExecutableScene
reject_executable_scene(std::string diagnostic) {
  SurfaceSvmExecutableScene result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] constexpr std::size_t bank_index(SurfaceValueBank bank) noexcept {
  switch (bank) {
  case SurfaceValueBank::scalar:
    return 0u;
  case SurfaceValueBank::vector:
    return 1u;
  case SurfaceValueBank::unsigned_integer:
    return 2u;
  }
  return 0u;
}

[[nodiscard]] bool float_bits_equal(float lhs, float rhs) noexcept {
  return std::bit_cast<std::uint32_t>(lhs) == std::bit_cast<std::uint32_t>(rhs);
}

[[nodiscard]] bool
source_representatives(const SurfaceProgram &program,
                       std::vector<std::uint32_t> &representatives,
                       std::string &diagnostic) {
  const auto &values = program.value_instructions();
  representatives.resize(values.size(), invalid_source);
  for (auto index = std::size_t{}; index < values.size(); ++index) {
    const auto &value = values[index];
    if (value.operands.size() !=
        value_operation_operand_count(value.operation)) {
      diagnostic = "source value " + std::to_string(index) +
                   " has an invalid operand arity";
      return false;
    }
    auto result_bank = SurfaceValueBank::scalar;
    if (!classify_surface_value_type(value.result_type, result_bank)) {
      diagnostic = "source value " + std::to_string(index) +
                   " has no physical execution bank";
      return false;
    }
    for (const auto operand : value.operands) {
      if (!operand.valid() || operand.value >= index) {
        diagnostic = "source value " + std::to_string(index) +
                     " violates strict topological order";
        return false;
      }
    }

    if (value.operation == ValueOperation::parameter) {
      if (!value.parameter.valid() ||
          value.parameter.value >= program.parameters().size()) {
        diagnostic = "source parameter " + std::to_string(index) +
                     " has an invalid binding";
        return false;
      }
      const auto &parameter = program.parameters()[value.parameter.value];
      if (parameter.id != value.parameter ||
          parameter.type != value.result_type) {
        diagnostic = "source parameter " + std::to_string(index) +
                     " disagrees with its descriptor";
        return false;
      }
      representatives[index] = static_cast<std::uint32_t>(index);
      continue;
    }
    if (value.operation == ValueOperation::passthrough) {
      const auto source = value.operands.front().value;
      const auto representative = representatives[source];
      if (representative == invalid_source) {
        diagnostic = "source passthrough " + std::to_string(index) +
                     " has no quotient representative";
        return false;
      }
      auto source_bank = SurfaceValueBank::scalar;
      if (!classify_surface_value_type(values[representative].result_type,
                                       source_bank) ||
          source_bank != result_bank) {
        diagnostic = "source passthrough " + std::to_string(index) +
                     " crosses physical execution banks";
        return false;
      }
      representatives[index] = representative;
      continue;
    }
    representatives[index] = static_cast<std::uint32_t>(index);
  }
  return true;
}

[[nodiscard]] std::optional<SurfaceValueOperandAddress>
decode_operand(const SurfaceSvmProgramImage &image,
               const SurfaceValueBytecodeInstruction &instruction,
               std::size_t operand_index) noexcept {
  const auto operand_count = surface_value_operand_count(instruction);
  auto word = instruction.operand_payload;
  if (operand_count > surface_value_inline_operand_capacity) {
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

[[nodiscard]] bool
metadata_matches_source(const ValueInstruction &source,
                        const SurfaceValueBytecodeInstruction &instruction,
                        const SurfaceSvmProgramImage &image,
                        std::vector<std::uint32_t> &uses,
                        std::string &diagnostic) {
  const auto has_metadata =
      surface_value_operation_uses_metadata_static_u0(source.operation) ||
      source.static_u0 != 0u || source.static_u1 != 0u ||
      std::bit_cast<std::uint32_t>(source.static_f0) != 0u ||
      std::bit_cast<std::uint32_t>(source.static_f1) != 0u ||
      source.parameter.valid() || !source.static_table.empty();
  if (!has_metadata) {
    if (instruction.metadata_index != invalid_source) {
      diagnostic = "a metadata-free source owns a bytecode metadata record";
      return false;
    }
    return true;
  }
  if (instruction.metadata_index >= image.value_metadata.size()) {
    diagnostic = "a source metadata record is missing from bytecode";
    return false;
  }
  auto &use_count = uses[instruction.metadata_index];
  if (use_count == std::numeric_limits<std::uint32_t>::max()) {
    diagnostic = "a metadata use count exceeds the proof encoding";
    return false;
  }
  ++use_count;
  const auto &metadata = image.value_metadata[instruction.metadata_index];
  const auto expected_parameter =
      source.parameter.valid() ? source.parameter.value : invalid_source;
  if (metadata.static_u0 != source.static_u0 ||
      metadata.static_u1 != source.static_u1 ||
      !float_bits_equal(metadata.static_f0, source.static_f0) ||
      !float_bits_equal(metadata.static_f1, source.static_f1) ||
      metadata.parameter != expected_parameter || metadata.reserved != 0u ||
      metadata.static_table_count != source.static_table.size() ||
      metadata.static_table_begin > image.static_data.size() ||
      metadata.static_table_count >
          image.static_data.size() - metadata.static_table_begin) {
    diagnostic = "bytecode metadata disagrees with its exact source value";
    return false;
  }
  for (auto offset = std::size_t{}; offset < source.static_table.size();
       ++offset) {
    if (!float_bits_equal(
            image.static_data[metadata.static_table_begin + offset],
            source.static_table[offset])) {
      diagnostic =
          "bytecode static-table data disagrees with its exact source value";
      return false;
    }
  }
  return true;
}

using DefinitionState = std::array<std::vector<std::uint32_t>, 3u>;

[[nodiscard]] DefinitionState
make_empty_state(const SurfaceSvmProgramImage &image) {
  return DefinitionState{
      std::vector<std::uint32_t>(image.stack_lanes, invalid_source),
      std::vector<std::uint32_t>(image.stack_lanes, invalid_source),
      std::vector<std::uint32_t>(image.stack_lanes, invalid_source)};
}

void clear_state(DefinitionState &state) {
  for (auto &bank : state) {
    std::fill(bank.begin(), bank.end(), invalid_source);
  }
}

void kill_scalar_slot(DefinitionState &state, std::uint32_t slot) {
  if (slot != surface_svm_invalid_payload &&
      slot != surface_svm_invalid_packed_weight_slot &&
      slot < state[bank_index(SurfaceValueBank::scalar)].size()) {
    state[bank_index(SurfaceValueBank::scalar)][slot] = invalid_source;
  }
}

void merge_state(std::optional<DefinitionState> &destination,
                 const DefinitionState &source) {
  if (!destination.has_value()) {
    destination = source;
    return;
  }
  for (auto bank = std::size_t{}; bank < destination->size(); ++bank) {
    auto &target_bank = (*destination)[bank];
    const auto &source_bank = source[bank];
    for (auto slot = std::size_t{}; slot < target_bank.size(); ++slot) {
      if (target_bank[slot] != source_bank[slot]) {
        target_bank[slot] = invalid_source;
      }
    }
  }
}

[[nodiscard]] bool validate_value_record(
    const SurfaceProgram &program, const SurfaceSvmProgramImage &image,
    const std::vector<std::uint32_t> &representatives, std::uint32_t source_id,
    const SurfaceValueBytecodeInstruction &record, DefinitionState &state,
    std::vector<std::uint32_t> &metadata_uses, std::string &diagnostic) {
  const auto &values = program.value_instructions();
  if (source_id >= values.size()) {
    diagnostic = "a Value record has no valid source expression";
    return false;
  }
  const auto &source = values[source_id];
  if (representatives[source_id] != source_id ||
      source.operation == ValueOperation::parameter ||
      source.operation == ValueOperation::passthrough) {
    diagnostic = "a Value record names a non-executable quotient member";
    return false;
  }
  if (surface_value_operation(record) != source.operation ||
      surface_value_operand_count(record) != source.operands.size()) {
    diagnostic = "a Value record opcode or arity disagrees with its source";
    return false;
  }
  auto expected_result_bank = SurfaceValueBank::scalar;
  if (!classify_surface_value_type(source.result_type, expected_result_bank) ||
      surface_value_result_bank(record) != expected_result_bank) {
    diagnostic = "a Value record result bank disagrees with its source";
    return false;
  }
  const auto expected_immediate =
      static_cast<std::uint16_t>(make_surface_value_svm_immediate(
          source.operation, source.static_u0, source.static_u1));
  if (surface_value_svm_immediate(record) != expected_immediate) {
    diagnostic = "a Value record immediate disagrees with its source";
    return false;
  }
  if (!metadata_matches_source(source, record, image, metadata_uses,
                               diagnostic)) {
    return false;
  }

  for (auto operand_index = std::size_t{};
       operand_index < source.operands.size(); ++operand_index) {
    const auto compact = decode_operand(image, record, operand_index);
    if (!compact.has_value() || !compact->valid()) {
      diagnostic = "a Value record has an invalid physical operand";
      return false;
    }
    const auto actual = compact->expanded();
    const auto expected_source =
        representatives[source.operands[operand_index].value];
    if (expected_source == invalid_source) {
      diagnostic = "a Value record operand has no quotient representative";
      return false;
    }
    const auto &expected_value = values[expected_source];
    auto expected_bank = SurfaceValueBank::scalar;
    if (!classify_surface_value_type(expected_value.result_type,
                                     expected_bank) ||
        actual.bank() != expected_bank) {
      diagnostic = "a Value record operand bank disagrees with its source";
      return false;
    }
    if (actual.parameter()) {
      if (expected_value.operation != ValueOperation::parameter ||
          !expected_value.parameter.valid() ||
          actual.index() != expected_value.parameter.value) {
        diagnostic =
            "a Value record parameter operand disagrees with its source";
        return false;
      }
    } else {
      const auto &bank = state[bank_index(actual.bank())];
      if (actual.index() >= bank.size() ||
          bank[actual.index()] != expected_source) {
        diagnostic =
            "a Value record local operand does not contain its source value";
        return false;
      }
    }
  }

  const auto result = SurfaceValueAddress{record.result};
  if (!result.valid() || result.parameter() ||
      result.bank() != expected_result_bank) {
    diagnostic = "a Value record has no exact local result address";
    return false;
  }
  auto &result_bank = state[bank_index(result.bank())];
  if (result.index() >= result_bank.size()) {
    diagnostic = "a Value record result exceeds its physical bank";
    return false;
  }
  result_bank[result.index()] = source_id;
  return true;
}

[[nodiscard]] bool
validate_program_and_intern(const SurfaceSvmEvaluatorProgramInput &input,
                            detail::SurfaceValueVariantInterner &interner,
                            std::vector<std::uint32_t> &instruction_variants,
                            std::string &diagnostic) {
  if (input.program == nullptr || input.image == nullptr) {
    diagnostic = "evaluator provenance input is incomplete";
    return false;
  }
  const auto &program = *input.program;
  const auto &image = *input.image;
  if (const auto image_diagnostic = validate_surface_svm_program_image(image);
      !image_diagnostic.empty()) {
    diagnostic = "unified image: " + image_diagnostic;
    return false;
  }
  if (input.instruction_sources.size() != image.instructions.size()) {
    diagnostic = "source provenance is not parallel to unified bytecode";
    return false;
  }

  std::vector<std::uint32_t> representatives;
  if (!source_representatives(program, representatives, diagnostic)) {
    return false;
  }
  std::vector<std::uint32_t> metadata_uses(image.value_metadata.size(), 0u);
  std::vector<std::optional<DefinitionState>> incoming(
      image.instructions.size());
  if (incoming.empty()) {
    diagnostic = "a unified evaluator program is empty";
    return false;
  }
  incoming.front() = make_empty_state(image);
  auto set_normal_count = std::size_t{};

  for (auto pc = std::size_t{}; pc < image.instructions.size(); ++pc) {
    if (!incoming[pc].has_value()) {
      diagnostic =
          "unified PC " + std::to_string(pc) + " is structurally unreachable";
      return false;
    }
    auto state = std::move(*incoming[pc]);
    const auto &instruction = image.instructions[pc];
    const auto source = input.instruction_sources[pc];
    const auto kind = surface_svm_bytecode_kind(instruction);
    if (kind == SurfaceSvmBytecodeKind::value) {
      if (!validate_value_record(program, image, representatives, source,
                                 surface_svm_value_instruction(instruction),
                                 state, metadata_uses, diagnostic)) {
        diagnostic = "unified PC " + std::to_string(pc) + ": " + diagnostic;
        return false;
      }
      auto variant = invalid_source;
      if (!interner.intern(program, program.value_instructions()[source],
                           variant, diagnostic)) {
        diagnostic = "unified PC " + std::to_string(pc) +
                     ": evaluator interning: " + diagnostic;
        return false;
      }
      instruction_variants.emplace_back(variant);
    } else {
      if (source != invalid_source) {
        diagnostic = "unified PC " + std::to_string(pc) +
                     " assigns source provenance to a non-value record";
        return false;
      }
      instruction_variants.emplace_back(invalid_source);
      switch (kind) {
      case SurfaceSvmBytecodeKind::mix_closure:
        kill_scalar_slot(state, surface_svm_mix_left_weight_slot(instruction));
        kill_scalar_slot(state, surface_svm_mix_right_weight_slot(instruction));
        break;
      case SurfaceSvmBytecodeKind::add_closure_weight:
        kill_scalar_slot(state, instruction.payload2);
        break;
      case SurfaceSvmBytecodeKind::set_normal: {
        ++set_normal_count;
        if (!input.surface_normal_output.valid() ||
            input.surface_normal_output.value >= representatives.size()) {
          diagnostic = "SetNormal has no source output proof";
          return false;
        }
        const auto expected =
            representatives[input.surface_normal_output.value];
        const auto actual = SurfaceValueAddress{instruction.payload0};
        if (expected == invalid_source || !actual.valid() ||
            actual.bank() != SurfaceValueBank::vector) {
          diagnostic =
              "SetNormal does not commit its exact source vector value";
          return false;
        }
        const auto &expected_value = program.value_instructions()[expected];
        const auto exact_parameter =
            actual.parameter() &&
            expected_value.operation == ValueOperation::parameter &&
            expected_value.parameter.valid() &&
            actual.index() == expected_value.parameter.value;
        const auto &vector_state = state[bank_index(SurfaceValueBank::vector)];
        const auto exact_local = !actual.parameter() &&
                                 actual.index() < vector_state.size() &&
                                 vector_state[actual.index()] == expected;
        if (!exact_parameter && !exact_local) {
          diagnostic =
              "SetNormal does not commit its exact source vector value";
          return false;
        }
        clear_state(state);
        break;
      }
      case SurfaceSvmBytecodeKind::value:
      case SurfaceSvmBytecodeKind::jump_if_one:
      case SurfaceSvmBytecodeKind::jump_if_zero:
      case SurfaceSvmBytecodeKind::closure_leaf:
      case SurfaceSvmBytecodeKind::end:
        break;
      case SurfaceSvmBytecodeKind::invalid:
        diagnostic =
            "unified PC " + std::to_string(pc) + " has an invalid opcode";
        return false;
      }
    }

    const auto propagate = [&](std::uint32_t successor) {
      if (successor >= image.instructions.size()) {
        diagnostic = "unified CFG successor exceeds the program";
        return false;
      }
      merge_state(incoming[successor], state);
      return true;
    };
    if (kind == SurfaceSvmBytecodeKind::jump_if_one ||
        kind == SurfaceSvmBytecodeKind::jump_if_zero) {
      if (!propagate(static_cast<std::uint32_t>(pc + 1u)) ||
          !propagate(instruction.payload1)) {
        return false;
      }
    } else if (kind != SurfaceSvmBytecodeKind::end &&
               !propagate(static_cast<std::uint32_t>(pc + 1u))) {
      return false;
    }
  }

  if ((set_normal_count == 0u) == input.surface_normal_output.valid()) {
    diagnostic = "SetNormal cardinality disagrees with its source-output proof";
    return false;
  }
  if (std::any_of(metadata_uses.begin(), metadata_uses.end(),
                  [](std::uint32_t count) noexcept { return count != 1u; })) {
    diagnostic = "unified bytecode metadata is not an exact one-use partition";
    return false;
  }
  auto static_cursor = std::size_t{};
  for (const auto &metadata : image.value_metadata) {
    if (static_cursor > image.static_data.size() ||
        metadata.static_table_begin != static_cursor ||
        metadata.static_table_count >
            image.static_data.size() - static_cursor) {
      diagnostic = "unified bytecode static tables are not a dense partition";
      return false;
    }
    static_cursor += metadata.static_table_count;
  }
  if (static_cursor != image.static_data.size()) {
    diagnostic = "unified bytecode has an unowned static-data suffix";
    return false;
  }
  return true;
}

} // namespace

SurfaceSvmExecutableScene build_surface_svm_executable_scene(
    std::span<const SurfaceSvmEvaluatorProgramInput> inputs) {
  SurfaceSvmExecutableScene result;
  detail::SurfaceValueVariantInterner interner;
  std::vector<SurfaceSvmProgramImage> images;
  images.reserve(inputs.size());
  for (auto index = std::size_t{}; index < inputs.size(); ++index) {
    std::string diagnostic;
    if (!validate_program_and_intern(inputs[index], interner,
                                     result.instruction_variants, diagnostic)) {
      return reject_executable_scene("surface program " +
                                     std::to_string(index) + ": " + diagnostic);
    }
    images.emplace_back(*inputs[index].image);
  }

  result.value_variants = interner.finish();
  result.image = build_surface_svm_scene_image(images);
  if (!result.image.valid) {
    return reject_executable_scene("unified surface scene: " +
                                   result.image.diagnostic);
  }
  if (result.instruction_variants.size() != result.image.instructions.size()) {
    return reject_executable_scene(
        "the evaluator-variant stream is not parallel to the unified scene");
  }
  if (auto diagnostic = detail::populate_surface_value_operand_routes(
          result.image, result.instruction_variants, result.value_variants);
      !diagnostic.empty()) {
    return reject_executable_scene(std::move(diagnostic));
  }
  result.valid = true;
  return result;
}

} // namespace psycles::compiler
