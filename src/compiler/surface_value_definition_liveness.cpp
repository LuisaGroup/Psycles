#include <psycles/compiler/surface_execution_plan.h>

#include <map>
#include <utility>

namespace psycles::compiler {
namespace {

[[nodiscard]] bool
decode_operand(const SurfaceValueSceneImage &image,
               const SurfaceValueBytecodeInstruction &instruction,
               std::size_t operand_index,
               SurfaceValueOperandAddress &operand) noexcept {
  const auto operand_count = surface_value_operand_count(instruction);
  if (operand_index >= operand_count) {
    return false;
  }
  const auto word_index = operand_index / surface_value_operands_per_word;
  const auto lane = operand_index % surface_value_operands_per_word;
  auto word = instruction.operand_payload;
  if (operand_count > surface_value_inline_operand_capacity) {
    if (instruction.operand_payload >= image.operands.size() ||
        word_index >= image.operands.size() - instruction.operand_payload) {
      return false;
    }
    word = image.operands[instruction.operand_payload + word_index];
  }
  operand = surface_value_operand_from_word(word, lane);
  return operand.valid();
}

} // namespace

SurfaceValueDefinitionLiveness analyze_surface_value_definition_liveness(
    const SurfaceValueSceneImage &image,
    const SurfaceValueProgramDescriptor &program) {
  SurfaceValueDefinitionLiveness result;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    return result;
  };
  if (!image.valid) {
    return reject("the surface value scene image is invalid");
  }
  if (program.instruction_begin > image.instructions.size() ||
      program.instruction_count >
          image.instructions.size() - program.instruction_begin ||
      program.closure_begin > image.closure_instructions.size() ||
      program.closure_count >
          image.closure_instructions.size() - program.closure_begin) {
    return reject("the program descriptor exceeds the scene image");
  }

  result.last_use_offsets.assign(program.instruction_count,
                                 surface_value_definition_no_use);
  std::map<std::uint32_t, std::uint32_t> active_definitions;
  const auto record_local_use = [&](std::uint32_t encoded_address,
                                    std::uint32_t use_offset,
                                    bool invalid_is_optional = false) noexcept {
    const auto address = SurfaceValueAddress{encoded_address};
    if (!address.valid()) {
      return invalid_is_optional;
    }
    if (address.parameter()) {
      return true;
    }
    const auto definition = active_definitions.find(address.encoded());
    if (definition == active_definitions.end() ||
        definition->second >= result.last_use_offsets.size()) {
      return false;
    }
    result.last_use_offsets[definition->second] = use_offset;
    return true;
  };

  for (auto offset = std::uint32_t{}; offset < program.instruction_count;
       ++offset) {
    const auto &instruction =
        image.instructions[program.instruction_begin + offset];
    if (is_surface_value_surface_normal_transition(instruction)) {
      if (!record_local_use(instruction.result, offset)) {
        return reject("a normal commit reads an undefined local value");
      }
      active_definitions.clear();
      continue;
    }
    const auto operand_count = surface_value_operand_count(instruction);
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      auto operand = SurfaceValueOperandAddress{};
      if (!decode_operand(image, instruction, operand_index, operand)) {
        return reject("an ordinary instruction has an invalid operand");
      }
      if (!operand.parameter() &&
          !record_local_use(operand.expanded().encoded(), offset)) {
        return reject("an ordinary instruction reads an undefined local "
                      "value");
      }
    }
    const auto definition = SurfaceValueAddress{instruction.result};
    if (!definition.valid() || definition.parameter() ||
        definition.bank() != surface_value_result_bank(instruction)) {
      return reject("an ordinary instruction has an invalid local result");
    }
    // Operands are read before the result is written. insert_or_assign is
    // therefore the definition-epoch transition for an in-place colored slot.
    active_definitions.insert_or_assign(definition.encoded(), offset);
  }

  const auto terminal_use = program.instruction_count;
  for (auto closure_offset = std::uint32_t{};
       closure_offset < program.closure_count; ++closure_offset) {
    const auto &instruction =
        image.closure_instructions[program.closure_begin + closure_offset];
    if (!surface_closure_is_leaf(instruction)) {
      if (!record_local_use(surface_closure_mix_factor_address(instruction),
                            terminal_use)) {
        return reject("a closure Mix reads an undefined local factor");
      }
      continue;
    }
    const auto operation = surface_closure_operation(instruction);
    const auto operand_begin = surface_closure_leaf_operand_begin(instruction);
    const auto operand_count = surface_closure_operand_count(operation);
    if (operand_begin > image.closure_operands.size() ||
        operand_count > image.closure_operands.size() - operand_begin) {
      return reject("a closure leaf exceeds the operand stream");
    }
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      if (!record_local_use(
              image.closure_operands[operand_begin + operand_index],
              terminal_use, true)) {
        return reject("a closure leaf reads an undefined local value");
      }
    }
  }

  result.valid = true;
  return result;
}

SurfaceValueForwardingPlan plan_surface_value_forwarding(
    const SurfaceValueSceneImage &image,
    const SurfaceValueProgramDescriptor &program) {
  SurfaceValueForwardingPlan result;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    return result;
  };
  const auto liveness =
      analyze_surface_value_definition_liveness(image, program);
  if (!liveness.valid) {
    return reject("definition liveness: " + liveness.diagnostic);
  }
  if (liveness.last_use_offsets.size() != program.instruction_count) {
    return reject("definition liveness is not parallel to the program");
  }
  result.successor_operand_masks.assign(program.instruction_count, 0u);

  // Let D_i be the definition produced by ordinary instruction i and U_i(j)
  // the definition read by successor operand j. Forwarding is legal exactly
  // when {j | U_i(j) = D_i} is non-empty and last_use(D_i) = i + 1. The first
  // equality supplies every substituted use; the second proves that omitting
  // D_i's bank write cannot affect any instruction or closure after the
  // successor. This is a local implementation of that relation, not an
  // opcode-specific peephole.
  for (auto source_offset = std::uint32_t{};
       source_offset + 1u < program.instruction_count; ++source_offset) {
    const auto target_offset = source_offset + 1u;
    const auto &source =
        image.instructions[program.instruction_begin + source_offset];
    const auto &target =
        image.instructions[program.instruction_begin + target_offset];
    if (is_surface_value_surface_normal_transition(source) ||
        is_surface_value_surface_normal_transition(target)) {
      continue;
    }
    const auto definition = SurfaceValueAddress{source.result};
    if (!definition.valid() || definition.parameter() ||
        definition.bank() != surface_value_result_bank(source)) {
      return reject("an ordinary source has an invalid local result");
    }

    auto operand_mask = std::uint32_t{};
    const auto operand_count = surface_value_operand_count(target);
    if (operand_count > surface_value_max_operand_count) {
      return reject("a successor exceeds the forwarding mask domain");
    }
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      auto operand = SurfaceValueOperandAddress{};
      if (!decode_operand(image, target, operand_index, operand)) {
        return reject("a successor has an invalid operand");
      }
      if (!operand.parameter() && operand.expanded() == definition) {
        operand_mask |= std::uint32_t{1u} << operand_index;
      }
    }
    if (liveness.last_use_offsets[source_offset] == target_offset) {
      if (operand_mask == 0u) {
        return reject("definition liveness names a successor without a use");
      }
      result.successor_operand_masks[source_offset] = operand_mask;
    }
  }

  result.valid = true;
  return result;
}

} // namespace psycles::compiler
