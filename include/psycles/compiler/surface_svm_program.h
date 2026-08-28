#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <psycles/compiler/surface_svm_schedule.h>

namespace psycles::compiler {

// One 16-byte instruction stream for value evaluation, closure-weight SSA,
// structured guards and closure setup. Ordinary value records retain their
// established bit-for-bit ABI. Opcodes above the closed ValueOperation range
// select the six unified control records; 0xff remains reserved by the legacy
// automatic-normal transaction and is deliberately not accepted here.
struct SurfaceSvmBytecodeInstruction {
  std::uint32_t control{};
  std::uint32_t payload0{};
  std::uint32_t payload1{};
  std::uint32_t payload2{};

  auto operator<=>(const SurfaceSvmBytecodeInstruction &) const noexcept =
      default;
};

enum class SurfaceSvmBytecodeKind : std::uint8_t {
  value,
  mix_closure,
  add_closure_weight,
  jump_if_one,
  jump_if_zero,
  closure_leaf,
  end,
  invalid,
};

inline constexpr std::uint32_t surface_svm_mix_closure_opcode = 248u;
inline constexpr std::uint32_t surface_svm_add_closure_weight_opcode = 249u;
inline constexpr std::uint32_t surface_svm_jump_if_one_opcode = 250u;
inline constexpr std::uint32_t surface_svm_jump_if_zero_opcode = 251u;
inline constexpr std::uint32_t surface_svm_closure_leaf_opcode = 252u;
inline constexpr std::uint32_t surface_svm_end_opcode = 253u;
inline constexpr std::uint32_t surface_svm_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_svm_invalid_payload =
    ~std::uint32_t{0u};

inline constexpr std::uint32_t surface_svm_mix_left_result_bit = 1u << 8u;
inline constexpr std::uint32_t surface_svm_mix_right_result_bit = 1u << 9u;
inline constexpr std::uint32_t surface_svm_mix_result_mask =
    surface_svm_mix_left_result_bit | surface_svm_mix_right_result_bit;

inline constexpr std::uint32_t surface_svm_closure_control_shift = 8u;
inline constexpr std::uint32_t surface_svm_root_weight_slot =
    surface_svm_invalid_payload;
inline constexpr std::uint32_t surface_svm_packed_weight_slot_mask = 0xffffu;
inline constexpr std::uint32_t surface_svm_invalid_packed_weight_slot =
    0xffffu;

static_assert(static_cast<std::uint32_t>(ValueOperation::ambient_occlusion) <
              surface_svm_mix_closure_opcode);
static_assert((surface_closure_control_mask <<
               surface_svm_closure_control_shift) >>
                  surface_svm_closure_control_shift ==
              surface_closure_control_mask);

[[nodiscard]] constexpr SurfaceSvmBytecodeKind surface_svm_bytecode_kind(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  const auto opcode = instruction.control & surface_svm_opcode_mask;
  if (opcode <=
      static_cast<std::uint32_t>(ValueOperation::ambient_occlusion)) {
    return SurfaceSvmBytecodeKind::value;
  }
  switch (opcode) {
  case surface_svm_mix_closure_opcode:
    return SurfaceSvmBytecodeKind::mix_closure;
  case surface_svm_add_closure_weight_opcode:
    return SurfaceSvmBytecodeKind::add_closure_weight;
  case surface_svm_jump_if_one_opcode:
    return SurfaceSvmBytecodeKind::jump_if_one;
  case surface_svm_jump_if_zero_opcode:
    return SurfaceSvmBytecodeKind::jump_if_zero;
  case surface_svm_closure_leaf_opcode:
    return SurfaceSvmBytecodeKind::closure_leaf;
  case surface_svm_end_opcode:
    return SurfaceSvmBytecodeKind::end;
  default:
    return SurfaceSvmBytecodeKind::invalid;
  }
}

[[nodiscard]] constexpr SurfaceValueBytecodeInstruction
surface_svm_value_instruction(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  return SurfaceValueBytecodeInstruction{
      .control = instruction.control,
      .result = instruction.payload0,
      .operand_payload = instruction.payload1,
      .metadata_index = instruction.payload2};
}

[[nodiscard]] constexpr SurfaceSvmBytecodeInstruction
make_surface_svm_value_instruction(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return SurfaceSvmBytecodeInstruction{
      .control = instruction.control,
      .payload0 = instruction.result,
      .payload1 = instruction.operand_payload,
      .payload2 = instruction.metadata_index};
}

[[nodiscard]] constexpr std::uint32_t surface_svm_closure_control(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  return instruction.control >> surface_svm_closure_control_shift;
}

[[nodiscard]] constexpr std::uint32_t surface_svm_mix_left_weight_slot(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  return instruction.payload2 & surface_svm_packed_weight_slot_mask;
}

[[nodiscard]] constexpr std::uint32_t surface_svm_mix_right_weight_slot(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  return instruction.payload2 >> 16u;
}

// Side streams keep their physical element type unambiguous: value overflow
// operands are packed pairs of 16-bit addresses, while closure operands are
// full 32-bit typed addresses. Metadata and static tables retain the existing
// value evaluator ABI.
struct SurfaceSvmProgramImage {
  bool valid{};
  std::string diagnostic;
  SurfaceClosureEndpointMask endpoints{};
  std::vector<SurfaceSvmBytecodeInstruction> instructions;
  std::vector<std::uint32_t> value_operands;
  std::vector<SurfaceValueBytecodeMetadata> value_metadata;
  std::vector<float> static_data;
  std::vector<std::uint32_t> closure_operands;
  std::vector<std::uint32_t> value_addresses;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t flags{};
  std::uint32_t value_instruction_count{};
  std::uint32_t mix_instruction_count{};
  std::uint32_t weight_add_instruction_count{};
  std::uint32_t conditional_branch_count{};
  std::uint32_t closure_leaf_count{};
  std::uint32_t used_closure_operations{};
  PrincipledClosureFeatureMask used_principled_features{};
};

// Returns an empty string iff every serialized relation is valid: value ABI,
// forward CFG, exact typed read-before-write initialization, weight/value slot
// ownership, dense side streams, closure operand banks and aggregate masks.
[[nodiscard]] std::string validate_surface_svm_program_image(
    const SurfaceSvmProgramImage &program);

// Transactionally lowers one proven structured schedule. Passthrough aliases
// have no bytecode record; jump targets are remapped across those erased
// semantic points. No material value is baked into the image.
[[nodiscard]] SurfaceSvmProgramImage lower_surface_svm_program(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    const SurfaceSvmSchedulePlan &schedule,
    const SurfaceSvmStoragePlan &storage);

} // namespace psycles::compiler
