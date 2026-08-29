#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_svm_schedule.h>

namespace psycles::compiler {

// Cycles 5.2 intern/cycles/kernel/svm/types.h::SVM_STACK_SIZE. Local value
// addresses name 32-bit lanes in this closed domain; 255 remains the invalid
// uint8 stack offset in Cycles and is never a legal lane here.
inline constexpr std::uint32_t surface_svm_stack_lane_capacity = 255u;

// One 16-byte instruction stream for value evaluation, closure-weight SSA,
// structured guards and closure setup. Ordinary value records retain their
// established bit-for-bit ABI. Opcodes above the closed ValueOperation range
// select the unified control records. The established 0xff SetNormal record
// is retained as the explicit transaction boundary between an automatic-
// normal prefix and the structured closure body; 0xfe remains reserved.
struct SurfaceSvmBytecodeInstruction {
  std::uint32_t control{};
  std::uint32_t payload0{};
  std::uint32_t payload1{};
  std::uint32_t payload2{};

  auto
  operator<=>(const SurfaceSvmBytecodeInstruction &) const noexcept = default;
};

enum class SurfaceSvmBytecodeKind : std::uint8_t {
  value,
  mix_closure,
  add_closure_weight,
  jump_if_one,
  jump_if_zero,
  closure_leaf,
  set_normal,
  end,
  invalid,
};

inline constexpr std::uint32_t surface_svm_mix_closure_opcode = 248u;
inline constexpr std::uint32_t surface_svm_add_closure_weight_opcode = 249u;
inline constexpr std::uint32_t surface_svm_jump_if_one_opcode = 250u;
inline constexpr std::uint32_t surface_svm_jump_if_zero_opcode = 251u;
inline constexpr std::uint32_t surface_svm_closure_leaf_opcode = 252u;
inline constexpr std::uint32_t surface_svm_end_opcode = 253u;
inline constexpr std::uint32_t surface_svm_reserved_opcode = 254u;
inline constexpr std::uint32_t surface_svm_set_normal_opcode =
    surface_value_surface_normal_transition_control;
inline constexpr std::uint32_t surface_svm_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_svm_invalid_payload = ~std::uint32_t{0u};

inline constexpr std::uint32_t surface_svm_mix_left_result_bit = 1u << 8u;
inline constexpr std::uint32_t surface_svm_mix_right_result_bit = 1u << 9u;
inline constexpr std::uint32_t surface_svm_mix_result_mask =
    surface_svm_mix_left_result_bit | surface_svm_mix_right_result_bit;

inline constexpr std::uint32_t surface_svm_closure_control_shift = 8u;
inline constexpr std::uint32_t surface_svm_root_weight_slot =
    surface_svm_invalid_payload;
inline constexpr std::uint32_t surface_svm_packed_weight_slot_mask = 0xffffu;
inline constexpr std::uint32_t surface_svm_invalid_packed_weight_slot = 0xffffu;

static_assert(surface_svm_value_opcode_count <
              surface_svm_mix_closure_opcode);
static_assert((surface_closure_control_mask
               << surface_svm_closure_control_shift) >>
                  surface_svm_closure_control_shift ==
              surface_closure_control_mask);

[[nodiscard]] constexpr SurfaceSvmBytecodeKind surface_svm_bytecode_kind(
    const SurfaceSvmBytecodeInstruction &instruction) noexcept {
  const auto opcode = instruction.control & surface_svm_opcode_mask;
  if (opcode < surface_svm_value_opcode_count) {
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
  case surface_svm_set_normal_opcode:
    return SurfaceSvmBytecodeKind::set_normal;
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
  return SurfaceSvmBytecodeInstruction{.control = instruction.control,
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
  std::uint32_t stack_lanes{};
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t flags{};
  std::uint32_t value_instruction_count{};
  std::uint32_t mix_instruction_count{};
  std::uint32_t weight_add_instruction_count{};
  std::uint32_t conditional_branch_count{};
  std::uint32_t closure_leaf_count{};
  std::uint32_t surface_normal_transition_count{};
  std::uint32_t used_closure_operations{};
  PrincipledClosureFeatureMask used_principled_features{};
};

// Compact device descriptor. Side-stream ranges are host validation data and
// intentionally live in the parallel structure below instead of inflating the
// per-invocation descriptor fetched by the shader.
struct SurfaceSvmProgramDescriptor {
  std::uint32_t instruction_begin{};
  std::uint32_t instruction_count{};
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t flags{};
  SurfaceClosureEndpointMask endpoints{};
  std::uint32_t stack_lanes{};

  auto
  operator<=>(const SurfaceSvmProgramDescriptor &) const noexcept = default;
};

// Exact host-side partition of the four independently typed side streams.
// The runtime does not upload this table: every device instruction already
// contains its absolute scene offset.
struct SurfaceSvmProgramSideRange {
  std::uint32_t value_operand_begin{};
  std::uint32_t value_operand_count{};
  std::uint32_t metadata_begin{};
  std::uint32_t metadata_count{};
  std::uint32_t static_data_begin{};
  std::uint32_t static_data_count{};
  std::uint32_t closure_operand_begin{};
  std::uint32_t closure_operand_count{};

  auto operator<=>(const SurfaceSvmProgramSideRange &) const noexcept = default;
};

// Runtime tags index `programs` directly. Instructions and all side streams
// are dense concatenations in the same tag order. Jumps and side references
// are absolute scene offsets; typed local addresses remain invocation-local.
struct SurfaceSvmSceneImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceSvmProgramDescriptor> programs;
  std::vector<SurfaceSvmProgramSideRange> side_ranges;
  std::vector<SurfaceSvmBytecodeInstruction> instructions;
  std::vector<std::uint32_t> value_operands;
  std::vector<SurfaceValueBytecodeMetadata> value_metadata;
  std::vector<float> static_data;
  std::vector<std::uint32_t> closure_operands;
  std::uint32_t maximum_instruction_count{};
  std::uint32_t maximum_stack_lanes{};
  std::uint32_t maximum_scalar_slots{};
  std::uint32_t maximum_vector_slots{};
  std::uint32_t maximum_unsigned_integer_slots{};
  std::uint32_t used_closure_operations{};
  PrincipledClosureFeatureMask used_principled_features{};
};

// Host-only proof input for assigning exact evaluator bodies to a unified
// program. `instruction_sources` is parallel to `image->instructions`: every
// Value record names its original ValueExpressionId and every other record
// carries the invalid sentinel. When SetNormal is present, its committed
// source is named separately so the new lifetime epoch is also proven against
// the source graph rather than inferred from a physical slot.
struct SurfaceSvmEvaluatorProgramInput {
  const SurfaceProgram *program{};
  const SurfaceSvmProgramImage *image{};
  std::span<const std::uint32_t> instruction_sources{};
  ValueExpressionId surface_normal_output{};
};

// Complete runtime-ready unified scene. `instruction_variants` is parallel to
// `image.instructions`; it selects an exact value evaluator for Value records
// and contains the invalid sentinel for control, closure, SetNormal and End.
// Variant equality is exact tuple equality, never hash-only equality.
struct SurfaceSvmExecutableScene {
  bool valid{};
  std::string diagnostic;
  SurfaceSvmSceneImage image;
  std::vector<SurfaceValueStaticVariant> value_variants;
  std::vector<std::uint32_t> instruction_variants;
};

// Returns an empty string iff every serialized relation is valid: value ABI,
// forward CFG, exact typed read-before-write initialization, weight/value slot
// ownership, dense side streams, closure operand banks and aggregate masks.
[[nodiscard]] std::string
validate_surface_svm_program_image(const SurfaceSvmProgramImage &program);

// Transactionally lowers one proven structured schedule. Passthrough aliases
// have no bytecode record; jump targets are remapped across those erased
// semantic points. No material value is baked into the image.
[[nodiscard]] SurfaceSvmProgramImage
lower_surface_svm_program(const SurfaceProgram &program,
                          const SurfaceClosurePlan &closure_plan,
                          const SurfaceValueDependencyPlan &dependencies,
                          const SurfaceSvmSchedulePlan &schedule,
                          const SurfaceSvmStoragePlan &storage);

// Formal sequential composition of Cycles' automatic-normal transaction:
// evaluate `normal`, commit its selected vector to ShaderData::N, discard the
// prefix locals, then execute `root`. Independent slot colorings may overlap
// because the SetNormal boundary kills every prefix local after its one read.
[[nodiscard]] SurfaceSvmProgramImage compose_surface_svm_normal_transaction(
    const SurfaceValueProgramImage &normal, std::uint32_t normal_output,
    const SurfaceSvmProgramImage &root, bool uses_undisplaced_geometry);

// Concatenates programs in runtime-tag order and rebases every jump, overflow
// operand, metadata/static-table reference, and closure operand begin to an
// absolute scene offset. Local typed addresses and parameter ids are not
// changed.
[[nodiscard]] SurfaceSvmSceneImage
build_surface_svm_scene_image(std::span<const SurfaceSvmProgramImage> programs);

// Validates the source-provenance relation directly on the unified CFG,
// interns the exact evaluator domain, joins operand storage routes, and then
// aggregates programs in input/tag order. This is the sole runtime assembly
// path; no legacy split value/closure executable participates in the proof.
[[nodiscard]] SurfaceSvmExecutableScene build_surface_svm_executable_scene(
    std::span<const SurfaceSvmEvaluatorProgramInput> inputs);

// Returns an empty string iff descriptors form exact dense partitions and
// every de-relocated per-tag program satisfies the complete program verifier.
[[nodiscard]] std::string
validate_surface_svm_scene_image(const SurfaceSvmSceneImage &scene);

} // namespace psycles::compiler
