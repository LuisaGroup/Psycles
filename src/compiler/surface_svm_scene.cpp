#include <psycles/compiler/surface_svm_program.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace psycles::compiler {
namespace {

[[nodiscard]] SurfaceSvmProgramImage reject_program(std::string diagnostic) {
  SurfaceSvmProgramImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceSvmSceneImage reject_scene(std::string diagnostic) {
  SurfaceSvmSceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool add_extent(std::size_t &total, std::size_t count) noexcept {
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (count > limit - total) {
    return false;
  }
  total += count;
  return true;
}

[[nodiscard]] bool range_fits(std::uint32_t begin, std::uint32_t count,
                              std::size_t extent) noexcept {
  return begin <= extent && count <= extent - begin;
}

[[nodiscard]] bool
address_fits_normal_output(const SurfaceValueProgramImage &normal,
                           std::uint32_t encoded) noexcept {
  const auto address = SurfaceValueAddress{encoded};
  return address.valid() && address.bank() == SurfaceValueBank::vector &&
         (address.parameter() || address.index() < normal.vector_slots);
}

[[nodiscard]] constexpr std::uint32_t
bank_lane_width(SurfaceValueBank bank) noexcept {
  switch (bank) {
  case SurfaceValueBank::scalar:
    return 1u;
  case SurfaceValueBank::vector:
    return 3u;
  case SurfaceValueBank::unsigned_integer:
    return 2u;
  }
  return 0u;
}

struct SurfaceSvmLaneLayout {
  std::array<std::uint32_t, 3u> slots{};
  std::array<std::uint32_t, 3u> bases{};
  std::uint32_t lanes{};
};

[[nodiscard]] bool make_lane_layout(std::uint32_t scalar_slots,
                                    std::uint32_t vector_slots,
                                    std::uint32_t unsigned_integer_slots,
                                    SurfaceSvmLaneLayout &layout) noexcept {
  layout.slots = {scalar_slots, vector_slots, unsigned_integer_slots};
  auto lane_end = std::uint64_t{};
  for (auto bank = std::size_t{}; bank < layout.slots.size(); ++bank) {
    if (lane_end > surface_svm_stack_lane_capacity) {
      return false;
    }
    layout.bases[bank] = static_cast<std::uint32_t>(lane_end);
    lane_end += static_cast<std::uint64_t>(layout.slots[bank]) *
                bank_lane_width(static_cast<SurfaceValueBank>(bank));
  }
  if (lane_end > surface_svm_stack_lane_capacity) {
    return false;
  }
  layout.lanes = static_cast<std::uint32_t>(lane_end);
  return true;
}

[[nodiscard]] bool remap_local_address_to_lane(
    const SurfaceSvmLaneLayout &layout, std::uint32_t &encoded) noexcept {
  const auto address = SurfaceValueAddress{encoded};
  if (!address.valid() || address.parameter()) {
    return true;
  }
  const auto bank = static_cast<std::uint32_t>(address.bank());
  if (bank >= layout.slots.size() || address.index() >= layout.slots[bank]) {
    return false;
  }
  const auto width = bank_lane_width(address.bank());
  const auto lane =
      static_cast<std::uint64_t>(layout.bases[bank]) +
      static_cast<std::uint64_t>(address.index()) * width;
  if (width == 0u || lane + width > layout.lanes) {
    return false;
  }
  encoded = (bank << SurfaceValueAddress::bank_shift) |
            static_cast<std::uint32_t>(lane);
  return true;
}

[[nodiscard]] bool remap_operand_word_to_lanes(
    const SurfaceSvmLaneLayout &layout, std::uint32_t &word) noexcept {
  for (auto lane = std::uint32_t{};
       lane < surface_value_operands_per_word; ++lane) {
    const auto shift = surface_value_operand_lane_bits * lane;
    auto compact = SurfaceValueOperandAddress{static_cast<std::uint16_t>(
        (word >> shift) & std::numeric_limits<std::uint16_t>::max())};
    if (!compact.valid()) {
      if (compact.encoded() != SurfaceValueOperandAddress::invalid_value) {
        return false;
      }
      continue;
    }
    auto expanded = compact.expanded().encoded();
    if (!remap_local_address_to_lane(layout, expanded) ||
        !encode_surface_value_operand_address(
            SurfaceValueAddress{expanded}, compact)) {
      return false;
    }
    word = (word & ~(std::uint32_t{0xffffu} << shift)) |
           (static_cast<std::uint32_t>(compact.encoded()) << shift);
  }
  return true;
}

[[nodiscard]] bool remap_value_program_to_lanes(
    SurfaceValueProgramImage &program, const SurfaceSvmLaneLayout &layout,
    std::uint32_t &selected_output) {
  if (!remap_local_address_to_lane(layout, selected_output)) {
    return false;
  }
  for (auto &encoded : program.value_addresses) {
    if (!remap_local_address_to_lane(layout, encoded)) {
      return false;
    }
  }
  for (auto &instruction : program.instructions) {
    if (!remap_local_address_to_lane(layout, instruction.result)) {
      return false;
    }
    const auto operand_count = surface_value_operand_count(instruction);
    if (operand_count <= surface_value_inline_operand_capacity) {
      if (!remap_operand_word_to_lanes(layout,
                                       instruction.operand_payload)) {
        return false;
      }
      continue;
    }
    const auto word_count = surface_value_operand_word_count(operand_count);
    if (instruction.operand_payload > program.operands.size() ||
        word_count > program.operands.size() - instruction.operand_payload) {
      return false;
    }
    for (auto word = std::size_t{}; word < word_count; ++word) {
      if (!remap_operand_word_to_lanes(
              layout, program.operands[instruction.operand_payload + word])) {
        return false;
      }
    }
  }
  program.scalar_slots = layout.lanes;
  program.vector_slots = layout.lanes;
  program.unsigned_integer_slots = layout.lanes;
  return true;
}

void rebase_value_instruction(SurfaceSvmBytecodeInstruction &instruction,
                              std::uint32_t operand_begin,
                              std::uint32_t metadata_begin) noexcept {
  auto value = surface_svm_value_instruction(instruction);
  if (surface_value_operand_count(value) >
      surface_value_inline_operand_capacity) {
    value.operand_payload += operand_begin;
  }
  if (value.metadata_index != SurfaceValueAddress::invalid_value) {
    value.metadata_index += metadata_begin;
  }
  instruction = make_surface_svm_value_instruction(value);
}

[[nodiscard]] bool
de_rebase_program(const SurfaceSvmSceneImage &scene,
                  const SurfaceSvmProgramDescriptor &descriptor,
                  const SurfaceSvmProgramSideRange &side,
                  SurfaceSvmProgramImage &program, std::string &diagnostic) {
  program.endpoints = descriptor.endpoints;
  program.stack_lanes = descriptor.stack_lanes;
  program.scalar_slots = descriptor.scalar_slots;
  program.vector_slots = descriptor.vector_slots;
  program.unsigned_integer_slots = descriptor.unsigned_integer_slots;
  program.flags = descriptor.flags;
  program.value_operands.assign(
      scene.value_operands.begin() + side.value_operand_begin,
      scene.value_operands.begin() + side.value_operand_begin +
          side.value_operand_count);
  program.value_metadata.assign(
      scene.value_metadata.begin() + side.metadata_begin,
      scene.value_metadata.begin() + side.metadata_begin + side.metadata_count);
  program.static_data.assign(scene.static_data.begin() + side.static_data_begin,
                             scene.static_data.begin() +
                                 side.static_data_begin +
                                 side.static_data_count);
  program.closure_operands.assign(
      scene.closure_operands.begin() + side.closure_operand_begin,
      scene.closure_operands.begin() + side.closure_operand_begin +
          side.closure_operand_count);

  for (auto &metadata : program.value_metadata) {
    if (metadata.static_table_begin < side.static_data_begin) {
      diagnostic = "a scene metadata table begins before its program slice";
      return false;
    }
    metadata.static_table_begin -= side.static_data_begin;
  }

  program.instructions.reserve(descriptor.instruction_count);
  const auto instruction_end =
      descriptor.instruction_begin + descriptor.instruction_count;
  for (auto global_pc = descriptor.instruction_begin;
       global_pc < instruction_end; ++global_pc) {
    auto instruction = scene.instructions[global_pc];
    switch (surface_svm_bytecode_kind(instruction)) {
    case SurfaceSvmBytecodeKind::value: {
      auto value = surface_svm_value_instruction(instruction);
      if (surface_value_operand_count(value) >
          surface_value_inline_operand_capacity) {
        if (value.operand_payload < side.value_operand_begin) {
          diagnostic =
              "a scene value operand range begins before its program slice";
          return false;
        }
        value.operand_payload -= side.value_operand_begin;
      }
      if (value.metadata_index != SurfaceValueAddress::invalid_value) {
        if (value.metadata_index < side.metadata_begin) {
          diagnostic = "a scene metadata index begins before its program slice";
          return false;
        }
        value.metadata_index -= side.metadata_begin;
      }
      instruction = make_surface_svm_value_instruction(value);
      ++program.value_instruction_count;
      break;
    }
    case SurfaceSvmBytecodeKind::mix_closure:
      ++program.mix_instruction_count;
      break;
    case SurfaceSvmBytecodeKind::add_closure_weight:
      ++program.weight_add_instruction_count;
      break;
    case SurfaceSvmBytecodeKind::jump_if_one:
    case SurfaceSvmBytecodeKind::jump_if_zero:
      if (instruction.payload1 < descriptor.instruction_begin ||
          instruction.payload1 >= instruction_end) {
        diagnostic = "a scene closure guard leaves its program slice";
        return false;
      }
      instruction.payload1 -= descriptor.instruction_begin;
      ++program.conditional_branch_count;
      break;
    case SurfaceSvmBytecodeKind::closure_leaf: {
      if (instruction.payload0 < side.closure_operand_begin) {
        diagnostic =
            "a scene closure operand range begins before its program slice";
        return false;
      }
      instruction.payload0 -= side.closure_operand_begin;
      const auto closure_control = surface_svm_closure_control(instruction);
      if ((closure_control & ~surface_closure_control_mask) != 0u) {
        diagnostic = "a scene closure leaf has invalid control bits";
        return false;
      }
      const auto operation = surface_closure_operation(
          SurfaceClosureBytecodeInstruction{.control = closure_control});
      if (!surface_closure_is_leaf_operation(operation) ||
          static_cast<std::uint32_t>(operation) >= 32u) {
        diagnostic = "a scene closure leaf has no physical operation";
        return false;
      }
      program.used_closure_operations |=
          1u << static_cast<std::uint32_t>(operation);
      program.used_principled_features |= instruction.payload2;
      ++program.closure_leaf_count;
      break;
    }
    case SurfaceSvmBytecodeKind::set_normal:
      ++program.surface_normal_transition_count;
      break;
    case SurfaceSvmBytecodeKind::end:
      break;
    case SurfaceSvmBytecodeKind::invalid:
      diagnostic = "a scene program contains an unknown opcode";
      return false;
    }
    program.instructions.emplace_back(instruction);
  }
  program.valid = true;
  if (const auto local_diagnostic = validate_surface_svm_program_image(program);
      !local_diagnostic.empty()) {
    diagnostic = std::move(local_diagnostic);
    return false;
  }
  return true;
}

} // namespace

SurfaceSvmProgramImage compose_surface_svm_normal_transaction(
    const SurfaceValueProgramImage &normal, std::uint32_t normal_output,
    const SurfaceSvmProgramImage &root, bool uses_undisplaced_geometry) {
  if (const auto diagnostic = validate_surface_value_program_image(normal);
      !diagnostic.empty()) {
    return reject_program("automatic-normal prefix: " + diagnostic);
  }
  if (const auto diagnostic = validate_surface_svm_program_image(root);
      !diagnostic.empty()) {
    return reject_program("structured surface root: " + diagnostic);
  }
  if (normal.flags != 0u ||
      std::any_of(normal.instructions.begin(), normal.instructions.end(),
                  is_surface_value_surface_normal_transition) ||
      root.flags != 0u || root.surface_normal_transition_count != 0u) {
    return reject_program("a unified SetNormal transaction cannot nest");
  }
  if (!address_fits_normal_output(normal, normal_output)) {
    return reject_program("the automatic-normal output is not a vector");
  }

  SurfaceSvmLaneLayout normal_layout;
  if (!make_lane_layout(normal.scalar_slots, normal.vector_slots,
                        normal.unsigned_integer_slots, normal_layout)) {
    return reject_program(
        "the automatic-normal prefix exceeds the Cycles SVM stack");
  }
  auto remapped_normal = normal;
  auto remapped_normal_output = normal_output;
  if (!remap_value_program_to_lanes(remapped_normal, normal_layout,
                                    remapped_normal_output)) {
    return reject_program(
        "the automatic-normal prefix could not be mapped to SVM lanes");
  }

  // Appending only the established commit record turns the legacy value
  // verifier into an exact proof that the chosen output is defined by the
  // prefix. No source ValueExpressionId or heuristic last-definition scan is
  // needed here.
  auto normal_probe = remapped_normal;
  normal_probe.instructions.emplace_back(SurfaceValueBytecodeInstruction{
      .control = surface_value_surface_normal_transition_control,
      .result = remapped_normal_output,
      .operand_payload = surface_value_invalid_operand_word,
      .metadata_index = SurfaceValueAddress::invalid_value});
  normal_probe.flags =
      uses_undisplaced_geometry
          ? surface_value_program_automatic_normal_uses_undisplaced_geometry
          : 0u;
  if (const auto diagnostic =
          validate_surface_value_program_image(normal_probe);
      !diagnostic.empty()) {
    return reject_program("automatic-normal commit: " + diagnostic);
  }

  auto instruction_count = remapped_normal.instructions.size();
  auto value_operand_count = remapped_normal.operands.size();
  auto metadata_count = remapped_normal.metadata.size();
  auto static_data_count = remapped_normal.static_data.size();
  if (!add_extent(instruction_count, 1u) ||
      !add_extent(instruction_count, root.instructions.size()) ||
      !add_extent(value_operand_count, root.value_operands.size()) ||
      !add_extent(metadata_count, root.value_metadata.size()) ||
      !add_extent(static_data_count, root.static_data.size())) {
    return reject_program(
        "the unified SetNormal transaction exceeds 32-bit offsets");
  }

  SurfaceSvmProgramImage result;
  result.endpoints = root.endpoints;
  result.instructions.reserve(instruction_count);
  result.value_operands = remapped_normal.operands;
  result.value_operands.reserve(value_operand_count);
  result.value_metadata = remapped_normal.metadata;
  result.value_metadata.reserve(metadata_count);
  result.static_data = remapped_normal.static_data;
  result.static_data.reserve(static_data_count);
  result.closure_operands = root.closure_operands;
  result.value_addresses = root.value_addresses;
  result.stack_lanes = std::max(normal_layout.lanes, root.stack_lanes);
  result.scalar_slots = std::max(normal.scalar_slots, root.scalar_slots);
  result.vector_slots = std::max(normal.vector_slots, root.vector_slots);
  result.unsigned_integer_slots =
      std::max(normal.unsigned_integer_slots, root.unsigned_integer_slots);
  result.flags = normal_probe.flags;
  result.value_instruction_count =
      static_cast<std::uint32_t>(remapped_normal.instructions.size()) +
      root.value_instruction_count;
  result.mix_instruction_count = root.mix_instruction_count;
  result.weight_add_instruction_count = root.weight_add_instruction_count;
  result.conditional_branch_count = root.conditional_branch_count;
  result.closure_leaf_count = root.closure_leaf_count;
  result.surface_normal_transition_count = 1u;
  result.used_closure_operations = root.used_closure_operations;
  result.used_principled_features = root.used_principled_features;

  for (const auto &instruction : remapped_normal.instructions) {
    result.instructions.emplace_back(
        make_surface_svm_value_instruction(instruction));
  }
  result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
      .control = surface_svm_set_normal_opcode,
      .payload0 = remapped_normal_output,
      .payload1 = surface_value_invalid_operand_word,
      .payload2 = SurfaceValueAddress::invalid_value});

  const auto root_instruction_begin =
      static_cast<std::uint32_t>(result.instructions.size());
  const auto root_value_operand_begin =
      static_cast<std::uint32_t>(result.value_operands.size());
  const auto root_metadata_begin =
      static_cast<std::uint32_t>(result.value_metadata.size());
  const auto root_static_data_begin =
      static_cast<std::uint32_t>(result.static_data.size());
  for (auto instruction : root.instructions) {
    switch (surface_svm_bytecode_kind(instruction)) {
    case SurfaceSvmBytecodeKind::value:
      rebase_value_instruction(instruction, root_value_operand_begin,
                               root_metadata_begin);
      break;
    case SurfaceSvmBytecodeKind::jump_if_one:
    case SurfaceSvmBytecodeKind::jump_if_zero:
      instruction.payload1 += root_instruction_begin;
      break;
    case SurfaceSvmBytecodeKind::mix_closure:
    case SurfaceSvmBytecodeKind::add_closure_weight:
    case SurfaceSvmBytecodeKind::closure_leaf:
    case SurfaceSvmBytecodeKind::end:
    case SurfaceSvmBytecodeKind::invalid:
      break;
    case SurfaceSvmBytecodeKind::set_normal:
      return reject_program("a unified SetNormal transaction is nested");
    }
    result.instructions.emplace_back(instruction);
  }
  result.value_operands.insert(result.value_operands.end(),
                               root.value_operands.begin(),
                               root.value_operands.end());
  for (auto metadata : root.value_metadata) {
    metadata.static_table_begin += root_static_data_begin;
    result.value_metadata.emplace_back(metadata);
  }
  result.static_data.insert(result.static_data.end(), root.static_data.begin(),
                            root.static_data.end());
  result.valid = true;
  if (const auto diagnostic = validate_surface_svm_program_image(result);
      !diagnostic.empty()) {
    return reject_program("composed unified surface SVM: " + diagnostic);
  }
  return result;
}

std::string
validate_surface_svm_scene_image(const SurfaceSvmSceneImage &scene) {
  static_assert(std::is_trivially_copyable_v<SurfaceSvmProgramDescriptor>);
  static_assert(sizeof(SurfaceSvmProgramDescriptor) == 32u);
  static_assert(std::is_trivially_copyable_v<SurfaceSvmProgramSideRange>);
  static_assert(sizeof(SurfaceSvmProgramSideRange) == 32u);
  static_assert(static_cast<std::uint32_t>(ClosureOperation::refraction) < 32u);
  if (!scene.valid) {
    return scene.diagnostic.empty() ? "the unified surface scene is invalid"
                                    : scene.diagnostic;
  }
  if (scene.programs.size() != scene.side_ranges.size()) {
    return "surface scene descriptors and side ranges are not parallel";
  }

  auto instruction_cursor = std::size_t{};
  auto value_operand_cursor = std::size_t{};
  auto metadata_cursor = std::size_t{};
  auto static_data_cursor = std::size_t{};
  auto closure_operand_cursor = std::size_t{};
  auto maximum_instruction_count = std::uint32_t{};
  auto maximum_stack_lanes = std::uint32_t{};
  auto maximum_scalar_slots = std::uint32_t{};
  auto maximum_vector_slots = std::uint32_t{};
  auto maximum_unsigned_integer_slots = std::uint32_t{};
  auto used_closure_operations = std::uint32_t{};
  auto used_principled_features = PrincipledClosureFeatureMask{};

  for (auto index = std::size_t{}; index < scene.programs.size(); ++index) {
    const auto &descriptor = scene.programs[index];
    const auto &side = scene.side_ranges[index];
    if (descriptor.instruction_begin != instruction_cursor ||
        side.value_operand_begin != value_operand_cursor ||
        side.metadata_begin != metadata_cursor ||
        side.static_data_begin != static_data_cursor ||
        side.closure_operand_begin != closure_operand_cursor) {
      return "surface scene program partitions are not dense and canonical";
    }
    if (!range_fits(descriptor.instruction_begin, descriptor.instruction_count,
                    scene.instructions.size()) ||
        !range_fits(side.value_operand_begin, side.value_operand_count,
                    scene.value_operands.size()) ||
        !range_fits(side.metadata_begin, side.metadata_count,
                    scene.value_metadata.size()) ||
        !range_fits(side.static_data_begin, side.static_data_count,
                    scene.static_data.size()) ||
        !range_fits(side.closure_operand_begin, side.closure_operand_count,
                    scene.closure_operands.size())) {
      return "a surface scene program partition exceeds its stream";
    }

    SurfaceSvmProgramImage local;
    std::string diagnostic;
    if (!de_rebase_program(scene, descriptor, side, local, diagnostic)) {
      return "surface scene program " + std::to_string(index) + ": " +
             diagnostic;
    }
    instruction_cursor += descriptor.instruction_count;
    value_operand_cursor += side.value_operand_count;
    metadata_cursor += side.metadata_count;
    static_data_cursor += side.static_data_count;
    closure_operand_cursor += side.closure_operand_count;
    maximum_instruction_count =
        std::max(maximum_instruction_count, descriptor.instruction_count);
    maximum_stack_lanes =
        std::max(maximum_stack_lanes, descriptor.stack_lanes);
    maximum_scalar_slots =
        std::max(maximum_scalar_slots, descriptor.scalar_slots);
    maximum_vector_slots =
        std::max(maximum_vector_slots, descriptor.vector_slots);
    maximum_unsigned_integer_slots = std::max(
        maximum_unsigned_integer_slots, descriptor.unsigned_integer_slots);
    used_closure_operations |= local.used_closure_operations;
    used_principled_features |= local.used_principled_features;
  }
  if (instruction_cursor != scene.instructions.size() ||
      value_operand_cursor != scene.value_operands.size() ||
      metadata_cursor != scene.value_metadata.size() ||
      static_data_cursor != scene.static_data.size() ||
      closure_operand_cursor != scene.closure_operands.size()) {
    return "the surface scene has an unowned stream suffix";
  }
  if (maximum_instruction_count != scene.maximum_instruction_count ||
      maximum_stack_lanes != scene.maximum_stack_lanes ||
      maximum_scalar_slots != scene.maximum_scalar_slots ||
      maximum_vector_slots != scene.maximum_vector_slots ||
      maximum_unsigned_integer_slots != scene.maximum_unsigned_integer_slots ||
      used_closure_operations != scene.used_closure_operations ||
      used_principled_features != scene.used_principled_features) {
    return "the surface scene aggregate capabilities are inconsistent";
  }
  return {};
}

SurfaceSvmSceneImage build_surface_svm_scene_image(
    std::span<const SurfaceSvmProgramImage> programs) {
  auto instruction_count = std::size_t{};
  auto value_operand_count = std::size_t{};
  auto metadata_count = std::size_t{};
  auto static_data_count = std::size_t{};
  auto closure_operand_count = std::size_t{};
  for (auto index = std::size_t{}; index < programs.size(); ++index) {
    if (const auto diagnostic =
            validate_surface_svm_program_image(programs[index]);
        !diagnostic.empty()) {
      return reject_scene("surface program " + std::to_string(index) + ": " +
                          diagnostic);
    }
    if (!add_extent(instruction_count, programs[index].instructions.size()) ||
        !add_extent(value_operand_count,
                    programs[index].value_operands.size()) ||
        !add_extent(metadata_count, programs[index].value_metadata.size()) ||
        !add_extent(static_data_count, programs[index].static_data.size()) ||
        !add_extent(closure_operand_count,
                    programs[index].closure_operands.size())) {
      return reject_scene(
          "the unified surface scene exceeds 32-bit device offsets");
    }
  }

  SurfaceSvmSceneImage result;
  result.programs.reserve(programs.size());
  result.side_ranges.reserve(programs.size());
  result.instructions.reserve(instruction_count);
  result.value_operands.reserve(value_operand_count);
  result.value_metadata.reserve(metadata_count);
  result.static_data.reserve(static_data_count);
  result.closure_operands.reserve(closure_operand_count);
  for (const auto &program : programs) {
    const auto instruction_begin =
        static_cast<std::uint32_t>(result.instructions.size());
    const auto value_operand_begin =
        static_cast<std::uint32_t>(result.value_operands.size());
    const auto metadata_begin =
        static_cast<std::uint32_t>(result.value_metadata.size());
    const auto static_data_begin =
        static_cast<std::uint32_t>(result.static_data.size());
    const auto closure_operand_begin =
        static_cast<std::uint32_t>(result.closure_operands.size());
    result.programs.emplace_back(SurfaceSvmProgramDescriptor{
        .instruction_begin = instruction_begin,
        .instruction_count =
            static_cast<std::uint32_t>(program.instructions.size()),
        .scalar_slots = program.scalar_slots,
        .vector_slots = program.vector_slots,
        .unsigned_integer_slots = program.unsigned_integer_slots,
        .flags = program.flags,
        .endpoints = program.endpoints,
        .stack_lanes = program.stack_lanes});
    result.side_ranges.emplace_back(SurfaceSvmProgramSideRange{
        .value_operand_begin = value_operand_begin,
        .value_operand_count =
            static_cast<std::uint32_t>(program.value_operands.size()),
        .metadata_begin = metadata_begin,
        .metadata_count =
            static_cast<std::uint32_t>(program.value_metadata.size()),
        .static_data_begin = static_data_begin,
        .static_data_count =
            static_cast<std::uint32_t>(program.static_data.size()),
        .closure_operand_begin = closure_operand_begin,
        .closure_operand_count =
            static_cast<std::uint32_t>(program.closure_operands.size())});

    for (auto instruction : program.instructions) {
      switch (surface_svm_bytecode_kind(instruction)) {
      case SurfaceSvmBytecodeKind::value:
        rebase_value_instruction(instruction, value_operand_begin,
                                 metadata_begin);
        break;
      case SurfaceSvmBytecodeKind::jump_if_one:
      case SurfaceSvmBytecodeKind::jump_if_zero:
        instruction.payload1 += instruction_begin;
        break;
      case SurfaceSvmBytecodeKind::closure_leaf:
        instruction.payload0 += closure_operand_begin;
        break;
      case SurfaceSvmBytecodeKind::mix_closure:
      case SurfaceSvmBytecodeKind::add_closure_weight:
      case SurfaceSvmBytecodeKind::set_normal:
      case SurfaceSvmBytecodeKind::end:
      case SurfaceSvmBytecodeKind::invalid:
        break;
      }
      result.instructions.emplace_back(instruction);
    }
    result.value_operands.insert(result.value_operands.end(),
                                 program.value_operands.begin(),
                                 program.value_operands.end());
    for (auto metadata : program.value_metadata) {
      metadata.static_table_begin += static_data_begin;
      result.value_metadata.emplace_back(metadata);
    }
    result.static_data.insert(result.static_data.end(),
                              program.static_data.begin(),
                              program.static_data.end());
    result.closure_operands.insert(result.closure_operands.end(),
                                   program.closure_operands.begin(),
                                   program.closure_operands.end());
    result.maximum_instruction_count =
        std::max(result.maximum_instruction_count,
                 static_cast<std::uint32_t>(program.instructions.size()));
    result.maximum_stack_lanes =
        std::max(result.maximum_stack_lanes, program.stack_lanes);
    result.maximum_scalar_slots =
        std::max(result.maximum_scalar_slots, program.scalar_slots);
    result.maximum_vector_slots =
        std::max(result.maximum_vector_slots, program.vector_slots);
    result.maximum_unsigned_integer_slots = std::max(
        result.maximum_unsigned_integer_slots, program.unsigned_integer_slots);
    result.used_closure_operations |= program.used_closure_operations;
    result.used_principled_features |= program.used_principled_features;
  }
  result.valid = true;
  if (const auto diagnostic = validate_surface_svm_scene_image(result);
      !diagnostic.empty()) {
    return reject_scene("built unified surface scene: " + diagnostic);
  }
  return result;
}

} // namespace psycles::compiler
