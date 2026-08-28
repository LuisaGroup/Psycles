#include <psycles/compiler/surface_svm_program.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

inline constexpr auto invalid_index = ~std::uint32_t{0u};
using IndexSet = std::vector<std::uint32_t>;

[[nodiscard]] SurfaceSvmProgramImage reject(std::string diagnostic) {
  SurfaceSvmProgramImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] constexpr std::size_t bank_index(
    SurfaceValueBank bank) noexcept {
  return static_cast<std::size_t>(bank);
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

[[nodiscard]] bool contains(const IndexSet &set,
                            std::uint32_t value) noexcept {
  return std::binary_search(set.begin(), set.end(), value);
}

void insert(IndexSet &set, std::uint32_t value) {
  const auto position = std::lower_bound(set.begin(), set.end(), value);
  if (position == set.end() || *position != value) {
    set.insert(position, value);
  }
}

[[nodiscard]] IndexSet set_intersection(const IndexSet &a,
                                        const IndexSet &b) {
  IndexSet result;
  result.reserve(std::min(a.size(), b.size()));
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
  return result;
}

[[nodiscard]] bool address_fits(const SurfaceSvmProgramImage &program,
                                std::uint32_t encoded,
                                bool allow_invalid,
                                SurfaceValueBank expected,
                                bool require_expected) noexcept {
  const auto address = SurfaceValueAddress{encoded};
  if (!address.valid()) {
    return allow_invalid;
  }
  if (static_cast<std::uint32_t>(address.bank()) >
          static_cast<std::uint32_t>(
              SurfaceValueBank::unsigned_integer) ||
      (require_expected && address.bank() != expected)) {
    return false;
  }
  if (address.parameter()) {
    return true;
  }
  const auto width = bank_lane_width(address.bank());
  return width != 0u && address.index() < program.stack_lanes &&
         width <= program.stack_lanes - address.index();
}

[[nodiscard]] bool valid_weight_slot(const SurfaceSvmProgramImage &program,
                                     std::uint32_t slot,
                                     bool allow_root) noexcept {
  return (allow_root && slot == surface_svm_root_weight_slot) ||
         (slot < program.stack_lanes &&
          slot != surface_svm_invalid_packed_weight_slot);
}

[[nodiscard]] SurfaceClosureEndpointMask closure_endpoints(
    const SurfaceValueDependencyPlan &dependencies,
    ClosureExpressionId closure,
    SurfaceClosureEndpointMask selected) noexcept {
  auto endpoints = SurfaceClosureEndpointMask{};
  if (dependencies.physical_closures[closure.value]) {
    endpoints |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::physical);
  }
  if (dependencies.emission_closures[closure.value]) {
    endpoints |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::emission);
  }
  return endpoints & selected;
}

struct Definition {
  bool weight{};
  std::uint32_t value{};
};

struct LaneSpan {
  std::uint32_t begin{};
  std::uint32_t end{};
};

[[nodiscard]] constexpr LaneSpan value_lane_span(
    std::uint32_t encoded) noexcept {
  const auto address = SurfaceValueAddress{encoded};
  return {.begin = address.index(),
          .end = address.index() + bank_lane_width(address.bank())};
}

[[nodiscard]] constexpr bool lane_spans_overlap(LaneSpan a,
                                                LaneSpan b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

struct ProgramPoint {
  IndexSet value_uses;
  IndexSet weight_uses;
  std::vector<Definition> definitions;
  std::vector<std::uint32_t> successors;
  // SetNormal consumes its selected vector and then starts a disjoint local
  // lifetime epoch. Prefix and root colorings may use the same physical slots,
  // so no prefix definition is available after this point.
  bool clear_definitions_after_uses{};
};

[[nodiscard]] bool add_value_use(ProgramPoint &point,
                                 const SurfaceSvmProgramImage &program,
                                 std::uint32_t encoded,
                                 bool allow_invalid,
                                 SurfaceValueBank expected =
                                     SurfaceValueBank::scalar,
                                 bool require_expected = false) {
  if (!address_fits(program, encoded, allow_invalid, expected,
                    require_expected)) {
    return false;
  }
  const auto address = SurfaceValueAddress{encoded};
  if (address.valid() && !address.parameter()) {
    insert(point.value_uses, address.encoded());
  }
  return true;
}

[[nodiscard]] bool add_weight_use(ProgramPoint &point,
                                  const SurfaceSvmProgramImage &program,
                                  std::uint32_t slot) {
  if (!valid_weight_slot(program, slot, true)) {
    return false;
  }
  if (slot != surface_svm_root_weight_slot) {
    insert(point.weight_uses, slot);
  }
  return true;
}

[[nodiscard]] bool verify_definite_initialization(
    const std::vector<ProgramPoint> &points,
    std::string &diagnostic) {
  if (points.empty()) {
    diagnostic = "the unified surface SVM has no End record";
    return false;
  }
  std::vector<std::vector<std::uint32_t>> predecessors(points.size());
  for (auto index = std::size_t{}; index < points.size(); ++index) {
    for (const auto successor : points[index].successors) {
      if (successor >= points.size()) {
        diagnostic = "the unified surface SVM has an invalid successor";
        return false;
      }
      predecessors[successor].emplace_back(
          static_cast<std::uint32_t>(index));
    }
  }

  struct State {
    IndexSet values;
    IndexSet weights;

    // The SVM stack is physically untyped. A write invalidates every logical
    // value whose occupied 32-bit lane interval overlaps it, independently of
    // the address's semantic bank tag. This is the bytecode-level proof that
    // typed coloring and lifetime-epoch reuse cannot silently alias at run
    // time.
    void invalidate(LaneSpan written) {
      std::erase_if(values, [written](std::uint32_t encoded) noexcept {
        return lane_spans_overlap(value_lane_span(encoded), written);
      });
      std::erase_if(weights, [written](std::uint32_t slot) noexcept {
        return lane_spans_overlap(
            LaneSpan{.begin = slot, .end = slot + 1u}, written);
      });
    }
  };
  std::vector<State> defined_out(points.size());
  for (auto index = std::size_t{}; index < points.size(); ++index) {
    State state;
    if (index != 0u) {
      if (predecessors[index].empty()) {
        diagnostic = "the unified surface SVM contains unreachable bytecode";
        return false;
      }
      state = defined_out[predecessors[index].front()];
      for (auto predecessor = std::size_t{1u};
           predecessor < predecessors[index].size(); ++predecessor) {
        const auto &incoming =
            defined_out[predecessors[index][predecessor]];
        state.values = set_intersection(state.values, incoming.values);
        state.weights = set_intersection(state.weights, incoming.weights);
      }
    }
    for (const auto value : points[index].value_uses) {
      if (!contains(state.values, value)) {
        diagnostic =
            "the unified surface SVM reads an undefined value local";
        return false;
      }
    }
    for (const auto weight : points[index].weight_uses) {
      if (!contains(state.weights, weight)) {
        diagnostic =
            "the unified surface SVM reads an undefined weight local";
        return false;
      }
    }
    if (points[index].clear_definitions_after_uses) {
      state.values.clear();
      state.weights.clear();
    }
    for (const auto &definition : points[index].definitions) {
      if (definition.weight) {
        state.invalidate(
            LaneSpan{.begin = definition.value,
                     .end = definition.value + 1u});
        insert(state.weights, definition.value);
      } else {
        state.invalidate(value_lane_span(definition.value));
        insert(state.values, definition.value);
      }
    }
    defined_out[index] = std::move(state);
  }
  return true;
}

[[nodiscard]] bool scheduled_value_emits(
    const SurfaceSvmStoragePlan &storage,
    const SurfaceSvmScheduleInstruction &instruction) noexcept {
  return instruction.kind == SurfaceSvmScheduleInstructionKind::value &&
         instruction.source < storage.representatives.size() &&
         storage.representatives[instruction.source] == instruction.source;
}

} // namespace

std::string validate_surface_svm_program_image(
    const SurfaceSvmProgramImage &program) {
  static_assert(
      std::is_trivially_copyable_v<SurfaceSvmBytecodeInstruction>);
  static_assert(sizeof(SurfaceSvmBytecodeInstruction) == 16u);
  if (!program.valid) {
    return program.diagnostic.empty()
               ? "the unified surface SVM image is invalid"
               : program.diagnostic;
  }
  if (program.endpoints == 0u ||
      (program.endpoints & ~all_surface_closure_endpoints) != 0u) {
    return "the unified surface SVM has invalid endpoint projection";
  }
  if (program.stack_lanes > surface_svm_stack_lane_capacity) {
    return "the unified surface SVM exceeds Cycles' 255-lane stack";
  }
  if (program.stack_lanes > surface_svm_invalid_packed_weight_slot) {
    return "the unified lane stack exceeds packed weight addresses";
  }
  if (program.instructions.empty()) {
    return "the unified surface SVM has no instruction stream";
  }

  SurfaceValueProgramImage value_projection;
  value_projection.valid = true;
  value_projection.operands = program.value_operands;
  value_projection.metadata = program.value_metadata;
  value_projection.static_data = program.static_data;
  value_projection.value_addresses = program.value_addresses;
  value_projection.scalar_slots = program.stack_lanes;
  value_projection.vector_slots = program.stack_lanes;
  value_projection.unsigned_integer_slots = program.stack_lanes;
  value_projection.flags = program.flags;

  SurfaceClosureProgramImage closure_projection;
  closure_projection.valid = true;
  closure_projection.operands = program.closure_operands;
  closure_projection.used_operations = program.used_closure_operations;
  closure_projection.used_principled_features =
      program.used_principled_features;

  auto value_count = std::uint32_t{};
  auto mix_count = std::uint32_t{};
  auto add_count = std::uint32_t{};
  auto branch_count = std::uint32_t{};
  auto leaf_count = std::uint32_t{};
  auto normal_transition_count = std::uint32_t{};
  auto end_count = std::uint32_t{};
  auto closure_operand_cursor = std::size_t{};
  auto saw_structured_surface_control = false;

  for (auto index = std::size_t{}; index < program.instructions.size();
       ++index) {
    const auto &instruction = program.instructions[index];
    switch (surface_svm_bytecode_kind(instruction)) {
    case SurfaceSvmBytecodeKind::value:
      value_projection.instructions.emplace_back(
          surface_svm_value_instruction(instruction));
      ++value_count;
      break;
    case SurfaceSvmBytecodeKind::mix_closure: {
      saw_structured_surface_control = true;
      if ((instruction.control &
           ~(surface_svm_opcode_mask | surface_svm_mix_result_mask)) != 0u ||
          (instruction.control & surface_svm_opcode_mask) !=
              surface_svm_mix_closure_opcode ||
          (instruction.control & surface_svm_mix_result_mask) == 0u) {
        return "a unified Mix record has an invalid control word";
      }
      ++mix_count;
      break;
    }
    case SurfaceSvmBytecodeKind::add_closure_weight:
      saw_structured_surface_control = true;
      if (instruction.control != surface_svm_add_closure_weight_opcode) {
        return "a unified weight Add has undefined control bits";
      }
      ++add_count;
      break;
    case SurfaceSvmBytecodeKind::jump_if_one:
    case SurfaceSvmBytecodeKind::jump_if_zero:
      saw_structured_surface_control = true;
      if (instruction.control !=
              (surface_svm_bytecode_kind(instruction) ==
                       SurfaceSvmBytecodeKind::jump_if_one
                   ? surface_svm_jump_if_one_opcode
                   : surface_svm_jump_if_zero_opcode) ||
          instruction.payload1 <= index ||
          instruction.payload1 >= program.instructions.size() ||
          instruction.payload2 != surface_svm_invalid_payload) {
        return "a unified closure guard is malformed or not forward";
      }
      ++branch_count;
      break;
    case SurfaceSvmBytecodeKind::closure_leaf: {
      saw_structured_surface_control = true;
      const auto closure_control = surface_svm_closure_control(instruction);
      if ((closure_control & ~surface_closure_control_mask) != 0u ||
          instruction.control !=
              (surface_svm_closure_leaf_opcode |
               (closure_control << surface_svm_closure_control_shift))) {
        return "a unified closure leaf has undefined control bits";
      }
      const auto legacy = SurfaceClosureBytecodeInstruction{
          .control = closure_control,
          .payload0 = instruction.payload0,
          .payload1 = surface_closure_root_weight_slot,
          .payload2 = 0u};
      const auto operand_count = surface_closure_operand_count(
          surface_closure_operation(legacy));
      if (instruction.payload0 != closure_operand_cursor ||
          operand_count >
              program.closure_operands.size() - closure_operand_cursor) {
        return "the unified closure operand stream is not dense";
      }
      closure_operand_cursor += operand_count;
      closure_projection.instructions.emplace_back(legacy);
      closure_projection.principled_features.emplace_back(
          instruction.payload2);
      ++leaf_count;
      break;
    }
    case SurfaceSvmBytecodeKind::set_normal:
      if (saw_structured_surface_control ||
          instruction.control != surface_svm_set_normal_opcode ||
          ++normal_transition_count != 1u) {
        return "the unified SetNormal boundary is nested or follows surface "
               "control";
      }
      value_projection.instructions.emplace_back(
          surface_svm_value_instruction(instruction));
      break;
    case SurfaceSvmBytecodeKind::end:
      saw_structured_surface_control = true;
      if (instruction.control != surface_svm_end_opcode ||
          instruction.payload0 != surface_svm_invalid_payload ||
          instruction.payload1 != surface_svm_invalid_payload ||
          instruction.payload2 != surface_svm_invalid_payload ||
          index + 1u != program.instructions.size() || ++end_count != 1u) {
        return "the unified surface SVM End record is not canonical and final";
      }
      break;
    case SurfaceSvmBytecodeKind::invalid:
      return "the unified surface SVM contains an unknown opcode";
    }
  }
  if (end_count != 1u) {
    return "the unified surface SVM has no unique final End record";
  }
  if (closure_operand_cursor != program.closure_operands.size()) {
    return "the unified closure operand stream has an unreferenced suffix";
  }
  if (value_count != program.value_instruction_count ||
      mix_count != program.mix_instruction_count ||
      add_count != program.weight_add_instruction_count ||
      branch_count != program.conditional_branch_count ||
      leaf_count != program.closure_leaf_count ||
      normal_transition_count != program.surface_normal_transition_count) {
    return "the unified surface SVM instruction counts are inconsistent";
  }
  if (const auto diagnostic =
          validate_surface_value_program_image(value_projection);
      !diagnostic.empty()) {
    return "unified value projection: " + diagnostic;
  }
  if (const auto diagnostic = validate_surface_closure_program_image(
          closure_projection, value_projection);
      !diagnostic.empty()) {
    return "unified closure projection: " + diagnostic;
  }

  std::vector<ProgramPoint> points(program.instructions.size());
  for (auto index = std::size_t{}; index < program.instructions.size();
       ++index) {
    const auto &instruction = program.instructions[index];
    auto &point = points[index];
    switch (surface_svm_bytecode_kind(instruction)) {
    case SurfaceSvmBytecodeKind::value: {
      const auto value = surface_svm_value_instruction(instruction);
      const auto operand_count = surface_value_operand_count(value);
      const auto inline_operands =
          operand_count <= surface_value_inline_operand_capacity;
      for (auto operand = std::size_t{}; operand < operand_count; ++operand) {
        const auto word =
            inline_operands
                ? value.operand_payload
                : program.value_operands[
                      value.operand_payload +
                      operand / surface_value_operands_per_word];
        const auto compact = surface_value_operand_from_word(
            word, operand % surface_value_operands_per_word);
        if (!add_value_use(point, program, compact.expanded().encoded(),
                           false)) {
          return "a unified value operand exceeds its typed bank";
        }
      }
      point.definitions.emplace_back(
          Definition{.weight = false, .value = value.result});
      break;
    }
    case SurfaceSvmBytecodeKind::mix_closure: {
      if (!add_value_use(point, program, instruction.payload0, false,
                         SurfaceValueBank::scalar, true) ||
          !add_weight_use(point, program, instruction.payload1)) {
        return "a unified Mix reads an invalid factor or parent weight";
      }
      const auto left = surface_svm_mix_left_weight_slot(instruction);
      const auto right = surface_svm_mix_right_weight_slot(instruction);
      const auto has_left =
          (instruction.control & surface_svm_mix_left_result_bit) != 0u;
      const auto has_right =
          (instruction.control & surface_svm_mix_right_result_bit) != 0u;
      if ((has_left && !valid_weight_slot(program, left, false)) ||
          (!has_left && left != surface_svm_invalid_packed_weight_slot) ||
          (has_right && !valid_weight_slot(program, right, false)) ||
          (!has_right && right != surface_svm_invalid_packed_weight_slot) ||
          (has_left && has_right && left == right)) {
        return "a unified Mix has invalid or aliased result weights";
      }
      if (has_left) {
        point.definitions.emplace_back(
            Definition{.weight = true, .value = left});
      }
      if (has_right) {
        point.definitions.emplace_back(
            Definition{.weight = true, .value = right});
      }
      break;
    }
    case SurfaceSvmBytecodeKind::add_closure_weight:
      if (!add_weight_use(point, program, instruction.payload0) ||
          !add_weight_use(point, program, instruction.payload1) ||
          !valid_weight_slot(program, instruction.payload2, false)) {
        return "a unified weight Add has invalid operands or result";
      }
      point.definitions.emplace_back(
          Definition{.weight = true, .value = instruction.payload2});
      break;
    case SurfaceSvmBytecodeKind::jump_if_one:
    case SurfaceSvmBytecodeKind::jump_if_zero:
      if (!add_value_use(point, program, instruction.payload0, false,
                         SurfaceValueBank::scalar, true)) {
        return "a unified closure guard has no scalar factor";
      }
      point.successors.emplace_back(
          static_cast<std::uint32_t>(index + 1u));
      if (instruction.payload1 != index + 1u) {
        point.successors.emplace_back(instruction.payload1);
      }
      continue;
    case SurfaceSvmBytecodeKind::closure_leaf: {
      const auto legacy = SurfaceClosureBytecodeInstruction{
          .control = surface_svm_closure_control(instruction)};
      const auto operand_count = surface_closure_operand_count(
          surface_closure_operation(legacy));
      for (auto operand = std::size_t{}; operand < operand_count; ++operand) {
        if (!add_value_use(
                point, program,
                program.closure_operands[instruction.payload0 + operand],
                true)) {
          return "a unified closure operand exceeds its typed bank";
        }
      }
      if (!add_weight_use(point, program, instruction.payload1)) {
        return "a unified closure leaf reads an invalid weight";
      }
      break;
    }
    case SurfaceSvmBytecodeKind::set_normal:
      if (!add_value_use(point, program, instruction.payload0, false,
                         SurfaceValueBank::vector, true)) {
        return "the unified SetNormal boundary has no initialized vector";
      }
      point.clear_definitions_after_uses = true;
      break;
    case SurfaceSvmBytecodeKind::end:
      continue;
    case SurfaceSvmBytecodeKind::invalid:
      return "the unified surface SVM contains an unknown opcode";
    }
    point.successors.emplace_back(
        static_cast<std::uint32_t>(index + 1u));
  }

  std::string initialization_diagnostic;
  if (!verify_definite_initialization(points, initialization_diagnostic)) {
    return initialization_diagnostic;
  }
  return {};
}

SurfaceSvmProgramImage lower_surface_svm_program(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    const SurfaceSvmSchedulePlan &schedule,
    const SurfaceSvmStoragePlan &storage) {
  static_assert(
      std::is_trivially_copyable_v<SurfaceSvmBytecodeInstruction>);
  static_assert(sizeof(SurfaceSvmBytecodeInstruction) == 16u);
  static_assert(static_cast<std::uint32_t>(ClosureOperation::refraction) <
                32u);
  static_assert(static_cast<std::uint32_t>(BssrdfMethod::random_walk_skin) <
                4u);
  if (!closure_plan.compatible(program) ||
      !dependencies.compatible(program) ||
      !storage.compatible(program, schedule)) {
    return reject("cannot lower incompatible unified surface SVM inputs");
  }
  if (schedule.endpoints == 0u ||
      (schedule.endpoints & ~all_surface_closure_endpoints) != 0u) {
    return reject("cannot lower an invalid endpoint projection");
  }
  if (storage.stack_lanes > surface_svm_invalid_packed_weight_slot) {
    return reject("the unified lane stack exceeds packed weight addresses");
  }

  SurfaceValueStoragePlan value_storage;
  value_storage.valid = true;
  value_storage.locations = storage.locations;
  const std::array typed_slot_counts{
      storage.scalar_slots, storage.vector_slots,
      storage.unsigned_integer_slots};
  for (auto &location : value_storage.locations) {
    if (location.storage != SurfaceValueStorageClass::local_slot) {
      continue;
    }
    const auto typed_bank = bank_index(location.bank);
    if (typed_bank >= typed_slot_counts.size() ||
        location.index >= typed_slot_counts[typed_bank]) {
      return reject("the unified value location exceeds its typed coloring");
    }
    const auto lane =
        static_cast<std::uint64_t>(storage.lane_bases[typed_bank]) +
        static_cast<std::uint64_t>(location.index) *
            bank_lane_width(location.bank);
    if (lane >= storage.stack_lanes) {
      return reject("the unified value location exceeds its lane stack");
    }
    location.index = static_cast<std::uint32_t>(lane);
  }
  // The established value serializer treats each type as an independent
  // address domain. Supplying the common physical extent makes it serialize
  // the already remapped lane offsets without changing its record format.
  value_storage.scalar_slots = storage.stack_lanes;
  value_storage.vector_slots = storage.stack_lanes;
  value_storage.unsigned_integer_slots = storage.stack_lanes;
  value_storage.active_values = storage.active_values;
  value_storage.parameter_values = storage.parameter_values;
  value_storage.alias_values = storage.alias_values;
  for (const auto &instruction : schedule.instructions) {
    if (instruction.kind != SurfaceSvmScheduleInstructionKind::value) {
      continue;
    }
    if (instruction.source >= program.value_instructions().size() ||
        instruction.source >= storage.representatives.size()) {
      return reject("a scheduled value is outside the surface program");
    }
    if (storage.representatives[instruction.source] == instruction.source) {
      value_storage.instructions.emplace_back(
          ValueExpressionId{instruction.source});
    } else if (program.value_instructions()[instruction.source].operation !=
               ValueOperation::passthrough) {
      return reject("a non-Passthrough scheduled value was aliased");
    }
  }
  if (!value_storage.compatible(program)) {
    return reject("the unified value projection is not a total quotient");
  }
  auto values = lower_surface_value_program(program, value_storage);
  if (!values.valid) {
    return reject("unified value lowering: " + values.diagnostic);
  }
  if (values.instructions.size() != value_storage.instructions.size()) {
    return reject("unified value lowering changed instruction cardinality");
  }

  std::vector<std::uint32_t> lowered_value_index(
      program.value_instructions().size(), invalid_index);
  for (auto index = std::size_t{};
       index < value_storage.instructions.size(); ++index) {
    lowered_value_index[value_storage.instructions[index].value] =
        static_cast<std::uint32_t>(index);
  }

  std::vector<std::uint32_t> bytecode_pc(schedule.instructions.size());
  auto bytecode_count = std::size_t{};
  for (auto index = std::size_t{}; index < schedule.instructions.size();
       ++index) {
    bytecode_pc[index] = static_cast<std::uint32_t>(bytecode_count);
    const auto &instruction = schedule.instructions[index];
    if (instruction.kind != SurfaceSvmScheduleInstructionKind::value ||
        scheduled_value_emits(storage, instruction)) {
      if (++bytecode_count >
          std::numeric_limits<std::uint32_t>::max()) {
        return reject("the unified instruction stream exceeds 32-bit PCs");
      }
    }
  }

  SurfaceSvmProgramImage result;
  result.endpoints = schedule.endpoints;
  result.instructions.reserve(bytecode_count);
  result.value_operands = std::move(values.operands);
  result.value_metadata = std::move(values.metadata);
  result.static_data = std::move(values.static_data);
  result.value_addresses = std::move(values.value_addresses);
  result.stack_lanes = storage.stack_lanes;
  result.scalar_slots = storage.scalar_slots;
  result.vector_slots = storage.vector_slots;
  result.unsigned_integer_slots = storage.unsigned_integer_slots;
  result.flags = values.flags;

  const auto value_address = [&](ValueExpressionId value,
                                 bool required,
                                 std::uint32_t &encoded) {
    encoded = SurfaceValueAddress::invalid_value;
    if (!value.valid()) {
      return !required;
    }
    if (value.value >= result.value_addresses.size() ||
        value.value >= schedule.value_regions.size()) {
      return false;
    }
    const auto active = schedule.value_regions[value.value] !=
                        surface_svm_invalid_region;
    if (!active) {
      return !required;
    }
    const auto address =
        SurfaceValueAddress{result.value_addresses[value.value]};
    if (!address.valid()) {
      return false;
    }
    encoded = address.encoded();
    return true;
  };
  const auto weight_slot = [&](SurfaceSvmWeightId weight,
                               std::uint32_t &slot) {
    if (!weight.valid()) {
      slot = surface_svm_root_weight_slot;
      return true;
    }
    if (weight.value >= storage.weight_locations.size()) {
      return false;
    }
    slot = storage.weight_locations[weight.value];
    return slot < result.stack_lanes &&
           slot != surface_svm_invalid_packed_weight_slot;
  };

  const auto &closures = program.closure_instructions();
  for (auto semantic_pc = std::size_t{};
       semantic_pc < schedule.instructions.size(); ++semantic_pc) {
    const auto &instruction = schedule.instructions[semantic_pc];
    switch (instruction.kind) {
    case SurfaceSvmScheduleInstructionKind::value: {
      if (!scheduled_value_emits(storage, instruction)) {
        break;
      }
      const auto lowered = lowered_value_index[instruction.source];
      if (lowered == invalid_index || lowered >= values.instructions.size()) {
        return reject("a scheduled value has no lowered bytecode record");
      }
      result.instructions.emplace_back(
          make_surface_svm_value_instruction(values.instructions[lowered]));
      ++result.value_instruction_count;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::mix_closure: {
      if (!instruction.weight.valid() ||
          instruction.weight.value >= schedule.weight_expressions.size() ||
          instruction.source >= closures.size()) {
        return reject("a scheduled Mix has no weight expression");
      }
      const auto &primary =
          schedule.weight_expressions[instruction.weight.value];
      if ((primary.operation != SurfaceSvmWeightOperation::mix_left &&
           primary.operation != SurfaceSvmWeightOperation::mix_right) ||
          primary.source_mix.value != instruction.source ||
          closures[instruction.source].operation != ClosureOperation::mix ||
          closures[instruction.source].factor != primary.factor) {
        return reject("a scheduled Mix disagrees with its source closure");
      }
      auto factor = std::uint32_t{};
      auto parent = std::uint32_t{};
      if (!value_address(primary.factor, true, factor) ||
          !weight_slot(primary.a, parent)) {
        return reject("a scheduled Mix has no factor or parent address");
      }
      auto left = surface_svm_invalid_packed_weight_slot;
      auto right = surface_svm_invalid_packed_weight_slot;
      auto control = surface_svm_mix_closure_opcode;
      const auto publish = [&](SurfaceSvmWeightId output) {
        if (!output.valid() ||
            output.value >= schedule.weight_expressions.size()) {
          return false;
        }
        const auto &expression = schedule.weight_expressions[output.value];
        auto slot = std::uint32_t{};
        if (expression.source_mix != primary.source_mix ||
            expression.factor != primary.factor ||
            expression.a != primary.a || !weight_slot(output, slot) ||
            slot > surface_svm_packed_weight_slot_mask) {
          return false;
        }
        if (expression.operation == SurfaceSvmWeightOperation::mix_left) {
          if (left != surface_svm_invalid_packed_weight_slot) {
            return false;
          }
          left = slot;
          control |= surface_svm_mix_left_result_bit;
          return true;
        }
        if (expression.operation == SurfaceSvmWeightOperation::mix_right) {
          if (right != surface_svm_invalid_packed_weight_slot) {
            return false;
          }
          right = slot;
          control |= surface_svm_mix_right_result_bit;
          return true;
        }
        return false;
      };
      if (!publish(instruction.weight) ||
          (instruction.secondary_weight.valid() &&
           !publish(instruction.secondary_weight)) ||
          left == right) {
        return reject("a scheduled Mix has malformed result weights");
      }
      result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
          .control = control,
          .payload0 = factor,
          .payload1 = parent,
          .payload2 = left | (right << 16u)});
      ++result.mix_instruction_count;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::add_closure_weight: {
      if (!instruction.weight.valid() ||
          instruction.weight.value >= schedule.weight_expressions.size()) {
        return reject("a scheduled weight Add has no expression");
      }
      const auto &expression =
          schedule.weight_expressions[instruction.weight.value];
      auto a = std::uint32_t{};
      auto b = std::uint32_t{};
      auto output = std::uint32_t{};
      if (expression.operation != SurfaceSvmWeightOperation::add ||
          instruction.source != instruction.weight.value ||
          !weight_slot(expression.a, a) || !weight_slot(expression.b, b) ||
          !weight_slot(instruction.weight, output) ||
          output == surface_svm_root_weight_slot) {
        return reject("a scheduled weight Add is inconsistent");
      }
      result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
          .control = surface_svm_add_closure_weight_opcode,
          .payload0 = a,
          .payload1 = b,
          .payload2 = output});
      ++result.weight_add_instruction_count;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::jump_if_one:
    case SurfaceSvmScheduleInstructionKind::jump_if_zero: {
      if (instruction.source >= closures.size() ||
          closures[instruction.source].operation != ClosureOperation::mix ||
          instruction.target <= semantic_pc ||
          instruction.target >= schedule.instructions.size()) {
        return reject("a scheduled closure guard is malformed");
      }
      auto factor = std::uint32_t{};
      if (!value_address(closures[instruction.source].factor, true, factor)) {
        return reject("a scheduled closure guard has no factor address");
      }
      const auto target = bytecode_pc[instruction.target];
      if (target <= result.instructions.size()) {
        return reject("an erased alias collapsed a guard target");
      }
      result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
          .control =
              instruction.kind ==
                      SurfaceSvmScheduleInstructionKind::jump_if_one
                  ? surface_svm_jump_if_one_opcode
                  : surface_svm_jump_if_zero_opcode,
          .payload0 = factor,
          .payload1 = target,
          .payload2 = surface_svm_invalid_payload});
      ++result.conditional_branch_count;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::closure_leaf: {
      if (instruction.source >= closures.size()) {
        return reject("a scheduled closure leaf is outside the program");
      }
      const auto id = ClosureExpressionId{instruction.source};
      const auto &closure = closures[instruction.source];
      if (!surface_closure_is_leaf_operation(closure.operation)) {
        return reject("a scheduled closure leaf names a control operation");
      }
      const auto endpoints =
          closure_endpoints(dependencies, id, schedule.endpoints);
      if (endpoints == 0u) {
        return reject("a scheduled closure leaf has no active endpoint");
      }
      const auto operands = surface_closure_operands(closure);
      if (operands.size() !=
          surface_closure_operand_count(closure.operation) ||
          result.closure_operands.size() >
              std::numeric_limits<std::uint32_t>::max() ||
          operands.size() >
              std::numeric_limits<std::uint32_t>::max() -
                  result.closure_operands.size()) {
        return reject("a scheduled closure leaf has invalid operand extent");
      }
      const auto operand_begin =
          static_cast<std::uint32_t>(result.closure_operands.size());
      for (const auto operand : operands) {
        auto address = std::uint32_t{};
        const auto required =
            operand.valid() &&
            operand.value < schedule.value_regions.size() &&
            schedule.value_regions[operand.value] !=
                surface_svm_invalid_region;
        if (!value_address(operand, required, address)) {
          return reject("a scheduled closure operand has no typed address");
        }
        result.closure_operands.emplace_back(address);
      }
      auto features = closure.operation == ClosureOperation::principled
                          ? closure_plan.entry(id).principled_features
                          : PrincipledClosureFeatureMask{};
      if ((endpoints & surface_closure_endpoint_bit(
                           SurfaceClosureEndpoint::physical)) == 0u) {
        features &= surface_closure_emission_principled_features;
      }
      const auto physical =
          (endpoints & surface_closure_endpoint_bit(
                           SurfaceClosureEndpoint::physical)) != 0u;
      auto weight = std::uint32_t{};
      if (!weight_slot(instruction.weight, weight)) {
        return reject("a scheduled closure leaf has no weight address");
      }
      const auto closure_control = make_surface_closure_control(
          closure, endpoints,
          physical && closure_plan.entry(id).microfacet_anisotropy,
          physical && closure_plan.entry(id).thin_film);
      result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
          .control = surface_svm_closure_leaf_opcode |
                     (closure_control << surface_svm_closure_control_shift),
          .payload0 = operand_begin,
          .payload1 = weight,
          .payload2 = features});
      result.used_closure_operations |=
          1u << static_cast<std::uint32_t>(closure.operation);
      result.used_principled_features |= features;
      ++result.closure_leaf_count;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::end:
      result.instructions.emplace_back(SurfaceSvmBytecodeInstruction{
          .control = surface_svm_end_opcode,
          .payload0 = surface_svm_invalid_payload,
          .payload1 = surface_svm_invalid_payload,
          .payload2 = surface_svm_invalid_payload});
      break;
    }
  }
  if (result.instructions.size() != bytecode_count ||
      result.mix_instruction_count != schedule.mix_instruction_count ||
      result.weight_add_instruction_count !=
          schedule.weight_add_instruction_count ||
      result.conditional_branch_count != schedule.conditional_branch_count ||
      result.closure_leaf_count != schedule.closure_leaf_count) {
    return reject("unified lowering changed schedule cardinality");
  }
  result.valid = true;
  if (const auto diagnostic = validate_surface_svm_program_image(result);
      !diagnostic.empty()) {
    return reject("lowered unified surface SVM: " + diagnostic);
  }
  return result;
}

} // namespace psycles::compiler
