#include <psycles/compiler/surface_execution_plan.h>

#include "surface_value_variant_interner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

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

[[nodiscard]] SurfaceValueStoragePlan reject(std::string diagnostic) {
  SurfaceValueStoragePlan result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceValueProgramImage reject_image(std::string diagnostic) {
  SurfaceValueProgramImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceValueSceneImage
reject_scene_image(std::string diagnostic) {
  SurfaceValueSceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceValueExecutableScene
reject_executable_scene(std::string diagnostic) {
  SurfaceValueExecutableScene result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool encode_location(const SurfaceValueLocation &location,
                                   SurfaceValueAddress &address) noexcept {
  if (location.storage == SurfaceValueStorageClass::inactive ||
      location.index > SurfaceValueAddress::index_mask) {
    return false;
  }
  const auto bank = static_cast<std::uint32_t>(location.bank);
  if (bank > static_cast<std::uint32_t>(SurfaceValueBank::unsigned_integer)) {
    return false;
  }
  address = SurfaceValueAddress{
      (location.storage == SurfaceValueStorageClass::parameter
           ? SurfaceValueAddress::parameter_bit
           : 0u) |
      (bank << SurfaceValueAddress::bank_shift) | location.index};
  return address.valid();
}

[[nodiscard]] constexpr std::uint32_t
pack_operand_lane(std::uint32_t word, SurfaceValueOperandAddress operand,
                  std::size_t lane) noexcept {
  const auto shift = surface_value_operand_lane_bits * lane;
  const auto mask = std::uint32_t{0xffffu} << shift;
  return (word & ~mask) |
         (static_cast<std::uint32_t>(operand.encoded()) << shift);
}

[[nodiscard]] bool address_fits_program(SurfaceValueAddress address,
                                        const SurfaceValueProgramImage &program,
                                        bool allow_invalid) noexcept {
  if (!address.valid()) {
    return allow_invalid;
  }
  const auto bank = address.bank();
  if (static_cast<std::uint32_t>(bank) >
      static_cast<std::uint32_t>(SurfaceValueBank::unsigned_integer)) {
    return false;
  }
  if (address.parameter()) {
    return true;
  }
  switch (bank) {
  case SurfaceValueBank::scalar:
    return address.index() < program.scalar_slots;
  case SurfaceValueBank::vector:
    return address.index() < program.vector_slots;
  case SurfaceValueBank::unsigned_integer:
    return address.index() < program.unsigned_integer_slots;
  }
  return false;
}

[[nodiscard]] std::string validate_surface_value_program_image_impl(
    const SurfaceValueProgramImage &program) {
  if (!program.valid) {
    return "source value program is invalid: " + program.diagnostic;
  }
  if (program.scalar_slots > SurfaceValueAddress::index_mask + 1u ||
      program.vector_slots > SurfaceValueAddress::index_mask + 1u ||
      program.unsigned_integer_slots > SurfaceValueAddress::index_mask + 1u) {
    return "a typed local bank exceeds the address encoding";
  }
  if ((program.flags & ~surface_value_program_flag_mask) != 0u) {
    return "a value program has unknown flags";
  }
  for (const auto encoded : program.value_addresses) {
    if (!address_fits_program(SurfaceValueAddress{encoded}, program, true)) {
      return "a value address exceeds its typed local bank";
    }
  }

  auto operand_word_cursor = std::size_t{0u};
  auto transition_count = std::size_t{0u};
  // Definite initialization is a forward must-property. Parameters are
  // immutable inputs and therefore initialized at entry; a local slot is
  // initialized exactly after an ordinary instruction writes it. Checking
  // operands before inserting the result preserves the interpreter's
  // read-before-write contract when the allocator reuses a dying operand.
  // Sparse sets make validation proportional to the bytecode itself instead
  // of trusting potentially malformed bank extents for host allocation.
  std::array<std::unordered_set<std::uint32_t>, 3u> initialized_locals;
  const auto local_is_initialized = [&](SurfaceValueAddress address) {
    return address.parameter() ||
           initialized_locals[bank_index(address.bank())].contains(
               address.index());
  };
  for (const auto &instruction : program.instructions) {
    if (is_surface_value_surface_normal_transition(instruction)) {
      if (instruction.operand_payload != surface_value_invalid_operand_word ||
          instruction.metadata_index != SurfaceValueAddress::invalid_value) {
        return "a surface-normal transition owns operands or metadata";
      }
      const auto output = SurfaceValueAddress{instruction.result};
      if (!address_fits_program(output, program, false) ||
          output.bank() != SurfaceValueBank::vector) {
        return "a surface-normal transition has no vector output";
      }
      if (!local_is_initialized(output)) {
        return "a surface-normal transition reads an uninitialized local";
      }
      if (++transition_count != 1u) {
        return "a value program contains multiple surface-normal "
               "transitions";
      }
      continue;
    }
    if ((instruction.control & ~surface_value_control_mask) != 0u ||
        static_cast<std::uint32_t>(surface_value_operation(instruction)) >
            static_cast<std::uint32_t>(ValueOperation::ambient_occlusion)) {
      return "an instruction has an invalid control word";
    }
    const auto operation = surface_value_operation(instruction);
    const auto operand_count = value_operation_operand_count(operation);
    const auto inline_operands =
        operand_count <= surface_value_inline_operand_capacity;
    const auto operand_word_count =
        inline_operands ? std::size_t{}
                        : static_cast<std::size_t>(
                              surface_value_operand_word_count(operand_count));
    if (!inline_operands) {
      if (instruction.operand_payload != operand_word_cursor) {
        return "the packed operand stream is not densely ordered";
      }
      if (operand_word_count > program.operands.size() - operand_word_cursor) {
        return "an instruction operand range exceeds the packed stream";
      }
    } else if (operand_count == 0u && instruction.operand_payload !=
                                          surface_value_invalid_operand_word) {
      return "a nullary instruction owns an operand payload";
    }
    const auto result = SurfaceValueAddress{instruction.result};
    if (!address_fits_program(result, program, false) || result.parameter() ||
        result.bank() != surface_value_result_bank(instruction)) {
      return "an instruction result is inconsistent with its typed bank";
    }
    for (auto operand_index = std::size_t{0u}; operand_index < operand_count;
         ++operand_index) {
      const auto word =
          inline_operands
              ? instruction.operand_payload
              : program
                    .operands[operand_word_cursor +
                              operand_index / surface_value_operands_per_word];
      const auto compact = surface_value_operand_from_word(
          word, operand_index % surface_value_operands_per_word);
      if (!compact.valid()) {
        return "an instruction contains an invalid packed operand address";
      }
      const auto operand = compact.expanded();
      if (!address_fits_program(operand, program, false)) {
        return "an instruction operand exceeds its typed bank";
      }
      if (!local_is_initialized(operand)) {
        return "an instruction reads an uninitialized local";
      }
    }
    if ((operand_count % surface_value_operands_per_word) != 0u) {
      const auto last_word =
          inline_operands
              ? instruction.operand_payload
              : program.operands[operand_word_cursor + operand_word_count - 1u];
      if (surface_value_operand_from_word(last_word, 1u).encoded() !=
          SurfaceValueOperandAddress::invalid_value) {
        return "an odd-arity instruction has non-canonical operand padding";
      }
    }
    if (instruction.metadata_index != SurfaceValueAddress::invalid_value &&
        instruction.metadata_index >= program.metadata.size()) {
      return "an instruction metadata index exceeds the side table";
    }
    const auto actual_immediate = surface_value_svm_immediate(instruction);
    if (!surface_value_operation_uses_svm_immediate(operation)) {
      if (actual_immediate != 0u) {
        return "an instruction assigns immediate data to an opcode without "
               "an immediate contract";
      }
    } else {
      auto static_u0 = std::uint64_t{};
      auto static_u1 = std::uint64_t{};
      if (instruction.metadata_index != SurfaceValueAddress::invalid_value) {
        const auto &metadata = program.metadata[instruction.metadata_index];
        static_u0 = metadata.static_u0;
        static_u1 = metadata.static_u1;
      }
      if (!surface_value_svm_static_fields_valid(operation, static_u0,
                                                 static_u1)) {
        return "an instruction has immutable fields outside its immediate "
               "contract";
      }
      const auto expected_immediate =
          make_surface_value_svm_immediate(operation, static_u0, static_u1);
      if (actual_immediate != expected_immediate) {
        return "an instruction immediate disagrees with immutable metadata";
      }
    }
    initialized_locals[bank_index(result.bank())].emplace(result.index());
    operand_word_cursor += operand_word_count;
  }
  if (operand_word_cursor != program.operands.size()) {
    return "the packed operand stream has an unreferenced suffix";
  }
  if ((program.flags &
       surface_value_program_automatic_normal_uses_undisplaced_geometry) !=
          0u &&
      transition_count != 1u) {
    return "undisplaced automatic-normal evaluation has no transition";
  }
  for (const auto &metadata : program.metadata) {
    if (metadata.static_table_begin > program.static_data.size() ||
        metadata.static_table_count >
            program.static_data.size() - metadata.static_table_begin) {
      return "a metadata static-table range exceeds the data stream";
    }
  }
  return {};
}

[[nodiscard]] bool add_scene_extent(std::size_t &total, std::size_t count) {
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (count > limit - total) {
    return false;
  }
  total += count;
  return true;
}

// Sequential composition theorem used here:
//
//   eval(N, automatic_point(P), L); n = read(output_N, L);
//   eval(R, P[N := n], L)
//
// is observationally equal to interpreting `N; commit(output_N); R` once.
// N and R are independently topologically closed, and commit is sequenced
// before every R instruction. Therefore their typed-slot allocations may
// overlap: no N slot is live after commit except the value consumed by commit
// itself. Parameters remain late-bound addresses and require no instructions.
[[nodiscard]] SurfaceValueProgramImage
make_surface_normal_transaction_image(const SurfaceValueProgramImage &normal,
                                      const SurfaceValueProgramImage &root,
                                      std::uint32_t normal_output,
                                      bool uses_undisplaced_geometry) {
  if (const auto diagnostic = validate_surface_value_program_image_impl(normal);
      !diagnostic.empty()) {
    return reject_image("automatic-normal prefix: " + diagnostic);
  }
  if (const auto diagnostic = validate_surface_value_program_image_impl(root);
      !diagnostic.empty()) {
    return reject_image("endpoint root: " + diagnostic);
  }
  if (normal.flags != 0u || root.flags != 0u ||
      std::any_of(normal.instructions.begin(), normal.instructions.end(),
                  is_surface_value_surface_normal_transition) ||
      std::any_of(root.instructions.begin(), root.instructions.end(),
                  is_surface_value_surface_normal_transition)) {
    return reject_image("a surface-normal transaction cannot nest");
  }

  auto instruction_count = normal.instructions.size();
  auto operand_count = normal.operands.size();
  auto metadata_count = normal.metadata.size();
  auto static_data_count = normal.static_data.size();
  if (!add_scene_extent(instruction_count, 1u) ||
      !add_scene_extent(instruction_count, root.instructions.size()) ||
      !add_scene_extent(operand_count, root.operands.size()) ||
      !add_scene_extent(metadata_count, root.metadata.size()) ||
      !add_scene_extent(static_data_count, root.static_data.size())) {
    return reject_image("a surface-normal transaction exceeds 32-bit offsets");
  }

  SurfaceValueProgramImage result;
  result.instructions.reserve(instruction_count);
  result.operands.reserve(operand_count);
  result.metadata.reserve(metadata_count);
  result.static_data.reserve(static_data_count);
  result.value_addresses = root.value_addresses;
  result.scalar_slots = std::max(normal.scalar_slots, root.scalar_slots);
  result.vector_slots = std::max(normal.vector_slots, root.vector_slots);
  result.unsigned_integer_slots =
      std::max(normal.unsigned_integer_slots, root.unsigned_integer_slots);
  result.flags =
      uses_undisplaced_geometry
          ? surface_value_program_automatic_normal_uses_undisplaced_geometry
          : 0u;

  const auto append = [&](const SurfaceValueProgramImage &source) {
    const auto operand_word_begin =
        static_cast<std::uint32_t>(result.operands.size());
    const auto metadata_begin =
        static_cast<std::uint32_t>(result.metadata.size());
    const auto static_data_begin =
        static_cast<std::uint32_t>(result.static_data.size());
    for (auto instruction : source.instructions) {
      if (!is_surface_value_surface_normal_transition(instruction) &&
          surface_value_operand_count(instruction) >
              surface_value_inline_operand_capacity) {
        instruction.operand_payload += operand_word_begin;
      }
      if (instruction.metadata_index != SurfaceValueAddress::invalid_value) {
        instruction.metadata_index += metadata_begin;
      }
      result.instructions.emplace_back(instruction);
    }
    result.operands.insert(result.operands.end(), source.operands.begin(),
                           source.operands.end());
    for (auto metadata : source.metadata) {
      metadata.static_table_begin += static_data_begin;
      result.metadata.emplace_back(metadata);
    }
    result.static_data.insert(result.static_data.end(),
                              source.static_data.begin(),
                              source.static_data.end());
  };

  append(normal);
  result.instructions.emplace_back(SurfaceValueBytecodeInstruction{
      .control = surface_value_surface_normal_transition_control,
      .result = normal_output,
      .operand_payload = surface_value_invalid_operand_word,
      .metadata_index = SurfaceValueAddress::invalid_value});
  append(root);
  result.valid = true;
  if (const auto diagnostic = validate_surface_value_program_image_impl(result);
      !diagnostic.empty()) {
    return reject_image("composed surface-normal transaction: " + diagnostic);
  }
  return result;
}

[[nodiscard]] std::uint32_t
value_stack_words(const ValueInstruction &instruction) noexcept {
  SurfaceValueBank bank;
  if (!classify_surface_value_type(instruction.result_type, bank)) {
    return 0u;
  }
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

struct SurfaceValueSchedulePressure {
  bool valid{};
  std::array<std::uint32_t, 3u> slots{};

  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return static_cast<std::uint64_t>(slots[0u]) * sizeof(float) +
           static_cast<std::uint64_t>(slots[1u]) * sizeof(float) * 3u +
           static_cast<std::uint64_t>(slots[2u]) * sizeof(std::uint64_t);
  }
};

[[nodiscard]] SurfaceValueSchedulePressure
measure_schedule_pressure(const std::vector<ValueInstruction> &values,
                          const std::vector<bool> &active,
                          const std::vector<bool> &outputs,
                          const std::vector<std::uint32_t> &representatives,
                          const std::vector<ValueExpressionId> &schedule) {
  std::vector<std::uint32_t> remaining_uses(values.size(), 0u);
  std::vector<std::uint32_t> output_uses(values.size(), 0u);
  auto computed_count = std::size_t{0u};
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    const auto representative = representatives[index];
    const auto executable =
        representative == index &&
        values[index].operation != ValueOperation::parameter;
    computed_count += executable ? 1u : 0u;
    if (executable) {
      for (const auto operand : values[index].operands) {
        const auto source = representatives[operand.value];
        if (remaining_uses[source] ==
            std::numeric_limits<std::uint32_t>::max()) {
          return {};
        }
        ++remaining_uses[source];
      }
    }
    if (outputs[index]) {
      if (remaining_uses[representative] ==
              std::numeric_limits<std::uint32_t>::max() ||
          output_uses[representative] ==
              std::numeric_limits<std::uint32_t>::max()) {
        return {};
      }
      ++remaining_uses[representative];
      ++output_uses[representative];
    }
  }
  if (schedule.size() != computed_count) {
    return {};
  }

  std::vector<bool> emitted(values.size(), false);
  std::array<std::uint32_t, 3u> live{};
  SurfaceValueSchedulePressure result;
  for (const auto id : schedule) {
    if (!id.valid() || id.value >= values.size() || !active[id.value] ||
        emitted[id.value] || representatives[id.value] != id.value ||
        values[id.value].operation == ValueOperation::parameter) {
      return {};
    }
    const auto &instruction = values[id.value];

    // All operands are read before any dying location is made reusable.
    for (const auto operand : instruction.operands) {
      const auto source = representatives[operand.value];
      if (values[source].operation != ValueOperation::parameter &&
          !emitted[source]) {
        return {};
      }
    }
    for (const auto operand : instruction.operands) {
      const auto source = representatives[operand.value];
      if (remaining_uses[source] == 0u) {
        return {};
      }
      --remaining_uses[source];
      if (remaining_uses[source] == 0u &&
          values[source].operation != ValueOperation::parameter) {
        SurfaceValueBank operand_bank;
        if (!classify_surface_value_type(values[source].result_type,
                                         operand_bank)) {
          return {};
        }
        auto &bank_live = live[bank_index(operand_bank)];
        if (bank_live == 0u) {
          return {};
        }
        --bank_live;
      }
    }
    if (remaining_uses[id.value] == 0u) {
      return {};
    }
    SurfaceValueBank result_bank;
    if (!classify_surface_value_type(instruction.result_type, result_bank)) {
      return {};
    }
    const auto result_bank_index = bank_index(result_bank);
    if (live[result_bank_index] == std::numeric_limits<std::uint32_t>::max()) {
      return {};
    }
    ++live[result_bank_index];
    result.slots[result_bank_index] =
        std::max(result.slots[result_bank_index], live[result_bank_index]);
    emitted[id.value] = true;
  }
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (active[index] && representatives[index] == index &&
        remaining_uses[index] != output_uses[index]) {
      return {};
    }
  }
  result.valid = true;
  return result;
}

// Cycles schedules the value DAG from its output roots with a Sethi-Ullman
// heuristic before assigning SVM stack slots. SurfaceProgram already proves
// that every operand precedes its consumer, so the same recurrence can be
// evaluated in one forward pass without mutating the graph.
//
// Let w(v) be the result width in 32-bit stack words and c(v) the number of
// distinct live consumer instructions, including one synthetic consumer when
// v is a terminal closure input. For producers p_i sorted by descending
// (current(p_i) - w(p_i)),
//
//   current(v) = c(v) > 1 ? w(v) : SU(v)
//   SU(v) = max_i(sum_{j<i} w(p_j) + current(p_i),
//                  sum_i w(p_i) + w(v)).
//
// The recurrence is exact for trees and a deterministic heuristic for DAGs.
// A second dependency walk emits each instruction once. Reordering is sound
// because ValueInstruction operations are pure: all device state is explicit
// in their operands, immutable metadata, parameters, ShaderServices, and
// SurfacePoint.
[[nodiscard]] bool make_sethi_ullman_value_schedule(
    const std::vector<ValueInstruction> &values,
    const std::vector<bool> &active, const std::vector<bool> &outputs,
    const std::vector<std::uint32_t> &representatives,
    std::vector<ValueExpressionId> &schedule, std::string &diagnostic) {
  const auto count = values.size();
  std::vector<std::vector<std::uint32_t>> producers(count);
  std::vector<std::uint32_t> instruction_consumers(count, 0u);
  std::vector<std::uint32_t> total_consumers(count, 0u);
  auto computed_count = std::size_t{0u};

  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (!active[index] || representatives[index] != index ||
        values[index].operation == ValueOperation::parameter) {
      continue;
    }
    ++computed_count;
    auto &dependencies = producers[index];
    dependencies.reserve(values[index].operands.size());
    for (const auto operand : values[index].operands) {
      const auto source = representatives[operand.value];
      if (values[source].operation == ValueOperation::parameter ||
          std::find(dependencies.begin(), dependencies.end(), source) !=
              dependencies.end()) {
        continue;
      }
      dependencies.emplace_back(source);
      if (instruction_consumers[source] ==
          std::numeric_limits<std::uint32_t>::max()) {
        diagnostic = "the value consumer count exceeds the scheduler encoding";
        return false;
      }
      ++instruction_consumers[source];
    }
  }
  total_consumers = instruction_consumers;
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (outputs[index]) {
      const auto representative = representatives[index];
      if (total_consumers[representative] ==
          std::numeric_limits<std::uint32_t>::max()) {
        diagnostic =
            "the terminal value consumer count exceeds the scheduler encoding";
        return false;
      }
      ++total_consumers[representative];
    }
  }

  std::vector<std::uint32_t> sethi_ullman(count, 0u);
  const auto current_number = [&](std::uint32_t index) noexcept {
    const auto width = value_stack_words(values[index]);
    return total_consumers[index] > 1u ? width : sethi_ullman[index];
  };
  const auto order_key = [&](std::uint32_t index) noexcept {
    return current_number(index) - value_stack_words(values[index]);
  };
  const auto order_before = [&](std::uint32_t lhs, std::uint32_t rhs) noexcept {
    const auto lhs_key = order_key(lhs);
    const auto rhs_key = order_key(rhs);
    return lhs_key > rhs_key || (lhs_key == rhs_key && lhs < rhs);
  };

  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (!active[index] || representatives[index] != index ||
        values[index].operation == ValueOperation::parameter) {
      continue;
    }
    auto &dependencies = producers[index];
    std::sort(dependencies.begin(), dependencies.end(), order_before);
    auto live_words = std::uint64_t{0u};
    auto peak_words = std::uint64_t{0u};
    for (const auto dependency : dependencies) {
      peak_words = std::max(
          peak_words,
          live_words + static_cast<std::uint64_t>(current_number(dependency)));
      live_words += value_stack_words(values[dependency]);
    }
    peak_words =
        std::max(peak_words, live_words + value_stack_words(values[index]));
    if (peak_words > std::numeric_limits<std::uint32_t>::max()) {
      diagnostic = "the Sethi-Ullman stack estimate exceeds its encoding";
      return false;
    }
    sethi_ullman[index] = static_cast<std::uint32_t>(peak_words);
  }

  std::vector<std::uint32_t> sinks;
  sinks.reserve(computed_count);
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (active[index] && representatives[index] == index &&
        values[index].operation != ValueOperation::parameter &&
        instruction_consumers[index] == 0u) {
      sinks.emplace_back(static_cast<std::uint32_t>(index));
    }
  }
  std::sort(sinks.begin(), sinks.end(), order_before);

  std::vector<bool> emitted(count, false);
  schedule.clear();
  schedule.reserve(computed_count);
  std::function<void(std::uint32_t)> emit = [&](std::uint32_t index) {
    if (emitted[index]) {
      return;
    }
    for (const auto dependency : producers[index]) {
      emit(dependency);
    }
    emitted[index] = true;
    schedule.emplace_back(ValueExpressionId{index});
  };
  for (const auto sink : sinks) {
    emit(sink);
  }
  if (schedule.size() != computed_count) {
    diagnostic = "the value scheduler did not reach every active instruction";
    return false;
  }
  return true;
}

[[nodiscard]] bool schedule_surface_value_instructions(
    const std::vector<ValueInstruction> &values,
    const std::vector<bool> &active, const std::vector<bool> &outputs,
    const std::vector<std::uint32_t> &representatives,
    SurfaceValueStorageCapacity capacity,
    std::vector<ValueExpressionId> &schedule, std::string &diagnostic) {
  if (!make_sethi_ullman_value_schedule(
          values, active, outputs, representatives, schedule, diagnostic)) {
    return false;
  }
  const auto count = values.size();
  const auto computed_count = schedule.size();

  // Cycles' scalar-stack recurrence is a heuristic for DAGs, while Psycles
  // has three separately allocated typed banks. Retain two independent
  // witnesses in addition to the quotient schedule:
  //
  // 1. Source order is legal because SurfaceProgram is strictly topological.
  // 2. Projecting a legal schedule of the uncontracted graph through the
  //    identity quotient remains topological: deleting p = id(x) preserves
  //    x-before-use, and every surviving instruction keeps its relative
  //    order. The projected schedule cannot have greater pressure than that
  //    pre-optimization witness because it removes one definition and one
  //    copy edge without adding a value.
  //
  // Reject candidates outside the interpreter's component-wise capacity
  // vector, then minimize exact payload bytes. This prevents a byte-saving
  // vector-to-scalar trade from silently overflowing an independent bank and
  // gives identity contraction a formal no-regression schedule.
  std::vector<ValueExpressionId> source_schedule;
  source_schedule.reserve(computed_count);
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (active[index] && representatives[index] == index &&
        values[index].operation != ValueOperation::parameter) {
      source_schedule.emplace_back(
          ValueExpressionId{static_cast<std::uint32_t>(index)});
    }
  }
  std::vector<std::uint32_t> uncontracted_representatives(count);
  for (auto index = std::size_t{0u}; index < count; ++index) {
    uncontracted_representatives[index] = static_cast<std::uint32_t>(index);
  }
  std::vector<ValueExpressionId> uncontracted_schedule;
  if (!make_sethi_ullman_value_schedule(values, active, outputs,
                                        uncontracted_representatives,
                                        uncontracted_schedule, diagnostic)) {
    return false;
  }
  std::vector<ValueExpressionId> projected_schedule;
  projected_schedule.reserve(computed_count);
  for (const auto id : uncontracted_schedule) {
    if (representatives[id.value] == id.value) {
      projected_schedule.emplace_back(id);
    }
  }
  const auto cycles_pressure = measure_schedule_pressure(
      values, active, outputs, representatives, schedule);
  const auto source_pressure = measure_schedule_pressure(
      values, active, outputs, representatives, source_schedule);
  const auto projected_pressure = measure_schedule_pressure(
      values, active, outputs, representatives, projected_schedule);
  if (!cycles_pressure.valid || !source_pressure.valid ||
      !projected_pressure.valid) {
    diagnostic = "a value schedule violates read-before-write liveness";
    return false;
  }
  const std::array capacity_slots{capacity.scalar_slots, capacity.vector_slots,
                                  capacity.unsigned_integer_slots};
  const auto fits_capacity = [&](const SurfaceValueSchedulePressure &pressure) {
    return std::equal(pressure.slots.begin(), pressure.slots.end(),
                      capacity_slots.begin(),
                      [](std::uint32_t slots, std::uint32_t limit) noexcept {
                        return slots <= limit;
                      });
  };
  struct ScheduleCandidate {
    const std::vector<ValueExpressionId> *instructions;
    const SurfaceValueSchedulePressure *pressure;
  };
  const std::array candidates{
      ScheduleCandidate{.instructions = &schedule,
                        .pressure = &cycles_pressure},
      ScheduleCandidate{.instructions = &projected_schedule,
                        .pressure = &projected_pressure},
      ScheduleCandidate{.instructions = &source_schedule,
                        .pressure = &source_pressure}};
  const ScheduleCandidate *best = nullptr;
  for (const auto &candidate : candidates) {
    if (!fits_capacity(*candidate.pressure) ||
        (best != nullptr && candidate.pressure->payload_bytes() >=
                                best->pressure->payload_bytes())) {
      continue;
    }
    best = &candidate;
  }
  if (best == nullptr) {
    diagnostic = "no candidate value schedule fits the typed storage capacity";
    return false;
  }
  if (best->instructions != &schedule) {
    schedule = *best->instructions;
  }
  return true;
}

} // namespace

std::string
validate_surface_value_program_image(const SurfaceValueProgramImage &program) {
  return validate_surface_value_program_image_impl(program);
}

bool SurfaceValueStoragePlan::compatible(
    const SurfaceProgram &program) const noexcept {
  return valid && locations.size() == program.value_instructions().size() &&
         static_cast<std::size_t>(active_values) >= instructions.size() &&
         static_cast<std::size_t>(active_values) ==
             instructions.size() + parameter_values + alias_values;
}

std::size_t SurfaceValueStoragePlan::payload_bytes() const noexcept {
  return static_cast<std::size_t>(scalar_slots) * sizeof(float) +
         static_cast<std::size_t>(vector_slots) * sizeof(float) * 3u +
         static_cast<std::size_t>(unsigned_integer_slots) *
             sizeof(std::uint64_t);
}

SurfaceValueStoragePlan plan_surface_value_storage(
    const SurfaceProgram &program, const std::vector<bool> &active,
    const std::vector<bool> &outputs, SurfaceValueStorageCapacity capacity) {
  const auto &values = program.value_instructions();
  if (active.size() != values.size() || outputs.size() != values.size()) {
    return reject("value storage masks do not match the program");
  }

  SurfaceValueStoragePlan result;
  result.locations.resize(values.size());
  result.instructions.reserve(values.size());
  constexpr auto invalid_representative =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> representatives(values.size(),
                                             invalid_representative);
  std::vector<std::uint32_t> remaining_uses(values.size(), 0u);
  std::vector<std::uint32_t> output_uses(values.size(), 0u);

  // Construct the congruence quotient before liveness. A Passthrough is
  // defined by every evaluator as the identity on one physical execution
  // bank. Contracting p = id(x) is therefore semantics preserving exactly
  // when bank(p) == bank(x): every use and terminal observation of p becomes
  // a use of representative(x), while the nominal socket spelling remains in
  // SurfaceProgram for diagnostics and graph round-tripping.
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (outputs[index] && !active[index]) {
      return reject("a value output is not active");
    }
    if (!active[index]) {
      continue;
    }
    if (result.active_values == std::numeric_limits<std::uint32_t>::max()) {
      return reject("the active value count exceeds the plan encoding");
    }
    ++result.active_values;
    const auto &instruction = values[index];
    SurfaceValueBank bank;
    if (!classify_surface_value_type(instruction.result_type, bank)) {
      return reject("an active value has no supported typed storage bank");
    }
    if (instruction.operation == ValueOperation::parameter) {
      if (!instruction.parameter.valid() ||
          instruction.parameter.value >= program.parameters().size()) {
        return reject("an active parameter value has an invalid binding");
      }
      const auto &parameter = program.parameters()[instruction.parameter.value];
      if (parameter.id != instruction.parameter ||
          parameter.type != instruction.result_type) {
        return reject("an active parameter value has an inconsistent type");
      }
      result.locations[index] = {.storage = SurfaceValueStorageClass::parameter,
                                 .bank = bank,
                                 .index = instruction.parameter.value};
      representatives[index] = static_cast<std::uint32_t>(index);
      ++result.parameter_values;
    }
    for (const auto operand : instruction.operands) {
      if (!operand.valid() || operand.value >= values.size()) {
        return reject("an active instruction has an invalid operand");
      }
      if (operand.value >= index) {
        return reject("the value stream is not in strict topological order");
      }
      if (!active[operand.value]) {
        return reject("the active value mask is not transitively closed");
      }
    }
    if (instruction.operation == ValueOperation::passthrough) {
      if (instruction.operands.size() != 1u) {
        return reject("a passthrough does not have one source");
      }
      const auto source = instruction.operands.front().value;
      const auto representative = representatives[source];
      if (representative == invalid_representative) {
        return reject("a passthrough source has no representative");
      }
      SurfaceValueBank source_bank;
      if (!classify_surface_value_type(values[representative].result_type,
                                       source_bank) ||
          source_bank != bank) {
        return reject("a passthrough crosses physical execution banks");
      }
      representatives[index] = representative;
      if (result.alias_values == std::numeric_limits<std::uint32_t>::max()) {
        return reject("the value alias count exceeds the plan encoding");
      }
      ++result.alias_values;
    } else if (instruction.operation != ValueOperation::parameter) {
      representatives[index] = static_cast<std::uint32_t>(index);
    }
  }

  // Count uses only on quotient representatives. Identity nodes neither read
  // nor write device storage; keeping their apparent edges here would extend
  // live ranges and defeat the contraction.
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    const auto representative = representatives[index];
    if (representative == invalid_representative) {
      return reject("an active value has no quotient representative");
    }
    if (representative == index &&
        values[index].operation != ValueOperation::parameter) {
      for (const auto operand : values[index].operands) {
        const auto source = representatives[operand.value];
        if (remaining_uses[source] ==
            std::numeric_limits<std::uint32_t>::max()) {
          return reject("the value use count exceeds the plan encoding");
        }
        ++remaining_uses[source];
      }
    }
    if (outputs[index]) {
      if (remaining_uses[representative] ==
              std::numeric_limits<std::uint32_t>::max() ||
          output_uses[representative] ==
              std::numeric_limits<std::uint32_t>::max()) {
        return reject("the value output count exceeds the plan encoding");
      }
      ++remaining_uses[representative];
      ++output_uses[representative];
    }
  }

  if (!schedule_surface_value_instructions(
          values, active, outputs, representatives, capacity,
          result.instructions, result.diagnostic)) {
    return result;
  }

  std::array<std::vector<std::uint32_t>, 3u> free_slots;
  std::array<std::uint32_t, 3u> slot_counts{};
  for (const auto id : result.instructions) {
    const auto &instruction = values[id.value];

    // This is the formal read phase. Expired operands become reusable only
    // after every operand location has already been established.
    for (const auto operand : instruction.operands) {
      const auto source = representatives[operand.value];
      const auto &location = result.locations[source];
      if (location.storage == SurfaceValueStorageClass::inactive ||
          remaining_uses[source] == 0u) {
        return reject("an operand has no live storage at its use");
      }
    }
    for (const auto operand : instruction.operands) {
      const auto source = representatives[operand.value];
      auto &uses = remaining_uses[source];
      --uses;
      const auto &location = result.locations[source];
      if (uses == 0u &&
          location.storage == SurfaceValueStorageClass::local_slot) {
        free_slots[bank_index(location.bank)].emplace_back(location.index);
      }
    }

    auto result_bank = SurfaceValueBank::scalar;
    if (!classify_surface_value_type(instruction.result_type, result_bank)) {
      return reject("an active value has no supported typed storage bank");
    }
    const auto typed_index = bank_index(result_bank);
    auto slot = std::uint32_t{};
    if (free_slots[typed_index].empty()) {
      slot = slot_counts[typed_index]++;
    } else {
      slot = free_slots[typed_index].back();
      free_slots[typed_index].pop_back();
    }
    result.locations[id.value] = {.storage =
                                      SurfaceValueStorageClass::local_slot,
                                  .bank = result_bank,
                                  .index = slot};
    if (remaining_uses[id.value] == 0u) {
      return reject("an active computed value has no consumer");
    }
  }

  // Publish the quotient without changing any source ValueExpressionId.
  // Representatives are strictly earlier for aliases, and every executable
  // representative has now received its final colored location.
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (active[index] && representatives[index] != index) {
      const auto &location = result.locations[representatives[index]];
      if (location.storage == SurfaceValueStorageClass::inactive) {
        return reject("a value alias representative has no storage");
      }
      result.locations[index] = location;
    }
  }

  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (!active[index] || representatives[index] != index) {
      continue;
    }
    const auto expected = output_uses[index];
    if (remaining_uses[index] != expected) {
      return reject("value liveness did not converge at the stream boundary");
    }
  }

  result.scalar_slots = slot_counts[0u];
  result.vector_slots = slot_counts[1u];
  result.unsigned_integer_slots = slot_counts[2u];
  result.valid = true;
  return result;
}

SurfaceValueProgramImage
lower_surface_value_program(const SurfaceProgram &program,
                            const SurfaceValueStoragePlan &storage) {
  static_assert(std::is_trivially_copyable_v<SurfaceValueBytecodeInstruction>);
  static_assert(std::is_trivially_copyable_v<SurfaceValueBytecodeMetadata>);
  static_assert(sizeof(SurfaceValueBytecodeInstruction) == 16u);
  static_assert(sizeof(SurfaceValueBytecodeMetadata) == 40u);
  static_assert(static_cast<std::uint32_t>(ValueOperation::ambient_occlusion) <=
                surface_value_opcode_mask);
  if (!storage.compatible(program)) {
    return reject_image("cannot lower an incompatible value storage plan");
  }

  const auto &values = program.value_instructions();
  SurfaceValueProgramImage result;
  result.instructions.reserve(storage.instructions.size());
  result.value_addresses.resize(values.size(),
                                SurfaceValueAddress::invalid_value);
  result.scalar_slots = storage.scalar_slots;
  result.vector_slots = storage.vector_slots;
  result.unsigned_integer_slots = storage.unsigned_integer_slots;
  for (auto index = std::size_t{0u}; index < storage.locations.size();
       ++index) {
    if (storage.locations[index].storage ==
        SurfaceValueStorageClass::inactive) {
      continue;
    }
    SurfaceValueAddress address;
    if (!encode_location(storage.locations[index], address)) {
      return reject_image("a planned value address cannot be encoded");
    }
    result.value_addresses[index] = address.encoded();
  }
  for (const auto id : storage.instructions) {
    if (!id.valid() || id.value >= values.size()) {
      return reject_image("the storage schedule contains an invalid value");
    }
    const auto &instruction = values[id.value];
    if (instruction.operation == ValueOperation::parameter) {
      return reject_image(
          "a parameter leaked into the runtime instruction stream");
    }
    if (instruction.operands.size() !=
        value_operation_operand_count(instruction.operation)) {
      return reject_image(
          "an instruction arity disagrees with its opcode contract");
    }
    if (!surface_value_svm_static_fields_valid(instruction.operation,
                                               instruction.static_u0,
                                               instruction.static_u1)) {
      return reject_image(
          "an instruction has immutable fields outside its immediate contract");
    }
    const auto result_address =
        SurfaceValueAddress{result.value_addresses[id.value]};
    if (!result_address.valid() || result_address.parameter()) {
      return reject_image("an instruction result has no local typed address");
    }

    std::vector<SurfaceValueOperandAddress> compact_operands;
    compact_operands.reserve(instruction.operands.size());
    for (const auto operand : instruction.operands) {
      if (!operand.valid() || operand.value >= storage.locations.size()) {
        return reject_image("an instruction operand has no planned address");
      }
      const auto operand_address =
          SurfaceValueAddress{result.value_addresses[operand.value]};
      if (!operand_address.valid()) {
        return reject_image("an instruction operand address cannot be encoded");
      }
      SurfaceValueOperandAddress compact;
      if (!encode_surface_value_operand_address(operand_address, compact)) {
        return reject_image(
            "an instruction operand exceeds the compact 13-bit address "
            "domain");
      }
      compact_operands.emplace_back(compact);
    }

    auto operand_payload = surface_value_invalid_operand_word;
    const auto pack_word = [&](std::size_t begin) noexcept {
      auto word = surface_value_invalid_operand_word;
      for (auto lane = std::size_t{0u};
           lane < surface_value_operands_per_word &&
           begin + lane < compact_operands.size();
           ++lane) {
        word = pack_operand_lane(word, compact_operands[begin + lane], lane);
      }
      return word;
    };
    if (compact_operands.size() <= surface_value_inline_operand_capacity) {
      operand_payload = pack_word(0u);
    } else {
      const auto word_count =
          surface_value_operand_word_count(compact_operands.size());
      if (result.operands.size() > std::numeric_limits<std::uint32_t>::max() ||
          word_count > std::numeric_limits<std::uint32_t>::max() -
                           result.operands.size()) {
        return reject_image(
            "the packed operand stream exceeds the device encoding");
      }
      operand_payload = static_cast<std::uint32_t>(result.operands.size());
      for (auto begin = std::size_t{0u}; begin < compact_operands.size();
           begin += surface_value_operands_per_word) {
        result.operands.emplace_back(pack_word(begin));
      }
    }

    auto metadata_index = ~std::uint32_t{0u};
    const auto has_metadata =
        instruction.static_u0 != 0u || instruction.static_u1 != 0u ||
        std::bit_cast<std::uint32_t>(instruction.static_f0) != 0u ||
        std::bit_cast<std::uint32_t>(instruction.static_f1) != 0u ||
        instruction.parameter.valid() || !instruction.static_table.empty();
    if (has_metadata) {
      if (result.metadata.size() >= std::numeric_limits<std::uint32_t>::max() ||
          result.static_data.size() >
              std::numeric_limits<std::uint32_t>::max() ||
          instruction.static_table.size() >
              std::numeric_limits<std::uint32_t>::max() -
                  result.static_data.size()) {
        return reject_image("value metadata exceeds the device encoding");
      }
      metadata_index = static_cast<std::uint32_t>(result.metadata.size());
      result.metadata.emplace_back(SurfaceValueBytecodeMetadata{
          .static_u0 = instruction.static_u0,
          .static_u1 = instruction.static_u1,
          .static_f0 = instruction.static_f0,
          .static_f1 = instruction.static_f1,
          .parameter = instruction.parameter.valid()
                           ? instruction.parameter.value
                           : ~std::uint32_t{0u},
          .static_table_begin =
              static_cast<std::uint32_t>(result.static_data.size()),
          .static_table_count =
              static_cast<std::uint32_t>(instruction.static_table.size())});
      result.static_data.insert(result.static_data.end(),
                                instruction.static_table.begin(),
                                instruction.static_table.end());
    }
    result.instructions.emplace_back(SurfaceValueBytecodeInstruction{
        .control = make_surface_value_control(
            instruction.operation, result_address.bank(),
            make_surface_value_svm_immediate(instruction.operation,
                                             instruction.static_u0,
                                             instruction.static_u1)),
        .result = result_address.encoded(),
        .operand_payload = operand_payload,
        .metadata_index = metadata_index});
  }
  result.valid = true;
  return result;
}

SurfaceValueSceneImage build_surface_value_scene_image(
    std::span<const SurfaceValueProgramImage> programs) {
  static_assert(std::is_trivially_copyable_v<SurfaceValueProgramDescriptor>);
  static_assert(sizeof(SurfaceValueProgramDescriptor) == 32u);

  auto instruction_count = std::size_t{0u};
  auto operand_count = std::size_t{0u};
  auto metadata_count = std::size_t{0u};
  auto static_data_count = std::size_t{0u};
  for (auto program_index = std::size_t{0u}; program_index < programs.size();
       ++program_index) {
    const auto diagnostic =
        validate_surface_value_program_image(programs[program_index]);
    if (!diagnostic.empty()) {
      return reject_scene_image(
          "value program " + std::to_string(program_index) + ": " + diagnostic);
    }
    if (!add_scene_extent(instruction_count,
                          programs[program_index].instructions.size()) ||
        !add_scene_extent(operand_count,
                          programs[program_index].operands.size()) ||
        !add_scene_extent(metadata_count,
                          programs[program_index].metadata.size()) ||
        !add_scene_extent(static_data_count,
                          programs[program_index].static_data.size())) {
      return reject_scene_image(
          "the aggregate value program exceeds 32-bit device offsets");
    }
  }

  SurfaceValueSceneImage result;
  result.programs.reserve(programs.size());
  result.instructions.reserve(instruction_count);
  result.operands.reserve(operand_count);
  result.metadata.reserve(metadata_count);
  result.static_data.reserve(static_data_count);
  for (const auto &program : programs) {
    const auto instruction_begin =
        static_cast<std::uint32_t>(result.instructions.size());
    const auto operand_word_begin =
        static_cast<std::uint32_t>(result.operands.size());
    const auto metadata_begin =
        static_cast<std::uint32_t>(result.metadata.size());
    const auto static_data_begin =
        static_cast<std::uint32_t>(result.static_data.size());
    result.programs.emplace_back(SurfaceValueProgramDescriptor{
        .instruction_begin = instruction_begin,
        .instruction_count =
            static_cast<std::uint32_t>(program.instructions.size()),
        .scalar_slots = program.scalar_slots,
        .vector_slots = program.vector_slots,
        .unsigned_integer_slots = program.unsigned_integer_slots,
        .flags = program.flags});
    for (auto instruction : program.instructions) {
      if (!is_surface_value_surface_normal_transition(instruction) &&
          surface_value_operand_count(instruction) >
              surface_value_inline_operand_capacity) {
        instruction.operand_payload += operand_word_begin;
      }
      if (instruction.metadata_index != SurfaceValueAddress::invalid_value) {
        instruction.metadata_index += metadata_begin;
      }
      result.instructions.emplace_back(instruction);
    }
    result.operands.insert(result.operands.end(), program.operands.begin(),
                           program.operands.end());
    for (auto metadata : program.metadata) {
      metadata.static_table_begin += static_data_begin;
      result.metadata.emplace_back(metadata);
    }
    result.static_data.insert(result.static_data.end(),
                              program.static_data.begin(),
                              program.static_data.end());
  }
  result.valid = true;
  return result;
}

SurfaceValueExecutableScene build_surface_value_executable_scene(
    std::span<const SurfaceValueExecutionInput> inputs) {
  std::vector<SurfaceValueProgramImage> program_images;
  program_images.reserve(inputs.size());
  std::vector<SurfaceClosureProgramImage> closure_images;
  closure_images.reserve(inputs.size());
  SurfaceValueExecutableScene result;
  detail::SurfaceValueVariantInterner variant_interner;
  for (auto input_index = std::size_t{0u}; input_index < inputs.size();
       ++input_index) {
    const auto &input = inputs[input_index];
    const auto *program = input.program;
    const auto *storage = input.storage;
    if (program == nullptr || storage == nullptr ||
        !storage->compatible(*program)) {
      return reject_executable_scene(
          "value program " + std::to_string(input_index) +
          ": execution input is incomplete or incompatible");
    }
    const auto has_surface_normal = input.surface_normal_storage != nullptr;
    if (has_surface_normal != input.surface_normal_output.valid() ||
        (has_surface_normal &&
         !input.surface_normal_storage->compatible(*program)) ||
        (!has_surface_normal &&
         input.surface_normal_uses_undisplaced_geometry)) {
      return reject_executable_scene(
          "value program " + std::to_string(input_index) +
          ": automatic-normal transaction is incomplete or incompatible");
    }
    auto image = lower_surface_value_program(*program, *storage);
    if (!image.valid) {
      return reject_executable_scene("value program " +
                                     std::to_string(input_index) + ": " +
                                     image.diagnostic);
    }
    std::optional<SurfaceValueProgramImage> surface_normal_image;
    auto surface_normal_output = SurfaceValueAddress::invalid_value;
    if (has_surface_normal) {
      surface_normal_image.emplace(
          lower_surface_value_program(*program, *input.surface_normal_storage));
      if (!surface_normal_image->valid ||
          input.surface_normal_output.value >=
              surface_normal_image->value_addresses.size()) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": automatic-normal prefix cannot be lowered");
      }
      surface_normal_output =
          surface_normal_image
              ->value_addresses[input.surface_normal_output.value];
      if (surface_normal_output == SurfaceValueAddress::invalid_value) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": automatic-normal output has no typed address");
      }
    }
    SurfaceClosureProgramImage closure_image;
    closure_image.valid = true;
    if (input.closure_plan != nullptr) {
      if (!input.closure_plan->compatible(*program)) {
        return reject_executable_scene("value program " +
                                       std::to_string(input_index) +
                                       ": closure plan is incompatible");
      }
      const auto dependencies =
          analyze_surface_value_dependencies(*program, *input.closure_plan);
      closure_image = lower_surface_closure_program(
          *program, *input.closure_plan, dependencies, image.value_addresses,
          input.closure_endpoints);
      if (!closure_image.valid) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": closure lowering: " + closure_image.diagnostic);
      }
    }
    closure_images.emplace_back(std::move(closure_image));
    if (surface_normal_image) {
      image = make_surface_normal_transaction_image(
          *surface_normal_image, image, surface_normal_output,
          input.surface_normal_uses_undisplaced_geometry);
      if (!image.valid) {
        return reject_executable_scene("value program " +
                                       std::to_string(input_index) + ": " +
                                       image.diagnostic);
      }
    }
    program_images.emplace_back(std::move(image));
    const std::array<const SurfaceValueStoragePlan *, 2u> schedules{
        input.surface_normal_storage, storage};
    for (auto schedule_index = std::size_t{0u};
         schedule_index < schedules.size(); ++schedule_index) {
      const auto *schedule = schedules[schedule_index];
      if (schedule == nullptr) {
        continue;
      }
      for (const auto id : schedule->instructions) {
        if (!id.valid() || id.value >= program->value_instructions().size()) {
          return reject_executable_scene(
              "value program " + std::to_string(input_index) +
              ": storage schedule contains an invalid instruction");
        }
        const auto &instruction = program->value_instructions()[id.value];
        auto variant = SurfaceValueAddress::invalid_value;
        std::string interning_diagnostic;
        if (!variant_interner.intern(*program, instruction, variant,
                                     interning_diagnostic)) {
          return reject_executable_scene(
              "value program " + std::to_string(input_index) +
              ": evaluator interning: " + interning_diagnostic);
        }
        result.instruction_variants.emplace_back(variant);
      }
      if (schedule_index == 0u) {
        result.instruction_variants.emplace_back(
            SurfaceValueAddress::invalid_value);
      }
    }
  }
  result.variants = variant_interner.finish();
  result.values =
      build_surface_execution_scene_image(program_images, closure_images);
  if (!result.values.valid) {
    return reject_executable_scene(result.values.diagnostic);
  }
  if (auto diagnostic = detail::populate_surface_value_operand_routes(
          result.values, result.instruction_variants, result.variants);
      !diagnostic.empty()) {
    return reject_executable_scene(std::move(diagnostic));
  }
  result.valid = true;
  return result;
}

} // namespace psycles::compiler
