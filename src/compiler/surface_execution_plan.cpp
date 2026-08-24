#include <psycles/compiler/surface_execution_plan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

[[nodiscard]] bool classify_value(contract::SocketType type,
                                  SurfaceValueBank &bank) noexcept {
  using contract::SocketType;
  switch (type) {
  case SocketType::boolean:
  case SocketType::integer:
  case SocketType::floating:
    bank = SurfaceValueBank::scalar;
    return true;
  case SocketType::float2:
  case SocketType::float3:
  case SocketType::color:
  case SocketType::spectrum:
  case SocketType::point:
  case SocketType::vector:
  case SocketType::normal:
    bank = SurfaceValueBank::vector;
    return true;
  case SocketType::unsigned_integer:
    bank = SurfaceValueBank::unsigned_integer;
    return true;
  case SocketType::transform:
  case SocketType::string:
  case SocketType::closure:
  case SocketType::volume_closure:
    return false;
  }
  return false;
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

[[nodiscard]] SurfaceValueSceneImage reject_scene_image(
    std::string diagnostic) {
  SurfaceValueSceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceValueExecutableScene reject_executable_scene(
    std::string diagnostic) {
  SurfaceValueExecutableScene result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] SurfaceValueBumpExecutableScene reject_bump_scene(
    std::string diagnostic) {
  SurfaceValueBumpExecutableScene result;
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
  if (bank > static_cast<std::uint32_t>(
                 SurfaceValueBank::unsigned_integer)) {
    return false;
  }
  address = SurfaceValueAddress{
      (location.storage == SurfaceValueStorageClass::parameter
           ? SurfaceValueAddress::parameter_bit
           : 0u) |
      (bank << SurfaceValueAddress::bank_shift) | location.index};
  return address.valid();
}

[[nodiscard]] bool address_fits_program(
    SurfaceValueAddress address, const SurfaceValueProgramImage &program,
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

[[nodiscard]] std::string validate_surface_value_program_image(
    const SurfaceValueProgramImage &program) {
  if (!program.valid) {
    return "source value program is invalid: " + program.diagnostic;
  }
  if (program.scalar_slots > SurfaceValueAddress::index_mask + 1u ||
      program.vector_slots > SurfaceValueAddress::index_mask + 1u ||
      program.unsigned_integer_slots >
          SurfaceValueAddress::index_mask + 1u) {
    return "a typed local bank exceeds the address encoding";
  }
  for (const auto encoded : program.value_addresses) {
    if (!address_fits_program(SurfaceValueAddress{encoded}, program, true)) {
      return "a value address exceeds its typed local bank";
    }
  }

  auto operand_cursor = std::size_t{0u};
  for (const auto &instruction : program.instructions) {
    if ((instruction.control & ~surface_value_control_mask) != 0u ||
        static_cast<std::uint32_t>(surface_value_operation(instruction)) >
            static_cast<std::uint32_t>(ValueOperation::nishita_sky)) {
      return "an instruction has an invalid control word";
    }
    const auto operation = surface_value_operation(instruction);
    if (surface_value_noise_normalize(instruction) &&
        !surface_value_operation_uses_noise_normalize(operation)) {
      return "an instruction assigns Noise Normalize data to a non-Noise "
             "opcode";
    }
    if (surface_value_operand_count(instruction) !=
        value_operation_operand_count(
            surface_value_operation(instruction))) {
      return "an instruction arity disagrees with its opcode contract";
    }
    if (instruction.operand_begin != operand_cursor) {
      return "the operand stream is not densely ordered";
    }
    const auto operand_count =
        static_cast<std::size_t>(surface_value_operand_count(instruction));
    if (operand_count > program.operands.size() - operand_cursor) {
      return "an instruction operand range exceeds the stream";
    }
    const auto result = SurfaceValueAddress{instruction.result};
    if (!address_fits_program(result, program, false) ||
        result.parameter() ||
        result.bank() != surface_value_result_bank(instruction)) {
      return "an instruction result is inconsistent with its typed bank";
    }
    for (auto operand_index = std::size_t{0u};
         operand_index < operand_count; ++operand_index) {
      if (!address_fits_program(
              SurfaceValueAddress{
                  program.operands[operand_cursor + operand_index]},
              program, false)) {
        return "an instruction operand exceeds its typed bank";
      }
    }
    if (instruction.metadata_index !=
            SurfaceValueAddress::invalid_value &&
        instruction.metadata_index >= program.metadata.size()) {
      return "an instruction metadata index exceeds the side table";
    }
    if (surface_value_operation_uses_noise_normalize(operation)) {
      const auto metadata_normalize =
          instruction.metadata_index != SurfaceValueAddress::invalid_value &&
          (program.metadata[instruction.metadata_index].static_u1 & 1u) != 0u;
      if (surface_value_noise_normalize(instruction) != metadata_normalize) {
        return "a Noise Normalize control bit disagrees with immutable "
               "metadata";
      }
    }
    operand_cursor += operand_count;
  }
  if (operand_cursor != program.operands.size()) {
    return "the operand stream has an unreferenced suffix";
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

[[nodiscard]] std::vector<std::uint64_t> make_static_variant_key(
    const SurfaceProgram &program,
    const ValueInstruction &instruction) {
  std::vector<std::uint64_t> key;
  key.reserve(9u + instruction.operands.size());
  key.emplace_back(static_cast<std::uint64_t>(instruction.operation));
  key.emplace_back(static_cast<std::uint64_t>(instruction.result_type));
  key.emplace_back(instruction.static_u0);
  // Noise Normalize selects only a terminal range mapping and is encoded in
  // each bytecode instruction. Removing exactly that owned bit from the
  // semantic AST key preserves all remaining static dimensions, fractal type,
  // output, and operand-type information.
  key.emplace_back(
      surface_value_operation_uses_noise_normalize(instruction.operation)
          ? instruction.static_u1 & ~std::uint64_t{1u}
          : instruction.static_u1);
  key.emplace_back(std::bit_cast<std::uint32_t>(instruction.static_f0));
  key.emplace_back(std::bit_cast<std::uint32_t>(instruction.static_f1));
  // Shader-table ParameterId is a late-bound address already preserved in
  // bytecode metadata. It changes which material data is read, not the Luisa
  // operation body. All other current non-parameter operations have no
  // ParameterId; retaining the field in their key catches future semantic use.
  const auto dynamic_parameter =
      instruction.operation == ValueOperation::color_ramp ||
      instruction.operation == ValueOperation::rgb_curve;
  key.emplace_back(!dynamic_parameter && instruction.parameter.valid()
                       ? instruction.parameter.value
                       : ~std::uint64_t{0u});
  key.emplace_back(instruction.operands.size());
  for (const auto operand : instruction.operands) {
    key.emplace_back(static_cast<std::uint64_t>(
        program.value_instructions()[operand.value].result_type));
  }
  // Static-table values are bytecode data, just like a Cycles SVM node's
  // constant payload. Only the shape can affect the host-recorded evaluator
  // body: the two current consumers require fixed 16- and 33-float layouts.
  // Keeping the length in the exact key proves that merged instructions use
  // the same indexing domain while allowing authored transforms/sky data to
  // share one semantic AST body.
  key.emplace_back(instruction.static_table.size());
  return key;
}

[[nodiscard]] std::vector<bool> transitive_value_mask(
    const SurfaceProgram &program, ValueExpressionId root) {
  std::vector<bool> active(program.value_instructions().size(), false);
  std::vector<ValueExpressionId> pending;
  pending.emplace_back(root);
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!id.valid() || id.value >= active.size() || active[id.value]) {
      continue;
    }
    active[id.value] = true;
    for (const auto operand : program.value_instructions()[id.value].operands) {
      pending.emplace_back(operand);
    }
  }
  return active;
}

[[nodiscard]] std::uint32_t value_stack_words(
    const ValueInstruction &instruction) noexcept {
  SurfaceValueBank bank;
  if (!classify_value(instruction.result_type, bank)) {
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

[[nodiscard]] SurfaceValueSchedulePressure measure_schedule_pressure(
    const std::vector<ValueInstruction> &values,
    const std::vector<bool> &active,
    const std::vector<bool> &outputs,
    const std::vector<ValueExpressionId> &schedule) {
  std::vector<std::uint32_t> remaining_uses(values.size(), 0u);
  auto computed_count = std::size_t{0u};
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    computed_count +=
        values[index].operation == ValueOperation::parameter ? 0u : 1u;
    for (const auto operand : values[index].operands) {
      if (remaining_uses[operand.value] ==
          std::numeric_limits<std::uint32_t>::max()) {
        return {};
      }
      ++remaining_uses[operand.value];
    }
    if (outputs[index]) {
      if (remaining_uses[index] ==
          std::numeric_limits<std::uint32_t>::max()) {
        return {};
      }
      ++remaining_uses[index];
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
        emitted[id.value] ||
        values[id.value].operation == ValueOperation::parameter) {
      return {};
    }
    const auto &instruction = values[id.value];

    // All operands are read before any dying location is made reusable.
    for (const auto operand : instruction.operands) {
      if (values[operand.value].operation != ValueOperation::parameter &&
          !emitted[operand.value]) {
        return {};
      }
    }
    for (const auto operand : instruction.operands) {
      if (remaining_uses[operand.value] == 0u) {
        return {};
      }
      --remaining_uses[operand.value];
      if (remaining_uses[operand.value] == 0u &&
          values[operand.value].operation != ValueOperation::parameter) {
        SurfaceValueBank operand_bank;
        if (!classify_value(values[operand.value].result_type, operand_bank)) {
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
    if (!classify_value(instruction.result_type, result_bank)) {
      return {};
    }
    const auto result_bank_index = bank_index(result_bank);
    if (live[result_bank_index] ==
        std::numeric_limits<std::uint32_t>::max()) {
      return {};
    }
    ++live[result_bank_index];
    result.slots[result_bank_index] =
        std::max(result.slots[result_bank_index], live[result_bank_index]);
    emitted[id.value] = true;
  }
  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (active[index] &&
        values[index].operation != ValueOperation::parameter &&
        remaining_uses[index] != (outputs[index] ? 1u : 0u)) {
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
[[nodiscard]] bool schedule_surface_value_instructions(
    const std::vector<ValueInstruction> &values,
    const std::vector<bool> &active,
    const std::vector<bool> &outputs,
    std::vector<ValueExpressionId> &schedule,
    std::string &diagnostic) {
  const auto count = values.size();
  std::vector<std::vector<std::uint32_t>> producers(count);
  std::vector<std::uint32_t> instruction_consumers(count, 0u);
  std::vector<std::uint32_t> total_consumers(count, 0u);
  auto computed_count = std::size_t{0u};

  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (!active[index] ||
        values[index].operation == ValueOperation::parameter) {
      continue;
    }
    ++computed_count;
    auto &dependencies = producers[index];
    dependencies.reserve(values[index].operands.size());
    for (const auto operand : values[index].operands) {
      if (values[operand.value].operation == ValueOperation::parameter ||
          std::find(dependencies.begin(), dependencies.end(), operand.value) !=
              dependencies.end()) {
        continue;
      }
      dependencies.emplace_back(operand.value);
      if (instruction_consumers[operand.value] ==
          std::numeric_limits<std::uint32_t>::max()) {
        diagnostic = "the value consumer count exceeds the scheduler encoding";
        return false;
      }
      ++instruction_consumers[operand.value];
    }
  }
  total_consumers = instruction_consumers;
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (outputs[index]) {
      if (total_consumers[index] ==
          std::numeric_limits<std::uint32_t>::max()) {
        diagnostic =
            "the terminal value consumer count exceeds the scheduler encoding";
        return false;
      }
      ++total_consumers[index];
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
  const auto order_before = [&](std::uint32_t lhs,
                                std::uint32_t rhs) noexcept {
    const auto lhs_key = order_key(lhs);
    const auto rhs_key = order_key(rhs);
    return lhs_key > rhs_key || (lhs_key == rhs_key && lhs < rhs);
  };

  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (!active[index] ||
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
          live_words + static_cast<std::uint64_t>(
                           current_number(dependency)));
      live_words += value_stack_words(values[dependency]);
    }
    peak_words = std::max(
        peak_words,
        live_words + value_stack_words(values[index]));
    if (peak_words > std::numeric_limits<std::uint32_t>::max()) {
      diagnostic = "the Sethi-Ullman stack estimate exceeds its encoding";
      return false;
    }
    sethi_ullman[index] = static_cast<std::uint32_t>(peak_words);
  }

  std::vector<std::uint32_t> sinks;
  sinks.reserve(computed_count);
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (active[index] &&
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

  // Cycles' scalar-stack recurrence is a heuristic for DAGs, while Psycles
  // has three separately allocated typed banks. Preserve the original legal
  // topological order as a formal no-regression candidate and select it only
  // when its exact read-before-write payload is smaller. Equal pressure keeps
  // the Cycles schedule and its locality/order characteristics.
  std::vector<ValueExpressionId> source_schedule;
  source_schedule.reserve(computed_count);
  for (auto index = std::size_t{0u}; index < count; ++index) {
    if (active[index] &&
        values[index].operation != ValueOperation::parameter) {
      source_schedule.emplace_back(
          ValueExpressionId{static_cast<std::uint32_t>(index)});
    }
  }
  const auto cycles_pressure =
      measure_schedule_pressure(values, active, outputs, schedule);
  const auto source_pressure =
      measure_schedule_pressure(values, active, outputs, source_schedule);
  if (!cycles_pressure.valid || !source_pressure.valid) {
    diagnostic = "a value schedule violates read-before-write liveness";
    return false;
  }
  if (source_pressure.payload_bytes() < cycles_pressure.payload_bytes()) {
    schedule = std::move(source_schedule);
  }
  return true;
}

} // namespace

bool SurfaceValueStoragePlan::compatible(
    const SurfaceProgram &program) const noexcept {
  return valid && locations.size() == program.value_instructions().size() &&
         static_cast<std::size_t>(active_values) >= instructions.size() &&
         static_cast<std::size_t>(active_values) ==
             instructions.size() + parameter_values;
}

std::size_t SurfaceValueStoragePlan::payload_bytes() const noexcept {
  return static_cast<std::size_t>(scalar_slots) * sizeof(float) +
         static_cast<std::size_t>(vector_slots) * sizeof(float) * 3u +
         static_cast<std::size_t>(unsigned_integer_slots) *
             sizeof(std::uint64_t);
}

SurfaceValueStoragePlan
plan_surface_value_storage(const SurfaceProgram &program,
                           const std::vector<bool> &active,
                           const std::vector<bool> &outputs) {
  const auto &values = program.value_instructions();
  if (active.size() != values.size() || outputs.size() != values.size()) {
    return reject("value storage masks do not match the program");
  }

  SurfaceValueStoragePlan result;
  result.locations.resize(values.size());
  result.instructions.reserve(values.size());
  std::vector<std::uint32_t> remaining_uses(values.size(), 0u);

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
    if (!classify_value(instruction.result_type, bank)) {
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
      if (remaining_uses[operand.value] ==
          std::numeric_limits<std::uint32_t>::max()) {
        return reject("the value use count exceeds the plan encoding");
      }
      ++remaining_uses[operand.value];
    }
    if (outputs[index]) {
      if (remaining_uses[index] ==
          std::numeric_limits<std::uint32_t>::max()) {
        return reject("the value use count exceeds the plan encoding");
      }
      ++remaining_uses[index];
    }
  }

  if (!schedule_surface_value_instructions(
          values, active, outputs, result.instructions, result.diagnostic)) {
    return result;
  }

  std::array<std::vector<std::uint32_t>, 3u> free_slots;
  std::array<std::uint32_t, 3u> slot_counts{};
  for (const auto id : result.instructions) {
    const auto &instruction = values[id.value];

    // This is the formal read phase. Expired operands become reusable only
    // after every operand location has already been established.
    for (const auto operand : instruction.operands) {
      const auto &location = result.locations[operand.value];
      if (location.storage == SurfaceValueStorageClass::inactive ||
          remaining_uses[operand.value] == 0u) {
        return reject("an operand has no live storage at its use");
      }
    }
    for (const auto operand : instruction.operands) {
      auto &uses = remaining_uses[operand.value];
      --uses;
      const auto &location = result.locations[operand.value];
      if (uses == 0u &&
          location.storage == SurfaceValueStorageClass::local_slot) {
        free_slots[bank_index(location.bank)].emplace_back(location.index);
      }
    }

    auto result_bank = SurfaceValueBank::scalar;
    if (!classify_value(instruction.result_type, result_bank)) {
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

  for (auto index = std::size_t{0u}; index < values.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    const auto expected = outputs[index] ? 1u : 0u;
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

SurfaceValueProgramImage lower_surface_value_program(
    const SurfaceProgram &program,
    const SurfaceValueStoragePlan &storage) {
  static_assert(std::is_trivially_copyable_v<SurfaceValueBytecodeInstruction>);
  static_assert(std::is_trivially_copyable_v<SurfaceValueBytecodeMetadata>);
  static_assert(sizeof(SurfaceValueBytecodeInstruction) == 16u);
  static_assert(sizeof(SurfaceValueBytecodeMetadata) == 40u);
  static_assert(static_cast<std::uint32_t>(ValueOperation::nishita_sky) <=
                surface_value_opcode_mask);
  if (!storage.compatible(program)) {
    return reject_image("cannot lower an incompatible value storage plan");
  }

  const auto &values = program.value_instructions();
  SurfaceValueProgramImage result;
  result.instructions.reserve(storage.instructions.size());
  result.value_addresses.resize(
      values.size(), SurfaceValueAddress::invalid_value);
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
    if (instruction.operands.size() >
        std::numeric_limits<std::uint8_t>::max()) {
      return reject_image("an instruction exceeds the encoded operand count");
    }
    if (result.operands.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        instruction.operands.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.operands.size()) {
      return reject_image("the operand stream exceeds the device encoding");
    }
    const auto result_address =
        SurfaceValueAddress{result.value_addresses[id.value]};
    if (!result_address.valid() || result_address.parameter()) {
      return reject_image("an instruction result has no local typed address");
    }

    const auto operand_begin =
        static_cast<std::uint32_t>(result.operands.size());
    for (const auto operand : instruction.operands) {
      if (!operand.valid() || operand.value >= storage.locations.size()) {
        return reject_image("an instruction operand has no planned address");
      }
      const auto operand_address =
          SurfaceValueAddress{result.value_addresses[operand.value]};
      if (!operand_address.valid()) {
        return reject_image("an instruction operand address cannot be encoded");
      }
      result.operands.emplace_back(operand_address.encoded());
    }

    auto metadata_index = ~std::uint32_t{0u};
    const auto has_metadata = instruction.static_u0 != 0u ||
                              instruction.static_u1 != 0u ||
                              std::bit_cast<std::uint32_t>(
                                  instruction.static_f0) != 0u ||
                              std::bit_cast<std::uint32_t>(
                                  instruction.static_f1) != 0u ||
                              instruction.parameter.valid() ||
                              !instruction.static_table.empty();
    if (has_metadata) {
      if (result.metadata.size() >=
              std::numeric_limits<std::uint32_t>::max() ||
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
          .static_table_count = static_cast<std::uint32_t>(
              instruction.static_table.size())});
      result.static_data.insert(result.static_data.end(),
                                instruction.static_table.begin(),
                                instruction.static_table.end());
    }
    result.instructions.emplace_back(SurfaceValueBytecodeInstruction{
        .control = make_surface_value_control(
            instruction.operation,
            static_cast<std::uint8_t>(instruction.operands.size()),
            result_address.bank(),
            surface_value_operation_uses_noise_normalize(
                instruction.operation) &&
                (instruction.static_u1 & 1u) != 0u),
        .result = result_address.encoded(),
        .operand_begin = operand_begin,
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
  for (auto program_index = std::size_t{0u};
       program_index < programs.size(); ++program_index) {
    const auto diagnostic =
        validate_surface_value_program_image(programs[program_index]);
    if (!diagnostic.empty()) {
      return reject_scene_image(
          "value program " + std::to_string(program_index) + ": " +
          diagnostic);
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
    const auto operand_begin =
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
        .unsigned_integer_slots = program.unsigned_integer_slots});
    for (auto instruction : program.instructions) {
      instruction.operand_begin += operand_begin;
      if (instruction.metadata_index !=
          SurfaceValueAddress::invalid_value) {
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
  std::map<std::vector<std::uint64_t>, std::uint32_t> variant_indices;
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
    auto image = lower_surface_value_program(*program, *storage);
    if (!image.valid) {
      return reject_executable_scene(
          "value program " + std::to_string(input_index) + ": " +
          image.diagnostic);
    }
    SurfaceClosureProgramImage closure_image;
    closure_image.valid = true;
    if (input.closure_plan != nullptr) {
      if (!input.closure_plan->compatible(*program)) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": closure plan is incompatible");
      }
      const auto dependencies = analyze_surface_value_dependencies(
          *program, *input.closure_plan);
      closure_image = lower_surface_closure_program(
          *program, *input.closure_plan, dependencies,
          image.value_addresses, input.closure_endpoints);
      if (!closure_image.valid) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": closure lowering: " + closure_image.diagnostic);
      }
    }
    closure_images.emplace_back(std::move(closure_image));
    program_images.emplace_back(std::move(image));
    for (const auto id : storage->instructions) {
      if (!id.valid() || id.value >= program->value_instructions().size()) {
        return reject_executable_scene(
            "value program " + std::to_string(input_index) +
            ": storage schedule contains an invalid instruction");
      }
      const auto &instruction = program->value_instructions()[id.value];
      auto key = make_static_variant_key(*program, instruction);
      auto [iter, inserted] = variant_indices.try_emplace(
          key, static_cast<std::uint32_t>(result.variants.size()));
      if (inserted) {
        if (result.variants.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
          return reject_executable_scene(
              "the scene has too many immutable value variants");
        }
        auto normalized = instruction;
        if (normalized.operation == ValueOperation::color_ramp ||
            normalized.operation == ValueOperation::rgb_curve) {
          normalized.parameter = {};
        }
        if (surface_value_operation_uses_noise_normalize(
                normalized.operation)) {
          normalized.static_u1 &= ~std::uint64_t{1u};
        }
        // Preserve the statically proven shape but erase authored payloads
        // from the host AST representative. Compact execution must obtain
        // every entry from the instruction metadata/static-data streams;
        // zeroing here makes an accidental bake observable in regressions.
        std::fill(normalized.static_table.begin(),
                  normalized.static_table.end(), 0.0f);
        std::vector<contract::SocketType> operand_types;
        operand_types.reserve(normalized.operands.size());
        for (auto operand_index = std::size_t{0u};
             operand_index < normalized.operands.size(); ++operand_index) {
          const auto source = normalized.operands[operand_index];
          operand_types.emplace_back(
              program->value_instructions()[source.value].result_type);
          normalized.operands[operand_index] = ValueExpressionId{
              static_cast<std::uint32_t>(operand_index)};
        }
        normalized.source_node = {};
        result.variants.emplace_back(SurfaceValueStaticVariant{
            .instruction = std::move(normalized),
            .operand_types = std::move(operand_types)});
      }
      result.instruction_variants.emplace_back(iter->second);
    }
  }
  result.values = build_surface_execution_scene_image(
      program_images, closure_images);
  if (!result.values.valid) {
    return reject_executable_scene(result.values.diagnostic);
  }
  if (result.instruction_variants.size() !=
      result.values.instructions.size()) {
    return reject_executable_scene(
        "the immutable-variant stream is not parallel to the instructions");
  }
  result.valid = true;
  return result;
}

SurfaceValueBumpExecutableScene build_surface_value_bump_executable_scene(
    std::span<const SurfaceValueExecutionInput> root_inputs) {
  if (root_inputs.size() >
      std::numeric_limits<std::uint32_t>::max()) {
    return reject_bump_scene("the root program count exceeds device tags");
  }

  for (const auto &input : root_inputs) {
    const auto *program = input.program;
    const auto *storage = input.storage;
    if (program == nullptr || storage == nullptr ||
        !storage->compatible(*program)) {
      return reject_bump_scene(
          "a root execution input is incomplete or incompatible");
    }
  }

  // deque keeps every storage pointer stable while recursively discovered
  // height programs extend the execution-input array.
  std::deque<SurfaceValueStoragePlan> height_storage;
  std::vector<SurfaceValueExecutionInput> expanded_inputs;
  expanded_inputs.reserve(root_inputs.size());
  expanded_inputs.insert(expanded_inputs.end(), root_inputs.begin(),
                         root_inputs.end());

  struct HeightProgramKey {
    const SurfaceProgram *program{};
    std::uint32_t height{};
  };
  struct HeightProgramKeyLess {
    [[nodiscard]] bool operator()(const HeightProgramKey &lhs,
                                  const HeightProgramKey &rhs) const noexcept {
      const auto less = std::less<const SurfaceProgram *>{};
      if (less(lhs.program, rhs.program)) {
        return true;
      }
      if (less(rhs.program, lhs.program)) {
        return false;
      }
      return lhs.height < rhs.height;
    }
  };
  std::map<HeightProgramKey, std::uint32_t, HeightProgramKeyLess>
      height_programs;

  struct PendingBump {
    std::uint32_t instruction{};
    std::uint32_t program{};
    std::uint32_t output{};
  };
  std::vector<PendingBump> pending_bumps;
  std::vector<std::uint32_t> program_outputs(
      root_inputs.size(), SurfaceValueAddress::invalid_value);
  std::vector<std::vector<std::uint32_t>> program_children(
      root_inputs.size());
  auto instruction_begin = std::size_t{0u};
  for (auto program_index = std::size_t{0u};
       program_index < expanded_inputs.size(); ++program_index) {
    // Copy the two stable pointers before appending to expanded_inputs, which
    // may reallocate its own descriptor storage.
    const auto input = expanded_inputs[program_index];
    const auto &program = *input.program;
    const auto &storage = *input.storage;
    for (auto schedule_index = std::size_t{0u};
         schedule_index < storage.instructions.size(); ++schedule_index) {
      const auto id = storage.instructions[schedule_index];
      if (!id.valid() || id.value >= program.value_instructions().size()) {
        return reject_bump_scene(
            "a value program contains an invalid storage schedule");
      }
      const auto &instruction = program.value_instructions()[id.value];
      if (instruction.operation != ValueOperation::bump) {
        continue;
      }
      const auto height = instruction.operand(value_operand::bump::height);
      if (!height.valid() || height.value >=
                                 program.value_instructions().size()) {
        return reject_bump_scene(
            "a Bump instruction has an invalid height dependency");
      }
      const auto key = HeightProgramKey{.program = &program,
                                        .height = height.value};
      auto child_program = std::uint32_t{};
      if (const auto iter = height_programs.find(key);
          iter != height_programs.end()) {
        child_program = iter->second;
      } else {
        auto active = transitive_value_mask(program, height);
        auto outputs = std::vector<bool>(active.size(), false);
        outputs[height.value] = true;
        auto subprogram_storage =
            plan_surface_value_storage(program, active, outputs);
        if (!subprogram_storage.compatible(program)) {
          return reject_bump_scene(
              "a Bump height subprogram cannot be planned: " +
              subprogram_storage.diagnostic);
        }
        auto subprogram_image =
            lower_surface_value_program(program, subprogram_storage);
        if (!subprogram_image.valid ||
            height.value >= subprogram_image.value_addresses.size() ||
            subprogram_image.value_addresses[height.value] ==
                SurfaceValueAddress::invalid_value) {
          return reject_bump_scene(
              "a Bump height subprogram has no typed output address");
        }
        if (expanded_inputs.size() >
            std::numeric_limits<std::uint32_t>::max()) {
          return reject_bump_scene(
              "Bump program references exceed device indices");
        }
        child_program =
            static_cast<std::uint32_t>(expanded_inputs.size());
        height_storage.emplace_back(std::move(subprogram_storage));
        expanded_inputs.emplace_back(SurfaceValueExecutionInput{
            .program = &program,
            .storage = &height_storage.back()});
        program_outputs.emplace_back(
            subprogram_image.value_addresses[height.value]);
        program_children.emplace_back();
        height_programs.emplace(key, child_program);
      }
      if (instruction_begin + schedule_index >
          std::numeric_limits<std::uint32_t>::max()) {
        return reject_bump_scene(
            "Bump program references exceed device indices");
      }
      pending_bumps.emplace_back(PendingBump{
          .instruction = static_cast<std::uint32_t>(
              instruction_begin + schedule_index),
          .program = child_program,
          .output = program_outputs[child_program]});
      program_children[program_index].emplace_back(child_program);
    }
    if (storage.instructions.size() >
        std::numeric_limits<std::size_t>::max() - instruction_begin) {
      return reject_bump_scene(
          "the aggregate Bump instruction stream overflows host indices");
    }
    instruction_begin += storage.instructions.size();
  }

  // The height-program graph is a DAG because every Bump height is a
  // predecessor in the source value program. Compute the longest root-to-leaf
  // chain to emit exactly that many non-recursive callable strata.
  std::vector<std::uint8_t> depth_state(expanded_inputs.size(), 0u);
  std::vector<std::uint32_t> program_depths(expanded_inputs.size(), 0u);
  std::string depth_diagnostic;
  const auto visit_depth = [&](auto &&self,
                               std::uint32_t program_index) -> bool {
    if (program_index >= program_children.size()) {
      depth_diagnostic = "a Bump height-program edge is out of range";
      return false;
    }
    if (depth_state[program_index] == 2u) {
      return true;
    }
    if (depth_state[program_index] == 1u) {
      depth_diagnostic = "the Bump height-program graph contains a cycle";
      return false;
    }
    depth_state[program_index] = 1u;
    auto depth = std::uint32_t{0u};
    for (const auto child : program_children[program_index]) {
      if (!self(self, child)) {
        return false;
      }
      if (program_depths[child] ==
          std::numeric_limits<std::uint32_t>::max()) {
        depth_diagnostic = "the Bump evaluator depth exceeds device tags";
        return false;
      }
      depth = std::max(depth, program_depths[child] + 1u);
    }
    depth_state[program_index] = 2u;
    program_depths[program_index] = depth;
    return true;
  };
  auto maximum_bump_depth = std::uint32_t{0u};
  for (auto root_index = std::size_t{0u}; root_index < root_inputs.size();
       ++root_index) {
    if (!visit_depth(visit_depth,
                     static_cast<std::uint32_t>(root_index))) {
      return reject_bump_scene(std::move(depth_diagnostic));
    }
    maximum_bump_depth =
        std::max(maximum_bump_depth, program_depths[root_index]);
  }

  auto executable = build_surface_value_executable_scene(expanded_inputs);
  if (!executable.valid) {
    return reject_bump_scene(executable.diagnostic);
  }
  SurfaceValueBumpExecutableScene result;
  result.executable = std::move(executable);
  result.root_program_count =
      static_cast<std::uint32_t>(root_inputs.size());
  result.maximum_bump_depth = maximum_bump_depth;
  result.bump_height_programs.assign(
      result.executable.values.instructions.size(),
      SurfaceValueAddress::invalid_value);
  if (program_outputs.size() != result.executable.values.programs.size()) {
    return reject_bump_scene(
        "the Bump output stream is not parallel to the value programs");
  }
  result.program_outputs = std::move(program_outputs);
  for (const auto &bump : pending_bumps) {
    if (bump.instruction >= result.bump_height_programs.size() ||
        bump.program >= result.program_outputs.size()) {
      return reject_bump_scene(
          "a relocated Bump program reference is out of range");
    }
    result.bump_height_programs[bump.instruction] = bump.program;
    if (result.program_outputs[bump.program] != bump.output) {
      return reject_bump_scene(
          "a shared Bump height program has inconsistent output addresses");
    }
  }
  result.valid = true;
  return result;
}

} // namespace psycles::compiler
