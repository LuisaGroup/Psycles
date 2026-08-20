#include <psycles/compiler/surface_execution_plan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
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
  return {.valid = false, .diagnostic = std::move(diagnostic)};
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
    } else {
      result.instructions.emplace_back(
          ValueExpressionId{static_cast<std::uint32_t>(index)});
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
      ++remaining_uses[index];
    }
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

    SurfaceValueBank result_bank;
    static_cast<void>(classify_value(instruction.result_type, result_bank));
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

} // namespace psycles::compiler
