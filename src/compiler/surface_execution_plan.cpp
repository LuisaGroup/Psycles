#include <psycles/compiler/surface_execution_plan.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  return {.valid = false, .diagnostic = std::move(diagnostic)};
}

[[nodiscard]] SurfaceValueProgramImage reject_image(std::string diagnostic) {
  return {.valid = false, .diagnostic = std::move(diagnostic)};
}

[[nodiscard]] SurfaceValueSceneImage reject_scene_image(
    std::string diagnostic) {
  return {.valid = false, .diagnostic = std::move(diagnostic)};
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
      if (remaining_uses[index] ==
          std::numeric_limits<std::uint32_t>::max()) {
        return reject("the value use count exceeds the plan encoding");
      }
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
            result_address.bank()),
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

} // namespace psycles::compiler
