#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler {

// Physical storage for a strongly typed value. Parameters remain references
// into the material block; only computed values occupy interpreter-local
// slots. No device-side type tag is required.
enum class SurfaceValueStorageClass : std::uint8_t {
  inactive,
  parameter,
  local_slot
};

enum class SurfaceValueBank : std::uint8_t { scalar, vector, unsigned_integer };

struct SurfaceValueLocation {
  SurfaceValueStorageClass storage{SurfaceValueStorageClass::inactive};
  SurfaceValueBank bank{SurfaceValueBank::scalar};
  std::uint32_t index{};
};

// A deterministic allocation for one topologically closed value domain.
// `instructions` excludes parameters, which are direct material-data reads.
//
// The allocation contract is read-before-write: an evaluator must load all
// operands of an instruction before writing its result. Under that contract,
// the result may reuse a same-bank operand slot whose final use is the current
// instruction. Since every value has one interval in the fixed topological
// order, greedy expiration colors this interval graph with exactly its peak
// number of simultaneously live values.
struct SurfaceValueStoragePlan {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueLocation> locations;
  std::vector<ValueExpressionId> instructions;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t active_values{};
  std::uint32_t parameter_values{};

  [[nodiscard]] bool compatible(const SurfaceProgram &program) const noexcept;

  // Logical SoA payload. A backend may impose stronger alignment, but this
  // metric intentionally excludes type tags and float4 padding.
  [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

// Compact device-program address. The high bit selects the immutable material
// parameter block; computed values use a typed local bank. The next two bits
// encode the bank and the low 29 bits encode its index. 0xffffffff remains
// invalid so accidental sentinel use cannot alias valid storage.
class SurfaceValueAddress {

private:
  std::uint32_t _value{invalid_value};

public:
  static constexpr std::uint32_t invalid_value = ~std::uint32_t{0u};
  static constexpr std::uint32_t parameter_bit = 1u << 31u;
  static constexpr std::uint32_t bank_shift = 29u;
  static constexpr std::uint32_t bank_mask = 0x3u << bank_shift;
  static constexpr std::uint32_t index_mask = (1u << bank_shift) - 1u;

  SurfaceValueAddress() noexcept = default;
  explicit constexpr SurfaceValueAddress(std::uint32_t value) noexcept
      : _value{value} {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return _value != invalid_value;
  }
  [[nodiscard]] constexpr bool parameter() const noexcept {
    return (_value & parameter_bit) != 0u;
  }
  [[nodiscard]] constexpr SurfaceValueBank bank() const noexcept {
    return static_cast<SurfaceValueBank>((_value & bank_mask) >> bank_shift);
  }
  [[nodiscard]] constexpr std::uint32_t index() const noexcept {
    return _value & index_mask;
  }
  [[nodiscard]] constexpr std::uint32_t encoded() const noexcept {
    return _value;
  }

  auto operator<=>(const SurfaceValueAddress &) const noexcept = default;
};

// The hot stream is deliberately 16 bytes. Operand arity is an opcode
// invariant, and uncommon immutable fields live in a side table so ordinary
// arithmetic does not fetch two uint64 values and a static-table descriptor.
struct SurfaceValueBytecodeInstruction {
  // Packed as [reserved:14 | result bank:2 | operand count:8 | opcode:8].
  // ValueOperation is a closed uint8_t enum and every operation has a fixed
  // arity. Keeping those facts in the stream makes a serialized image
  // independently verifiable without enlarging the hot 16-byte record.
  std::uint32_t control{};
  std::uint32_t result{};
  std::uint32_t operand_begin{};
  std::uint32_t metadata_index{~std::uint32_t{0u}};
};

inline constexpr std::uint32_t surface_value_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_value_operand_count_shift = 8u;
inline constexpr std::uint32_t surface_value_operand_count_mask =
    0xffu << surface_value_operand_count_shift;
inline constexpr std::uint32_t surface_value_result_bank_shift = 16u;
inline constexpr std::uint32_t surface_value_result_bank_mask =
    0x3u << surface_value_result_bank_shift;
inline constexpr std::uint32_t surface_value_control_mask =
    surface_value_opcode_mask | surface_value_operand_count_mask |
    surface_value_result_bank_mask;

[[nodiscard]] constexpr std::uint32_t make_surface_value_control(
    ValueOperation operation, std::uint8_t operand_count,
    SurfaceValueBank result_bank) noexcept {
  return static_cast<std::uint32_t>(operation) |
         (static_cast<std::uint32_t>(operand_count)
          << surface_value_operand_count_shift) |
         (static_cast<std::uint32_t>(result_bank)
          << surface_value_result_bank_shift);
}

[[nodiscard]] constexpr ValueOperation surface_value_operation(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return static_cast<ValueOperation>(instruction.control &
                                     surface_value_opcode_mask);
}

[[nodiscard]] constexpr std::uint32_t surface_value_operand_count(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return (instruction.control & surface_value_operand_count_mask) >>
         surface_value_operand_count_shift;
}

[[nodiscard]] constexpr SurfaceValueBank surface_value_result_bank(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return static_cast<SurfaceValueBank>(
      (instruction.control & surface_value_result_bank_mask) >>
      surface_value_result_bank_shift);
}

// Ordered largest-to-smallest alignment. A metadata record exists only when
// at least one field differs from its canonical zero/invalid default.
struct SurfaceValueBytecodeMetadata {
  std::uint64_t static_u0{};
  std::uint64_t static_u1{};
  float static_f0{};
  float static_f1{};
  std::uint32_t parameter{~std::uint32_t{0u}};
  std::uint32_t static_table_begin{};
  std::uint32_t static_table_count{};
  std::uint32_t reserved{};
};

struct SurfaceValueProgramImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueBytecodeInstruction> instructions;
  std::vector<std::uint32_t> operands;
  std::vector<SurfaceValueBytecodeMetadata> metadata;
  std::vector<float> static_data;
  // Host-side bridge used while closure records are lowered. Inactive values
  // retain SurfaceValueAddress::invalid_value and are never device operands.
  std::vector<std::uint32_t> value_addresses;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
};

// One entry per runtime surface tag. All offsets in the aggregate streams are
// absolute, while typed local addresses remain relative to an invocation's
// small scalar/vector/uint64 banks. Eight uint32 fields keep the descriptor a
// naturally aligned 32-byte device record and leave named extension fields for
// the subsequent closure-program lowering.
struct SurfaceValueProgramDescriptor {
  std::uint32_t instruction_begin{};
  std::uint32_t instruction_count{};
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t closure_begin{};
  std::uint32_t closure_count{};
  std::uint32_t flags{};
};

// Scene-wide immutable image consumed by one shared device evaluator. The
// builder is transactional: any malformed source program rejects the complete
// image, so a runtime tag can never observe a partially relocated stream.
struct SurfaceValueSceneImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueProgramDescriptor> programs;
  std::vector<SurfaceValueBytecodeInstruction> instructions;
  std::vector<std::uint32_t> operands;
  std::vector<SurfaceValueBytecodeMetadata> metadata;
  std::vector<float> static_data;
};

// One AST body per exact immutable instruction configuration. Operands are
// renumbered to [0, arity), and their original socket types are retained so a
// shared evaluator can load typed addresses before invoking the existing
// Luisa node implementation. Source-node identity is deliberately absent: it
// is provenance, not executable semantics.
struct SurfaceValueStaticVariant {
  ValueInstruction instruction;
  std::vector<contract::SocketType> operand_types;
};

struct SurfaceValueExecutionInput {
  const SurfaceProgram *program{};
  const SurfaceValueStoragePlan *storage{};
};

// `instruction_variants` is parallel to `values.instructions`. Interning is
// exact and bit-preserving (including NaN payloads and signed zero); it never
// relies on a collision-prone hash equivalence.
struct SurfaceValueExecutableScene {
  bool valid{};
  std::string diagnostic;
  SurfaceValueSceneImage values;
  std::vector<SurfaceValueStaticVariant> variants;
  std::vector<std::uint32_t> instruction_variants;
};

// `active` must be transitively closed over ValueInstruction operands.
// `outputs` names values consumed after the stream (normally closure roots),
// and must be a subset of `active`.
[[nodiscard]] SurfaceValueStoragePlan
plan_surface_value_storage(const SurfaceProgram &program,
                           const std::vector<bool> &active,
                           const std::vector<bool> &outputs);

// Lowers the proven storage plan without changing graph semantics. Every
// original closure/output remains a typed address into this image; parameters
// remain late-bound material data and are never baked into the stream.
[[nodiscard]] SurfaceValueProgramImage lower_surface_value_program(
    const SurfaceProgram &program,
    const SurfaceValueStoragePlan &storage);

// Concatenates topology programs in runtime-tag order and rebases every
// operand, metadata, and static-table reference to the scene-wide streams.
// Local typed addresses and material ParameterId references are intentionally
// unchanged.
[[nodiscard]] SurfaceValueSceneImage build_surface_value_scene_image(
    std::span<const SurfaceValueProgramImage> programs);

// Builds the aggregate scene image and interns immutable instruction
// configurations for a scene-pruned shared Luisa evaluator. Inputs and output
// descriptors remain in runtime surface-tag order.
[[nodiscard]] SurfaceValueExecutableScene
build_surface_value_executable_scene(
    std::span<const SurfaceValueExecutionInput> inputs);

} // namespace psycles::compiler
