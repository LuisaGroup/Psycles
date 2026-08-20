#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
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
  std::uint32_t operation{};
  std::uint32_t result{};
  std::uint32_t operand_begin{};
  std::uint32_t metadata_index{~std::uint32_t{0u}};
};

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

} // namespace psycles::compiler
