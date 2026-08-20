#pragma once

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

// `active` must be transitively closed over ValueInstruction operands.
// `outputs` names values consumed after the stream (normally closure roots),
// and must be a subset of `active`.
[[nodiscard]] SurfaceValueStoragePlan
plan_surface_value_storage(const SurfaceProgram &program,
                           const std::vector<bool> &active,
                           const std::vector<bool> &outputs);

} // namespace psycles::compiler
