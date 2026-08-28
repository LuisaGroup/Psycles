#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <psycles/compiler/surface_execution_plan.h>

namespace psycles::compiler {

// Structured control image for the Cycles-style surface SVM replacement.
//
// This is deliberately a semantic schedule, not device bytecode yet. Value
// and closure ids still name the immutable SurfaceProgram, while forward jump
// targets name entries in `instructions`. Keeping this layer independent of
// address coloring lets the compiler prove control placement before storage
// decisions obscure the original graph relation.
enum class SurfaceSvmScheduleInstructionKind : std::uint8_t {
  value,
  mix_closure,
  jump_if_one,
  jump_if_zero,
  closure_leaf,
  end,
};

struct SurfaceSvmScheduleInstruction {
  SurfaceSvmScheduleInstructionKind kind{};
  // ValueExpressionId for `value`; ClosureExpressionId for Mix, jumps and
  // leaves; invalid for `end`.
  std::uint32_t source{~std::uint32_t{0u}};
  // Absolute forward target for jumps; zero for every other instruction.
  std::uint32_t target{};

  auto
  operator<=>(const SurfaceSvmScheduleInstruction &) const noexcept = default;
};

inline constexpr std::uint32_t surface_svm_invalid_region = ~std::uint32_t{0u};

// Formal host plan for one endpoint projection of one surface topology.
//
// Every active computed value is assigned to exactly one structured region.
// A region is the closure-tree node whose prefix is the lowest common
// dominator of every use of that value. Dependencies are propagated upward,
// so the region of an operand always dominates the region of its user. The
// emitted stream therefore evaluates shared inputs once and leaves values
// private to a Mix branch behind the corresponding forward guard.
struct SurfaceSvmSchedulePlan {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceSvmScheduleInstruction> instructions;
  // Parallel to SurfaceProgram::value_instructions(). Parameters receive a
  // region as semantic uses are propagated, but do not emit value records.
  // Inactive values retain surface_svm_invalid_region.
  std::vector<std::uint32_t> value_regions;
  std::uint32_t region_count{};
  std::uint32_t value_instruction_count{};
  std::uint32_t closure_leaf_count{};
  std::uint32_t mix_instruction_count{};
  std::uint32_t conditional_branch_count{};
  // Computed value instructions placed strictly below at least one dynamic
  // Mix guard. These are the instructions the old dependency-union stream
  // necessarily executes but the Cycles-style stream can skip at factor 0/1.
  std::uint32_t conditional_value_instruction_count{};
};

// Build one exact endpoint projection. `dependencies` and `closure_plan` must
// have been derived from the same program. The result is deterministic and
// contains only forward control flow; malformed/cyclic source relations are
// rejected transactionally.
[[nodiscard]] SurfaceSvmSchedulePlan plan_surface_svm_schedule(
    const SurfaceProgram &program, const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    SurfaceClosureEndpointMask endpoints = all_surface_closure_endpoints);

// Exact local storage for the structured schedule. Unlike the legacy linear
// value plan, liveness includes closure and control uses and is solved on the
// forward CFG. The quotient contracts same-bank Passthrough identities before
// analysis; locations remain parallel to the immutable SurfaceProgram ids.
//
// Each bank's interference graph is required to be chordal, as follows from
// strict SSA dominance. Maximum-cardinality search supplies and verifies a
// perfect-elimination ordering, so `*_slots` are optimal for this schedule,
// rather than merely the result of a greedy source-order heuristic.
struct SurfaceSvmStoragePlan {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueLocation> locations;
  // Quotient representative for every active value. Inactive entries retain
  // the invalid sentinel; a representative always names itself or a strict
  // topological predecessor.
  std::vector<std::uint32_t> representatives;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t active_values{};
  std::uint32_t parameter_values{};
  std::uint32_t local_values{};
  std::uint32_t alias_values{};
  std::uint64_t interference_edges{};
  std::array<std::uint32_t, 3u> maximum_interference_clique{};

  [[nodiscard]] bool
  compatible(const SurfaceProgram &program,
             const SurfaceSvmSchedulePlan &schedule) const noexcept;
  [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

// Computes exact read-before-write liveness over the acyclic schedule CFG,
// proves definite assignment on every path, and optimally colors each typed
// local bank. Parameters remain direct material reads and consume no slots.
[[nodiscard]] SurfaceSvmStoragePlan
plan_surface_svm_storage(const SurfaceProgram &program,
                         const SurfaceSvmSchedulePlan &schedule,
                         SurfaceValueStorageCapacity capacity = {});

} // namespace psycles::compiler
