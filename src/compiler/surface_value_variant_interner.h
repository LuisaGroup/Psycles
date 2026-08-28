#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_svm_program.h>

namespace psycles::compiler::detail {

// Exact host/JIT equivalence classes for value evaluators. The key is an
// ordered semantic tuple, not a hash: every source field not owned by the
// bytecode immediate/metadata ABI participates bit-for-bit. Authored data that
// is read from the bytecode remains data and therefore does not multiply AST
// bodies.
class SurfaceValueVariantInterner {
public:
  [[nodiscard]] bool intern(const SurfaceProgram &program,
                            const ValueInstruction &instruction,
                            std::uint32_t &variant_index,
                            std::string &diagnostic);

  // Canonicalizes the finite immediate domain of every exact evaluator.
  // Calling this before all insertions is a contract violation.
  [[nodiscard]] std::vector<SurfaceValueStaticVariant> finish();

private:
  std::map<std::vector<std::uint64_t>, std::uint32_t> _indices;
  std::vector<SurfaceValueStaticVariant> _variants;
  bool _finished{};
};

// Computes the exact scene-wide storage-class product abstraction for each
// evaluator operand. The two overloads share one lattice implementation; the
// unified overload additionally proves that every non-value PC has the
// invalid evaluator sentinel.
[[nodiscard]] std::string populate_surface_value_operand_routes(
    const SurfaceValueSceneImage &image,
    std::span<const std::uint32_t> instruction_variants,
    std::vector<SurfaceValueStaticVariant> &variants);

[[nodiscard]] std::string populate_surface_value_operand_routes(
    const SurfaceSvmSceneImage &image,
    std::span<const std::uint32_t> instruction_variants,
    std::vector<SurfaceValueStaticVariant> &variants);

} // namespace psycles::compiler::detail
