#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_svm_program.h>

namespace psycles::compiler::detail {

// Exact typed-handler equivalence classes for value evaluators. The primary
// key is an ordered AST-shape tuple, not a hash. A second map proves that the
// device-decodable handler key is injective over those tuples; an opcode ABI
// collision is rejected instead of repaired with a material-variant switch.
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
  std::map<std::uint32_t, std::uint32_t> _handler_indices;
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
