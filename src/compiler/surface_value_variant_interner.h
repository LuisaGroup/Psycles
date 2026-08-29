#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_svm_program.h>

namespace psycles::compiler::detail {

// Exact host provenance classes for value evaluators. The primary key is an
// ordered semantic tuple, not a hash. Several such tuples may intentionally
// inhabit one device SVM family: the family-local semantic/result-bank subtype
// is the complete discriminator, exactly as in Cycles' multi-mode and
// multi-output node handlers.
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
  // The device-visible product (family, semantic operation, result bank) must
  // select one and only one typed evaluator shape. Unlike the removed
  // family-only map, this relation permits Cycles-style multi-operation and
  // multi-output families while rejecting an actually incomplete bytecode
  // discriminator at compile time.
  std::map<std::uint64_t, std::uint32_t> _family_subtype_indices;
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
