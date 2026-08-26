#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler {

// Result of converting recursive Bump height evaluation into one pure SSA
// value graph. `root_values[i]` is the image of source ValueExpressionId{i}
// in the unshifted domain; every public SurfaceProgram reference is remapped
// through this bijection onto the expanded graph.
struct SurfaceBumpExpansion {
  bool valid{};
  std::string diagnostic;
  std::shared_ptr<const SurfaceProgram> program;
  std::vector<ValueExpressionId> root_values;
  std::uint32_t bump_count{};
  std::uint32_t sampled_instruction_count{};
};

// Cycles-style Bump graph refinement for the compact SVM backend.
//
// For a source Bump B in differential context C, the pass constructs:
//
//   hc = eval(height(B), C)
//   w  = max(eval(filter_width(B), C), 0)
//   hx = eval(height(B), C + (w, 0))
//   hy = eval(height(B), C + (0, w))
//   B' = bump_samples(hc, hx, hy, ...)
//
// A context is a pair of scalar SSA coefficients applied to every
// point-dependent leaf. Thus nested contexts compose algebraically, all
// dependencies remain explicit, and an ordinary topological scheduler may
// reorder only operations whose values are provably independent. Exact GVN
// merges duplicated invariant nodes after comparing the complete semantic
// key; hashes are never accepted as equality proofs.
[[nodiscard]] SurfaceBumpExpansion
expand_surface_bump_program(const SurfaceProgram &program);

} // namespace psycles::compiler
