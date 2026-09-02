#pragma once

namespace psycles::luisa_backend::detail {

// Host/JIT surface execution policy. The route switches default to the
// canonical Cycles-style SVM route; zero is retained solely as an explicit
// expanded-AST diagnostic oracle and never becomes a device-side branch.
[[nodiscard]] bool compact_surface_values_requested() noexcept;
[[nodiscard]] bool populate_surface_once_requested() noexcept;
// Explicit migration gate for the copied Cycles 5.2.1 SVM evaluator. Unlike
// the established routes above, this remains opt-in until production scene
// validation covers every primitive and shader domain.
[[nodiscard]] bool native_cycles_svm_surface_requested() noexcept;

} // namespace psycles::luisa_backend::detail
