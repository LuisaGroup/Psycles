#pragma once

#include <cstdint>

namespace psycles::luisa_backend::detail {

// Host/JIT surface execution policy. The route switches default to the
// canonical Cycles-style SVM route; zero is retained solely as an explicit
// expanded-AST diagnostic oracle and never becomes a device-side branch.
[[nodiscard]] bool compact_surface_values_requested() noexcept;
[[nodiscard]] bool populate_surface_once_requested() noexcept;
[[nodiscard]] std::uint32_t
surface_value_region_handler_site_budget_requested() noexcept;
[[nodiscard]] bool surface_value_region_diagnostics_requested() noexcept;

} // namespace psycles::luisa_backend::detail
