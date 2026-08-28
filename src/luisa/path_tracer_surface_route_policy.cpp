#include "path_tracer_surface_route_policy.h"

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool
environment_route_enabled_by_default(const char *name) noexcept {
  const auto *value = std::getenv(name);
  return value == nullptr ||
         (std::string_view{value} != "0" && std::string_view{value} != "false");
}

} // namespace

bool compact_surface_values_requested() noexcept {
  return environment_route_enabled_by_default("PSYCLES_COMPACT_SURFACE_VALUES");
}

bool populate_surface_once_requested() noexcept {
  return environment_route_enabled_by_default("PSYCLES_POPULATE_SURFACE_ONCE");
}

bool surface_value_region_diagnostics_requested() noexcept {
  const auto *value =
      std::getenv("PSYCLES_LOG_SURFACE_VALUE_REGION_SPECIALIZATIONS");
  return value != nullptr && std::string_view{value} != "0" &&
         std::string_view{value} != "false";
}

std::uint32_t surface_value_region_handler_site_budget_requested() noexcept {
  constexpr auto name = "PSYCLES_SURFACE_VALUE_REGION_HANDLER_SITE_BUDGET";
  const auto *value = std::getenv(name);
  if (value == nullptr) {
    // The hybrid lowering remains opt-in until object size, private memory,
    // and real HIP render time have all passed the validation gate.
    return 0u;
  }
  const auto text = std::string_view{value};
  std::uint32_t result{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                      result);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()
             ? result
             : 0u;
}

} // namespace psycles::luisa_backend::detail
