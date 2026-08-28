#include "path_tracer_surface_route_policy.h"

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

} // namespace psycles::luisa_backend::detail
