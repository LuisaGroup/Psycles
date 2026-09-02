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

[[nodiscard]] bool
environment_route_enabled_explicitly(const char *name) noexcept {
  const auto *value = std::getenv(name);
  return value != nullptr &&
         (std::string_view{value} == "1" ||
          std::string_view{value} == "true");
}

} // namespace

bool compact_surface_values_requested() noexcept {
  return environment_route_enabled_by_default("PSYCLES_COMPACT_SURFACE_VALUES");
}

bool populate_surface_once_requested() noexcept {
  return environment_route_enabled_by_default("PSYCLES_POPULATE_SURFACE_ONCE");
}

bool native_cycles_svm_surface_requested() noexcept {
  return environment_route_enabled_explicitly(
      "PSYCLES_NATIVE_CYCLES_SVM_SURFACE");
}

} // namespace psycles::luisa_backend::detail
