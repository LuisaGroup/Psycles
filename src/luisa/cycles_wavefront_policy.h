#pragma once

#include <cstdint>

namespace psycles::luisa_backend::detail {

struct CyclesShadowCompaction {
  std::uint32_t extent;
  bool compact;
  bool upload_extent;
};

// Cycles 5.2.1 PathTraceWorkGPU::compact_shadow_paths. The caller supplies
// the sum of all live shadow stages, including pending MNEE ownership.
// These are host scheduling decisions, never a shader/compiler heuristic.
[[nodiscard]] constexpr CyclesShadowCompaction
cycles_shadow_compaction(std::uint32_t extent, std::uint32_t live) noexcept {
  if (live == 0u) {
    return {0u, false, extent != 0u};
  }
  // Keep Cycles' floating-point comparison, including representability at
  // very large host capacities. This is not device-side software arithmetic.
  if (static_cast<float>(extent) < static_cast<float>(live) * 2.0f || extent < 32u) {
    return {extent, false, false};
  }
  return {live, true, true};
}

} // namespace psycles::luisa_backend::detail
