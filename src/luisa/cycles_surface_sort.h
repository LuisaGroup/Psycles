#pragma once

#include <algorithm>
#include <cstdint>

namespace psycles::luisa_backend::detail {

// Cycles 5.2.1 DeviceQueue::num_sort_partitions(), followed by
// PathTraceWorkGPU::alloc_integrator_sorting(). HIP inherits this policy.
// The frame-index partition is the outer key; shader identity is the inner
// key. A single partition is still represented by its exact divisor.
[[nodiscard]] constexpr std::uint32_t
cycles_surface_sort_partition_size(std::uint32_t capacity,
                                   std::uint32_t shader_count) noexcept {
  const auto partitions = shader_count < 300u ? std::max(capacity / 65536u, 1u) : 1u;
  return capacity / partitions + (capacity % partitions != 0u);
}

} // namespace psycles::luisa_backend::detail
