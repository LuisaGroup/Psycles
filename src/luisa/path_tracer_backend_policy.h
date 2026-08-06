#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

namespace psycles::luisa_backend::detail {

// Watchdog-managed GPU backends must make forward progress between command
// submissions. The session partitions complete, contiguous row bands under
// this pixel/sample-work bound, preserving film coordinates, sample indices,
// and accumulation order exactly.
inline constexpr auto watchdog_max_pixel_samples_per_dispatch =
    std::uint32_t{131072u};

[[nodiscard]] constexpr std::uint32_t
backend_max_pixel_samples_per_dispatch(
    std::string_view backend) noexcept {
    return backend == "metal" || backend == "vk"
               ? watchdog_max_pixel_samples_per_dispatch
               : std::numeric_limits<std::uint32_t>::max();
}

}// namespace psycles::luisa_backend::detail
