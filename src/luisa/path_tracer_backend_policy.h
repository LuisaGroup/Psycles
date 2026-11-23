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

// Wavefront schedulers decompose each sample batch into many short per-stage
// kernel dispatches bounded by `wavefront_frame_capacity`, naturally making
// forward progress between GPU submissions. Applying the megakernel
// pixel-sample watchdog clamp to Vulkan or Metal wavefront slices a full-HD
// frame into thousands of tiny row-band passes with host synchronization barriers.
[[nodiscard]] constexpr std::uint32_t
backend_max_pixel_samples_per_dispatch(
    std::string_view backend,
    bool wavefront_scheduler) noexcept {
    if ((backend == "vk" || backend == "metal") && wavefront_scheduler) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    return backend_max_pixel_samples_per_dispatch(backend);
}

}// namespace psycles::luisa_backend::detail
