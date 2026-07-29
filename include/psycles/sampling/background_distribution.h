#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <psycles/core/math.h>

namespace psycles::sampling {

// Cycles stores every background CDF entry as (piecewise-constant function
// value, normalized CDF). The sentinel function value stores the unnormalized
// integral, which is needed to recover a solid-angle PDF on the device.
struct BackgroundCdfEntry {
    float function{};
    float cumulative{};
};

struct CyclesBackgroundMapDistribution {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<BackgroundCdfEntry> conditional;
    std::vector<BackgroundCdfEntry> marginal;

    [[nodiscard]] bool valid() const noexcept;
};

// Build the same piecewise-constant latitude-longitude distribution as
// Cycles' background_cdf(). `radiance` is evaluated at pixel centers in
// Cycles equirectangular row order (south pole to north pole).
[[nodiscard]] CyclesBackgroundMapDistribution
build_cycles_background_map_distribution(std::span<const Vec3f> radiance,
                                         std::uint32_t width,
                                         std::uint32_t height);

} // namespace psycles::sampling
