#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace psycles::sampling {

enum class LightDistributionEmitterKind : std::uint32_t {
    emissive_triangle,
    analytic_light,
    environment,
    sentinel
};

struct LightDistributionEntry {
    float cumulative{};
    float selection_pdf{};
    LightDistributionEmitterKind kind{
        LightDistributionEmitterKind::sentinel};
    std::uint32_t index{};
};

struct CyclesLightDistribution {
    // Cycles stores one leading-CDF record per emitter plus a sentinel.
    // The sentinel's cumulative value is the total (one for a usable
    // normalized distribution).
    std::vector<LightDistributionEntry> entries;
    std::uint32_t emitter_count{};
    float triangle_area_pdf{};
    float light_selection_pdf{};

    [[nodiscard]] bool usable() const noexcept;
};

// Build Cycles 4.5's flat light distribution (light tree disabled).
// Emissive triangles are ordered first and weighted by world-space area.
// Analytic lights follow in index order, with the optional environment
// treated as the last lamp. When both emitter classes are present, triangles
// and lamps receive half of the total probability each.
[[nodiscard]] CyclesLightDistribution
build_cycles_light_distribution(
    std::span<const float> emissive_triangle_areas,
    std::uint32_t analytic_light_count,
    bool include_environment);

}// namespace psycles::sampling
