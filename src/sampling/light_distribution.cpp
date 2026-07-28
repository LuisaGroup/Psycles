#include <psycles/sampling/light_distribution.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace psycles::sampling {

namespace {

[[nodiscard]] float valid_area(float area) noexcept {
    return std::isfinite(area) && area > 0.0f
               ? area
               : 0.0f;
}

}// namespace

bool CyclesLightDistribution::usable() const noexcept {
    return emitter_count > 0u &&
           entries.size() ==
               static_cast<std::size_t>(emitter_count) + 1u &&
           entries.back().cumulative > 0.0f;
}

CyclesLightDistribution build_cycles_light_distribution(
    std::span<const float> emissive_triangle_areas,
    std::uint32_t analytic_light_count,
    bool include_environment) {
    CyclesLightDistribution result;
    const auto triangle_count =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                emissive_triangle_areas.size(),
                std::numeric_limits<std::uint32_t>::max()));
    const auto environment_count =
        include_environment ? 1u : 0u;
    const auto light_count =
        analytic_light_count + environment_count;
    result.emitter_count = triangle_count + light_count;
    result.entries.reserve(
        static_cast<std::size_t>(result.emitter_count) + 1u);

    float total_area = 0.0f;
    for (std::uint32_t index = 0u;
         index < triangle_count;
         ++index) {
        result.entries.emplace_back(
            LightDistributionEntry{
                .cumulative = total_area,
                .selection_pdf = 0.0f,
                .kind =
                    LightDistributionEmitterKind::
                        emissive_triangle,
                .index = index});
        total_area += valid_area(
            emissive_triangle_areas[index]);
    }
    const auto triangle_area = total_area;

    const auto light_area =
        light_count > 0u
            ? (total_area > 0.0f
                   ? total_area /
                         static_cast<float>(light_count)
                   : 1.0f)
            : 0.0f;
    for (std::uint32_t index = 0u;
         index < analytic_light_count;
         ++index) {
        result.entries.emplace_back(
            LightDistributionEntry{
                .cumulative = total_area,
                .selection_pdf = 0.0f,
                .kind =
                    LightDistributionEmitterKind::
                        analytic_light,
                .index = index});
        total_area += light_area;
    }
    if (include_environment) {
        result.entries.emplace_back(
            LightDistributionEntry{
                .cumulative = total_area,
                .selection_pdf = 0.0f,
                .kind =
                    LightDistributionEmitterKind::environment,
                .index = 0u});
        total_area += light_area;
    }

    result.entries.emplace_back(
        LightDistributionEntry{
            .cumulative = total_area,
            .selection_pdf = 0.0f,
            .kind =
                LightDistributionEmitterKind::sentinel,
            .index = 0u});

    if (triangle_area > 0.0f) {
        result.triangle_area_pdf = 1.0f / triangle_area;
        if (light_count > 0u) {
            result.triangle_area_pdf *= 0.5f;
        }
    }
    if (light_count > 0u) {
        result.light_selection_pdf =
            1.0f / static_cast<float>(light_count);
        if (triangle_area > 0.0f) {
            result.light_selection_pdf *= 0.5f;
        }
    }

    for (std::uint32_t index = 0u;
         index < triangle_count;
         ++index) {
        result.entries[index].selection_pdf =
            valid_area(emissive_triangle_areas[index]) *
            result.triangle_area_pdf;
    }
    for (std::uint32_t index = triangle_count;
         index < result.emitter_count;
         ++index) {
        result.entries[index].selection_pdf =
            result.light_selection_pdf;
    }

    if (total_area > 0.0f) {
        const auto inverse_total = 1.0f / total_area;
        for (std::uint32_t index = 0u;
             index < result.emitter_count;
             ++index) {
            result.entries[index].cumulative *=
                inverse_total;
        }
        result.entries.back().cumulative = 1.0f;
    }
    return result;
}

std::optional<std::uint32_t>
select_cycles_light_distribution(
    const CyclesLightDistribution &distribution,
    float sample) noexcept {
    if (!distribution.usable()) {
        return std::nullopt;
    }

    std::uint32_t first = 0u;
    std::uint32_t length =
        distribution.emitter_count + 1u;
    while (length > 0u) {
        const auto half_length = length >> 1u;
        const auto middle = first + half_length;
        if (sample <
            distribution.entries[middle].cumulative) {
            length = half_length;
        } else {
            first = middle + 1u;
            length =
                length - half_length - 1u;
        }
    }
    const auto index =
        std::clamp<std::int64_t>(
            static_cast<std::int64_t>(first) - 1,
            0,
            static_cast<std::int64_t>(
                distribution.emitter_count) -
                1);
    return static_cast<std::uint32_t>(index);
}

}// namespace psycles::sampling
