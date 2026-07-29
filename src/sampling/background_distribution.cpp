#include <psycles/sampling/background_distribution.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace psycles::sampling {
namespace {

[[nodiscard]] float average_absolute(Vec3f value) noexcept {
    return (std::abs(value.x) + std::abs(value.y) + std::abs(value.z)) / 3.0f;
}

} // namespace

bool CyclesBackgroundMapDistribution::valid() const noexcept {
    if (width == 0u || height == 0u) {
        return false;
    }
    const auto conditional_size =
        static_cast<std::size_t>(width + 1u) * static_cast<std::size_t>(height);
    return conditional.size() == conditional_size &&
           marginal.size() == static_cast<std::size_t>(height) + 1u;
}

CyclesBackgroundMapDistribution
build_cycles_background_map_distribution(std::span<const Vec3f> radiance,
                                         std::uint32_t width,
                                         std::uint32_t height) {
    if (width == 0u || height == 0u) {
        throw std::invalid_argument(
            "background distribution dimensions must be nonzero");
    }
    if (static_cast<std::size_t>(width) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(height)) {
        throw std::invalid_argument(
            "background distribution dimensions overflow");
    }
    const auto pixel_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (radiance.size() != pixel_count) {
        throw std::invalid_argument(
            "background distribution radiance size mismatch");
    }

    CyclesBackgroundMapDistribution result{
        .width = width,
        .height = height,
        .conditional = {},
        .marginal = {}};
    const auto cdf_width = static_cast<std::size_t>(width) + 1u;
    result.conditional.resize(cdf_width * static_cast<std::size_t>(height));
    result.marginal.resize(static_cast<std::size_t>(height) + 1u);

    constexpr auto pi = 3.14159265358979323846f;
    for (std::uint32_t y = 0u; y < height; ++y) {
        const auto row_offset = static_cast<std::size_t>(y) * cdf_width;
        const auto pixel_offset =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const auto sine_theta = std::sin(pi * (static_cast<float>(y) + 0.5f) /
                                         static_cast<float>(height));

        auto &first = result.conditional[row_offset];
        first.function = average_absolute(radiance[pixel_offset]) * sine_theta;
        first.cumulative = 0.0f;
        for (std::uint32_t x = 1u; x < width; ++x) {
            auto &entry =
                result.conditional[row_offset + static_cast<std::size_t>(x)];
            const auto &previous =
                result
                    .conditional[row_offset + static_cast<std::size_t>(x - 1u)];
            entry.function =
                average_absolute(
                    radiance[pixel_offset + static_cast<std::size_t>(x)]) *
                sine_theta;
            entry.cumulative = previous.cumulative +
                               previous.function / static_cast<float>(width);
        }

        const auto &last =
            result
                .conditional[row_offset + static_cast<std::size_t>(width - 1u)];
        const auto total =
            last.cumulative + last.function / static_cast<float>(width);
        auto &sentinel =
            result.conditional[row_offset + static_cast<std::size_t>(width)];
        sentinel.function = total;
        sentinel.cumulative = 1.0f;
        if (total > 0.0f) {
            const auto inverse_total = 1.0f / total;
            for (std::uint32_t x = 1u; x < width; ++x) {
                result.conditional[row_offset + static_cast<std::size_t>(x)]
                    .cumulative *= inverse_total;
            }
        }
    }

    result.marginal[0u] = {
        .function =
            result.conditional[static_cast<std::size_t>(width)].function,
        .cumulative = 0.0f};
    for (std::uint32_t y = 1u; y < height; ++y) {
        const auto row_offset = static_cast<std::size_t>(y) * cdf_width;
        const auto &previous =
            result.marginal[static_cast<std::size_t>(y - 1u)];
        result.marginal[static_cast<std::size_t>(y)] = {
            .function =
                result.conditional[row_offset + static_cast<std::size_t>(width)]
                    .function,
            .cumulative = previous.cumulative +
                          previous.function / static_cast<float>(height)};
    }
    const auto &last = result.marginal[static_cast<std::size_t>(height - 1u)];
    const auto total =
        last.cumulative + last.function / static_cast<float>(height);
    result.marginal[static_cast<std::size_t>(height)] = {.function = total,
                                                         .cumulative = 1.0f};
    if (total > 0.0f) {
        const auto inverse_total = 1.0f / total;
        for (std::uint32_t y = 1u; y < height; ++y) {
            result.marginal[static_cast<std::size_t>(y)].cumulative *=
                inverse_total;
        }
    }
    return result;
}

} // namespace psycles::sampling
