#include <psycles/sampling/pixel_filter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace psycles::sampling {
namespace {

using FilterFunction = float (*)(float, float);

[[nodiscard]] float box_filter(float, float) noexcept { return 1.0f; }

[[nodiscard]] float gaussian_filter(float value, float width) noexcept {
    value *= 6.0f / width;
    return std::exp(-2.0f * value * value);
}

[[nodiscard]] float blackman_harris_filter(float value, float width) noexcept {
    constexpr auto two_pi = 6.28318530717958647692f;
    value = two_pi * (value / width + 0.5f);
    return 0.35875f - 0.48829f * std::cos(value) +
           0.14128f * std::cos(2.0f * value) -
           0.01168f * std::cos(3.0f * value);
}

} // namespace

PixelFilterTable make_pixel_filter_table(contract::PixelFilter filter,
                                         float width) {
    width = std::max(width, 1.0e-5f);
    FilterFunction function = box_filter;
    switch (filter) {
    case contract::PixelFilter::box:
        break;
    case contract::PixelFilter::gaussian:
        function = gaussian_filter;
        width *= 3.0f;
        break;
    case contract::PixelFilter::blackman_harris:
        function = blackman_harris_filter;
        width *= 2.0f;
        break;
    }

    // This follows Cycles' util_cdf_evaluate(resolution - 1, ...) and
    // util_cdf_invert(..., make_symmetric=true) construction. In particular,
    // the even-sized table leaves its final element at zero. Although unusual,
    // interpolating against that endpoint is observable and must not be
    // "cleaned up" independently.
    constexpr auto cdf_resolution = pixel_filter_table_size - 1u;
    std::array<float, pixel_filter_table_size> cdf{};
    const auto range = width * 0.5f;
    for (std::size_t i = 0u; i < cdf_resolution; ++i) {
        const auto value = range * static_cast<float>(i) /
                           static_cast<float>(cdf_resolution - 1u);
        cdf[i + 1u] = cdf[i] + std::abs(function(value, width));
    }
    const auto normalization =
        cdf[cdf_resolution] == 0.0f ? 0.0f : 1.0f / cdf[cdf_resolution];
    for (auto &value : cdf) {
        value *= normalization;
    }
    cdf[cdf_resolution] = 1.0f;

    PixelFilterTable inverse{};
    constexpr auto half_size = (pixel_filter_table_size - 1u) / 2u;
    for (std::size_t i = 0u; i <= half_size; ++i) {
        const auto x = static_cast<float>(i) / static_cast<float>(half_size);
        auto iterator = std::upper_bound(cdf.begin(), cdf.end(), x);
        auto index = static_cast<std::size_t>(iterator - cdf.begin());
        float interpolation = 0.0f;
        if (index < cdf_resolution) {
            interpolation = (x - cdf[index]) / (cdf[index + 1u] - cdf[index]);
        } else {
            index = cdf_resolution;
        }
        const auto y = (static_cast<float>(index) + interpolation) /
                       static_cast<float>(pixel_filter_table_size - 1u) *
                       (2.0f * range);
        inverse[half_size + i] = 0.5f * (1.0f + y);
        inverse[half_size - i] = 0.5f * (1.0f - y);
    }
    return inverse;
}

} // namespace psycles::sampling
