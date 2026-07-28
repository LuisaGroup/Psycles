#include "path_tracer_sampling.h"

namespace psycles::luisa_backend::detail {

PixelFilterCallable make_pixel_filter_callable(
    contract::PixelFilter pixel_filter) noexcept {
    PixelFilterCallable sample_box_filter =
        [](Float u, Float width) noexcept {
            return 0.5f +
                   (u - 0.5f) * width;
        };
    PixelFilterCallable sample_gaussian_filter =
        [](Float u, Float width) noexcept {
            // Cycles' Gaussian filter is exp(-2*x*x) over
            // [-width/2, width/2]. Invert its normalized CDF
            // numerically so raster sampling remains entirely in
            // the Luisa program.
            const auto radius = width * 0.5f;
            const auto target =
                abs(u * 2.0f - 1.0f);
            Float x = radius * target;
            Float total = 0.0f;
            Float integral = 0.0f;
            constexpr auto integration_steps = 32u;
            $for (i, integration_steps) {
                const auto x0 =
                    radius *
                    cast<float>(i) /
                    static_cast<float>(
                        integration_steps);
                const auto x1 =
                    radius *
                    cast<float>(i + 1u) /
                    static_cast<float>(
                        integration_steps);
                total +=
                    0.5f * (x1 - x0) *
                    (exp(-2.0f * x0 * x0) +
                     exp(-2.0f * x1 * x1));
            };
            $for (iteration, 8u) {
                static_cast<void>(iteration);
                integral = 0.0f;
                $for (i, integration_steps) {
                    const auto x0 =
                        x *
                        cast<float>(i) /
                        static_cast<float>(
                            integration_steps);
                    const auto x1 =
                        x *
                        cast<float>(i + 1u) /
                        static_cast<float>(
                            integration_steps);
                    integral +=
                        0.5f * (x1 - x0) *
                        (exp(-2.0f * x0 * x0) +
                         exp(-2.0f * x1 * x1));
                };
                x = clamp(
                    x -
                        (integral - target * total) /
                            max(
                                exp(-2.0f * x * x),
                                1.0e-7f),
                    0.0f,
                    radius);
            };
            return 0.5f +
                   select(-x, x, u >= 0.5f);
        };
    PixelFilterCallable sample_blackman_harris_filter =
        [](Float u, Float width) noexcept {
            // This is the continuous form of Cycles'
            // BLACKMAN_HARRIS inverse-CDF table. Cycles doubles
            // filter_width before building this symmetric filter,
            // so the positive support radius is filter_width.
            constexpr auto a0 = 0.35875f;
            constexpr auto a1 = 0.48829f;
            constexpr auto a2 = 0.14128f;
            constexpr auto a3 = 0.01168f;
            constexpr auto two_pi =
                6.28318530717958647692f;
            const auto radius = width;
            const auto full_width = radius * 2.0f;
            const auto target_fraction =
                abs(u * 2.0f - 1.0f);
            const auto target =
                target_fraction * a0 * radius;
            Float x = radius * target_fraction;
            $for (iteration, 8u) {
                static_cast<void>(iteration);
                const auto phase =
                    two_pi * x / full_width;
                const auto integral =
                    a0 * x +
                    a1 * full_width / two_pi *
                        sin(phase) +
                    a2 * full_width /
                        (2.0f * two_pi) *
                        sin(2.0f * phase) +
                    a3 * full_width /
                        (3.0f * two_pi) *
                        sin(3.0f * phase);
                const auto density =
                    a0 + a1 * cos(phase) +
                    a2 * cos(2.0f * phase) +
                    a3 * cos(3.0f * phase);
                x = clamp(
                    x -
                        (integral - target) /
                            max(density, 1.0e-7f),
                    0.0f,
                    radius);
            };
            return 0.5f +
                   select(-x, x, u >= 0.5f);
        };
    return [=](Float u, Float width) noexcept {
        if (pixel_filter ==
            contract::PixelFilter::box) {
            return sample_box_filter(u, width);
        }
        if (pixel_filter ==
            contract::PixelFilter::gaussian) {
            return sample_gaussian_filter(u, width);
        }
        return sample_blackman_harris_filter(
            u, width);
    };
}

}// namespace psycles::luisa_backend::detail
