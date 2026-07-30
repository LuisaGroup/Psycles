#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/camera_sampling.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::camera_sampling {

inline constexpr float pi = 3.1415926535897932f;

// Cycles addresses film pixels from the lower-left, whereas Psycles stores
// output rows from the upper-left. A displayed output row must therefore use
// the vertically mirrored Cycles row for every pixel-keyed sample dimension.
// Keeping this conversion next to the corresponding sub-pixel conversion
// makes the raster-coordinate invariant explicit:
//
//   2 * (cycles_y + cycles_filter_y) / height - 1
//     == 1 - 2 * (output_y + output_filter_y) / height.
[[nodiscard]] inline luisa::compute::UInt
cycles_pixel_y(luisa::compute::UInt output_y,
               luisa::compute::UInt height) noexcept {
    return height - 1u - output_y;
}

[[nodiscard]] inline luisa::compute::Float
output_filter_y(luisa::compute::Float cycles_filter_y) noexcept {
    return 1.0f - cycles_filter_y;
}

// Cycles stores camera rays with the origin already advanced to the near
// clipping plane. The traversal interval therefore starts at zero and ends
// after (far - near), scaled by the camera-space direction cosine for a
// perspective ray. Orthographic and panoramic callers pass cosine = 1.
[[nodiscard]] inline luisa::compute::Float2 camera_clip_range(
    luisa::compute::Float near_clip,
    luisa::compute::Float far_clip,
    luisa::compute::Float direction_cosine) noexcept {
    const auto safe_cosine = luisa::compute::max(
        luisa::compute::abs(direction_cosine),
        1.0e-20f);
    return luisa::compute::make_float2(
        near_clip / safe_cosine,
        (far_clip - near_clip) / safe_cosine);
}

// Shirley-Chiu concentric square-to-disk map. Uniform polar sampling has the
// same density, but it destroys the two-dimensional stratification of the
// Sobol lens sample. Camera coverage must preserve this exact mapping so
// depth-of-field silhouettes use the same sample placement as Cycles.
[[nodiscard]] inline luisa::compute::Float2
sample_uniform_disk(luisa::compute::Float2 random) noexcept {
    const auto a = 2.0f * random.x - 1.0f;
    const auto b = 2.0f * random.y - 1.0f;
    const auto x_major = a * a > b * b;
    const auto safe_a = luisa::compute::select(1.0f, a, a != 0.0f);
    const auto safe_b = luisa::compute::select(1.0f, b, b != 0.0f);
    const auto radius = luisa::compute::select(b, a, x_major);
    const auto angle =
        luisa::compute::select(0.5f * pi - 0.25f * pi * (a / safe_b),
                               0.25f * pi * (b / safe_a),
                               x_major);
    const auto mapped =
        luisa::compute::make_float2(luisa::compute::cos(angle),
                                    luisa::compute::sin(angle)) *
        radius;
    return luisa::compute::select(
        mapped, luisa::compute::make_float2(0.0f), (a == 0.0f) & (b == 0.0f));
}

// Uniformly sample a regular polygon by selecting a corner triangle and
// sampling barycentric area within it. This is the finite-blade aperture
// counterpart of the concentric disk map.
[[nodiscard]] inline luisa::compute::Float2
sample_regular_polygon(luisa::compute::Float2 random,
                       std::uint32_t blade_count,
                       float rotation) noexcept {
    const auto corners = static_cast<float>(blade_count);
    auto u = random.x;
    auto v = random.y;
    const auto corner = floor(u * corners);
    u = u * corners - corner;
    u = sqrt(u);
    v *= u;
    u = 1.0f - u;

    const auto half_corner_angle = pi / corners;
    const auto point = luisa::compute::make_float2(
        (u + v) * cos(half_corner_angle), (u - v) * sin(half_corner_angle));
    const auto angle = rotation + corner * (2.0f * half_corner_angle);
    const auto cosine = cos(angle);
    const auto sine = sin(angle);
    return luisa::compute::make_float2(cosine * point.x - sine * point.y,
                                       sine * point.x + cosine * point.y);
}

[[nodiscard]] inline luisa::compute::Float2
sample_aperture(luisa::compute::Float2 random,
                std::uint32_t blade_count,
                float rotation) noexcept {
    if (blade_count == 0u) {
        return sample_uniform_disk(random);
    }
    return sample_regular_polygon(random, blade_count, rotation);
}

} // namespace psycles::luisa_backend::camera_sampling
