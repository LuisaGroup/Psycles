#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/camera_sampling.h> through the Psycles::luisa target."
#endif

#include <cmath>
#include <cstdint>

#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_transform.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::camera_sampling {

inline constexpr float pi = 3.1415926535897932f;

struct OrthographicViewplaneSpan {
    float horizontal{};
    float vertical{};
};

struct CameraToWorldRay {
    luisa::compute::Float3 origin;
    luisa::compute::Float3 direction;
};

// Camera rays participate in the same geometric-predicate contract as
// surface rays. A backend-native matrix lowering may round a product before
// translation and move a ray across a thin or coincident primitive. Preserve
// Cycles' explicit affine FMA tree here; normalization remains a separate
// stage because depth-of-field updates the camera-space direction first.
[[nodiscard]] inline CameraToWorldRay camera_to_world_ray(
    luisa::compute::Float4x4 transform,
    luisa::compute::Float3 origin,
    luisa::compute::Float3 direction) noexcept {
    return {
        .origin = cycles_transform::point(transform, origin),
        .direction = cycles_transform::direction(transform, direction)};
}

// Cycles defines orthographic_scale along the fitted sensor dimension. The
// other dimension follows from the render aspect ratio. Expressing that rule
// once at the host stage keeps camera-ray generation and conservative volume
// bounds on the same viewplane:
//
//   horizontal fit: (scale, scale / aspect)
//   vertical fit:   (scale * aspect, scale)
[[nodiscard]] constexpr OrthographicViewplaneSpan
orthographic_viewplane_span(float scale,
                            float aspect,
                            bool horizontal_fit) noexcept {
    return horizontal_fit
               ? OrthographicViewplaneSpan{scale, scale / aspect}
               : OrthographicViewplaneSpan{scale * aspect, scale};
}

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

// Advancing a ray origin by distance s changes the positional differential
// by s times the directional differential. Cycles applies this invariant when
// it moves camera rays onto the near clipping plane:
//
//   dP_clipped = dP_camera + s_near * dD_camera.
//
// Keep the compact scalar form here so ray construction cannot advance P
// without advancing its differential by the same parameter distance.
[[nodiscard]] inline luisa::compute::Float
advance_compact_differential_position(
    luisa::compute::Float differential_position,
    luisa::compute::Float differential_direction,
    luisa::compute::Float distance) noexcept {
    return differential_position + distance * differential_direction;
}

// Shirley-Chiu concentric square-to-disk map. Uniform polar sampling has the
// same density, but it destroys the two-dimensional stratification of the
// Sobol lens sample. Camera coverage must preserve this exact mapping so
// depth-of-field silhouettes use the same sample placement as Cycles.
[[nodiscard]] inline luisa::compute::Float2
sample_uniform_disk(luisa::compute::Float2 random) noexcept {
    return cycles_sample_mapping::sample_uniform_disk(random);
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
        (u + v) * std::cos(half_corner_angle),
        (u - v) * std::sin(half_corner_angle));
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
