#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error                                                                         \
    "Include <psycles/luisa/background_sampling.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/spherical_geometry.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::background_sampling {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float two_pi = 6.28318530717958647692f;
inline constexpr float uniform_sphere_pdf = 1.0f / (4.0f * pi);

struct BackgroundMapSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

struct BackgroundSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

template<typename Cdf, typename Index>
[[nodiscard]] inline auto
read_cdf(const Cdf &cdf, Index index) noexcept {
    if constexpr (requires {
                      cdf.read(index);
                  }) {
        return cdf.read(index);
    } else {
        return cdf->read(index);
    }
}

[[nodiscard]] inline luisa::compute::Float3
equirectangular_to_direction(luisa::compute::Float u,
                             luisa::compute::Float v) noexcept {
    const auto phi = pi - two_pi * u;
    const auto theta = pi * (1.0f - v);
    const auto sine_theta = sin(theta);
    return luisa::compute::make_float3(
        sine_theta * cos(phi), sine_theta * sin(phi), cos(theta));
}

[[nodiscard]] inline luisa::compute::Float2
direction_to_equirectangular(luisa::compute::Float3 direction) noexcept {
    direction = spherical_geometry::normalize_or(
        direction, luisa::compute::make_float3(0.0f, 0.0f, -1.0f));
    const auto u = fract(
        (pi - spherical_geometry::canonical_direction_azimuth(direction)) /
        two_pi);
    const auto v = 1.0f - acos(clamp(direction.z, -1.0f, 1.0f)) / pi;
    return luisa::compute::make_float2(u, v);
}

template<typename Cdf>
[[nodiscard]] inline luisa::compute::UInt
lower_bound_cdf(const Cdf &cdf,
                luisa::compute::UInt offset,
                std::uint32_t count,
                luisa::compute::Float sample) noexcept {
    luisa::compute::UInt first = 0u;
    luisa::compute::UInt length = count;
    $while(length > 0u) {
        const auto step = length >> 1u;
        const auto middle = first + step;
        const auto value =
            read_cdf(cdf, offset + middle).y;
        $if(value < sample) {
            first = middle + 1u;
            length -= step + 1u;
        }
        $else { length = step; };
    };
    return luisa::compute::select(0u, first - 1u, first > 0u);
}

template<typename Conditional, typename Marginal>
[[nodiscard]] inline BackgroundMapSample
sample_map(const Conditional &conditional,
           const Marginal &marginal,
           std::uint32_t width,
           std::uint32_t height,
           luisa::compute::Float2 random) noexcept {
    const auto index_v = lower_bound_cdf(marginal, 0u, height, random.y);
    const auto cdf_v =
        read_cdf(marginal, index_v);
    const auto cdf_next_v =
        read_cdf(marginal, index_v + 1u);
    const auto cdf_last_v =
        read_cdf(marginal, height);
    const auto dv = (random.y - cdf_v.y) / (cdf_next_v.y - cdf_v.y);
    const auto v = (luisa::compute::cast<float>(index_v) + dv) /
                   static_cast<float>(height);

    const auto cdf_width = width + 1u;
    const auto row_offset = index_v * cdf_width;
    const auto index_u =
        lower_bound_cdf(conditional, row_offset, width, random.x);
    const auto cdf_u =
        read_cdf(conditional, row_offset + index_u);
    const auto cdf_next_u =
        read_cdf(
            conditional,
            row_offset + index_u + 1u);
    const auto cdf_last_u =
        read_cdf(conditional, row_offset + width);
    const auto du = (random.x - cdf_u.y) / (cdf_next_u.y - cdf_u.y);
    const auto u =
        (luisa::compute::cast<float>(index_u) + du) / static_cast<float>(width);

    const auto sine_theta = sin(pi * v);
    const auto denominator =
        two_pi * pi * sine_theta * cdf_last_u.x * cdf_last_v.x;
    const auto pdf =
        luisa::compute::select(0.0f,
                               cdf_u.x * cdf_v.x / denominator,
                               (sine_theta != 0.0f) & (denominator != 0.0f));
    return {.direction = equirectangular_to_direction(u, v), .pdf = pdf};
}

template<typename Conditional, typename Marginal>
[[nodiscard]] inline luisa::compute::Float
map_pdf(const Conditional &conditional,
        const Marginal &marginal,
        std::uint32_t width,
        std::uint32_t height,
        luisa::compute::Float3 direction) noexcept {
    const auto uv = direction_to_equirectangular(direction);
    const auto sine_theta = sin(uv.y * pi);
    const auto index_u = min(
        luisa::compute::cast<std::uint32_t>(uv.x * static_cast<float>(width)),
        width - 1u);
    const auto index_v = min(
        luisa::compute::cast<std::uint32_t>(uv.y * static_cast<float>(height)),
        height - 1u);
    const auto cdf_width = width + 1u;
    const auto row_offset = index_v * cdf_width;
    const auto cdf_u =
        read_cdf(conditional, row_offset + index_u);
    const auto cdf_v =
        read_cdf(marginal, index_v);
    const auto cdf_last_u =
        read_cdf(conditional, row_offset + width);
    const auto cdf_last_v =
        read_cdf(marginal, height);
    const auto denominator =
        two_pi * pi * sine_theta * cdf_last_u.x * cdf_last_v.x;
    return luisa::compute::select(0.0f,
                                  cdf_u.x * cdf_v.x / denominator,
                                  (sine_theta != 0.0f) & (denominator != 0.0f));
}

[[nodiscard]] inline luisa::compute::Float3
sample_uniform_sphere(luisa::compute::Float2 random) noexcept {
    const auto z = 1.0f - 2.0f * random.x;
    const auto radius = sqrt(max(1.0f - z * z, 0.0f));
    const auto phi = two_pi * random.y;
    return luisa::compute::make_float3(radius * cos(phi), radius * sin(phi), z);
}

[[nodiscard]] inline luisa::compute::Float3
sample_sun(luisa::compute::Float3 axis,
           float angular_radius,
           luisa::compute::Float2 random) noexcept {
    axis = spherical_geometry::normalize_or(
        axis, luisa::compute::make_float3(0.0f, 0.0f, 1.0f));
    const auto cap_height = spherical_geometry::unit_cap_height(angular_radius);
    const auto cosine_theta = 1.0f - random.x * cap_height;
    const auto sine_theta = sqrt(max(1.0f - cosine_theta * cosine_theta, 0.0f));
    const auto reference =
        luisa::compute::select(luisa::compute::make_float3(0.0f, 0.0f, 1.0f),
                               luisa::compute::make_float3(0.0f, 1.0f, 0.0f),
                               abs(axis.z) > 0.999f);
    const auto tangent = spherical_geometry::normalize_or(
        cross(reference, axis), luisa::compute::make_float3(1.0f, 0.0f, 0.0f));
    const auto bitangent = cross(axis, tangent);
    const auto phi = two_pi * random.y;
    return tangent * (cos(phi) * sine_theta) +
           bitangent * (sin(phi) * sine_theta) + axis * cosine_theta;
}

[[nodiscard]] inline luisa::compute::Float
sun_pdf(luisa::compute::Float3 axis,
        float angular_radius,
        luisa::compute::Float3 direction) noexcept {
    if (angular_radius <= 0.0f) {
        return luisa::compute::Float{0.0f};
    }
    axis = spherical_geometry::normalize_or(
        axis, luisa::compute::make_float3(0.0f, 0.0f, 1.0f));
    direction = spherical_geometry::normalize_or(direction, axis);
    const auto cap_height = spherical_geometry::unit_cap_height(angular_radius);
    const auto inside = dot(axis, direction) >= 1.0f - cap_height;
    return luisa::compute::select(
        0.0f,
        1.0f / spherical_geometry::cap_solid_angle(angular_radius),
        inside);
}

template<typename Conditional, typename Marginal>
[[nodiscard]] inline luisa::compute::Float
pdf(const Conditional &conditional,
    const Marginal &marginal,
    std::uint32_t width,
    std::uint32_t height,
    float map_weight,
    float guided_sun_weight,
    luisa::compute::Float3 guided_sun_axis,
    float guided_sun_radius,
    luisa::compute::Float3 direction) noexcept {
    const auto total_weight = map_weight + guided_sun_weight;
    if (total_weight <= 0.0f) {
        return luisa::compute::Float{uniform_sphere_pdf};
    }
    const auto map_probability = map_weight / total_weight;
    const auto sun_probability = guided_sun_weight / total_weight;
    auto result = luisa::compute::Float{0.0f};
    if (map_weight > 0.0f) {
        result += map_probability *
                  map_pdf(conditional, marginal, width, height, direction);
    }
    if (guided_sun_weight > 0.0f) {
        result += sun_probability *
                  sun_pdf(guided_sun_axis, guided_sun_radius, direction);
    }
    return result;
}

template<typename Conditional, typename Marginal>
[[nodiscard]] inline BackgroundSample
sample(const Conditional &conditional,
       const Marginal &marginal,
       std::uint32_t width,
       std::uint32_t height,
       float map_weight,
       float guided_sun_weight,
       luisa::compute::Float3 guided_sun_axis,
       float guided_sun_radius,
       luisa::compute::Float2 random) noexcept {
    const auto total_weight = map_weight + guided_sun_weight;
    if (total_weight <= 0.0f) {
        return {.direction = sample_uniform_sphere(random),
                .pdf = uniform_sphere_pdf};
    }

    const auto map_probability = map_weight / total_weight;
    const auto sun_probability = guided_sun_weight / total_weight;
    luisa::compute::Float3 direction;
    luisa::compute::Float result_pdf;
    if (guided_sun_weight <= 0.0f) {
        const auto map =
            sample_map(conditional, marginal, width, height, random);
        return {.direction = map.direction, .pdf = map.pdf};
    }
    if (map_weight <= 0.0f) {
        direction = sample_sun(guided_sun_axis, guided_sun_radius, random);
        return {.direction = direction,
                .pdf = sun_pdf(guided_sun_axis, guided_sun_radius, direction)};
    }

    $if(random.x < sun_probability) {
        const auto sun_random =
            luisa::compute::make_float2(random.x / sun_probability, random.y);
        direction = sample_sun(guided_sun_axis, guided_sun_radius, sun_random);
        result_pdf =
            sun_probability *
                sun_pdf(guided_sun_axis, guided_sun_radius, direction) +
            map_probability *
                map_pdf(conditional, marginal, width, height, direction);
    }
    $else {
        const auto map_random = luisa::compute::make_float2(
            (random.x - sun_probability) / map_probability, random.y);
        const auto map =
            sample_map(conditional, marginal, width, height, map_random);
        direction = map.direction;
        result_pdf = map_probability * map.pdf +
                     sun_probability *
                         sun_pdf(guided_sun_axis, guided_sun_radius, direction);
    };
    return {.direction = direction, .pdf = result_pdf};
}

} // namespace psycles::luisa_backend::background_sampling
