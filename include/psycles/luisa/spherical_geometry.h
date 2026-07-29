#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/spherical_geometry.h> through the Psycles::luisa target."
#endif

#include <cmath>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::spherical_geometry {

inline constexpr float pi = 3.1415926535897932f;
inline constexpr float two_pi = 6.2831853071795864f;

struct TriangleDirectionalPdf {
    luisa::compute::Float value;
    luisa::compute::Bool uses_solid_angle;
    luisa::compute::Bool valid;
};

struct TriangleSample {
    luisa::compute::Float3 position;
    // We use the same two barycentric coordinates as Luisa surface hits:
    // p = (1 - u - v) p0 + u p1 + v p2.
    luisa::compute::Float2 barycentric;
    luisa::compute::Float3 direction;
    luisa::compute::Float distance;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Bool uses_solid_angle;
    luisa::compute::Bool valid;
};

[[nodiscard]] inline luisa::compute::Float3
normalize_or(luisa::compute::Float3 value,
             luisa::compute::Float3 fallback) noexcept {
    const auto length_squared = dot(value, value);
    return select(fallback,
                  value / sqrt(max(length_squared, 1.0e-20f)),
                  length_squared > 1.0e-20f);
}

// Height of a spherical cap on the unit sphere. The direct expression
// 1 - cos(radius) catastrophically cancels for sun-sized angles when a GPU
// backend lowers cos to a fast approximation. This equivalent half-angle
// identity remains well-conditioned as radius approaches zero.
[[nodiscard]] inline float unit_cap_height(float angular_radius) noexcept {
    const auto sine_half = std::sin(0.5f * angular_radius);
    return 2.0f * sine_half * sine_half;
}

[[nodiscard]] inline luisa::compute::Float
unit_cap_height(luisa::compute::Float angular_radius) noexcept {
    const auto sine_half = sin(0.5f * angular_radius);
    return 2.0f * sine_half * sine_half;
}

// Exact solid angle of a cone/spherical cap with angular radius r:
//   Omega = 2 pi (1 - cos(r)) = 4 pi sin^2(r / 2).
[[nodiscard]] inline float cap_solid_angle(float angular_radius) noexcept {
    return two_pi * unit_cap_height(angular_radius);
}

[[nodiscard]] inline luisa::compute::Float
cap_solid_angle(luisa::compute::Float angular_radius) noexcept {
    return two_pi * unit_cap_height(angular_radius);
}

// Exact solid angle subtended by a triangle at a reference point. Expressing
// this with atan2 keeps both very small and greater-than-hemisphere triangles
// well conditioned:
//   Omega = 2 atan2(|a . (b x c)|,
//                   1 + a.b + b.c + c.a).
[[nodiscard]] inline luisa::compute::Float
triangle_solid_angle(luisa::compute::Float3 reference,
                     luisa::compute::Float3 p0,
                     luisa::compute::Float3 p1,
                     luisa::compute::Float3 p2) noexcept {
    const auto a = normalize_or(p0 - reference,
                                luisa::compute::make_float3(1.0f, 0.0f, 0.0f));
    const auto b = normalize_or(p1 - reference,
                                luisa::compute::make_float3(0.0f, 1.0f, 0.0f));
    const auto c = normalize_or(p2 - reference,
                                luisa::compute::make_float3(0.0f, 0.0f, 1.0f));
    const auto numerator = abs(dot(a, cross(b, c)));
    const auto denominator = 1.0f + dot(a, b) + dot(b, c) + dot(c, a);
    return 2.0f * atan2(numerator, denominator);
}

// Cycles selects the more expensive solid-angle strategy exactly when the
// triangle is large relative to its plane distance. This predicate is shared
// by NEE sampling and the competing forward-hit PDF; keeping it in one
// function is the invariant that makes MIS consistent.
[[nodiscard]] inline luisa::compute::Bool
use_triangle_solid_angle_sampling(luisa::compute::Float3 reference,
                                  luisa::compute::Float3 p0,
                                  luisa::compute::Float3 p1,
                                  luisa::compute::Float3 p2) noexcept {
    const auto e0 = p1 - p0;
    const auto e1 = p2 - p0;
    const auto e2 = p2 - p1;
    const auto longest_edge_squared =
        max(dot(e0, e0), max(dot(e1, e1), dot(e2, e2)));
    const auto unnormalized_normal = cross(e0, e1);
    const auto normal_length_squared =
        dot(unnormalized_normal, unnormalized_normal);
    const auto safe_normal_length_squared =
        max(normal_length_squared, 1.0e-20f);
    const auto distance_to_plane =
        dot(unnormalized_normal, p0 - reference) / safe_normal_length_squared;
    return (normal_length_squared > 1.0e-20f) &
           (longest_edge_squared > distance_to_plane * distance_to_plane);
}

// Eric Heitz's low-distortion square-to-triangle map. It remains uniform in
// area like the familiar square-root map, but matching this map also preserves
// Cycles' Sobol sample placement rather than merely its expectation.
[[nodiscard]] inline luisa::compute::Float2
low_distortion_triangle_barycentric(luisa::compute::Float2 random) noexcept {
    const auto x = clamp(random.x, 0.0f, 1.0f);
    const auto y = clamp(random.y, 0.0f, 1.0f);
    const auto upper = y > x;
    const auto u_upper = 0.5f * x;
    const auto v_upper = y - u_upper;
    const auto v_lower = 0.5f * y;
    const auto u_lower = x - v_lower;
    return luisa::compute::make_float2(select(u_lower, u_upper, upper),
                                       select(v_lower, v_upper, upper));
}

[[nodiscard]] inline TriangleDirectionalPdf
triangle_directional_pdf(luisa::compute::Float3 reference,
                         luisa::compute::Float3 light_position,
                         luisa::compute::Float3 p0,
                         luisa::compute::Float3 p1,
                         luisa::compute::Float3 p2) noexcept {
    const auto e0 = p1 - p0;
    const auto e1 = p2 - p0;
    const auto unnormalized_normal = cross(e0, e1);
    const auto doubled_area =
        sqrt(max(dot(unnormalized_normal, unnormalized_normal), 0.0f));
    const auto area = 0.5f * doubled_area;
    const auto normal = unnormalized_normal / max(doubled_area, 1.0e-20f);
    const auto offset = light_position - reference;
    const auto distance_squared = dot(offset, offset);
    const auto direction = offset / sqrt(max(distance_squared, 1.0e-20f));
    const auto cosine = abs(dot(normal, -direction));
    const auto uses_solid_angle =
        use_triangle_solid_angle_sampling(reference, p0, p1, p2);
    const auto solid_angle = triangle_solid_angle(reference, p0, p1, p2);
    const auto area_pdf = distance_squared / max(cosine * area, 1.0e-20f);
    const auto solid_angle_pdf = 1.0f / max(solid_angle, 1.0e-20f);
    const auto valid_area =
        (area > 0.0f) & (distance_squared > 1.0e-12f) & (cosine > 0.0f);
    const auto valid_solid_angle = valid_area & (solid_angle > 0.0f);
    return {.value = select(area_pdf, solid_angle_pdf, uses_solid_angle),
            .uses_solid_angle = uses_solid_angle,
            .valid = select(valid_area, valid_solid_angle, uses_solid_angle)};
}

// Uniformly sample the subtended spherical triangle when it is large and use
// the low-distortion area map otherwise. The returned conditional PDF excludes
// emitter-selection probability; callers compose that exactly once.
[[nodiscard]] inline TriangleSample
sample_triangle(luisa::compute::Float3 reference,
                luisa::compute::Float3 p0,
                luisa::compute::Float3 p1,
                luisa::compute::Float3 p2,
                luisa::compute::Float2 random) noexcept {
    const auto area_barycentric = low_distortion_triangle_barycentric(random);
    const auto area_position =
        (1.0f - area_barycentric.x - area_barycentric.y) * p0 +
        area_barycentric.x * p1 + area_barycentric.y * p2;
    const auto area_offset = area_position - reference;
    const auto area_distance_squared = dot(area_offset, area_offset);
    const auto area_distance = sqrt(max(area_distance_squared, 1.0e-20f));
    const auto area_direction = area_offset / area_distance;

    const auto a = normalize_or(p0 - reference,
                                luisa::compute::make_float3(1.0f, 0.0f, 0.0f));
    const auto b = normalize_or(p1 - reference,
                                luisa::compute::make_float3(0.0f, 1.0f, 0.0f));
    const auto c = normalize_or(p2 - reference,
                                luisa::compute::make_float3(0.0f, 0.0f, 1.0f));
    const auto cosine_a = clamp(dot(b, c), -1.0f, 1.0f);
    const auto cosine_b = clamp(dot(a, c), -1.0f, 1.0f);
    const auto cosine_c = clamp(dot(a, b), -1.0f, 1.0f);
    const auto solid_angle = triangle_solid_angle(reference, p0, p1, p2);
    const auto ab_normal = normalize_or(
        cross(a, b), luisa::compute::make_float3(1.0f, 0.0f, 0.0f));
    const auto ac_normal = normalize_or(
        cross(a, c), luisa::compute::make_float3(0.0f, 1.0f, 0.0f));
    const auto cosine_alpha = clamp(dot(ab_normal, ac_normal), -1.0f, 1.0f);
    const auto sine_alpha = sqrt(max(1.0f - cosine_alpha * cosine_alpha, 0.0f));
    const auto alpha = acos(cosine_alpha);
    const auto phi = clamp(random.x, 0.0f, 1.0f) * solid_angle - alpha;
    const auto sine_phi = sin(phi);
    const auto cosine_phi = cos(phi);
    const auto u = cosine_phi - cosine_alpha;
    const auto v = sine_phi + sine_alpha * cosine_c;
    const auto numerator = (v * cosine_phi - u * sine_phi) * cosine_alpha - v;
    const auto denominator = (v * sine_phi + u * cosine_phi) * sine_alpha;
    const auto safe_denominator =
        select(select(-1.0e-20f, 1.0e-20f, denominator >= 0.0f),
               denominator,
               abs(denominator) > 1.0e-20f);
    const auto q = select(
        numerator / safe_denominator, 1.0f, abs(denominator) <= 1.0e-20f);
    const auto tangent = normalize_or(
        c - cosine_b * a, luisa::compute::make_float3(0.0f, 1.0f, 0.0f));
    const auto c_prime = normalize_or(
        clamp(q, -1.0f, 1.0f) * a + sqrt(max(1.0f - q * q, 0.0f)) * tangent, a);
    const auto z =
        1.0f - clamp(random.y, 0.0f, 1.0f) * (1.0f - dot(c_prime, b));
    const auto edge_direction = normalize_or(c_prime - dot(c_prime, b) * b, a);
    const auto solid_direction =
        normalize_or(z * b + sqrt(max(1.0f - z * z, 0.0f)) * edge_direction, b);

    // Intersect the sampled direction with the original planar triangle so
    // attributes can use the same barycentric point as the light PDF.
    const auto edge01 = p1 - p0;
    const auto edge02 = p2 - p0;
    const auto p_vector = cross(solid_direction, edge02);
    const auto determinant = dot(edge01, p_vector);
    const auto safe_determinant =
        select(select(-1.0e-20f, 1.0e-20f, determinant >= 0.0f),
               determinant,
               abs(determinant) > 1.0e-20f);
    const auto inverse_determinant = 1.0f / safe_determinant;
    const auto t_vector = reference - p0;
    const auto solid_u = dot(t_vector, p_vector) * inverse_determinant;
    const auto q_vector = cross(t_vector, edge01);
    const auto solid_v = dot(solid_direction, q_vector) * inverse_determinant;
    const auto solid_distance = dot(edge02, q_vector) * inverse_determinant;
    const auto solid_barycentric =
        luisa::compute::make_float2(solid_u, solid_v);
    const auto solid_position = reference + solid_direction * solid_distance;
    const auto solid_valid =
        (abs(determinant) > 1.0e-20f) & (solid_distance > 0.0f) &
        (solid_u >= -1.0e-5f) & (solid_v >= -1.0e-5f) &
        (solid_u + solid_v <= 1.0f + 1.0e-5f) & (solid_angle > 0.0f);

    const auto uses_solid_angle =
        use_triangle_solid_angle_sampling(reference, p0, p1, p2);
    const auto position =
        select(area_position, solid_position, uses_solid_angle);
    const auto pdf = triangle_directional_pdf(reference, position, p0, p1, p2);
    return {.position = position,
            .barycentric =
                select(area_barycentric, solid_barycentric, uses_solid_angle),
            .direction =
                select(area_direction, solid_direction, uses_solid_angle),
            .distance = select(area_distance, solid_distance, uses_solid_angle),
            .conditional_pdf = pdf.value,
            .uses_solid_angle = uses_solid_angle,
            .valid = pdf.valid & select(area_distance_squared > 1.0e-12f,
                                        solid_valid,
                                        uses_solid_angle)};
}

} // namespace psycles::luisa_backend::spherical_geometry
