#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/analytic_light_sampling.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::analytic_light_sampling {

inline constexpr float pi = 3.1415926535897932f;
inline constexpr float inverse_pi = 0.3183098861837907f;

struct RectangleSolidAngleSample {
    luisa::compute::Float3 position;
    luisa::compute::Float pdf;
};

[[nodiscard]] inline luisa::compute::Float
safe_divide(
    luisa::compute::Float numerator,
    luisa::compute::Float denominator) noexcept {
    return luisa::compute::select(
        0.0f,
        numerator / denominator,
        denominator != 0.0f);
}

// Uniformly sample the solid angle subtended by a rectangle using the
// area-preserving spherical-rectangle parametrization of Ureña et al. The
// construction is expressed in the light's orthonormal frame and returns a
// directional PDF directly, so no area-to-solid-angle conversion is composed
// a second time. The planar PDF limit handles the single-precision
// cancellation of the internal-angle sum for tiny or grazing rectangles.
[[nodiscard]] inline RectangleSolidAngleSample
sample_rectangle_solid_angle(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 center,
    luisa::compute::Float3 axis_u,
    luisa::compute::Float length_u,
    luisa::compute::Float3 axis_v,
    luisa::compute::Float length_v,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    auto z = cross(axis_u, axis_v);
    const auto center_offset = center - reference;
    auto z0 = dot(center_offset, z);
    const auto flip = z0 > 0.0f;
    z = select(z, -z, flip);
    z0 = select(z0, -z0, flip);

    const auto center_u = dot(center_offset, axis_u);
    const auto center_v = dot(center_offset, axis_v);
    const auto x0 = center_u - 0.5f * length_u;
    const auto x1 = center_u + 0.5f * length_u;
    const auto y0 = center_v - 0.5f * length_v;
    const auto y1 = center_v + 0.5f * length_v;

    auto normalized_edges =
        make_float4(-y0, x1, y1, -x0);
    normalized_edges /=
        sqrt(
            normalized_edges * normalized_edges +
            z0 * z0);
    const auto g0 = asin(clamp(
        -normalized_edges.x * normalized_edges.y,
        -1.0f,
        1.0f));
    const auto g1 = asin(clamp(
        -normalized_edges.y * normalized_edges.z,
        -1.0f,
        1.0f));
    const auto g2 = asin(clamp(
        -normalized_edges.z * normalized_edges.w,
        -1.0f,
        1.0f));
    const auto g3 = asin(clamp(
        -normalized_edges.w * normalized_edges.x,
        -1.0f,
        1.0f));
    const auto solid_angle = -(g0 + g1 + g2 + g3);

    const auto b0 = normalized_edges.x;
    const auto b1 = normalized_edges.z;
    const auto b0_squared = b0 * b0;
    const auto alpha =
        random.x * solid_angle + g2 + g3;
    const auto f_u = safe_divide(
        cos(alpha) * b0 + b1,
        sin(alpha));
    auto cos_u = copysign(
        1.0f /
            sqrt(
                f_u * f_u +
                b0_squared),
        f_u);
    cos_u = clamp(cos_u, -1.0f, 1.0f);
    auto sampled_u =
        -(cos_u * z0) /
        max(
            sqrt(max(
                1.0f - cos_u * cos_u,
                0.0f)),
            1.0e-7f);
    sampled_u = clamp(sampled_u, x0, x1);

    const auto distance_u_squared =
        sampled_u * sampled_u + z0 * z0;
    const auto h0 =
        y0 /
        sqrt(
            distance_u_squared +
            y0 * y0);
    const auto h1 =
        y1 /
        sqrt(
            distance_u_squared +
            y1 * y1);
    const auto h_v =
        h0 + random.y * (h1 - h0);
    const auto h_v_squared = h_v * h_v;
    const auto sampled_v = select(
        y1,
        h_v *
            sqrt(
                distance_u_squared /
                (1.0f - h_v_squared)),
        h_v_squared < 1.0f - 1.0e-6f);

    const auto position =
        reference +
        sampled_u * axis_u +
        sampled_v * axis_v +
        z0 * z;
    const auto minimum_edge_squared = min(
        min(
            normalized_edges.x * normalized_edges.x,
            normalized_edges.y * normalized_edges.y),
        min(
            normalized_edges.z * normalized_edges.z,
            normalized_edges.w * normalized_edges.w));
    const auto center_distance =
        length(center_offset);
    const auto planar_pdf = safe_divide(
        -(center_distance *
          center_distance *
          center_distance),
        z0 * length_u * length_v);
    const auto use_planar_limit =
        (solid_angle < 1.0e-5f) |
        (minimum_edge_squared > 0.99999f);
    return {
        .position = position,
        .pdf = select(
            1.0f / solid_angle,
            planar_pdf,
            use_planar_limit)};
}

// Cycles represents a zero-radius point as a unit-area delta emitter for the
// area-to-solid-angle conversion. Keeping the inverse-square term in the PDF,
// rather than pre-dividing radiance, gives one common LightSample contract for
// delta and finite oriented-disk point lights:
//
//   pdf_omega = pdf_area * distance^2 / abs(Ng . -D).
//
// For a delta point Ng == -D, pdf_area and the cosine are both one, so the
// conditional PDF is exactly distance^2.
[[nodiscard]] inline luisa::compute::Float
point_disk_pdf(luisa::compute::Float distance_squared,
               luisa::compute::Float light_cosine,
               luisa::compute::Float radius) noexcept {
    const auto radius_squared = radius * radius;
    const auto inverse_area = luisa::compute::select(
        1.0f,
        1.0f /
            luisa::compute::max(
                radius_squared * pi,
                1.0e-20f),
        radius_squared > 0.0f);
    return inverse_area * distance_squared /
           luisa::compute::max(light_cosine, 1.0e-20f);
}

// Cycles converts point-light power to radiance/intensity using
// eval_fac = invarea / pi. A zero-radius normalized point has the explicit
// surrogate area four and therefore eval_fac = 1 / (4 pi). A finite point
// uses its spherical area, even when its sampled geometry is an oriented
// disk; this is the normalization authored by the source light.
[[nodiscard]] inline luisa::compute::Float
point_eval_factor(luisa::compute::Float radius,
                  luisa::compute::Bool normalize_power) noexcept {
    const auto radius_squared = radius * radius;
    const auto finite_area =
        4.0f * pi * radius_squared;
    const auto area = luisa::compute::select(
        4.0f,
        finite_area,
        radius_squared > 0.0f);
    const auto inverse_area = luisa::compute::select(
        1.0f,
        1.0f /
            luisa::compute::max(area, 1.0e-20f),
        normalize_power);
    return inverse_area * inverse_pi;
}

// A finite point/spot can be intersected by the competing forward-BSDF
// technique. A zero-radius delta cannot, regardless of the Blender use-MIS
// property. This is a property of the sampling measures, not a scene-specific
// exception.
[[nodiscard]] inline luisa::compute::Bool
point_has_competing_bsdf_technique(
    luisa::compute::Float radius,
    luisa::compute::Bool use_mis) noexcept {
    return use_mis & (radius > 0.0f);
}

}// namespace psycles::luisa_backend::analytic_light_sampling
