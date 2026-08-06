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

struct RectangleSolidAngleContext {
    luisa::compute::Float3 z;
    luisa::compute::Float3 center_offset;
    luisa::compute::Float z0;
    luisa::compute::Float x0;
    luisa::compute::Float x1;
    luisa::compute::Float y0;
    luisa::compute::Float y1;
    luisa::compute::Float4 normalized_edges;
    luisa::compute::Float4 internal_angles;
    luisa::compute::Float solid_angle;
    luisa::compute::Float pdf;
};

struct LightLinearTransform {
    luisa::compute::Float3 column_x;
    luisa::compute::Float3 column_y;
    luisa::compute::Float3 column_z;
    luisa::compute::Float3 inverse_row_x;
    luisa::compute::Float3 inverse_row_y;
    luisa::compute::Float3 inverse_row_z;
    luisa::compute::Bool valid;
};

struct RayPrimitiveIntersection {
    luisa::compute::Bool valid;
    luisa::compute::Float distance;
    luisa::compute::Float3 position;
};

struct FiniteLightGeometrySample {
    luisa::compute::Bool valid;
    luisa::compute::Float3 direction;
    luisa::compute::Float3 position;
    luisa::compute::Float3 normal;
    luisa::compute::Float distance;
    luisa::compute::Float conditional_pdf;
};

struct FiniteLightSample {
    luisa::compute::Bool valid;
    luisa::compute::Float3 direction;
    luisa::compute::Float3 position;
    luisa::compute::Float3 normal;
    luisa::compute::Float2 uv;
    luisa::compute::Float distance;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Float evaluation_factor;
};

struct DistantLightSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Float evaluation_factor;
};

// Cycles stores a sun's outward light normal in KernelLight::co and samples
// the cone around that normal before negating the result to obtain LightSample
// ::D. LightGpu::axis_z already denotes D at the cone center, so sampling a
// cone directly around axis_z would mirror the deterministic azimuth even
// though it preserves the same density. Keep the complete mapping here so
// surface and volume NEE cannot silently choose different Sobol-to-direction
// conventions.
[[nodiscard]] inline DistantLightSample sample_distant_light(
    luisa::compute::Float3 axis_z,
    luisa::compute::Float angle,
    luisa::compute::Float2 random,
    luisa::compute::Bool normalize_power) noexcept {
    using namespace luisa::compute;

    const auto half_angle = 0.5f * max(angle, 0.0f);
    const auto sine_quarter = sin(0.5f * half_angle);
    const auto one_minus_cosine =
        2.0f * sine_quarter * sine_quarter;
    const auto normal_sample =
        cycles_sample_mapping::sample_uniform_cone(
            -axis_z, one_minus_cosine, random);
    const auto finite = half_angle > 0.0f;
    const auto sine_half = sin(half_angle);
    const auto disk_area = pi * sine_half * sine_half;
    return {
        .direction = -normal_sample.direction,
        .conditional_pdf = normal_sample.pdf,
        .evaluation_factor = select(
            1.0f,
            1.0f / max(disk_area, 1.0e-20f),
            normalize_power & finite)};
}

[[nodiscard]] inline luisa::compute::Float
safe_divide(
    luisa::compute::Float numerator,
    luisa::compute::Float denominator) noexcept {
    return luisa::compute::select(
        0.0f,
        numerator / denominator,
        denominator != 0.0f);
}

[[nodiscard]] inline luisa::compute::Float3
safe_normalize(
    luisa::compute::Float3 value,
    luisa::compute::Float3 fallback) noexcept {
    using namespace luisa::compute;
    const auto magnitude_squared = dot(value, value);
    return select(
        fallback,
        value /
            sqrt(max(magnitude_squared, 1.0e-30f)),
        magnitude_squared > 0.0f);
}

// A lamp shader and a spot profile use Cycles' complete object transform,
// including non-uniform scale. Store normalized columns plus their lengths in
// LightGpu, then reconstruct the inverse exactly through the adjugate instead
// of treating the normalized frame as an inverse transform.
[[nodiscard]] inline LightLinearTransform
light_linear_transform(
    luisa::compute::Float3 axis_x,
    luisa::compute::Float3 axis_y,
    luisa::compute::Float3 axis_z,
    luisa::compute::Float3 axis_scale) noexcept {
    using namespace luisa::compute;
    const auto column_x = axis_x * axis_scale.x;
    const auto column_y = axis_y * axis_scale.y;
    const auto column_z = axis_z * axis_scale.z;
    const auto adjugate_x = cross(column_y, column_z);
    const auto determinant = dot(column_x, adjugate_x);
    const auto valid = abs(determinant) > 1.0e-20f;
    const auto inverse_determinant =
        1.0f /
        select(1.0f, determinant, valid);
    return {
        .column_x = column_x,
        .column_y = column_y,
        .column_z = column_z,
        .inverse_row_x =
            adjugate_x * inverse_determinant,
        .inverse_row_y =
            cross(column_z, column_x) *
            inverse_determinant,
        .inverse_row_z =
            cross(column_x, column_y) *
            inverse_determinant,
        .valid = valid};
}

[[nodiscard]] inline luisa::compute::Float3
world_to_light_direction(
    luisa::compute::Float3 direction,
    const LightLinearTransform &transform) noexcept {
    using namespace luisa::compute;
    return make_float3(
        dot(transform.inverse_row_x, direction),
        dot(transform.inverse_row_y, direction),
        dot(transform.inverse_row_z, direction));
}

[[nodiscard]] inline luisa::compute::Float3
world_to_light_normal(
    luisa::compute::Float3 normal,
    const LightLinearTransform &transform) noexcept {
    using namespace luisa::compute;
    return safe_normalize(
        make_float3(
            dot(transform.column_x, normal),
            dot(transform.column_y, normal),
            dot(transform.column_z, normal)),
        normal);
}

// Exact open-interval Cycles sphere intersection. Keeping this primitive
// shared between NEE's constrained spot-cone branch and forward light hits
// makes their support mathematically identical.
[[nodiscard]] inline RayPrimitiveIntersection
intersect_sphere(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float radius) noexcept {
    using namespace luisa::compute;
    const auto center_offset = center - ray_origin;
    const auto radius_squared = radius * radius;
    const auto center_distance_squared =
        dot(center_offset, center_offset);
    const auto projected_distance =
        dot(center_offset, ray_direction);
    const auto perpendicular =
        center_offset -
        projected_distance * ray_direction;
    const auto perpendicular_squared =
        dot(perpendicular, perpendicular);
    const auto distance =
        projected_distance -
        copysign(
            sqrt(max(
                radius_squared -
                    perpendicular_squared,
                0.0f)),
            center_distance_squared -
                radius_squared);
    const auto valid =
        !((center_distance_squared >
           radius_squared) &
          (projected_distance < 0.0f)) &
        (perpendicular_squared <=
         radius_squared) &
        (distance > ray_minimum) &
        (distance < ray_maximum);
    return {
        .valid = valid,
        .distance = distance,
        .position =
            ray_origin +
            ray_direction * distance};
}

[[nodiscard]] inline RayPrimitiveIntersection
intersect_disk(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float3 normal,
    luisa::compute::Float radius) noexcept {
    using namespace luisa::compute;
    const auto center_offset = ray_origin - center;
    const auto plane_distance =
        dot(center_offset, normal);
    const auto cosine =
        dot(normal, -ray_direction);
    const auto distance =
        safe_divide(plane_distance, cosine);
    const auto position =
        ray_origin +
        distance * ray_direction;
    const auto radial_offset =
        position - center;
    const auto valid =
        (plane_distance * cosine > 0.0f) &
        (distance >= 0.0f) &
        (dot(radial_offset, radial_offset) <
         radius * radius) &
        (distance > ray_minimum) &
        (distance < ray_maximum);
    return {
        .valid = valid,
        .distance = distance,
        .position = position};
}

[[nodiscard]] inline luisa::compute::Float
point_disk_pdf(
    luisa::compute::Float distance_squared,
    luisa::compute::Float light_cosine,
    luisa::compute::Float radius) noexcept;

[[nodiscard]] inline luisa::compute::Float
sphere_light_pdf(
    luisa::compute::Float center_distance_squared,
    luisa::compute::Float radius_squared,
    luisa::compute::Float3 shading_normal,
    luisa::compute::Float3 direction,
    luisa::compute::Bool had_transmission,
    luisa::compute::Float outside_one_minus_cosine) noexcept {
    using namespace luisa::compute;
    const auto outside_pdf =
        cycles_sample_mapping::inverse_two_pi /
        max(outside_one_minus_cosine, 1.0e-30f);
    const auto inside_pdf = select(
        cycles_sample_mapping::
            cosine_hemisphere_pdf(
                shading_normal,
                direction),
        0.5f *
            cycles_sample_mapping::
                inverse_two_pi,
        had_transmission);
    return select(
        inside_pdf,
        outside_pdf,
        center_distance_squared >
            radius_squared);
}

[[nodiscard]] inline luisa::compute::Float3
spot_light_to_local(
    luisa::compute::Float3 ray,
    const LightLinearTransform &transform) noexcept {
    using namespace luisa::compute;
    auto local = safe_normalize(
        world_to_light_direction(
            ray, transform),
        make_float3(0.0f, 0.0f, -1.0f));
    local.z = -local.z;
    return local;
}

[[nodiscard]] inline luisa::compute::Float
spot_light_attenuation(
    luisa::compute::Float3 local_ray,
    luisa::compute::Float spot_angle,
    luisa::compute::Float spot_smooth) noexcept {
    using namespace luisa::compute;
    const auto cosine_half_angle =
        cos(0.5f * spot_angle);
    const auto blend_width =
        (1.0f - cosine_half_angle) *
        spot_smooth;
    const auto linear = clamp(
        safe_divide(
            local_ray.z -
                cosine_half_angle,
            blend_width),
        0.0f,
        1.0f);
    const auto smooth =
        linear * linear *
        (3.0f - 2.0f * linear);
    const auto hard = select(
        0.0f,
        1.0f,
        local_ray.z >=
            cosine_half_angle);
    return select(
        hard,
        smooth,
        blend_width > 0.0f);
}

[[nodiscard]] inline luisa::compute::Float
spot_one_minus_cosine_larger_spread(
    luisa::compute::Float spot_angle,
    luisa::compute::Float3 axis_scale) noexcept {
    using namespace luisa::compute;
    const auto tangent =
        tan(0.5f * spot_angle);
    const auto transverse_scale_squared =
        max(
            axis_scale.x * axis_scale.x,
            axis_scale.y * axis_scale.y);
    const auto axial_scale_squared =
        axis_scale.z * axis_scale.z;
    const auto cosine =
        rsqrt(
            1.0f +
            tangent * tangent *
                transverse_scale_squared /
                max(
                    axial_scale_squared,
                    1.0e-30f));
    return 1.0f - cosine;
}

[[nodiscard]] inline luisa::compute::Float2
spot_light_uv(
    luisa::compute::Float3 local_ray,
    luisa::compute::Float spot_angle) noexcept {
    using namespace luisa::compute;
    const auto half_cotangent =
        0.5f /
        tan(0.5f * spot_angle);
    const auto factor =
        half_cotangent / local_ray.z;
    return make_float2(
        local_ray.y * factor + 0.5f,
        -(local_ray.x + local_ray.y) *
            factor);
}

[[nodiscard]] inline luisa::compute::Float2
point_light_uv(
    luisa::compute::Float3 normal,
    const LightLinearTransform &transform) noexcept {
    using namespace luisa::compute;
    const auto local =
        world_to_light_direction(
            normal, transform);
    const auto magnitude_squared =
        dot(local, local);
    const auto has_azimuth =
        (local.x != 0.0f) |
        (local.y != 0.0f);
    const auto u = select(
        0.0f,
        0.5f -
            atan2(local.x, local.y) *
                cycles_sample_mapping::
                    inverse_two_pi,
        has_azimuth);
    const auto v = select(
        0.0f,
        1.0f -
            acos(clamp(
                local.z /
                    sqrt(max(
                        magnitude_squared,
                        1.0e-30f)),
                -1.0f,
                1.0f)) *
                inverse_pi,
        magnitude_squared > 0.0f);
    return make_float2(v, 1.0f - u - v);
}

[[nodiscard]] inline FiniteLightGeometrySample
sample_sphere_geometry(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 shading_normal,
    luisa::compute::Bool has_transmission,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Float3 outside_axis,
    luisa::compute::Float outside_one_minus_cosine,
    luisa::compute::Bool intersect_sampled_ray,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    const auto radius_squared =
        radius * radius;
    auto light_normal =
        reference - center;
    const auto center_distance_squared =
        dot(light_normal, light_normal);
    const auto center_distance =
        sqrt(max(
            center_distance_squared,
            1.0e-30f));
    light_normal /= center_distance;
    const auto outside =
        center_distance_squared >
        radius_squared;

    Float3 direction =
        make_float3(0.0f);
    Float cosine = 1.0f;
    Float conditional_pdf = 0.0f;
    Float distance = 0.0f;
    Float3 position = reference;
    Bool distance_known = false;
    Bool valid =
        (radius > 0.0f) &
        (center_distance_squared > 0.0f);

    $if (outside) {
        const auto cone =
            cycles_sample_mapping::
                sample_uniform_cone(
                    outside_axis,
                    outside_one_minus_cosine,
                    random);
        direction = cone.direction;
        cosine = cone.cosine;
        conditional_pdf = cone.pdf;
        $if (intersect_sampled_ray) {
            const auto intersection =
                intersect_sphere(
                    reference,
                    direction,
                    0.0f,
                    1.0e30f,
                    center,
                    radius);
            valid &= intersection.valid;
            distance = intersection.distance;
            position = intersection.position;
            distance_known =
                intersection.valid;
        };
    }
    $else {
        $if (has_transmission) {
            direction =
                cycles_sample_mapping::
                    sample_uniform_sphere(random);
            conditional_pdf =
                0.5f *
                cycles_sample_mapping::
                    inverse_two_pi;
        }
        $else {
            const auto hemisphere =
                cycles_sample_mapping::
                    sample_cosine_hemisphere(
                        shading_normal,
                        random);
            direction = hemisphere.direction;
            conditional_pdf = hemisphere.pdf;
        };
        cosine =
            -dot(direction, light_normal);
    };

    $if (!distance_known) {
        distance =
            center_distance * cosine -
            copysign(
                sqrt(max(
                    radius_squared -
                        center_distance_squared +
                        center_distance_squared *
                            cosine * cosine,
                    0.0f)),
                center_distance_squared -
                    radius_squared);
        position =
            reference +
            direction * distance;
    };

    const auto normal = safe_normalize(
        position - center,
        -direction);
    position =
        normal * radius + center;
    return {
        .valid = valid,
        .direction = direction,
        .position = position,
        .normal = normal,
        .distance = distance,
        .conditional_pdf =
            conditional_pdf};
}

[[nodiscard]] inline FiniteLightGeometrySample
sample_disk_geometry(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    auto light_normal =
        reference - center;
    const auto center_distance_squared =
        dot(light_normal, light_normal);
    const auto center_distance =
        sqrt(max(
            center_distance_squared,
            1.0e-30f));
    light_normal /= center_distance;
    const auto basis =
        cycles_sample_mapping::
            make_orthonormals(light_normal);
    const auto disk =
        cycles_sample_mapping::
            sample_uniform_disk(random);
    const auto position =
        center +
        radius *
            (disk.x * basis.tangent +
             disk.y * basis.bitangent);
    const auto offset = position - reference;
    const auto distance_squared =
        dot(offset, offset);
    const auto distance =
        sqrt(max(
            distance_squared,
            1.0e-30f));
    const auto direction =
        offset / distance;
    const auto cosine =
        dot(light_normal, -direction);
    const auto conditional_pdf =
        point_disk_pdf(
            distance_squared,
            cosine,
            radius);
    return {
        .valid =
            (center_distance_squared > 0.0f) &
            (cosine > 0.0f),
        .direction = direction,
        .position = position,
        .normal = -direction,
        .distance = distance,
        .conditional_pdf =
            conditional_pdf};
}

[[nodiscard]] inline RectangleSolidAngleContext
rectangle_solid_angle_context(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 center,
    luisa::compute::Float3 axis_u,
    luisa::compute::Float length_u,
    luisa::compute::Float3 axis_v,
    luisa::compute::Float length_v) noexcept {
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
    const auto internal_angles = make_float4(
        asin(clamp(
            -normalized_edges.x * normalized_edges.y,
            -1.0f,
            1.0f)),
        asin(clamp(
            -normalized_edges.y * normalized_edges.z,
            -1.0f,
            1.0f)),
        asin(clamp(
            -normalized_edges.z * normalized_edges.w,
            -1.0f,
            1.0f)),
        asin(clamp(
            -normalized_edges.w * normalized_edges.x,
            -1.0f,
            1.0f)));
    const auto solid_angle =
        -(internal_angles.x +
          internal_angles.y +
          internal_angles.z +
          internal_angles.w);
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
        .z = z,
        .center_offset = center_offset,
        .z0 = z0,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1,
        .normalized_edges = normalized_edges,
        .internal_angles = internal_angles,
        .solid_angle = solid_angle,
        .pdf = select(
            1.0f / solid_angle,
            planar_pdf,
            use_planar_limit)};
}

[[nodiscard]] inline luisa::compute::Float
rectangle_solid_angle_pdf(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 center,
    luisa::compute::Float3 axis_u,
    luisa::compute::Float length_u,
    luisa::compute::Float3 axis_v,
    luisa::compute::Float length_v) noexcept {
    return rectangle_solid_angle_context(
               reference,
               center,
               axis_u,
               length_u,
               axis_v,
               length_v)
        .pdf;
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

    const auto context =
        rectangle_solid_angle_context(
            reference,
            center,
            axis_u,
            length_u,
            axis_v,
            length_v);
    const auto b0 = context.normalized_edges.x;
    const auto b1 = context.normalized_edges.z;
    const auto b0_squared = b0 * b0;
    const auto alpha =
        random.x * context.solid_angle +
        context.internal_angles.z +
        context.internal_angles.w;
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
        -(cos_u * context.z0) /
        max(
            sqrt(max(
                1.0f - cos_u * cos_u,
                0.0f)),
            1.0e-7f);
    sampled_u = clamp(
        sampled_u, context.x0, context.x1);

    const auto distance_u_squared =
        sampled_u * sampled_u +
        context.z0 * context.z0;
    const auto h0 =
        context.y0 /
        sqrt(
            distance_u_squared +
            context.y0 * context.y0);
    const auto h1 =
        context.y1 /
        sqrt(
            distance_u_squared +
            context.y1 * context.y1);
    const auto h_v =
        h0 + random.y * (h1 - h0);
    const auto h_v_squared = h_v * h_v;
    const auto sampled_v = select(
        context.y1,
        h_v *
            sqrt(
                distance_u_squared /
                (1.0f - h_v_squared)),
        h_v_squared < 1.0f - 1.0e-6f);

    const auto position =
        reference +
        sampled_u * axis_u +
        sampled_v * axis_v +
        context.z0 * context.z;
    return {
        .position = position,
        .pdf = context.pdf};
}

[[nodiscard]] inline luisa::compute::Float
area_spread_attenuation(
    luisa::compute::Float3 direction_to_light,
    luisa::compute::Float3 light_normal,
    luisa::compute::Float spread) noexcept {
    using namespace luisa::compute;

    const auto cosine = max(
        dot(light_normal, -direction_to_light),
        0.0f);
    const auto half_spread =
        0.5f * max(spread, 0.0f);
    const auto sine_angle = sqrt(max(
        1.0f - cosine * cosine,
        0.0f));
    const auto tangent_angle =
        sine_angle / max(cosine, 1.0e-20f);
    const auto tangent_spread =
        tan(half_spread);
    const auto normalization = select(
        3.0f /
            max(
                half_spread *
                    half_spread *
                    half_spread,
                1.0e-20f),
        1.0f /
            max(
                tangent_spread -
                    half_spread,
                1.0e-20f),
        half_spread > 0.05f);
    const auto finite_spread =
        max(
            (tangent_spread -
             tangent_angle) *
                normalization,
            0.0f);
    const auto zero_spread = select(
        pi,
        0.0f,
        tangent_angle > 1.0e-5f);
    const auto narrowed = select(
        finite_spread,
        zero_spread,
        half_spread <= 0.0f);
    return select(
        narrowed,
        1.0f,
        spread >= pi - 1.0e-6f);
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
    return luisa::compute::select(
        0.0f,
        inverse_area * distance_squared /
            light_cosine,
        light_cosine > 0.0f);
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

[[nodiscard]] inline FiniteLightSample
sample_point_light(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 shading_normal,
    luisa::compute::Bool has_transmission,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Bool sphere,
    luisa::compute::Float3 axis_x,
    luisa::compute::Float3 axis_y,
    luisa::compute::Float3 axis_z,
    luisa::compute::Float3 axis_scale,
    luisa::compute::Float2 random,
    luisa::compute::Bool normalize_power) noexcept {
    using namespace luisa::compute;

    const auto center_offset =
        reference - center;
    const auto center_distance_squared =
        dot(center_offset, center_offset);
    const auto radius_squared =
        radius * radius;
    const auto light_normal = safe_normalize(
        center_offset,
        make_float3(0.0f, 0.0f, -1.0f));
    const auto sphere_cap =
        cycles_sample_mapping::
            sin_squared_to_one_minus_cosine(
                radius_squared /
                max(
                    center_distance_squared,
                    1.0e-30f));
    FiniteLightGeometrySample geometry{
        .valid = false,
        .direction = make_float3(0.0f),
        .position = center,
        .normal = make_float3(0.0f),
        .distance = 0.0f,
        .conditional_pdf = 0.0f};
    const auto finite_sphere =
        sphere & (radius > 0.0f);
    $if (finite_sphere) {
        geometry = sample_sphere_geometry(
            reference,
            shading_normal,
            has_transmission,
            center,
            radius,
            -light_normal,
            sphere_cap,
            false,
            random);
    }
    $else {
        geometry = sample_disk_geometry(
            reference,
            center,
            radius,
            random);
    };
    const auto transform =
        light_linear_transform(
            axis_x,
            axis_y,
            axis_z,
            axis_scale);
    return {
        .valid = geometry.valid,
        .direction = geometry.direction,
        .position = geometry.position,
        .normal = geometry.normal,
        .uv = point_light_uv(
            geometry.normal,
            transform),
        .distance = geometry.distance,
        .conditional_pdf =
            geometry.conditional_pdf,
        .evaluation_factor =
            point_eval_factor(
                radius,
                normalize_power)};
}

[[nodiscard]] inline FiniteLightSample
sample_spot_light(
    luisa::compute::Float3 reference,
    luisa::compute::Float3 shading_normal,
    luisa::compute::Bool has_transmission,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Bool sphere,
    luisa::compute::Float3 axis_x,
    luisa::compute::Float3 axis_y,
    luisa::compute::Float3 axis_z,
    luisa::compute::Float3 axis_scale,
    luisa::compute::Float spot_angle,
    luisa::compute::Float spot_smooth,
    luisa::compute::Float2 random,
    luisa::compute::Bool normalize_power) noexcept {
    using namespace luisa::compute;

    const auto center_offset =
        reference - center;
    const auto center_distance_squared =
        dot(center_offset, center_offset);
    const auto radius_squared =
        radius * radius;
    const auto light_normal = safe_normalize(
        center_offset,
        make_float3(0.0f, 0.0f, -1.0f));
    const auto sphere_cap =
        cycles_sample_mapping::
            sin_squared_to_one_minus_cosine(
                radius_squared /
                max(
                    center_distance_squared,
                    1.0e-30f));
    const auto spot_cap =
        spot_one_minus_cosine_larger_spread(
            spot_angle,
            axis_scale);
    // Cycles selects the light's visible cap only when it is strictly
    // narrower. Equality belongs to the spread-cone branch.
    const auto sample_spread_cone =
        (center_distance_squared >
         radius_squared) &
        !(sphere_cap < spot_cap);
    const auto outside_axis = select(
        -light_normal,
        axis_z,
        sample_spread_cone);
    const auto outside_cap = select(
        sphere_cap,
        spot_cap,
        sample_spread_cone);

    FiniteLightGeometrySample geometry{
        .valid = false,
        .direction = make_float3(0.0f),
        .position = center,
        .normal = make_float3(0.0f),
        .distance = 0.0f,
        .conditional_pdf = 0.0f};
    const auto finite_sphere =
        sphere & (radius > 0.0f);
    $if (finite_sphere) {
        geometry = sample_sphere_geometry(
            reference,
            shading_normal,
            has_transmission,
            center,
            radius,
            outside_axis,
            outside_cap,
            sample_spread_cone,
            random);
    }
    $else {
        geometry = sample_disk_geometry(
            reference,
            center,
            radius,
            random);
    };

    const auto transform =
        light_linear_transform(
            axis_x,
            axis_y,
            axis_z,
            axis_scale);
    const auto local_ray =
        spot_light_to_local(
            -geometry.direction,
            transform);
    const auto attenuation =
        spot_light_attenuation(
            local_ray,
            spot_angle,
            spot_smooth);
    const auto use_attenuation =
        !finite_sphere |
        (center_distance_squared >
         radius_squared);
    const auto evaluation_factor =
        point_eval_factor(
            radius,
            normalize_power) *
        select(
            1.0f,
            attenuation,
            use_attenuation);
    return {
        .valid =
            geometry.valid &
            (evaluation_factor != 0.0f),
        .direction = geometry.direction,
        .position = geometry.position,
        .normal = geometry.normal,
        .uv = spot_light_uv(
            local_ray,
            spot_angle),
        .distance = geometry.distance,
        .conditional_pdf =
            geometry.conditional_pdf,
        .evaluation_factor =
            evaluation_factor};
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
