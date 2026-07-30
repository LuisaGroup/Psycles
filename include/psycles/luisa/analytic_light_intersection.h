#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/analytic_light_intersection.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/analytic_light_sampling.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::analytic_light_intersection {

struct AreaIntersection {
    luisa::compute::Bool valid;
    luisa::compute::Float distance;
    luisa::compute::Float3 position;
    luisa::compute::Float3 normal;
    luisa::compute::Float2 uv;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Float evaluation_factor;
};

// Intersect and evaluate one Cycles area-light primitive in a common
// directional measure. Intersection is strictly inside the ray interval,
// matching ray_quad_intersect. A full-spread rectangle uses its solid-angle
// PDF; ellipse and currently unclamped narrow-spread shapes use the
// area-to-solid-angle Jacobian.
[[nodiscard]] inline AreaIntersection
intersect_area(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float3 axis_u,
    luisa::compute::Float length_u,
    luisa::compute::Float3 axis_v,
    luisa::compute::Float length_v,
    luisa::compute::Float3 axis_z,
    luisa::compute::Bool ellipse,
    luisa::compute::Bool full_spread,
    luisa::compute::Float spread,
    luisa::compute::Bool normalize_power) noexcept {
    using namespace luisa::compute;
    namespace sampling =
        analytic_light_sampling;

    const auto normal = -axis_z;
    const auto direction_normal =
        dot(ray_direction, normal);
    const auto distance = sampling::safe_divide(
        dot(center - ray_origin, normal),
        direction_normal);
    const auto position =
        ray_origin + distance * ray_direction;
    const auto inplane = position - center;
    const auto u = sampling::safe_divide(
        dot(inplane, axis_u), length_u);
    const auto v = sampling::safe_divide(
        dot(inplane, axis_v), length_v);
    const auto inside_rectangle =
        (u >= -0.5f) & (u <= 0.5f) &
        (v >= -0.5f) & (v <= 0.5f);
    const auto inside_shape =
        inside_rectangle &
        (!ellipse |
         (u * u + v * v <= 0.25f));

    auto area = length_u * length_v;
    area *= select(
        1.0f,
        0.25f * sampling::pi,
        ellipse);
    const auto positive_area = area > 0.0f;
    area = max(area, 1.0e-20f);
    const auto cosine = max(
        dot(normal, -ray_direction),
        0.0f);
    const auto area_pdf =
        distance * distance /
        max(cosine * area, 1.0e-20f);
    const auto solid_angle_rectangle =
        !ellipse & full_spread;
    Float conditional_pdf = area_pdf;
    $if (solid_angle_rectangle) {
        conditional_pdf =
            sampling::rectangle_solid_angle_pdf(
                ray_origin,
                center,
                axis_u,
                length_u,
                axis_v,
                length_v);
    };

    const auto inverse_area = select(
        1.0f,
        1.0f / area,
        normalize_power);
    const auto spread_attenuation =
        sampling::area_spread_attenuation(
            ray_direction,
            normal,
            spread);
    const auto evaluation_factor =
        inverse_area *
        sampling::inverse_pi *
        spread_attenuation;
    const auto valid =
        (direction_normal < 0.0f) &
        (distance > ray_minimum) &
        (distance < ray_maximum) &
        positive_area &
        inside_shape &
        (evaluation_factor > 0.0f);
    return {
        .valid = valid,
        .distance = distance,
        .position = position,
        .normal = normal,
        // Cycles exposes lamp UV in Embree/OptiX barycentric notation.
        .uv = make_float2(v + 0.5f, -u - v),
        .conditional_pdf = conditional_pdf,
        .evaluation_factor = evaluation_factor};
}

}// namespace psycles::luisa_backend::analytic_light_intersection
