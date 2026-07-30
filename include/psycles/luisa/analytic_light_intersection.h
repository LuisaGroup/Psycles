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

struct PointIntersection {
    luisa::compute::Bool valid;
    luisa::compute::Float distance;
    luisa::compute::Float3 position;
    luisa::compute::Float3 normal;
    luisa::compute::Float2 uv;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Float evaluation_factor;
};

[[nodiscard]] inline PointIntersection
intersect_point_geometry(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Bool sphere,
    luisa::compute::Float3 previous_normal,
    luisa::compute::Bool had_transmission,
    luisa::compute::Float outside_one_minus_cosine) noexcept {
    using namespace luisa::compute;
    namespace sampling =
        analytic_light_sampling;

    const auto radius_squared =
        radius * radius;
    const auto center_offset =
        ray_origin - center;
    const auto center_distance_squared =
        dot(center_offset, center_offset);
    const auto light_normal =
        sampling::safe_normalize(
            center_offset,
            -ray_direction);
    sampling::RayPrimitiveIntersection geometry{
        .valid = false,
        .distance = 0.0f,
        .position = ray_origin};
    Float3 normal = -ray_direction;
    Float conditional_pdf = 0.0f;
    $if (sphere) {
        geometry =
            sampling::intersect_sphere(
                ray_origin,
                ray_direction,
                ray_minimum,
                ray_maximum,
                center,
                radius);
        normal = sampling::safe_normalize(
            geometry.position - center,
            -ray_direction);
        conditional_pdf =
            sampling::sphere_light_pdf(
                center_distance_squared,
                radius_squared,
                previous_normal,
                ray_direction,
                had_transmission,
                outside_one_minus_cosine);
    }
    $else {
        geometry =
            sampling::intersect_disk(
                ray_origin,
                ray_direction,
                ray_minimum,
                ray_maximum,
                center,
                light_normal,
                radius);
        const auto light_cosine =
            dot(light_normal, -ray_direction);
        conditional_pdf =
            sampling::point_disk_pdf(
                geometry.distance *
                    geometry.distance,
                light_cosine,
                radius);
    };
    return {
        .valid =
            geometry.valid &
            (radius > 0.0f),
        .distance = geometry.distance,
        .position = geometry.position,
        .normal = normal,
        .uv = make_float2(0.0f),
        .conditional_pdf =
            conditional_pdf,
        .evaluation_factor = 0.0f};
}

[[nodiscard]] inline PointIntersection
intersect_point(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Bool sphere,
    luisa::compute::Float3 axis_x,
    luisa::compute::Float3 axis_y,
    luisa::compute::Float3 axis_z,
    luisa::compute::Float3 axis_scale,
    luisa::compute::Bool normalize_power,
    luisa::compute::Float3 previous_normal,
    luisa::compute::Bool had_transmission) noexcept {
    using namespace luisa::compute;
    namespace sampling =
        analytic_light_sampling;

    const auto center_distance_squared =
        dot(ray_origin - center,
            ray_origin - center);
    const auto sphere_cap =
        cycles_sample_mapping::
            sin_squared_to_one_minus_cosine(
                radius * radius /
                max(
                    center_distance_squared,
                    1.0e-30f));
    auto result =
        intersect_point_geometry(
            ray_origin,
            ray_direction,
            ray_minimum,
            ray_maximum,
            center,
            radius,
            sphere,
            previous_normal,
            had_transmission,
            sphere_cap);
    const auto transform =
        sampling::light_linear_transform(
            axis_x,
            axis_y,
            axis_z,
            axis_scale);
    result.uv =
        sampling::point_light_uv(
            result.normal,
            transform);
    result.evaluation_factor =
        sampling::point_eval_factor(
            radius,
            normalize_power);
    result.valid &=
        result.evaluation_factor > 0.0f;
    return result;
}

[[nodiscard]] inline PointIntersection
intersect_spot(
    luisa::compute::Float3 ray_origin,
    luisa::compute::Float3 ray_direction,
    luisa::compute::Float ray_minimum,
    luisa::compute::Float ray_maximum,
    luisa::compute::Float3 center,
    luisa::compute::Float radius,
    luisa::compute::Bool sphere,
    luisa::compute::Float3 axis_x,
    luisa::compute::Float3 axis_y,
    luisa::compute::Float3 axis_z,
    luisa::compute::Float3 axis_scale,
    luisa::compute::Float spot_angle,
    luisa::compute::Float spot_smooth,
    luisa::compute::Bool normalize_power,
    luisa::compute::Float3 previous_normal,
    luisa::compute::Bool had_transmission) noexcept {
    using namespace luisa::compute;
    namespace sampling =
        analytic_light_sampling;

    const auto center_distance_squared =
        dot(ray_origin - center,
            ray_origin - center);
    const auto radius_squared =
        radius * radius;
    const auto sphere_cap =
        cycles_sample_mapping::
            sin_squared_to_one_minus_cosine(
                radius_squared /
                max(
                    center_distance_squared,
                    1.0e-30f));
    const auto spot_cap =
        sampling::
            spot_one_minus_cosine_larger_spread(
                spot_angle,
                axis_scale);
    auto result =
        intersect_point_geometry(
            ray_origin,
            ray_direction,
            ray_minimum,
            ray_maximum,
            center,
            radius,
            sphere,
            previous_normal,
            had_transmission,
            luisa::compute::min(
                sphere_cap, spot_cap));
    const auto transform =
        sampling::light_linear_transform(
            axis_x,
            axis_y,
            axis_z,
            axis_scale);
    const auto local_ray =
        sampling::spot_light_to_local(
            -ray_direction,
            transform);
    result.uv =
        sampling::spot_light_uv(
            local_ray,
            spot_angle);
    const auto use_attenuation =
        !sphere |
        (center_distance_squared >
         radius_squared);
    const auto attenuation =
        sampling::spot_light_attenuation(
            local_ray,
            spot_angle,
            spot_smooth);
    result.evaluation_factor =
        sampling::point_eval_factor(
            radius,
            normalize_power) *
        select(
            1.0f,
            attenuation,
            use_attenuation);
    result.valid &=
        (dot(
             ray_direction,
             ray_origin - center) <
         0.0f) &
        (result.evaluation_factor > 0.0f);
    return result;
}

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
