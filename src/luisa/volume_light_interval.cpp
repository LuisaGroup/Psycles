#include <psycles/luisa/volume_light_interval.h>

#include <psycles/luisa/analytic_light_sampling.h>

#include <limits>

namespace psycles::luisa_backend {

using namespace luisa::compute;

namespace {

struct QuadraticRoots {
    Float minimum;
    Float maximum;
    Bool valid;
};

[[nodiscard]] Float scalar_min(
    Float first,
    Float second) noexcept {
    return select(
        first,
        second,
        second < first);
}

[[nodiscard]] Float scalar_max(
    Float first,
    Float second) noexcept {
    return select(
        first,
        second,
        second > first);
}

[[nodiscard]] VolumeDirectSampleInterval
select_interval(
    const VolumeDirectSampleInterval
        &false_interval,
    const VolumeDirectSampleInterval
        &true_interval,
    Bool predicate) noexcept {
    return {
        .minimum =
            select(
                false_interval.minimum,
                true_interval.minimum,
                predicate),
        .maximum =
            select(
                false_interval.maximum,
                true_interval.maximum,
                predicate)};
}

[[nodiscard]] QuadraticRoots solve_quadratic(
    Float a,
    Float b,
    Float c) noexcept {
    const auto valid_linear =
        (a == 0.0f) & (b != 0.0f);
    const auto safe_b =
        select(1.0f, b, b != 0.0f);
    const auto linear_root = -c / safe_b;

    const auto discriminant =
        b * b - 4.0f * a * c;
    const auto valid_quadratic =
        (a != 0.0f) &
        (discriminant > -1.0e-5f);
    const auto square_root =
        sqrt(
            scalar_max(
                discriminant,
                0.0f));
    const Float temporary =
        -0.5f *
        (b + copysign(square_root, b));
    const auto safe_a =
        select(1.0f, a, a != 0.0f);
    const auto temporary_valid =
        temporary != 0.0f;
    const auto safe_temporary =
        select(
            1.0f,
            temporary,
            temporary_valid);
    const Float root_1 =
        temporary / safe_a;
    const Float root_2 =
        select(
            root_1,
            c / safe_temporary,
            temporary_valid);
    const auto quadratic_minimum =
        select(
            root_1,
            root_2,
            root_2 < root_1);
    const auto quadratic_maximum =
        select(
            root_1,
            root_2,
            root_2 > root_1);
    return {
        .minimum = select(
            linear_root,
            quadratic_minimum,
            valid_quadratic),
        .maximum = select(
            linear_root,
            quadratic_maximum,
            valid_quadratic),
        .valid =
            valid_linear |
            valid_quadratic};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_intervals(
    const VolumeDirectSampleInterval &first,
    const VolumeDirectSampleInterval &second,
    Bool valid) noexcept {
    const VolumeDirectSampleInterval interval{
        .minimum =
            scalar_max(
                first.minimum,
                second.minimum),
        .maximum =
            scalar_min(
                first.maximum,
                second.maximum)};
    return {
        .interval = interval,
        .valid =
            valid &
            (interval.minimum <
             interval.maximum)};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_plane(
    Float3 normal,
    Float3 plane_to_origin,
    Float3 ray_direction,
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto direction_cosine =
        dot(ray_direction, normal);
    const auto numerator =
        -dot(plane_to_origin, normal);
    const auto distance =
        numerator / direction_cosine;
    const auto minimum =
        select(
            interval.minimum,
            scalar_max(
                interval.minimum,
                distance),
            direction_cosine > 0.0f);
    const auto maximum =
        select(
            scalar_min(
                interval.maximum,
                distance),
            interval.maximum,
            direction_cosine > 0.0f);
    const auto result =
        VolumeDirectSampleInterval{
            .minimum = minimum,
            .maximum = maximum};
    return {
        .interval = result,
        .valid =
            result.minimum <
            result.maximum};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_slab(
    Float minimum_bound,
    Float maximum_bound,
    Float origin,
    Float direction,
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto moving =
        direction != 0.0f;
    const auto safe_direction =
        select(1.0f, direction, moving);
    const auto lower =
        (minimum_bound - origin) /
        safe_direction;
    const auto upper =
        (maximum_bound - origin) /
        safe_direction;
    const auto bounded =
        intersect_intervals(
            interval,
            {.minimum =
                 scalar_min(lower, upper),
             .maximum =
                 scalar_max(lower, upper)},
            moving);
    const auto parallel_inside =
        !moving &
        (origin >= minimum_bound) &
        (origin <= maximum_bound) &
        (interval.minimum <
         interval.maximum);
    return {
        .interval = select_interval(
            bounded.interval,
            interval,
            !moving),
        .valid = select(
            bounded.valid,
            parallel_inside,
            !moving)};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_aabb(
    Float3 minimum_bound,
    Float3 maximum_bound,
    Float3 origin,
    Float3 direction,
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto x =
        intersect_slab(
            minimum_bound.x,
            maximum_bound.x,
            origin.x,
            direction.x,
            interval);
    const auto y =
        intersect_slab(
            minimum_bound.y,
            maximum_bound.y,
            origin.y,
            direction.y,
            x.interval);
    const auto z =
        intersect_slab(
            minimum_bound.z,
            maximum_bound.z,
            origin.z,
            direction.z,
            y.interval);
    return {
        .interval = z.interval,
        .valid =
            x.valid &
            y.valid &
            z.valid};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_infinite_elliptic_cylinder(
    Float3 origin,
    Float3 direction,
    Float radius_u,
    Float radius_v,
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto valid_radii =
        (radius_u > 0.0f) &
        (radius_v > 0.0f);
    const auto safe_u =
        select(1.0f, radius_u, valid_radii);
    const auto safe_v =
        select(1.0f, radius_v, valid_radii);
    auto projected_origin =
        make_float2(
            origin.x / safe_u,
            origin.y / safe_v);
    const auto projected_direction =
        make_float2(
            direction.x / safe_u,
            direction.y / safe_v);
    const auto a =
        dot(
            projected_direction,
            projected_direction);
    auto b =
        dot(
            projected_origin,
            projected_direction);
    const auto movable = a != 0.0f;
    const auto safe_a =
        select(1.0f, a, movable);
    const auto middle = -b / safe_a;
    projected_origin +=
        projected_direction * middle;
    b = dot(
        projected_origin,
        projected_direction);
    const auto c =
        dot(
            projected_origin,
            projected_origin) -
        1.0f;
    const auto roots =
        solve_quadratic(
            a, 2.0f * b, c);
    const auto bounded =
        intersect_intervals(
            interval,
            {.minimum =
                 roots.minimum + middle,
             .maximum =
                 roots.maximum + middle},
            roots.valid);
    return {
        .interval = bounded.interval,
        .valid =
            valid_radii &
            movable &
            bounded.valid};
}

[[nodiscard]] VolumeLightIntervalResult
intersect_cone(
    Float3 axis,
    Float3 apex_to_origin,
    Float3 direction,
    Float cosine_angle_squared,
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto plane_limit =
        cosine_angle_squared <
        1.0e-4f;
    const auto plane =
        intersect_plane(
            axis,
            apex_to_origin,
            direction,
            interval);

    const auto direction_length_squared =
        dot(direction, direction);
    const auto valid_direction =
        direction_length_squared > 0.0f;
    const auto inverse_length =
        rsqrt(
            scalar_max(
                direction_length_squared,
                1.0e-30f));
    const auto normalized_direction =
        direction * inverse_length;
    const auto axis_direction =
        dot(axis, normalized_direction);
    const auto axis_origin =
        dot(axis, apex_to_origin);
    const auto a =
        axis_direction *
            axis_direction -
        cosine_angle_squared;
    const auto b =
        2.0f *
        (axis_direction *
             axis_origin -
         cosine_angle_squared *
             dot(
                 normalized_direction,
                 apex_to_origin));
    const auto c =
        axis_origin * axis_origin -
        cosine_angle_squared *
            dot(
                apex_to_origin,
                apex_to_origin);
    const auto roots =
        solve_quadratic(a, b, c);
    const auto minimum_valid =
        axis_origin +
            roots.minimum *
                axis_direction >
        0.0f;
    const auto maximum_valid =
        axis_origin +
            roots.maximum *
                axis_direction >
        0.0f;
    const auto neither_maximum =
        !maximum_valid;
    const auto only_maximum =
        maximum_valid &
        !minimum_valid;
    const auto clipped_minimum =
        select(
            roots.minimum,
            0.0f,
            neither_maximum);
    const auto hemisphere_minimum =
        select(
            clipped_minimum,
            roots.maximum,
            only_maximum);
    const auto clipped_maximum =
        select(
            roots.maximum,
            roots.minimum,
            neither_maximum);
    const auto hemisphere_maximum =
        select(
            clipped_maximum,
            std::numeric_limits<
                float>::max(),
            only_maximum);
    const auto cone =
        intersect_intervals(
            interval,
            {.minimum =
                 hemisphere_minimum *
                 inverse_length,
             .maximum =
                 hemisphere_maximum *
                 inverse_length},
            valid_direction &
                roots.valid &
                (minimum_valid |
                 maximum_valid));
    return {
        .interval =
            select_interval(
                cone.interval,
                plane.interval,
                plane_limit),
        .valid =
            select(
                cone.valid,
                plane.valid,
                plane_limit)};
}

}// namespace

VolumeLightIntervalResult
VolumeLightInterval::spot(
    const VolumeSpotIntervalInput
        &input) const noexcept {
    namespace sampling =
        analytic_light_sampling;
    const auto transform =
        sampling::light_linear_transform(
            input.axis_x,
            input.axis_y,
            input.axis_z,
            input.axis_scale);
    const auto half_angle =
        0.5f * input.spot_angle;
    const auto tangent =
        tan(half_angle);
    const auto minimum_transverse_scale =
        scalar_min(
            input.axis_scale.x *
                input.axis_scale.x,
            input.axis_scale.y *
                input.axis_scale.y);
    const auto axial_scale =
        input.axis_scale.z *
        input.axis_scale.z;
    const auto denominator =
        tangent * tangent *
        minimum_transverse_scale;
    const auto geometry_valid =
        transform.valid &
        (denominator > 0.0f);
    const auto safe_denominator =
        select(
            1.0f,
            denominator,
            geometry_valid);
    const auto apex_shift =
        input.radius *
        sqrt(
            1.0f +
            axial_scale /
                safe_denominator);
    const auto emission_axis =
        -input.axis_z;
    const auto shifted_origin =
        input.ray_origin +
        emission_axis * apex_shift -
        input.center;
    const auto local_origin =
        sampling::
            world_to_light_direction(
                shifted_origin,
                transform);
    const auto local_direction =
        sampling::
            world_to_light_direction(
                input.ray_direction,
                transform);
    auto result =
        intersect_cone(
            make_float3(
                0.0f, 0.0f, -1.0f),
            local_origin,
            local_direction,
            cos(half_angle) *
                cos(half_angle),
            input.interval);
    result.valid &=
        geometry_valid;
    return result;
}

VolumeLightIntervalResult
VolumeLightInterval::area(
    const VolumeAreaIntervalInput
        &input) const noexcept {
    namespace sampling =
        analytic_light_sampling;
    const auto normal =
        -input.axis_z;
    const auto origin =
        input.ray_origin -
        input.center;
    const auto local_origin =
        make_float3(
            dot(origin, input.axis_u),
            dot(origin, input.axis_v),
            dot(origin, normal));
    const auto local_direction =
        make_float3(
            dot(
                input.ray_direction,
                input.axis_u),
            dot(
                input.ray_direction,
                input.axis_v),
            dot(
                input.ray_direction,
                normal));
    const auto half_u =
        0.5f * input.length_u;
    const auto half_v =
        0.5f * input.length_v;
    const auto tangent =
        select(
            tan(
                0.5f *
                scalar_max(
                    input.spread,
                    0.0f)),
            std::numeric_limits<
                float>::max(),
            input.spread ==
                sampling::pi);
    const auto nearly_parallel =
        tangent < 1.0e-5f;
    const auto cylinder =
        intersect_infinite_elliptic_cylinder(
            local_origin,
            local_direction,
            half_u,
            half_v,
            input.interval);
    const auto box =
        intersect_aabb(
            make_float3(
                -half_u,
                -half_v,
                0.0f),
            make_float3(
                half_u,
                half_v,
                std::numeric_limits<
                    float>::max()),
            local_origin,
            local_direction,
            input.interval);
    const auto zero_spread =
        VolumeLightIntervalResult{
            .interval =
                select_interval(
                    box.interval,
                    cylinder.interval,
                    input.ellipse),
            .valid =
                select(
                    box.valid,
                    cylinder.valid,
                    input.ellipse)};

    const auto maximum_extent =
        0.5f *
        select(
            sqrt(
                input.length_u *
                    input.length_u +
                input.length_v *
                    input.length_v),
            scalar_max(
                input.length_u,
                input.length_v),
            input.ellipse);
    const auto safe_tangent =
        select(
            1.0f,
            tangent,
            tangent != 0.0f);
    const auto apex_to_origin =
        origin +
        maximum_extent /
            safe_tangent *
            normal;
    const auto cone =
        intersect_cone(
            normal,
            apex_to_origin,
            input.ray_direction,
            1.0f /
                (1.0f +
                 tangent * tangent),
            input.interval);
    const auto spread_region =
        VolumeLightIntervalResult{
            .interval =
                select_interval(
                    cone.interval,
                    zero_spread.interval,
                    nearly_parallel),
            .valid =
                select(
                    cone.valid,
                    zero_spread.valid,
                    nearly_parallel)};
    const auto positive_side =
        intersect_plane(
            normal,
            origin,
            input.ray_direction,
            spread_region.interval);
    return {
        .interval =
            positive_side.interval,
        .valid =
            spread_region.valid &
            positive_side.valid};
}

VolumeLightIntervalResult
VolumeLightInterval::triangle(
    const VolumeTriangleIntervalInput
        &input) const noexcept {
    const auto samples_both =
        input.sample_front &
        input.sample_back;
    const auto samples_one =
        input.sample_front ^
        input.sample_back;
    const auto oriented_normal =
        select(
            -input.normal,
            input.normal,
            input.sample_front);
    const auto clipped =
        intersect_plane(
            oriented_normal,
            input.ray_origin -
                input.plane_point,
            input.ray_direction,
            input.interval);
    return {
        .interval =
            select_interval(
                clipped.interval,
                input.interval,
                samples_both),
        .valid =
            select(
                clipped.valid &
                    samples_one,
                input.interval.minimum <
                    input.interval.maximum,
                samples_both)};
}

}// namespace psycles::luisa_backend
