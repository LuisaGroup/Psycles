#include <psycles/luisa/area_light_sampling.h>

#include <limits>

namespace psycles::luisa_backend {

using namespace luisa::compute;

namespace sampling =
    analytic_light_sampling;

namespace {

struct AreaSamplingShape {
    Bool valid;
    Float3 center;
    Float3 axis_u;
    Float3 axis_v;
    Float length_u;
    Float length_v;
    Bool rectangle;
};

[[nodiscard]] AreaSamplingShape
original_shape(
    const AreaLightSampleInput
        &input) noexcept {
    return {
        .valid = true,
        .center = input.center,
        .axis_u = input.axis_u,
        .axis_v = input.axis_v,
        .length_u = input.length_u,
        .length_v = input.length_v,
        .rectangle = !input.ellipse};
}

[[nodiscard]] Float
area_value(
    const AreaLightSampleInput
        &input) noexcept {
    auto area =
        input.length_u *
        input.length_v;
    area *= select(
        1.0f,
        0.25f * sampling::pi,
        input.ellipse);
    return area;
}

[[nodiscard]] Float
inverse_area(
    const AreaLightSampleInput
        &input) noexcept {
    const auto area =
        area_value(input);
    return select(
        1.0f,
        sampling::safe_divide(
            1.0f, area),
        input.normalize_power &
            (area != 0.0f));
}

[[nodiscard]] Float
tangent_half_spread(
    Float spread) noexcept {
    return select(
        tan(0.5f *
            luisa::compute::max(
                spread, 0.0f)),
        std::numeric_limits<
            float>::max(),
        spread == sampling::pi);
}

[[nodiscard]] Float3
sample_rectangle(
    Float3 center,
    Float3 axis_u,
    Float length_u,
    Float3 axis_v,
    Float length_v,
    Float2 random) noexcept {
    return center +
           axis_u *
               (length_u *
                (random.x - 0.5f)) +
           axis_v *
               (length_v *
                (random.y - 0.5f));
}

[[nodiscard]] Float3
sample_ellipse(
    Float3 center,
    Float3 axis_u,
    Float length_u,
    Float3 axis_v,
    Float length_v,
    Float2 random) noexcept {
    const auto disk =
        cycles_sample_mapping::
            sample_uniform_disk(random);
    return center +
           axis_u *
               (0.5f * length_u *
                disk.x) +
           axis_v *
               (0.5f * length_v *
                disk.y);
}

struct DirectionToLight {
    Float3 direction;
    Float distance;
    Float distance_squared;
};

[[nodiscard]] DirectionToLight
direction_to_light(
    Float3 reference,
    Float3 position) noexcept {
    const auto offset =
        position - reference;
    const auto distance_squared =
        dot(offset, offset);
    const auto distance =
        sqrt(luisa::compute::max(
            distance_squared,
            0.0f));
    return {
        .direction =
            offset /
            luisa::compute::max(
                distance, 1.0e-30f),
        .distance = distance,
        .distance_squared =
            distance_squared};
}

[[nodiscard]] Float
area_to_solid_angle(
    Float3 normal,
    const DirectionToLight
        &direction) noexcept {
    const auto cosine =
        dot(
            normal,
            -direction.direction);
    return select(
        0.0f,
        direction.distance_squared /
            cosine,
        cosine > 0.0f);
}

[[nodiscard]] Float2
area_uv(
    const AreaLightSampleInput
        &input,
    Float3 position) noexcept {
    const auto inplane =
        position - input.center;
    const auto u =
        sampling::safe_divide(
            dot(inplane, input.axis_u),
            input.length_u);
    const auto v =
        sampling::safe_divide(
            dot(inplane, input.axis_v),
            input.length_v);
    return make_float2(
        v + 0.5f,
        -u - v);
}

[[nodiscard]] AreaSamplingShape
clamp_to_spread(
    const AreaLightSampleInput
        &input,
    Float3 normal,
    Float tangent) noexcept {
    auto result =
        original_shape(input);

    const auto plane_distance =
        dot(
            normal,
            input.reference -
                input.center);
    const auto closest =
        input.reference -
        plane_distance * normal;
    const auto spread_radius =
        plane_distance * tangent;
    const auto spread_u =
        dot(
            input.axis_u,
            closest - input.center);
    const auto spread_v =
        dot(
            input.axis_v,
            closest - input.center);
    const auto round =
        (!result.rectangle) &
        (result.length_u ==
         result.length_v);
    Bool sample_spread =
        spread_radius == 0.0f;
    Bool resolved = false;

    $if(round & (!sample_spread)) {
        const auto center_distance =
            length(make_float2(
                spread_u,
                spread_v));
        const auto light_radius =
            0.5f * result.length_u;
        $if(center_distance >=
            light_radius +
                spread_radius) {
            result.valid = false;
        }
        $else {
            sample_spread =
                (center_distance <=
                 abs(light_radius -
                     spread_radius)) &
                (spread_radius <
                 light_radius);
            $if(center_distance >
                abs(light_radius -
                    spread_radius)) {
                const auto rectangle_u =
                    light_radius +
                    spread_radius -
                    center_distance;
                Float rectangle_v =
                    0.0f;
                $if(abs(
                        light_radius *
                            light_radius -
                        spread_radius *
                            spread_radius) >=
                    center_distance *
                        center_distance) {
                    rectangle_v =
                        2.0f *
                        luisa::compute::min(
                            light_radius,
                            spread_radius);
                }
                $else {
                    const auto offset =
                        center_distance +
                        (spread_radius *
                             spread_radius -
                         light_radius *
                             light_radius) /
                            center_distance;
                    rectangle_v =
                        sqrt(
                            4.0f *
                                spread_radius *
                                spread_radius -
                            offset * offset);
                };
                const auto rectangle_area =
                    rectangle_u *
                    rectangle_v;
                const auto light_area =
                    sampling::pi *
                    light_radius *
                    light_radius;
                const auto spread_area =
                    sampling::pi *
                    spread_radius *
                    spread_radius;
                $if(rectangle_area <
                    luisa::compute::min(
                        light_area,
                        spread_area)) {
                    result.rectangle =
                        true;
                    result.axis_u =
                        sampling::safe_normalize(
                            input.center -
                                closest,
                            input.axis_u);
                    result.axis_v =
                        cross(
                            normal,
                            result.axis_u);
                    result.length_u =
                        rectangle_u;
                    result.length_v =
                        rectangle_v;
                    result.center =
                        0.5f *
                        (input.center +
                         closest +
                         result.axis_u *
                             (spread_radius -
                              light_radius));
                    resolved = true;
                }
                $else {
                    sample_spread =
                        spread_area <
                        light_area;
                };
            };
        };
    }
    $elif((!round) & (!sample_spread)) {
        const auto minimum_u =
            luisa::compute::max(
                spread_u -
                    spread_radius,
                -0.5f *
                    result.length_u);
        const auto maximum_u =
            luisa::compute::min(
                spread_u +
                    spread_radius,
                0.5f *
                    result.length_u);
        const auto minimum_v =
            luisa::compute::max(
                spread_v -
                    spread_radius,
                -0.5f *
                    result.length_v);
        const auto maximum_v =
            luisa::compute::min(
                spread_v +
                    spread_radius,
                0.5f *
                    result.length_v);
        $if((minimum_u >= maximum_u) |
            (minimum_v >= maximum_v)) {
            result.valid = false;
        }
        $else {
            const auto rectangle_u =
                maximum_u - minimum_u;
            const auto rectangle_v =
                maximum_v - minimum_v;
            const auto rectangle_area =
                rectangle_u *
                rectangle_v;
            const auto ellipse_area =
                select(
                    0.25f *
                        sampling::pi *
                        result.length_u *
                        result.length_v,
                    std::numeric_limits<
                        float>::max(),
                    result.rectangle);
            const auto spread_area =
                sampling::pi *
                spread_radius *
                spread_radius;
            $if(result.rectangle |
                (rectangle_area <
                 luisa::compute::min(
                     ellipse_area,
                     spread_area))) {
                const auto center_u =
                    0.5f *
                    (minimum_u +
                     maximum_u);
                const auto center_v =
                    0.5f *
                    (minimum_v +
                     maximum_v);
                result.center +=
                    result.axis_u *
                        center_u +
                    result.axis_v *
                        center_v;
                result.length_u =
                    rectangle_u;
                result.length_v =
                    rectangle_v;
                result.rectangle = true;
                resolved = true;
            }
            $else {
                result.rectangle = false;
                sample_spread =
                    spread_area <
                    ellipse_area;
            };
        };
    };

    $if(result.valid &
        (!resolved) &
        sample_spread) {
        result.rectangle = false;
        result.center =
            input.center +
            result.axis_u * spread_u +
            result.axis_v * spread_v;
        result.length_u =
            2.0f * spread_radius;
        result.length_v =
            2.0f * spread_radius;
    };
    return result;
}

[[nodiscard]] Float
evaluation_factor(
    const AreaLightSampleInput
        &input,
    Float3 direction,
    Float3 normal) noexcept {
    return sampling::inverse_pi *
           inverse_area(input) *
           sampling::
               area_spread_attenuation(
                   direction,
                   normal,
                   input.spread);
}

}// namespace

sampling::FiniteLightSample
AreaLightSampling::from_segment(
    const AreaLightSampleInput
        &input) const noexcept {
    const auto rectangle =
        !input.ellipse;
    Float3 position =
        input.center;
    $if(rectangle) {
        position = sample_rectangle(
            input.center,
            input.axis_u,
            input.length_u,
            input.axis_v,
            input.length_v,
            input.random);
    }
    $else {
        position = sample_ellipse(
            input.center,
            input.axis_u,
            input.length_u,
            input.axis_v,
            input.length_v,
            input.random);
    };
    const auto direction =
        direction_to_light(
            input.reference,
            position);
    const auto normal =
        -input.axis_z;
    const auto conditional_pdf =
        inverse_area(input) *
        area_to_solid_angle(
            normal,
            direction);
    return {
        // Cycles' in-volume contract preserves the proposal even when its
        // directional PDF or spread attenuation is zero. The independently
        // clipped segment decides whether a contributing collision exists.
        .valid = true,
        .direction =
            direction.direction,
        .position = position,
        .normal = normal,
        .uv = area_uv(
            input, position),
        .distance =
            direction.distance,
        .conditional_pdf =
            conditional_pdf,
        .evaluation_factor =
            evaluation_factor(
                input,
                direction.direction,
                normal)};
}

sampling::FiniteLightSample
AreaLightSampling::from_position(
    const AreaLightSampleInput
        &input) const noexcept {
    sampling::FiniteLightSample result{
        .valid = false,
        .direction =
            make_float3(0.0f),
        .position = input.center,
        .normal = -input.axis_z,
        .uv = make_float2(0.0f),
        .distance = 0.0f,
        .conditional_pdf = 0.0f,
        .evaluation_factor = 0.0f};
    const auto normal =
        -input.axis_z;
    const auto front_facing =
        dot(
            input.center -
                input.reference,
            normal) <= 0.0f;
    $if(front_facing) {
        const auto tangent =
            tangent_half_spread(
                input.spread);
        auto shape =
            original_shape(input);
        $if(!input.full_spread) {
            shape =
                clamp_to_spread(
                    input,
                    normal,
                    tangent);
        };
        $if(shape.valid) {
            Float3 position =
                shape.center;
            Float conditional_pdf =
                0.0f;
            $if(shape.rectangle) {
                const auto rectangle =
                    sampling::
                        sample_rectangle_solid_angle(
                            input.reference,
                            shape.center,
                            shape.axis_u,
                            shape.length_u,
                            shape.axis_v,
                            shape.length_v,
                            input.random);
                position =
                    rectangle.position;
                conditional_pdf =
                    rectangle.pdf;
            }
            $else {
                $if(tangent == 0.0f) {
                    conditional_pdf =
                        1.0f;
                }
                $else {
                    position =
                        sample_ellipse(
                            shape.center,
                            shape.axis_u,
                            shape.length_u,
                            shape.axis_v,
                            shape.length_v,
                            input.random);
                    conditional_pdf =
                        4.0f *
                        sampling::inverse_pi /
                        (shape.length_u *
                         shape.length_v);
                };
            };

            const auto direction =
                direction_to_light(
                    input.reference,
                    position);
            $if((!shape.rectangle) &
                (tangent > 0.0f)) {
                conditional_pdf *=
                    area_to_solid_angle(
                        normal,
                        direction);
            };
            const auto factor =
                evaluation_factor(
                    input,
                    direction.direction,
                    normal);

            const auto inplane =
                position -
                input.center;
            const auto light_u =
                dot(
                    inplane,
                    input.axis_u);
            const auto light_v =
                dot(
                    inplane,
                    input.axis_v);
            const auto epsilon_u =
                (0.5f + 1.0e-7f) *
                    input.length_u +
                1.0e-6f;
            const auto epsilon_v =
                (0.5f + 1.0e-7f) *
                    input.length_v +
                1.0e-6f;
            const auto inside_rectangle =
                (abs(light_u) <=
                 epsilon_u) &
                (abs(light_v) <=
                 epsilon_v);
            const auto inside_ellipse =
                light_u * light_u /
                        (epsilon_u *
                         epsilon_u) +
                    light_v * light_v /
                        (epsilon_v *
                         epsilon_v) <=
                1.0f;
            result = {
                .valid =
                    select(
                        inside_rectangle,
                        inside_ellipse,
                        input.ellipse) &
                    (factor > 0.0f),
                .direction =
                    direction.direction,
                .position = position,
                .normal = normal,
                .uv = area_uv(
                    input, position),
                .distance =
                    direction.distance,
                .conditional_pdf =
                    conditional_pdf,
                .evaluation_factor =
                    factor};
        };
    };
    return result;
}

sampling::FiniteLightSample
AreaLightSampling::from_intersection(
    const AreaLightSampleInput
        &input,
    Float3 direction,
    Float distance) const noexcept {
    const auto normal =
        -input.axis_z;
    const auto position =
        input.reference +
        direction * distance;
    sampling::FiniteLightSample result{
        .valid = false,
        .direction = direction,
        .position = position,
        .normal = normal,
        .uv = area_uv(
            input, position),
        .distance = distance,
        .conditional_pdf = 0.0f,
        .evaluation_factor = 0.0f};
    const auto tangent =
        tangent_half_spread(
            input.spread);
    auto shape =
        original_shape(input);
    $if(!input.full_spread) {
        shape =
            clamp_to_spread(
                input,
                normal,
                tangent);
    };
    $if(shape.valid) {
        Float conditional_pdf = 0.0f;
        $if(shape.rectangle) {
            conditional_pdf =
                sampling::
                    rectangle_solid_angle_pdf(
                        input.reference,
                        shape.center,
                        shape.axis_u,
                        shape.length_u,
                        shape.axis_v,
                        shape.length_v);
        }
        $else {
            $if(tangent == 0.0f) {
                conditional_pdf =
                    1.0f;
            }
            $else {
                conditional_pdf =
                    4.0f *
                    sampling::inverse_pi /
                    (shape.length_u *
                     shape.length_v);
                conditional_pdf *=
                    area_to_solid_angle(
                        normal,
                        {.direction =
                             direction,
                         .distance =
                             distance,
                         .distance_squared =
                             distance *
                             distance});
            };
        };
        const auto factor =
            evaluation_factor(
                input,
                direction,
                normal);
        result.valid =
            (area_value(input) > 0.0f) &
            (factor > 0.0f);
        result.conditional_pdf =
            conditional_pdf;
        result.evaluation_factor =
            factor;
    };
    return result;
}

}// namespace psycles::luisa_backend
