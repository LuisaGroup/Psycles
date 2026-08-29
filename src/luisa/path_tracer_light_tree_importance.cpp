#include "path_tracer_light_tree_importance.h"

#include <cmath>
#include <cstdint>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto pi = 3.14159265358979323846f;
constexpr auto half_pi = 0.5f * pi;
constexpr auto inverse_sqrt_two = 0.70710678118654752440f;
constexpr auto ray_infinity = 1.0e30f;
constexpr auto volume_projection_limit = 1.0e12f;
constexpr auto safe_distance = 1.0e-20f;

struct ExactEmitterState {
    Float3 centroid{make_float3(0.0f)};
    Float3 cone_axis{make_float3(0.0f, 0.0f, 1.0f)};
    Float theta_o{0.0f};
    Float theta_e{0.0f};
    Float energy{0.0f};
    Float3 point_to_centroid{make_float3(0.0f)};
    Float cos_theta_u{1.0f};
    Float2 distances{make_float2(1.0f)};
    Float theta_d{1.0f};
    Bool visible{false};
};

[[nodiscard]] UInt emitter_source(
    Var<LightTreeEmitterGpu> emitter) noexcept {
    return (emitter.identity.y & light_tree_emitter_source_mask) >>
           light_tree_emitter_source_shift;
}

[[nodiscard]] Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept {
    const auto squared_length = dot(value, value);
    return select(
        fallback,
        value * rsqrt(max(squared_length, safe_distance)),
        squared_length > safe_distance);
}

[[nodiscard]] Float3 safe_normalize_length(
    Float3 value,
    Float &value_length) noexcept {
    value_length = length(value);
    return select(
        value,
        value / value_length,
        value_length != 0.0f);
}

[[nodiscard]] Float sin_from_cos(Float cosine) noexcept {
    return sqrt(max(1.0f - cosine * cosine, 0.0f));
}

[[nodiscard]] Float cos_from_sin(Float sine) noexcept {
    return sqrt(max(1.0f - sine * sine, 0.0f));
}

[[nodiscard]] Float safe_divide(
    Float numerator,
    Float denominator) noexcept {
    return select(
        0.0f,
        numerator / denominator,
        denominator != 0.0f);
}

[[nodiscard]] Float smooth_unit_interval(Float value) noexcept {
    value = clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

[[nodiscard]] Float cos_bound_subtended_angle(
    Float3 bounds_max,
    Float3 centroid,
    Float3 point) noexcept {
    const auto distance_squared = dot(point - centroid, point - centroid);
    const auto radius_squared =
        dot(bounds_max - centroid, bounds_max - centroid);
    return select(
        sqrt(max(1.0f - radius_squared /
                           max(distance_squared, safe_distance),
                 0.0f)),
        -1.0f,
        distance_squared <= radius_squared);
}

[[nodiscard]] Float3 compute_v(
    Float3 centroid,
    Float3 point,
    Float3 direction,
    Float3 cone_axis,
    Float distance) noexcept {
    const auto v0 = safe_normalize(
        point - centroid, make_float3(0.0f, 0.0f, 1.0f));
    const auto v1 = safe_normalize(
        point - centroid + direction * min(distance, volume_projection_limit),
        v0);
    const auto o2 = safe_normalize(
        cross(v0, v1),
        safe_normalize(
            cross(v0, make_float3(1.0f, 0.0f, 0.0f)),
            make_float3(0.0f, 1.0f, 0.0f)));
    const auto o1 = cross(o2, v0);
    const auto dot_o0_axis = dot(v0, cone_axis);
    const auto dot_o1_axis = dot(o1, cone_axis);
    const auto inverse_length = rsqrt(max(
        dot_o0_axis * dot_o0_axis + dot_o1_axis * dot_o1_axis,
        safe_distance));
    const auto cos_phi0 = dot_o0_axis * inverse_length;
    const auto endpoint = select(
        v1,
        v0,
        dot_o0_axis > dot(v1, cone_axis));
    return select(
        cos_phi0 * v0 + dot_o1_axis * inverse_length * o1,
        endpoint,
        (dot_o1_axis < 0.0f) | (dot(v0, v1) > cos_phi0));
}

[[nodiscard]] Float2 bounded_importance(
    Float3 normal_or_direction,
    Bool has_transmission,
    Float3 point_to_centroid,
    Float cos_theta_u,
    Float3 cone_axis,
    Float theta_o,
    Float theta_e,
    Float maximum_distance,
    Float minimum_distance,
    Float energy,
    Float theta_d,
    bool in_volume) noexcept {
    cos_theta_u = clamp(cos_theta_u, -1.0f, 1.0f);
    const auto sin_theta_u = sin_from_cos(cos_theta_u);

    Float minimum_incidence = 1.0f;
    Float maximum_incidence = 1.0f;
    Bool visible = energy > 0.0f;
    if (!in_volume) {
        const auto incidence_cosine = select(
            dot(point_to_centroid, normal_or_direction),
            abs(dot(point_to_centroid, normal_or_direction)),
            has_transmission);
        const auto incidence_sine = sin_from_cos(
            clamp(incidence_cosine, -1.0f, 1.0f));
        minimum_incidence = select(
            incidence_cosine * cos_theta_u +
                incidence_sine * sin_theta_u,
            1.0f,
            incidence_cosine >= cos_theta_u);
        maximum_incidence = max(
            incidence_cosine * cos_theta_u -
                incidence_sine * sin_theta_u,
            0.0f);
        visible &= has_transmission | (minimum_incidence >= 0.0f);
    }

    const auto axes_are_exactly_opposed =
        all(cone_axis == -point_to_centroid);
    const auto theta_cosine = select(
        clamp(dot(cone_axis, -point_to_centroid), -1.0f, 1.0f),
        1.0f,
        axes_are_exactly_opposed);
    const auto theta_sine = select(
        sin_from_cos(theta_cosine),
        0.0f,
        axes_are_exactly_opposed);
    const auto theta_minus_u_cosine =
        theta_cosine * cos_theta_u + theta_sine * sin_theta_u;
    const auto theta_o_cosine = cos(theta_o);
    const auto theta_o_sine = sin(theta_o);
    const auto fully_covered =
        (theta_cosine >= cos_theta_u) |
        (theta_minus_u_cosine >= theta_o_cosine);
    const auto partially_covered =
        !fully_covered &
        ((theta_o + theta_e > pi) |
         (theta_minus_u_cosine > cos(theta_o + theta_e)));
    const auto theta_minus_u_sine =
        sin_from_cos(clamp(theta_minus_u_cosine, -1.0f, 1.0f));
    const auto minimum_outgoing = select(
        select(
            0.0f,
            theta_minus_u_cosine * theta_o_cosine +
                theta_minus_u_sine * theta_o_sine,
            partially_covered),
        1.0f,
        fully_covered);
    visible &= fully_covered | partially_covered;

    minimum_distance = max(minimum_distance, safe_distance);
    maximum_distance = max(maximum_distance, safe_distance);
    const auto geometric_factor = in_volume
                                      ? theta_d / minimum_distance
                                      : 1.0f /
                                            (minimum_distance *
                                             minimum_distance);
    const auto maximum = select(
        0.0f,
        abs(minimum_incidence * energy * minimum_outgoing *
            geometric_factor),
        visible);
    if (in_volume) {
        return make_float2(maximum, 0.0f);
    }

    const auto theta_plus_u_cosine =
        theta_cosine * cos_theta_u - theta_sine * sin_theta_u;
    const auto minimum_is_zero =
        (theta_e - theta_o < 0.0f) |
        (theta_cosine < 0.0f) |
        (cos_theta_u < 0.0f) |
        (theta_plus_u_cosine < cos(theta_e - theta_o));
    const auto theta_plus_u_sine =
        sin_from_cos(clamp(theta_plus_u_cosine, -1.0f, 1.0f));
    const auto maximum_outgoing =
        theta_plus_u_cosine * theta_o_cosine -
        theta_plus_u_sine * theta_o_sine;
    const auto minimum = select(
        abs(maximum_incidence * energy * maximum_outgoing /
            (maximum_distance * maximum_distance)),
        0.0f,
        minimum_is_zero | !visible);
    return make_float2(maximum, minimum);
}

template<typename PackedMeasure>
[[nodiscard]] Float2 measure_importance(
    const PackedMeasure &measure,
    UInt flags,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    Float2 result = make_float2(0.0f);
    const auto has_orientation =
        (flags & light_tree_measure_has_orientation) != 0u;
    const auto has_bounds =
        (flags & light_tree_measure_has_bounds) != 0u;
    const auto distant =
        (flags & light_tree_measure_is_distant) != 0u;
    $if (has_orientation & (measure.bounds_min_energy.w > 0.0f)) {
        const auto cone_axis = safe_normalize(
            measure.cone_axis_theta_e.xyz(),
            make_float3(0.0f, 0.0f, 1.0f));
        Float3 point_to_centroid = -cone_axis;
        Float cos_theta_u = cos(
            measure.bounds_max_theta_o.w +
            measure.cone_axis_theta_e.w);
        Float emitter_distance = 1.0f;
        Float theta_d = select(
            distance,
            1.0f,
            distance >= ray_infinity);
        Bool measure_visible = true;
        $if (!distant) {
            measure_visible = has_bounds;
            const auto bounds_min = measure.bounds_min_energy.xyz();
            const auto bounds_max = measure.bounds_max_theta_o.xyz();
            const auto centroid = 0.5f * (bounds_min + bounds_max);
            const auto extent = bounds_max - centroid;
            if (in_volume) {
                const auto closest_t =
                    dot(centroid - point, normal_or_direction);
                const auto closest_point =
                    point + normal_or_direction *
                                clamp(closest_t, 0.0f, distance);
                emitter_distance = length(
                    centroid - point -
                    normal_or_direction * closest_t);
                theta_d = select(
                    atan2(closest_t, emitter_distance) + half_pi,
                    atan2(
                        distance,
                        emitter_distance -
                            closest_t * safe_divide(
                                            distance - closest_t,
                                            emitter_distance)),
                    distance < ray_infinity);
                point_to_centroid = -compute_v(
                    centroid,
                    point,
                    normal_or_direction,
                    cone_axis,
                    distance);
                cos_theta_u = cos_bound_subtended_angle(
                    bounds_max, centroid, closest_point);
            } else {
                measure_visible =
                    has_transmission |
                    (dot(normal_or_direction, centroid - point) +
                         dot(abs(normal_or_direction), abs(extent)) >
                     0.0f);
                const auto delta = centroid - point;
                emitter_distance = length(delta);
                point_to_centroid = safe_normalize(delta, -cone_axis);
                cos_theta_u = cos_bound_subtended_angle(
                    bounds_max, centroid, point);
                theta_d = 1.0f;
            }
            emitter_distance = max(
                0.5f * length(centroid - bounds_max),
                emitter_distance);
        };
        $if (measure_visible) {
            result = bounded_importance(
                normal_or_direction,
                has_transmission,
                point_to_centroid,
                cos_theta_u,
                cone_axis,
                measure.bounds_max_theta_o.w,
                measure.cone_axis_theta_e.w,
                emitter_distance,
                emitter_distance,
                measure.bounds_min_energy.w,
                theta_d,
                in_volume);
        };
    };
    return result;
}

void prepare_volume_query(
    ExactEmitterState &state,
    Float3 point,
    Float3 direction,
    Float distance,
    Float3 &evaluation_point,
    bool in_volume) noexcept {
    evaluation_point = point;
    if (!in_volume) {
        return;
    }
    const auto closest_t = dot(state.centroid - point, direction);
    evaluation_point += direction * clamp(closest_t, 0.0f, distance);
    const auto perpendicular_distance = length(
        state.centroid - point - direction * closest_t);
    state.theta_d = select(
        atan2(closest_t, perpendicular_distance) + half_pi,
        atan2(
            distance,
            perpendicular_distance -
                closest_t * safe_divide(
                                distance - closest_t,
                                perpendicular_distance)),
        distance < ray_infinity);
}

void triangle_parameters(
    Var<EmissiveTriangleGpu> reference,
    Var<InstanceGpu> representative_instance,
    Float3 p0,
    Float3 p1,
    Float3 p2,
    Float3 point,
    Float3 normal_or_direction,
    ExactEmitterState &state,
    bool in_volume) noexcept {
    const auto front_only =
        reference.emission_sampling ==
        static_cast<std::uint32_t>(contract::EmissionSampling::front);
    const auto back_only =
        reference.emission_sampling ==
        static_cast<std::uint32_t>(contract::EmissionSampling::back);
    const auto one_sided = front_only | back_only;
    auto triangle_axis = safe_normalize(
        cross(p1 - p0, p2 - p0),
        make_float3(0.0f, 0.0f, 1.0f));
    triangle_axis = select(
        triangle_axis,
        -triangle_axis,
        back_only);
    const auto inverse_x =
        (representative_instance.cycles_world_to_object *
         make_float4(1.0f, 0.0f, 0.0f, 0.0f))
            .xyz();
    const auto inverse_y =
        (representative_instance.cycles_world_to_object *
         make_float4(0.0f, 1.0f, 0.0f, 0.0f))
            .xyz();
    const auto inverse_z =
        (representative_instance.cycles_world_to_object *
         make_float4(0.0f, 0.0f, 1.0f, 0.0f))
            .xyz();
    const auto negative_applied_transform =
        (representative_instance.cycles_transform_applied != 0u) &
        (dot(inverse_x, cross(inverse_y, inverse_z)) < 0.0f);
    triangle_axis = select(
        triangle_axis,
        -triangle_axis,
        one_sided & negative_applied_transform);
    state.cone_axis = select(
        safe_normalize(p0 - p1, make_float3(1.0f, 0.0f, 0.0f)),
        triangle_axis,
        one_sided);
    state.theta_o = select(half_pi, 0.0f, one_sided);

    Float minimum_distance = 0.0f;
    state.point_to_centroid = safe_normalize_length(
        state.centroid - point, minimum_distance);
    state.distances = make_float2(minimum_distance);
    state.cos_theta_u = 1.0f;
    Bool shape_above_surface = false;
    const auto visit_corner = [&](Float3 corner) noexcept {
        Float corner_distance = 0.0f;
        const auto point_to_corner = safe_normalize_length(
            corner - point, corner_distance);
        state.cos_theta_u = min(
            state.cos_theta_u,
            dot(state.point_to_centroid, point_to_corner));
        shape_above_surface |=
            dot(point_to_corner, normal_or_direction) > 0.0f;
        if (!in_volume) {
            state.distances.x = max(
                state.distances.x, corner_distance);
        }
    };
    visit_corner(p0);
    visit_corner(p1);
    visit_corner(p2);
    const auto front_facing =
        (state.theta_o != 0.0f) |
        (dot(state.cone_axis, state.point_to_centroid) < 0.0f);
    state.visible = front_facing & shape_above_surface;
}

void area_parameters(
    Var<LightGpu> light,
    Float3 point,
    Float3 normal_or_direction,
    ExactEmitterState &state,
    bool in_volume) noexcept {
    Float minimum_distance = 0.0f;
    state.point_to_centroid = safe_normalize_length(
        state.centroid - point, minimum_distance);
    state.distances = make_float2(minimum_distance);
    state.cos_theta_u = 1.0f;
    const auto extent_u = light.axis_x * light.size_u;
    const auto extent_v = light.axis_y * light.size_v;
    const auto visit_corner = [&](Float3 corner) noexcept {
        Float corner_distance = 0.0f;
        const auto point_to_corner = safe_normalize_length(
            corner - point, corner_distance);
        state.cos_theta_u = min(
            state.cos_theta_u,
            dot(state.point_to_centroid, point_to_corner));
        if (!in_volume) {
            state.distances.x = max(
                state.distances.x, corner_distance);
        }
    };
    visit_corner(state.centroid - 0.5f * extent_u - 0.5f * extent_v);
    visit_corner(state.centroid + 0.5f * extent_u - 0.5f * extent_v);
    visit_corner(state.centroid - 0.5f * extent_u + 0.5f * extent_v);
    visit_corner(state.centroid + 0.5f * extent_u + 0.5f * extent_v);
    const auto front_facing =
        dot(state.cone_axis, state.point_to_centroid) < 0.0f;
    const auto shape_above_surface =
        dot(normal_or_direction, state.centroid - point) +
            abs(dot(normal_or_direction, extent_u)) +
            abs(dot(normal_or_direction, extent_v)) >
        0.0f;
    state.visible = front_facing & shape_above_surface;
}

void point_parameters(
    Var<LightGpu> light,
    Float3 original_point,
    Float3 point,
    ExactEmitterState &state,
    bool in_volume) noexcept {
    Float minimum_distance = 0.0f;
    state.point_to_centroid = safe_normalize_length(
        state.centroid - point, minimum_distance);
    state.distances = make_float2(minimum_distance);
    state.cone_axis = safe_normalize(
        original_point - state.centroid,
        make_float3(0.0f, 0.0f, 1.0f));
    state.theta_o = 0.0f;
    state.visible = true;
    if (in_volume) {
        state.cos_theta_u = 1.0f;
        return;
    }
    const auto sphere =
        ((light.flags & light_flag_sphere) != 0u) &
        (light.radius != 0.0f);
    $if (sphere) {
        $if (minimum_distance > light.radius) {
            state.cos_theta_u = cos_from_sin(
                light.radius / minimum_distance);
            state.distances.x =
                minimum_distance / state.cos_theta_u;
        }
        $else {
            state.cos_theta_u = -1.0f;
            state.distances = make_float2(
                light.radius * inverse_sqrt_two);
        };
    }
    $else {
        const auto hypotenuse = sqrt(
            light.radius * light.radius +
            minimum_distance * minimum_distance);
        state.cos_theta_u = safe_divide(
            minimum_distance, hypotenuse);
        state.distances.x = hypotenuse;
    };
}

void spot_parameters(
    Var<LightGpu> light,
    Float3 point,
    ExactEmitterState &state,
    bool in_volume) noexcept {
    Float minimum_distance = 0.0f;
    state.point_to_centroid = safe_normalize_length(
        state.centroid - point, minimum_distance);
    state.distances = make_float2(minimum_distance);
    const auto sphere =
        ((light.flags & light_flag_sphere) != 0u) &
        (light.radius != 0.0f);
    $if (sphere) {
        state.cos_theta_u = select(
            -1.0f,
            cos_from_sin(light.radius / minimum_distance),
            minimum_distance > light.radius);
        if (!in_volume) {
            state.distances = select(
                make_float2(light.radius * inverse_sqrt_two),
                minimum_distance *
                    make_float2(1.0f / state.cos_theta_u, 1.0f),
                minimum_distance > light.radius);
        }
    }
    $else {
        const auto hypotenuse = sqrt(
            light.radius * light.radius +
            minimum_distance * minimum_distance);
        state.cos_theta_u = safe_divide(
            minimum_distance, hypotenuse);
        if (!in_volume) {
            state.distances.x = hypotenuse;
        }
    };
    state.visible = true;
    if (in_volume) {
        return;
    }
    const auto cosine_minimum_outgoing = cos(max(
        0.0f,
        acos(clamp(
            dot(state.cone_axis, -state.point_to_centroid),
            -1.0f,
            1.0f)) -
            acos(clamp(state.cos_theta_u, -1.0f, 1.0f))));
    const auto cosine_half_angle = cos(0.5f * light.spot_angle);
    const auto blend_width =
        (1.0f - cosine_half_angle) * light.spot_smooth;
    const auto cosine_delta =
        cosine_minimum_outgoing - cos(state.theta_e);
    state.energy *= select(
        cast<float>(cosine_delta >= 0.0f),
        smooth_unit_interval(cosine_delta / blend_width),
        blend_width > 0.0f);
}

void analytic_parameters(
    Var<LightGpu> light,
    Float3 original_point,
    Float3 evaluation_point,
    Float3 normal_or_direction,
    Float distance,
    ExactEmitterState &state,
    bool in_volume) noexcept {
    const auto type = light.type;
    $if (type == static_cast<std::uint32_t>(contract::LightType::area)) {
        area_parameters(
            light,
            evaluation_point,
            normal_or_direction,
            state,
            in_volume);
    }
    $elif (type == static_cast<std::uint32_t>(contract::LightType::point)) {
        point_parameters(
            light,
            original_point,
            evaluation_point,
            state,
            in_volume);
    }
    $elif (type == static_cast<std::uint32_t>(contract::LightType::spot)) {
        spot_parameters(light, evaluation_point, state, in_volume);
    }
    $elif (type == static_cast<std::uint32_t>(contract::LightType::distant)) {
        state.centroid = state.cone_axis;
        state.point_to_centroid = -state.centroid;
        state.cos_theta_u = cos(state.theta_e);
        state.distances = make_float2(
            1.0f / max(state.cos_theta_u, safe_distance),
            1.0f);
        state.theta_d = select(
            distance,
            1.0f,
            distance >= ray_infinity);
        state.visible = true;
    };
}

[[nodiscard]] Float2 exact_emitter_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    Var<LightTreeEmitterGpu> emitter,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    ExactEmitterState state;
    state.theta_o = emitter.bounds_max_theta_o.w;
    state.theta_e = emitter.cone_axis_theta_e.w;
    state.energy = emitter.bounds_min_energy.w;
    state.cone_axis = safe_normalize(
        emitter.cone_axis_theta_e.xyz(),
        make_float3(0.0f, 0.0f, 1.0f));
    const auto source = emitter_source(emitter);
    Float3 evaluation_point = point;
    Bool exact_source = false;
    // Mapping cardinality and source residency are independent invariants.
    // Renderer-neutral tests may exercise the hierarchy quotient without
    // allocating production geometry. Only trace resource references into
    // the callable when the complete triangle source relation is resident.
    const auto triangle_source_resident =
        static_cast<bool>(scene->emissive_triangle_buffer) &&
        static_cast<bool>(scene->geometry_buffer) &&
        static_cast<bool>(scene->instance_buffer) &&
        static_cast<bool>(scene->heap);
    const auto analytic_source_resident =
        scene->light_count > 0u &&
        static_cast<bool>(scene->light_buffer);

    $if (source == static_cast<std::uint32_t>(
                       LightTreeEmitterSource::triangle)) {
        if (triangle_source_resident) {
            exact_source = true;
            const auto reference =
                scene->emissive_triangle_buffer->read(emitter.identity.z);
            const auto geometry =
                scene->geometry_buffer->read(reference.geometry_index);
            const auto triangle =
                scene->heap->buffer<Triangle>(geometry.bindless_base)
                    .read(reference.primitive_index);
            const auto positions = scene->heap->buffer<luisa::float3>(
                geometry.bindless_base + 1u);
            const auto applied_positions =
                scene->heap->buffer<luisa::float3>(
                    geometry.bindless_base + 9u);
            const auto representative_instance =
                scene->instance_buffer->read(reference.instance_index);
            const auto transform_applied =
                representative_instance.cycles_transform_applied != 0u;
            const auto p0 = select(
                positions.read(triangle.i0),
                applied_positions.read(triangle.i0),
                transform_applied);
            const auto p1 = select(
                positions.read(triangle.i1),
                applied_positions.read(triangle.i1),
                transform_applied);
            const auto p2 = select(
                positions.read(triangle.i2),
                applied_positions.read(triangle.i2),
                transform_applied);
            state.centroid = (p0 + p1 + p2) * (1.0f / 3.0f);
            prepare_volume_query(
                state,
                point,
                normal_or_direction,
                distance,
                evaluation_point,
                in_volume);
            triangle_parameters(
                reference,
                representative_instance,
                p0,
                p1,
                p2,
                evaluation_point,
                normal_or_direction,
                state,
                in_volume);
        }
    }
    $elif (source == static_cast<std::uint32_t>(
                         LightTreeEmitterSource::analytic_light)) {
        if (analytic_source_resident) {
            exact_source = true;
            const auto light = scene->light_buffer->read(emitter.identity.z);
            state.centroid = light.position;
            state.cone_axis = safe_normalize(
                -light.axis_z,
                make_float3(0.0f, 0.0f, -1.0f));
            const auto distant =
                light.type == static_cast<std::uint32_t>(
                                  contract::LightType::distant);
            $if (distant) {
                state.centroid = state.cone_axis;
            };
            prepare_volume_query(
                state,
                point,
                normal_or_direction,
                distance,
                evaluation_point,
                in_volume);
            analytic_parameters(
                light,
                point,
                evaluation_point,
                normal_or_direction,
                distance,
                state,
                in_volume);
        }
    }
    $elif (source == static_cast<std::uint32_t>(
                         LightTreeEmitterSource::environment)) {
        exact_source = true;
        state.centroid = make_float3(0.0f, 0.0f, 1.0f);
        state.cone_axis = make_float3(0.0f, 0.0f, -1.0f);
        state.point_to_centroid = -state.centroid;
        state.cos_theta_u = -1.0f;
        state.distances = make_float2(1.0f);
        state.theta_d = select(
            distance,
            1.0f,
            distance >= ray_infinity);
        state.visible = true;
    };

    state.visible |= has_transmission;
    if (in_volume) {
        state.point_to_centroid = -compute_v(
            state.centroid,
            point,
            normal_or_direction,
            state.cone_axis,
            distance);
        const auto analytic_sun =
            source == static_cast<std::uint32_t>(
                          LightTreeEmitterSource::analytic_light);
        if (analytic_source_resident) {
            $if (analytic_sun) {
                const auto light =
                    scene->light_buffer->read(emitter.identity.z);
                $if (light.type == static_cast<std::uint32_t>(
                                      contract::LightType::distant)) {
                    state.point_to_centroid = -state.cone_axis;
                };
            };
        }
    }

    Float2 result = make_float2(0.0f);
    $if (exact_source) {
        $if (state.visible) {
            result = bounded_importance(
                normal_or_direction,
                has_transmission,
                state.point_to_centroid,
                state.cos_theta_u,
                state.cone_axis,
                state.theta_o,
                state.theta_e,
                state.distances.x,
                state.distances.y,
                state.energy,
                state.theta_d,
                in_volume);
        };
    }
    $else {
        result = measure_importance(
            emitter,
            emitter.identity.y,
            point,
            normal_or_direction,
            distance,
            has_transmission,
            in_volume);
    };
    return result;
}

}// namespace

Float3 cycles_light_tree_volume_projection_direction(
    Float3 centroid,
    Float3 point,
    Float3 direction,
    Float3 cone_axis,
    Float distance) noexcept {
    return compute_v(
        centroid, point, direction, cone_axis, distance);
}

LightTreeImportanceComponent::LightTreeImportanceComponent(
    std::shared_ptr<LuisaSceneData> scene,
    bool in_volume) noexcept
    : _scene{std::move(scene)}, _in_volume{in_volume} {}

Float2 LightTreeImportanceComponent::emitter(
    UInt emitter_index,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission) const noexcept {
    Var<LightTreeEmitterGpu> emitter_value =
        _scene->light_tree_emitter_buffer->read(emitter_index);
    return exact_emitter_importance(
        _scene,
        emitter_value,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        _in_volume);
}

Float2 LightTreeImportanceComponent::node(
    Var<LightTreeNodeGpu> node_value,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission) const noexcept {
    return measure_importance(
        node_value,
        node_value.emitters.z,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        _in_volume);
}

}// namespace psycles::luisa_backend::detail
