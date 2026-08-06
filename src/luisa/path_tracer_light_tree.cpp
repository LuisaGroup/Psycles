#include "path_tracer_light_tree.h"

#include <psycles/sampling/light_distribution.h>
#include <psycles/sampling/light_tree.h>
#include <psycles/luisa/cycles_path_state.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto pi = 3.14159265358979323846f;
constexpr auto half_pi = 0.5f * pi;
constexpr auto finite_volume_limit = 1.0e30f;
constexpr auto safe_distance = 1.0e-20f;

[[nodiscard]] Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept {
    const auto squared_length = dot(value, value);
    return select(
        fallback,
        value * rsqrt(max(squared_length, safe_distance)),
        squared_length > safe_distance);
}

[[nodiscard]] Float sin_from_cos(Float cosine) noexcept {
    return sqrt(max(1.0f - cosine * cosine, 0.0f));
}

[[nodiscard]] Float safe_divide(Float numerator, Float denominator) noexcept {
    return select(
        0.0f,
        numerator / denominator,
        abs(denominator) > safe_distance);
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
        point - centroid + direction * min(distance, 1.0e12f), v0);
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

    const auto theta_cosine = clamp(
        dot(cone_axis, -point_to_centroid), -1.0f, 1.0f);
    const auto theta_sine = sin_from_cos(theta_cosine);
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
            distance >= finite_volume_limit);
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
                    distance < finite_volume_limit);
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

[[nodiscard]] Float2 emitter_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt emitter_index,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    Var<LightTreeEmitterGpu> emitter =
        scene->light_tree_emitter_buffer->read(emitter_index);
    return measure_importance(
        emitter,
        emitter.identity.y,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        in_volume);
}

[[nodiscard]] Float2 node_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    Var<LightTreeNodeGpu> node,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    return measure_importance(
        node,
        node.emitters.z,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        in_volume);
}

[[nodiscard]] Float2 child_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt node_index,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    Var<LightTreeNodeGpu> node =
        scene->light_tree_node_buffer->read(node_index);
    Float2 result = make_float2(0.0f);
    const auto inner = node.topology.w == static_cast<std::uint32_t>(
        sampling::LightTreeNodeKind::inner);
    $if (inner | (node.emitters.y > 1u)) {
        result = node_importance(
            scene,
            node,
            point,
            normal_or_direction,
            distance,
            has_transmission,
            in_volume);
    }
    $elif (node.emitters.y == 1u) {
        result = emitter_importance(
            scene,
            node.emitters.x,
            point,
            normal_or_direction,
            distance,
            has_transmission,
            in_volume);
    };
    return result;
}

[[nodiscard]] Float2 left_probability(
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt left,
    UInt right,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume) noexcept {
    const auto left_importance = child_importance(
        scene,
        left,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        in_volume);
    const auto right_importance = child_importance(
        scene,
        right,
        point,
        normal_or_direction,
        distance,
        has_transmission,
        in_volume);
    const auto maximum_total =
        left_importance.x + right_importance.x;
    const auto minimum_total =
        left_importance.y + right_importance.y;
    const auto maximum_probability =
        left_importance.x / max(maximum_total, safe_distance);
    const auto minimum_probability = select(
        0.5f * (cast<float>(left_importance.x > 0.0f) +
                cast<float>(right_importance.x == 0.0f)),
        left_importance.y / max(minimum_total, safe_distance),
        minimum_total > 0.0f);
    return make_float2(
        0.5f * (maximum_probability + minimum_probability),
        cast<float>(maximum_total > 0.0f));
}

void reservoir_update(
    UInt current,
    Float weight,
    UInt &selected,
    Float &selected_weight,
    Float &total_weight,
    Float &random) noexcept {
    $if (weight > 0.0f) {
        total_weight += weight;
        $if (selected == sampling::invalid_light_tree_index) {
            selected = current;
            selected_weight = weight;
        }
        $else {
            const auto threshold = weight / total_weight;
            $if (random <= threshold) {
                selected = current;
                selected_weight = weight;
                random = random / threshold;
            }
            $else {
                random = (random - threshold) /
                         max(1.0f - threshold, safe_distance);
            };
            random = clamp(random, 0.0f, 1.0f);
        };
    };
}

void select_leaf_emitter(
    const std::shared_ptr<LuisaSceneData> &scene,
    Var<LightTreeNodeGpu> node,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
    bool in_volume,
    Float &random,
    UInt &selected,
    Float &selection_pdf) noexcept {
    Float selected_maximum = 0.0f;
    Float selected_minimum = 0.0f;
    Float total_maximum = 0.0f;
    Float total_minimum = 0.0f;
    UInt positive_count = 0u;
    const auto sample_maximum = random > 0.5f;
    $if (node.emitters.y > 1u) {
        random = random * 2.0f - cast<float>(sample_maximum);
    };

    $for (i, node.emitters.y) {
        const auto emitter_index = node.emitters.x + i;
        const auto importance = emitter_importance(
            scene,
            emitter_index,
            point,
            normal_or_direction,
            distance,
            has_transmission,
            in_volume);
        total_maximum += importance.x;
        total_minimum += importance.y;
        positive_count += cast<uint>(importance.x > 0.0f);
        const auto primary = select(
            importance.y, importance.x, sample_maximum);
        const auto primary_total = select(
            total_minimum, total_maximum, sample_maximum);
        $if (primary > 0.0f) {
            $if (selected == sampling::invalid_light_tree_index) {
                selected = emitter_index;
                selected_maximum = importance.x;
                selected_minimum = importance.y;
            }
            $else {
                const auto threshold = primary / primary_total;
                $if (random <= threshold) {
                    selected = emitter_index;
                    selected_maximum = importance.x;
                    selected_minimum = importance.y;
                    random = random / threshold;
                }
                $else {
                    random = (random - threshold) /
                             max(1.0f - threshold, safe_distance);
                };
                random = clamp(random, 0.0f, 1.0f);
            };
        };
    };

    $if ((positive_count > 0u) & (total_minimum == 0.0f)) {
        $if (sample_maximum) {
            selected_minimum = 1.0f;
            total_minimum = cast<float>(positive_count);
        }
        $else {
            selected = sampling::invalid_light_tree_index;
            Float selected_uniform = 0.0f;
            Float total_uniform = 0.0f;
            $for (i, node.emitters.y) {
                const auto emitter_index = node.emitters.x + i;
                const auto importance = emitter_importance(
                    scene,
                    emitter_index,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission,
                    in_volume);
                reservoir_update(
                    emitter_index,
                    cast<float>(importance.x > 0.0f),
                    selected,
                    selected_uniform,
                    total_uniform,
                    random);
                $if (selected == emitter_index) {
                    selected_maximum = importance.x;
                    selected_minimum = 1.0f;
                };
            };
            total_minimum = total_uniform;
        };
    };

    $if (selected != sampling::invalid_light_tree_index) {
        selection_pdf *= 0.5f *
                         (selected_maximum /
                              max(total_maximum, safe_distance) +
                          selected_minimum /
                              max(total_minimum, safe_distance));
    };
}

void initialize_invalid_light_selection(
    Var<LightDistributionGpu> &result) noexcept {
    result.cumulative = 0.0f;
    result.selection_pdf = 0.0f;
    result.kind = static_cast<std::uint32_t>(
        sampling::LightDistributionEmitterKind::sentinel);
    result.index = 0u;
    result.emitter_id = sampling::invalid_light_tree_index;
    result.padding_0 = 0u;
    result.padding_1 = 0u;
    result.padding_2 = 0u;
}

[[nodiscard]] LightTreeSampleCallable make_sample_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    bool in_volume) noexcept {
    return [scene, in_volume](
               Float sample,
               Float3 point,
               Float3 normal_or_direction,
               Float distance,
               Bool has_transmission) noexcept {
        Var<LightDistributionGpu> result;
        initialize_invalid_light_selection(result);
        if (scene->light_tree_node_count == 0u ||
            scene->light_tree_emitter_count == 0u ||
            scene->light_tree_root >= scene->light_tree_node_count) {
            return result;
        }

        Float random = clamp(sample, 0.0f, 1.0f);
        Float selection_pdf = 1.0f;
        UInt node_index = scene->light_tree_root;
        UInt selected = sampling::invalid_light_tree_index;
        Bool active = true;
        UInt steps = 0u;
        $while (active & (steps <= scene->light_tree_node_count)) {
            steps += 1u;
            Var<LightTreeNodeGpu> node =
                scene->light_tree_node_buffer->read(node_index);
            const auto leaf =
                node.topology.w != static_cast<std::uint32_t>(
                    sampling::LightTreeNodeKind::inner);
            $if (leaf) {
                select_leaf_emitter(
                    scene,
                    node,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission,
                    in_volume,
                    random,
                    selected,
                    selection_pdf);
                active = false;
            }
            $else {
                const auto probability = left_probability(
                    scene,
                    node.topology.y,
                    node.topology.z,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission,
                    in_volume);
                $if (probability.y <= 0.0f) {
                    active = false;
                }
                $else {
                    const auto right_probability = 1.0f - probability.x;
                    const auto choose_right = random <= right_probability;
                    node_index = select(
                        node.topology.y,
                        node.topology.z,
                        choose_right);
                    selection_pdf *= select(
                        probability.x,
                        right_probability,
                        choose_right);
                    random = select(
                        (random - right_probability) /
                            max(probability.x, safe_distance),
                        random /
                            max(right_probability, safe_distance),
                        choose_right);
                    random = clamp(random, 0.0f, 1.0f);
                };
            };
        };

        $if (selected != sampling::invalid_light_tree_index) {
            Var<LightTreeEmitterGpu> emitter =
                scene->light_tree_emitter_buffer->read(selected);
            $if (emitter.identity.x < scene->light_distribution_count) {
                result = scene->light_distribution_buffer->read(
                    emitter.identity.x);
                result.selection_pdf = selection_pdf;
            };
        };
        return result;
    };
}

[[nodiscard]] LightTreePdfCallable make_pdf_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    bool in_volume) noexcept {
    return [scene, in_volume](
               UInt emitter_id,
               Float3 point,
               Float3 normal_or_direction,
               Float distance,
               Bool has_transmission) noexcept {
        Float result = 0.0f;
        if (scene->light_tree_node_count == 0u ||
            scene->light_tree_emitter_count == 0u) {
            return result;
        }

        $if (emitter_id < scene->light_tree_emitter_count) {
            const auto mapping =
                scene->light_tree_emitter_mapping_buffer->read(emitter_id);
            const auto target_emitter = mapping.x;
            UInt current_node = mapping.y;
            $if ((target_emitter < scene->light_tree_emitter_count) &
                 (current_node < scene->light_tree_node_count)) {
                Var<LightTreeNodeGpu> leaf =
                    scene->light_tree_node_buffer->read(current_node);
                Float target_maximum = 0.0f;
                Float target_minimum = 0.0f;
                Float total_maximum = 0.0f;
                Float total_minimum = 0.0f;
                UInt positive_count = 0u;
                $for (i, leaf.emitters.y) {
                    const auto current_emitter = leaf.emitters.x + i;
                    const auto importance = emitter_importance(
                        scene,
                        current_emitter,
                        point,
                        normal_or_direction,
                        distance,
                        has_transmission,
                        in_volume);
                    total_maximum += importance.x;
                    total_minimum += importance.y;
                    positive_count += cast<uint>(importance.x > 0.0f);
                    $if (current_emitter == target_emitter) {
                        target_maximum = importance.x;
                        target_minimum = importance.y;
                    };
                };
                $if (target_maximum > 0.0f) {
                    result = 0.5f *
                             (target_maximum /
                                  max(total_maximum, safe_distance) +
                              select(
                                  1.0f /
                                      max(cast<float>(positive_count), 1.0f),
                                  target_minimum /
                                      max(total_minimum, safe_distance),
                                  total_minimum > 0.0f));
                    UInt steps = 0u;
                    Bool valid = true;
                    $while (valid &
                            (current_node != scene->light_tree_root) &
                            (steps <= scene->light_tree_node_count)) {
                        steps += 1u;
                        const auto parent_index =
                            scene->light_tree_node_buffer
                                ->read(current_node)
                                .topology.x;
                        $if (parent_index >= scene->light_tree_node_count) {
                            valid = false;
                            result = 0.0f;
                        }
                        $else {
                            Var<LightTreeNodeGpu> parent =
                                scene->light_tree_node_buffer->read(
                                    parent_index);
                            const auto probability = left_probability(
                                scene,
                                parent.topology.y,
                                parent.topology.z,
                                point,
                                normal_or_direction,
                                distance,
                                has_transmission,
                                in_volume);
                            $if (probability.y <= 0.0f) {
                                valid = false;
                                result = 0.0f;
                            }
                            $else {
                                result *= select(
                                    1.0f - probability.x,
                                    probability.x,
                                    current_node == parent.topology.y);
                                current_node = parent_index;
                            };
                        };
                    };
                    $if (current_node != scene->light_tree_root) {
                        result = 0.0f;
                    };
                };
            };
        };
        return result;
    };
}

[[nodiscard]] LightTreeTriangleEmitterCallable
make_triangle_emitter_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (scene->light_tree_triangle_lookup_count == 0u) {
        return [](UInt, UInt) noexcept {
            return UInt{sampling::invalid_light_tree_index};
        };
    }
    return [scene](UInt object, UInt primitive) noexcept {
        UInt result = sampling::invalid_light_tree_index;
        UInt first = 0u;
        UInt length = scene->light_tree_triangle_lookup_count;
        $while (length > 0u) {
            const auto half_length = length >> 1u;
            const auto middle = first + half_length;
            const auto entry =
                scene->light_tree_triangle_lookup_buffer->read(middle);
            const auto entry_less =
                (entry.x < object) |
                ((entry.x == object) & (entry.y < primitive));
            $if (entry_less) {
                first = middle + 1u;
                length = length - half_length - 1u;
            }
            $else {
                length = half_length;
            };
        };
        $if (first < scene->light_tree_triangle_lookup_count) {
            const auto entry =
                scene->light_tree_triangle_lookup_buffer->read(first);
            $if ((entry.x == object) & (entry.y == primitive)) {
                result = entry.z;
            };
        };
        return result;
    };
}

[[nodiscard]] LightTreeForwardPdfCallable make_forward_pdf_callable(
    LightTreePdfCallable surface_pdf,
    LightTreePdfCallable volume_pdf) noexcept {
    return [surface_pdf, volume_pdf](
               UInt emitter_id,
               Float3 point,
               Float3 mis_origin_normal,
               Float previous_dt,
               UInt path_visibility,
               UInt path_flags) noexcept {
        const auto has_transmission =
            (path_flags &
             cycles_path_state::flag_mis_had_transmission) != 0u;
        const auto from_volume =
            (path_visibility &
             cycles_path_state::visibility_volume_scatter) != 0u;
        const auto volume_direction = safe_normalize(
            mis_origin_normal,
            make_float3(0.0f, 0.0f, 1.0f));
        Float result;
        $if (from_volume) {
            result = volume_pdf(
                emitter_id,
                point - mis_origin_normal,
                volume_direction,
                previous_dt,
                has_transmission);
        }
        $else {
            result = surface_pdf(
                emitter_id,
                point,
                mis_origin_normal,
                0.0f,
                has_transmission);
        };
        return result;
    };
}

}// namespace

LightTreeCallables make_light_tree_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    auto surface_sample = make_sample_callable(scene, false);
    auto volume_sample = make_sample_callable(scene, true);
    auto surface_pdf = make_pdf_callable(scene, false);
    auto volume_pdf = make_pdf_callable(scene, true);
    auto forward_pdf = make_forward_pdf_callable(
        surface_pdf, volume_pdf);
    return {
        .surface_sample = std::move(surface_sample),
        .volume_sample = std::move(volume_sample),
        .surface_pdf = std::move(surface_pdf),
        .volume_pdf = std::move(volume_pdf),
        .forward_pdf = std::move(forward_pdf),
        .triangle_emitter = make_triangle_emitter_callable(scene)};
}

}// namespace psycles::luisa_backend::detail
