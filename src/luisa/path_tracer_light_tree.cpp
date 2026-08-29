#include "path_tracer_light_tree.h"
#include "path_tracer_light_tree_importance.h"

#include <psycles/sampling/light_distribution.h>
#include <psycles/sampling/light_tree.h>
#include <psycles/luisa/cycles_path_state.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

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

[[nodiscard]] UInt emitter_kind(
    Var<LightTreeEmitterGpu> emitter) noexcept {
    return (emitter.identity.y & light_tree_emitter_kind_mask) >>
           light_tree_emitter_kind_shift;
}

void to_mesh_local_space(
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt instance_index,
    Float3 &point,
    Float3 &normal_or_direction,
    Float &distance,
    bool in_volume) noexcept {
    const auto instance = scene->instance_buffer->read(instance_index);
    const auto world_to_object = instance.cycles_world_to_object;
    point = (world_to_object * make_float4(point, 1.0f)).xyz();
    if (in_volume) {
        const auto local_direction =
            (world_to_object * make_float4(normal_or_direction, 0.0f)).xyz();
        const auto scale = sqrt(dot(local_direction, local_direction));
        normal_or_direction = select(
            normal_or_direction,
            local_direction / max(scale, safe_distance),
            scale > safe_distance);
        distance *= scale;
    } else {
        const auto object_to_world =
            scene->accel->instance_transform(instance_index);
        const auto local_normal =
            (transpose(object_to_world) *
             make_float4(normal_or_direction, 0.0f))
                .xyz();
        const auto squared_length = dot(local_normal, local_normal);
        normal_or_direction = select(
            normal_or_direction,
            local_normal * rsqrt(max(squared_length, safe_distance)),
            squared_length > safe_distance);
    }
}

[[nodiscard]] Float2 child_importance(
    const LightTreeImportanceComponent &importance,
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt node_index,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission) noexcept {
    Var<LightTreeNodeGpu> node =
        scene->light_tree_node_buffer->read(node_index);
    Float2 result = make_float2(0.0f);
    const auto inner = node.topology.w == static_cast<std::uint32_t>(
        sampling::LightTreeNodeKind::inner);
    $if (inner | (node.emitters.y > 1u)) {
        result = importance.node(
            node,
            point,
            normal_or_direction,
            distance,
            has_transmission);
    }
    $elif (node.emitters.y == 1u) {
        result = importance.emitter(
            node.emitters.x,
            point,
            normal_or_direction,
            distance,
            has_transmission);
    };
    return result;
}

[[nodiscard]] Float2 left_probability(
    const LightTreeImportanceComponent &importance,
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt left,
    UInt right,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission) noexcept {
    const auto left_importance = child_importance(
        importance,
        scene,
        left,
        point,
        normal_or_direction,
        distance,
        has_transmission);
    const auto right_importance = child_importance(
        importance,
        scene,
        right,
        point,
        normal_or_direction,
        distance,
        has_transmission);
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
    const LightTreeImportanceComponent &importance_component,
    Var<LightTreeNodeGpu> node,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission,
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
        const auto importance = importance_component.emitter(
            emitter_index,
            point,
            normal_or_direction,
            distance,
            has_transmission);
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
                const auto importance = importance_component.emitter(
                    emitter_index,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission);
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
    const auto importance =
        LightTreeImportanceComponent{scene, in_volume};
    return [scene, in_volume, importance](
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
        UInt mesh_triangle_offset = 0u;
        Bool inside_mesh = false;
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
                    importance,
                    node,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission,
                    random,
                    selected,
                    selection_pdf);
                active = false;
                $if (selected != sampling::invalid_light_tree_index) {
                    const auto emitter =
                        scene->light_tree_emitter_buffer->read(selected);
                    if (scene->light_tree_mesh_triangle_count > 0u) {
                        $if (emitter_kind(emitter) ==
                             static_cast<std::uint32_t>(
                                 LightTreeEmitterKind::mesh_instance)) {
                            to_mesh_local_space(
                                scene,
                                emitter.identity.x,
                                point,
                                normal_or_direction,
                                distance,
                                in_volume);
                            node_index = emitter.identity.z;
                            mesh_triangle_offset = emitter.identity.w;
                            inside_mesh = true;
                            selected = sampling::invalid_light_tree_index;
                            active = node_index < scene->light_tree_node_count;
                        };
                    }
                };
            }
            $else {
                const auto probability = left_probability(
                    importance,
                    scene,
                    node.topology.y,
                    node.topology.z,
                    point,
                    normal_or_direction,
                    distance,
                    has_transmission);
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
            UInt distribution_emitter = sampling::invalid_light_tree_index;
            const auto kind = emitter_kind(emitter);
            $if (kind == static_cast<std::uint32_t>(
                             LightTreeEmitterKind::direct)) {
                distribution_emitter = emitter.identity.x;
            };
            if (scene->light_tree_mesh_triangle_count > 0u) {
                $if (inside_mesh &
                     (kind == static_cast<std::uint32_t>(
                                  LightTreeEmitterKind::mesh_triangle))) {
                    const auto mapping =
                        mesh_triangle_offset + emitter.identity.x;
                    $if (mapping < scene->light_tree_mesh_triangle_count) {
                        distribution_emitter =
                            scene->light_tree_mesh_triangle_buffer->read(mapping);
                    };
                };
            }
            $if (distribution_emitter < scene->light_distribution_count) {
                result = scene->light_distribution_buffer->read(
                    distribution_emitter);
                result.selection_pdf = selection_pdf;
            };
        };
        return result;
    };
}

[[nodiscard]] Float tree_path_pdf(
    const LightTreeImportanceComponent &importance_component,
    const std::shared_ptr<LuisaSceneData> &scene,
    UInt target_emitter,
    UInt leaf_index,
    UInt root_index,
    Float3 point,
    Float3 normal_or_direction,
    Float distance,
    Bool has_transmission) noexcept {
    Float result = 0.0f;
    $if ((target_emitter < scene->light_tree_emitter_count) &
         (leaf_index < scene->light_tree_node_count) &
         (root_index < scene->light_tree_node_count)) {
        const auto leaf = scene->light_tree_node_buffer->read(leaf_index);
        Float target_maximum = 0.0f;
        Float target_minimum = 0.0f;
        Float total_maximum = 0.0f;
        Float total_minimum = 0.0f;
        UInt positive_count = 0u;
        $for (i, leaf.emitters.y) {
            const auto current_emitter = leaf.emitters.x + i;
            const auto importance = importance_component.emitter(
                current_emitter,
                point,
                normal_or_direction,
                distance,
                has_transmission);
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
                     (target_maximum / max(total_maximum, safe_distance) +
                      select(
                          1.0f / max(cast<float>(positive_count), 1.0f),
                          target_minimum /
                              max(total_minimum, safe_distance),
                          total_minimum > 0.0f));
            UInt current_node = leaf_index;
            UInt steps = 0u;
            Bool valid = true;
            $while (valid & (current_node != root_index) &
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
                    const auto parent =
                        scene->light_tree_node_buffer->read(parent_index);
                    const auto probability = left_probability(
                        importance_component,
                        scene,
                        parent.topology.y,
                        parent.topology.z,
                        point,
                        normal_or_direction,
                        distance,
                        has_transmission);
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
            $if (current_node != root_index) {
                result = 0.0f;
            };
        };
    };
    return result;
}

[[nodiscard]] LightTreePdfCallable make_pdf_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    bool in_volume) noexcept {
    const auto importance =
        LightTreeImportanceComponent{scene, in_volume};
    return [scene, in_volume, importance](
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

        $if (emitter_id < scene->light_distribution_count) {
            const auto top =
                scene->light_tree_emitter_mapping_buffer->read(emitter_id);
            result = tree_path_pdf(
                importance,
                scene,
                top.x,
                top.y,
                scene->light_tree_root,
                point,
                normal_or_direction,
                distance,
                has_transmission);
            if (scene->emissive_triangle_count > 0u) {
                $if (emitter_id < scene->emissive_triangle_count) {
                    const auto proxy =
                        scene->light_tree_emitter_buffer->read(top.x);
                    const auto local = scene
                                           ->light_tree_triangle_emitter_mapping_buffer
                                           ->read(emitter_id);
                    Float3 local_point = point;
                    Float3 local_normal_or_direction = normal_or_direction;
                    Float local_distance = distance;
                    to_mesh_local_space(
                        scene,
                        proxy.identity.x,
                        local_point,
                        local_normal_or_direction,
                        local_distance,
                        in_volume);
                    const auto local_pdf = tree_path_pdf(
                        importance,
                        scene,
                        local.x,
                        local.y,
                        proxy.identity.z,
                        local_point,
                        local_normal_or_direction,
                        local_distance,
                        has_transmission);
                    const auto proxy_is_mesh =
                        emitter_kind(proxy) == static_cast<std::uint32_t>(
                                                   LightTreeEmitterKind::mesh_instance);
                    result = select(0.0f, result * local_pdf, proxy_is_mesh);
                };
            }
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
