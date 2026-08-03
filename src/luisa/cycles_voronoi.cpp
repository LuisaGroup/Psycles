#include <psycles/luisa/cycles_voronoi.h>

#include <psycles/luisa/cycles_noise.h>

#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_voronoi {
namespace {

using luisa::compute::Int;
using luisa::compute::Int4;
using luisa::compute::make_int4;

struct Parameters {
    Configuration configuration;
    Float scale;
    Float detail;
    Float roughness;
    Float lacunarity;
    Float smoothness;
    Float exponent;
    Float randomness;
    Float max_distance;
};

struct Octave {
    Float distance{0.0f};
    Float3 color{make_float3(0.0f)};
    Float4 position{make_float4(0.0f)};
};

using VoronoiCallable = luisa::compute::Callable<luisa::float4(
    luisa::float3, float, float, float, float, float, float, float, float)>;

[[nodiscard]] std::uint64_t
configuration_key(const Configuration &configuration) noexcept {
    return static_cast<std::uint64_t>(configuration.dimensions) |
           (static_cast<std::uint64_t>(configuration.feature) << 8u) |
           (static_cast<std::uint64_t>(configuration.metric) << 16u) |
           (static_cast<std::uint64_t>(configuration.output) << 24u) |
           (static_cast<std::uint64_t>(configuration.normalize) << 32u);
}

[[nodiscard]] auto &callable_registry() noexcept {
    static std::map<std::uint64_t, std::unique_ptr<VoronoiCallable>> registry;
    return registry;
}

[[nodiscard]] auto &callable_registry_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] bool needs_color(const Configuration &configuration) noexcept {
    return configuration.output == compiler::VoronoiOutput::color;
}

[[nodiscard]] bool needs_position(const Configuration &configuration) noexcept {
    return configuration.output == compiler::VoronoiOutput::position ||
           configuration.output == compiler::VoronoiOutput::w;
}

[[nodiscard]] bool
is_distance_feature(const Configuration &configuration) noexcept {
    return configuration.feature == compiler::VoronoiFeature::f1 ||
           configuration.feature == compiler::VoronoiFeature::f2 ||
           configuration.feature == compiler::VoronoiFeature::smooth_f1;
}

[[nodiscard]] Float4 input_coordinate(std::uint32_t dimensions, Float3 vector,
                                      Float w) noexcept {
    switch (dimensions) {
        case 1u:
            return make_float4(w, 0.0f, 0.0f, 0.0f);
        case 2u:
            return make_float4(vector.x, vector.y, 0.0f, 0.0f);
        case 4u:
            return make_float4(vector, w);
        case 3u:
        default:
            return make_float4(vector, 0.0f);
    }
}

[[nodiscard]] Float4 voronoi_position(std::uint32_t dimensions,
                                      Float4 coordinate) noexcept {
    switch (dimensions) {
        case 1u:
            return make_float4(0.0f, 0.0f, 0.0f, coordinate.x);
        case 2u:
            return make_float4(coordinate.x, coordinate.y, 0.0f, 0.0f);
        case 4u:
            return coordinate;
        case 3u:
        default:
            return make_float4(coordinate.xyz(), 0.0f);
    }
}

[[nodiscard]] Float component_sum(Float4 value,
                                  std::uint32_t dimensions) noexcept {
    auto result = value.x;
    if (dimensions >= 2u) {
        result += value.y;
    }
    if (dimensions >= 3u) {
        result += value.z;
    }
    if (dimensions >= 4u) {
        result += value.w;
    }
    return result;
}

[[nodiscard]] Float component_max(Float4 value,
                                  std::uint32_t dimensions) noexcept {
    auto result = value.x;
    if (dimensions >= 2u) {
        result = max(result, value.y);
    }
    if (dimensions >= 3u) {
        result = max(result, value.z);
    }
    if (dimensions >= 4u) {
        result = max(result, value.w);
    }
    return result;
}

[[nodiscard]] Float distance_squared(Float4 a, Float4 b,
                                     std::uint32_t dimensions) noexcept {
    const auto delta = a - b;
    return component_sum(delta * delta, dimensions);
}

[[nodiscard]] Float voronoi_distance(Float4 a, Float4 b,
                                     const Parameters &parameters) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto delta = abs(a - b);
    if (dimensions == 1u) {
        return delta.x;
    }
    switch (parameters.configuration.metric) {
        case compiler::VoronoiDistanceMetric::euclidean:
            return sqrt(component_sum(delta * delta, dimensions));
        case compiler::VoronoiDistanceMetric::manhattan:
            return component_sum(delta, dimensions);
        case compiler::VoronoiDistanceMetric::chebychev:
            return component_max(delta, dimensions);
        case compiler::VoronoiDistanceMetric::minkowski: {
            auto powered = make_float4(
                pow(delta.x, parameters.exponent), pow(delta.y, parameters.exponent),
                pow(delta.z, parameters.exponent), pow(delta.w, parameters.exponent));
            return pow(component_sum(powered, dimensions), 1.0f / parameters.exponent);
        }
    }
    return 0.0f;
}

[[nodiscard]] Float
voronoi_distance_bound(Float4 a, Float4 b,
                       const Parameters &parameters) noexcept {
    if (parameters.configuration.dimensions == 1u) {
        return abs(a.x - b.x);
    }
    if (parameters.configuration.metric !=
        compiler::VoronoiDistanceMetric::euclidean) {
        if (parameters.configuration.metric ==
            compiler::VoronoiDistanceMetric::minkowski) {
            const auto delta = abs(a - b);
            const auto powered = make_float4(
                pow(delta.x, parameters.exponent), pow(delta.y, parameters.exponent),
                pow(delta.z, parameters.exponent), pow(delta.w, parameters.exponent));
            return component_sum(powered, parameters.configuration.dimensions);
        }
        return voronoi_distance(a, b, parameters);
    }
    return distance_squared(a, b, parameters.configuration.dimensions);
}

template<typename Function>
void for_each_offset(std::uint32_t dimensions, int radius,
                     Function &&function) noexcept {
    const auto extent = static_cast<std::uint32_t>(radius * 2 + 1);
    auto offset_component = [radius](UInt index) noexcept {
        return cast<int>(index) - radius;
    };
    if (dimensions == 1u) {
        $for (x, extent) { function(make_int4(offset_component(x), 0, 0, 0)); };
        return;
    }
    if (dimensions == 2u) {
        $for (y, extent) {
            $for (x, extent) {
                function(make_int4(offset_component(x), offset_component(y), 0, 0));
            };
        };
        return;
    }
    if (dimensions == 3u) {
        $for (z, extent) {
            $for (y, extent) {
                $for (x, extent) {
                    function(make_int4(offset_component(x), offset_component(y),
                                       offset_component(z), 0));
                };
            };
        };
        return;
    }
    $for (w, extent) {
        $for (z, extent) {
            $for (y, extent) {
                $for (x, extent) {
                    function(make_int4(offset_component(x), offset_component(y),
                                       offset_component(z), offset_component(w)));
                };
            };
        };
    };
}

[[nodiscard]] Int4 floor_cell(Float4 coordinate) noexcept {
    return make_int4(cast<int>(coordinate.x), cast<int>(coordinate.y),
                     cast<int>(coordinate.z), cast<int>(coordinate.w));
}

[[nodiscard]] Float4 float_cell(Int4 cell) noexcept {
    return make_float4(cast<float>(cell.x), cast<float>(cell.y),
                       cast<float>(cell.z), cast<float>(cell.w));
}

[[nodiscard]] Int4 pcg_hash(Int4 value, std::uint32_t dimensions) noexcept {
    value = value * 1664525 + 1013904223;
    if (dimensions == 2u) {
        value.x += value.y * 1664525;
        value.y += value.x * 1664525;
        value.x = value.x ^ (value.x >> 16);
        value.y = value.y ^ (value.y >> 16);
        value.x += value.y * 1664525;
        value.y += value.x * 1664525;
    } else if (dimensions == 3u) {
        value.x += value.y * value.z;
        value.y += value.z * value.x;
        value.z += value.x * value.y;
        value.x = value.x ^ (value.x >> 16);
        value.y = value.y ^ (value.y >> 16);
        value.z = value.z ^ (value.z >> 16);
        value.x += value.y * value.z;
        value.y += value.z * value.x;
        value.z += value.x * value.y;
    } else {
        value.x += value.y * value.w;
        value.y += value.z * value.x;
        value.z += value.x * value.y;
        value.w += value.y * value.z;
        value.x = value.x ^ (value.x >> 16);
        value.y = value.y ^ (value.y >> 16);
        value.z = value.z ^ (value.z >> 16);
        value.w = value.w ^ (value.w >> 16);
        value.x += value.y * value.w;
        value.y += value.z * value.x;
        value.z += value.x * value.y;
        value.w += value.y * value.z;
    }
    value.x &= 0x7fffffff;
    value.y &= 0x7fffffff;
    value.z &= 0x7fffffff;
    value.w &= 0x7fffffff;
    return value;
}

[[nodiscard]] Float4 point_hash(Int4 cell, std::uint32_t dimensions) noexcept {
    const auto hash = pcg_hash(cell, dimensions);
    constexpr auto inverse_maximum = 1.0f / static_cast<float>(0x7fffffffu);
    return make_float4(cast<float>(hash.x), cast<float>(hash.y),
                       cast<float>(hash.z), cast<float>(hash.w)) *
           inverse_maximum;
}

[[nodiscard]] Float3 color_hash(Float4 cell_position, Int4 cell,
                                std::uint32_t dimensions) noexcept {
    if (dimensions == 1u) {
        return cycles_noise::hash_float_to_color(cell_position.x);
    }
    const auto hash = dimensions == 2u ? pcg_hash(make_int4(cell.x, cell.y, 0, 0), 3u) : pcg_hash(cell, dimensions);
    constexpr auto inverse_maximum = 1.0f / static_cast<float>(0x7fffffffu);
    return make_float3(cast<float>(hash.x), cast<float>(hash.y),
                       cast<float>(hash.z)) *
           inverse_maximum;
}

[[nodiscard]] Float4 random_point(Float4 cell_position, Int4 cell, Int4 offset,
                                  const Parameters &parameters) noexcept {
    const auto offset_float = float_cell(offset);
    if (parameters.configuration.dimensions == 1u) {
        return offset_float + make_float4(cycles_noise::hash_float(
                                  cell_position.x + offset_float.x)) *
                                  parameters.randomness;
    }
    return offset_float +
           point_hash(cell + offset, parameters.configuration.dimensions) *
               parameters.randomness;
}

[[nodiscard]] Octave voronoi_f1(const Parameters &parameters,
                                Float4 coordinate) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto cell_position = floor(coordinate);
    const auto local_position = coordinate - cell_position;
    const auto cell = floor_cell(cell_position);
    Float minimum_distance = std::numeric_limits<float>::max();
    Int4 target_offset = make_int4(0);
    Float4 target_position = make_float4(0.0f);
    for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
        const auto point_position =
            random_point(cell_position, cell, offset, parameters);
        const auto distance_to_point =
            voronoi_distance_bound(point_position, local_position, parameters);
        $if (distance_to_point < minimum_distance) {
            target_offset = offset;
            minimum_distance = distance_to_point;
            target_position = point_position;
        };
    });
    Octave octave;
    octave.distance =
        voronoi_distance(target_position, local_position, parameters);
    if (needs_color(parameters.configuration)) {
        octave.color = color_hash(cell_position + float_cell(target_offset),
                                  cell + target_offset, dimensions);
    }
    if (needs_position(parameters.configuration)) {
        octave.position =
            voronoi_position(dimensions, target_position + cell_position);
    }
    return octave;
}

[[nodiscard]] Octave voronoi_f2(const Parameters &parameters,
                                Float4 coordinate) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto cell_position = floor(coordinate);
    const auto local_position = coordinate - cell_position;
    const auto cell = floor_cell(cell_position);
    Float distance_f1 = std::numeric_limits<float>::max();
    Float distance_f2 = std::numeric_limits<float>::max();
    Int4 offset_f1 = make_int4(0);
    Int4 offset_f2 = make_int4(0);
    Float4 position_f1 = make_float4(0.0f);
    Float4 position_f2 = make_float4(0.0f);
    for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
        const auto point_position =
            random_point(cell_position, cell, offset, parameters);
        const auto distance_to_point =
            voronoi_distance(point_position, local_position, parameters);
        $if (distance_to_point < distance_f1) {
            distance_f2 = distance_f1;
            distance_f1 = distance_to_point;
            offset_f2 = offset_f1;
            offset_f1 = offset;
            position_f2 = position_f1;
            position_f1 = point_position;
        }
        $else {
            $if (distance_to_point < distance_f2) {
                distance_f2 = distance_to_point;
                offset_f2 = offset;
                position_f2 = point_position;
            };
        };
    });
    Octave octave;
    octave.distance = distance_f2;
    if (needs_color(parameters.configuration)) {
        octave.color = color_hash(cell_position + float_cell(offset_f2),
                                  cell + offset_f2, dimensions);
    }
    if (needs_position(parameters.configuration)) {
        octave.position = voronoi_position(dimensions, position_f2 + cell_position);
    }
    return octave;
}

[[nodiscard]] Octave voronoi_smooth_f1(const Parameters &parameters,
                                       Float4 coordinate) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto cell_position = floor(coordinate);
    const auto local_position = coordinate - cell_position;
    const auto cell = floor_cell(cell_position);
    Float smooth_distance = 0.0f;
    Float3 smooth_color = make_float3(0.0f);
    Float4 smooth_position = make_float4(0.0f);
    Float h = -1.0f;
    for_each_offset(dimensions, 2, [&](Int4 offset) noexcept {
        const auto point_position =
            random_point(cell_position, cell, offset, parameters);
        const auto distance_to_point =
            voronoi_distance(point_position, local_position, parameters);
        h = select(smoothstep(0.0f, 1.0f,
                              0.5f + 0.5f * (smooth_distance - distance_to_point) /
                                         parameters.smoothness),
                   1.0f, h == -1.0f);
        auto correction = parameters.smoothness * h * (1.0f - h);
        smooth_distance = lerp(smooth_distance, distance_to_point, h) - correction;
        correction /= 1.0f + 3.0f * parameters.smoothness;
        if (needs_color(parameters.configuration)) {
            const auto cell_color = color_hash(cell_position + float_cell(offset),
                                               cell + offset, dimensions);
            smooth_color = lerp(smooth_color, cell_color, h) - correction;
        }
        if (needs_position(parameters.configuration)) {
            smooth_position = lerp(smooth_position, point_position, h) - correction;
        }
    });
    Octave octave;
    octave.distance = smooth_distance;
    if (needs_color(parameters.configuration)) {
        octave.color = smooth_color;
    }
    if (needs_position(parameters.configuration)) {
        octave.position =
            voronoi_position(dimensions, cell_position + smooth_position);
    }
    return octave;
}

void assign_octave(Octave &destination, const Octave &source) noexcept {
    destination.distance = source.distance;
    destination.color = source.color;
    destination.position = source.position;
}

[[nodiscard]] Octave evaluate_octave(const Parameters &parameters,
                                     Float4 coordinate) noexcept {
    switch (parameters.configuration.feature) {
        case compiler::VoronoiFeature::f2:
            return voronoi_f2(parameters, coordinate);
        case compiler::VoronoiFeature::smooth_f1: {
            Octave octave;
            $if (parameters.smoothness != 0.0f) {
                const auto smooth = voronoi_smooth_f1(parameters, coordinate);
                assign_octave(octave, smooth);
            }
            $else {
                const auto f1 = voronoi_f1(parameters, coordinate);
                assign_octave(octave, f1);
            };
            return octave;
        }
        case compiler::VoronoiFeature::f1:
        default:
            return voronoi_f1(parameters, coordinate);
    }
}

[[nodiscard]] Float distance_to_edge(const Parameters &parameters,
                                     Float4 coordinate) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto cell_position = floor(coordinate);
    const auto local_position = coordinate - cell_position;
    if (dimensions == 1u) {
        const auto middle =
            cycles_noise::hash_float(cell_position.x) * parameters.randomness;
        const auto left = -1.0f + cycles_noise::hash_float(cell_position.x - 1.0f) *
                                      parameters.randomness;
        const auto right = 1.0f + cycles_noise::hash_float(cell_position.x + 1.0f) *
                                      parameters.randomness;
        return min(abs((middle + left) * 0.5f - local_position.x),
                   abs((middle + right) * 0.5f - local_position.x));
    }
    const auto cell = floor_cell(cell_position);
    Float4 vector_to_closest = make_float4(0.0f);
    Float minimum_distance = std::numeric_limits<float>::max();
    for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
        const auto vector_to_point =
            random_point(cell_position, cell, offset, parameters) - local_position;
        const auto distance_to_point =
            component_sum(vector_to_point * vector_to_point, dimensions);
        $if (distance_to_point < minimum_distance) {
            minimum_distance = distance_to_point;
            vector_to_closest = vector_to_point;
        };
    });
    minimum_distance = std::numeric_limits<float>::max();
    for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
        const auto vector_to_point =
            random_point(cell_position, cell, offset, parameters) - local_position;
        const auto perpendicular = vector_to_point - vector_to_closest;
        const auto perpendicular_squared =
            component_sum(perpendicular * perpendicular, dimensions);
        $if (perpendicular_squared > 0.0001f) {
            const auto distance =
                component_sum((vector_to_closest + vector_to_point) * perpendicular,
                              dimensions) /
                (2.0f * sqrt(perpendicular_squared));
            minimum_distance = min(minimum_distance, distance);
        };
    });
    return minimum_distance;
}

[[nodiscard]] Bool nonzero_offset(Int4 offset,
                                  std::uint32_t dimensions) noexcept {
    auto result = offset.x != 0;
    if (dimensions >= 2u) {
        result |= offset.y != 0;
    }
    if (dimensions >= 3u) {
        result |= offset.z != 0;
    }
    if (dimensions >= 4u) {
        result |= offset.w != 0;
    }
    return result;
}

[[nodiscard]] Float n_sphere_radius(const Parameters &parameters,
                                    Float4 coordinate) noexcept {
    const auto dimensions = parameters.configuration.dimensions;
    const auto cell_position = floor(coordinate);
    const auto local_position = coordinate - cell_position;
    const auto cell = floor_cell(cell_position);
    if (dimensions == 1u) {
        Float closest_point = 0.0f;
        Int closest_offset = 0;
        Float minimum_distance = std::numeric_limits<float>::max();
        for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
            const auto point = random_point(cell_position, cell, offset, parameters);
            const auto distance = abs(point.x - local_position.x);
            $if (distance < minimum_distance) {
                minimum_distance = distance;
                closest_point = point.x;
                closest_offset = offset.x;
            };
        });
        minimum_distance = std::numeric_limits<float>::max();
        Float closest_neighbor = 0.0f;
        for_each_offset(dimensions, 1, [&](Int4 neighbor) noexcept {
            $if (neighbor.x != 0) {
                const auto offset =
                    make_int4(neighbor.x + closest_offset, 0, 0, 0);
                const auto point =
                    random_point(cell_position, cell, offset, parameters);
                const auto distance = abs(closest_point - point.x);
                $if (distance < minimum_distance) {
                    minimum_distance = distance;
                    closest_neighbor = point.x;
                };
            };
        });
        return abs(closest_neighbor - closest_point) * 0.5f;
    }
    Float4 closest_point = make_float4(0.0f);
    Int4 closest_offset = make_int4(0);
    Float minimum_distance_squared = std::numeric_limits<float>::max();
    for_each_offset(dimensions, 1, [&](Int4 offset) noexcept {
        const auto point = random_point(cell_position, cell, offset, parameters);
        const auto squared = distance_squared(point, local_position, dimensions);
        $if (squared < minimum_distance_squared) {
            minimum_distance_squared = squared;
            closest_point = point;
            closest_offset = offset;
        };
    });
    minimum_distance_squared = std::numeric_limits<float>::max();
    Float4 closest_neighbor = make_float4(0.0f);
    for_each_offset(dimensions, 1, [&](Int4 neighbor_offset) noexcept {
        $if (nonzero_offset(neighbor_offset, dimensions)) {
            const auto offset = neighbor_offset + closest_offset;
            const auto point = random_point(cell_position, cell, offset, parameters);
            const auto squared = distance_squared(closest_point, point, dimensions);
            $if (squared < minimum_distance_squared) {
                minimum_distance_squared = squared;
                closest_neighbor = point;
            };
        };
    });
    return sqrt(distance_squared(closest_neighbor, closest_point, dimensions)) *
           0.5f;
}

[[nodiscard]] Octave fractal_distance_feature(const Parameters &parameters,
                                              Float4 coordinate) noexcept {
    Float amplitude = 1.0f;
    Float maximum_amplitude = 0.0f;
    Float frequency = 1.0f;
    Octave output;
    const auto zero_input =
        (parameters.detail == 0.0f) | (parameters.roughness == 0.0f);
    const auto iterations = cast<luisa::uint>(ceil(parameters.detail)) + 1u;
    $for (iteration, iterations) {
        const auto octave = evaluate_octave(parameters, coordinate * frequency);
        $if (zero_input) {
            maximum_amplitude = 1.0f;
            assign_octave(output, octave);
            $break;
        };
        $if (cast<float>(iteration) <= parameters.detail) {
            maximum_amplitude += amplitude;
            output.distance += octave.distance * amplitude;
            if (needs_color(parameters.configuration)) {
                output.color += octave.color * amplitude;
            }
            if (needs_position(parameters.configuration)) {
                output.position =
                    lerp(output.position, octave.position / frequency, amplitude);
            }
            frequency *= parameters.lacunarity;
            amplitude *= parameters.roughness;
        }
        $else {
            const auto remainder = parameters.detail - floor(parameters.detail);
            $if (remainder != 0.0f) {
                maximum_amplitude =
                    lerp(maximum_amplitude, maximum_amplitude + amplitude, remainder);
                output.distance =
                    lerp(output.distance, output.distance + octave.distance * amplitude,
                         remainder);
                if (needs_color(parameters.configuration)) {
                    output.color = lerp(
                        output.color, output.color + octave.color * amplitude, remainder);
                }
                if (needs_position(parameters.configuration)) {
                    output.position = lerp(
                        output.position,
                        lerp(output.position, octave.position / frequency, amplitude),
                        remainder);
                }
            };
        };
    };
    if (parameters.configuration.normalize) {
        output.distance /= maximum_amplitude * parameters.max_distance;
        if (needs_color(parameters.configuration)) {
            output.color /= maximum_amplitude;
        }
    }
    if (needs_position(parameters.configuration)) {
        output.position =
            select(make_float4(0.0f), output.position / parameters.scale,
                   parameters.scale != 0.0f);
    }
    return output;
}

[[nodiscard]] Float fractal_distance_to_edge(const Parameters &parameters,
                                             Float4 coordinate) noexcept {
    Float amplitude = 1.0f;
    Float maximum_amplitude = parameters.max_distance;
    Float frequency = 1.0f;
    Float distance = 8.0f;
    const auto zero_input =
        (parameters.detail == 0.0f) | (parameters.roughness == 0.0f);
    const auto iterations = cast<luisa::uint>(ceil(parameters.detail)) + 1u;
    $for (iteration, iterations) {
        const auto octave_distance =
            distance_to_edge(parameters, coordinate * frequency);
        $if (zero_input) {
            distance = octave_distance;
            $break;
        };
        $if (cast<float>(iteration) <= parameters.detail) {
            maximum_amplitude = lerp(maximum_amplitude,
                                     parameters.max_distance / frequency, amplitude);
            distance =
                lerp(distance, min(distance, octave_distance / frequency), amplitude);
            frequency *= parameters.lacunarity;
            amplitude *= parameters.roughness;
        }
        $else {
            const auto remainder = parameters.detail - floor(parameters.detail);
            $if (remainder != 0.0f) {
                const auto lerp_amplitude = lerp(
                    maximum_amplitude, parameters.max_distance / frequency, amplitude);
                maximum_amplitude = lerp(maximum_amplitude, lerp_amplitude, remainder);
                const auto lerp_distance = lerp(
                    distance, min(distance, octave_distance / frequency), amplitude);
                distance = lerp(distance, min(distance, lerp_distance), remainder);
            };
        };
    };
    if (parameters.configuration.normalize) {
        distance /= maximum_amplitude;
    }
    return distance;
}

[[nodiscard]] Float maximum_distance(const Parameters &parameters) noexcept {
    auto result = 0.5f + 0.5f * parameters.randomness;
    if (parameters.configuration.dimensions > 1u) {
        auto extent = make_float4(result);
        result = voronoi_distance(make_float4(0.0f), extent, parameters);
    }
    if (parameters.configuration.feature == compiler::VoronoiFeature::f2) {
        result *= 2.0f;
    }
    return result;
}

[[nodiscard]] Float4 evaluate_specialized(const Configuration &configuration,
                                          Float3 vector, Float w, Float scale,
                                          Float detail, Float roughness,
                                          Float lacunarity, Float smoothness,
                                          Float exponent,
                                          Float randomness) noexcept {
    Parameters parameters{.configuration = configuration,
                          .scale = scale,
                          .detail = clamp(detail, 0.0f, 15.0f),
                          .roughness = clamp(roughness, 0.0f, 1.0f),
                          .lacunarity = lacunarity,
                          .smoothness = clamp(smoothness * 0.5f, 0.0f, 0.5f),
                          .exponent = exponent,
                          .randomness = clamp(randomness, 0.0f, 1.0f),
                          .max_distance = 0.0f};
    const auto coordinate =
        input_coordinate(configuration.dimensions, vector, w) * scale;
    if (configuration.feature == compiler::VoronoiFeature::distance_to_edge) {
        if (configuration.output != compiler::VoronoiOutput::distance) {
            return make_float4(0.0f);
        }
        parameters.max_distance = 0.5f + 0.5f * parameters.randomness;
        return make_float4(fractal_distance_to_edge(parameters, coordinate));
    }
    if (configuration.feature == compiler::VoronoiFeature::n_sphere_radius) {
        if (configuration.output != compiler::VoronoiOutput::radius) {
            return make_float4(0.0f);
        }
        return make_float4(n_sphere_radius(parameters, coordinate));
    }
    if (!is_distance_feature(configuration) ||
        configuration.output == compiler::VoronoiOutput::radius) {
        return make_float4(0.0f);
    }
    parameters.max_distance = maximum_distance(parameters);
    const auto output = fractal_distance_feature(parameters, coordinate);
    switch (configuration.output) {
        case compiler::VoronoiOutput::distance:
            return make_float4(output.distance);
        case compiler::VoronoiOutput::color:
            return make_float4(output.color, 1.0f);
        case compiler::VoronoiOutput::position:
            return output.position;
        case compiler::VoronoiOutput::w:
            return make_float4(output.position.w);
        case compiler::VoronoiOutput::radius:
            return make_float4(0.0f);
    }
    return make_float4(0.0f);
}

[[nodiscard]] std::unique_ptr<VoronoiCallable>
make_callable(Configuration configuration) noexcept {
    return std::make_unique<VoronoiCallable>(
        [configuration](Float3 vector, Float w, Float scale, Float detail,
                        Float roughness, Float lacunarity, Float smoothness,
                        Float exponent, Float randomness) noexcept {
            return evaluate_specialized(configuration, vector, w, scale, detail,
                                        roughness, lacunarity, smoothness, exponent,
                                        randomness);
        });
}

}// namespace

bool is_operation(compiler::ValueOperation operation) noexcept {
    switch (operation) {
        case compiler::ValueOperation::voronoi_distance:
        case compiler::ValueOperation::voronoi_color:
        case compiler::ValueOperation::voronoi_position:
        case compiler::ValueOperation::voronoi_w:
        case compiler::ValueOperation::voronoi_radius:
            return true;
        default:
            return false;
    }
}

Configuration
decode_configuration(const compiler::ValueInstruction &instruction) noexcept {
    const auto encoded = instruction.static_u0;
    auto dimensions = static_cast<std::uint32_t>(encoded & 0xffu);
    if (dimensions < 1u || dimensions > 4u) {
        dimensions = 3u;
    }
    auto feature = compiler::VoronoiFeature::f1;
    switch (static_cast<compiler::VoronoiFeature>((encoded >> 8u) & 0xffu)) {
        case compiler::VoronoiFeature::f1:
        case compiler::VoronoiFeature::f2:
        case compiler::VoronoiFeature::smooth_f1:
        case compiler::VoronoiFeature::distance_to_edge:
        case compiler::VoronoiFeature::n_sphere_radius:
            feature = static_cast<compiler::VoronoiFeature>(
                (encoded >> 8u) & 0xffu);
            break;
    }
    auto metric = compiler::VoronoiDistanceMetric::euclidean;
    switch (static_cast<compiler::VoronoiDistanceMetric>(
        (encoded >> 16u) & 0xffu)) {
        case compiler::VoronoiDistanceMetric::euclidean:
        case compiler::VoronoiDistanceMetric::manhattan:
        case compiler::VoronoiDistanceMetric::chebychev:
        case compiler::VoronoiDistanceMetric::minkowski:
            metric = static_cast<compiler::VoronoiDistanceMetric>(
                (encoded >> 16u) & 0xffu);
            break;
    }
    auto output = compiler::VoronoiOutput::distance;
    switch (instruction.operation) {
        case compiler::ValueOperation::voronoi_color:
            output = compiler::VoronoiOutput::color;
            break;
        case compiler::ValueOperation::voronoi_position:
            output = compiler::VoronoiOutput::position;
            break;
        case compiler::ValueOperation::voronoi_w:
            output = compiler::VoronoiOutput::w;
            break;
        case compiler::ValueOperation::voronoi_radius:
            output = compiler::VoronoiOutput::radius;
            break;
        case compiler::ValueOperation::voronoi_distance:
        default:
            break;
    }
    return {.dimensions = dimensions,
            .feature = feature,
            .metric = metric,
            .output = output,
            .normalize = ((encoded >> 24u) & 1u) != 0u};
}

void prepare(const Configuration &configuration) noexcept {
    const auto key = configuration_key(configuration);
    std::scoped_lock lock{callable_registry_mutex()};
    auto &registry = callable_registry();
    if (!registry.contains(key)) {
        registry.emplace(key, make_callable(configuration));
    }
}

Float4 evaluate(const Configuration &configuration, Float3 vector, Float w,
                Float scale, Float detail, Float roughness, Float lacunarity,
                Float smoothness, Float exponent, Float randomness) noexcept {
    const VoronoiCallable *callable = nullptr;
    {
        std::scoped_lock lock{callable_registry_mutex()};
        const auto iterator =
            callable_registry().find(configuration_key(configuration));
        if (iterator != callable_registry().end()) {
            callable = iterator->second.get();
        }
    }
    if (callable == nullptr) {
        std::abort();
    }
    return (*callable)(vector, w, scale, detail, roughness, lacunarity,
                       smoothness, exponent, randomness);
}

}// namespace psycles::luisa_backend::cycles_voronoi
