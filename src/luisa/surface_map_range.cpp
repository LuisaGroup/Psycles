#include "surface_map_range.h"

#include <array>
#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using SurfaceClampSvmCallable =
    luisa::compute::Callable<float(luisa::uint, float, float, float)>;
using SurfaceMapRangeFloatSvmCallable = luisa::compute::Callable<float(
    luisa::uint, float, float, float, float, float, float)>;
using SurfaceMapRangeVectorSvmCallable = luisa::compute::Callable<luisa::float3(
    luisa::uint, luisa::float3, luisa::float3, luisa::float3, luisa::float3,
    luisa::float3, luisa::float3)>;

[[nodiscard]] auto
active_clamp_modes(std::span<const std::uint16_t> immediate_domain) noexcept {
    constexpr auto mode_count =
        static_cast<std::size_t>(compiler::ClampMode::range) + 1u;
    std::array<bool, mode_count> result{};
    if (immediate_domain.empty()) {
        std::abort();
    }
    for (const auto encoded : immediate_domain) {
        if ((encoded & ~compiler::surface_value_clamp_mode_mask) != 0u) {
            std::abort();
        }
        result[encoded & compiler::surface_value_clamp_mode_mask] = true;
    }
    return result;
}

[[nodiscard]] auto active_map_range_interpolations(
    std::span<const std::uint16_t> immediate_domain) noexcept {
    std::array<bool, compiler::map_range_interpolation_count> result{};
    if (immediate_domain.empty()) {
        std::abort();
    }
    for (const auto encoded : immediate_domain) {
        if ((encoded & ~compiler::surface_value_map_range_configuration_mask) !=
            0u) {
            std::abort();
        }
        const auto interpolation = static_cast<std::size_t>(
            encoded & compiler::surface_value_map_range_interpolation_mask);
        if (interpolation >= result.size()) {
            std::abort();
        }
        result[interpolation] = true;
    }
    return result;
}

[[nodiscard]] Float
map_range_float_unclamped(compiler::MapRangeInterpolation interpolation,
                          Float value, Float from_min, Float from_max,
                          Float to_min, Float to_max, Float steps) noexcept {
    const auto denominator = from_max - from_min;
    const auto has_range = denominator != 0.0f;
    auto factor = (value - from_min) / select(1.0f, denominator, has_range);
    switch (interpolation) {
        case compiler::MapRangeInterpolation::linear:
            break;
        case compiler::MapRangeInterpolation::stepped: {
            const auto valid_steps = steps > 0.0f;
            factor = select(
                0.0f, floor(factor * (steps + 1.0f)) / select(1.0f, steps, valid_steps),
                valid_steps);
            break;
        }
        case compiler::MapRangeInterpolation::smoothstep:
            factor = clamp(factor, 0.0f, 1.0f);
            factor = (3.0f - 2.0f * factor) * (factor * factor);
            break;
        case compiler::MapRangeInterpolation::smootherstep:
            factor = clamp(factor, 0.0f, 1.0f);
            factor =
                factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
            break;
    }
    return select(0.0f, to_min + factor * (to_max - to_min), has_range);
}

[[nodiscard]] Float safe_divide_component(Float numerator,
                                          Float denominator) noexcept {
    const auto nonzero = denominator != 0.0f;
    return select(0.0f, numerator / select(1.0f, denominator, nonzero), nonzero);
}

[[nodiscard]] Float stepped_component(Float factor, Float steps) noexcept {
    const auto valid_steps = steps > 0.0f;
    return select(
        0.0f, floor(factor * (steps + 1.0f)) / select(1.0f, steps, valid_steps),
        valid_steps);
}

[[nodiscard]] Float3
map_range_vector_unclamped(compiler::MapRangeInterpolation interpolation,
                           Float3 value, Float3 from_min, Float3 from_max,
                           Float3 to_min, Float3 to_max,
                           Float3 steps) noexcept {
    const auto numerator = value - from_min;
    const auto denominator = from_max - from_min;
    auto factor = make_float3(safe_divide_component(numerator.x, denominator.x),
                              safe_divide_component(numerator.y, denominator.y),
                              safe_divide_component(numerator.z, denominator.z));
    switch (interpolation) {
        case compiler::MapRangeInterpolation::linear:
            break;
        case compiler::MapRangeInterpolation::stepped:
            factor = make_float3(stepped_component(factor.x, steps.x),
                                 stepped_component(factor.y, steps.y),
                                 stepped_component(factor.z, steps.z));
            break;
        case compiler::MapRangeInterpolation::smoothstep:
            factor = clamp(factor, 0.0f, 1.0f);
            factor = (make_float3(3.0f) - 2.0f * factor) * (factor * factor);
            break;
        case compiler::MapRangeInterpolation::smootherstep:
            factor = clamp(factor, 0.0f, 1.0f);
            factor =
                factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
            break;
    }
    return to_min + factor * (to_max - to_min);
}

} // namespace

Float evaluate_surface_clamp(compiler::ClampMode mode, Float value,
                             Float minimum, Float maximum) noexcept {
    if (mode == compiler::ClampMode::range) {
        const auto reversed = minimum > maximum;
        const auto original_minimum = minimum;
        minimum = select(minimum, maximum, reversed);
        maximum = select(maximum, original_minimum, reversed);
    }
    return min(max(value, minimum), maximum);
}

Float evaluate_surface_clamp_svm(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    Float value, Float minimum, Float maximum) noexcept {
    const auto active_modes = active_clamp_modes(immediate_domain);
    SurfaceClampSvmCallable callable = [active_modes](UInt encoded, Float input,
                                                      Float lower,
                                                      Float upper) noexcept {
        Float result = 0.0f;
        const UInt mode = encoded & compiler::surface_value_clamp_mode_mask;
        luisa::compute::detail::SwitchStmtBuilder{mode} % [&] {
            for (auto index = std::size_t{0u}; index < active_modes.size(); ++index) {
                if (!active_modes[index]) {
                    continue;
                }
                luisa::compute::detail::SwitchCaseStmtBuilder{
                    static_cast<luisa::uint>(index)} %
                    [&, index] {
                        result = evaluate_surface_clamp(
                            static_cast<compiler::ClampMode>(index), input, lower, upper);
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable("invalid compact surface Clamp mode");
            };
        };
        return result;
    };
    callable.set_name("surface_clamp_svm");
    return callable(immediate, value, minimum, maximum);
}

Float evaluate_surface_map_range_float(
    compiler::MapRangeInterpolation interpolation, bool clamp_result,
    Float value, Float from_min, Float from_max, Float to_min, Float to_max,
    Float steps) noexcept {
    auto result = map_range_float_unclamped(interpolation, value, from_min,
                                            from_max, to_min, to_max, steps);
    if (clamp_result) {
        result = min(max(result, min(to_min, to_max)), max(to_min, to_max));
    }
    return result;
}

Float evaluate_surface_map_range_float_svm(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    Float value, Float from_min, Float from_max, Float to_min, Float to_max,
    Float steps) noexcept {
    const auto active_interpolations =
        active_map_range_interpolations(immediate_domain);
    SurfaceMapRangeFloatSvmCallable callable =
        [active_interpolations](UInt encoded, Float input, Float input_from_min,
                                Float input_from_max, Float input_to_min,
                                Float input_to_max, Float input_steps) noexcept {
            Float result = 0.0f;
            const UInt interpolation =
                encoded & compiler::surface_value_map_range_interpolation_mask;
            luisa::compute::detail::SwitchStmtBuilder{interpolation} % [&] {
                for (auto index = std::size_t{0u};
                     index < active_interpolations.size(); ++index) {
                    if (!active_interpolations[index]) {
                        continue;
                    }
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        static_cast<luisa::uint>(index)} %
                        [&, index] {
                            result = map_range_float_unclamped(
                                static_cast<compiler::MapRangeInterpolation>(index),
                                input, input_from_min, input_from_max, input_to_min,
                                input_to_max, input_steps);
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface Map Range interpolation");
                };
            };
            const Bool use_clamp =
                (encoded & compiler::surface_value_map_range_clamp_bit) != 0u;
            return select(result,
                          min(max(result, min(input_to_min, input_to_max)),
                              max(input_to_min, input_to_max)),
                          use_clamp);
        };
    callable.set_name("surface_map_range_float_svm");
    return callable(immediate, value, from_min, from_max, to_min, to_max, steps);
}

Float3 evaluate_surface_map_range_vector(
    compiler::MapRangeInterpolation interpolation, bool clamp_result,
    Float3 value, Float3 from_min, Float3 from_max, Float3 to_min,
    Float3 to_max, Float3 steps) noexcept {
    auto result = map_range_vector_unclamped(interpolation, value, from_min,
                                             from_max, to_min, to_max, steps);
    if (clamp_result &&
        (interpolation == compiler::MapRangeInterpolation::linear ||
         interpolation == compiler::MapRangeInterpolation::stepped)) {
        result = min(max(result, min(to_min, to_max)), max(to_min, to_max));
    }
    return result;
}

Float3 evaluate_surface_map_range_vector_svm(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    Float3 value, Float3 from_min, Float3 from_max, Float3 to_min,
    Float3 to_max, Float3 steps) noexcept {
    const auto active_interpolations =
        active_map_range_interpolations(immediate_domain);
    SurfaceMapRangeVectorSvmCallable callable =
        [active_interpolations](UInt encoded, Float3 input, Float3 input_from_min,
                                Float3 input_from_max, Float3 input_to_min,
                                Float3 input_to_max,
                                Float3 input_steps) noexcept {
            Float3 result = make_float3(0.0f);
            const UInt interpolation =
                encoded & compiler::surface_value_map_range_interpolation_mask;
            luisa::compute::detail::SwitchStmtBuilder{interpolation} % [&] {
                for (auto index = std::size_t{0u};
                     index < active_interpolations.size(); ++index) {
                    if (!active_interpolations[index]) {
                        continue;
                    }
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        static_cast<luisa::uint>(index)} %
                        [&, index] {
                            result = map_range_vector_unclamped(
                                static_cast<compiler::MapRangeInterpolation>(index),
                                input, input_from_min, input_from_max, input_to_min,
                                input_to_max, input_steps);
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface Vector Map Range interpolation");
                };
            };
            const Bool use_clamp =
                ((encoded & compiler::surface_value_map_range_clamp_bit) != 0u) &
                (interpolation < static_cast<luisa::uint>(
                                     compiler::MapRangeInterpolation::smoothstep));
            return select(result,
                          min(max(result, min(input_to_min, input_to_max)),
                              max(input_to_min, input_to_max)),
                          use_clamp);
        };
    callable.set_name("surface_map_range_vector_svm");
    return callable(immediate, value, from_min, from_max, to_min, to_max, steps);
}

} // namespace psycles::luisa_backend::detail
