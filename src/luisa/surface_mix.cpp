#include "surface_mix.h"

#include "graph_surface_internal.h"

#include <array>
#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using SurfaceMixSvmCallable = luisa::compute::Callable<
    luisa::float3(luisa::uint, float, luisa::float3, luisa::float3)>;

[[nodiscard]] auto active_mix_operations(
    std::span<const std::uint16_t> immediate_domain) noexcept {
    constexpr auto operation_count =
        static_cast<std::size_t>(compiler::BlendOperation::value) + 1u;
    constexpr auto valid_bits = compiler::surface_value_mix_operation_mask |
                                compiler::surface_value_mix_factor_clamp_bit |
                                compiler::surface_value_mix_result_clamp_bit;
    std::array<bool, operation_count> result{};
    for (const auto encoded : immediate_domain) {
        if ((encoded & ~valid_bits) != 0u) {
            std::abort();
        }
        const auto operation = static_cast<std::size_t>(
            encoded & compiler::surface_value_mix_operation_mask);
        if (operation >= operation_count) {
            std::abort();
        }
        result[operation] = true;
    }
    return result;
}

} // namespace

Float3 evaluate_surface_mix_operation(const ShaderServices &services,
                                      compiler::BlendOperation operation,
                                      Float factor,
                                      Float3 a,
                                      Float3 b) noexcept {
    switch (operation) {
        case compiler::BlendOperation::mix:
            return lerp(a, b, factor);
        case compiler::BlendOperation::darken:
            return lerp(a, min(a, b), factor);
        case compiler::BlendOperation::multiply:
            return lerp(a, a * b, factor);
        case compiler::BlendOperation::burn: {
            const auto denominator = 1.0f - factor + factor * b;
            const auto burned =
                clamp(1.0f - (make_float3(1.0f) - a) / denominator, 0.0f, 1.0f);
            return select(burned, make_float3(0.0f), denominator <= 0.0f);
        }
        case compiler::BlendOperation::lighten:
            return lerp(a, max(a, b), factor);
        case compiler::BlendOperation::screen:
            return 1.0f - (1.0f - factor + factor * (make_float3(1.0f) - b)) *
                              (make_float3(1.0f) - a);
        case compiler::BlendOperation::dodge: {
            const auto denominator = 1.0f - factor * b;
            auto dodged = min(a / denominator, make_float3(1.0f));
            dodged = select(dodged, make_float3(1.0f), denominator <= 0.0f);
            return select(a, dodged, a != 0.0f);
        }
        case compiler::BlendOperation::add:
            return lerp(a, a + b, factor);
        case compiler::BlendOperation::overlay: {
            const auto low = a * (1.0f - factor + 2.0f * factor * b);
            const auto high = 1.0f - (1.0f - factor +
                                      2.0f * factor * (make_float3(1.0f) - b)) *
                                         (make_float3(1.0f) - a);
            return select(high, low, a < 0.5f);
        }
        case compiler::BlendOperation::soft_light: {
            const auto screen =
                1.0f - (make_float3(1.0f) - b) * (make_float3(1.0f) - a);
            return (1.0f - factor) * a +
                   factor * ((make_float3(1.0f) - a) * b * a + a * screen);
        }
        case compiler::BlendOperation::linear_light:
            return a + factor * (2.0f * b - 1.0f);
        case compiler::BlendOperation::difference:
            return lerp(a, abs(a - b), factor);
        case compiler::BlendOperation::exclusion:
            return max(lerp(a, a + b - 2.0f * a * b, factor),
                       make_float3(0.0f));
        case compiler::BlendOperation::subtract:
            return lerp(a, a - b, factor);
        case compiler::BlendOperation::divide: {
            const auto divided = (1.0f - factor) * a + factor * a / b;
            return select(a, divided, b != 0.0f);
        }
        case compiler::BlendOperation::hue: {
            const auto hsv_b = rgb_to_hsv(services, b);
            auto hsv = rgb_to_hsv(services, a);
            hsv.x = hsv_b.x;
            const auto recolored = hsv_to_rgb(services, hsv);
            return select(a, lerp(a, recolored, factor), hsv_b.y != 0.0f);
        }
        case compiler::BlendOperation::saturation: {
            auto hsv = rgb_to_hsv(services, a);
            const auto hsv_b = rgb_to_hsv(services, b);
            const auto has_saturation = hsv.y != 0.0f;
            hsv.y = lerp(hsv.y, hsv_b.y, factor);
            return select(a, hsv_to_rgb(services, hsv), has_saturation);
        }
        case compiler::BlendOperation::color: {
            const auto hsv_b = rgb_to_hsv(services, b);
            auto hsv = rgb_to_hsv(services, a);
            hsv.x = hsv_b.x;
            hsv.y = hsv_b.y;
            const auto recolored = hsv_to_rgb(services, hsv);
            return select(a, lerp(a, recolored, factor), hsv_b.y != 0.0f);
        }
        case compiler::BlendOperation::value: {
            auto hsv = rgb_to_hsv(services, a);
            const auto hsv_b = rgb_to_hsv(services, b);
            hsv.z = lerp(hsv.z, hsv_b.z, factor);
            return hsv_to_rgb(services, hsv);
        }
    }
    std::abort();
}

Float3 evaluate_surface_mix(const ShaderServices &services,
                            compiler::BlendOperation operation,
                            bool clamp_factor,
                            bool clamp_result,
                            Float factor,
                            Float3 a,
                            Float3 b) noexcept {
    if (clamp_factor) {
        factor = clamp(factor, 0.0f, 1.0f);
    }
    auto result =
        evaluate_surface_mix_operation(services, operation, factor, a, b);
    if (clamp_result) {
        result = clamp(result, 0.0f, 1.0f);
    }
    return result;
}

Float3 evaluate_surface_mix_svm(const ShaderServices &services,
                                UInt immediate,
                                std::span<const std::uint16_t> immediate_domain,
                                Float factor,
                                Float3 a,
                                Float3 b) noexcept {
    const auto active_operations = active_mix_operations(immediate_domain);
    SurfaceMixSvmCallable callable =
        [&services, active_operations](UInt encoded,
                                       Float input_factor,
                                       Float3 input_a,
                                       Float3 input_b) noexcept {
            const Bool clamp_factor =
                (encoded & compiler::surface_value_mix_factor_clamp_bit) != 0u;
            input_factor = select(input_factor,
                                  clamp(input_factor, 0.0f, 1.0f),
                                  clamp_factor);

            auto result = def(input_a);
            const UInt operation =
                encoded & compiler::surface_value_mix_operation_mask;
            luisa::compute::detail::SwitchStmtBuilder{operation} % [&] {
                for (auto index = std::size_t{0u};
                     index < active_operations.size();
                     ++index) {
                    if (!active_operations[index]) {
                        continue;
                    }
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        static_cast<luisa::uint>(index)} %
                        [&, index] {
                            result = evaluate_surface_mix_operation(
                                services,
                                static_cast<compiler::BlendOperation>(index),
                                input_factor,
                                input_a,
                                input_b);
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface Mix operation");
                };
            };
            const Bool clamp_result =
                (encoded & compiler::surface_value_mix_result_clamp_bit) != 0u;
            return select(result,
                          clamp(result, 0.0f, 1.0f),
                          clamp_result);
        };
    callable.set_name("surface_mix_svm");
    return callable(immediate, factor, a, b);
}

} // namespace psycles::luisa_backend::detail
