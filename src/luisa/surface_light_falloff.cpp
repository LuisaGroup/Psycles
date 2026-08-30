#include "surface_light_falloff.h"

#include <array>
#include <cstdlib>
#include <limits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr auto light_falloff_type_count = std::size_t{3u};

using SurfaceLightFalloffSvmCallable =
    luisa::compute::Callable<float(luisa::uint, float, float, float)>;

[[nodiscard]] auto active_light_falloff_types(
    std::span<const std::uint16_t> immediate_domain) noexcept {
    std::array<bool, light_falloff_type_count> result{};
    for (const auto encoded : immediate_domain) {
        if (encoded >= result.size()) {
            std::abort();
        }
        result[encoded] = true;
    }
    return result;
}

} // namespace

Float evaluate_surface_light_falloff(
    compiler::LightFalloffType type,
    Float strength,
    Float smooth,
    Float ray_length) noexcept {
    Float result = strength;

    // This structured branch is the semantic boundary in Cycles'
    // svm_node_light_falloff. It is deliberately not expressed with select:
    // the distant arm must not record or evaluate FLT_MAX * FLT_MAX.
    $if(ray_length != std::numeric_limits<float>::max()) {
        switch (type) {
            case compiler::LightFalloffType::quadratic:
                break;
            case compiler::LightFalloffType::linear:
                result *= ray_length;
                break;
            case compiler::LightFalloffType::constant:
                result *= ray_length * ray_length;
                break;
        }

        // Cycles loads and applies Smooth only after the output-type scaling.
        // Preserve that order and control dependence exactly.
        $if(smooth > 0.0f) {
            const auto squared = ray_length * ray_length;
            result *= squared / (smooth + squared);
        };
    };
    return result;
}

Float evaluate_surface_light_falloff_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float strength,
    Float smooth,
    Float ray_length) noexcept {
    const auto active_types = active_light_falloff_types(immediate_domain);
    SurfaceLightFalloffSvmCallable callable =
        [active_types](UInt type,
                       Float input_strength,
                       Float input_smooth,
                       Float input_ray_length) noexcept {
            Float result = 0.0f;
            luisa::compute::detail::SwitchStmtBuilder{type} % [&] {
                for (auto index = std::size_t{0u};
                     index < active_types.size();
                     ++index) {
                    if (!active_types[index]) {
                        continue;
                    }
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        static_cast<luisa::uint>(index)} %
                        [&, index] {
                            result = evaluate_surface_light_falloff(
                                static_cast<compiler::LightFalloffType>(index),
                                input_strength,
                                input_smooth,
                                input_ray_length);
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface Light Falloff output");
                };
            };
            return result;
        };
    callable.set_name("surface_light_falloff_svm");
    return callable(immediate, strength, smooth, ray_length);
}

} // namespace psycles::luisa_backend::detail
