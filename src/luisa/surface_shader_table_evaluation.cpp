#include "surface_shader_table_evaluation.h"

#include <array>
#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

#include <psycles/compiler/surface_execution_plan.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::uint32_t color_ramp_constant_bit = 1u;
inline constexpr std::uint32_t color_ramp_sampled_bit = 2u;

[[nodiscard]] Float4 color_ramp_sampled_impl(
    const SurfaceShaderTableReader &table,
    Float factor,
    bool constant) noexcept {
    Float4 result = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const auto count = table.count();
    $if(count >= 2u) {
        const auto last = count - 1u;
        const auto scaled =
            clamp(factor, 0.0f, 1.0f) *
            cast<float>(last);
        const auto index = min(cast<uint>(scaled), last);
        const auto t = scaled - cast<float>(index);
        auto sampled = make_float4(
            table.read(index, 0u),
            table.read(index, 1u),
            table.read(index, 2u),
            table.read(index, 3u));
        if (!constant) {
            const auto next_index = min(index + 1u, last);
            const auto next = make_float4(
                table.read(next_index, 0u),
                table.read(next_index, 1u),
                table.read(next_index, 2u),
                table.read(next_index, 3u));
            sampled = select(
                sampled,
                lerp(sampled, next, t),
                t > 0.0f);
        }
        result = sampled;
    };
    return result;
}

[[nodiscard]] Float4 color_ramp_control_impl(
    const SurfaceShaderTableReader &table,
    Float factor,
    bool constant) noexcept {
    Float3 color = make_float3(0.0f);
    Float alpha = 1.0f;
    const auto count = table.count();
    $if(count != 0u) {
        color = make_float3(
            table.read(0u, 1u),
            table.read(0u, 2u),
            table.read(0u, 3u));
        alpha = table.read(0u, 4u);
        UInt element = 1u;
        $while(element < count) {
            const auto previous = element - 1u;
            const auto p0 = table.read(previous, 0u);
            const auto p1 = table.read(element, 0u);
            auto t = clamp(
                (factor - p0) / max(p1 - p0, 1.0e-20f),
                0.0f,
                1.0f);
            if (constant) {
                t = 0.0f;
            }
            const auto c0 = make_float3(
                table.read(previous, 1u),
                table.read(previous, 2u),
                table.read(previous, 3u));
            const auto c1 = make_float3(
                table.read(element, 1u),
                table.read(element, 2u),
                table.read(element, 3u));
            const auto a0 = table.read(previous, 4u);
            const auto a1 = table.read(element, 4u);
            const auto use = factor >= p0;
            color = select(color, lerp(c0, c1, t), use);
            alpha = select(alpha, lerp(a0, a1, t), use);
            element += 1u;
        };
        const auto last = count - 1u;
        const auto use_last =
            factor >= table.read(last, 0u);
        color = select(
            color,
            make_float3(
                table.read(last, 1u),
                table.read(last, 2u),
                table.read(last, 3u)),
            use_last);
        alpha = select(
            alpha,
            table.read(last, 4u),
            use_last);
    };
    return make_float4(color, alpha);
}

}// namespace

SurfaceShaderTableView
surface_shader_table_view(const ShaderServices &services,
                          const SurfacePoint &point,
                          Expr<std::uint32_t> parameter) noexcept {
    const auto descriptor =
        services.parameter_float3(point.parameter_block, parameter)
            .template bitcast<luisa::uint3>();
    return {
        .offset = descriptor.x, .count = descriptor.y, .width = descriptor.z};
}

ServiceSurfaceShaderTableReader::
    ServiceSurfaceShaderTableReader(
        const ShaderServices &services,
        SurfaceShaderTableView table) noexcept
    : _services{services},
      _table{std::move(table)} {}

UInt ServiceSurfaceShaderTableReader::count() const noexcept {
    return _table.count;
}

Float ServiceSurfaceShaderTableReader::read(
    Expr<std::uint32_t> element,
    std::uint32_t component) const noexcept {
    return _services.parameter_float(
        0u,
        _table.offset + element * _table.width + component);
}

BufferSurfaceShaderTableReader::
    BufferSurfaceShaderTableReader(
        const luisa::compute::BufferFloat &data,
        SurfaceShaderTableView table) noexcept
    : _data{data},
      _table{std::move(table)} {}

UInt BufferSurfaceShaderTableReader::count() const noexcept {
    return _table.count;
}

Float BufferSurfaceShaderTableReader::read(
    Expr<std::uint32_t> element,
    std::uint32_t component) const noexcept {
    return _data.read(
        _table.offset + element * _table.width + component);
}

Float4 color_ramp_sampled_linear_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept {
    return color_ramp_sampled_impl(table, factor, false);
}

Float4 color_ramp_sampled_constant_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept {
    return color_ramp_sampled_impl(table, factor, true);
}

Float4 color_ramp_control_linear_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept {
    return color_ramp_control_impl(table, factor, false);
}

Float4 color_ramp_control_constant_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept {
    return color_ramp_control_impl(table, factor, true);
}

Float3 rgb_curve_sampled_inline(
    const SurfaceShaderTableReader &table,
    Float3 input,
    Float factor,
    Float min_x,
    Float max_x,
    Float extrapolate) noexcept {
    Float3 mapped = input;
    const auto count = table.count();
    $if(count >= 2u) {
        const auto lookup = [&](Float coordinate,
                                std::uint32_t channel) noexcept {
            const auto last = count - 1u;
            const auto scaled =
                clamp(coordinate, 0.0f, 1.0f) *
                cast<float>(last);
            const auto index = min(cast<uint>(scaled), last);
            const auto t = scaled - cast<float>(index);
            auto sampled = table.read(index, channel);
            const auto next = table.read(
                min(index + 1u, last), channel);
            sampled = select(
                sampled,
                lerp(sampled, next, t),
                t > 0.0f);
            $if(extrapolate != 0.0f) {
                const auto first = table.read(0u, channel);
                const auto second = table.read(1u, channel);
                const auto final_value = table.read(last, channel);
                const auto previous = table.read(last - 1u, channel);
                const auto below =
                    first +
                    (first - second) *
                        (-coordinate) *
                        cast<float>(last);
                const auto above =
                    final_value +
                    (final_value - previous) *
                        (coordinate - 1.0f) *
                        cast<float>(last);
                sampled = select(
                    sampled, below, coordinate < 0.0f);
                sampled = select(
                    sampled, above, coordinate > 1.0f);
            };
            return sampled;
        };
        const auto range = max_x - min_x;
        const auto relative = (input - min_x) / range;
        mapped = make_float3(
            lookup(relative.x, 0u),
            lookup(relative.y, 1u),
            lookup(relative.z, 2u));
    };
    return lerp(input, mapped, factor);
}

Float3 rgb_curve_control_inline(
    const SurfaceShaderTableReader &table,
    Float3 input,
    Float factor) noexcept {
    Float3 mapped = input;
    const auto count = table.count();
    $if(count >= 2u) {
        mapped = make_float3(0.0f);
        UInt element = 1u;
        $while(element < count) {
            const auto previous = element - 1u;
            const auto x0 = table.read(previous, 0u);
            const auto x1 = table.read(element, 0u);
            const auto t = clamp(
                (input - x0) /
                    max(x1 - x0, 1.0e-20f),
                make_float3(0.0f),
                make_float3(1.0f));
            const auto y0 = make_float3(
                table.read(previous, 1u),
                table.read(previous, 2u),
                table.read(previous, 3u));
            const auto y1 = make_float3(
                table.read(element, 1u),
                table.read(element, 2u),
                table.read(element, 3u));
            mapped = select(
                mapped,
                lerp(y0, y1, t),
                input >= x0);
            element += 1u;
        };
    };
    return lerp(input, mapped, factor);
}

Float4 color_ramp_sampled_linear(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->color_ramp_sampled_linear(table, factor);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return color_ramp_sampled_linear_inline(reader, factor);
}

Float4 color_ramp_sampled_constant(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->color_ramp_sampled_constant(table, factor);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return color_ramp_sampled_constant_inline(reader, factor);
}

Float4 color_ramp_control_linear(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->color_ramp_control_linear(table, factor);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return color_ramp_control_linear_inline(reader, factor);
}

Float4 color_ramp_control_constant(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->color_ramp_control_constant(table, factor);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return color_ramp_control_constant_inline(reader, factor);
}

Float4 evaluate_surface_color_ramp(const ShaderServices &services,
                                   const SurfaceShaderTableView &table,
                                   Float factor, std::uint32_t mode) noexcept {
    const auto sampled = (mode & color_ramp_sampled_bit) != 0u;
    const auto constant = (mode & color_ramp_constant_bit) != 0u;
    if (sampled && constant) {
        return color_ramp_sampled_constant(services, table, factor);
    }
    if (sampled) {
        return color_ramp_sampled_linear(services, table, factor);
    }
    if (constant) {
        return color_ramp_control_constant(services, table, factor);
    }
    return color_ramp_control_linear(services, table, factor);
}

Float4
evaluate_surface_color_ramp_svm(const ShaderServices &services, UInt immediate,
                                std::span<const std::uint16_t> immediate_domain,
                                const SurfaceShaderTableView &table,
                                Float factor) noexcept {
    std::array<bool, 4u> active{};
    for (const auto encoded : immediate_domain) {
        const auto mode = static_cast<std::uint32_t>(encoded) &
                          compiler::surface_value_color_ramp_mode_mask;
        if (mode >= active.size()) {
            std::abort();
        }
        active[mode] = true;
    }
    Float4 ramp = make_float4(0.0f);
    luisa::compute::detail::SwitchStmtBuilder{
        immediate & compiler::surface_value_color_ramp_mode_mask} %
        [&] {
            for (auto index = std::size_t{}; index < active.size(); ++index) {
                if (!active[index]) {
                    continue;
                }
                luisa::compute::detail::SwitchCaseStmtBuilder{
                    static_cast<luisa::uint>(index)} %
                    [&, index] {
                        ramp = evaluate_surface_color_ramp(
                            services, table, factor,
                            static_cast<std::uint32_t>(index));
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid compact surface Color Ramp mode");
            };
        };
    return ramp;
}

Float3 rgb_curve_sampled(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float3 input,
    Float factor,
    Float min_x,
    Float max_x,
    Float extrapolate) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->rgb_curve_sampled(
            table,
            input,
            factor,
            min_x,
            max_x,
            extrapolate);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return rgb_curve_sampled_inline(
        reader,
        input,
        factor,
        min_x,
        max_x,
        extrapolate);
}

Float3 rgb_curve_control(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float3 input,
    Float factor) noexcept {
    if (const auto provider =
            services.surface_shader_table_provider()) {
        return provider->rgb_curve_control(
            table, input, factor);
    }
    ServiceSurfaceShaderTableReader reader{services, table};
    return rgb_curve_control_inline(
        reader, input, factor);
}

Float3 evaluate_surface_rgb_curve_svm(
    const ShaderServices &services, UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    const SurfaceShaderTableView &table, Float3 input, Float factor,
    Float min_x, Float max_x, Float extrapolate) noexcept {
    std::array<bool, 2u> active{};
    if (immediate_domain.empty()) {
        std::abort();
    }
    for (const auto encoded : immediate_domain) {
        if ((encoded & ~compiler::surface_value_rgb_curve_sampled_bit) != 0u) {
            std::abort();
        }
        active[(encoded & compiler::surface_value_rgb_curve_sampled_bit) != 0u] =
            true;
    }
    Float3 result = make_float3(0.0f);
    const UInt sampled =
        immediate & compiler::surface_value_rgb_curve_sampled_bit;
    luisa::compute::detail::SwitchStmtBuilder{sampled} % [&] {
        if (active[0u]) {
            luisa::compute::detail::SwitchCaseStmtBuilder{0u} % [&] {
                result = rgb_curve_control(services, table, input, factor);
            };
        }
        if (active[1u]) {
            luisa::compute::detail::SwitchCaseStmtBuilder{
                compiler::surface_value_rgb_curve_sampled_bit} % [&] {
                result = rgb_curve_sampled(services, table, input, factor,
                                           min_x, max_x, extrapolate);
            };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid compact surface RGB Curve mode");
        };
    };
    return result;
}

}// namespace psycles::luisa_backend::detail
