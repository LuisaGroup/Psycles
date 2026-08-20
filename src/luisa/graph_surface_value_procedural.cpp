#include "graph_surface_internal.h"
#include "surface_shader_table_evaluation.h"

#include <psycles/luisa/cycles_noise.h>
#include <luisa/dsl/sugar.h>

using namespace luisa::compute;

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

inline constexpr std::uint64_t color_ramp_constant_bit = 1u;
inline constexpr std::uint64_t color_ramp_sampled_bit = 2u;
inline constexpr std::uint64_t rgb_curve_sampled_bit = 1u;

// Variable-length node tables are material data. The descriptor keeps payload
// cardinality out of shader structure; sampled and legacy representations are
// selected from immutable instruction metadata while recording the graph.
[[nodiscard]] SurfaceShaderTableView shader_table_view(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<std::uint32_t> parameter) noexcept {
    const auto descriptor = services
                                .parameter_float3(
                                    point.parameter_block,
                                    parameter)
                                .template bitcast<luisa::uint3>();
    return {
        .offset = descriptor.x,
        .count = descriptor.y,
        .width = descriptor.z};
}

[[nodiscard]] bool supports_procedural_value(
    compiler::ValueOperation operation) noexcept {
    switch (operation) {
        case compiler::ValueOperation::noise_factor:
        case compiler::ValueOperation::noise_color:
        case compiler::ValueOperation::white_noise_value:
        case compiler::ValueOperation::white_noise_color:
        case compiler::ValueOperation::checker_color:
        case compiler::ValueOperation::checker_factor:
        case compiler::ValueOperation::brick_color:
        case compiler::ValueOperation::brick_factor:
        case compiler::ValueOperation::gradient:
        case compiler::ValueOperation::color_ramp:
        case compiler::ValueOperation::rgb_curve:
        case compiler::ValueOperation::separate_r:
        case compiler::ValueOperation::separate_g:
        case compiler::ValueOperation::separate_b:
        case compiler::ValueOperation::combine_color:
        case compiler::ValueOperation::hosek_wilkie_sky:
        case compiler::ValueOperation::nishita_sky:
            return true;
        default:
            return false;
    }
}

class ProceduralValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        [[maybe_unused]] const auto &services = context.services;
        [[maybe_unused]] const auto &point = context.point;
        [[maybe_unused]] auto &result = context.result;
        const auto &instruction = this->instruction();
        Float4 value = make_float4(0.0f);
        switch (instruction.operation) {
                case compiler::ValueOperation::noise_factor:
                case compiler::ValueOperation::noise_color: {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::noise_color;
                    const auto normalize =
                        (instruction.static_u1 & 1u) != 0u;
                    const auto noise_type =
                        static_cast<cycles_noise::Type>(
                            (instruction.static_u1 >> 8u) &
                            0xffu);
                    auto scale = scalar(
                        instruction.operand(operand::noise::scale), result);
                    value = cycles_noise::
                        evaluate_texture_shared(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        noise_type,
                        normalize,
                        color_needed,
                        vector(
                            instruction.operand(operand::noise::vector),
                            result) * scale,
                        scalar(
                            instruction.operand(operand::noise::w), result) *
                            scale,
                        scalar(
                            instruction.operand(operand::noise::detail),
                            result),
                        scalar(
                            instruction.operand(operand::noise::roughness),
                            result),
                        scalar(
                            instruction.operand(operand::noise::lacunarity),
                            result),
                        scalar(
                            instruction.operand(operand::noise::offset),
                            result),
                        scalar(
                            instruction.operand(operand::noise::gain), result),
                        scalar(
                            instruction.operand(operand::noise::distortion),
                            result));
                    break;
                }
                case compiler::ValueOperation::white_noise_value:
                case compiler::ValueOperation::white_noise_color: {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::
                            white_noise_color;
                    value =
                        cycles_noise::evaluate_white_shared(
                            static_cast<std::uint32_t>(
                                instruction.static_u0),
                            color_needed,
                            vector(
                                instruction.operand(
                                    operand::white_noise::vector),
                                result),
                            scalar(
                                instruction.operand(operand::white_noise::w),
                                result));
                    break;
                }
                case compiler::ValueOperation::checker_color:
                case compiler::ValueOperation::checker_factor: {
                    auto scaled_raw =
                        vector(
                            instruction.operand(operand::checker::vector),
                            result) *
                        scalar(
                            instruction.operand(operand::checker::scale),
                            result);
                    auto p = select(
                        scaled_raw,
                        make_float3(0.0f),
                        luisa::compute::dsl::isnan(
                            scaled_raw));
                    // Cycles computes `(p + 1e-6f) * 0.999999f`.
                    // Express the second factor as `1 - 0x1.1p-20f`,
                    // which is bit-identical in float32 but remains below
                    // the unit boundary even if a backend contracts the
                    // arithmetic.
                    auto shifted_raw = p + 0.000001f;
                    auto shifted = select(
                        shifted_raw,
                        make_float3(0.0f),
                        luisa::compute::dsl::isnan(
                            shifted_raw));
                    p =
                        shifted -
                        shifted * 0x1.1p-20f;
                    auto xi = abs(cast<int>(floor(p.x)));
                    auto yi = abs(cast<int>(floor(p.y)));
                    auto zi = abs(cast<int>(floor(p.z)));
                    auto xy_same = select(
                        0,
                        1,
                        (xi % 2) == (yi % 2));
                    auto is_first =
                        xy_same == (zi % 2);
                    auto factor = select(
                        0.0f, 1.0f, is_first);
                    if (instruction.operation ==
                        compiler::ValueOperation::
                            checker_color) {
                        value = make_float4(
                            lerp(
                                vector(
                                    instruction.operand(
                                        operand::checker::color2),
                                    result),
                                vector(
                                    instruction.operand(
                                        operand::checker::color1),
                                    result),
                                factor),
                            1.0f);
                    } else {
                        value = make_float4(factor);
                    }
                    break;
                }
                case compiler::ValueOperation::brick_color:
                case compiler::ValueOperation::brick_factor: {
                    auto p =
                        vector(
                            instruction.operand(operand::brick::vector),
                            result) *
                        scalar(
                            instruction.operand(operand::brick::scale),
                            result);
                    auto mortar_size = max(
                        scalar(
                            instruction.operand(operand::brick::mortar_size),
                            result),
                        0.0f);
                    auto mortar_smooth = max(
                        scalar(
                            instruction.operand(
                                operand::brick::mortar_smooth),
                            result),
                        0.0f);
                    auto bias = scalar(
                        instruction.operand(operand::brick::bias), result);
                    auto brick_width = max(
                        abs(scalar(
                            instruction.operand(operand::brick::brick_width),
                            result)),
                        1.0e-20f);
                    auto row_height = max(
                        abs(scalar(
                            instruction.operand(operand::brick::row_height),
                            result)),
                        1.0e-20f);
                    auto row = cast<int>(
                        floor(p.y / row_height));
                    const auto offset_frequency = cast<int>(min(
                        unsigned_integer(
                            instruction.operand(
                                operand::brick::offset_frequency),
                            result),
                        luisa::ulong{0x7fffffffull}));
                    const auto squash_frequency = cast<int>(min(
                        unsigned_integer(
                            instruction.operand(
                                operand::brick::squash_frequency),
                            result),
                        luisa::ulong{0x7fffffffull}));
                    Float offset = 0.0f;
                    $if ((offset_frequency != 0) &
                         (squash_frequency != 0)) {
                        auto squash_row =
                            (row % squash_frequency) == 0;
                        brick_width *= select(
                            1.0f,
                            scalar(
                                instruction.operand(
                                    operand::brick::squash_amount),
                                result),
                            squash_row);
                        auto offset_row =
                            (row % offset_frequency) == 0;
                        offset = select(
                            0.0f,
                            brick_width *
                                scalar(
                                    instruction.operand(
                                        operand::brick::offset_amount),
                                    result),
                            offset_row);
                    };
                    auto brick = cast<int>(floor(
                        (p.x + offset) / brick_width));
                    auto x =
                        p.x + offset -
                        brick_width * cast<float>(brick);
                    auto y =
                        p.y - row_height * cast<float>(row);
                    auto n =
                        (cast<uint>(row) << 16u) +
                        (cast<uint>(brick) & 0xffffu);
                    n = (n + 1013u) & 0x7fffffffu;
                    n = (n >> 13u) ^ n;
                    auto nn =
                        (n * (n * n * 60493u + 19990303u) +
                         1376312589u) &
                        0x7fffffffu;
                    auto tint = clamp(
                        0.5f *
                                cast<float>(nn) /
                                1073741824.0f +
                            bias,
                        0.0f,
                        1.0f);
                    auto min_distance = min(
                        min(x, y),
                        min(
                            brick_width - x,
                            row_height - y));
                    Float mortar = select(
                        1.0f,
                        0.0f,
                        min_distance >= mortar_size);
                    auto edge =
                        1.0f -
                        min_distance /
                            max(mortar_size, 1.0e-20f);
                    auto smooth = clamp(
                        edge /
                            max(mortar_smooth, 1.0e-20f),
                        0.0f,
                        1.0f);
                    smooth =
                        smooth * smooth *
                        (3.0f - 2.0f * smooth);
                    auto smooth_mortar = select(
                        smooth,
                        0.0f,
                        min_distance >= mortar_size);
                    mortar = select(
                        mortar,
                        smooth_mortar,
                        mortar_smooth != 0.0f);
                    if (instruction.operation ==
                        compiler::ValueOperation::brick_factor) {
                        value = make_float4(mortar);
                    } else {
                        auto brick_color = lerp(
                            vector(
                                instruction.operand(operand::brick::color1),
                                result),
                            vector(
                                instruction.operand(operand::brick::color2),
                                result),
                            tint);
                        value = make_float4(
                            lerp(
                                brick_color,
                                vector(
                                    instruction.operand(
                                        operand::brick::mortar),
                                    result),
                                mortar),
                            1.0f);
                    }
                    break;
                }
                case compiler::ValueOperation::gradient: {
                    auto p = vector(
                        instruction.operand(operand::gradient::vector),
                        result);
                    Float gradient = p.x;
                    if (instruction.static_u0 == 1u) {
                        gradient = max(p.x, 0.0f);
                        gradient *= gradient;
                    } else if (instruction.static_u0 == 2u) {
                        auto t = clamp(p.x, 0.0f, 1.0f);
                        gradient = t * t * (3.0f - 2.0f * t);
                    } else if (instruction.static_u0 == 3u) {
                        gradient = (p.x + p.y) * 0.5f;
                    } else if (instruction.static_u0 == 4u) {
                        gradient =
                            atan2(p.y, p.x) / two_pi + 0.5f;
                    } else if (instruction.static_u0 == 5u) {
                        gradient = max(
                            0.999999f - length(p),
                            0.0f);
                    } else if (instruction.static_u0 == 6u) {
                        gradient = max(
                            0.999999f - length(p),
                            0.0f);
                        gradient *= gradient;
                    }
                    gradient = clamp(
                        gradient, 0.0f, 1.0f);
                    value = make_float4(gradient);
                    break;
                }
                case compiler::ValueOperation::color_ramp: {
                    const auto factor = scalar(
                        instruction.operand(operand::color_ramp::factor),
                        result);
                    const auto parameter =
                        context.parameter_override != nullptr
                            ? *context.parameter_override
                            : Expr<std::uint32_t>{
                                  instruction.parameter.value};
                    const auto table = shader_table_view(
                        services, point, parameter);
                    const auto sampled =
                        (instruction.static_u0 &
                         color_ramp_sampled_bit) != 0u;
                    const auto constant =
                        (instruction.static_u0 &
                         color_ramp_constant_bit) != 0u;
                    Float4 ramp;
                    if (sampled && constant) {
                        ramp = color_ramp_sampled_constant(
                            services, table, factor);
                    } else if (sampled) {
                        ramp = color_ramp_sampled_linear(
                            services, table, factor);
                    } else if (constant) {
                        ramp = color_ramp_control_constant(
                            services, table, factor);
                    } else {
                        ramp = color_ramp_control_linear(
                            services, table, factor);
                    }
                    value =
                        instruction.static_u1 != 0u
                            ? make_float4(ramp.w)
                            : ramp;
                    break;
                }
                case compiler::ValueOperation::rgb_curve: {
                    const auto input = vector(
                        instruction.operand(operand::rgb_curve::color),
                        result);
                    const auto factor = scalar(
                        instruction.operand(operand::rgb_curve::factor),
                        result);
                    const auto parameter =
                        context.parameter_override != nullptr
                            ? *context.parameter_override
                            : Expr<std::uint32_t>{
                                  instruction.parameter.value};
                    const auto table = shader_table_view(
                        services, point, parameter);
                    Float3 mapped;
                    if ((instruction.static_u0 &
                         rgb_curve_sampled_bit) != 0u) {
                        mapped = rgb_curve_sampled(
                            services,
                            table,
                            input,
                            factor,
                            scalar(
                                instruction.operand(
                                    operand::rgb_curve::min_x),
                                result),
                            scalar(
                                instruction.operand(
                                    operand::rgb_curve::max_x),
                                result),
                            scalar(
                                instruction.operand(
                                    operand::rgb_curve::extrapolate),
                                result));
                    } else {
                        mapped = rgb_curve_control(
                            services, table, input, factor);
                    }
                    value = make_float4(mapped, 1.0f);
                    break;
                }
                case compiler::ValueOperation::separate_r:
                    value = make_float4(
                        separate_color(
                            services,
                            vector(
                                instruction.operand(
                                    operand::separate_color::color),
                                result),
                            instruction.static_u0)
                            .x);
                    break;
                case compiler::ValueOperation::separate_g:
                    value = make_float4(
                        separate_color(
                            services,
                            vector(
                                instruction.operand(
                                    operand::separate_color::color),
                                result),
                            instruction.static_u0)
                            .y);
                    break;
                case compiler::ValueOperation::separate_b:
                    value = make_float4(
                        separate_color(
                            services,
                            vector(
                                instruction.operand(
                                    operand::separate_color::color),
                                result),
                            instruction.static_u0)
                            .z);
                    break;
                case compiler::ValueOperation::combine_color: {
                    auto channels = make_float3(
                        scalar(
                            instruction.operand(operand::combine_color::r),
                            result),
                        scalar(
                            instruction.operand(operand::combine_color::g),
                            result),
                        scalar(
                            instruction.operand(operand::combine_color::b),
                            result));
                    value = make_float4(
                        combine_color(
                            services,
                            channels,
                            instruction.static_u0),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::hosek_wilkie_sky: {
                    if (instruction.static_table.size() != 33u) {
                        break;
                    }
                    auto direction = safe_normalize(
                        vector(
                            instruction.operand(operand::sky::direction),
                            result),
                        make_float3(0.0f, 0.0f, 1.0f));
                    const auto sun_direction = make_float3(
                        instruction.static_table[0u],
                        instruction.static_table[1u],
                        instruction.static_table[2u]);
                    auto theta = min(
                        acos(clamp(direction.z, -1.0f, 1.0f)),
                        0.5f * pi - 0.001f);
                    auto gamma = acos(clamp(
                        dot(direction, sun_direction),
                        -1.0f,
                        1.0f));
                    const auto radiance = make_float3(
                        instruction.static_table[3u],
                        instruction.static_table[4u],
                        instruction.static_table[5u]);
                    const auto sky_channel =
                        [&](std::size_t channel) noexcept {
                            const auto offset =
                                6u + channel * 9u;
                            const auto ctheta = cos(theta);
                            const auto cgamma = cos(gamma);
                            const auto ray = cgamma * cgamma;
                            const auto g = instruction.static_table[
                                offset + 8u];
                            const auto mie =
                                (1.0f + ray) /
                                pow(
                                    max(
                                        1.0f + g * g -
                                            2.0f * g * cgamma,
                                        1.0e-20f),
                                    1.5f);
                            return
                                (1.0f +
                                 instruction.static_table[offset] *
                                     exp(
                                         instruction.static_table[
                                             offset + 1u] /
                                         (ctheta + 0.01f))) *
                                (instruction.static_table[
                                     offset + 2u] +
                                 instruction.static_table[
                                     offset + 3u] *
                                     exp(
                                         instruction.static_table[
                                             offset + 4u] *
                                         gamma) +
                                 instruction.static_table[
                                     offset + 5u] *
                                     ray +
                                 instruction.static_table[
                                     offset + 6u] *
                                     mie +
                                 instruction.static_table[
                                     offset + 7u] *
                                     sqrt(max(ctheta, 0.0f)));
                        };
                    const auto xyz = make_float3(
                        sky_channel(0u) * radiance.x,
                        sky_channel(1u) * radiance.y,
                        sky_channel(2u) * radiance.z);
                    value = make_float4(
                        max(
                            services.xyz_to_rgb(xyz),
                            make_float3(0.0f)) *
                            (two_pi / 683.0f),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::nishita_sky: {
                    auto direction = safe_normalize(
                        vector(
                            instruction.operand(
                                operand::nishita_sky::direction),
                            result),
                        make_float3(0.0f, 0.0f, 1.0f));
                    value = make_float4(
                        services.nishita_sky(
                            point.parameter_block,
                            static_cast<std::uint32_t>(
                                instruction.static_u0),
                            direction,
                            scalar(
                                instruction.operand(
                                    operand::nishita_sky::elevation),
                                result),
                            scalar(
                                instruction.operand(
                                    operand::nishita_sky::rotation),
                                result),
                            scalar(
                                instruction.operand(
                                    operand::nishita_sky::size),
                                result),
                            scalar(
                                instruction.operand(
                                    operand::nishita_sky::intensity),
                                result)),
                        1.0f);
                    break;
                }
            default:
                break;
        }
        return project_surface_value(
            instruction.result_type, value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_procedural_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (!supports_procedural_value(instruction.operation)) {
        return nullptr;
    }
    return std::make_unique<ProceduralValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail
