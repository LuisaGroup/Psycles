#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_noise.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

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
        case compiler::ValueOperation::nishita_sky:
            return true;
        default:
            return false;
    }
}

class ProceduralValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] Float4 evaluate(
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
                    auto scale =
                        scalar(instruction.b, result);
                    value = cycles_noise::
                        evaluate_texture_shared(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        noise_type,
                        normalize,
                        color_needed,
                        vector(instruction.a, result) * scale,
                        scalar(instruction.g, result) * scale,
                        scalar(instruction.c, result),
                        scalar(instruction.d, result),
                        scalar(instruction.e, result),
                        scalar(instruction.h, result),
                        scalar(instruction.i, result),
                        scalar(instruction.f, result));
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
                            vector(instruction.a, result),
                            scalar(instruction.b, result));
                    break;
                }
                case compiler::ValueOperation::checker_color:
                case compiler::ValueOperation::checker_factor: {
                    auto scaled_raw =
                        vector(instruction.a, result) *
                        scalar(instruction.d, result);
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
                                    instruction.c, result),
                                vector(
                                    instruction.b, result),
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
                        vector(instruction.a, result) *
                        scalar(instruction.e, result);
                    auto mortar_size = max(
                        scalar(instruction.f, result),
                        0.0f);
                    auto mortar_smooth = max(
                        scalar(instruction.g, result),
                        0.0f);
                    auto bias =
                        scalar(instruction.h, result);
                    auto brick_width = max(
                        abs(scalar(instruction.i, result)),
                        1.0e-20f);
                    auto row_height = max(
                        abs(scalar(instruction.j, result)),
                        1.0e-20f);
                    auto row = cast<int>(
                        floor(p.y / row_height));
                    Float offset = 0.0f;
                    if (instruction.static_u0 != 0u &&
                        instruction.static_u1 != 0u) {
                        auto squash_row =
                            (row %
                             static_cast<int>(
                                 instruction.static_u1)) ==
                            0;
                        brick_width *= select(
                            1.0f,
                            instruction.static_f1,
                            squash_row);
                        auto offset_row =
                            (row %
                             static_cast<int>(
                                 instruction.static_u0)) ==
                            0;
                        offset = select(
                            0.0f,
                            brick_width *
                                instruction.static_f0,
                            offset_row);
                    }
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
                            vector(instruction.b, result),
                            vector(instruction.c, result),
                            tint);
                        value = make_float4(
                            lerp(
                                brick_color,
                                vector(
                                    instruction.d, result),
                                mortar),
                            1.0f);
                    }
                    break;
                }
                case compiler::ValueOperation::gradient: {
                    auto p = vector(instruction.a, result);
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
                    auto factor =
                        scalar(instruction.a, result);
                    Float3 color = make_float3(0.0f);
                    Float alpha = 1.0f;
                    const auto count =
                        instruction.static_table.size() / 5u;
                    if (
                        count >= 2u &&
                        (instruction.static_u0 & 2u) != 0u) {
                        std::vector<luisa::float4> samples;
                        samples.reserve(count);
                        for (std::size_t i = 0u;
                             i < count;
                             ++i) {
                            samples.emplace_back(
                                instruction.static_table[
                                    i * 5u + 1u],
                                instruction.static_table[
                                    i * 5u + 2u],
                                instruction.static_table[
                                    i * 5u + 3u],
                                instruction.static_table[
                                    i * 5u + 4u]);
                        }
                        luisa::compute::Constant<luisa::float4>
                            table{samples};
                        auto scaled =
                            clamp(factor, 0.0f, 1.0f) *
                            static_cast<float>(count - 1u);
                        auto index = min(
                            cast<luisa::uint>(scaled),
                            static_cast<std::uint32_t>(
                                count - 1u));
                        auto t =
                            scaled - cast<float>(index);
                        auto sampled = table.read(index);
                        if ((instruction.static_u0 & 1u) == 0u) {
                            auto next = table.read(min(
                                index + 1u,
                                static_cast<std::uint32_t>(
                                    count - 1u)));
                            sampled = select(
                                sampled,
                                lerp(sampled, next, t),
                                t > 0.0f);
                        }
                        color = sampled.xyz();
                        alpha = sampled.w;
                    } else if (count != 0u) {
                        color = make_float3(
                            instruction.static_table[1u],
                            instruction.static_table[2u],
                            instruction.static_table[3u]);
                        alpha = instruction.static_table[4u];
                        for (std::size_t i = 1u; i < count; ++i) {
                            const auto p0 =
                                instruction.static_table[
                                    (i - 1u) * 5u];
                            const auto p1 =
                                instruction.static_table[i * 5u];
                            auto t = clamp(
                                (factor - p0) /
                                    std::max(p1 - p0, 1.0e-20f),
                                0.0f,
                                1.0f);
                            if (
                                (instruction.static_u0 & 1u) !=
                                0u) {
                                t = 0.0f;
                            }
                            auto c0 = make_float3(
                                instruction.static_table[
                                    (i - 1u) * 5u + 1u],
                                instruction.static_table[
                                    (i - 1u) * 5u + 2u],
                                instruction.static_table[
                                    (i - 1u) * 5u + 3u]);
                            auto c1 = make_float3(
                                instruction.static_table[
                                    i * 5u + 1u],
                                instruction.static_table[
                                    i * 5u + 2u],
                                instruction.static_table[
                                    i * 5u + 3u]);
                            auto a0 =
                                instruction.static_table[
                                    (i - 1u) * 5u + 4u];
                            auto a1 =
                                instruction.static_table[
                                    i * 5u + 4u];
                            auto use =
                                factor >= p0;
                            color = select(
                                color, lerp(c0, c1, t), use);
                            alpha = select(
                                alpha, lerp(a0, a1, t), use);
                        }
                        const auto last =
                            (count - 1u) * 5u;
                        auto use_last =
                            factor >=
                            instruction.static_table[last];
                        color = select(
                            color,
                            make_float3(
                                instruction.static_table[
                                    last + 1u],
                                instruction.static_table[
                                    last + 2u],
                                instruction.static_table[
                                    last + 3u]),
                            use_last);
                        alpha = select(
                            alpha,
                            instruction.static_table[last + 4u],
                            use_last);
                    }
                    value =
                        instruction.static_u1 != 0u
                            ? make_float4(alpha)
                            : make_float4(color, alpha);
                    break;
                }
                case compiler::ValueOperation::rgb_curve: {
                    auto input = vector(instruction.a, result);
                    auto factor = scalar(instruction.b, result);
                    Float3 mapped = input;
                    const auto count =
                        instruction.static_table.size() / 4u;
                    if (
                        count >= 2u &&
                        (instruction.static_u0 & 1u) != 0u) {
                        std::vector<luisa::float3> samples;
                        samples.reserve(count);
                        for (std::size_t i = 0u;
                             i < count;
                             ++i) {
                            samples.emplace_back(
                                instruction.static_table[
                                    i * 4u + 1u],
                                instruction.static_table[
                                    i * 4u + 2u],
                                instruction.static_table[
                                    i * 4u + 3u]);
                        }
                        luisa::compute::Constant<luisa::float3>
                            table{samples};
                        const auto component =
                            [](Float3 value,
                               std::uint32_t channel) {
                                return channel == 0u
                                           ? value.x
                                           : channel == 1u
                                                 ? value.y
                                                 : value.z;
                            };
                        const auto lookup =
                            [&](Float coordinate,
                                std::uint32_t channel) {
                                auto scaled =
                                    clamp(
                                        coordinate,
                                        0.0f,
                                        1.0f) *
                                    static_cast<float>(
                                        count - 1u);
                                auto index = min(
                                    cast<luisa::uint>(scaled),
                                    static_cast<std::uint32_t>(
                                        count - 1u));
                                auto t =
                                    scaled - cast<float>(index);
                                auto sampled = component(
                                    table.read(index), channel);
                                auto next = component(
                                    table.read(min(
                                        index + 1u,
                                        static_cast<
                                            std::uint32_t>(
                                            count - 1u))),
                                    channel);
                                sampled = select(
                                    sampled,
                                    lerp(sampled, next, t),
                                    t > 0.0f);
                                if (
                                    (instruction.static_u0 &
                                     2u) != 0u) {
                                    auto first = component(
                                        table.read(0u),
                                        channel);
                                    auto second = component(
                                        table.read(1u),
                                        channel);
                                    auto last = component(
                                        table.read(
                                            static_cast<
                                                std::uint32_t>(
                                                count - 1u)),
                                        channel);
                                    auto previous = component(
                                        table.read(
                                            static_cast<
                                                std::uint32_t>(
                                                count - 2u)),
                                        channel);
                                    auto below =
                                        first +
                                        (first - second) *
                                            (-coordinate) *
                                            static_cast<float>(
                                                count - 1u);
                                    auto above =
                                        last +
                                        (last - previous) *
                                            (coordinate - 1.0f) *
                                            static_cast<float>(
                                                count - 1u);
                                    sampled = select(
                                        sampled,
                                        below,
                                        coordinate < 0.0f);
                                    sampled = select(
                                        sampled,
                                        above,
                                        coordinate > 1.0f);
                                }
                                return sampled;
                            };
                        const auto range =
                            instruction.static_f1 -
                            instruction.static_f0;
                        auto relative =
                            (input -
                             instruction.static_f0) /
                            range;
                        mapped = make_float3(
                            lookup(relative.x, 0u),
                            lookup(relative.y, 1u),
                            lookup(relative.z, 2u));
                    } else if (count >= 2u) {
                        mapped = make_float3(0.0f);
                        for (std::size_t i = 1u; i < count; ++i) {
                            const auto x0 =
                                instruction.static_table[
                                    (i - 1u) * 4u];
                            const auto x1 =
                                instruction.static_table[i * 4u];
                            auto t = clamp(
                                (input - x0) /
                                    std::max(x1 - x0, 1.0e-20f),
                                make_float3(0.0f),
                                make_float3(1.0f));
                            auto y0 = make_float3(
                                instruction.static_table[
                                    (i - 1u) * 4u + 1u],
                                instruction.static_table[
                                    (i - 1u) * 4u + 2u],
                                instruction.static_table[
                                    (i - 1u) * 4u + 3u]);
                            auto y1 = make_float3(
                                instruction.static_table[
                                    i * 4u + 1u],
                                instruction.static_table[
                                    i * 4u + 2u],
                                instruction.static_table[
                                    i * 4u + 3u]);
                            mapped = select(
                                mapped,
                                lerp(y0, y1, t),
                                input >= x0);
                        }
                    }
                    value = make_float4(
                        lerp(input, mapped, factor),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::separate_r:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .x);
                    break;
                case compiler::ValueOperation::separate_g:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .y);
                    break;
                case compiler::ValueOperation::separate_b:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .z);
                    break;
                case compiler::ValueOperation::combine_color: {
                    auto channels = make_float3(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result),
                        scalar(instruction.c, result));
                    value = make_float4(
                        combine_color(
                            channels,
                            instruction.static_u0),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::nishita_sky: {
                    auto direction = safe_normalize(
                        vector(instruction.i, result),
                        make_float3(0.0f, 0.0f, 1.0f));
                    value = make_float4(
                        services.nishita_sky(
                            point.parameter_block,
                            static_cast<std::uint32_t>(
                                instruction.static_u0),
                            direction,
                            scalar(instruction.a, result),
                            scalar(instruction.b, result),
                            scalar(instruction.c, result),
                            scalar(instruction.d, result)),
                        1.0f);
                    break;
                }
            default:
                break;
        }
        return value;
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
