#include "graph_surface_internal.h"
#include "surface_shader_table_evaluation.h"

#include <luisa/dsl/sugar.h>

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/cycles_noise.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace luisa::compute;

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

template <std::size_t DomainSize, typename Evaluate>
void dispatch_surface_value_immediate(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    std::uint32_t mask, Evaluate &&evaluate) noexcept {
  std::array<bool, DomainSize> active{};
  for (const auto encoded : immediate_domain) {
    const auto value = static_cast<std::uint32_t>(encoded) & mask;
    if (value >= active.size()) {
      std::abort();
    }
    active[value] = true;
  }
  luisa::compute::detail::SwitchStmtBuilder{immediate & mask} % [&] {
    for (auto value = std::size_t{}; value < active.size(); ++value) {
      if (!active[value]) {
        continue;
      }
      luisa::compute::detail::SwitchCaseStmtBuilder{
          static_cast<luisa::uint>(value)} %
          [&, value] { evaluate(static_cast<std::uint32_t>(value)); };
    }
    luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
      luisa::compute::dsl::unreachable(
          "invalid compact surface opcode immediate");
    };
  };
}

[[nodiscard]] Float evaluate_gradient_mode(
    Float3 point, std::uint32_t mode) noexcept {
    switch (mode) {
        case 1u: {
            auto gradient = max(point.x, 0.0f);
            return gradient * gradient;
        }
        case 2u: {
            auto t = clamp(point.x, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }
        case 3u:
            return (point.x + point.y) * 0.5f;
        case 4u:
            return atan2(point.y, point.x) / two_pi + 0.5f;
        case 5u:
            return max(0.999999f - length(point), 0.0f);
        case 6u: {
            auto gradient = max(0.999999f - length(point), 0.0f);
            return gradient * gradient;
        }
        case 0u:
        default:
            return point.x;
    }
}

[[nodiscard]] Float evaluate_gradient_svm(
    UInt mode,
    std::span<const std::uint16_t> immediate_domain,
    Float3 point) noexcept {
    std::array<bool, 7u> active{};
    for (const auto encoded : immediate_domain) {
        if (encoded >= active.size()) {
            std::abort();
        }
        active[encoded] = true;
    }
    Float gradient = 0.0f;
    luisa::compute::detail::SwitchStmtBuilder{mode} % [&] {
        for (auto index = std::size_t{0u}; index < active.size(); ++index) {
            if (!active[index]) {
                continue;
            }
            luisa::compute::detail::SwitchCaseStmtBuilder{
                static_cast<luisa::uint>(index)} %
                [&, index] {
                    gradient = evaluate_gradient_mode(
                        point, static_cast<std::uint32_t>(index));
                };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid compact surface Gradient mode");
        };
    };
    return gradient;
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
                    auto scale = scalar(
                        instruction.operand(operand::noise::scale), result);
                    const auto evaluate = [&, color_needed](
                                              std::uint32_t dimensions,
                                              cycles_noise::Type noise_type,
                                              auto &&normalize) noexcept {
                        return cycles_noise::
                            evaluate_texture_shared(
                            dimensions,
                            noise_type,
                            std::forward<decltype(normalize)>(normalize),
                            color_needed,
                            vector(
                                instruction.operand(operand::noise::vector),
                                result) * scale,
                            scalar(
                                instruction.operand(operand::noise::w),
                                result) * scale,
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
                                instruction.operand(operand::noise::gain),
                                result),
                            scalar(
                                instruction.operand(operand::noise::distortion),
                                result));
                    };
                    if (context.svm_immediate_override != nullptr) {
                        constexpr auto shape_mask =
                            compiler::surface_value_noise_dimensions_mask |
                            compiler::surface_value_noise_type_mask;
                        std::vector<std::uint16_t> active_shapes;
                        active_shapes.reserve(
                            context.svm_immediate_domain.size());
                        for (const auto encoded :
                             context.svm_immediate_domain) {
                            const auto shape = static_cast<std::uint16_t>(
                                encoded & shape_mask);
                            if (std::find(active_shapes.begin(),
                                          active_shapes.end(),
                                          shape) == active_shapes.end()) {
                                active_shapes.emplace_back(shape);
                            }
                        }
                        const auto immediate =
                            *context.svm_immediate_override;
                        const auto shape = immediate & shape_mask;
                        const auto normalize =
                            (immediate & compiler::
                                              surface_value_noise_normalize_immediate_bit) !=
                            0u;
                        luisa::compute::detail::SwitchStmtBuilder{shape} % [&] {
                            for (const auto active_shape : active_shapes) {
                                luisa::compute::detail::SwitchCaseStmtBuilder{
                                    active_shape} %
                                    [&, active_shape] {
                                        const auto dimensions =
                                            (active_shape &
                                             compiler::
                                                 surface_value_noise_dimensions_mask) >>
                                            compiler::
                                                surface_value_noise_dimensions_shift;
                                        const auto type =
                                            (active_shape &
                                             compiler::
                                                 surface_value_noise_type_mask) >>
                                            compiler::
                                                surface_value_noise_type_shift;
                                        value = evaluate(
                                            dimensions,
                                            static_cast<cycles_noise::Type>(
                                                type),
                                            normalize);
                                    };
                            }
                            luisa::compute::detail::SwitchDefaultStmtBuilder{} %
                                [] {
                                    luisa::compute::dsl::unreachable(
                                        "invalid compact surface Noise shape");
                                };
                        };
                    } else {
                        value = evaluate(
                            static_cast<std::uint32_t>(instruction.static_u0),
                            static_cast<cycles_noise::Type>(
                                (instruction.static_u1 >> 8u) & 0xffu),
                            (instruction.static_u1 & 1u) != 0u);
                    }
                    break;
                }
                case compiler::ValueOperation::white_noise_value:
                case compiler::ValueOperation::white_noise_color: {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::
                            white_noise_color;
                    const auto vector_value = vector(
                        instruction.operand(operand::white_noise::vector),
                        result);
                    const auto w = scalar(
                        instruction.operand(operand::white_noise::w), result);
                    if (context.svm_immediate_override != nullptr) {
                      dispatch_surface_value_immediate<5u>(
                          *context.svm_immediate_override,
                          context.svm_immediate_domain,
                          compiler::surface_value_white_noise_dimensions_mask,
                          [&](std::uint32_t dimensions) noexcept {
                            if (dimensions == 0u) {
                              std::abort();
                            }
                            value = cycles_noise::evaluate_white_shared(
                                dimensions, color_needed, vector_value, w);
                          });
                    } else {
                      value = cycles_noise::evaluate_white_shared(
                          static_cast<std::uint32_t>(instruction.static_u0),
                          color_needed, vector_value, w);
                    }
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
                    auto gradient =
                        context.svm_immediate_override != nullptr
                            ? evaluate_gradient_svm(
                                  *context.svm_immediate_override,
                                  context.svm_immediate_domain,
                                  p)
                            : evaluate_gradient_mode(
                                  p,
                                  static_cast<std::uint32_t>(
                                      instruction.static_u0));
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
                    const auto table =
                        surface_shader_table_view(services, point, parameter);
                    const auto ramp =
                        context.svm_immediate_override != nullptr
                            ? evaluate_surface_color_ramp_svm(
                                  services, *context.svm_immediate_override,
                                  context.svm_immediate_domain, table, factor)
                            : evaluate_surface_color_ramp(
                                  services, table, factor,
                                  static_cast<std::uint32_t>(
                                      instruction.static_u0));
                    const auto alpha_output =
                        context.svm_immediate_override != nullptr
                            ? surface_value_category(instruction.result_type) ==
                                  SurfaceValueCategory::scalar
                            : instruction.static_u1 != 0u;
                    value = alpha_output ? make_float4(ramp.w) : ramp;
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
                    const auto table =
                        surface_shader_table_view(services, point, parameter);
                    const auto min_x = scalar(
                        instruction.operand(operand::rgb_curve::min_x), result);
                    const auto max_x = scalar(
                        instruction.operand(operand::rgb_curve::max_x), result);
                    const auto extrapolate = scalar(
                        instruction.operand(operand::rgb_curve::extrapolate),
                        result);
                    Float3 mapped = make_float3(0.0f);
                    const auto evaluate_curve = [&](bool sampled) noexcept {
                      mapped = sampled ? rgb_curve_sampled(services, table,
                                                           input, factor, min_x,
                                                           max_x, extrapolate)
                                       : rgb_curve_control(services, table,
                                                           input, factor);
                    };
                    if (context.svm_immediate_override != nullptr) {
                      dispatch_surface_value_immediate<2u>(
                          *context.svm_immediate_override,
                          context.svm_immediate_domain,
                          compiler::surface_value_rgb_curve_sampled_bit,
                          [&](std::uint32_t sampled) noexcept {
                            evaluate_curve(sampled != 0u);
                          });
                    } else {
                      evaluate_curve((instruction.static_u0 & 1u) != 0u);
                    }
                    value = make_float4(mapped, 1.0f);
                    break;
                }
                case compiler::ValueOperation::separate_r:
                case compiler::ValueOperation::separate_g:
                case compiler::ValueOperation::separate_b: {
                  const auto color = vector(
                      instruction.operand(operand::separate_color::color),
                      result);
                  Float3 channels = make_float3(0.0f);
                  if (context.svm_immediate_override != nullptr) {
                    dispatch_surface_value_immediate<3u>(
                        *context.svm_immediate_override,
                        context.svm_immediate_domain,
                        compiler::surface_value_color_mode_mask,
                        [&](std::uint32_t mode) noexcept {
                          channels = separate_color(services, color, mode);
                        });
                  } else {
                    channels =
                        separate_color(services, color, instruction.static_u0);
                  }
                  const auto channel =
                      instruction.operation ==
                              compiler::ValueOperation::separate_r
                          ? channels.x
                      : instruction.operation ==
                              compiler::ValueOperation::separate_g
                          ? channels.y
                          : channels.z;
                  value = make_float4(channel);
                  break;
                }
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
                    Float3 combined = make_float3(0.0f);
                    if (context.svm_immediate_override != nullptr) {
                      dispatch_surface_value_immediate<3u>(
                          *context.svm_immediate_override,
                          context.svm_immediate_domain,
                          compiler::surface_value_color_mode_mask,
                          [&](std::uint32_t mode) noexcept {
                            combined = combine_color(services, channels, mode);
                          });
                    } else {
                      combined = combine_color(services, channels,
                                               instruction.static_u0);
                    }
                    value = make_float4(combined, 1.0f);
                    break;
                }
                case compiler::ValueOperation::hosek_wilkie_sky: {
                    if (instruction.static_table.size() != 33u) {
                        break;
                    }
                    const auto table = [&](std::size_t index) noexcept {
                        return value_static_table_entry(
                            context, instruction, index);
                    };
                    auto direction = safe_normalize(
                        vector(
                            instruction.operand(operand::sky::direction),
                            result),
                        make_float3(0.0f, 0.0f, 1.0f));
                    const auto sun_direction = make_float3(
                        table(0u), table(1u), table(2u));
                    auto theta = min(
                        acos(clamp(direction.z, -1.0f, 1.0f)),
                        0.5f * pi - 0.001f);
                    auto gamma = acos(clamp(
                        dot(direction, sun_direction),
                        -1.0f,
                        1.0f));
                    const auto radiance = make_float3(
                        table(3u), table(4u), table(5u));
                    const auto sky_channel =
                        [&](std::size_t channel) noexcept {
                            const auto offset =
                                6u + channel * 9u;
                            const auto ctheta = cos(theta);
                            const auto cgamma = cos(gamma);
                            const auto ray = cgamma * cgamma;
                            const auto g = table(offset + 8u);
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
                                 table(offset) *
                                     exp(
                                         table(offset + 1u) /
                                         (ctheta + 0.01f))) *
                                (table(offset + 2u) +
                                 table(offset + 3u) *
                                     exp(
                                         table(offset + 4u) *
                                         gamma) +
                                 table(offset + 5u) *
                                     ray +
                                 table(offset + 6u) *
                                     mie +
                                 table(offset + 7u) *
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
                            context.static_u0_override != nullptr
                                ? Expr<std::uint32_t>{context
                                                          .static_u0_override
                                                          ->expression()}
                                : Expr<std::uint32_t>{static_cast<
                                      std::uint32_t>(instruction.static_u0)},
                            direction,
                            scalar(instruction.operand(
                                       operand::nishita_sky::elevation),
                                   result),
                            scalar(instruction.operand(
                                       operand::nishita_sky::rotation),
                                   result),
                            scalar(
                                instruction.operand(operand::nishita_sky::size),
                                result),
                            scalar(instruction.operand(
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
