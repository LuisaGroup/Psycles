#include "graph_surface_internal.h"

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/cycles_wave.h>

#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

enum class WaveType : std::uint8_t {
    bands,
    rings
};

enum class WaveDirection : std::uint8_t {
    x,
    y,
    z,
    fourth
};

enum class WaveProfile : std::uint8_t {
    sine,
    saw,
    triangle
};

struct WaveConfiguration {
    WaveType type;
    WaveDirection bands_direction;
    WaveDirection rings_direction;
    WaveProfile profile;
};

[[nodiscard]] WaveConfiguration decode_configuration(
    std::uint64_t encoded) noexcept {
    return {
        .type = static_cast<WaveType>(encoded & 0xffu),
        .bands_direction = static_cast<WaveDirection>(
            (encoded >> 8u) & 0xffu),
        .rings_direction = static_cast<WaveDirection>(
            (encoded >> 16u) & 0xffu),
        .profile = static_cast<WaveProfile>(
            (encoded >> 24u) & 0xffu)};
}

[[nodiscard]] WaveConfiguration decode_svm_configuration(
    std::uint32_t encoded) noexcept {
    return {
        .type = static_cast<WaveType>(
            encoded & compiler::surface_value_wave_type_mask),
        .bands_direction = static_cast<WaveDirection>(
            (encoded & compiler::surface_value_wave_bands_direction_mask) >>
            compiler::surface_value_wave_bands_direction_shift),
        .rings_direction = static_cast<WaveDirection>(
            (encoded & compiler::surface_value_wave_rings_direction_mask) >>
            compiler::surface_value_wave_rings_direction_shift),
        .profile = static_cast<WaveProfile>(
            (encoded & compiler::surface_value_wave_profile_mask) >>
            compiler::surface_value_wave_profile_shift)};
}

[[nodiscard]] Float wave_coordinate(
    Float3 point,
    const WaveConfiguration &configuration) noexcept {
    if (configuration.type == WaveType::bands) {
        switch (configuration.bands_direction) {
            case WaveDirection::x:
                return point.x * 20.0f;
            case WaveDirection::y:
                return point.y * 20.0f;
            case WaveDirection::z:
                return point.z * 20.0f;
            case WaveDirection::fourth:
                return (point.x + point.y + point.z) * 10.0f;
        }
    }
    auto radial = point;
    switch (configuration.rings_direction) {
        case WaveDirection::x:
            radial *= make_float3(0.0f, 1.0f, 1.0f);
            break;
        case WaveDirection::y:
            radial *= make_float3(1.0f, 0.0f, 1.0f);
            break;
        case WaveDirection::z:
            radial *= make_float3(1.0f, 1.0f, 0.0f);
            break;
        case WaveDirection::fourth:
            break;
    }
    return sqrt(dot(radial, radial)) * 20.0f;
}

[[nodiscard]] Float wave_profile(
    Float coordinate,
    WaveProfile profile) noexcept {
    if (profile == WaveProfile::sine) {
        return 0.5f + 0.5f * sin(coordinate - 0.5f * pi);
    }
    coordinate /= two_pi;
    if (profile == WaveProfile::saw) {
        return coordinate - floor(coordinate);
    }
    return abs(coordinate - floor(coordinate + 0.5f)) * 2.0f;
}

[[nodiscard]] Float wave_coordinate_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float3 point) noexcept {
    constexpr auto coordinate_mask =
        compiler::surface_value_wave_type_mask |
        compiler::surface_value_wave_bands_direction_mask |
        compiler::surface_value_wave_rings_direction_mask;
    std::vector<std::uint16_t> active_shapes;
    active_shapes.reserve(immediate_domain.size());
    for (const auto encoded : immediate_domain) {
        const auto shape = static_cast<std::uint16_t>(
            encoded & coordinate_mask);
        if (std::find(active_shapes.begin(), active_shapes.end(), shape) ==
            active_shapes.end()) {
            active_shapes.emplace_back(shape);
        }
    }
    Float coordinate = 0.0f;
    luisa::compute::detail::SwitchStmtBuilder{
        immediate & coordinate_mask} % [&] {
        for (const auto shape : active_shapes) {
            luisa::compute::detail::SwitchCaseStmtBuilder{shape} %
                [&, shape] {
                    coordinate = wave_coordinate(
                        point, decode_svm_configuration(shape));
                };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid compact surface Wave coordinate shape");
        };
    };
    return coordinate;
}

[[nodiscard]] Float wave_profile_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float coordinate) noexcept {
    std::vector<std::uint16_t> active_profiles;
    active_profiles.reserve(immediate_domain.size());
    for (const auto encoded : immediate_domain) {
        const auto profile = static_cast<std::uint16_t>(
            encoded & compiler::surface_value_wave_profile_mask);
        if (std::find(active_profiles.begin(), active_profiles.end(),
                      profile) == active_profiles.end()) {
            active_profiles.emplace_back(profile);
        }
    }
    Float factor = 0.0f;
    luisa::compute::detail::SwitchStmtBuilder{
        immediate & compiler::surface_value_wave_profile_mask} % [&] {
        for (const auto profile : active_profiles) {
            luisa::compute::detail::SwitchCaseStmtBuilder{profile} %
                [&, profile] {
                    factor = wave_profile(
                        coordinate,
                        decode_svm_configuration(profile).profile);
                };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid compact surface Wave profile");
        };
    };
    return factor;
}

class WaveValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        auto point =
            vector(
                instruction.operand(operand::wave::vector),
                context.result) *
            scalar(
                instruction.operand(operand::wave::scale),
                context.result);
        // Exact Cycles precision correction from svm_wave(). Ordering is
        // intentionally preserved because unit-coordinate boundaries are
        // visible in the saw and triangle profiles.
        point = (point + 0.000001f) * 0.999999f;
        auto coordinate =
            context.svm_immediate_override != nullptr
                ? wave_coordinate_svm(
                      *context.svm_immediate_override,
                      context.svm_immediate_domain,
                      point)
                : wave_coordinate(
                      point,
                      decode_configuration(instruction.static_u0));
        coordinate += scalar(
            instruction.operand(operand::wave::phase), context.result);
        auto distortion = scalar(
            instruction.operand(operand::wave::distortion),
            context.result);
        $if (distortion != 0.0f) {
            coordinate +=
                distortion *
                (cycles_wave::distortion_noise(
                     point * scalar(
                                 instruction.operand(
                                     operand::wave::detail_scale),
                                 context.result),
                     scalar(
                         instruction.operand(operand::wave::detail),
                         context.result),
                     scalar(
                         instruction.operand(
                             operand::wave::detail_roughness),
                         context.result)) *
                     2.0f -
                 1.0f);
        };
        const auto factor =
            context.svm_immediate_override != nullptr
                ? wave_profile_svm(
                      *context.svm_immediate_override,
                      context.svm_immediate_domain,
                      coordinate)
                : wave_profile(
                      coordinate,
                      decode_configuration(instruction.static_u0).profile);
        const auto value =
            instruction.operation == compiler::ValueOperation::wave_color
                ? make_float4(make_float3(factor), 1.0f)
                : make_float4(factor);
        return project_surface_value(
            instruction.result_type,
            value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_wave_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (instruction.operation != compiler::ValueOperation::wave_color &&
        instruction.operation != compiler::ValueOperation::wave_factor) {
        return nullptr;
    }
    return std::make_unique<WaveValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail
