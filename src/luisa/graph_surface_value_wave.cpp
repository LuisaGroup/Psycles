#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_wave.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

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

class WaveValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto configuration =
            decode_configuration(instruction.static_u0);
        auto point =
            vector(instruction.a, context.result) *
            scalar(instruction.b, context.result);
        // Exact Cycles precision correction from svm_wave(). Ordering is
        // intentionally preserved because unit-coordinate boundaries are
        // visible in the saw and triangle profiles.
        point = (point + 0.000001f) * 0.999999f;
        auto coordinate = wave_coordinate(point, configuration) +
                          scalar(instruction.g, context.result);
        auto distortion = scalar(instruction.c, context.result);
        $if (distortion != 0.0f) {
            coordinate +=
                distortion *
                (cycles_wave::distortion_noise(
                     point * scalar(instruction.e, context.result),
                     scalar(instruction.d, context.result),
                     scalar(instruction.f, context.result)) *
                     2.0f -
                 1.0f);
        };
        const auto factor = wave_profile(
            coordinate,
            configuration.profile);
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
