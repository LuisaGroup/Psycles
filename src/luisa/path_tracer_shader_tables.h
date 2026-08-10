#pragma once

#include "path_tracer_internal.h"

#include <optional>

namespace psycles::luisa_backend::detail {

using ColorRampTableCallable =
    Callable<luisa::float4(
        Buffer<float>,
        luisa::uint3,
        float)>;
using SampledRgbCurveTableCallable =
    Callable<luisa::float3(
        Buffer<float>,
        luisa::uint3,
        luisa::float3,
        float,
        float,
        float,
        float)>;
using ControlRgbCurveTableCallable =
    Callable<luisa::float3(
        Buffer<float>,
        luisa::uint3,
        luisa::float3,
        float)>;

[[nodiscard]] ColorRampTableCallable
make_color_ramp_sampled_linear_callable() noexcept;
[[nodiscard]] ColorRampTableCallable
make_color_ramp_sampled_constant_callable() noexcept;
[[nodiscard]] ColorRampTableCallable
make_color_ramp_control_linear_callable() noexcept;
[[nodiscard]] ColorRampTableCallable
make_color_ramp_control_constant_callable() noexcept;
[[nodiscard]] SampledRgbCurveTableCallable
make_rgb_curve_sampled_callable() noexcept;
[[nodiscard]] ControlRgbCurveTableCallable
make_rgb_curve_control_callable() noexcept;

template<typename ScalarBuffer>
class CallableSurfaceShaderTableProvider final
    : public SurfaceShaderTableProvider {

private:
    const ScalarBuffer &_data;
    mutable std::optional<ColorRampTableCallable>
        _color_ramp_sampled_linear;
    mutable std::optional<ColorRampTableCallable>
        _color_ramp_sampled_constant;
    mutable std::optional<ColorRampTableCallable>
        _color_ramp_control_linear;
    mutable std::optional<ColorRampTableCallable>
        _color_ramp_control_constant;
    mutable std::optional<SampledRgbCurveTableCallable>
        _rgb_curve_sampled;
    mutable std::optional<ControlRgbCurveTableCallable>
        _rgb_curve_control;

    [[nodiscard]] static luisa::compute::UInt3 descriptor(
        const SurfaceShaderTableView &table) noexcept {
        return make_uint3(
            table.offset,
            table.count,
            table.width);
    }

public:
    explicit CallableSurfaceShaderTableProvider(
        const ScalarBuffer &data) noexcept
        : _data{data} {}

    [[nodiscard]] Float4 color_ramp_sampled_linear(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept override {
        if (!_color_ramp_sampled_linear) {
            _color_ramp_sampled_linear.emplace(
                make_color_ramp_sampled_linear_callable());
        }
        return (*_color_ramp_sampled_linear)(
            _data, descriptor(table), factor);
    }

    [[nodiscard]] Float4 color_ramp_sampled_constant(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept override {
        if (!_color_ramp_sampled_constant) {
            _color_ramp_sampled_constant.emplace(
                make_color_ramp_sampled_constant_callable());
        }
        return (*_color_ramp_sampled_constant)(
            _data, descriptor(table), factor);
    }

    [[nodiscard]] Float4 color_ramp_control_linear(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept override {
        if (!_color_ramp_control_linear) {
            _color_ramp_control_linear.emplace(
                make_color_ramp_control_linear_callable());
        }
        return (*_color_ramp_control_linear)(
            _data, descriptor(table), factor);
    }

    [[nodiscard]] Float4 color_ramp_control_constant(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept override {
        if (!_color_ramp_control_constant) {
            _color_ramp_control_constant.emplace(
                make_color_ramp_control_constant_callable());
        }
        return (*_color_ramp_control_constant)(
            _data, descriptor(table), factor);
    }

    [[nodiscard]] Float3 rgb_curve_sampled(
        const SurfaceShaderTableView &table,
        Float3 input,
        Float factor,
        Float min_x,
        Float max_x,
        Float extrapolate) const noexcept override {
        if (!_rgb_curve_sampled) {
            _rgb_curve_sampled.emplace(
                make_rgb_curve_sampled_callable());
        }
        return (*_rgb_curve_sampled)(
            _data,
            descriptor(table),
            input,
            factor,
            min_x,
            max_x,
            extrapolate);
    }

    [[nodiscard]] Float3 rgb_curve_control(
        const SurfaceShaderTableView &table,
        Float3 input,
        Float factor) const noexcept override {
        if (!_rgb_curve_control) {
            _rgb_curve_control.emplace(
                make_rgb_curve_control_callable());
        }
        return (*_rgb_curve_control)(
            _data,
            descriptor(table),
            input,
            factor);
    }
};

}// namespace psycles::luisa_backend::detail
