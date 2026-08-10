#include "path_tracer_shader_tables.h"

#include "surface_shader_table_evaluation.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceShaderTableView table_view(
    luisa::compute::UInt3 descriptor) noexcept {
    return {
        .offset = descriptor.x,
        .count = descriptor.y,
        .width = descriptor.z};
}

}// namespace

ColorRampTableCallable
make_color_ramp_sampled_linear_callable() noexcept {
    ColorRampTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float factor) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return color_ramp_sampled_linear_inline(
            table, factor);
    };
    callable.set_name("surface_color_ramp_sampled_linear");
    return callable;
}

ColorRampTableCallable
make_color_ramp_sampled_constant_callable() noexcept {
    ColorRampTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float factor) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return color_ramp_sampled_constant_inline(
            table, factor);
    };
    callable.set_name("surface_color_ramp_sampled_constant");
    return callable;
}

ColorRampTableCallable
make_color_ramp_control_linear_callable() noexcept {
    ColorRampTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float factor) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return color_ramp_control_linear_inline(
            table, factor);
    };
    callable.set_name("surface_color_ramp_control_linear");
    return callable;
}

ColorRampTableCallable
make_color_ramp_control_constant_callable() noexcept {
    ColorRampTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float factor) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return color_ramp_control_constant_inline(
            table, factor);
    };
    callable.set_name("surface_color_ramp_control_constant");
    return callable;
}

SampledRgbCurveTableCallable
make_rgb_curve_sampled_callable() noexcept {
    SampledRgbCurveTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float3 input,
           Float factor,
           Float min_x,
           Float max_x,
           Float extrapolate) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return rgb_curve_sampled_inline(
            table,
            input,
            factor,
            min_x,
            max_x,
            extrapolate);
    };
    callable.set_name("surface_rgb_curve_sampled");
    return callable;
}

ControlRgbCurveTableCallable
make_rgb_curve_control_callable() noexcept {
    ControlRgbCurveTableCallable callable =
        [](BufferFloat data,
           luisa::compute::UInt3 descriptor,
           Float3 input,
           Float factor) noexcept {
        BufferSurfaceShaderTableReader table{
            data, table_view(descriptor)};
        return rgb_curve_control_inline(
            table, input, factor);
    };
    callable.set_name("surface_rgb_curve_control");
    return callable;
}

}// namespace psycles::luisa_backend::detail
