#pragma once

#include <cstdint>
#include <span>

#include <psycles/luisa/surface.h>

#include <luisa/dsl/resource.h>

namespace psycles::luisa_backend::detail {

// Runtime-polymorphic C++ readers are resolved while Luisa records the AST.
// They abstract the storage source without creating device virtual calls or a
// weak table protocol in the generated shader.
class SurfaceShaderTableReader {

public:
    virtual ~SurfaceShaderTableReader() noexcept = default;

    [[nodiscard]] virtual UInt count() const noexcept = 0;

    [[nodiscard]] virtual Float read(
        Expr<std::uint32_t> element,
        std::uint32_t component) const noexcept = 0;
};

class ServiceSurfaceShaderTableReader final
    : public SurfaceShaderTableReader {

private:
    const ShaderServices &_services;
    SurfaceShaderTableView _table;

public:
    ServiceSurfaceShaderTableReader(
        const ShaderServices &services,
        SurfaceShaderTableView table) noexcept;

    [[nodiscard]] UInt count() const noexcept override;

    [[nodiscard]] Float read(
        Expr<std::uint32_t> element,
        std::uint32_t component) const noexcept override;
};

class BufferSurfaceShaderTableReader final
    : public SurfaceShaderTableReader {

private:
    const luisa::compute::BufferFloat &_data;
    SurfaceShaderTableView _table;

public:
    BufferSurfaceShaderTableReader(
        const luisa::compute::BufferFloat &data,
        SurfaceShaderTableView table) noexcept;

    [[nodiscard]] UInt count() const noexcept override;

    [[nodiscard]] Float read(
        Expr<std::uint32_t> element,
        std::uint32_t component) const noexcept override;
};

// Decode one variable-length shader-table descriptor from the same parameter
// ABI used by Cycles-style SVM records. The table payload remains runtime data;
// neither its offset nor cardinality enters the recorded shader identity.
[[nodiscard]] SurfaceShaderTableView
surface_shader_table_view(const ShaderServices &services,
                          const SurfacePoint &point,
                          Expr<std::uint32_t> parameter) noexcept;

[[nodiscard]] Float4 color_ramp_sampled_linear_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_sampled_constant_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_control_linear_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_control_constant_inline(
    const SurfaceShaderTableReader &table,
    Float factor) noexcept;
[[nodiscard]] Float3 rgb_curve_sampled_inline(
    const SurfaceShaderTableReader &table,
    Float3 input,
    Float factor,
    Float min_x,
    Float max_x,
    Float extrapolate) noexcept;
[[nodiscard]] Float3 rgb_curve_control_inline(
    const SurfaceShaderTableReader &table,
    Float3 input,
    Float factor) noexcept;

[[nodiscard]] Float4 color_ramp_sampled_linear(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_sampled_constant(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_control_linear(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept;
[[nodiscard]] Float4 color_ramp_control_constant(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float factor) noexcept;
[[nodiscard]] Float4
evaluate_surface_color_ramp(const ShaderServices &services,
                            const SurfaceShaderTableView &table, Float factor,
                            std::uint32_t mode) noexcept;

// `immediate_domain` is the exact finite image of the family subtype. It
// determines which host-recorded branches exist, while `immediate` selects
// one of those branches at device runtime.
[[nodiscard]] Float4
evaluate_surface_color_ramp_svm(const ShaderServices &services, UInt immediate,
                                std::span<const std::uint16_t> immediate_domain,
                                const SurfaceShaderTableView &table,
                                Float factor) noexcept;
[[nodiscard]] Float3 rgb_curve_sampled(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float3 input,
    Float factor,
    Float min_x,
    Float max_x,
    Float extrapolate) noexcept;
[[nodiscard]] Float3 rgb_curve_control(
    const ShaderServices &services,
    const SurfaceShaderTableView &table,
    Float3 input,
    Float factor) noexcept;

[[nodiscard]] Float3 evaluate_surface_rgb_curve_svm(
    const ShaderServices &services, UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    const SurfaceShaderTableView &table, Float3 input, Float factor,
    Float min_x, Float max_x, Float extrapolate) noexcept;

}// namespace psycles::luisa_backend::detail
