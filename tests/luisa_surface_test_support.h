#pragma once

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

#include <luisa/luisa-compute.h>

namespace psycles::test_support {

using namespace luisa::compute;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;

class ParameterShaderServices : public ShaderServices {

private:
    const BufferFloat4 &_parameters;
    Float _cycles_value;

public:
    explicit ParameterShaderServices(
        const BufferFloat4 &parameters,
        Expr<float> cycles_value = 0.5f) noexcept
        : _parameters{parameters},
          _cycles_value{cycles_value} {
    }

    [[nodiscard]] Float4 texture_2d(Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(Expr<luisa::ulong>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).xyz();
    }

    [[nodiscard]] ULong
    parameter_uint64(Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot)
            .xy()
            .bitcast<luisa::ulong>();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        // Structural tests only need a finite, non-extinguishing Cycles LUT
        // value. Scene probes cover the production versioned table data.
        return _cycles_value;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

template<typename Device, typename Kernel>
[[nodiscard]] auto compile_named_kernel(
    Device &device, std::string_view name, const Kernel &kernel) {
    const auto trace_compilation =
        std::getenv("PSYCLES_TRACE_SHADER_COMPILATION") != nullptr;
    if (trace_compilation) {
        std::cerr << "compiling " << name << "..." << std::endl;
    }
    auto shader = device.compile(kernel);
    if (trace_compilation) {
        std::cerr << "compiled " << name << std::endl;
    }
    return shader;
}

[[nodiscard]] inline std::vector<luisa::float4> parameter_data(
    const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        const auto &value = parameter.default_value;
        if (parameter.type != value.type || !value.well_typed()) {
            throw std::runtime_error{
                "surface fixture has an ill-typed parameter"};
        }
        using contract::SocketType;
        switch (parameter.type) {
            case SocketType::boolean:
                result.emplace_back(
                    std::get<bool>(value.value) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f,
                    0.0f);
                break;
            case SocketType::integer:
                result.emplace_back(
                    static_cast<float>(
                        std::get<std::int64_t>(value.value)),
                    0.0f,
                    0.0f,
                    0.0f);
                break;
            case SocketType::unsigned_integer:
                {
                    const auto word =
                        std::get<std::uint64_t>(value.value);
                    result.emplace_back(
                        std::bit_cast<float>(
                            static_cast<std::uint32_t>(word)),
                        std::bit_cast<float>(
                            static_cast<std::uint32_t>(word >> 32u)),
                        0.0f,
                        0.0f);
                }
                break;
            case SocketType::floating:
                result.emplace_back(
                    std::get<float>(value.value),
                    0.0f,
                    0.0f,
                    0.0f);
                break;
            case SocketType::float2: {
                const auto vector = std::get<Vec2f>(value.value);
                result.emplace_back(
                    vector.x, vector.y, 0.0f, 0.0f);
                break;
            }
            case SocketType::float3:
            case SocketType::color:
            case SocketType::spectrum:
            case SocketType::point:
            case SocketType::vector:
            case SocketType::normal: {
                const auto vector = std::get<Vec3f>(value.value);
                result.emplace_back(
                    vector.x, vector.y, vector.z, 0.0f);
                break;
            }
            case SocketType::transform:
            case SocketType::string:
            case SocketType::closure:
            case SocketType::volume_closure:
                throw std::runtime_error{
                    "surface fixture has an unsupported parameter type"};
        }
    }
    if (result.empty()) {
        result.emplace_back(0.0f);
    }
    return result;
}

// Reconstruct the host-side abstract-interpretation input from the exact
// packed parameter blocks dispatched by a Luisa regression. This lets tests
// join the same per-material closure plans as the production scene builder,
// instead of accidentally compiling the topology-only conservative top.
[[nodiscard]] inline SurfaceParameterBlock surface_parameter_block(
    const SurfaceProgram &program,
    std::span<const luisa::float4> packed) {
    if (packed.size() != program.parameters().size()) {
        throw std::runtime_error{
            "surface fixture parameter block has the wrong extent"};
    }
    SurfaceParameterBlock result{program};
    for (std::size_t index = 0u; index < packed.size(); ++index) {
        const auto &parameter = program.parameters()[index];
        const auto value = packed[index];
        contract::SocketValue decoded = parameter.default_value;
        using contract::SocketType;
        switch (parameter.type) {
            case SocketType::boolean:
                decoded = contract::SocketValue::boolean(value.x != 0.0f);
                break;
            case SocketType::integer:
                decoded = contract::SocketValue::integer(
                    static_cast<std::int64_t>(value.x));
                break;
            case SocketType::unsigned_integer: {
                const auto low = std::bit_cast<std::uint32_t>(value.x);
                const auto high = std::bit_cast<std::uint32_t>(value.y);
                decoded = contract::SocketValue::unsigned_integer(
                    static_cast<std::uint64_t>(low) |
                    (static_cast<std::uint64_t>(high) << 32u));
                break;
            }
            case SocketType::floating:
                decoded = contract::SocketValue::floating(value.x);
                break;
            case SocketType::float2:
                decoded = contract::SocketValue::float2({value.x, value.y});
                break;
            case SocketType::float3:
                decoded = contract::SocketValue::float3(
                    {value.x, value.y, value.z});
                break;
            case SocketType::color:
                decoded = contract::SocketValue::color(
                    {value.x, value.y, value.z});
                break;
            case SocketType::spectrum:
                decoded = contract::SocketValue::spectrum(
                    {value.x, value.y, value.z});
                break;
            case SocketType::point:
                decoded = contract::SocketValue::point(
                    {value.x, value.y, value.z});
                break;
            case SocketType::vector:
                decoded = contract::SocketValue::vector(
                    {value.x, value.y, value.z});
                break;
            case SocketType::normal:
                decoded = contract::SocketValue::normal(
                    {value.x, value.y, value.z});
                break;
            case SocketType::transform:
            case SocketType::string:
            case SocketType::closure:
            case SocketType::volume_closure:
                throw std::runtime_error{
                    "surface fixture has an unsupported packed parameter type"};
        }
        if (!result.set(program, parameter.id, std::move(decoded))) {
            throw std::runtime_error{
                "surface fixture failed to reconstruct a parameter block"};
        }
    }
    return result;
}

[[nodiscard]] inline SurfaceClosurePlan merged_surface_closure_plan(
    const SurfaceProgram &program,
    std::span<const luisa::float4> packed) {
    const auto stride = program.parameters().size();
    if (stride == 0u) {
        return analyze_surface_closure_plan(
            program, SurfaceParameterBlock{program});
    }
    if (packed.empty() || packed.size() % stride != 0u) {
        throw std::runtime_error{
            "surface fixture parameter stream is not block-aligned"};
    }
    SurfaceClosurePlan result;
    for (std::size_t offset = 0u; offset < packed.size(); offset += stride) {
        const auto parameters = surface_parameter_block(
            program, packed.subspan(offset, stride));
        result.merge(analyze_surface_closure_plan(program, parameters));
    }
    return result;
}

[[nodiscard]] inline SurfacePoint make_surface_point() noexcept {
    return {.position = make_float3(0.0f),
        .object_position = make_float3(0.0f),
        .object_location = make_float3(0.0f),
        .generated = make_float3(0.5f),
        .geometric_normal = make_float3(0.0f, 0.0f, 1.0f),
        .shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .object_shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .object_tangent = make_float3(1.0f, 0.0f, 0.0f),
        .tangent_sign = 1.0f,
        .undisplaced_position = make_float3(0.0f),
        .undisplaced_object_position = make_float3(0.0f),
        .undisplaced_shading_normal =
            make_float3(0.0f, 0.0f, 1.0f),
        .undisplaced_object_shading_normal =
            make_float3(0.0f, 0.0f, 1.0f),
        .undisplaced_object_tangent =
            make_float3(1.0f, 0.0f, 0.0f),
        .undisplaced_tangent_sign = 1.0f,
        .normal_to_world_x = make_float3(1.0f, 0.0f, 0.0f),
        .normal_to_world_y = make_float3(0.0f, 1.0f, 0.0f),
        .normal_to_world_z = make_float3(0.0f, 0.0f, 1.0f),
        .dpdu = make_float3(1.0f, 0.0f, 0.0f),
        .dpdv = make_float3(0.0f, 1.0f, 0.0f),
        .dPdx = make_float3(0.0f),
        .dPdy = make_float3(0.0f),
        .object_dPdx = make_float3(0.0f),
        .object_dPdy = make_float3(0.0f),
        .undisplaced_dPdx = make_float3(0.0f),
        .undisplaced_dPdy = make_float3(0.0f),
        .undisplaced_object_dPdx = make_float3(0.0f),
        .undisplaced_object_dPdy = make_float3(0.0f),
        .generated_dx = make_float3(0.0f),
        .generated_dy = make_float3(0.0f),
        .incoming = make_float3(0.0f, 0.0f, 1.0f),
        .uv = make_float2(0.0f),
        .uv_dx = make_float2(0.0f),
        .uv_dy = make_float2(0.0f),
        .geometry_index = 0u,
        .barycentric = make_float2(0.0f),
        .barycentric_dx = make_float2(0.0f),
        .barycentric_dy = make_float2(0.0f),
        .instance_id = 0u,
        .primitive_id = 0u,
        .parameter_block = 0u,
        .object_random = 0.0f,
        .particle_index = 0u,
        .random_per_island = 0.0f,
        .triangle_smooth = true,
        .ray_visibility = 1u,
        .ray_events = 0u,
        .ray_depth = 0u,
        .diffuse_depth = 0u,
        .glossy_depth = 0u,
        .transparent_depth = 0u,
        .transmission_depth = 0u,
        .ray_length = 0.0f,
        .time = 0.0f,
        .use_bump_map_correction = true,
        .back_facing = false};
}

// AOV-focused unit tests exercise the reduction directly instead of recording
// unrelated emission and runtime-flag projections into already large closure
// kernels. Production path hits still have only the fused prepare boundary.
[[nodiscard]] inline SurfaceAov surface_aov(
    const SurfaceDispatch &surfaces,
    Expr<std::uint32_t> tag,
    const ShaderServices &services,
    const SurfacePoint &point) noexcept {
    const auto operation = make_surface_closure_aov_callable();
    SurfaceAovVisitor visitor{
        point,
        maximum_surface_closure_capacity,
        operation};
    static_cast<void>(surfaces.collect_closures(
        tag, services, point, true, true, visitor));
    return visitor.result();
}

[[nodiscard]] inline bool approximately_equal(
    float actual, float expected, float tolerance = 3.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(1.0f,
                   std::max(std::abs(actual), std::abs(expected)));
}

[[nodiscard]] inline bool approximately_equal(luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 3.0e-6f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance) &&
           approximately_equal(actual.w, expected.w, tolerance);
}

}// namespace psycles::test_support
