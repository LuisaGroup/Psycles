#pragma once

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <variant>
#include <vector>

#include <luisa/luisa-compute.h>

namespace psycles::test_support {

using namespace luisa::compute;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;

class ParameterShaderServices final : public ShaderServices {

private:
    const BufferFloat4 &_parameters;
    float _cycles_value;

public:
    explicit ParameterShaderServices(
        const BufferFloat4 &parameters,
        float cycles_value = 0.5f) noexcept
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

    [[nodiscard]] ShaderAttribute attribute(Expr<std::uint64_t>,
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
                result.emplace_back(
                    static_cast<float>(
                        std::get<std::uint64_t>(value.value)),
                    0.0f,
                    0.0f,
                    0.0f);
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
