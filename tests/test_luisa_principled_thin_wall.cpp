#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/cycles_abi.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"
#include "luisa_shader_shape_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;
using psycles::test_support::surface_aov;

constexpr std::uint32_t case_count = 7u;
constexpr std::uint32_t record_count = 51u;
constexpr luisa::float3 base_color{0.25f, 0.64f, 1.0f};
constexpr luisa::float3 specular_tint{0.5f, 1.0f, 0.25f};

namespace trace_case {
constexpr std::uint32_t thick_glass = 0u;
constexpr std::uint32_t smooth_begin = 1u;
constexpr std::uint32_t smooth_end = 4u;
constexpr std::uint32_t noncamera_begin = 5u;
constexpr std::uint32_t noncamera_end = 6u;
constexpr std::uint32_t rough_thin_begin = 7u;
constexpr std::uint32_t rough_thin_end = 8u;
constexpr std::uint32_t smooth_subsurface_begin = 9u;
constexpr std::uint32_t smooth_subsurface_end = 10u;
constexpr std::uint32_t rough_subsurface_begin = 11u;
constexpr std::uint32_t rough_subsurface_end = 12u;
constexpr std::uint32_t thick_subsurface = 13u;
constexpr std::uint32_t count = 14u;
}// namespace trace_case

namespace sample_case {
constexpr std::uint32_t smooth_thin = 0u;
constexpr std::uint32_t rough_subsurface = 1u;
constexpr std::uint32_t tilted_corrected = 2u;
constexpr std::uint32_t tilted_uncorrected = 3u;
constexpr std::uint32_t count = 4u;
}// namespace sample_case

namespace evaluation_case {
constexpr std::uint32_t rough_transmission = 0u;
constexpr std::uint32_t rough_reflection = 1u;
constexpr std::uint32_t rough_subsurface = 2u;
constexpr std::uint32_t count = 3u;
}// namespace evaluation_case

namespace light_case {
constexpr std::uint32_t rough_exclude_glossy = 0u;
constexpr std::uint32_t rough_exclude_transmit = 1u;
constexpr std::uint32_t subsurface_exclude_transmit = 2u;
constexpr std::uint32_t subsurface_exclude_diffuse = 3u;
constexpr std::uint32_t count = 4u;
}// namespace light_case

[[nodiscard]] ShaderGraph make_thin_wall_graph() {
    ShaderGraph graph;
    const auto thin_wall_value = graph.add_node(
        node_type::constant_float,
        "Linked Cycles FLOAT to Boolean");
    const auto thin_wall_conversion = graph.add_node(
        node_type::scalar_to_boolean,
        "Cycles FLOAT to INT Boolean conversion");
    const auto normal_color = graph.add_node(
        node_type::constant_color,
        "Linked closure normal");
    const auto normal_vector = graph.add_node(
        node_type::color_to_vector,
        "Closure normal color to vector");
    const auto normal = graph.add_node(
        node_type::vector_to_normal,
        "Closure normal vector to normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Raw dynamic Thin Wall regression");
    const auto configured =
        graph.set_input(thin_wall_value,
            "Value",
            SocketValue::floating(0.0f)) &&
        graph.connect(
            OutputRef{.node = thin_wall_value, .socket = "Value"},
            thin_wall_conversion,
            "Value") &&
        graph.connect(
            OutputRef{
                .node = thin_wall_conversion,
                .socket = "Boolean"},
            principled,
            "ThinWall") &&
        graph.connect(
            OutputRef{.node = normal_color, .socket = "Color"},
            normal_vector,
            "Color") &&
        graph.connect(
            OutputRef{.node = normal_vector, .socket = "Vector"},
            normal,
            "Vector") &&
        graph.connect(
            OutputRef{.node = normal, .socket = "Normal"},
            principled,
            "Normal") &&
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color(
                {base_color.x, base_color.y, base_color.z})) &&
        graph.set_input(principled,
            "Metallic",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "Roughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "SubsurfaceRadius",
            SocketValue::vector({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled,
            "SubsurfaceScale",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SubsurfaceAnisotropy",
            SocketValue::floating(0.2f)) &&
        graph.set_input(principled,
            "TransmissionWeight",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "IOR",
            SocketValue::floating(1.5f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled,
            "SpecularTint",
            SocketValue::color(
                {specular_tint.x,
                    specular_tint.y,
                    specular_tint.z})) &&
        graph.set_input(principled,
            "Alpha",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SheenWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "CoatWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_property(principled,
            "Distribution",
            SocketValue::string("GGX")) &&
        graph.set_property(principled,
            "SubsurfaceMethod",
            SocketValue::string("RANDOM_WALK"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure dynamic Thin Wall graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

void set_parameter(
    std::vector<luisa::float4> &values,
    const SurfaceProgram &program,
    std::string_view socket,
    luisa::float4 value) {
    for (std::size_t index = 0u;
         index < program.parameters().size();
         ++index) {
        if (program.parameters()[index].socket == socket) {
            values[index] = value;
            return;
        }
    }
    throw std::runtime_error{
        "missing Thin Wall parameter: " + std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4> make_parameters(
    const SurfaceProgram &program) {
    const auto defaults = parameter_data(program);
    std::vector<luisa::float4> result;
    result.reserve(defaults.size() * case_count);
    const auto append = [&](float thin_wall,
                            float transmission,
                            float roughness,
                            float subsurface,
                            float diffuse_roughness,
                            float ior,
                            float alpha,
                            luisa::float3 normal) {
        auto values = defaults;
        set_parameter(values,
            program,
            "Value",
            {thin_wall, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "TransmissionWeight",
            {transmission, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Roughness",
            {roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "SubsurfaceWeight",
            {subsurface, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "DiffuseRoughness",
            {diffuse_roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "IOR",
            {ior, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Alpha",
            {alpha, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Color",
            {normal.x, normal.y, normal.z, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    };

    // Cycles converts FLOAT to INT by truncation before treating the target
    // Boolean socket as true/false. The non-integral values pin both sides of
    // that boundary and prevent replacement with Blender's generic > 0 cast.
    constexpr luisa::float3 geometric_normal{0.0f, 0.0f, 1.0f};
    append(0.75f, 1.0f, 0.0f, 0.0f, 0.0f, 1.5f, 1.0f,
        geometric_normal);
    append(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.5f, 0.8f,
        geometric_normal);
    append(1.25f, 1.0f, 0.35f, 0.0f, 0.0f, 1.5f, 1.0f,
        geometric_normal);
    append(-1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        geometric_normal);
    append(1.0f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        geometric_normal);
    append(-0.75f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        geometric_normal);
    append(1.0f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        {0.6f, 0.0f, 0.8f});
    return result;
}

[[nodiscard]] bool rgb_equal(
    luisa::float4 actual,
    luisa::float3 expected,
    float tolerance = 6.0e-5f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

[[nodiscard]] constexpr luisa::float3 scale(
    luisa::float3 value,
    float factor) noexcept {
    return {value.x * factor, value.y * factor, value.z * factor};
}

[[nodiscard]] constexpr luisa::float3 add(
    luisa::float3 lhs,
    luisa::float3 rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] constexpr float average(
    luisa::float3 value) noexcept {
    return (value.x + value.y + value.z) / 3.0f;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(make_thin_wall_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile Thin Wall graph\n";
        return EXIT_FAILURE;
    }
    const auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok()) {
        std::cerr << "failed to lower Thin Wall surface program\n";
        return EXIT_FAILURE;
    }
    const auto &program = *lowered.program;
    const auto &closure = program.closure_instructions().front();
    bool raw_thin_wall_source = false;
    for (const auto &parameter : program.parameters()) {
        raw_thin_wall_source |= parameter.socket == "Value";
    }
    bool exact_float_to_boolean = false;
    for (const auto &instruction : program.value_instructions()) {
        exact_float_to_boolean |=
            instruction.operation ==
            ValueOperation::scalar_to_boolean;
    }
    if (!raw_thin_wall_source || !exact_float_to_boolean ||
        !closure.thin_wall.valid()) {
        std::cerr << "raw Thin Wall expression was lost before Luisa setup\n";
        return EXIT_FAILURE;
    }

    const auto parameters = make_parameters(program);
    const auto parameter_stride = static_cast<std::uint32_t>(
        program.parameters().size());
    SurfaceDispatch surfaces;
    const auto surface_tag = surfaces.create<GraphSurface>(
        lowered.program,
        merged_surface_closure_plan(program, parameters));

    Kernel1D trace_closures = [&](BufferFloat4 parameter_buffer,
                                  BufferFloat4 output) noexcept {
        // A unit table value isolates the closure algebra from GGX energy
        // table interpolation. Production scene regressions use the exact
        // versioned Cycles tables.
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        const auto smooth =
            (case_index >= trace_case::smooth_begin) &
            (case_index <= trace_case::smooth_end);
        const auto noncamera =
            (case_index >= trace_case::noncamera_begin) &
            (case_index <= trace_case::noncamera_end);
        const auto rough_thin =
            (case_index >= trace_case::rough_thin_begin) &
            (case_index <= trace_case::rough_thin_end);
        const auto smooth_subsurface =
            (case_index >= trace_case::smooth_subsurface_begin) &
            (case_index <= trace_case::smooth_subsurface_end);
        const auto rough_subsurface =
            (case_index >= trace_case::rough_subsurface_begin) &
            (case_index <= trace_case::rough_subsurface_end);
        auto point = make_surface_point();
        UInt parameter_case = 0u;
        parameter_case = select(
            parameter_case, 1u, smooth | noncamera);
        parameter_case = select(parameter_case, 2u, rough_thin);
        parameter_case =
            select(parameter_case, 3u, smooth_subsurface);
        parameter_case =
            select(parameter_case, 4u, rough_subsurface);
        parameter_case = select(parameter_case,
            5u,
            case_index == trace_case::thick_subsurface);
        point.parameter_block = parameter_case * parameter_stride;
        point.ray_visibility = select(
            point.ray_visibility, 1u << 1u, noncamera);
        UInt closure_index = 0u;
        closure_index = select(closure_index,
            case_index - trace_case::smooth_begin,
            smooth);
        closure_index = select(closure_index,
            case_index - trace_case::noncamera_begin,
            noncamera);
        closure_index = select(closure_index,
            case_index - trace_case::rough_thin_begin,
            rough_thin);
        closure_index = select(closure_index,
            case_index - trace_case::smooth_subsurface_begin,
            smooth_subsurface);
        closure_index = select(closure_index,
            case_index - trace_case::rough_subsurface_begin,
            rough_subsurface);
        const auto value = surfaces.closure_trace(UInt{surface_tag},
            services,
            point,
            closure_index,
            true,
            true);
        UInt record = 0u;
        record = select(record, case_index * 2u, smooth | noncamera);
        record = select(record,
            21u + (case_index - trace_case::rough_thin_begin) * 2u,
            rough_thin);
        record = select(record,
            29u +
                (case_index - trace_case::smooth_subsurface_begin) * 2u,
            smooth_subsurface);
        record = select(record,
            33u +
                (case_index - trace_case::rough_subsurface_begin) * 2u,
            rough_subsurface);
        record = select(record,
            43u,
            case_index == trace_case::thick_subsurface);
        output.write(record,
            make_float4(cast<float>(value.count),
                cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
        output.write(record + 1u, make_float4(value.weight, 0.0f));
    };

    Kernel1D evaluate_extinction = [&](BufferFloat4 parameter_buffer,
                                       BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        auto point = make_surface_point();
        point.parameter_block = parameter_stride;
        point.ray_visibility = 1u << 1u;
        const auto extinction = surfaces.transparent_extinction(
            UInt{surface_tag}, services, point);
        output.write(14u, make_float4(extinction, 0.0f));
    };

    Kernel1D sample_closures = [&](BufferFloat4 parameter_buffer,
                                   BufferFloat3 tilted_directions,
                                   BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        UInt parameter_case = 1u;
        parameter_case = select(parameter_case,
            4u,
            case_index == sample_case::rough_subsurface);
        parameter_case = select(parameter_case,
            6u,
            case_index >= sample_case::tilted_corrected);
        point.parameter_block = parameter_case * parameter_stride;
        point.use_bump_map_correction =
            case_index != sample_case::tilted_uncorrected;
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(event_diffuse |
                                                    event_glossy |
                                                    event_transmission |
                                                    event_transparent),
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = surfaces.sample_trace(UInt{surface_tag},
            services,
            point,
            select(0.9f,
                0.99f,
                case_index == sample_case::smooth_thin),
            select(make_float2(0.23f, 0.79f),
                make_float2(0.37f, 0.73f),
                case_index == sample_case::smooth_thin),
            query);
        UInt record = 15u;
        record = select(record,
            40u,
            case_index == sample_case::rough_subsurface);
        record = select(record,
            47u + (case_index - sample_case::tilted_corrected),
            case_index >= sample_case::tilted_corrected);
        output.write(record,
            make_float4(
                value.sample.evaluation.f, value.sample.evaluation.pdf));
        $if(case_index < sample_case::tilted_corrected) {
            output.write(record + 1u,
                make_float4(value.sample.wi,
                    select(0.0f, 1.0f, value.sample.valid)));
            output.write(record + 2u,
                make_float4(value.sample.roughness.x,
                    value.sample.eta,
                    cast<float>(value.sample.evaluation.events),
                    cast<float>(value.closure_type)));
        };
        $if(case_index >= sample_case::tilted_corrected) {
            tilted_directions.write(
                case_index - sample_case::tilted_corrected,
                value.sample.wi);
        };
    };

    Kernel1D evaluate_aov = [&](BufferFloat4 parameter_buffer,
                                BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block =
            select(1u, 4u, case_index != 0u) * parameter_stride;
        const auto aov =
            surface_aov(surfaces, UInt{surface_tag}, services, point);
        $if(case_index == 0u) {
            output.write(18u,
                make_float4(aov.glossy_albedo, aov.roughness.x));
            output.write(19u,
                make_float4(
                    aov.transmission_albedo, aov.roughness.y));
            output.write(20u, make_float4(aov.transparency, 0.0f));
        }
        $else {
            output.write(
                45u, make_float4(aov.albedo, aov.roughness.x));
            output.write(46u,
                make_float4(aov.normal, aov.transmission_albedo.x));
        };
    };

    Kernel1D evaluate_surface = [&](BufferFloat4 parameter_buffer,
                                    BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block =
            select(2u,
                4u,
                case_index == evaluation_case::rough_subsurface) *
            parameter_stride;
        const auto outgoing = select(make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            case_index == evaluation_case::rough_reflection);
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(event_diffuse |
                                                    event_glossy |
                                                    event_transmission |
                                                    event_transparent),
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = surfaces.evaluate(
            UInt{surface_tag}, services, point, outgoing, query);
        UInt record = 25u;
        record = select(record,
            28u,
            case_index == evaluation_case::rough_reflection);
        record = select(record,
            37u,
            case_index == evaluation_case::rough_subsurface);
        output.write(record, make_float4(value.f, value.pdf));
    };

    Kernel1D evaluate_light = [&](BufferFloat4 parameter_buffer,
                                  BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block =
            select(2u,
                4u,
                case_index >= light_case::subsurface_exclude_transmit) *
            parameter_stride;
        UInt shader_flags = cycles_abi::shader_exclude_transmit;
        shader_flags = select(shader_flags,
            cycles_abi::shader_exclude_glossy,
            case_index == light_case::rough_exclude_glossy);
        shader_flags = select(shader_flags,
            cycles_abi::shader_exclude_diffuse,
            case_index == light_case::subsurface_exclude_diffuse);
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(event_diffuse |
                                                    event_glossy |
                                                    event_transmission |
                                                    event_transparent),
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = surfaces.evaluate_light(UInt{surface_tag},
            services,
            point,
            make_float3(0.0f, 0.0f, -1.0f),
            SurfaceLightQuery{
                .surface = query,
                .shader_flags =
                    shader_flags | cycles_abi::shader_use_mis});
        const auto record = select(26u + case_index,
            36u + case_index,
            case_index >= light_case::subsurface_exclude_transmit);
        output.write(record, make_float4(value.f, value.pdf));
    };

    Kernel1D evaluate_tilted = [&](BufferFloat4 parameter_buffer,
                                   BufferFloat3 tilted_directions,
                                   BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = 6u * parameter_stride;
        point.use_bump_map_correction = case_index == 0u;
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(event_diffuse |
                                                    event_glossy |
                                                    event_transmission |
                                                    event_transparent),
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = surfaces.evaluate(UInt{surface_tag},
            services,
            point,
            tilted_directions.read(0u),
            query);
        output.write(49u + case_index, make_float4(value.f, value.pdf));
    };

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "thin_wall_trace_closures", trace_closures, 35000u);
        bounded &= require_bounded_xir(
            "thin_wall_extinction", evaluate_extinction, 22000u);
        bounded &= require_bounded_xir(
            "thin_wall_sample_closures", sample_closures, 390000u);
        bounded &= require_bounded_xir(
            "thin_wall_aov", evaluate_aov, 27000u);
        bounded &= require_bounded_xir(
            "thin_wall_evaluate", evaluate_surface, 205000u);
        bounded &= require_bounded_xir(
            "thin_wall_evaluate_light", evaluate_light, 205000u);
        bounded &= require_bounded_xir(
            "thin_wall_evaluate_tilted", evaluate_tilted, 205000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    auto tilted_directions = device.create_buffer<luisa::float3>(2u);
    auto output_buffer =
        device.create_buffer<luisa::float4>(record_count);
    auto trace_kernel = compile_named_kernel(
        device, "thin_wall_trace_closures", trace_closures);
    auto extinction_kernel = compile_named_kernel(
        device, "thin_wall_extinction", evaluate_extinction);
    auto sample_kernel = compile_named_kernel(
        device, "thin_wall_sample_closures", sample_closures);
    auto aov_kernel =
        compile_named_kernel(device, "thin_wall_aov", evaluate_aov);
    auto evaluation_kernel = compile_named_kernel(
        device, "thin_wall_evaluate", evaluate_surface);
    auto light_kernel = compile_named_kernel(
        device, "thin_wall_evaluate_light", evaluate_light);
    auto tilted_kernel = compile_named_kernel(
        device, "thin_wall_evaluate_tilted", evaluate_tilted);
    std::array<luisa::float4, record_count> actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << trace_kernel(parameter_buffer, output_buffer)
                  .dispatch(trace_case::count)
           << extinction_kernel(parameter_buffer, output_buffer)
                  .dispatch(1u)
           << sample_kernel(
                  parameter_buffer, tilted_directions, output_buffer)
                  .dispatch(sample_case::count)
           << aov_kernel(parameter_buffer, output_buffer).dispatch(2u)
           << evaluation_kernel(parameter_buffer, output_buffer)
                  .dispatch(evaluation_case::count)
           << light_kernel(parameter_buffer, output_buffer)
                  .dispatch(light_case::count)
           << tilted_kernel(
                  parameter_buffer, tilted_directions, output_buffer)
                  .dispatch(2u)
           << output_buffer.copy_to(luisa::span{actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_THIN_WALL_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < actual.size(); ++index) {
            const auto value = actual[index];
            std::cerr << index << ": {" << value.x << ", "
                      << value.y << ", " << value.z << ", "
                      << value.w << "}\n";
        }
    }

    const auto meta = [&](std::uint32_t record,
                          float count,
                          std::uint32_t type,
                          bool valid) noexcept {
        return approximately_equal(actual[record].x, count) &&
               approximately_equal(
                   actual[record].y, static_cast<float>(type)) &&
               approximately_equal(
                   actual[record].w, valid ? 1.0f : 0.0f);
    };
    constexpr luisa::float3 reflectance{
        0.021200530013f,
        0.055109396494f,
        0.019801980198f};
    constexpr luisa::float3 transmittance{
        0.240106002650f,
        0.590210800550f,
        0.980198019802f};
    constexpr auto alpha = 0.8f;
    const auto thin_reflection = scale(reflectance, alpha);
    const auto thin_transmission = scale(transmittance, alpha);
    constexpr luisa::float3 alpha_transparency{0.2f, 0.2f, 0.2f};
    const auto merged_transparency = add(
        alpha_transparency, thin_transmission);

    if (!meta(0u,
            1.0f,
            cycles_closure::type_microfacet_ggx_glass,
            true) ||
        !meta(2u, 3.0f, cycles_closure::type_transparent, true) ||
        !rgb_equal(actual[3u], alpha_transparency) ||
        !meta(4u, 3.0f, cycles_closure::type_microfacet_ggx, true) ||
        !rgb_equal(actual[5u], thin_reflection) ||
        !meta(6u,
            3.0f,
            cycles_closure::type_thin_glass_transmission,
            true) ||
        !rgb_equal(actual[7u], thin_transmission) ||
        !meta(8u, 3.0f, cycles_closure::type_none, false)) {
        std::cerr << "dynamic Thin Wall camera topology regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(10u, 2.0f, cycles_closure::type_transparent, true) ||
        !rgb_equal(actual[11u], merged_transparency) ||
        !meta(12u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !rgb_equal(actual[13u], thin_reflection) ||
        !rgb_equal(actual[14u], merged_transparency)) {
        std::cerr << "smooth non-camera transparent merge regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto total_selection_weight =
        average(alpha_transparency) +
        average(thin_reflection) +
        average(thin_transmission);
    const auto transmission_probability =
        average(thin_transmission) / total_selection_weight;
    constexpr auto singular_transmission = static_cast<float>(
        event_singular | event_transmission);
    if (!rgb_equal(actual[15u], scale(thin_transmission, 1.0e6f)) ||
        !approximately_equal(
            actual[15u].w,
            transmission_probability * 1.0e6f,
            8.0e-5f) ||
        !approximately_equal(
            actual[16u], {0.0f, 0.0f, -1.0f, 1.0f}) ||
        !approximately_equal(actual[17u].x, 0.0f) ||
        !approximately_equal(actual[17u].y, 1.0f) ||
        !approximately_equal(actual[17u].z, singular_transmission) ||
        !approximately_equal(
            actual[17u].w,
            static_cast<float>(
                cycles_closure::type_thin_glass_transmission))) {
        std::cerr << "thin-glass singular sampling regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[18u], thin_reflection) ||
        !rgb_equal(actual[19u], thin_transmission) ||
        !rgb_equal(actual[20u], alpha_transparency)) {
        std::cerr << "thin-glass split AOV regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(21u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !meta(23u,
            2.0f,
            cycles_closure::type_thin_glass_transmission,
            true) ||
        !finite(actual[25u]) ||
        !finite(actual[28u]) ||
        !(actual[25u].w > 0.0f) ||
        !(actual[28u].w > 0.0f) ||
        !approximately_equal(actual[26u], actual[25u], 8.0e-5f) ||
        !rgb_equal(actual[27u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(
            actual[27u].w, actual[25u].w, 8.0e-5f)) {
        std::cerr << "rough thin-glass eval/type classifier regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto subsurface_reflection = scale(base_color, 0.4f);
    const auto subsurface_transmission = scale(base_color, 0.6f);
    if (!meta(29u, 2.0f, cycles_closure::type_diffuse, true) ||
        !rgb_equal(actual[30u], subsurface_reflection) ||
        !meta(31u, 2.0f, cycles_closure::type_translucent, true) ||
        !rgb_equal(actual[32u], subsurface_transmission) ||
        !meta(33u, 2.0f, cycles_closure::type_oren_nayar, true) ||
        !rgb_equal(actual[34u], subsurface_reflection) ||
        !meta(35u,
            2.0f,
            cycles_closure::type_rough_translucent,
            true) ||
        !rgb_equal(actual[36u], subsurface_transmission)) {
        std::cerr << "thin-subsurface topology/anisotropy regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    // Rough Translucent is in Cycles' diffuse closure interval even though
    // its sample label transmits. EXCLUDE_TRANSMIT retains it;
    // EXCLUDE_DIFFUSE removes its value but preserves the mixture PDF.
    if (!finite(actual[37u]) || !(actual[37u].w > 0.0f) ||
        !approximately_equal(actual[38u], actual[37u], 8.0e-5f) ||
        !rgb_equal(actual[39u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(
            actual[39u].w, actual[37u].w, 8.0e-5f) ||
        !finite(actual[40u]) || !(actual[40u].w > 0.0f) ||
        !(actual[41u].z < 0.0f) ||
        !approximately_equal(actual[41u].w, 1.0f) ||
        !approximately_equal(
            actual[42u].z,
            static_cast<float>(event_diffuse | event_transmission)) ||
        !approximately_equal(
            actual[42u].w,
            static_cast<float>(
                cycles_closure::type_rough_translucent))) {
        std::cerr << "rough-translucent eval/sample classifier regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    // Records 43/44 sample the thick BSSRDF, but 45/46 query the thin-wall
    // rough-diffuse pair (parameter block 4). Cycles includes Oren-Nayar and
    // Rough Translucent in its roughness pass; their value here is 0.4, not
    // the 1.0 fallback for an actual BSSRDF-only retained prefix. See the
    // independent test_luisa_surface_roughness.cpp external HIP oracle.
    if (!meta(43u,
            1.0f,
            cycles_closure::type_bssrdf_random_walk,
            true) ||
        !rgb_equal(actual[45u], base_color) ||
        !approximately_equal(actual[45u].w, 0.4f) ||
        !approximately_equal(
            actual[46u], {0.0f, 0.0f, -1.0f, 0.0f})) {
        std::cerr << "thick/thin subsurface dispatch and AOV regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto corrected_sample = actual[47u];
    const auto uncorrected_sample = actual[48u];
    const auto corrected_evaluation = actual[49u];
    const auto uncorrected_evaluation = actual[50u];
    if (!approximately_equal(
            corrected_sample, uncorrected_sample, 8.0e-5f) ||
        !approximately_equal(
            corrected_sample, uncorrected_evaluation, 8.0e-5f) ||
        !(corrected_evaluation.x <
            uncorrected_evaluation.x - 1.0e-5f) ||
        !approximately_equal(corrected_evaluation.w,
            uncorrected_evaluation.w,
            8.0e-5f)) {
        std::cerr << "Cycles transmissive sample bump-shadowing regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
