#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::adapter;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::surface_aov;

constexpr std::uint32_t case_count = 3u;
constexpr std::uint32_t records_per_case = 10u;
constexpr luisa::float3 authored_color{0.35f, 0.24f, 0.8f};
constexpr luisa::float3 authored_radius{2.7f, 1.58f, 1.0f};

[[nodiscard]] ShaderGraph make_barbershop_burley_graph() {
    CyclesNormalizedShaderGraph source;
    source.nodes = {{
        .id = 10u,
        .type = "subsurface_scattering",
        .variant = {},
        .label = "Barbershop agent_skin Burley",
        .inputs = {
            {.socket = "Color",
             .source = std::nullopt,
             .value = SocketValue::color(
                 {authored_color.x, authored_color.y, authored_color.z})},
            {.socket = "Scale",
             .source = std::nullopt,
             .value = SocketValue::floating(0.02f)},
            {.socket = "Radius",
             .source = std::nullopt,
             .value = SocketValue::vector(
                 {authored_radius.x,
                  authored_radius.y,
                  authored_radius.z})},
            {.socket = "IOR",
             .source = std::nullopt,
             .value = SocketValue::floating(1.4f)},
            {.socket = "Roughness",
             .source = std::nullopt,
             .value = SocketValue::floating(1.0f)},
            {.socket = "Anisotropy",
             .source = std::nullopt,
             .value = SocketValue::floating(0.0f)},
            {.socket = "Normal",
             .source = std::nullopt,
             .value = SocketValue::normal({0.0f, 0.0f, 0.0f})}},
        .properties = {{"method", SocketValue::string("BURLEY")}}}};
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 10u, .socket = "BSSRDF"});
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    if (!adapted.ok()) {
        throw std::runtime_error{
            "failed to adapt raw Cycles Burley graph"};
    }
    return std::move(*adapted.graph);
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
        "missing Burley parameter: " + std::string{socket}};
}

[[nodiscard]] bool rgb_equal(
    luisa::float4 actual,
    luisa::float3 expected,
    float tolerance = 4.0e-5f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

void dump_case(
    std::string_view backend,
    std::uint32_t case_index,
    const std::array<
        luisa::float4,
        case_count * records_per_case> &actual) {
    std::cerr << "Burley regression failed on " << backend
              << " in case " << case_index << '\n';
    const auto base = case_index * records_per_case;
    for (std::uint32_t record = 0u;
         record < records_per_case;
         ++record) {
        const auto value = actual[base + record];
        std::cerr << "  " << record << ": {" << value.x << ", "
                  << value.y << ", " << value.z << ", "
                  << value.w << "}\n";
    }
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(
        make_barbershop_burley_graph());
    if (!shader.ok()) {
        std::cerr << "raw Cycles Burley graph did not compile\n";
        return EXIT_FAILURE;
    }
    const auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok() ||
        lowered.program->closure_instructions().size() != 1u ||
        lowered.program->closure_instructions().front().operation !=
            ClosureOperation::subsurface ||
        lowered.program->closure_instructions()
                .front()
                .subsurface_method != BssrdfMethod::burley) {
        std::cerr
            << "raw Cycles BURLEY property did not remain a typed BSSRDF\n";
        return EXIT_FAILURE;
    }

    auto defaults = parameter_data(*lowered.program);
    std::vector<luisa::float4> parameters;
    parameters.reserve(defaults.size() * case_count);
    parameters.insert(parameters.end(), defaults.begin(), defaults.end());
    parameters.insert(parameters.end(), defaults.begin(), defaults.end());
    auto partial_parameters = defaults;
    set_parameter(
        partial_parameters,
        *lowered.program,
        "Radius",
        {authored_radius.x, 0.0f, authored_radius.z, 0.0f});
    parameters.insert(
        parameters.end(),
        partial_parameters.begin(),
        partial_parameters.end());

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(lowered.program);
    const auto parameter_count = static_cast<std::uint32_t>(
        lowered.program->parameters().size());
    Kernel1D probe = [&](BufferFloat4 parameter_buffer,
                         BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = case_index * parameter_count;
        point.diffuse_depth = select(
            0u, 1u, case_index == 1u);
        const auto first = surfaces.closure_trace(
            UInt{surface_tag}, services, point, 0u);
        const auto second = surfaces.closure_trace(
            UInt{surface_tag}, services, point, 1u);
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f};
        const auto sample = surfaces.sample_trace(
            UInt{surface_tag},
            services,
            point,
            0.0f,
            make_float2(0.37f, 0.61f),
            query);
        const auto aov = surface_aov(surfaces,
            UInt{surface_tag}, services, point);
        const auto base = case_index * records_per_case;
        output.write(base,
            make_float4(
                cast<float>(first.count),
                cast<float>(first.type),
                first.sample_weight,
                select(0.0f, 1.0f, first.valid)));
        output.write(base + 1u, make_float4(first.weight, 0.0f));
        output.write(base + 2u,
            make_float4(
                cast<float>(second.count),
                cast<float>(second.type),
                second.sample_weight,
                select(0.0f, 1.0f, second.valid)));
        output.write(base + 3u, make_float4(second.weight, 0.0f));
        output.write(base + 4u,
            make_float4(
                cast<float>(sample.closure_index),
                cast<float>(sample.closure_type),
                cast<float>(sample.sample.bssrdf_method),
                select(0.0f, 1.0f, sample.sample.valid)));
        output.write(base + 5u,
            make_float4(
                sample.sample.bssrdf_radius,
                sample.sample.bssrdf_ior));
        output.write(base + 6u,
            make_float4(
                sample.sample.bssrdf_albedo,
                sample.sample.bssrdf_roughness));
        output.write(base + 7u,
            make_float4(
                sample.sample.evaluation.f,
                sample.sample.evaluation.pdf));
        output.write(base + 8u,
            make_float4(
                aov.albedo,
                cast<float>(sample.sample.evaluation.events)));
        output.write(base + 9u,
            make_float4(
                sample.sample.bssrdf_anisotropy,
                select(0.0f, 1.0f, sample.closure_valid),
                select(0.0f, 1.0f, sample.sample.valid),
                cast<float>(sample.sample.runtime_flags)));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    auto output = device.create_buffer<luisa::float4>(
        case_count * records_per_case);
    auto kernel = device.compile(probe);
    std::array<
        luisa::float4,
        case_count * records_per_case> actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << kernel(parameter_buffer, output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    const auto radius_scale =
        0.02f * 0.25f / std::numbers::pi_v<float>;
    const auto expected_radius = luisa::float3{
        authored_radius.x * radius_scale,
        authored_radius.y * radius_scale,
        authored_radius.z * radius_scale};
    const auto case_ok = [&](std::uint32_t case_index) noexcept {
        const auto base = case_index * records_per_case;
        for (std::uint32_t record = 0u;
             record < records_per_case;
             ++record) {
            if (!finite(actual[base + record])) {
                return false;
            }
        }
        return rgb_equal(actual[base + 8u], authored_color);
    };

    const auto &full_first = actual[0u];
    const auto &full_second = actual[2u];
    const auto &full_sample = actual[4u];
    const auto &full_radius = actual[5u];
    const auto &full_albedo = actual[6u];
    const auto &full_transport = actual[7u];
    const auto full_ok = case_ok(0u) &&
                         approximately_equal(full_first.x, 1.0f) &&
                         approximately_equal(
                             full_first.y,
                             static_cast<float>(
                                 cycles_closure::type_bssrdf_burley)) &&
                         approximately_equal(full_first.z, 1.39f) &&
                         approximately_equal(full_first.w, 1.0f) &&
                         rgb_equal(actual[1u], authored_color) &&
                         approximately_equal(full_second.w, 0.0f) &&
                         approximately_equal(full_sample.x, 0.0f) &&
                         approximately_equal(
                             full_sample.y,
                             static_cast<float>(
                                 cycles_closure::type_bssrdf_burley)) &&
                         approximately_equal(
                             full_sample.z,
                             static_cast<float>(
                                 SurfaceBssrdfMethod::burley)) &&
                         approximately_equal(full_sample.w, 1.0f) &&
                         rgb_equal(full_radius, expected_radius) &&
                         approximately_equal(full_radius.w, 1.4f) &&
                         rgb_equal(full_albedo, authored_color) &&
                         approximately_equal(full_albedo.w, 1.0f) &&
                         rgb_equal(full_transport, authored_color) &&
                         approximately_equal(full_transport.w, 1.0f) &&
                         approximately_equal(
                             actual[8u].w,
                             static_cast<float>(event_subsurface));
    if (!full_ok) {
        dump_case(backend, 0u, actual);
        return EXIT_FAILURE;
    }

    const auto diffuse_base = records_per_case;
    const auto &occupied_none = actual[diffuse_base];
    const auto &diffuse = actual[diffuse_base + 2u];
    const auto &diffuse_sample = actual[diffuse_base + 4u];
    const auto diffuse_ok = case_ok(1u) &&
                            approximately_equal(occupied_none.x, 2.0f) &&
                            approximately_equal(
                                occupied_none.y,
                                static_cast<float>(
                                    cycles_closure::type_none)) &&
                            approximately_equal(occupied_none.z, 0.0f) &&
                            approximately_equal(occupied_none.w, 1.0f) &&
                            rgb_equal(
                                actual[diffuse_base + 1u],
                                luisa::float3{}) &&
                            approximately_equal(diffuse.x, 2.0f) &&
                            approximately_equal(
                                diffuse.y,
                                static_cast<float>(
                                    cycles_closure::type_diffuse)) &&
                            approximately_equal(
                                diffuse.z,
                                (authored_color.x + authored_color.y +
                                    authored_color.z) /
                                    3.0f) &&
                            approximately_equal(diffuse.w, 1.0f) &&
                            rgb_equal(
                                actual[diffuse_base + 3u],
                                authored_color) &&
                            approximately_equal(diffuse_sample.x, 1.0f) &&
                            approximately_equal(
                                diffuse_sample.y,
                                static_cast<float>(
                                    cycles_closure::type_diffuse)) &&
                            approximately_equal(diffuse_sample.w, 1.0f);
    if (!diffuse_ok) {
        dump_case(backend, 1u, actual);
        return EXIT_FAILURE;
    }

    const auto partial_base = 2u * records_per_case;
    const auto &partial_bssrdf = actual[partial_base];
    const auto &partial_diffuse = actual[partial_base + 2u];
    const auto &partial_sample = actual[partial_base + 4u];
    const auto &partial_radius = actual[partial_base + 5u];
    const auto &partial_albedo = actual[partial_base + 6u];
    const auto &partial_transport = actual[partial_base + 7u];
    const auto partial_ok = case_ok(2u) &&
                            approximately_equal(partial_bssrdf.x, 2.0f) &&
                            approximately_equal(
                                partial_bssrdf.y,
                                static_cast<float>(
                                    cycles_closure::type_bssrdf_burley)) &&
                            approximately_equal(
                                partial_bssrdf.z,
                                2.0f *
                                    (authored_color.x + authored_color.z) /
                                    3.0f) &&
                            rgb_equal(
                                actual[partial_base + 1u],
                                {authored_color.x, 0.0f, authored_color.z}) &&
                            approximately_equal(
                                partial_diffuse.y,
                                static_cast<float>(
                                    cycles_closure::type_diffuse)) &&
                            rgb_equal(
                                actual[partial_base + 3u],
                                {0.0f, authored_color.y, 0.0f}) &&
                            approximately_equal(partial_sample.x, 0.0f) &&
                            approximately_equal(
                                partial_sample.y,
                                static_cast<float>(
                                    cycles_closure::type_bssrdf_burley)) &&
                            approximately_equal(partial_radius.x,
                                expected_radius.x) &&
                            approximately_equal(partial_radius.y, 0.0f) &&
                            approximately_equal(partial_radius.z,
                                expected_radius.z) &&
                            rgb_equal(partial_albedo, authored_color) &&
                            partial_transport.x > authored_color.x &&
                            approximately_equal(partial_transport.y, 0.0f) &&
                            partial_transport.z > authored_color.z &&
                            approximately_equal(partial_transport.w, 1.0f);
    if (!partial_ok) {
        dump_case(backend, 2u, actual);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
