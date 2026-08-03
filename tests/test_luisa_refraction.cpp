#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/cycles_abi.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

constexpr std::uint32_t record_count = 28u;
constexpr luisa::float3 barbershop_color{
    0.382274f, 0.651278f, 0.868007f};

[[nodiscard]] ShaderGraph
make_refraction_graph(std::string_view distribution) {
    CyclesNormalizedShaderGraph source;
    source.nodes = {
        {.id = 10u,
         .type = "geometry",
         .variant = {},
         .label = "Geometry",
         .inputs = {},
         .properties = {}},
        {.id = 11u,
         .type = "refraction_bsdf",
         .variant = {},
         .label = "Standalone Refraction regression",
         .inputs = {
             {.socket = "Color",
              .source = std::nullopt,
              .value = SocketValue::color(
                  {barbershop_color.x,
                   barbershop_color.y,
                   barbershop_color.z})},
             {.socket = "Roughness",
              .source = std::nullopt,
              .value = SocketValue::floating(0.35f)},
             {.socket = "IOR",
              .source = std::nullopt,
              .value = SocketValue::floating(1.5f)},
             {.socket = "Normal",
              .source = CyclesOutputRef{
                  .node = 10u,
                  .socket = "Normal"},
              .value = std::nullopt}},
         .properties = {{
             "distribution",
             SocketValue::string(std::string{distribution})}}}};
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 11u, .socket = "BSDF"});
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    if (!adapted.ok()) {
        throw std::runtime_error{
            "failed to adapt normalized standalone Refraction graph"};
    }
    return std::move(*adapted.graph);
}

void set_parameter(std::vector<luisa::float4> &values,
                   const SurfaceProgram &program,
                   std::string_view socket,
                   luisa::float4 value) {
    const auto &parameters = program.parameters();
    for (std::size_t index = 0u; index < parameters.size(); ++index) {
        if (parameters[index].socket == socket) {
            values[index] = value;
            return;
        }
    }
    throw std::runtime_error{
        "missing raw Refraction parameter: " + std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4>
refraction_parameters(const SurfaceProgram &program) {
    const auto defaults = parameter_data(program);
    std::vector<luisa::float4> result;
    result.reserve(defaults.size() * 4u);
    const auto append_case = [&](float roughness,
                                 float ior,
                                 luisa::float3 color) {
        auto values = defaults;
        set_parameter(values,
                      program,
                      "Roughness",
                      {roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
                      program,
                      "IOR",
                      {ior, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
                      program,
                      "Color",
                      {color.x, color.y, color.z, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    };
    append_case(0.35f, 1.5f, barbershop_color);
    append_case(0.0f, 1.5f, barbershop_color);
    append_case(0.35f, 1.0f, barbershop_color);
    append_case(0.35f, 1.5f, {1.0e-6f, 1.0e-6f, 1.0e-6f});
    append_case(-0.35f, 1.5f, barbershop_color);
    return result;
}

[[nodiscard]] bool rgb_equal(luisa::float4 actual,
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

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto ggx_shader = compiler.compile(
        make_refraction_graph("GGX"));
    const auto beckmann_shader = compiler.compile(
        make_refraction_graph("BECKMANN"));
    if (!ggx_shader.ok() || !beckmann_shader.ok()) {
        std::cerr << "failed to compile standalone Refraction graphs\n";
        return EXIT_FAILURE;
    }
    const auto ggx_program =
        compile_surface_program(*ggx_shader.program);
    const auto beckmann_program =
        compile_surface_program(*beckmann_shader.program);
    if (!ggx_program.ok() || !beckmann_program.ok()) {
        std::cerr << "failed to lower standalone Refraction programs\n";
        return EXIT_FAILURE;
    }
    const auto &lowered =
        ggx_program.program->closure_instructions().front();
    if (lowered.operation != ClosureOperation::refraction ||
        !lowered.color.valid() || !lowered.roughness.valid() ||
        !lowered.ior.valid() || !lowered.normal.valid() ||
        lowered.preserve_ggx_energy || lowered.beckmann) {
        std::cerr << "standalone Refraction lost its typed closure identity\n";
        return EXIT_FAILURE;
    }

    const auto ggx_values =
        refraction_parameters(*ggx_program.program);
    const auto beckmann_values =
        parameter_data(*beckmann_program.program);
    const auto parameter_stride = static_cast<std::uint32_t>(
        ggx_program.program->parameters().size());

    SurfaceDispatch surfaces;
    const auto ggx_tag =
        surfaces.create<GraphSurface>(ggx_program.program);
    const auto beckmann_tag =
        surfaces.create<GraphSurface>(beckmann_program.program);

    Kernel1D evaluate = [&](BufferFloat4 ggx_parameter_buffer,
                            BufferFloat4 beckmann_parameter_buffer,
                            BufferFloat4 output) noexcept {
        ParameterShaderServices ggx_services{ggx_parameter_buffer};
        ParameterShaderServices beckmann_services{
            beckmann_parameter_buffer};
        auto point = make_surface_point();
        const auto trace = [&](UInt block,
                               Bool reflection,
                               Bool transmission) noexcept {
            auto local_point = point;
            local_point.parameter_block = block * parameter_stride;
            return surfaces.closure_trace(
                UInt{ggx_tag},
                ggx_services,
                local_point,
                0u,
                reflection,
                transmission);
        };
        const auto write_trace = [&](std::uint32_t record,
                                     const SurfaceClosureTrace &value) noexcept {
            output.write(record,
                         make_float4(cast<float>(value.count),
                                     cast<float>(value.type),
                                     value.sample_weight,
                                     select(0.0f, 1.0f, value.valid)));
        };

        const auto both = trace(0u, true, true);
        write_trace(0u, both);
        output.write(1u, make_float4(both.weight, 0.0f));
        write_trace(2u, trace(0u, true, false));
        write_trace(3u, trace(0u, false, true));
        write_trace(4u, trace(0u, false, false));
        write_trace(5u,
                    surfaces.closure_trace(UInt{beckmann_tag},
                                           beckmann_services,
                                           point,
                                           0u,
                                           true,
                                           true));

        constexpr auto all_lobes = static_cast<std::uint32_t>(
            event_diffuse | event_glossy | event_transmission |
            event_transparent);
        const auto query = SurfaceQuery{
            .lobe_mask = all_lobes,
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto reflected = surfaces.evaluate(
            UInt{ggx_tag},
            ggx_services,
            point,
            make_float3(0.0f, 0.0f, 1.0f),
            query);
        const auto transmitted = surfaces.evaluate(
            UInt{ggx_tag},
            ggx_services,
            point,
            make_float3(0.0f, 0.0f, -1.0f),
            query);
        output.write(6u, make_float4(reflected.f, reflected.pdf));
        output.write(7u, make_float4(transmitted.f, transmitted.pdf));

        const auto light = [&](std::uint32_t flags) noexcept {
            return surfaces.evaluate_light(
                UInt{ggx_tag},
                ggx_services,
                point,
                make_float3(0.0f, 0.0f, -1.0f),
                SurfaceLightQuery{
                    .surface = query,
                    .shader_flags =
                        flags | cycles_abi::shader_use_mis});
        };
        output.write(8u, make_float4(light(0u).f, light(0u).pdf));
        output.write(9u,
                     make_float4(
                         light(cycles_abi::shader_exclude_glossy).f,
                         light(cycles_abi::shader_exclude_glossy).pdf));
        output.write(10u,
                     make_float4(
                         light(cycles_abi::shader_exclude_transmit).f,
                         light(cycles_abi::shader_exclude_transmit).pdf));

        const auto aov = surfaces.aov(
            UInt{ggx_tag}, ggx_services, point);
        output.write(11u,
                     make_float4(aov.glossy_albedo, aov.roughness.x));
        output.write(12u,
                     make_float4(aov.transmission_albedo, aov.roughness.y));

        auto smooth_point = point;
        smooth_point.parameter_block = parameter_stride;
        const auto smooth = surfaces.sample_trace(
            UInt{ggx_tag},
            ggx_services,
            smooth_point,
            0.37f,
            make_float2(0.23f, 0.79f),
            query);
        output.write(13u,
                     make_float4(
                         smooth.sample.evaluation.f,
                         smooth.sample.evaluation.pdf));
        output.write(14u,
                     make_float4(smooth.sample.wi,
                                 select(0.0f, 1.0f, smooth.sample.valid)));
        output.write(15u,
                     make_float4(smooth.sample.roughness.x,
                                 smooth.sample.eta,
                                 cast<float>(smooth.sample.evaluation.events),
                                 cast<float>(smooth.closure_type)));

        const auto sample_mask = [&](std::uint32_t lobe_mask) noexcept {
            auto masked = query;
            masked.lobe_mask = lobe_mask;
            return surfaces.sample_trace(
                UInt{ggx_tag},
                ggx_services,
                smooth_point,
                0.37f,
                make_float2(0.23f, 0.79f),
                masked);
        };
        const auto glossy_only = sample_mask(
            static_cast<std::uint32_t>(event_glossy));
        const auto transmission_only = sample_mask(
            static_cast<std::uint32_t>(event_transmission));
        output.write(16u,
                     make_float4(glossy_only.sample.evaluation.f,
                                 select(0.0f, 1.0f, glossy_only.sample.valid)));
        output.write(17u,
                     make_float4(transmission_only.sample.evaluation.f,
                                 select(0.0f, 1.0f, transmission_only.sample.valid)));

        auto backface = smooth_point;
        backface.back_facing = true;
        const auto backface_sample = surfaces.sample_trace(
            UInt{ggx_tag},
            ggx_services,
            backface,
            0.37f,
            make_float2(0.23f, 0.79f),
            query);
        output.write(18u,
                     make_float4(backface_sample.sample.evaluation.f,
                                 backface_sample.sample.evaluation.pdf));
        output.write(19u,
                     make_float4(backface_sample.sample.wi,
                                 backface_sample.sample.eta));
        output.write(20u,
                     make_float4(
                         cast<float>(backface_sample.sample.evaluation.events),
                         select(0.0f, 1.0f, backface_sample.sample.valid),
                         0.0f,
                         0.0f));

        auto unit_ior = point;
        unit_ior.parameter_block = 2u * parameter_stride;
        const auto unit_ior_evaluation = surfaces.evaluate(
            UInt{ggx_tag},
            ggx_services,
            unit_ior,
            make_float3(0.0f, 0.0f, -1.0f),
            query);
        output.write(21u,
                     make_float4(unit_ior_evaluation.f,
                                 unit_ior_evaluation.pdf));
        write_trace(22u, trace(3u, true, true));

        auto tir_point = backface;
        tir_point.incoming = normalize(
            make_float3(0.8660254f, 0.0f, 0.5f));
        const auto tir = surfaces.sample_trace(
            UInt{ggx_tag},
            ggx_services,
            tir_point,
            0.37f,
            make_float2(0.23f, 0.79f),
            query);
        output.write(23u,
                     make_float4(tir.sample.evaluation.f,
                                 select(0.0f, 1.0f, tir.sample.valid)));
        output.write(24u,
                     make_float4(tir.sample.wi, tir.sample.eta));
        write_trace(25u,
                    surfaces.closure_trace(UInt{ggx_tag},
                                           ggx_services,
                                           tir_point,
                                           0u,
                                           true,
                                           true));

        auto negative_roughness = point;
        negative_roughness.parameter_block = 4u * parameter_stride;
        const auto negative_evaluation = surfaces.evaluate(
            UInt{ggx_tag},
            ggx_services,
            negative_roughness,
            make_float3(0.0f, 0.0f, -1.0f),
            query);
        output.write(26u,
                     make_float4(negative_evaluation.f,
                                 negative_evaluation.pdf));
        const auto negative_aov = surfaces.aov(
            UInt{ggx_tag}, ggx_services, negative_roughness);
        output.write(27u,
                     make_float4(negative_aov.transmission_albedo,
                                 negative_aov.roughness.y));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto ggx_buffer =
        device.create_buffer<luisa::float4>(ggx_values.size());
    auto beckmann_buffer =
        device.create_buffer<luisa::float4>(beckmann_values.size());
    auto output_buffer =
        device.create_buffer<luisa::float4>(record_count);
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, record_count> actual{};
    stream << ggx_buffer.copy_from(luisa::span{ggx_values})
           << beckmann_buffer.copy_from(luisa::span{beckmann_values})
           << kernel(ggx_buffer, beckmann_buffer, output_buffer)
                  .dispatch(1u)
           << output_buffer.copy_to(luisa::span{actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_REFRACTION_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < actual.size(); ++index) {
            const auto value = actual[index];
            std::cerr << index << ": {" << value.x << ", " << value.y
                      << ", " << value.z << ", " << value.w << "}\n";
        }
    }

    constexpr auto ggx_refraction = static_cast<float>(
        cycles_closure::type_microfacet_ggx_refraction);
    constexpr auto beckmann_refraction = static_cast<float>(
        cycles_closure::type_microfacet_beckmann_refraction);
    constexpr auto no_closure = static_cast<float>(
        cycles_closure::type_none);
    const auto meta = [&](std::uint32_t record,
                          float count,
                          float type,
                          bool valid) noexcept {
        return approximately_equal(actual[record].x, count) &&
               approximately_equal(actual[record].y, type) &&
               approximately_equal(
                   actual[record].w, valid ? 1.0f : 0.0f);
    };
    const auto color_average =
        (barbershop_color.x + barbershop_color.y +
         barbershop_color.z) /
        3.0f;
    if (!meta(0u, 1.0f, ggx_refraction, true) ||
        !approximately_equal(actual[0u].z, color_average) ||
        !rgb_equal(actual[1u], barbershop_color) ||
        !meta(2u, 0.0f, no_closure, false) ||
        !meta(3u, 1.0f, ggx_refraction, true) ||
        !meta(4u, 0.0f, no_closure, false) ||
        !meta(5u, 1.0f, beckmann_refraction, true)) {
        std::cerr << "Refraction allocation/type regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    constexpr auto pi = 3.14159265358979323846f;
    constexpr auto alpha = 0.35f * 0.35f;
    constexpr auto distribution = 1.0f / (pi * alpha * alpha);
    constexpr auto transmission_common = 9.0f * distribution;
    if (!finite(actual[6u]) || !finite(actual[7u]) ||
        !rgb_equal(actual[6u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[6u].w, 0.0f) ||
        !rgb_equal(actual[7u],
                   barbershop_color * transmission_common,
                   8.0e-5f) ||
        !approximately_equal(
            actual[7u].w, transmission_common, 8.0e-5f) ||
        !approximately_equal(actual[8u], actual[7u], 8.0e-5f) ||
        !rgb_equal(actual[9u], {0.0f, 0.0f, 0.0f}) ||
        !rgb_equal(actual[10u], {0.0f, 0.0f, 0.0f})) {
        std::cerr << "pure-transmission eval/light-mask regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[11u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[11u].w, 0.35f) ||
        !rgb_equal(actual[12u], barbershop_color) ||
        !approximately_equal(actual[12u].w, 0.35f)) {
        std::cerr << "Refraction split-AOV regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    constexpr auto singular_transmission = static_cast<float>(
        event_singular | event_transmission);
    if (!rgb_equal(actual[13u], barbershop_color * 1.0e6f) ||
        !approximately_equal(actual[13u].w, 1.0e6f) ||
        !approximately_equal(
            actual[14u], {0.0f, 0.0f, -1.0f, 1.0f}) ||
        !approximately_equal(actual[15u].x, 0.0f) ||
        !approximately_equal(actual[15u].y, 1.5f) ||
        !approximately_equal(actual[15u].z, singular_transmission) ||
        !approximately_equal(actual[15u].w, ggx_refraction) ||
        !rgb_equal(actual[16u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[16u].w, 0.0f) ||
        !rgb_equal(actual[17u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[17u].w, 0.0f)) {
        std::cerr << "Refraction singular/lobe-mask regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[18u], barbershop_color * 1.0e6f) ||
        !approximately_equal(actual[18u].w, 1.0e6f) ||
        !approximately_equal(
            actual[19u], {0.0f, 0.0f, -1.0f, 2.0f / 3.0f}) ||
        !approximately_equal(actual[20u].x, singular_transmission) ||
        !approximately_equal(actual[20u].y, 1.0f) ||
        !rgb_equal(actual[21u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[21u].w, 0.0f) ||
        !meta(22u, 0.0f, no_closure, false) ||
        !rgb_equal(actual[23u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[23u].w, 0.0f) ||
        !meta(25u, 1.0f, ggx_refraction, true) ||
        !approximately_equal(actual[25u].z, color_average) ||
        !approximately_equal(actual[26u], actual[7u], 8.0e-5f) ||
        !rgb_equal(actual[27u], barbershop_color) ||
        !approximately_equal(actual[27u].w, 0.35f)) {
        std::cerr << "Refraction eta/TIR/cutoff regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
