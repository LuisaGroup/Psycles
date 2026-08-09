#include <psycles/adapter/cycles_shader_graph.h>
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
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;
using psycles::test_support::surface_aov;

constexpr std::uint32_t record_count = 28u;
constexpr luisa::float3 barbershop_color{
    0.382274f, 0.651278f, 0.868007f};

namespace trace_case {
constexpr std::uint32_t both = 0u;
constexpr std::uint32_t reflection_only = 1u;
constexpr std::uint32_t transmission_only = 2u;
constexpr std::uint32_t neither = 3u;
constexpr std::uint32_t cutoff = 4u;
constexpr std::uint32_t total_internal_reflection = 5u;
constexpr std::uint32_t count = 6u;
}// namespace trace_case

namespace evaluation_case {
constexpr std::uint32_t front_reflection = 0u;
constexpr std::uint32_t front_transmission = 1u;
constexpr std::uint32_t unit_ior = 2u;
constexpr std::uint32_t negative_roughness = 3u;
constexpr std::uint32_t count = 4u;
}// namespace evaluation_case

namespace sample_case {
constexpr std::uint32_t smooth = 0u;
constexpr std::uint32_t glossy_mask = 1u;
constexpr std::uint32_t transmission_mask = 2u;
constexpr std::uint32_t backface = 3u;
constexpr std::uint32_t total_internal_reflection = 4u;
constexpr std::uint32_t count = 5u;
}// namespace sample_case

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

    SurfaceDispatch ggx_surfaces;
    const auto ggx_tag =
        ggx_surfaces.create<GraphSurface>(ggx_program.program);
    SurfaceDispatch beckmann_surfaces;
    const auto beckmann_tag =
        beckmann_surfaces.create<GraphSurface>(beckmann_program.program);

    Kernel1D trace_ggx = [&](BufferFloat4 ggx_parameter_buffer,
                             BufferFloat4 output) noexcept {
        ParameterShaderServices services{ggx_parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        UInt parameter_case = 0u;
        parameter_case = select(
            parameter_case, 3u, case_index == trace_case::cutoff);
        parameter_case = select(parameter_case,
            1u,
            case_index == trace_case::total_internal_reflection);
        point.parameter_block = parameter_case * parameter_stride;
        const auto tir =
            case_index == trace_case::total_internal_reflection;
        point.back_facing = tir;
        point.incoming = select(point.incoming,
            normalize(make_float3(0.8660254f, 0.0f, 0.5f)),
            tir);
        const auto reflection =
            (case_index != trace_case::transmission_only) &
            (case_index != trace_case::neither);
        const auto transmission =
            (case_index != trace_case::reflection_only) &
            (case_index != trace_case::neither);
        const auto value = ggx_surfaces.closure_trace(UInt{ggx_tag},
            services,
            point,
            0u,
            reflection,
            transmission);
        UInt record = case_index;
        record = select(record, 2u, case_index == trace_case::reflection_only);
        record = select(record, 3u, case_index == trace_case::transmission_only);
        record = select(record, 4u, case_index == trace_case::neither);
        record = select(record, 22u, case_index == trace_case::cutoff);
        record = select(record,
            25u,
            case_index == trace_case::total_internal_reflection);
        output.write(record,
            make_float4(cast<float>(value.count),
                cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
        $if(case_index == trace_case::both) {
            output.write(1u, make_float4(value.weight, 0.0f));
        };
    };

    Kernel1D trace_beckmann = [&](BufferFloat4 parameter_buffer,
                                  BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto point = make_surface_point();
        const auto value = beckmann_surfaces.closure_trace(
            UInt{beckmann_tag}, services, point, 0u, true, true);
        output.write(5u,
            make_float4(cast<float>(value.count),
                cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
    };

    Kernel1D evaluate_surface = [&](BufferFloat4 parameter_buffer,
                                    BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        UInt parameter_case = 0u;
        parameter_case = select(parameter_case,
            2u,
            case_index == evaluation_case::unit_ior);
        parameter_case = select(parameter_case,
            4u,
            case_index == evaluation_case::negative_roughness);
        point.parameter_block = parameter_case * parameter_stride;
        const auto outgoing = select(make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            case_index == evaluation_case::front_reflection);
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
        const auto value = ggx_surfaces.evaluate(
            UInt{ggx_tag}, services, point, outgoing, query);
        UInt record = 6u + case_index;
        record = select(record,
            21u,
            case_index == evaluation_case::unit_ior);
        record = select(record,
            26u,
            case_index == evaluation_case::negative_roughness);
        output.write(record, make_float4(value.f, value.pdf));
    };

    Kernel1D evaluate_light = [&](BufferFloat4 parameter_buffer,
                                  BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto case_index = dispatch_x();
        const auto point = make_surface_point();
        UInt shader_flags = 0u;
        shader_flags = select(shader_flags,
            cycles_abi::shader_exclude_glossy,
            case_index == 1u);
        shader_flags = select(shader_flags,
            cycles_abi::shader_exclude_transmit,
            case_index == 2u);
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
        const auto value = ggx_surfaces.evaluate_light(UInt{ggx_tag},
            services,
            point,
            make_float3(0.0f, 0.0f, -1.0f),
            SurfaceLightQuery{
                .surface = query,
                .shader_flags =
                    shader_flags | cycles_abi::shader_use_mis});
        output.write(8u + case_index, make_float4(value.f, value.pdf));
    };

    Kernel1D evaluate_aov = [&](BufferFloat4 parameter_buffer,
                                BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = select(0u, 4u, case_index != 0u) *
                                parameter_stride;
        const auto aov =
            surface_aov(ggx_surfaces, UInt{ggx_tag}, services, point);
        $if(case_index == 0u) {
            output.write(11u,
                make_float4(aov.glossy_albedo, aov.roughness.x));
            output.write(12u,
                make_float4(
                    aov.transmission_albedo, aov.roughness.y));
        }
        $else {
            output.write(27u,
                make_float4(
                    aov.transmission_albedo, aov.roughness.y));
        };
    };

    Kernel1D sample_surface = [&](BufferFloat4 parameter_buffer,
                                  BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = parameter_stride;
        point.back_facing = case_index >= sample_case::backface;
        point.incoming = select(point.incoming,
            normalize(make_float3(0.8660254f, 0.0f, 0.5f)),
            case_index == sample_case::total_internal_reflection);
        UInt lobe_mask = static_cast<std::uint32_t>(
            event_diffuse | event_glossy | event_transmission |
            event_transparent);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(event_glossy),
            case_index == sample_case::glossy_mask);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(event_transmission),
            case_index == sample_case::transmission_mask);
        const auto query = SurfaceQuery{
            .lobe_mask = lobe_mask,
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = ggx_surfaces.sample_trace(UInt{ggx_tag},
            services,
            point,
            0.37f,
            make_float2(0.23f, 0.79f),
            query);
        const auto valid = select(0.0f, 1.0f, value.sample.valid);
        $if(case_index == sample_case::smooth) {
            output.write(13u,
                make_float4(value.sample.evaluation.f,
                    value.sample.evaluation.pdf));
            output.write(
                14u, make_float4(value.sample.wi, valid));
            output.write(15u,
                make_float4(value.sample.roughness.x,
                    value.sample.eta,
                    cast<float>(value.sample.evaluation.events),
                    cast<float>(value.closure_type)));
        }
        $elif((case_index == sample_case::glossy_mask) |
              (case_index == sample_case::transmission_mask)) {
            output.write(15u + case_index,
                make_float4(value.sample.evaluation.f, valid));
        }
        $elif(case_index == sample_case::backface) {
            output.write(18u,
                make_float4(value.sample.evaluation.f,
                    value.sample.evaluation.pdf));
            output.write(19u,
                make_float4(value.sample.wi, value.sample.eta));
            output.write(20u,
                make_float4(
                    cast<float>(value.sample.evaluation.events),
                    valid,
                    0.0f,
                    0.0f));
        }
        $else {
            output.write(23u,
                make_float4(value.sample.evaluation.f, valid));
            output.write(24u,
                make_float4(value.sample.wi, value.sample.eta));
        };
    };

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "refraction_trace_ggx", trace_ggx, 3500u);
        bounded &= require_bounded_xir(
            "refraction_trace_beckmann", trace_beckmann, 3300u);
        bounded &= require_bounded_xir(
            "refraction_evaluate", evaluate_surface, 15000u);
        bounded &= require_bounded_xir(
            "refraction_evaluate_light", evaluate_light, 15000u);
        bounded &= require_bounded_xir(
            "refraction_aov", evaluate_aov, 4000u);
        bounded &= require_bounded_xir(
            "refraction_sample", sample_surface, 30000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto ggx_buffer =
        device.create_buffer<luisa::float4>(ggx_values.size());
    auto beckmann_buffer =
        device.create_buffer<luisa::float4>(beckmann_values.size());
    auto output_buffer =
        device.create_buffer<luisa::float4>(record_count);
    auto trace_ggx_kernel =
        compile_named_kernel(device, "refraction_trace_ggx", trace_ggx);
    auto trace_beckmann_kernel = compile_named_kernel(
        device, "refraction_trace_beckmann", trace_beckmann);
    auto evaluation_kernel = compile_named_kernel(
        device, "refraction_evaluate", evaluate_surface);
    auto light_kernel = compile_named_kernel(
        device, "refraction_evaluate_light", evaluate_light);
    auto aov_kernel =
        compile_named_kernel(device, "refraction_aov", evaluate_aov);
    auto sample_kernel = compile_named_kernel(
        device, "refraction_sample", sample_surface);
    std::array<luisa::float4, record_count> actual{};
    stream << ggx_buffer.copy_from(luisa::span{ggx_values})
           << beckmann_buffer.copy_from(luisa::span{beckmann_values})
           << trace_ggx_kernel(ggx_buffer, output_buffer)
                  .dispatch(trace_case::count)
           << trace_beckmann_kernel(beckmann_buffer, output_buffer)
                  .dispatch(1u)
           << evaluation_kernel(ggx_buffer, output_buffer)
                  .dispatch(evaluation_case::count)
           << light_kernel(ggx_buffer, output_buffer).dispatch(3u)
           << aov_kernel(ggx_buffer, output_buffer).dispatch(2u)
           << sample_kernel(ggx_buffer, output_buffer)
                  .dispatch(sample_case::count)
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
    // Cycles classifies refraction in CLOSURE_IS_BSDF_TRANSMISSION, outside
    // CLOSURE_IS_BSDF_GLOSSY. EXCLUDE_GLOSSY therefore retains the value and
    // PDF; EXCLUDE_TRANSMIT removes only the value from the same PDF mixture.
    if (!finite(actual[6u]) || !finite(actual[7u]) ||
        !rgb_equal(actual[6u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[6u].w, 0.0f) ||
        !rgb_equal(actual[7u],
                   barbershop_color * transmission_common,
                   8.0e-5f) ||
        !approximately_equal(
            actual[7u].w, transmission_common, 8.0e-5f) ||
        !approximately_equal(actual[8u], actual[7u], 8.0e-5f) ||
        !approximately_equal(actual[9u], actual[7u], 8.0e-5f) ||
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
