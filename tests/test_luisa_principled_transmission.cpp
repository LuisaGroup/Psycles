#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
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
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr std::uint32_t record_count = 32u;

[[nodiscard]] ShaderGraph
make_principled_transmission_graph(std::string_view distribution) {
    ShaderGraph graph;
    const auto principled = graph.add_node(
        node_type::principled_bsdf, "Raw Principled Transmission regression");
    const auto configured =
        graph.set_input(principled, "BaseColor",
                        SocketValue::color({0.25f, 0.64f, 1.0f})) &&
        graph.set_input(principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "Roughness", SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "DiffuseRoughness",
                        SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "SubsurfaceWeight",
                        SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "TransmissionWeight",
                        SocketValue::floating(1.0f)) &&
        graph.set_input(principled, "IOR", SocketValue::floating(1.5f)) &&
        graph.set_input(principled, "SpecularIORLevel",
                        SocketValue::floating(0.5f)) &&
        graph.set_input(principled, "SpecularTint",
                        SocketValue::color({0.5f, 1.0f, 0.25f})) &&
        graph.set_input(principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.set_property(principled, "Distribution",
                           SocketValue::string(std::string{distribution}));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure Principled Transmission graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

void set_parameter(std::vector<luisa::float4> &values,
                   const SurfaceProgram &program, std::string_view socket,
                   luisa::float4 value) {
    const auto &parameters = program.parameters();
    for (std::size_t index = 0u; index < parameters.size(); ++index) {
        if (parameters[index].socket == socket) {
            values[index] = value;
            return;
        }
    }
    throw std::runtime_error{"missing raw surface parameter: " +
                             std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4>
transmission_parameters(const SurfaceProgram &program) {
    const auto defaults = parameter_data(program);
    std::vector<luisa::float4> result;
    result.reserve(defaults.size() * 4u);
    const auto append_case = [&](float transmission, float roughness) {
        auto values = defaults;
        set_parameter(values, program, "TransmissionWeight",
                      {transmission, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "Roughness", {roughness, 0.0f, 0.0f, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    };
    append_case(1.0f, 0.0f);
    append_case(1.0e-5f, 0.0f);
    append_case(2.0e-5f, 0.0f);
    append_case(1.0f, 0.35f);
    return result;
}

[[nodiscard]] bool rgb_equal(luisa::float4 actual, luisa::float3 expected,
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
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto ggx_shader =
        compiler.compile(make_principled_transmission_graph("GGX"));
    const auto multi_shader =
        compiler.compile(make_principled_transmission_graph("MULTI_GGX"));
    if (!ggx_shader.ok() || !multi_shader.ok()) {
        std::cerr << "failed to compile Transmission graphs\n";
        return EXIT_FAILURE;
    }
    const auto ggx_program = compile_surface_program(*ggx_shader.program);
    const auto multi_program = compile_surface_program(*multi_shader.program);
    if (!ggx_program.ok() || !multi_program.ok()) {
        std::cerr << "failed to lower Transmission surface programs\n";
        return EXIT_FAILURE;
    }

    bool raw_socket_found = false;
    for (const auto &parameter : ggx_program.program->parameters()) {
        raw_socket_found |= parameter.socket == "TransmissionWeight";
    }
    const auto &lowered = ggx_program.program->closure_instructions().front();
    if (!raw_socket_found || !lowered.transmission_weight.valid()) {
        std::cerr << "raw Transmission Weight was lost before Luisa setup\n";
        return EXIT_FAILURE;
    }

    const auto ggx_values = transmission_parameters(*ggx_program.program);
    const auto multi_values = parameter_data(*multi_program.program);
    const auto parameter_stride =
        static_cast<std::uint32_t>(ggx_program.program->parameters().size());

    SurfaceDispatch surfaces;
    const auto ggx_tag = surfaces.create<GraphSurface>(ggx_program.program);
    const auto multi_tag = surfaces.create<GraphSurface>(multi_program.program);

    Kernel1D evaluate = [&](BufferFloat4 ggx_parameter_buffer,
                            BufferFloat4 multi_parameter_buffer,
                            BufferFloat4 output) noexcept {
        ParameterShaderServices ggx_services{ggx_parameter_buffer};
        ParameterShaderServices multi_services{multi_parameter_buffer};
        auto point = make_surface_point();
        const auto trace = [&](UInt block, UInt index, Bool reflection,
                               Bool transmission) noexcept {
            auto local_point = point;
            local_point.parameter_block = block * parameter_stride;
            return surfaces.closure_trace(UInt{ggx_tag}, ggx_services, local_point,
                                          index, reflection, transmission);
        };
        const auto write_trace = [&](std::uint32_t record,
                                     const SurfaceClosureTrace &value) noexcept {
            output.write(record,
                         make_float4(cast<float>(value.count),
                                     cast<float>(value.type), value.sample_weight,
                                     select(0.0f, 1.0f, value.valid)));
        };

        const auto both = trace(0u, 0u, true, true);
        const auto reflection_only = trace(0u, 0u, true, false);
        const auto transmission_only = trace(0u, 0u, false, true);
        const auto neither = trace(0u, 0u, false, false);
        write_trace(0u, both);
        output.write(1u, make_float4(both.weight, 0.0f));
        write_trace(2u, reflection_only);
        output.write(3u, make_float4(reflection_only.weight, 0.0f));
        write_trace(4u, transmission_only);
        output.write(5u, make_float4(transmission_only.weight, 0.0f));
        write_trace(6u, neither);
        write_trace(7u, trace(1u, 0u, true, true));
        write_trace(8u, trace(2u, 0u, true, true));
        write_trace(9u, trace(2u, 1u, true, true));
        write_trace(10u, trace(2u, 2u, true, true));

        const auto all_query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto reflection_query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = false};
        const auto transmission_query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = false,
            .refractive_caustics = true};
        const auto write_sample = [&](std::uint32_t record,
                                      const SurfaceSampleTrace &sample) noexcept {
            output.write(record, make_float4(sample.sample.evaluation.f,
                                             sample.sample.evaluation.pdf));
            output.write(record + 1u,
                         make_float4(sample.sample.wi,
                                     select(0.0f, 1.0f, sample.sample.valid)));
            output.write(record + 2u,
                         make_float4(sample.sample.roughness.x, sample.sample.eta,
                                     cast<float>(sample.sample.evaluation.events),
                                     cast<float>(sample.closure_type)));
        };
        write_sample(11u,
                     surfaces.sample_trace(UInt{ggx_tag}, ggx_services, point, 0.0f,
                                           make_float2(0.37f, 0.73f), all_query));
        write_sample(14u,
                     surfaces.sample_trace(UInt{ggx_tag}, ggx_services, point, 0.5f,
                                           make_float2(0.37f, 0.73f), all_query));
        write_sample(17u, surfaces.sample_trace(UInt{ggx_tag}, ggx_services, point,
                                                0.9f, make_float2(0.37f, 0.73f),
                                                reflection_query));
        write_sample(20u, surfaces.sample_trace(UInt{ggx_tag}, ggx_services, point,
                                                0.0f, make_float2(0.37f, 0.73f),
                                                transmission_query));
        auto backface = point;
        backface.back_facing = true;
        const auto backface_sample =
            surfaces.sample_trace(UInt{ggx_tag}, ggx_services, backface, 0.0f,
                                  make_float2(0.37f, 0.73f), transmission_query);
        output.write(23u, make_float4(backface_sample.sample.evaluation.f,
                                      backface_sample.sample.evaluation.pdf));
        output.write(24u, make_float4(backface_sample.sample.wi,
                                      backface_sample.sample.eta));
        output.write(
            25u, make_float4(cast<float>(backface_sample.sample.evaluation.events),
                             select(0.0f, 1.0f, backface_sample.sample.valid), 0.0f,
                             0.0f));

        const auto aov = surfaces.aov(UInt{ggx_tag}, ggx_services, point);
        output.write(26u, make_float4(aov.glossy_albedo, aov.roughness.x));
        output.write(27u, make_float4(aov.transmission_albedo, aov.roughness.y));

        auto rough_point = point;
        rough_point.parameter_block = 3u * parameter_stride;
        const auto reflected =
            surfaces.evaluate(UInt{ggx_tag}, ggx_services, rough_point,
                              make_float3(0.0f, 0.0f, 1.0f), all_query);
        const auto transmitted =
            surfaces.evaluate(UInt{ggx_tag}, ggx_services, rough_point,
                              make_float3(0.0f, 0.0f, -1.0f), all_query);
        output.write(28u, make_float4(reflected.f, reflected.pdf));
        output.write(29u, make_float4(transmitted.f, transmitted.pdf));

        const auto multi_trace = surfaces.closure_trace(
            UInt{multi_tag}, multi_services, point, 0u, true, true);
        write_trace(30u, multi_trace);
        output.write(31u, make_float4(multi_trace.weight, 0.0f));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto ggx_buffer = device.create_buffer<luisa::float4>(ggx_values.size());
    auto multi_buffer = device.create_buffer<luisa::float4>(multi_values.size());
    auto output_buffer = device.create_buffer<luisa::float4>(record_count);
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, record_count> actual{};
    stream << ggx_buffer.copy_from(luisa::span{ggx_values})
           << multi_buffer.copy_from(luisa::span{multi_values})
           << kernel(ggx_buffer, multi_buffer, output_buffer).dispatch(1u)
           << output_buffer.copy_to(luisa::span{actual}) << synchronize();

    if (std::getenv("PSYCLES_DUMP_TRANSMISSION_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < actual.size(); ++index) {
            const auto value = actual[index];
            std::cerr << index << ": {" << value.x << ", " << value.y << ", "
                      << value.z << ", " << value.w << "}\n";
        }
    }

    constexpr auto ggx_glass =
        static_cast<float>(cycles_closure::type_microfacet_ggx_glass);
    constexpr auto multi_ggx_glass =
        static_cast<float>(cycles_closure::type_microfacet_multi_ggx_glass);
    constexpr auto ggx = static_cast<float>(cycles_closure::type_microfacet_ggx);
    constexpr auto diffuse = static_cast<float>(cycles_closure::type_diffuse);
    const auto meta = [&](std::uint32_t record, float count, float type,
                          bool valid) noexcept {
        return approximately_equal(actual[record].x, count) &&
               approximately_equal(actual[record].y, type) &&
               approximately_equal(actual[record].w, valid ? 1.0f : 0.0f);
    };

    const auto f0 = luisa::float3{0.02f, 0.04f, 0.01f};
    const auto transmission = luisa::float3{0.49f, 0.768f, 0.99f};
    const auto estimated_reflection = luisa::float3{0.51f, 0.52f, 0.505f};
    const auto estimated_transmission = luisa::float3{0.245f, 0.384f, 0.495f};
    const auto both_sample_weight = (0.755f + 0.904f + 1.0f) / 3.0f;
    const auto reflection_sample_weight = (0.51f + 0.52f + 0.505f) / 3.0f;
    const auto transmission_sample_weight = (0.245f + 0.384f + 0.495f) / 3.0f;
    if (!meta(0u, 1.0f, ggx_glass, true) ||
        !approximately_equal(actual[0u].z, both_sample_weight) ||
        !meta(2u, 1.0f, ggx_glass, true) ||
        !approximately_equal(actual[2u].z, reflection_sample_weight) ||
        !meta(4u, 1.0f, ggx_glass, true) ||
        !approximately_equal(actual[4u].z, transmission_sample_weight) ||
        !meta(6u, 0.0f, static_cast<float>(cycles_closure::type_none), false) ||
        !rgb_equal(actual[1u], {1.0f, 1.0f, 1.0f}) ||
        !rgb_equal(actual[3u], {1.0f, 1.0f, 1.0f}) ||
        !rgb_equal(actual[5u], {1.0f, 1.0f, 1.0f})) {
        std::cerr << "Transmission caustic allocation/tint regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(7u, 2.0f, ggx, true) || !meta(8u, 3.0f, ggx_glass, true) ||
        !meta(9u, 3.0f, ggx, true) || !meta(10u, 3.0f, diffuse, true)) {
        std::cerr << "Transmission cutoff/closure-order regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto reflection_average = (f0.x + f0.y + f0.z) / 3.0f;
    const auto transmission_average =
        (transmission.x + transmission.y + transmission.z) / 3.0f;
    const auto reflection_probability =
        reflection_average / (reflection_average + transmission_average);
    constexpr auto singular_reflection =
        static_cast<float>(event_singular | event_reflection);
    constexpr auto singular_transmission =
        static_cast<float>(event_singular | event_transmission);
    if (!rgb_equal(actual[11u], f0 * 1.0e6f) ||
        !approximately_equal(actual[11u].w, reflection_probability * 1.0e6f) ||
        !approximately_equal(actual[12u], {0.0f, 0.0f, 1.0f, 1.0f}) ||
        !approximately_equal(actual[13u].x, 0.0f) ||
        !approximately_equal(actual[13u].y, 1.0f) ||
        !approximately_equal(actual[13u].z, singular_reflection) ||
        !approximately_equal(actual[13u].w, ggx_glass) ||
        !rgb_equal(actual[14u], transmission * 1.0e6f) ||
        !approximately_equal(actual[14u].w,
                             (1.0f - reflection_probability) * 1.0e6f) ||
        !approximately_equal(actual[15u], {0.0f, 0.0f, -1.0f, 1.0f}) ||
        !approximately_equal(actual[16u].y, 1.5f) ||
        !approximately_equal(actual[16u].z, singular_transmission)) {
        std::cerr << "spectral singular-lobe regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[17u], f0 * 1.0e6f) ||
        !approximately_equal(actual[17u].w, 1.0e6f) ||
        !approximately_equal(actual[19u].z, singular_reflection) ||
        !rgb_equal(actual[20u], transmission * 1.0e6f) ||
        !approximately_equal(actual[20u].w, 1.0e6f) ||
        !approximately_equal(actual[22u].z, singular_transmission) ||
        !rgb_equal(actual[23u], transmission * 1.0e6f) ||
        !approximately_equal(actual[23u].w, 1.0e6f) ||
        !approximately_equal(actual[24u], {0.0f, 0.0f, -1.0f, 2.0f / 3.0f}) ||
        !approximately_equal(actual[25u].x, singular_transmission) ||
        !approximately_equal(actual[25u].y, 1.0f)) {
        std::cerr << "caustic gate/backface eta regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[26u], estimated_reflection) ||
        !rgb_equal(actual[27u], estimated_transmission) ||
        !approximately_equal(actual[26u].w, 0.0f) ||
        !approximately_equal(actual[27u].w, 0.0f)) {
        std::cerr << "separate glass AOV regression failed on " << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto roughness = 0.35f;
    const auto alpha = roughness * roughness;
    const auto distribution_at_normal =
        1.0f / (3.14159265358979323846f * alpha * alpha);
    const auto reflection_common = 0.25f * distribution_at_normal;
    const auto transmission_common = 9.0f * distribution_at_normal;
    if (!finite(actual[28u]) || !finite(actual[29u]) ||
        !rgb_equal(actual[28u], f0 * reflection_common, 8.0e-5f) ||
        !approximately_equal(
            actual[28u].w, reflection_common * reflection_probability, 8.0e-5f) ||
        !rgb_equal(actual[29u], transmission * transmission_common, 8.0e-5f) ||
        !approximately_equal(
            actual[29u].w, transmission_common * (1.0f - reflection_probability),
            8.0e-5f)) {
        std::cerr << "rough generalized-glass eval/PDF regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto expected_multi_weight =
        luisa::float3{2.0f / 3.0f, 5.0f / 6.0f, 1.0f};
    if (!meta(30u, 1.0f, multi_ggx_glass, true) ||
        !rgb_equal(actual[31u], expected_multi_weight)) {
        std::cerr << "Multi-GGX glass energy regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
