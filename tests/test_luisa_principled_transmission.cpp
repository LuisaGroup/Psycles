#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
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

constexpr std::uint32_t record_count = 35u;

namespace output_record {
constexpr std::uint32_t trace_base = 0u;
constexpr std::uint32_t sample_base = 11u;
constexpr std::uint32_t aov_reflection = 26u;
constexpr std::uint32_t aov_transmission = 27u;
constexpr std::uint32_t rough_evaluation_base = 28u;
constexpr std::uint32_t multi_trace = 30u;
constexpr std::uint32_t multi_weight = 31u;
constexpr std::uint32_t rough_backface = 32u;
constexpr std::uint32_t unit_ior = 33u;
constexpr std::uint32_t beckmann = 34u;
}// namespace output_record

namespace trace_case {
constexpr std::uint32_t both = 0u;
constexpr std::uint32_t reflection_only = 1u;
constexpr std::uint32_t transmission_only = 2u;
constexpr std::uint32_t neither = 3u;
constexpr std::uint32_t below_cutoff = 4u;
constexpr std::uint32_t cutoff_glass = 5u;
constexpr std::uint32_t cutoff_reflection = 6u;
constexpr std::uint32_t cutoff_diffuse = 7u;
constexpr std::uint32_t count = 8u;
}// namespace trace_case

namespace sample_case {
constexpr std::uint32_t both_reflection = 0u;
constexpr std::uint32_t both_transmission = 1u;
constexpr std::uint32_t reflection_only = 2u;
constexpr std::uint32_t transmission_only = 3u;
constexpr std::uint32_t backface_transmission = 4u;
constexpr std::uint32_t count = 5u;
}// namespace sample_case

namespace evaluation_case {
constexpr std::uint32_t rough_reflection = 0u;
constexpr std::uint32_t rough_transmission = 1u;
constexpr std::uint32_t rough_backface = 2u;
constexpr std::uint32_t unit_ior = 3u;
constexpr std::uint32_t count = 4u;
}// namespace evaluation_case

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

[[nodiscard]] ShaderGraph make_beckmann_glass_graph() {
    ShaderGraph graph;
    const auto glass = graph.add_node(
        node_type::glass_bsdf, "Beckmann VNDF density regression");
    const auto configured =
        graph.set_input(glass, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(glass, "Roughness", SocketValue::floating(0.8f)) &&
        graph.set_input(glass, "IOR", SocketValue::floating(1.5f)) &&
        graph.set_property(glass, "Distribution", SocketValue::string("BECKMANN"));
    if (!configured) {
        throw std::runtime_error{"failed to configure Beckmann Glass graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = glass, .socket = "Closure"});
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
    result.reserve(defaults.size() * 5u);
    const auto append_case = [&](float transmission, float roughness,
                                 float ior = 1.5f) {
        auto values = defaults;
        set_parameter(values, program, "TransmissionWeight",
                      {transmission, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "Roughness", {roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "IOR", {ior, 0.0f, 0.0f, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    };
    append_case(1.0f, 0.0f);
    append_case(1.0e-5f, 0.0f);
    append_case(2.0e-5f, 0.0f);
    append_case(1.0f, 0.35f);
    append_case(1.0f, 0.35f, 1.0f);
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
    const auto beckmann_shader = compiler.compile(make_beckmann_glass_graph());
    if (!ggx_shader.ok() || !multi_shader.ok() || !beckmann_shader.ok()) {
        std::cerr << "failed to compile Transmission graphs\n";
        return EXIT_FAILURE;
    }
    const auto ggx_program = compile_surface_program(*ggx_shader.program);
    const auto multi_program = compile_surface_program(*multi_shader.program);
    const auto beckmann_program =
        compile_surface_program(*beckmann_shader.program);
    if (!ggx_program.ok() || !multi_program.ok() || !beckmann_program.ok()) {
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
    const auto beckmann_values = parameter_data(*beckmann_program.program);
    const auto parameter_stride =
        static_cast<std::uint32_t>(ggx_program.program->parameters().size());

    SurfaceDispatch ggx_surfaces;
    const auto ggx_tag = ggx_surfaces.create<GraphSurface>(
        ggx_program.program,
        merged_surface_closure_plan(*ggx_program.program, ggx_values));
    SurfaceDispatch multi_surfaces;
    const auto multi_tag = multi_surfaces.create<GraphSurface>(
        multi_program.program,
        merged_surface_closure_plan(*multi_program.program, multi_values));
    SurfaceDispatch beckmann_surfaces;
    const auto beckmann_tag = beckmann_surfaces.create<GraphSurface>(
        beckmann_program.program,
        merged_surface_closure_plan(
            *beckmann_program.program, beckmann_values));

    Kernel1D trace_ggx = [&](BufferFloat4 ggx_parameter_buffer,
                             BufferFloat4 output) noexcept {
        ParameterShaderServices ggx_services{ggx_parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        UInt parameter_case = 0u;
        parameter_case = select(parameter_case,
            1u,
            case_index == trace_case::below_cutoff);
        parameter_case = select(parameter_case,
            2u,
            case_index >= trace_case::cutoff_glass);
        point.parameter_block = parameter_case * parameter_stride;
        const auto closure_index = select(0u,
            case_index - trace_case::cutoff_glass,
            case_index >= trace_case::cutoff_glass);
        const auto reflection =
            (case_index != trace_case::transmission_only) &
            (case_index != trace_case::neither);
        const auto transmission =
            (case_index != trace_case::reflection_only) &
            (case_index != trace_case::neither);
        const auto value = ggx_surfaces.closure_trace(UInt{ggx_tag},
            ggx_services,
            point,
            closure_index,
            reflection,
            transmission);
        const auto record = select(case_index * 2u,
            case_index + 3u,
            case_index >= trace_case::below_cutoff);
        output.write(record,
            make_float4(cast<float>(value.count),
                cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
        $if(case_index < trace_case::neither) {
            output.write(record + 1u, make_float4(value.weight, 0.0f));
        };
    };

    Kernel1D sample_ggx = [&](BufferFloat4 ggx_parameter_buffer,
                              BufferFloat4 output) noexcept {
        ParameterShaderServices services{ggx_parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.back_facing =
            case_index == sample_case::backface_transmission;
        Float u_lobe = 0.0f;
        u_lobe = select(
            u_lobe, 0.5f, case_index == sample_case::both_transmission);
        u_lobe = select(
            u_lobe, 0.9f, case_index == sample_case::reflection_only);
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics =
                case_index < sample_case::transmission_only,
            .refractive_caustics =
                case_index != sample_case::reflection_only};
        const auto value = ggx_surfaces.sample_trace(UInt{ggx_tag},
            services,
            point,
            u_lobe,
            make_float2(0.37f, 0.73f),
            query);
        const auto backface =
            case_index == sample_case::backface_transmission;
        const auto valid = select(0.0f, 1.0f, value.sample.valid);
        const auto record =
            output_record::sample_base + case_index * 3u;
        output.write(record,
            make_float4(value.sample.evaluation.f,
                value.sample.evaluation.pdf));
        output.write(record + 1u,
            select(make_float4(value.sample.wi, valid),
                make_float4(value.sample.wi, value.sample.eta),
                backface));
        output.write(record + 2u,
            select(make_float4(value.sample.roughness.x,
                       value.sample.eta,
                       cast<float>(value.sample.evaluation.events),
                       cast<float>(value.closure_type)),
                make_float4(
                    cast<float>(value.sample.evaluation.events),
                    valid,
                    0.0f,
                    0.0f),
                backface));
    };

    Kernel1D evaluate_ggx = [&](BufferFloat4 ggx_parameter_buffer,
                                BufferFloat4 output) noexcept {
        ParameterShaderServices services{ggx_parameter_buffer};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block =
            select(3u, 4u, case_index == evaluation_case::unit_ior) *
            parameter_stride;
        point.back_facing =
            case_index == evaluation_case::rough_backface;
        const auto outgoing = select(make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            case_index == evaluation_case::rough_reflection);
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = ggx_surfaces.evaluate(
            UInt{ggx_tag}, services, point, outgoing, query);
        const auto record = output_record::rough_evaluation_base +
                            case_index +
                            select(0u, 2u, case_index >= 2u);
        output.write(record, make_float4(value.f, value.pdf));
    };

    Kernel1D evaluate_aov = [&](BufferFloat4 ggx_parameter_buffer,
                                BufferFloat4 output) noexcept {
        ParameterShaderServices services{ggx_parameter_buffer};
        const auto point = make_surface_point();
        const auto aov =
            surface_aov(ggx_surfaces, UInt{ggx_tag}, services, point);
        output.write(output_record::aov_reflection,
            make_float4(aov.glossy_albedo, aov.roughness.x));
        output.write(output_record::aov_transmission,
            make_float4(aov.transmission_albedo, aov.roughness.y));
    };

    Kernel1D trace_multi = [&](BufferFloat4 multi_parameter_buffer,
                               BufferFloat4 output) noexcept {
        ParameterShaderServices services{multi_parameter_buffer};
        const auto point = make_surface_point();
        const auto value = multi_surfaces.closure_trace(
            UInt{multi_tag}, services, point, 0u, true, true);
        output.write(output_record::multi_trace,
            make_float4(cast<float>(value.count),
                cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
        output.write(output_record::multi_weight,
            make_float4(value.weight, 0.0f));
    };

    Kernel1D evaluate_beckmann =
        [&](BufferFloat4 beckmann_parameter_buffer,
            BufferFloat4 output) noexcept {
        ParameterShaderServices services{beckmann_parameter_buffer};
        auto beckmann_point = make_surface_point();
        beckmann_point.incoming =
            normalize(make_float3(0.9539392f, 0.0f, 0.3f));
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto value = beckmann_surfaces.evaluate(UInt{beckmann_tag},
            services,
            beckmann_point,
            normalize(make_float3(-0.9539392f, 0.0f, 0.3f)),
            query);
        output.write(
            output_record::beckmann, make_float4(value.f, value.pdf));
    };

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "transmission_trace_ggx", trace_ggx, 35000u);
        bounded &= require_bounded_xir(
            "transmission_sample_ggx", sample_ggx, 390000u);
        bounded &= require_bounded_xir(
            "transmission_evaluate_ggx", evaluate_ggx, 205000u);
        bounded &= require_bounded_xir(
            "transmission_evaluate_aov", evaluate_aov, 27000u);
        bounded &= require_bounded_xir(
            "transmission_trace_multi", trace_multi, 37000u);
        bounded &= require_bounded_xir("transmission_evaluate_beckmann",
            evaluate_beckmann,
            16000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto ggx_buffer = device.create_buffer<luisa::float4>(ggx_values.size());
    auto multi_buffer = device.create_buffer<luisa::float4>(multi_values.size());
    auto beckmann_buffer =
        device.create_buffer<luisa::float4>(beckmann_values.size());
    auto output_buffer = device.create_buffer<luisa::float4>(record_count);
    auto trace_ggx_kernel =
        compile_named_kernel(device, "transmission_trace_ggx", trace_ggx);
    auto sample_ggx_kernel =
        compile_named_kernel(device, "transmission_sample_ggx", sample_ggx);
    auto evaluate_ggx_kernel = compile_named_kernel(
        device, "transmission_evaluate_ggx", evaluate_ggx);
    auto evaluate_aov_kernel = compile_named_kernel(
        device, "transmission_evaluate_aov", evaluate_aov);
    auto trace_multi_kernel = compile_named_kernel(
        device, "transmission_trace_multi", trace_multi);
    auto evaluate_beckmann_kernel = compile_named_kernel(
        device, "transmission_evaluate_beckmann", evaluate_beckmann);
    std::array<luisa::float4, record_count> actual{};
    stream << ggx_buffer.copy_from(luisa::span{ggx_values})
           << multi_buffer.copy_from(luisa::span{multi_values})
           << beckmann_buffer.copy_from(luisa::span{beckmann_values})
           << trace_ggx_kernel(ggx_buffer, output_buffer)
                  .dispatch(trace_case::count)
           << sample_ggx_kernel(ggx_buffer, output_buffer)
                  .dispatch(sample_case::count)
           << evaluate_ggx_kernel(ggx_buffer, output_buffer)
                  .dispatch(evaluation_case::count)
           << evaluate_aov_kernel(ggx_buffer, output_buffer).dispatch(1u)
           << trace_multi_kernel(multi_buffer, output_buffer).dispatch(1u)
           << evaluate_beckmann_kernel(beckmann_buffer, output_buffer)
                  .dispatch(1u)
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
    if (!meta(30u, 1.0f, ggx_glass, true) ||
        !rgb_equal(actual[31u], expected_multi_weight)) {
        std::cerr << "Multi-GGX glass energy regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    const auto backface_transmission_common = 4.0f * distribution_at_normal;
    if (!finite(actual[32u]) ||
        !rgb_equal(actual[32u], transmission * backface_transmission_common,
                   8.0e-5f) ||
        !approximately_equal(
            actual[32u].w,
            backface_transmission_common * (1.0f - reflection_probability),
            8.0e-5f)) {
        std::cerr << "rough backface generalized-glass regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (!finite(actual[33u]) || !rgb_equal(actual[33u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(actual[33u].w, 0.0f)) {
        std::cerr << "unit-IOR generalized-glass normalization failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    // Cycles samples both Beckmann and GGX from the visible-normal
    // distribution. At this symmetric oblique reflection H=N, so the Cycles
    // Beckmann result is f=0.094655011 and pdf=0.111128319. The former NDF PDF
    // omitted 1/(1+Lambda(I)) and produced 0.134543656 instead.
    if (!finite(actual[34u]) ||
        !rgb_equal(actual[34u], {0.094655011f, 0.094655011f, 0.094655011f},
                   8.0e-5f) ||
        !approximately_equal(actual[34u].w, 0.111128319f, 8.0e-5f)) {
        std::cerr << "Beckmann visible-normal PDF regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
