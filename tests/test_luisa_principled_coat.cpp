#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;

namespace coat_record {
constexpr std::uint32_t first_closure_meta = 0u;
constexpr std::uint32_t first_closure_value = 1u;
constexpr std::uint32_t second_closure_meta = 2u;
constexpr std::uint32_t second_closure_value = 3u;
constexpr std::uint32_t no_reflective_caustics_meta = 4u;
constexpr std::uint32_t no_reflective_caustics_value = 5u;
constexpr std::uint32_t evaluation = 6u;
constexpr std::uint32_t evaluation_diffuse = 7u;
constexpr std::uint32_t evaluation_glossy = 8u;
constexpr std::uint32_t sample = 9u;
constexpr std::uint32_t sample_diffuse = 10u;
constexpr std::uint32_t sample_glossy = 11u;
constexpr std::uint32_t sample_direction = 12u;
constexpr std::uint32_t sample_material = 13u;
constexpr std::uint32_t stride = 14u;
} // namespace coat_record

namespace coat_case {
constexpr std::uint32_t below_cutoff = 0u;
constexpr std::uint32_t above_cutoff = 1u;
constexpr std::uint32_t ior_one = 2u;
constexpr std::uint32_t singular = 3u;
constexpr std::uint32_t narrow = 4u;
constexpr std::uint32_t wide = 5u;
constexpr std::uint32_t overdriven = 6u;
constexpr std::uint32_t tinted = 7u;
constexpr std::uint32_t coat_disabled = 8u;
constexpr std::uint32_t count = 9u;
} // namespace coat_case

namespace aggregate_record {
constexpr std::uint32_t evaluation = 0u;
constexpr std::uint32_t evaluation_diffuse = 1u;
constexpr std::uint32_t evaluation_glossy = 2u;
constexpr std::uint32_t sample = 3u;
constexpr std::uint32_t sample_diffuse = 4u;
constexpr std::uint32_t sample_glossy = 5u;
constexpr std::uint32_t sample_direction = 6u;
constexpr std::uint32_t stride = 7u;
} // namespace aggregate_record

namespace aggregate_case {
constexpr std::uint32_t glass_base = 0u;
constexpr std::uint32_t transparent_base = aggregate_record::stride;
constexpr std::uint32_t count = 2u;
} // namespace aggregate_case

namespace coat_normal_case {
constexpr std::uint32_t coat = 0u;
constexpr std::uint32_t main = 1u;
constexpr std::uint32_t count = 2u;
} // namespace coat_normal_case

namespace coat_trace_variant {
constexpr std::uint32_t second_closure = 1u;
constexpr std::uint32_t without_reflective_caustics = 2u;
constexpr std::uint32_t record_stride = 2u;
constexpr std::uint32_t count = 3u;
} // namespace coat_trace_variant

struct CoatCase {
    float weight;
    float roughness;
    float ior;
    luisa::float3 tint;
    float base_ior;
};

constexpr std::array coat_cases{
    CoatCase{1.0e-5f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{2.0e-5f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.2f, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.0f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.01f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.02f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{1.3f, 0.31f, 1.45f, {0.45f, 0.8f, 0.2f}, 1.0f},
    CoatCase{0.7f, 0.2f, 1.5f, {0.2f, 0.55f, 0.9f}, 1.0f},
    CoatCase{0.0f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.5f}};

[[nodiscard]] ShaderGraph
make_principled_coat_graph(bool link_main_normal = false) {
    ShaderGraph graph;
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Principled Coat regression");
    auto configured =
        graph.set_input(principled, "BaseColor",
            SocketValue::color({0.38f, 0.16f, 0.07f})) &&
        graph.set_input(principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "Roughness", SocketValue::floating(0.4f)) &&
        graph.set_input(principled, "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "IOR", SocketValue::floating(1.0f)) &&
        graph.set_input(principled, "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled, "SpecularTint",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(principled, "CoatWeight", SocketValue::floating(0.7f)) &&
        graph.set_input(principled, "CoatRoughness",
            SocketValue::floating(0.2f)) &&
        graph.set_input(principled, "CoatIOR", SocketValue::floating(1.5f)) &&
        graph.set_input(principled, "CoatTint",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled, "SubsurfaceWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_property(principled, "Distribution",
            SocketValue::string("GGX"));
    if (link_main_normal) {
        const auto normal =
            graph.add_node(node_type::vector_to_normal,
                "Linked main normal, unlinked Coat Normal");
        configured = configured &&
                     graph.set_input(normal, "Vector",
                         SocketValue::vector({0.6f, 0.0f, 0.8f})) &&
                     graph.connect({.node = normal, .socket = "Normal"}, principled,
                         "Normal");
    }
    if (!configured) {
        throw std::runtime_error{
            "failed to configure Principled Coat regression graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_glass_diffuse_graph() {
    ShaderGraph graph;
    const auto glass = graph.add_node(node_type::glass_bsdf, "Delta glass");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Competing diffuse");
    const auto add = graph.add_node(node_type::add_closure, "Glass plus diffuse");
    const auto configured =
        graph.set_input(glass, "Color",
            SocketValue::color({0.72f, 0.86f, 0.94f})) &&
        graph.set_input(glass, "Roughness", SocketValue::floating(0.0f)) &&
        graph.set_input(glass, "IOR", SocketValue::floating(1.45f)) &&
        graph.set_property(glass, "Distribution", SocketValue::string("GGX")) &&
        graph.set_input(diffuse, "Color",
            SocketValue::color({0.31f, 0.19f, 0.11f})) &&
        graph.set_input(diffuse, "Roughness", SocketValue::floating(0.0f)) &&
        graph.connect({.node = glass, .socket = "Closure"}, add, "A") &&
        graph.connect({.node = diffuse, .socket = "Closure"}, add, "B");
    if (!configured) {
        throw std::runtime_error{"failed to configure delta-glass MIS graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = add, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transparent_translucent_graph() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(node_type::transparent_bsdf, "Delta transparent");
    const auto translucent =
        graph.add_node(node_type::translucent_bsdf, "Competing translucent");
    const auto add =
        graph.add_node(node_type::add_closure, "Transparent plus translucent");
    const auto configured =
        graph.set_input(transparent, "Color",
            SocketValue::color({0.42f, 0.36f, 0.24f})) &&
        graph.set_input(translucent, "Color",
            SocketValue::color({0.27f, 0.51f, 0.79f})) &&
        graph.connect({.node = transparent, .socket = "Closure"}, add, "A") &&
        graph.connect({.node = translucent, .socket = "Closure"}, add, "B");
    if (!configured) {
        throw std::runtime_error{"failed to configure transparent MIS graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = add, .socket = "Closure"});
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
    throw std::runtime_error{"missing surface parameter: " + std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4>
coat_parameter_data(const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    const auto defaults = parameter_data(program);
    result.reserve(defaults.size() * coat_cases.size());
    for (const auto &test_case : coat_cases) {
        auto values = defaults;
        set_parameter(values, program, "CoatWeight",
            {test_case.weight, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "CoatRoughness",
            {test_case.roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "CoatIOR",
            {test_case.ior, 0.0f, 0.0f, 0.0f});
        set_parameter(values, program, "CoatTint",
            {test_case.tint.x, test_case.tint.y, test_case.tint.z, 0.0f});
        set_parameter(values, program, "IOR",
            {test_case.base_ior, 0.0f, 0.0f, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

[[nodiscard]] bool rgb_equal(luisa::float4 lhs, luisa::float4 rhs,
    float tolerance = 3.0e-6f) noexcept {
    return approximately_equal(lhs.x, rhs.x, tolerance) &&
           approximately_equal(lhs.y, rhs.y, tolerance) &&
           approximately_equal(lhs.z, rhs.z, tolerance);
}

[[nodiscard]] bool rgb_positive(luisa::float4 value) noexcept {
    return value.x > 0.0f && value.y > 0.0f && value.z > 0.0f;
}

[[nodiscard]] bool finite_rgb(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};

    auto coat_shader = compiler.compile(make_principled_coat_graph());
    auto coat_normal_shader = compiler.compile(make_principled_coat_graph(true));
    auto glass_shader = compiler.compile(make_glass_diffuse_graph());
    auto transparent_shader =
        compiler.compile(make_transparent_translucent_graph());
    if (!coat_shader.ok() || !coat_normal_shader.ok() || !glass_shader.ok() ||
        !transparent_shader.ok()) {
        std::cerr << "failed to compile Coat/MIS regression graphs\n";
        return EXIT_FAILURE;
    }
    auto coat_program = compile_surface_program(*coat_shader.program);
    auto coat_normal_program =
        compile_surface_program(*coat_normal_shader.program);
    auto glass_program = compile_surface_program(*glass_shader.program);
    auto transparent_program =
        compile_surface_program(*transparent_shader.program);
    if (!coat_program.ok() || !coat_normal_program.ok() || !glass_program.ok() ||
        !transparent_program.ok()) {
        std::cerr << "failed to lower Coat/MIS surface programs\n";
        return EXIT_FAILURE;
    }
    const auto &coat_normal_closures =
        coat_normal_program.program->closure_instructions();
    if (coat_normal_closures.size() != 1u ||
        coat_normal_closures.front().coat_normal_linked) {
        std::cerr << "independent unlinked Coat Normal topology was not retained\n";
        return EXIT_FAILURE;
    }

    const auto coat_parameters = coat_parameter_data(*coat_program.program);
    const auto coat_normal_parameters =
        parameter_data(*coat_normal_program.program);
    const auto glass_parameters = parameter_data(*glass_program.program);
    const auto transparent_parameters =
        parameter_data(*transparent_program.program);
    const auto coat_parameter_stride =
        static_cast<std::uint32_t>(coat_program.program->parameters().size());

    SurfaceDispatch coat_surfaces;
    const auto coat_tag =
        coat_surfaces.create<GraphSurface>(coat_program.program);
    SurfaceDispatch coat_normal_surfaces;
    const auto coat_normal_tag =
        coat_normal_surfaces.create<GraphSurface>(coat_normal_program.program);
    SurfaceDispatch glass_surfaces;
    const auto glass_tag =
        glass_surfaces.create<GraphSurface>(glass_program.program);
    SurfaceDispatch transparent_surfaces;
    const auto transparent_tag =
        transparent_surfaces.create<GraphSurface>(transparent_program.program);

    Kernel1D trace_coat = [&](BufferFloat4 parameters,
                              BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto dispatch_index = dispatch_x();
        const auto case_index = dispatch_index / coat_trace_variant::count;
        const auto trace_variant = dispatch_index % coat_trace_variant::count;
        auto point = make_surface_point();
        point.parameter_block = case_index * coat_parameter_stride;
        const auto value = coat_surfaces.closure_trace(
            UInt{coat_tag}, services, point,
            select(0u, 1u, trace_variant == coat_trace_variant::second_closure),
            trace_variant != coat_trace_variant::without_reflective_caustics);
        const auto record = case_index * coat_record::stride +
                            trace_variant * coat_trace_variant::record_stride;
        output.write(record,
            make_float4(cast<float>(value.count), cast<float>(value.type),
                value.sample_weight,
                select(0.0f, 1.0f, value.valid)));
        output.write(record + 1u,
            make_float4(value.weight,
                select(0.0f, cast<float>(value.runtime_flags),
                    trace_variant !=
                        coat_trace_variant::second_closure)));
    };

    Kernel1D evaluate_coat = [&](BufferFloat4 parameters,
                                 BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = case_index * coat_parameter_stride;
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto evaluation = coat_surfaces.evaluate(
            UInt{coat_tag}, services, point, make_float3(0.0f, 0.0f, 1.0f), query);
        const auto base = case_index * coat_record::stride;
        output.write(base + coat_record::evaluation,
            make_float4(evaluation.f, evaluation.pdf));
        output.write(base + coat_record::evaluation_diffuse,
            make_float4(evaluation.diffuse_f, evaluation.diffuse_pdf));
        output.write(
            base + coat_record::evaluation_glossy,
            make_float4(evaluation.glossy_f, cast<float>(evaluation.events)));
    };

    Kernel1D sample_coat = [&](BufferFloat4 parameters,
                               BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto case_index = dispatch_x();
        auto point = make_surface_point();
        point.parameter_block = case_index * coat_parameter_stride;
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto sample =
            coat_surfaces.sample_trace(UInt{coat_tag}, services, point, 0.0f,
                make_float2(0.37f, 0.73f), query);
        const auto base = case_index * coat_record::stride;
        output.write(
            base + coat_record::sample,
            make_float4(sample.sample.evaluation.f, sample.sample.evaluation.pdf));
        output.write(base + coat_record::sample_diffuse,
            make_float4(sample.sample.evaluation.diffuse_f,
                sample.sample.evaluation.diffuse_pdf));
        output.write(base + coat_record::sample_glossy,
            make_float4(sample.sample.evaluation.glossy_f,
                cast<float>(sample.sample.evaluation.events)));
        output.write(
            base + coat_record::sample_direction,
            make_float4(sample.sample.wi, select(0.0f, 1.0f, sample.sample.valid)));
        output.write(base + coat_record::sample_material,
            make_float4(sample.sample.roughness, sample.sample.eta,
                sample.closure_sample_weight));
    };

    Kernel1D evaluate_glass = [&](BufferFloat4 parameters,
                                  BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto point = make_surface_point();
        const auto value = glass_surfaces.evaluate(
            UInt{glass_tag}, services, point, make_float3(0.0f, 0.0f, 1.0f), query);
        output.write(aggregate_record::evaluation, make_float4(value.f, value.pdf));
        output.write(aggregate_record::evaluation_diffuse,
            make_float4(value.diffuse_f, value.diffuse_pdf));
        output.write(aggregate_record::evaluation_glossy,
            make_float4(value.glossy_f, cast<float>(value.events)));
    };

    Kernel1D sample_glass = [&](BufferFloat4 parameters,
                                BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto point = make_surface_point();
        const auto value =
            glass_surfaces.sample_trace(UInt{glass_tag}, services, point, 0.0f,
                make_float2(0.29f, 0.61f), query);
        output.write(
            aggregate_record::sample,
            make_float4(value.sample.evaluation.f, value.sample.evaluation.pdf));
        output.write(aggregate_record::sample_diffuse,
            make_float4(value.sample.evaluation.diffuse_f,
                value.sample.evaluation.diffuse_pdf));
        output.write(aggregate_record::sample_glossy,
            make_float4(value.sample.evaluation.glossy_f,
                cast<float>(value.sample.evaluation.events)));
        output.write(
            aggregate_record::sample_direction,
            make_float4(value.sample.wi, select(0.0f, 1.0f, value.sample.valid)));
    };

    Kernel1D evaluate_transparent = [&](BufferFloat4 parameters,
                                        BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto point = make_surface_point();
        const auto value =
            transparent_surfaces.evaluate(UInt{transparent_tag}, services, point,
                make_float3(0.0f, 0.0f, -1.0f), query);
        constexpr auto base = aggregate_case::transparent_base;
        output.write(base + aggregate_record::evaluation,
            make_float4(value.f, value.pdf));
        output.write(base + aggregate_record::evaluation_diffuse,
            make_float4(value.diffuse_f, value.diffuse_pdf));
        output.write(base + aggregate_record::evaluation_glossy,
            make_float4(value.glossy_f, cast<float>(value.events)));
    };

    Kernel1D sample_transparent = [&](BufferFloat4 parameters,
                                      BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true};
        const auto point = make_surface_point();
        const auto value = transparent_surfaces.sample_trace(
            UInt{transparent_tag}, services, point, 0.0f, make_float2(0.29f, 0.61f),
            query);
        constexpr auto base = aggregate_case::transparent_base;
        output.write(
            base + aggregate_record::sample,
            make_float4(value.sample.evaluation.f, value.sample.evaluation.pdf));
        output.write(base + aggregate_record::sample_diffuse,
            make_float4(value.sample.evaluation.diffuse_f,
                value.sample.evaluation.diffuse_pdf));
        output.write(base + aggregate_record::sample_glossy,
            make_float4(value.sample.evaluation.glossy_f,
                cast<float>(value.sample.evaluation.events)));
        output.write(
            base + aggregate_record::sample_direction,
            make_float4(value.sample.wi, select(0.0f, 1.0f, value.sample.valid)));
    };

    Kernel1D evaluate_coat_normal_topology = [&](BufferFloat4 parameters,
                                                 BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        const auto point = make_surface_point();
        const auto requested = dispatch_x();
        const auto value = coat_normal_surfaces.closure_trace(
            UInt{coat_normal_tag}, services, point, requested, true);
        output.write(requested, make_float4(value.normal, 0.0f));
    };

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir("principled_coat_trace", trace_coat, 35000u);
        bounded &=
            require_bounded_xir("principled_coat_evaluate", evaluate_coat, 205000u);
        bounded &=
            require_bounded_xir("principled_coat_sample", sample_coat, 390000u);
        bounded &= require_bounded_xir("principled_coat_glass_evaluate",
            evaluate_glass, 30000u);
        bounded &= require_bounded_xir("principled_coat_glass_sample", sample_glass,
            55000u);
        bounded &= require_bounded_xir("principled_coat_transparent_evaluate",
            evaluate_transparent, 30000u);
        bounded &= require_bounded_xir("principled_coat_transparent_sample",
            sample_transparent, 55000u);
        bounded &= require_bounded_xir("principled_coat_normal_topology",
            evaluate_coat_normal_topology, 35000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto coat_parameter_buffer =
        device.create_buffer<luisa::float4>(coat_parameters.size());
    auto coat_normal_parameter_buffer =
        device.create_buffer<luisa::float4>(coat_normal_parameters.size());
    auto glass_parameter_buffer =
        device.create_buffer<luisa::float4>(glass_parameters.size());
    auto transparent_parameter_buffer =
        device.create_buffer<luisa::float4>(transparent_parameters.size());
    auto coat_output = device.create_buffer<luisa::float4>(coat_record::stride *
                                                           coat_case::count);
    auto aggregate_output = device.create_buffer<luisa::float4>(
        aggregate_record::stride * aggregate_case::count);
    auto coat_normal_output =
        device.create_buffer<luisa::float4>(coat_normal_case::count);
    auto coat_trace_kernel =
        compile_named_kernel(device, "principled_coat_trace", trace_coat);
    auto coat_evaluate_kernel =
        compile_named_kernel(device, "principled_coat_evaluate", evaluate_coat);
    auto coat_sample_kernel =
        compile_named_kernel(device, "principled_coat_sample", sample_coat);
    auto glass_evaluate_kernel = compile_named_kernel(
        device, "principled_coat_glass_evaluate", evaluate_glass);
    auto glass_sample_kernel = compile_named_kernel(
        device, "principled_coat_glass_sample", sample_glass);
    auto transparent_evaluate_kernel = compile_named_kernel(
        device, "principled_coat_transparent_evaluate", evaluate_transparent);
    auto transparent_sample_kernel = compile_named_kernel(
        device, "principled_coat_transparent_sample", sample_transparent);
    auto coat_normal_kernel = compile_named_kernel(
        device, "principled_coat_normal_topology", evaluate_coat_normal_topology);
    std::array<luisa::float4, coat_record::stride * coat_case::count>
        coat_actual{};
    std::array<luisa::float4, aggregate_record::stride * aggregate_case::count>
        aggregate_actual{};
    std::array<luisa::float4, coat_normal_case::count> coat_normal_actual{};
    stream << coat_parameter_buffer.copy_from(luisa::span{coat_parameters})
           << glass_parameter_buffer.copy_from(luisa::span{glass_parameters})
           << coat_normal_parameter_buffer.copy_from(
                  luisa::span{coat_normal_parameters})
           << transparent_parameter_buffer.copy_from(
                  luisa::span{transparent_parameters})
           << coat_trace_kernel(coat_parameter_buffer, coat_output)
                  .dispatch(coat_case::count * coat_trace_variant::count)
           << coat_evaluate_kernel(coat_parameter_buffer, coat_output)
                  .dispatch(coat_case::count)
           << coat_sample_kernel(coat_parameter_buffer, coat_output)
                  .dispatch(coat_case::count)
           << coat_output.copy_to(luisa::span{coat_actual})
           << glass_evaluate_kernel(glass_parameter_buffer, aggregate_output)
                  .dispatch(1u)
           << glass_sample_kernel(glass_parameter_buffer, aggregate_output)
                  .dispatch(1u)
           << transparent_evaluate_kernel(transparent_parameter_buffer,
                  aggregate_output)
                  .dispatch(1u)
           << transparent_sample_kernel(transparent_parameter_buffer,
                  aggregate_output)
                  .dispatch(1u)
           << aggregate_output.copy_to(luisa::span{aggregate_actual})
           << coat_normal_kernel(coat_normal_parameter_buffer, coat_normal_output)
                  .dispatch(coat_normal_case::count)
           << coat_normal_output.copy_to(luisa::span{coat_normal_actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_COAT_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < coat_actual.size(); ++index) {
            const auto value = coat_actual[index];
            std::cerr << "coat " << index << ": {" << value.x << ", " << value.y
                      << ", " << value.z << ", " << value.w << "}\n";
        }
        for (std::size_t index = 0u; index < aggregate_actual.size(); ++index) {
            const auto value = aggregate_actual[index];
            std::cerr << "aggregate " << index << ": {" << value.x << ", " << value.y
                      << ", " << value.z << ", " << value.w << "}\n";
        }
    }

    const auto coat =
        [&](std::uint32_t case_index,
            std::uint32_t record) noexcept -> const luisa::float4 & {
        return coat_actual[case_index * coat_record::stride + record];
    };
    const auto closure_meta_matches =
        [&](std::uint32_t case_index, std::uint32_t record, float count,
            std::uint32_t type, bool valid) noexcept {
            const auto value = coat(case_index, record);
            return approximately_equal(value.x, count) &&
                   approximately_equal(value.y, static_cast<float>(type)) &&
                   approximately_equal(value.w, valid ? 1.0f : 0.0f);
        };

    if (!rgb_equal(coat_normal_actual[coat_normal_case::coat],
            luisa::float4{0.0f, 0.0f, 1.0f, 0.0f}) ||
        !rgb_equal(coat_normal_actual[coat_normal_case::main],
            luisa::float4{0.6f, 0.0f, 0.8f, 0.0f})) {
        std::cerr << "Cycles independent unlinked Coat Normal regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!closure_meta_matches(coat_case::below_cutoff,
            coat_record::first_closure_meta, 1.0f,
            cycles_closure::type_diffuse, true) ||
        !closure_meta_matches(coat_case::below_cutoff,
            coat_record::second_closure_meta, 1.0f,
            cycles_closure::type_none, false) ||
        !closure_meta_matches(coat_case::above_cutoff,
            coat_record::first_closure_meta, 2.0f,
            cycles_closure::type_microfacet_ggx, true) ||
        !closure_meta_matches(coat_case::above_cutoff,
            coat_record::second_closure_meta, 2.0f,
            cycles_closure::type_diffuse, true)) {
        std::cerr << "Cycles Coat cutoff/order regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    if (!closure_meta_matches(coat_case::ior_one, coat_record::first_closure_meta,
            2.0f, cycles_closure::type_microfacet_ggx, true) ||
        !approximately_equal(
            coat(coat_case::ior_one, coat_record::first_closure_meta).z, 0.0f) ||
        !closure_meta_matches(coat_case::ior_one,
            coat_record::second_closure_meta, 2.0f,
            cycles_closure::type_diffuse, true)) {
        std::cerr << "Cycles IOR-one Coat allocation regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto singular_evaluation =
        coat(coat_case::singular, coat_record::evaluation);
    const auto singular_evaluation_diffuse =
        coat(coat_case::singular, coat_record::evaluation_diffuse);
    const auto singular_sample = coat(coat_case::singular, coat_record::sample);
    const auto singular_sample_diffuse =
        coat(coat_case::singular, coat_record::sample_diffuse);
    const auto singular_sample_glossy =
        coat(coat_case::singular, coat_record::sample_glossy);
    const auto singular_direction =
        coat(coat_case::singular, coat_record::sample_direction);
    const auto singular_roughness =
        coat(coat_case::singular, coat_record::sample_material);
    constexpr auto singular_reflection_events =
        static_cast<float>(event_singular | event_reflection);
    if (!rgb_positive(singular_evaluation_diffuse) ||
        !rgb_equal(singular_sample_diffuse, singular_evaluation_diffuse) ||
        !rgb_positive(singular_sample_glossy) ||
        !approximately_equal(singular_sample.x, singular_sample_diffuse.x +
                                                    singular_sample_glossy.x) ||
        !approximately_equal(singular_sample.y, singular_sample_diffuse.y +
                                                    singular_sample_glossy.y) ||
        !approximately_equal(singular_sample.z, singular_sample_diffuse.z +
                                                    singular_sample_glossy.z) ||
        !(singular_sample.w > singular_evaluation.w &&
            singular_sample.w < 1.0e6f) ||
        !approximately_equal(singular_sample_glossy.w,
            singular_reflection_events) ||
        !approximately_equal(singular_direction,
            luisa::float4{0.0f, 0.0f, 1.0f, 1.0f}) ||
        !approximately_equal(singular_roughness.x, 0.0f)) {
        std::cerr << "Cycles singular Coat MIS regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    const auto narrow_glossy =
        coat(coat_case::narrow, coat_record::evaluation_glossy);
    const auto wider_glossy =
        coat(coat_case::wide, coat_record::evaluation_glossy);
    if (!finite_rgb(narrow_glossy) || !finite_rgb(wider_glossy) ||
        !(narrow_glossy.x > 0.0f && wider_glossy.x > 0.0f) ||
        !approximately_equal(narrow_glossy.x / wider_glossy.x, 16.0f, 2.0e-5f)) {
        std::cerr << "Cycles stable GGX peak regression failed on " << backend
                  << ": ratio " << narrow_glossy.x / wider_glossy.x << '\n';
        return EXIT_FAILURE;
    }

    const auto overdriven_diffuse =
        coat(coat_case::overdriven, coat_record::second_closure_value);
    if (!(overdriven_diffuse.x > 0.0f && overdriven_diffuse.y > 0.0f) ||
        !approximately_equal(overdriven_diffuse.z, 0.0f)) {
        std::cerr << "Cycles bsdf_alloc channel clamp regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto tinted_without_reflection_meta =
        coat(coat_case::tinted, coat_record::no_reflective_caustics_meta);
    const auto tinted_without_reflection =
        coat(coat_case::tinted, coat_record::no_reflective_caustics_value);
    const auto dielectric_without_reflection_meta =
        coat(coat_case::coat_disabled, coat_record::no_reflective_caustics_meta);
    const auto dielectric_without_reflection =
        coat(coat_case::coat_disabled, coat_record::no_reflective_caustics_value);
    if (!approximately_equal(tinted_without_reflection_meta.x, 1.0f) ||
        !approximately_equal(tinted_without_reflection_meta.y,
            static_cast<float>(cycles_closure::type_diffuse)) ||
        !(tinted_without_reflection.x < 0.38f &&
            tinted_without_reflection.y < 0.16f &&
            tinted_without_reflection.z < 0.07f) ||
        !approximately_equal(dielectric_without_reflection_meta.x, 1.0f) ||
        !approximately_equal(dielectric_without_reflection_meta.y,
            static_cast<float>(cycles_closure::type_diffuse)) ||
        !rgb_equal(dielectric_without_reflection,
            luisa::float4{0.38f, 0.16f, 0.07f, 0.0f})) {
        std::cerr << "Cycles reflective-caustics layer regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto check_reflective_delta = [&](std::uint32_t base,
                                            std::string_view label) {
        const auto evaluation =
            aggregate_actual[base + aggregate_record::evaluation];
        const auto evaluation_diffuse =
            aggregate_actual[base + aggregate_record::evaluation_diffuse];
        const auto sample = aggregate_actual[base + aggregate_record::sample];
        const auto sample_diffuse =
            aggregate_actual[base + aggregate_record::sample_diffuse];
        const auto sample_glossy =
            aggregate_actual[base + aggregate_record::sample_glossy];
        const auto direction =
            aggregate_actual[base + aggregate_record::sample_direction];
        const auto ok =
            rgb_positive(evaluation_diffuse) &&
            rgb_equal(sample_diffuse, evaluation_diffuse) &&
            rgb_positive(sample_glossy) &&
            approximately_equal(sample.x, sample_diffuse.x + sample_glossy.x) &&
            approximately_equal(sample.y, sample_diffuse.y + sample_glossy.y) &&
            approximately_equal(sample.z, sample_diffuse.z + sample_glossy.z) &&
            sample.w > evaluation.w && sample.w < 1.0e6f &&
            approximately_equal(sample_glossy.w, singular_reflection_events) &&
            approximately_equal(direction, luisa::float4{0.0f, 0.0f, 1.0f, 1.0f});
        if (!ok) {
            std::cerr << "Cycles " << label << " delta MIS regression failed on "
                      << backend << '\n';
        }
        return ok;
    };
    if (!check_reflective_delta(aggregate_case::glass_base, "glass")) {
        return EXIT_FAILURE;
    }

    constexpr auto transparent_base = aggregate_case::transparent_base;
    const auto transparent_evaluation =
        aggregate_actual[transparent_base + aggregate_record::evaluation];
    const auto transparent_evaluation_diffuse =
        aggregate_actual[transparent_base + aggregate_record::evaluation_diffuse];
    const auto transparent_sample =
        aggregate_actual[transparent_base + aggregate_record::sample];
    const auto transparent_sample_diffuse =
        aggregate_actual[transparent_base + aggregate_record::sample_diffuse];
    const auto transparent_sample_glossy =
        aggregate_actual[transparent_base + aggregate_record::sample_glossy];
    const auto transparent_direction =
        aggregate_actual[transparent_base + aggregate_record::sample_direction];
    constexpr auto transparent_events =
        static_cast<float>(event_transmission | event_transparent);
    if (!rgb_positive(transparent_evaluation_diffuse) ||
        !rgb_equal(transparent_sample_diffuse, transparent_evaluation_diffuse) ||
        !(transparent_sample.x > transparent_sample_diffuse.x &&
            transparent_sample.y > transparent_sample_diffuse.y &&
            transparent_sample.z > transparent_sample_diffuse.z) ||
        !(transparent_sample.w > transparent_evaluation.w &&
            transparent_sample.w < 1.0e6f) ||
        !approximately_equal(
            transparent_sample_glossy,
            luisa::float4{0.0f, 0.0f, 0.0f, transparent_events}) ||
        !approximately_equal(transparent_direction,
            luisa::float4{0.0f, 0.0f, -1.0f, 1.0f})) {
        std::cerr << "Cycles transparent delta MIS regression failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
