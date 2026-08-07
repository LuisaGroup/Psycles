#include "subsurface_exit_closure_component.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_operations.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::luisa_backend::detail::SubsurfaceExitClosureComponent;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr luisa::float3 first_normal{0.0f, 0.6f, 0.8f};
constexpr luisa::float3 second_normal{0.8f, 0.0f, 0.6f};
constexpr luisa::float3 fallback_normal{0.0f, -0.8f, 0.6f};
constexpr luisa::float2 direction_random{0.6814339161f, 0.7971861959f};

[[nodiscard]] ShaderGraph make_exit_normal_graph() {
    ShaderGraph graph;
    const auto first = graph.add_node(
        node_type::subsurface_scattering, "First exit BSSRDF");
    const auto second = graph.add_node(
        node_type::subsurface_scattering, "Second exit BSSRDF");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Ignored non-BSSRDF closure");
    const auto bssrdf_sum = graph.add_node(
        node_type::add_closure, "BSSRDF sum");
    const auto root = graph.add_node(
        node_type::add_closure, "BSSRDF plus diffuse");
    const auto configured =
        graph.set_input(first,
            "Color",
            SocketValue::color({0.25f, 0.25f, 0.25f})) &&
        graph.set_input(first,
            "Normal",
            SocketValue::normal({0.0f, 0.6f, 0.8f})) &&
        graph.set_input(first,
            "Scale",
            SocketValue::floating(1.0f)) &&
        graph.set_input(second,
            "Color",
            SocketValue::color({0.75f, 0.75f, 0.75f})) &&
        graph.set_input(second,
            "Normal",
            SocketValue::normal({0.8f, 0.0f, 0.6f})) &&
        graph.set_input(second,
            "Scale",
            SocketValue::floating(1.0f)) &&
        graph.set_input(diffuse,
            "Color",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(diffuse,
            "Normal",
            SocketValue::normal({-0.6f, 0.0f, 0.8f})) &&
        graph.connect(
            {.node = first, .socket = "Closure"}, bssrdf_sum, "A") &&
        graph.connect(
            {.node = second, .socket = "Closure"}, bssrdf_sum, "B") &&
        graph.connect(
            {.node = bssrdf_sum, .socket = "Closure"}, root, "A") &&
        graph.connect(
            {.node = diffuse, .socket = "Closure"}, root, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure BSSRDF exit-normal graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = root, .socket = "Closure"});
    return graph;
}

[[nodiscard]] luisa::float3 host_normalize(luisa::float3 value) noexcept {
    const auto inverse_length = 1.0f / std::sqrt(
        value.x * value.x + value.y * value.y + value.z * value.z);
    return value * inverse_length;
}

[[nodiscard]] bool equal(
    luisa::float3 actual,
    luisa::float3 expected,
    float tolerance = 5.0e-6f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    auto shader = compiler.compile(make_exit_normal_graph());
    if (!shader.ok()) {
        std::cerr << "BSSRDF exit-normal graph did not compile\n";
        return EXIT_FAILURE;
    }
    auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok()) {
        std::cerr << "BSSRDF exit-normal graph did not lower\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(lowered.program);
    const auto parameters = parameter_data(*lowered.program);
    Kernel1D probe = [&](BufferFloat4 parameter_buffer,
                         BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameter_buffer};
        auto point = make_surface_point();
        point.shading_normal = fallback_normal;
        // This regression isolates the BSSRDF reduction law. Normal-correction
        // behavior has its own coverage and would intentionally alter the
        // authored closure normals before this reduction runs.
        point.use_bump_map_correction = false;

        SurfaceBssrdfNormalVisitor visitor{8u};
        static_cast<void>(surfaces.collect_closures(
            UInt{surface_tag}, services, point, true, true, visitor));
        const auto exit_normal = Float3{visitor.result()};
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(
                event_diffuse | event_reflection),
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .subsurface_exit = true,
            .subsurface_normal = exit_normal};
        const SubsurfaceExitClosureComponent exit;
        const auto sample = exit.sample(point, direction_random, query);
        const auto trace = exit.trace(point, query, 0u);
        const auto sample_trace = exit.sample_trace(
            point, direction_random, query);
        const auto evaluation = exit.evaluate_light(
            point,
            sample.wi,
            query,
            cycles_abi::shader_use_mis);
        const auto mapped =
            cycles_sample_mapping::sample_cosine_hemisphere(
                exit_normal, direction_random);

        // A retained non-BSSRDF closure must not prevent the exact Cycles
        // zero-sum fallback from selecting ShaderData::N.
        SurfaceBssrdfNormalVisitor fallback_visitor{8u};
        fallback_visitor.begin(point.shading_normal);
        auto diffuse = SurfaceClosureRecord::zero();
        diffuse.kind = static_cast<std::uint32_t>(
            SurfaceClosureKind::diffuse);
        diffuse.weight = make_float3(1.0f);
        diffuse.allocation_weight = 1.0f;
        diffuse.normal = make_float3(1.0f, 0.0f, 0.0f);
        fallback_visitor.add(diffuse);
        fallback_visitor.finish();

        output.write(0u, make_float4(exit_normal, 0.0f));
        output.write(1u, make_float4(sample.wi, sample.evaluation.pdf));
        output.write(2u, make_float4(
            trace.normal,
            cast<float>(trace.type)));
        output.write(3u, make_float4(evaluation.f, evaluation.pdf));
        output.write(4u, make_float4(
            sample_trace.closure_normal,
            select(0.0f, 1.0f, sample_trace.closure_valid)));
        output.write(5u, make_float4(
            Float3{fallback_visitor.result()},
            select(0.0f, 1.0f, sample.valid)));
        output.write(6u, make_float4(mapped.direction, mapped.pdf));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    auto output = device.create_buffer<luisa::float4>(7u);
    auto kernel = device.compile(probe);
    std::array<luisa::float4, 7u> actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << kernel(parameter_buffer, output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    const auto expected_normal = host_normalize(
        first_normal * 0.25f + second_normal * 0.75f);
    const auto ok =
        equal(actual[0u].xyz(), expected_normal) &&
        equal(actual[1u].xyz(), actual[6u].xyz()) &&
        approximately_equal(actual[1u].w, actual[6u].w) &&
        equal(actual[2u].xyz(), expected_normal) &&
        approximately_equal(
            actual[2u].w,
            static_cast<float>(cycles_closure::type_diffuse)) &&
        equal(actual[3u].xyz(), luisa::float3{actual[6u].w}) &&
        approximately_equal(actual[3u].w, actual[6u].w) &&
        equal(actual[4u].xyz(), expected_normal) &&
        approximately_equal(actual[4u].w, 1.0f) &&
        equal(actual[5u].xyz(), fallback_normal) &&
        approximately_equal(actual[5u].w, 1.0f);
    if (!ok) {
        std::cerr << "Cycles BSSRDF exit-normal regression failed on "
                  << backend << '\n';
        for (std::size_t index = 0u; index < actual.size(); ++index) {
            const auto value = actual[index];
            std::cerr << "  " << index << ": {" << value.x << ", "
                      << value.y << ", " << value.z << ", "
                      << value.w << "}\n";
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
