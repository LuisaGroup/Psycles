#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;

struct NormalMapCase {
    std::string_view base;
    std::string_view convention;
    psycles::Vec3f color;
    float strength;
    luisa::float3 expected;
};

[[nodiscard]] ShaderGraph make_graph(const NormalMapCase &test) {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::normal_map, "Normal Map execution contract");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Normal Map execution consumer");
    const auto configured =
        graph.set_input(
            normal,
            "Strength",
            SocketValue::floating(test.strength)) &&
        graph.set_input(
            normal,
            "Color",
            SocketValue::color(test.color)) &&
        graph.set_property(
            normal, "Space", SocketValue::string("TANGENT")) &&
        graph.set_property(
            normal,
            "Base",
            SocketValue::string(std::string{test.base})) &&
        graph.set_property(
            normal,
            "Convention",
            SocketValue::string(std::string{test.convention})) &&
        graph.set_input(
            diffuse,
            "Color",
            SocketValue::color({0.5f, 0.5f, 0.5f})) &&
        graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = normal, .socket = "Normal"},
            diffuse,
            "Normal");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure Normal Map execution graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] bool normal_equal(
    luisa::float3 actual,
    luisa::float3 expected) noexcept {
    return approximately_equal(actual.x, expected.x) &&
           approximately_equal(actual.y, expected.y) &&
           approximately_equal(actual.z, expected.z);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    constexpr auto inverse_sqrt_two = 0.7071067811865475244f;
    constexpr std::array cases{
        NormalMapCase{
            .base = "ORIGINAL",
            .convention = "OPENGL",
            .color = {1.0f, 0.5f, 0.5f},
            .strength = 1.0f,
            .expected = {1.0f, 0.0f, 0.0f}},
        NormalMapCase{
            .base = "DISPLACED",
            .convention = "OPENGL",
            .color = {1.0f, 0.5f, 0.5f},
            .strength = 1.0f,
            .expected = {0.0f, 1.0f, 0.0f}},
        NormalMapCase{
            .base = "ORIGINAL",
            .convention = "OPENGL",
            .color = {0.5f, 1.0f, 0.5f},
            .strength = 1.0f,
            .expected = {0.0f, 1.0f, 0.0f}},
        NormalMapCase{
            .base = "ORIGINAL",
            .convention = "DIRECTX",
            .color = {0.5f, 1.0f, 0.5f},
            .strength = 1.0f,
            .expected = {0.0f, -1.0f, 0.0f}},
        NormalMapCase{
            .base = "ORIGINAL",
            .convention = "OPENGL",
            .color = {1.0f, 0.5f, 0.5f},
            .strength = 0.0f,
            .expected = {0.0f, 0.0f, 1.0f}},
        NormalMapCase{
            .base = "ORIGINAL",
            .convention = "OPENGL",
            .color = {1.0f, 0.5f, 0.5f},
            .strength = 0.5f,
            .expected = {
                inverse_sqrt_two,
                0.0f,
                inverse_sqrt_two}}};

    ShaderCompiler compiler{make_core_node_registry()};
    SurfaceDispatch surfaces;
    std::vector<std::shared_ptr<const SurfaceProgram>> programs;
    std::vector<std::uint32_t> tags;
    std::vector<luisa::float4> parameters;
    std::size_t parameter_stride = 0u;
    programs.reserve(cases.size());
    tags.reserve(cases.size());
    for (const auto &test : cases) {
        const auto shader = compiler.compile(make_graph(test));
        if (!shader.ok()) {
            std::cerr << "Normal Map graph failed to compile\n";
            return EXIT_FAILURE;
        }
        const auto lowered = compile_surface_program(*shader.program);
        if (!lowered.ok()) {
            std::cerr << "Normal Map graph failed to lower\n";
            return EXIT_FAILURE;
        }
        auto values = parameter_data(*lowered.program);
        if (parameter_stride == 0u) {
            parameter_stride = values.size();
        } else if (values.size() != parameter_stride) {
            std::cerr << "Normal Map cases have incompatible parameter ABI\n";
            return EXIT_FAILURE;
        }
        parameters.insert(parameters.end(), values.begin(), values.end());
        programs.emplace_back(lowered.program);
        tags.emplace_back(
            surfaces.create<GraphSurface>(lowered.program));
    }

    Context context{argv[0]};
    auto device = context.create_device(std::string{backend});
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    auto tag_buffer = device.create_buffer<luisa::uint>(tags.size());
    auto output = device.create_buffer<luisa::float4>(cases.size());

    Kernel1D evaluate = [&](BufferFloat4 all_parameters,
                            BufferUInt surface_tags,
                            BufferFloat4 result) noexcept {
        const auto index = dispatch_x();
        ParameterShaderServices services{all_parameters};
        auto point = make_surface_point();
        // Current and original Mikk frames are intentionally orthogonal.
        // The saved smooth normal is non-unit so ORIGINAL also exercises
        // Cycles' raw attribute path rather than the DISPLACED normalization.
        point.object_tangent = make_float3(0.0f, 1.0f, 0.0f);
        point.undisplaced_object_tangent =
            make_float3(1.0f, 0.0f, 0.0f);
        point.object_shading_normal =
            make_float3(0.0f, 0.0f, 0.5f);
        point.undisplaced_object_shading_normal =
            make_float3(0.0f, 0.0f, 0.25f);
        point.parameter_block =
            index * static_cast<std::uint32_t>(parameter_stride);
        const auto closure = surfaces.closure_trace(
            surface_tags.read(index),
            services,
            point,
            0u);
        result.write(
            index,
            make_float4(
                closure.normal,
                select(0.0f, 1.0f, closure.valid)));
    };
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, cases.size()> actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << tag_buffer.copy_from(luisa::span{tags})
           << kernel(parameter_buffer, tag_buffer, output)
                  .dispatch(cases.size())
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (std::size_t index = 0u; index < cases.size(); ++index) {
        if (!normal_equal(actual[index].xyz(), cases[index].expected) ||
            !approximately_equal(actual[index].w, 1.0f)) {
            const auto value = actual[index];
            std::cerr << "Normal Map execution contract failed on "
                      << backend << " at case " << index << ": got {"
                      << value.x << ", " << value.y << ", " << value.z
                      << ", " << value.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
