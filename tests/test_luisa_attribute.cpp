#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
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

constexpr std::uint64_t attribute = 0x9b9c1b71d589d213ull;
constexpr luisa::float4 attribute_value{
    0.15f, 0.45f, 0.90f, 0.35f};

class AttributeShaderServices final : public ShaderServices {

private:
    const BufferFloat4 &_parameters;
    bool _found;

public:
    AttributeShaderServices(
        const BufferFloat4 &parameters,
        bool found) noexcept
        : _parameters{parameters}, _found{found} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<std::uint64_t> id,
        const SurfacePoint &) const noexcept override {
        const auto found = Bool{_found} & (id == ::attribute);
        return {
            .value = select(
                make_float4(0.0f),
                make_float4(attribute_value),
                found),
            .found = found};
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

[[nodiscard]] ShaderGraph attribute_graph(
    std::string_view output) {
    ShaderGraph graph;
    const auto source = graph.add_node(
        node_type::vertex_color, "Cycles Attribute");
    const auto emission = graph.add_node(
        node_type::emission, "Attribute oracle emission");
    static_cast<void>(graph.set_property(
        source,
        "AttributeId",
        SocketValue::unsigned_integer(attribute)));
    if (output == "Color" || output == "Vector") {
        if (output == "Vector") {
            const auto convert = graph.add_node(
                node_type::vector_to_color,
                "Cycles Vector to Color conversion");
            static_cast<void>(graph.connect(
                {.node = source, .socket = "Vector"},
                convert,
                "Vector"));
            static_cast<void>(graph.connect(
                {.node = convert, .socket = "Color"},
                emission,
                "Color"));
        } else {
            static_cast<void>(graph.connect(
                {.node = source, .socket = "Color"},
                emission,
                "Color"));
        }
        static_cast<void>(graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(1.0f)));
    } else {
        static_cast<void>(graph.set_input(
            emission,
            "Color",
            SocketValue::color({1.0f, 1.0f, 1.0f})));
        static_cast<void>(graph.connect(
            {.node = source, .socket = std::string{output}},
            emission,
            "Strength"));
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] std::shared_ptr<const SurfaceProgram>
compile_attribute_program(
    ShaderCompiler &compiler,
    std::string_view output,
    ValueOperation expected_operation,
    SocketType expected_type) {
    auto shader = compiler.compile(attribute_graph(output));
    if (!shader.ok()) {
        throw std::runtime_error{
            "failed to compile Attribute " + std::string{output}};
    }
    auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok()) {
        throw std::runtime_error{
            "failed to lower Attribute " + std::string{output}};
    }
    auto found = false;
    for (const auto &instruction :
         lowered.program->value_instructions()) {
        found |= instruction.operation == expected_operation &&
                 instruction.result_type == expected_type &&
                 instruction.static_u0 == attribute;
    }
    if (!found) {
        throw std::runtime_error{
            "Attribute " + std::string{output} +
            " lost its typed output operation"};
    }
    return std::move(lowered.program);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const std::array outputs{
        std::tuple{"Color", ValueOperation::attribute_color,
                   SocketType::color},
        std::tuple{"Vector", ValueOperation::attribute_color,
                   SocketType::vector},
        std::tuple{"Factor", ValueOperation::attribute_factor,
                   SocketType::floating},
        std::tuple{"Alpha", ValueOperation::attribute_alpha,
                   SocketType::floating}};
    std::array<std::shared_ptr<const SurfaceProgram>, 4u> programs;
    SurfaceDispatch surfaces;
    std::array<std::uint32_t, 4u> tags{};
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
        const auto &[name, operation, type] = outputs[index];
        programs[index] = compile_attribute_program(
            compiler, name, operation, type);
        tags[index] = surfaces.create<GraphSurface>(programs[index]);
    }

    std::array<std::vector<luisa::float4>, 4u> parameters;
    for (std::size_t index = 0u; index < programs.size(); ++index) {
        parameters[index] = parameter_data(*programs[index]);
    }

    Kernel1D evaluate = [&](BufferFloat4 color_parameters,
                            BufferFloat4 vector_parameters,
                            BufferFloat4 factor_parameters,
                            BufferFloat4 alpha_parameters,
                            BufferFloat4 result) noexcept {
        AttributeShaderServices color_services{
            color_parameters, true};
        AttributeShaderServices vector_services{
            vector_parameters, true};
        AttributeShaderServices factor_services{
            factor_parameters, true};
        AttributeShaderServices alpha_services{
            alpha_parameters, true};
        AttributeShaderServices missing_factor_services{
            factor_parameters, false};
        AttributeShaderServices missing_alpha_services{
            alpha_parameters, false};
        const auto point = make_surface_point();
        const auto emission = [&](std::uint32_t index,
                                  const ShaderServices &services) noexcept {
            return surfaces.emission(
                UInt{tags[index]},
                services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                true);
        };
        result.write(0u, make_float4(emission(0u, color_services), 0.0f));
        result.write(1u, make_float4(emission(1u, vector_services), 0.0f));
        result.write(2u, make_float4(emission(2u, factor_services), 0.0f));
        result.write(3u, make_float4(emission(3u, alpha_services), 0.0f));
        result.write(
            4u,
            make_float4(emission(2u, missing_factor_services), 0.0f));
        result.write(
            5u,
            make_float4(emission(3u, missing_alpha_services), 0.0f));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::array<Buffer<luisa::float4>, 4u> parameter_buffers;
    for (std::size_t index = 0u; index < parameter_buffers.size(); ++index) {
        parameter_buffers[index] =
            device.create_buffer<luisa::float4>(parameters[index].size());
    }
    auto result_buffer = device.create_buffer<luisa::float4>(6u);
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, 6u> actual{};
    for (std::size_t index = 0u; index < parameter_buffers.size(); ++index) {
        stream << parameter_buffers[index].copy_from(
            luisa::span{parameters[index]});
    }
    stream << kernel(parameter_buffers[0u],
                     parameter_buffers[1u],
                     parameter_buffers[2u],
                     parameter_buffers[3u],
                     result_buffer)
                  .dispatch(1u)
           << result_buffer.copy_to(luisa::span{actual})
           << synchronize();

    constexpr auto factor =
        (attribute_value.x + attribute_value.y + attribute_value.z) /
        3.0f;
    const std::array expected{
        luisa::float4{attribute_value.x, attribute_value.y,
                      attribute_value.z, 0.0f},
        luisa::float4{attribute_value.x, attribute_value.y,
                      attribute_value.z, 0.0f},
        luisa::float4{factor, factor, factor, 0.0f},
        luisa::float4{attribute_value.w, attribute_value.w,
                      attribute_value.w, 0.0f},
        luisa::float4{0.0f},
        luisa::float4{1.0f, 1.0f, 1.0f, 0.0f}};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        if (!approximately_equal(actual[index], expected[index])) {
            std::cerr << "Cycles Attribute output " << index
                      << " mismatch on " << backend << ": got {"
                      << actual[index].x << ", " << actual[index].y
                      << ", " << actual[index].z << ", "
                      << actual[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
