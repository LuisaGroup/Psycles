#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include "../src/luisa/shader_table_data.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <bit>
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
using namespace psycles::luisa_backend::detail;
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
        Expr<luisa::ulong> id,
        const SurfacePoint &) const noexcept override {
        const auto found = Bool{_found} &
                           (id == static_cast<luisa::ulong>(::attribute));
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

    [[nodiscard]] ULong
    parameter_uint64(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot)
            .xy()
            .bitcast<luisa::ulong>();
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

    [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>, Expr<std::uint32_t>,
                                     Expr<luisa::float3>, Expr<float>,
                                     Expr<float>, Expr<float>,
                                     Expr<float>) const noexcept override {
      return make_float3(0.0f);
    }
};

class TableShaderServices final : public ShaderServices {

private:
    const BufferFloat &_scalars;
    const BufferFloat3 &_vectors;

public:
    TableShaderServices(
        const BufferFloat &scalars,
        const BufferFloat3 &vectors) noexcept
        : _scalars{scalars}, _vectors{vectors} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>, Expr<luisa::float2>,
        Expr<luisa::float2>, Expr<luisa::float2>,
        std::uint32_t, std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<luisa::ulong>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _scalars.read(block + slot);
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vectors.read(block + slot);
    }

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _vectors.read(block + slot)
            .xy()
            .bitcast<luisa::ulong>();
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

    [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>, Expr<std::uint32_t>,
                                     Expr<luisa::float3>, Expr<float>,
                                     Expr<float>, Expr<float>,
                                     Expr<float>) const noexcept override {
      return make_float3(0.0f);
    }
};

struct PackedTableParameters {
    luisa::vector<float> scalars;
    luisa::vector<luisa::float3> vectors;
};

[[nodiscard]] PackedTableParameters pack_table_parameters(
    const SurfaceProgram &program) {
    PackedTableParameters result;
    SurfaceParameterBlock parameters{program};
    std::vector<PendingShaderTable> tables;
    for (const auto &parameter : program.parameters()) {
        const auto *value = parameters.find(parameter.id);
        if (value == nullptr) {
            throw std::runtime_error{
                "shader-table fixture lost a parameter"};
        }
        auto scalar = 0.0f;
        auto vector = luisa::make_float3(0.0f);
        using contract::SocketType;
        switch (parameter.type) {
            case SocketType::boolean:
                scalar = std::get<bool>(value->value) ? 1.0f : 0.0f;
                break;
            case SocketType::integer:
                scalar = static_cast<float>(
                    std::get<std::int64_t>(value->value));
                break;
            case SocketType::floating:
                scalar = std::get<float>(value->value);
                break;
            case SocketType::float2: {
                const auto authored = std::get<Vec2f>(value->value);
                vector = luisa::make_float3(
                    authored.x, authored.y, 0.0f);
                break;
            }
            case SocketType::float3:
            case SocketType::color:
            case SocketType::spectrum:
            case SocketType::point:
            case SocketType::vector:
            case SocketType::normal: {
                const auto authored = std::get<Vec3f>(value->value);
                vector = luisa::make_float3(
                    authored.x, authored.y, authored.z);
                break;
            }
            case SocketType::unsigned_integer: {
                const auto authored =
                    std::get<std::uint64_t>(value->value);
                vector = luisa::make_float3(
                    std::bit_cast<float>(
                        static_cast<std::uint32_t>(authored)),
                    std::bit_cast<float>(
                        static_cast<std::uint32_t>(authored >> 32u)),
                    0.0f);
                break;
            }
            case SocketType::string: {
                auto staged = stage_shader_table(
                    program,
                    parameter,
                    *value,
                    static_cast<std::uint32_t>(result.vectors.size()));
                if (!staged.valid) {
                    throw std::runtime_error{
                        "failed to stage shader table: " +
                        staged.diagnostic};
                }
                tables.emplace_back(std::move(staged.table));
                break;
            }
            case SocketType::transform:
            case SocketType::closure:
            case SocketType::volume_closure:
                throw std::runtime_error{
                    "shader-table fixture has an unsupported parameter"};
        }
        result.scalars.emplace_back(scalar);
        result.vectors.emplace_back(vector);
    }
    std::string diagnostic;
    if (!finalize_shader_tables(
            tables, result.scalars, result.vectors, diagnostic)) {
        throw std::runtime_error{
            "failed to finalize shader tables: " + diagnostic};
    }
    return result;
}

[[nodiscard]] ShaderGraph sampled_color_ramp_graph() {
    ShaderGraph graph;
    const auto ramp = graph.add_node(
        node_type::color_ramp, "Sampled ColorRamp oracle");
    const auto emission = graph.add_node(
        node_type::emission, "ColorRamp emission");
    if (!graph.set_input(
            ramp, "Factor", SocketValue::floating(0.25f)) ||
        !graph.set_property(
            ramp, "Sampled", SocketValue::boolean(true)) ||
        !graph.set_property(
            ramp,
            "Table",
            SocketValue::string(
                "0,0.1,0.2,0.3,0.4;"
                "0.5,0.5,0.6,0.7,0.8;"
                "1,0.9,1,0.1,0.2")) ||
        !graph.set_input(
            emission, "Strength", SocketValue::floating(1.0f)) ||
        !graph.connect(
            {.node = ramp, .socket = "Color"},
            emission,
            "Color")) {
        throw std::runtime_error{
            "failed to build sampled ColorRamp oracle"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph sampled_rgb_curve_graph() {
    ShaderGraph graph;
    const auto curve = graph.add_node(
        node_type::rgb_curve, "Sampled RGB Curve oracle");
    const auto emission = graph.add_node(
        node_type::emission, "RGB Curve emission");
    if (!graph.set_input(
            curve, "Factor", SocketValue::floating(1.0f)) ||
        !graph.set_input(
            curve,
            "Color",
            SocketValue::color({0.25f, 0.5f, 0.75f})) ||
        !graph.set_property(
            curve, "Sampled", SocketValue::boolean(true)) ||
        !graph.set_property(
            curve,
            "Table",
            SocketValue::string(
                "0,0,0.1,0.2;"
                "0.5,0.5,0.6,0.7;"
                "1,1,0.9,0.8")) ||
        !graph.set_input(
            emission, "Strength", SocketValue::floating(1.0f)) ||
        !graph.connect(
            {.node = curve, .socket = "Color"},
            emission,
            "Color")) {
        throw std::runtime_error{
            "failed to build sampled RGB Curve oracle"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] std::shared_ptr<const SurfaceProgram> compile_table_program(
    ShaderCompiler &compiler,
    const ShaderGraph &graph) {
    const auto shader = compiler.compile(graph);
    if (!shader.ok()) {
        throw std::runtime_error{
            "shader-table graph failed validation"};
    }
    auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok()) {
        throw std::runtime_error{
            "shader-table graph failed lowering"};
    }
    return std::move(lowered.program);
}

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
    const auto &values = lowered.program->value_instructions();
    const auto &parameters = lowered.program->parameters();
    auto found = false;
    for (const auto &instruction : values) {
        if (instruction.operation != expected_operation ||
            instruction.result_type != expected_type) {
            continue;
        }
        const auto id = instruction.operand(
            value_operand::attribute::id);
        if (!id.valid() || id.value >= values.size()) {
            continue;
        }
        const auto &id_value = values[id.value];
        if (id_value.operation != ValueOperation::parameter ||
            id_value.result_type != SocketType::unsigned_integer ||
            !id_value.parameter.valid() ||
            id_value.parameter.value >= parameters.size()) {
            continue;
        }
        const auto &parameter = parameters[id_value.parameter.value];
        found = parameter.source == ParameterSource::property &&
                parameter.socket == "AttributeId" &&
                parameter.type == SocketType::unsigned_integer &&
                std::get<std::uint64_t>(
                    parameter.default_value.value) == attribute;
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

    const std::array table_programs{
        compile_table_program(compiler, sampled_color_ramp_graph()),
        compile_table_program(compiler, sampled_rgb_curve_graph())};
    SurfaceDispatch table_surfaces;
    const std::array table_tags{
        table_surfaces.create<GraphSurface>(table_programs[0u]),
        table_surfaces.create<GraphSurface>(table_programs[1u])};
    const std::array table_parameters{
        pack_table_parameters(*table_programs[0u]),
        pack_table_parameters(*table_programs[1u])};

    Kernel1D evaluate_tables = [&](BufferFloat ramp_scalars,
                                    BufferFloat3 ramp_vectors,
                                    BufferFloat curve_scalars,
                                    BufferFloat3 curve_vectors,
                                    BufferFloat4 output) noexcept {
        TableShaderServices ramp_services{
            ramp_scalars, ramp_vectors};
        TableShaderServices curve_services{
            curve_scalars, curve_vectors};
        const auto point = make_surface_point();
        output.write(
            0u,
            make_float4(
                table_surfaces.emission(
                    table_tags[0u],
                    ramp_services,
                    point,
                    make_float3(0.0f, 0.0f, 1.0f),
                    true),
                1.0f));
        output.write(
            1u,
            make_float4(
                table_surfaces.emission(
                    table_tags[1u],
                    curve_services,
                    point,
                    make_float3(0.0f, 0.0f, 1.0f),
                    true),
                1.0f));
    };
    auto ramp_scalar_buffer = device.create_buffer<float>(
        table_parameters[0u].scalars.size());
    auto ramp_vector_buffer = device.create_buffer<luisa::float3>(
        table_parameters[0u].vectors.size());
    auto curve_scalar_buffer = device.create_buffer<float>(
        table_parameters[1u].scalars.size());
    auto curve_vector_buffer = device.create_buffer<luisa::float3>(
        table_parameters[1u].vectors.size());
    auto table_result_buffer =
        device.create_buffer<luisa::float4>(2u);
    auto table_kernel = device.compile(
        evaluate_tables,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    std::array<luisa::float4, 2u> table_actual{};
    stream << ramp_scalar_buffer.copy_from(
                  luisa::span{table_parameters[0u].scalars})
           << ramp_vector_buffer.copy_from(
                  luisa::span{table_parameters[0u].vectors})
           << curve_scalar_buffer.copy_from(
                  luisa::span{table_parameters[1u].scalars})
           << curve_vector_buffer.copy_from(
                  luisa::span{table_parameters[1u].vectors})
           << table_kernel(
                  ramp_scalar_buffer,
                  ramp_vector_buffer,
                  curve_scalar_buffer,
                  curve_vector_buffer,
                  table_result_buffer)
                  .dispatch(1u)
           << table_result_buffer.copy_to(
                  luisa::span{table_actual})
           << synchronize();
    constexpr std::array table_expected{
        luisa::float4{0.3f, 0.4f, 0.5f, 1.0f},
        luisa::float4{0.25f, 0.6f, 0.75f, 1.0f}};
    for (std::size_t index = 0u;
         index < table_actual.size();
         ++index) {
        if (!approximately_equal(
                table_actual[index], table_expected[index])) {
            std::cerr
                << "Runtime shader table " << index
                << " mismatch on " << backend << ": got {"
                << table_actual[index].x << ", "
                << table_actual[index].y << ", "
                << table_actual[index].z << ", "
                << table_actual[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
