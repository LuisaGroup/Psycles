#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"
#include "luisa_shader_shape_test_support.h"

#include <array>
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
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::require_bounded_xir;
using psycles::test_support::surface_aov;

namespace closure_case {
constexpr std::uint32_t valid = 0u;
constexpr std::uint32_t invalid_lut = 1u;
constexpr std::uint32_t count = 2u;
}// namespace closure_case

namespace closure_record {
constexpr std::uint32_t metadata = 0u;
constexpr std::uint32_t weight_and_flags = 1u;
constexpr std::uint32_t normal = 2u;
constexpr std::uint32_t aov_albedo = 3u;
constexpr std::uint32_t aov_normal = 4u;
constexpr std::uint32_t stride = 5u;
}// namespace closure_record

namespace evaluation_case {
constexpr std::uint32_t regular = 0u;
constexpr std::uint32_t invalid_lut = 1u;
constexpr std::uint32_t uncorrected = 2u;
constexpr std::uint32_t rejected_leak = 3u;
constexpr std::uint32_t rejected_grazing = 4u;
constexpr std::uint32_t count = 5u;
}// namespace evaluation_case

namespace evaluation_record {
constexpr std::uint32_t combined = 0u;
constexpr std::uint32_t diffuse = 1u;
constexpr std::uint32_t glossy = 2u;
constexpr std::uint32_t stride = 3u;
}// namespace evaluation_record

namespace sample_case {
constexpr std::uint32_t regular = 0u;
constexpr std::uint32_t invalid_lut = 1u;
constexpr std::uint32_t uncorrected = 2u;
constexpr std::uint32_t grazing = 3u;
constexpr std::uint32_t invalid_geometry = 4u;
constexpr std::uint32_t count = 5u;
}// namespace sample_case

namespace sample_record {
constexpr std::uint32_t selection = 0u;
constexpr std::uint32_t validity = 1u;
constexpr std::uint32_t direction = 2u;
constexpr std::uint32_t combined = 3u;
constexpr std::uint32_t diffuse = 4u;
constexpr std::uint32_t material = 5u;
constexpr std::uint32_t stride = 6u;
}// namespace sample_record

[[nodiscard]] ShaderGraph make_principled_sheen_graph() {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::vector_to_normal, "Linked Sheen Normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf, "Isolated Principled Sheen");
    const auto configured =
        graph.set_input(normal,
            "Vector",
            SocketValue::vector({0.3f, -0.2f, 0.9f})) &&
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color({0.0f, 0.0f, 0.0f})) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.27f)) &&
        graph.set_input(
            principled, "IOR", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SheenWeight",
            SocketValue::floating(0.75f)) &&
        graph.set_input(principled,
            "SheenRoughness",
            SocketValue::floating(0.37f)) &&
        graph.set_input(principled,
            "SheenTint",
            SocketValue::color({0.8f, 0.4f, 0.2f})) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = normal, .socket = "Normal"}, principled, "Normal") &&
        graph.set_property(
            principled, "Distribution", SocketValue::string("GGX"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure isolated Principled Sheen graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] constexpr auto record_index(
    std::uint32_t case_index, std::uint32_t stride,
    std::uint32_t record) noexcept {
    return case_index * stride + record;
}

[[nodiscard]] bool check_record(std::string_view backend,
    std::string_view family,
    std::uint32_t case_index,
    std::uint32_t record,
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    if (approximately_equal(actual, expected)) {
        return true;
    }
    std::cerr << "Cycles Principled Sheen " << family << " failed on "
              << backend << " at case " << case_index << ", record "
              << record << ": got {" << actual.x << ", " << actual.y
              << ", " << actual.z << ", " << actual.w << "}\n";
    return false;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    auto shader = compiler.compile(make_principled_sheen_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile Principled Sheen graph\n";
        return EXIT_FAILURE;
    }
    auto program = compile_surface_program(*shader.program);
    if (!program.ok()) {
        std::cerr << "failed to lower Principled Sheen surface program\n";
        return EXIT_FAILURE;
    }

    // Keep this fixture in its own dispatch domain. Mixing unrelated material
    // topologies into one test dispatcher multiplies every recorded surface
    // operation by the number of fixtures and hides real code-size regressions.
    SurfaceDispatch surfaces;
    const auto surface_tag = surfaces.create<GraphSurface>(program.program);

    Kernel1D trace_closure =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            const auto case_index = dispatch_x();
            const auto cycles_value = select(0.5f,
                0.0f,
                case_index == closure_case::invalid_lut);
            ParameterShaderServices services{parameters, cycles_value};
            const auto point = make_surface_point();
            const auto closure = surfaces.closure_trace(
                UInt{surface_tag}, services, point, 0u);
            const auto aov =
                surface_aov(surfaces, UInt{surface_tag}, services, point);
            const auto base = case_index * closure_record::stride;
            output.write(base + closure_record::metadata,
                make_float4(cast<float>(closure.count),
                    cast<float>(closure.type),
                    closure.sample_weight,
                    select(0.0f, 1.0f, closure.valid)));
            output.write(base + closure_record::weight_and_flags,
                make_float4(closure.weight,
                    cast<float>(closure.runtime_flags)));
            output.write(base + closure_record::normal,
                make_float4(closure.normal, 0.0f));
            output.write(base + closure_record::aov_albedo,
                make_float4(aov.albedo, aov.roughness.x));
            output.write(base + closure_record::aov_normal,
                make_float4(aov.normal, aov.roughness.y));
        };

    Kernel1D evaluate =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            const auto case_index = dispatch_x();
            const auto cycles_value = select(0.5f,
                0.0f,
                case_index == evaluation_case::invalid_lut);
            ParameterShaderServices services{parameters, cycles_value};
            auto point = make_surface_point();
            point.use_bump_map_correction =
                (case_index != evaluation_case::uncorrected) &
                (case_index != evaluation_case::rejected_leak);
            const auto regular_direction = normalize(
                make_float3(0.25f, 0.35f, 0.9027735f));
            auto outgoing = select(regular_direction,
                normalize(make_float3(0.8f, 0.0f, -0.1f)),
                case_index == evaluation_case::rejected_leak);
            outgoing = select(outgoing,
                normalize(make_float3(1.0f, 0.0f, 5.0e-7f)),
                case_index == evaluation_case::rejected_grazing);
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto value = surfaces.evaluate(
                UInt{surface_tag}, services, point, outgoing, query);
            const auto base = case_index * evaluation_record::stride;
            output.write(base + evaluation_record::combined,
                make_float4(value.f, value.pdf));
            output.write(base + evaluation_record::diffuse,
                make_float4(value.diffuse_f, value.diffuse_pdf));
            output.write(base + evaluation_record::glossy,
                make_float4(value.glossy_f, cast<float>(value.events)));
        };

    Kernel1D sample =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            const auto case_index = dispatch_x();
            const auto cycles_value = select(0.5f,
                0.0f,
                case_index == sample_case::invalid_lut);
            ParameterShaderServices services{parameters, cycles_value};
            auto point = make_surface_point();
            point.use_bump_map_correction =
                case_index != sample_case::uncorrected;
            Float2 u_direction = make_float2(0.23f, 0.71f);
            u_direction = select(u_direction,
                make_float2(0.200493872f, 0.5f),
                case_index == sample_case::grazing);
            u_direction = select(u_direction,
                make_float2(0.200491f, 0.5f),
                case_index == sample_case::invalid_geometry);
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto value = surfaces.sample_trace(UInt{surface_tag},
                services,
                point,
                0.41f,
                u_direction,
                query);
            const auto base = case_index * sample_record::stride;
            output.write(base + sample_record::selection,
                make_float4(cast<float>(value.closure_index),
                    cast<float>(value.closure_type),
                    value.closure_sample_weight,
                    value.selection_rescaled));
            output.write(base + sample_record::validity,
                make_float4(select(0.0f, 1.0f, value.closure_valid),
                    select(0.0f, 1.0f, value.sample.valid),
                    0.0f,
                    0.0f));
            output.write(base + sample_record::direction,
                make_float4(value.sample.wi,
                    select(0.0f, 1.0f, value.sample.valid)));
            output.write(base + sample_record::combined,
                make_float4(
                    value.sample.evaluation.f, value.sample.evaluation.pdf));
            output.write(base + sample_record::diffuse,
                make_float4(value.sample.evaluation.diffuse_f,
                    value.sample.evaluation.diffuse_pdf));
            output.write(base + sample_record::material,
                make_float4(value.sample.roughness,
                    value.sample.eta,
                    cast<float>(value.sample.evaluation.events)));
        };

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "principled_sheen_closure", trace_closure, 60000u);
        bounded &= require_bounded_xir(
            "principled_sheen_evaluate", evaluate, 205000u);
        bounded &= require_bounded_xir(
            "principled_sheen_sample", sample, 390000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    const auto parameters = parameter_data(*program.program);
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    constexpr auto closure_record_count =
        closure_case::count * closure_record::stride;
    constexpr auto evaluation_record_count =
        evaluation_case::count * evaluation_record::stride;
    constexpr auto sample_record_count =
        sample_case::count * sample_record::stride;
    auto closure_output =
        device.create_buffer<luisa::float4>(closure_record_count);
    auto evaluation_output =
        device.create_buffer<luisa::float4>(evaluation_record_count);
    auto sample_output =
        device.create_buffer<luisa::float4>(sample_record_count);
    auto closure_kernel = compile_named_kernel(
        device, "principled_sheen_closure", trace_closure);
    auto evaluation_kernel = compile_named_kernel(
        device, "principled_sheen_evaluate", evaluate);
    auto sample_kernel = compile_named_kernel(
        device, "principled_sheen_sample", sample);
    std::array<luisa::float4, closure_record_count> closure_actual{};
    std::array<luisa::float4, evaluation_record_count>
        evaluation_actual{};
    std::array<luisa::float4, sample_record_count> sample_actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << closure_kernel(parameter_buffer, closure_output)
                  .dispatch(closure_case::count)
           << closure_output.copy_to(luisa::span{closure_actual})
           << evaluation_kernel(parameter_buffer, evaluation_output)
                  .dispatch(evaluation_case::count)
           << evaluation_output.copy_to(luisa::span{evaluation_actual})
           << sample_kernel(parameter_buffer, sample_output)
                  .dispatch(sample_case::count)
           << sample_output.copy_to(luisa::span{sample_actual})
           << synchronize();

    constexpr auto diffuse_reflection_events =
        static_cast<float>(event_diffuse | event_reflection);
    constexpr std::array valid_closure_expected{
        luisa::float4{1.0f,
            static_cast<float>(cycles_closure::type_sheen),
            0.175f,
            1.0f},
        luisa::float4{0.3f, 0.15f, 0.075f, 24.0f},
        luisa::float4{0.309426f, -0.206284f, 0.928279f, 0.0f},
        luisa::float4{0.3f, 0.15f, 0.075f, 1.0f},
        luisa::float4{0.309426f, -0.206284f, 0.928279f, 1.0f}};
    for (std::uint32_t record = 0u;
         record < valid_closure_expected.size();
         ++record) {
        if (!check_record(backend,
                "closure",
                closure_case::valid,
                record,
                closure_actual[record_index(closure_case::valid,
                    closure_record::stride,
                    record)],
                valid_closure_expected[record])) {
            return EXIT_FAILURE;
        }
    }
    constexpr std::array invalid_closure_expected{
        luisa::float4{1.0f,
            static_cast<float>(cycles_closure::type_none),
            0.0f,
            1.0f},
        luisa::float4{0.6f, 0.3f, 0.15f, 0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 1.0f}};
    constexpr std::array invalid_closure_records{
        closure_record::metadata,
        closure_record::weight_and_flags,
        closure_record::aov_albedo,
        closure_record::aov_normal};
    for (std::uint32_t index = 0u;
         index < invalid_closure_records.size();
         ++index) {
        const auto record = invalid_closure_records[index];
        if (!check_record(backend,
                "closure",
                closure_case::invalid_lut,
                record,
                closure_actual[record_index(closure_case::invalid_lut,
                    closure_record::stride,
                    record)],
                invalid_closure_expected[index])) {
            return EXIT_FAILURE;
        }
    }

    constexpr std::array regular_evaluation_expected{
        luisa::float4{0.0167059f,
            0.00835297f,
            0.00417648f,
            0.0557498f},
        luisa::float4{0.0167059f,
            0.00835297f,
            0.00417648f,
            0.0557498f},
        luisa::float4{0.0f,
            0.0f,
            0.0f,
            diffuse_reflection_events}};
    for (std::uint32_t record = 0u;
         record < regular_evaluation_expected.size();
         ++record) {
        if (!check_record(backend,
                "evaluation",
                evaluation_case::regular,
                record,
                evaluation_actual[record_index(evaluation_case::regular,
                    evaluation_record::stride,
                    record)],
                regular_evaluation_expected[record])) {
            return EXIT_FAILURE;
        }
    }
    constexpr std::array evaluation_case_expected{
        luisa::float4{},
        luisa::float4{0.0167249f,
            0.00836247f,
            0.00418123f,
            0.0557498f},
        luisa::float4{},
        luisa::float4{}};
    constexpr std::array evaluation_cases{
        evaluation_case::invalid_lut,
        evaluation_case::uncorrected,
        evaluation_case::rejected_leak,
        evaluation_case::rejected_grazing};
    for (std::uint32_t index = 0u; index < evaluation_cases.size();
         ++index) {
        const auto case_index = evaluation_cases[index];
        if (!check_record(backend,
                "evaluation",
                case_index,
                evaluation_record::combined,
                evaluation_actual[record_index(case_index,
                    evaluation_record::stride,
                    evaluation_record::combined)],
                evaluation_case_expected[index])) {
            return EXIT_FAILURE;
        }
    }

    constexpr std::array regular_sample_expected{
        luisa::float4{0.0f,
            static_cast<float>(cycles_closure::type_sheen),
            0.175f,
            0.41f},
        luisa::float4{0.61952f, -0.78194f, 0.069027f, 1.0f},
        luisa::float4{0.100744f,
            0.0503718f,
            0.0251859f,
            0.550437f},
        luisa::float4{0.100744f,
            0.0503718f,
            0.0251859f,
            0.550437f},
        luisa::float4{1.0f,
            1.0f,
            1.0f,
            diffuse_reflection_events}};
    constexpr std::array regular_sample_records{
        sample_record::selection,
        sample_record::direction,
        sample_record::combined,
        sample_record::diffuse,
        sample_record::material};
    for (std::uint32_t index = 0u;
         index < regular_sample_records.size();
         ++index) {
        const auto record = regular_sample_records[index];
        if (!check_record(backend,
                "sample",
                sample_case::regular,
                record,
                sample_actual[record_index(sample_case::regular,
                    sample_record::stride,
                    record)],
                regular_sample_expected[index])) {
            return EXIT_FAILURE;
        }
    }
    const auto invalid_selection = sample_actual[record_index(
        sample_case::invalid_lut,
        sample_record::stride,
        sample_record::selection)];
    const auto invalid_validity = sample_actual[record_index(
        sample_case::invalid_lut,
        sample_record::stride,
        sample_record::validity)];
    if (!check_record(backend,
            "sample",
            sample_case::invalid_lut,
            sample_record::selection,
            luisa::float4{invalid_selection.y,
                invalid_selection.z,
                invalid_validity.x,
                invalid_validity.y},
            luisa::float4{
                static_cast<float>(cycles_closure::type_none),
                0.0f,
                0.0f,
                0.0f})) {
        return EXIT_FAILURE;
    }
    constexpr std::array sample_case_records{
        sample_record::combined,
        sample_record::direction,
        sample_record::combined,
        sample_record::direction,
        sample_record::combined};
    constexpr std::array sample_cases{
        sample_case::uncorrected,
        sample_case::grazing,
        sample_case::grazing,
        sample_case::invalid_geometry,
        sample_case::invalid_geometry};
    constexpr std::array sample_case_expected{
        luisa::float4{0.165131f,
            0.0825655f,
            0.0412828f,
            0.550437f},
        luisa::float4{0.83205f, -0.5547f, 0.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.636108f},
        luisa::float4{0.83205f, -0.5547f, -2.6e-6f, 0.0f},
        luisa::float4{}};
    for (std::uint32_t index = 0u; index < sample_cases.size();
         ++index) {
        const auto case_index = sample_cases[index];
        const auto record = sample_case_records[index];
        if (!check_record(backend,
                "sample",
                case_index,
                record,
                sample_actual[record_index(
                    case_index, sample_record::stride, record)],
                sample_case_expected[index])) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
