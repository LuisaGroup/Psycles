#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <tuple>
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

constexpr auto scenario_count = 8u;
constexpr auto material_count = 2u;
constexpr auto invocation_count = material_count * scenario_count;
constexpr auto closure_capacity = 12u;

struct ResultLayout {
    static constexpr auto emission = 0u;
    static constexpr auto shading_normal = 1u;
    static constexpr auto closure_meta = 2u;
    static constexpr auto closure_weight = 3u;
    static constexpr auto closure_normal = 4u;
    static constexpr auto aov_albedo = 5u;
    static constexpr auto aov_glossy = 6u;
    static constexpr auto aov_transmission = 7u;
    static constexpr auto aov_normal = 8u;
    static constexpr auto aov_transparency = 9u;
    static constexpr auto aov_roughness = 10u;
    static constexpr auto regular_evaluation = 11u;
    static constexpr auto light_evaluation = 15u;
    static constexpr auto sample_evaluation = 19u;
    static constexpr auto sample_direction = 23u;
    static constexpr auto sample_identity = 24u;
    static constexpr auto sample_bssrdf_meta = 25u;
    static constexpr auto sample_bssrdf_radius = 26u;
    static constexpr auto sample_bssrdf_albedo = 27u;
    static constexpr auto sample_bssrdf_normal = 28u;
    static constexpr auto selection_meta = 29u;
    static constexpr auto selection_weight = 30u;
    static constexpr auto selection_normal = 31u;
    static constexpr auto count = 32u;
};

class FixtureShaderServices final : public ParameterShaderServices {

private:
    std::size_t *_texture_recordings;

public:
    FixtureShaderServices(
        const BufferFloat4 &parameters,
        std::size_t *texture_recordings = nullptr) noexcept
        : ParameterShaderServices{parameters},
          _texture_recordings{texture_recordings} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        if (_texture_recordings != nullptr) {
            ++*_texture_recordings;
        }
        return make_float4(0.19f, 0.47f, 0.83f, 1.0f);
    }
};

[[nodiscard]] ShaderGraph make_shared_image_principled_graph() {
    ShaderGraph graph;
    const auto image = graph.add_node(
        node_type::image_texture,
        "Shared physical and emission image");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Layered image Principled");
    const auto configured =
        graph.set_property(
            image,
            "Image",
            SocketValue::unsigned_integer(17u)) &&
        graph.set_input(
            principled,
            "Metallic",
            SocketValue::floating(0.21f)) &&
        graph.set_input(
            principled,
            "Roughness",
            SocketValue::floating(0.34f)) &&
        graph.set_input(
            principled,
            "TransmissionWeight",
            SocketValue::floating(0.26f)) &&
        graph.set_input(
            principled,
            "Alpha",
            SocketValue::floating(0.72f)) &&
        graph.set_input(
            principled,
            "SheenWeight",
            SocketValue::floating(0.18f)) &&
        graph.set_input(
            principled,
            "CoatWeight",
            SocketValue::floating(0.15f)) &&
        graph.set_input(
            principled,
            "EmissionStrength",
            SocketValue::floating(1.7f)) &&
        graph.connect(
            {.node = image, .socket = "Color"},
            principled,
            "BaseColor") &&
        graph.connect(
            {.node = image, .socket = "Color"},
            principled,
            "EmissionColor");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure shared image population graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_glass_graph() {
    ShaderGraph graph;
    const auto glass = graph.add_node(
        node_type::glass_bsdf,
        "Population Beckmann glass");
    const auto configured =
        graph.set_input(
            glass,
            "Color",
            SocketValue::color({0.73f, 0.86f, 0.94f})) &&
        graph.set_input(
            glass,
            "Roughness",
            SocketValue::floating(0.24f)) &&
        graph.set_input(
            glass,
            "IOR",
            SocketValue::floating(1.37f)) &&
        graph.set_property(
            glass,
            "Distribution",
            SocketValue::string("BECKMANN"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure population glass graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = glass, .socket = "Closure"});
    return graph;
}

struct InvocationInputs {
    Float3 outgoing;
    SurfaceQuery query;
    UInt shader_flags;
    Float u_lobe;
    Float2 u_direction;
};

[[nodiscard]] InvocationInputs invocation_inputs(UInt scenario) noexcept {
    constexpr auto all_lobes = static_cast<std::uint32_t>(
        event_diffuse | event_glossy | event_transmission |
        event_transparent);
    UInt lobe_mask = all_lobes;
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_diffuse | event_transmission),
        scenario == 1u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_glossy),
        scenario == 2u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_glossy | event_transmission),
        scenario == 3u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_transparent),
        scenario == 4u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_diffuse),
        scenario == 6u);
    const auto outgoing = select(
        normalize(make_float3(0.31f, -0.17f, 0.936f)),
        normalize(make_float3(0.02f, 0.01f, -0.99975f)),
        (scenario == 1u) | (scenario == 3u) |
            (scenario == 6u));
    UInt shader_flags = cycles_abi::shader_use_mis;
    shader_flags |= select(
        0u,
        cycles_abi::shader_exclude_diffuse,
        (scenario == 1u) | (scenario == 7u));
    shader_flags |= select(
        0u,
        cycles_abi::shader_exclude_glossy,
        (scenario == 2u) | (scenario == 4u) |
            (scenario == 7u));
    shader_flags |= select(
        0u,
        cycles_abi::shader_exclude_transmit,
        (scenario == 3u) | (scenario == 4u) |
            (scenario == 6u));
    shader_flags = select(shader_flags, 0u, scenario == 5u);
    const auto scenario_float = cast<float>(scenario);
    return {
        .outgoing = outgoing,
        .query = {
            .lobe_mask = lobe_mask,
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = select(
                0.0f, 0.08f, scenario == 7u),
            .reflective_caustics = scenario != 6u,
            .refractive_caustics = scenario != 2u},
        .shader_flags = shader_flags,
        .u_lobe = min(
            0.02f + 0.137f * scenario_float,
            0.99999994f),
        .u_direction = make_float2(
            0.07f + 0.11f * scenario_float,
            0.91f - 0.09f * scenario_float)};
}

void write_evaluation(
    const BufferFloat4 &output,
    UInt base,
    const SurfaceEvaluation &evaluation) noexcept {
    output.write(base, make_float4(evaluation.f, evaluation.pdf));
    output.write(
        base + 1u,
        make_float4(evaluation.diffuse_f, evaluation.diffuse_pdf));
    output.write(
        base + 2u,
        make_float4(
            evaluation.glossy_f,
            cast<float>(evaluation.events)));
    output.write(
        base + 3u,
        make_float4(evaluation.average_roughness_squared));
}

void write_results(
    const BufferFloat4 &output,
    UInt base,
    Float3 emission,
    Float3 shading_normal,
    UInt runtime_flags,
    const SurfaceAov &aov,
    const SurfaceClosureTrace &closure,
    const SurfaceEvaluation &regular,
    const SurfaceEvaluation &light,
    const SurfaceSampleTrace &sample_trace) noexcept {
    output.write(
        base + ResultLayout::emission,
        make_float4(emission, cast<float>(closure.count)));
    output.write(
        base + ResultLayout::shading_normal,
        make_float4(shading_normal, cast<float>(runtime_flags)));
    output.write(
        base + ResultLayout::closure_meta,
        make_float4(
            cast<float>(closure.index),
            cast<float>(closure.type),
            closure.sample_weight,
            select(0.0f, 1.0f, closure.valid)));
    output.write(
        base + ResultLayout::closure_weight,
        make_float4(closure.weight, cast<float>(closure.runtime_flags)));
    output.write(
        base + ResultLayout::closure_normal,
        make_float4(closure.normal, 0.0f));
    output.write(
        base + ResultLayout::aov_albedo,
        make_float4(aov.albedo, 0.0f));
    output.write(
        base + ResultLayout::aov_glossy,
        make_float4(aov.glossy_albedo, 0.0f));
    output.write(
        base + ResultLayout::aov_transmission,
        make_float4(aov.transmission_albedo, 0.0f));
    output.write(
        base + ResultLayout::aov_normal,
        make_float4(aov.normal, 0.0f));
    output.write(
        base + ResultLayout::aov_transparency,
        make_float4(aov.transparency, 0.0f));
    output.write(
        base + ResultLayout::aov_roughness,
        make_float4(aov.roughness, 0.0f, 0.0f));
    write_evaluation(
        output, base + ResultLayout::regular_evaluation, regular);
    write_evaluation(
        output, base + ResultLayout::light_evaluation, light);
    const auto &sample = sample_trace.sample;
    write_evaluation(
        output,
        base + ResultLayout::sample_evaluation,
        sample.evaluation);
    output.write(
        base + ResultLayout::sample_direction,
        make_float4(sample.wi, sample.eta));
    output.write(
        base + ResultLayout::sample_identity,
        make_float4(
            sample.roughness,
            cast<float>(sample.runtime_flags),
            select(0.0f, 1.0f, sample.valid)));
    output.write(
        base + ResultLayout::sample_bssrdf_meta,
        make_float4(
            cast<float>(sample.bssrdf_method),
            sample.bssrdf_ior,
            sample.bssrdf_roughness,
            sample.bssrdf_anisotropy));
    output.write(
        base + ResultLayout::sample_bssrdf_radius,
        make_float4(sample.bssrdf_radius, 0.0f));
    output.write(
        base + ResultLayout::sample_bssrdf_albedo,
        make_float4(sample.bssrdf_albedo, 0.0f));
    output.write(
        base + ResultLayout::sample_bssrdf_normal,
        make_float4(sample.bssrdf_normal, 0.0f));
    output.write(
        base + ResultLayout::selection_meta,
        make_float4(
            cast<float>(sample_trace.closure_index),
            cast<float>(sample_trace.closure_type),
            sample_trace.closure_sample_weight,
            sample_trace.selection_rescaled));
    output.write(
        base + ResultLayout::selection_weight,
        make_float4(
            sample_trace.closure_weight,
            select(0.0f, 1.0f, sample_trace.closure_valid)));
    output.write(
        base + ResultLayout::selection_normal,
        make_float4(sample_trace.closure_normal, 0.0f));
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto compile = [&](ShaderGraph graph) {
        auto shader = compiler.compile(graph);
        if (!shader.ok()) {
            throw std::runtime_error{
                "failed to compile surface population graph"};
        }
        auto program = compile_surface_program(*shader.program);
        if (!program.ok()) {
            throw std::runtime_error{
                "failed to lower surface population program"};
        }
        return program.program;
    };
    const auto principled = compile(
        make_shared_image_principled_graph());
    const auto glass = compile(make_glass_graph());

    SurfaceDispatch surfaces;
    const auto principled_tag =
        surfaces.create<GraphSurface>(principled);
    const auto glass_tag = surfaces.create<GraphSurface>(glass);
    auto parameters = parameter_data(*principled);
    const auto glass_parameter_base =
        static_cast<std::uint32_t>(parameters.size());
    const auto glass_parameters = parameter_data(*glass);
    parameters.insert(
        parameters.end(),
        glass_parameters.begin(),
        glass_parameters.end());

    Kernel1D write_legacy = [&](BufferFloat4 parameter_buffer,
                                BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        const auto material = invocation / scenario_count;
        const auto scenario = invocation % scenario_count;
        auto point = make_surface_point();
        point.parameter_block = select(
            0u, glass_parameter_base, material != 0u);
        point.back_facing = scenario == 7u;
        FixtureShaderServices services{parameter_buffer};
        const auto tag = select(
            UInt{principled_tag}, UInt{glass_tag}, material != 0u);
        const auto inputs = invocation_inputs(scenario);
        const auto preparation = surfaces.prepare(
            tag,
            services,
            point,
            {.outgoing = point.incoming,
             .glossy_filter_roughness =
                 inputs.query.glossy_filter_roughness,
             .emission_reflective_caustics =
                 inputs.query.reflective_caustics,
             .reflective_caustics =
                 inputs.query.reflective_caustics,
             .refractive_caustics =
                 inputs.query.refractive_caustics,
             .include_runtime_flags = true,
             .include_aov = true});
        const auto closure = surfaces.closure_trace(
            tag,
            services,
            point,
            scenario,
            inputs.query.reflective_caustics,
            inputs.query.refractive_caustics);
        const auto regular = surfaces.evaluate(
            tag, services, point, inputs.outgoing, inputs.query);
        const auto light = surfaces.evaluate_light(
            tag,
            services,
            point,
            inputs.outgoing,
            {.surface = inputs.query,
             .shader_flags = inputs.shader_flags});
        const auto sample = surfaces.sample_trace(
            tag,
            services,
            point,
            inputs.u_lobe,
            inputs.u_direction,
            inputs.query);
        const auto shading_normal = surfaces.shading_normal(
            tag, services, point);
        write_results(
            output,
            invocation * ResultLayout::count,
            preparation.emission,
            shading_normal,
            preparation.runtime_flags,
            preparation.aov,
            closure,
            regular,
            light,
            sample);
    };

    auto texture_recordings = std::size_t{};
    Kernel1D write_populated = [&](BufferFloat4 parameter_buffer,
                                   BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        const auto material = invocation / scenario_count;
        const auto scenario = invocation % scenario_count;
        auto point = make_surface_point();
        point.parameter_block = select(
            0u, glass_parameter_base, material != 0u);
        point.back_facing = scenario == 7u;
        FixtureShaderServices services{
            parameter_buffer, &texture_recordings};
        const auto tag = select(
            UInt{principled_tag}, UInt{glass_tag}, material != 0u);
        const auto inputs = invocation_inputs(scenario);
        SurfaceClosureSet closures{closure_capacity};
        const auto population = surfaces.populate(
            tag,
            services,
            point,
            {.emission_reflective_caustics =
                 inputs.query.reflective_caustics,
             .reflective_caustics =
                 inputs.query.reflective_caustics,
             .refractive_caustics =
                 inputs.query.refractive_caustics},
            closures);
        const SurfaceClosureEvaluator evaluator{
            point, closures, population.shading_normal};
        const auto runtime_flags = evaluator.runtime_flags(
            inputs.query.glossy_filter_roughness);
        const auto aov = evaluator.aov();
        const auto closure = evaluator.closure_trace(scenario);
        const auto regular = evaluator.evaluate(
            services, inputs.outgoing, inputs.query);
        const auto light = evaluator.evaluate_light(
            services,
            inputs.outgoing,
            {.surface = inputs.query,
             .shader_flags = inputs.shader_flags});
        const auto sample = evaluator.sample_trace(
            services,
            inputs.u_lobe,
            inputs.u_direction,
            inputs.query);
        write_results(
            output,
            invocation * ResultLayout::count,
            population.emission,
            population.shading_normal,
            runtime_flags,
            aov,
            closure,
            regular,
            light,
            sample);
    };

    // The image value is a common predecessor of the physical and emission
    // endpoints. The population schedule is their dependency-union DAG, so
    // recording the entire consumer chain must visit the texture vertex once:
    // |recorded(texture)| = 1, independent of the number of consumers.
    if (texture_recordings != 1u) {
        std::cerr
            << "surface population recorded the shared texture "
            << texture_recordings << " times instead of once\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    constexpr auto output_record_count =
        invocation_count * ResultLayout::count;
    auto legacy_buffer =
        device.create_buffer<luisa::float4>(output_record_count);
    auto populated_buffer =
        device.create_buffer<luisa::float4>(output_record_count);
    auto legacy_kernel = device.compile(write_legacy);
    auto populated_kernel = device.compile(write_populated);
    std::array<luisa::float4, output_record_count> legacy{};
    std::array<luisa::float4, output_record_count> populated{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << legacy_kernel(parameter_buffer, legacy_buffer)
                  .dispatch(invocation_count)
           << legacy_buffer.copy_to(luisa::span{legacy})
           << populated_kernel(parameter_buffer, populated_buffer)
                  .dispatch(invocation_count)
           << populated_buffer.copy_to(luisa::span{populated})
           << synchronize();

    for (auto record = std::size_t{};
         record < output_record_count;
         ++record) {
        if (!finite(populated[record]) ||
            !approximately_equal(
                populated[record], legacy[record], 1.0e-4f)) {
            const auto invocation = record / ResultLayout::count;
            const auto local_record = record % ResultLayout::count;
            const auto actual = populated[record];
            const auto expected = legacy[record];
            std::cerr
                << "surface population mismatch on " << backend
                << " at material "
                << invocation / scenario_count
                << ", scenario " << invocation % scenario_count
                << ", record " << local_record
                << ": populated {" << actual.x << ", "
                << actual.y << ", " << actual.z << ", "
                << actual.w << "}, legacy {" << expected.x
                << ", " << expected.y << ", " << expected.z
                << ", " << expected.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
