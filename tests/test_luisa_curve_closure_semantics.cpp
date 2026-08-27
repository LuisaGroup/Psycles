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
#include <numbers>
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
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr luisa::float3 closure_normal{0.8f, 0.0f, 0.6f};
constexpr luisa::float3 geometric_normal{-0.8f, 0.0f, 0.6f};
constexpr luisa::float3 diffuse_color{0.62f, 0.41f, 0.23f};
constexpr float inverse_pi = std::numbers::inv_pi_v<float>;

namespace record {
constexpr std::uint32_t trace = 0u;
constexpr std::uint32_t evaluation = 1u;
constexpr std::uint32_t sample_direction = 2u;
constexpr std::uint32_t sample_evaluation = 3u;
constexpr std::uint32_t count = 4u;
}// namespace record

struct CompiledSurface {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceClosurePlan closure_plan;
    std::vector<luisa::float4> parameters;
};

[[nodiscard]] ShaderGraph make_graph(bool glossy) {
    ShaderGraph graph;
    const auto closure =
        graph.add_node(glossy ? node_type::glossy_bsdf : node_type::diffuse_bsdf,
                       glossy ? "Curve glossy" : "Curve diffuse");
    const auto color = glossy ? luisa::float3{1.0f} : diffuse_color;
    const auto configured =
        graph.set_input(closure, "Color",
                        SocketValue::color({color.x, color.y, color.z})) &&
        graph.set_input(closure, "Roughness", SocketValue::floating(0.0f)) &&
        graph.set_input(closure, "Normal",
                        SocketValue::normal({closure_normal.x, closure_normal.y,
                                             closure_normal.z}));
    if (!configured ||
        (glossy && !graph.set_property(closure, "Distribution",
                                       SocketValue::string("GGX")))) {
        throw std::runtime_error{"failed to configure curve closure graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = closure, .socket = "Closure"});
    return graph;
}

[[nodiscard]] CompiledSurface compile_graph(ShaderCompiler &compiler,
                                            bool glossy) {
    const auto shader = compiler.compile(make_graph(glossy));
    if (!shader.ok()) {
        throw std::runtime_error{"failed to compile curve closure graph"};
    }
    const auto surface = compile_surface_program(*shader.program);
    if (!surface.ok()) {
        throw std::runtime_error{"failed to lower curve closure graph"};
    }
    auto parameters = parameter_data(*surface.program);
    return {.program = surface.program,
            .closure_plan =
                merged_surface_closure_plan(*surface.program, parameters),
            .parameters = std::move(parameters)};
}

[[nodiscard]] SurfacePoint make_point(Expr<bool> is_curve) noexcept {
    auto point = make_surface_point();
    point.geometric_normal = make_float3(geometric_normal);
    point.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    point.object_shading_normal = point.shading_normal;
    point.undisplaced_shading_normal = point.shading_normal;
    point.undisplaced_object_shading_normal = point.shading_normal;
    point.incoming = make_float3(0.0f, 0.0f, 1.0f);
    point.is_curve = is_curve;
    point.use_bump_map_correction = true;
    return point;
}

[[nodiscard]] SurfaceQuery query() noexcept {
    return {.lobe_mask = static_cast<std::uint32_t>(event_diffuse | event_glossy |
                                                    event_transmission |
                                                    event_transparent),
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
}

[[nodiscard]] auto make_kernel(const SurfaceDispatch &surfaces,
                               std::uint32_t tag) {
    return Kernel1D{[surface = &surfaces, tag](BufferFloat4 parameters,
                                               BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto point = make_point(case_index != 0u);
        ParameterShaderServices services{parameters};
        const auto closure =
            surface->closure_trace(UInt{tag}, services, point, 0u, true, true);
        const auto evaluation = surface->evaluate(
            UInt{tag}, services, point, make_float3(closure_normal), query());
        // Concentric-disk center maps exactly to the closure normal for both
        // cosine and singular GGX sampling. The authored closure normal is in
        // the opposite hemisphere of the synthetic triangle Ng, so the two
        // primitive contracts are separated without a tolerance boundary.
        const auto sample = surface->sample_trace(UInt{tag}, services, point, 0.5f,
                                                  make_float2(0.5f), query());
        const auto base = case_index * record::count;
        output.write(
            base + record::trace,
            make_float4(closure.normal, select(0.0f, 1.0f, closure.valid)));
        output.write(base + record::evaluation,
                     make_float4(evaluation.f, evaluation.pdf));
        output.write(
            base + record::sample_direction,
            make_float4(sample.sample.wi, select(0.0f, 1.0f, sample.sample.valid)));
        output.write(
            base + record::sample_evaluation,
            make_float4(sample.sample.evaluation.f, sample.sample.evaluation.pdf));
    }};
}

[[nodiscard]] bool close(luisa::float4 actual, luisa::float4 expected,
                         float tolerance = 5.0e-5f) noexcept {
    return approximately_equal(actual, expected, tolerance);
}

void print_records(
    std::string_view backend, std::string_view model,
    const std::array<luisa::float4, 2u * record::count> &records) {
    std::cerr << "curve " << model << " regression failed on " << backend << '\n';
    for (auto index = std::size_t{0u}; index < records.size(); ++index) {
        const auto value = records[index];
        std::cerr << index << ": {" << value.x << ", " << value.y << ", " << value.z
                  << ", " << value.w << "}\n";
    }
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto diffuse = compile_graph(compiler, false);
    const auto glossy = compile_graph(compiler, true);
    SurfaceDispatch diffuse_surfaces;
    const auto diffuse_tag = diffuse_surfaces.create<GraphSurface>(
        diffuse.program, diffuse.closure_plan);
    SurfaceDispatch glossy_surfaces;
    const auto glossy_tag =
        glossy_surfaces.create<GraphSurface>(glossy.program, glossy.closure_plan);
    const auto diffuse_kernel = make_kernel(diffuse_surfaces, diffuse_tag);
    const auto glossy_kernel = make_kernel(glossy_surfaces, glossy_tag);

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto diffuse_parameters =
        device.create_buffer<luisa::float4>(diffuse.parameters.size());
    auto glossy_parameters =
        device.create_buffer<luisa::float4>(glossy.parameters.size());
    auto diffuse_output = device.create_buffer<luisa::float4>(2u * record::count);
    auto glossy_output = device.create_buffer<luisa::float4>(2u * record::count);
    auto diffuse_shader =
        compile_named_kernel(device, "curve diffuse semantics", diffuse_kernel);
    auto glossy_shader =
        compile_named_kernel(device, "curve glossy semantics", glossy_kernel);
    std::array<luisa::float4, 2u * record::count> diffuse_actual{};
    std::array<luisa::float4, 2u * record::count> glossy_actual{};
    stream << diffuse_parameters.copy_from(luisa::span{diffuse.parameters})
           << glossy_parameters.copy_from(luisa::span{glossy.parameters})
           << diffuse_shader(diffuse_parameters, diffuse_output).dispatch(2u)
           << glossy_shader(glossy_parameters, glossy_output).dispatch(2u)
           << diffuse_output.copy_to(luisa::span{diffuse_actual})
           << glossy_output.copy_to(luisa::span{glossy_actual}) << synchronize();

    const auto diffuse_curve = record::count;
    const auto expected_diffuse = diffuse_color * inverse_pi;
    const auto diffuse_ok =
        // Triangle Ng rejects the direction sampled about the authored N.
        close(diffuse_actual[record::sample_direction],
              {closure_normal.x, closure_normal.y, closure_normal.z, 0.0f}) &&
        // Curve Ng is the selected closure normal, and curve bump shadowing
        // is the identity for both evaluation and the selected sample.
        close(diffuse_actual[diffuse_curve + record::trace],
              {closure_normal.x, closure_normal.y, closure_normal.z, 1.0f}) &&
        close(diffuse_actual[diffuse_curve + record::evaluation],
              {expected_diffuse.x, expected_diffuse.y, expected_diffuse.z,
               inverse_pi}) &&
        close(diffuse_actual[diffuse_curve + record::sample_direction],
              {closure_normal.x, closure_normal.y, closure_normal.z, 1.0f}) &&
        close(diffuse_actual[diffuse_curve + record::sample_evaluation],
              {expected_diffuse.x, expected_diffuse.y, expected_diffuse.z,
               inverse_pi});
    if (!diffuse_ok) {
        print_records(backend, "diffuse", diffuse_actual);
        return EXIT_FAILURE;
    }

    const auto glossy_curve = record::count;
    const auto expected_reflection = luisa::float3{0.96f, 0.0f, -0.28f};
    const auto glossy_ok =
        // Triangle setup must correct this authored normal because its ideal
        // reflection lies below Ng; curve setup must preserve it exactly.
        !close(glossy_actual[record::trace],
               {closure_normal.x, closure_normal.y, closure_normal.z, 1.0f}) &&
        close(glossy_actual[glossy_curve + record::trace],
              {closure_normal.x, closure_normal.y, closure_normal.z, 1.0f}) &&
        close(glossy_actual[glossy_curve + record::sample_direction],
              {expected_reflection.x, expected_reflection.y,
               expected_reflection.z, 1.0f});
    if (!glossy_ok) {
        print_records(backend, "glossy", glossy_actual);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
