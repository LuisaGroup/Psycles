#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::adapter;
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

constexpr luisa::float3 f82_color{0.17f, 0.31f, 0.73f};
constexpr luisa::float3 f82_tint{0.91f, 0.67f, 0.43f};
constexpr luisa::float3 conductor_ior{1.9f, 2.7f, 3.4f};
constexpr luisa::float3 conductor_extinction{4.6f, 3.2f, 2.1f};
constexpr float roughness = 0.38f;
constexpr float anisotropy = 0.63f;
constexpr float rotation = 0.25f;
constexpr float film_thickness = 420.0f;
constexpr float film_ior = 1.37f;

namespace record {
constexpr std::uint32_t f82_identity = 0u;
constexpr std::uint32_t f82_parameters = 1u;
constexpr std::uint32_t f82_b = 2u;
constexpr std::uint32_t f82_microfacet = 3u;
constexpr std::uint32_t conductor_identity = 4u;
constexpr std::uint32_t conductor_parameters = 5u;
constexpr std::uint32_t conductor_extinction = 6u;
constexpr std::uint32_t conductor_microfacet = 7u;
constexpr std::uint32_t f82_normal_evaluation = 8u;
constexpr std::uint32_t conductor_normal_evaluation = 9u;
constexpr std::uint32_t f82_sample_evaluation = 10u;
constexpr std::uint32_t f82_resampled_evaluation = 11u;
constexpr std::uint32_t conductor_sample_evaluation = 12u;
constexpr std::uint32_t conductor_resampled_evaluation = 13u;
constexpr std::uint32_t no_film_oblique = 14u;
constexpr std::uint32_t film_oblique = 15u;
constexpr std::uint32_t film_identity = 16u;
constexpr std::uint32_t f82_microfacet_tail = 17u;
constexpr std::uint32_t conductor_microfacet_tail = 18u;
constexpr std::uint32_t f82_film_oblique = 19u;
constexpr std::uint32_t f82_film_identity = 20u;
constexpr std::uint32_t f82_no_film_oblique = 21u;
constexpr std::uint32_t count = 22u;
}// namespace record

struct CompiledSurface {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceClosurePlan closure_plan;
    std::vector<luisa::float4> parameters;
};

[[nodiscard]] ShaderGraph make_metallic_graph(
    std::string fresnel_type,
    float thin_film_thickness) {
    CyclesNormalizedShaderGraph source;
    using NormalizedNode = typename decltype(source.nodes)::value_type;
    source.nodes.emplace_back(NormalizedNode{
        .id = 10u,
        .type = "geometry",
        .variant = {},
        .label = "Geometry",
        .inputs = {},
        .properties = {}});
    source.nodes.emplace_back(NormalizedNode{
        .id = 11u,
        .type = "metallic_bsdf",
        .variant = {},
        .label = "Cycles 5.2 standalone Metallic regression",
        .inputs =
            {{.socket = "Base Color",
              .source = std::nullopt,
              .value = SocketValue::color(
                  {f82_color.x, f82_color.y, f82_color.z})},
             {.socket = "Edge Tint",
              .source = std::nullopt,
              .value = SocketValue::color(
                  {f82_tint.x, f82_tint.y, f82_tint.z})},
             {.socket = "IOR",
              .source = std::nullopt,
              .value = SocketValue::vector(
                  {conductor_ior.x, conductor_ior.y, conductor_ior.z})},
             {.socket = "Extinction",
              .source = std::nullopt,
              .value = SocketValue::vector(
                  {conductor_extinction.x,
                   conductor_extinction.y,
                   conductor_extinction.z})},
             {.socket = "Roughness",
              .source = std::nullopt,
              .value = SocketValue::floating(roughness)},
             {.socket = "Anisotropy",
              .source = std::nullopt,
              .value = SocketValue::floating(anisotropy)},
             {.socket = "Rotation",
              .source = std::nullopt,
              .value = SocketValue::floating(rotation)},
             {.socket = "Normal",
              .source = CyclesOutputRef{.node = 10u, .socket = "Normal"},
              .value = std::nullopt},
             {.socket = "Tangent",
              .source = CyclesOutputRef{.node = 10u, .socket = "Tangent"},
              .value = std::nullopt},
             {.socket = "Thin Film Thickness",
              .source = std::nullopt,
              .value = SocketValue::floating(thin_film_thickness)},
             {.socket = "Thin Film IOR",
              .source = std::nullopt,
              .value = SocketValue::floating(film_ior)}},
        .properties =
            {{"distribution", SocketValue::string("ggx")},
             {"fresnel_type", SocketValue::string(std::move(fresnel_type))}}});
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 11u, .socket = "BSDF"});
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    if (!adapted.ok()) {
        throw std::runtime_error{
            "failed to adapt standalone Metallic graph"};
    }
    return std::move(*adapted.graph);
}

[[nodiscard]] CompiledSurface compile_graph(
    ShaderCompiler &compiler,
    std::string fresnel_type,
    float thin_film_thickness) {
    const auto shader = compiler.compile(make_metallic_graph(
        std::move(fresnel_type), thin_film_thickness));
    if (!shader.ok()) {
        throw std::runtime_error{
            "failed to compile standalone Metallic graph"};
    }
    const auto surface = compile_surface_program(*shader.program);
    if (!surface.ok()) {
        throw std::runtime_error{
            "failed to lower standalone Metallic graph"};
    }
    auto parameters = parameter_data(*surface.program);
    return {
        .program = surface.program,
        .closure_plan = merged_surface_closure_plan(
            *surface.program, parameters),
        .parameters = std::move(parameters)};
}

[[nodiscard]] SurfacePoint metallic_point() noexcept {
    auto point = make_surface_point();
    point.geometric_normal = make_float3(0.0f, 0.0f, 1.0f);
    point.shading_normal = point.geometric_normal;
    point.object_shading_normal = point.geometric_normal;
    point.undisplaced_shading_normal = point.geometric_normal;
    point.undisplaced_object_shading_normal = point.geometric_normal;
    // Cycles Geometry Tangent is radial in Generated space for meshes. This
    // coordinate preserves the +X tangent assumed by the Metallic oracle.
    point.generated = make_float3(0.5f, 0.25f, 0.5f);
    point.dpdu = make_float3(1.0f, 0.0f, 0.0f);
    point.incoming = make_float3(0.0f, 0.0f, 1.0f);
    return point;
}

[[nodiscard]] SurfaceQuery metallic_query() noexcept {
    return {
        .lobe_mask = static_cast<std::uint32_t>(
            event_diffuse | event_glossy |
            event_transmission | event_transparent),
        .transport_mode =
            static_cast<std::uint32_t>(TransportMode::radiance),
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = true,
        .refractive_caustics = true};
}

[[nodiscard]] constexpr float f82_b_channel(
    float f0,
    float tint) noexcept {
    constexpr float f = 6.0f / 7.0f;
    constexpr float f5 = f * f * f * f * f;
    const auto schlick = f0 + (1.0f - f0) * f5;
    return schlick * (7.0f / (f5 * f)) * (1.0f - tint);
}

[[nodiscard]] constexpr float conductor_f0_channel(
    float n,
    float k) noexcept {
    const auto numerator = (n - 1.0f) * (n - 1.0f) + k * k;
    const auto denominator = (n + 1.0f) * (n + 1.0f) + k * k;
    return numerator / denominator;
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

[[nodiscard]] bool approximately_equal_rgb(
    luisa::float3 actual,
    luisa::float3 expected,
    float tolerance) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool proportional_rgb(
    luisa::float4 actual,
    luisa::float3 expected,
    float tolerance) noexcept {
    if (!(actual.x > 0.0f) || !(expected.x > 0.0f)) {
        return false;
    }
    return approximately_equal(
               actual.y / actual.x,
               expected.y / expected.x,
               tolerance) &&
           approximately_equal(
               actual.z / actual.x,
               expected.z / expected.x,
               tolerance);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto f82 = compile_graph(compiler, "f82", 0.0f);
    const auto conductor = compile_graph(
        compiler, "physical_conductor", 0.0f);
    const auto conductor_film = compile_graph(
        compiler, "physical_conductor", film_thickness);
    const auto f82_film = compile_graph(
        compiler, "f82", film_thickness);

    const auto &f82_instruction =
        f82.program->closure_instructions().front();
    const auto &conductor_instruction =
        conductor.program->closure_instructions().front();
    const auto &film_instruction =
        conductor_film.program->closure_instructions().front();
    const auto &f82_film_instruction =
        f82_film.program->closure_instructions().front();
    if (f82_instruction.operation != ClosureOperation::metallic_f82 ||
        conductor_instruction.operation !=
            ClosureOperation::metallic_conductor ||
        film_instruction.operation != ClosureOperation::metallic_conductor ||
        f82_film_instruction.operation != ClosureOperation::metallic_f82 ||
        !f82.closure_plan.entry(f82.program->root())
             .microfacet_anisotropy ||
        f82.closure_plan.entry(f82.program->root()).thin_film ||
        conductor.closure_plan.entry(conductor.program->root()).thin_film ||
        !conductor_film.closure_plan.entry(
             conductor_film.program->root()).thin_film ||
        !f82_film.closure_plan.entry(
             f82_film.program->root()).thin_film) {
        std::cerr << "standalone Metallic static topology did not survive "
                     "normalized Cycles adaptation\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch f82_surfaces;
    const auto f82_tag = f82_surfaces.create<GraphSurface>(
        f82.program, f82.closure_plan);
    SurfaceDispatch conductor_surfaces;
    const auto conductor_tag = conductor_surfaces.create<GraphSurface>(
        conductor.program, conductor.closure_plan);
    SurfaceDispatch film_surfaces;
    const auto film_tag = film_surfaces.create<GraphSurface>(
        conductor_film.program, conductor_film.closure_plan);
    SurfaceDispatch f82_film_surfaces;
    const auto f82_film_tag = f82_film_surfaces.create<GraphSurface>(
        f82_film.program, f82_film.closure_plan);

    Kernel1D test = [&](BufferFloat4 f82_parameters,
                        BufferFloat4 conductor_parameters,
                        BufferFloat4 film_parameters,
                        BufferFloat4 f82_film_parameters,
                        BufferFloat4 output) noexcept {
        const auto point = metallic_point();
        const auto query = metallic_query();
        ParameterShaderServices f82_services{f82_parameters};
        ParameterShaderServices conductor_services{conductor_parameters};
        ParameterShaderServices film_services{film_parameters};
        ParameterShaderServices f82_film_services{f82_film_parameters};

        SurfaceClosureSet f82_closures{1u};
        static_cast<void>(f82_surfaces.collect_closures(
            UInt{f82_tag}, f82_services, point, true, true, f82_closures));
        const auto f82_closure = f82_closures.entry(0u);
        output.write(
            record::f82_identity,
            make_float4(
                cast<float>(f82_closure.closure_type),
                cast<float>(f82_closure.microfacet_fresnel),
                cast<float>(f82_closures.count()),
                select(0.0f, 1.0f, f82_closure.setup_valid)));
        output.write(
            record::f82_parameters,
            make_float4(f82_closure.color, f82_closure.sample_weight));
        output.write(
            record::f82_b,
            make_float4(f82_closure.specular_tint, f82_closure.roughness));
        output.write(
            record::f82_microfacet,
            make_float4(
                f82_closure.microfacet_tangent,
                f82_closure.microfacet_alpha_x));
        output.write(
            record::f82_microfacet_tail,
            make_float4(
                f82_closure.microfacet_alpha_y,
                f82_closure.thin_film_thickness,
                f82_closure.thin_film_ior,
                0.0f));

        SurfaceClosureSet conductor_closures{1u};
        static_cast<void>(conductor_surfaces.collect_closures(
            UInt{conductor_tag}, conductor_services, point, true, true,
            conductor_closures));
        const auto conductor_closure = conductor_closures.entry(0u);
        output.write(
            record::conductor_identity,
            make_float4(
                cast<float>(conductor_closure.closure_type),
                cast<float>(conductor_closure.microfacet_fresnel),
                cast<float>(conductor_closures.count()),
                select(0.0f, 1.0f, conductor_closure.setup_valid)));
        output.write(
            record::conductor_parameters,
            make_float4(
                conductor_closure.color,
                conductor_closure.sample_weight));
        output.write(
            record::conductor_extinction,
            make_float4(
                conductor_closure.specular_tint,
                conductor_closure.roughness));
        output.write(
            record::conductor_microfacet,
            make_float4(
                conductor_closure.microfacet_tangent,
                conductor_closure.microfacet_alpha_x));
        output.write(
            record::conductor_microfacet_tail,
            make_float4(
                conductor_closure.microfacet_alpha_y,
                conductor_closure.thin_film_thickness,
                conductor_closure.thin_film_ior,
                0.0f));

        const auto normal = make_float3(0.0f, 0.0f, 1.0f);
        const auto f82_normal = f82_surfaces.evaluate(
            UInt{f82_tag}, f82_services, point, normal, query);
        const auto conductor_normal = conductor_surfaces.evaluate(
            UInt{conductor_tag}, conductor_services, point, normal, query);
        output.write(
            record::f82_normal_evaluation,
            make_float4(f82_normal.f, f82_normal.pdf));
        output.write(
            record::conductor_normal_evaluation,
            make_float4(conductor_normal.f, conductor_normal.pdf));

        const auto f82_sample = f82_surfaces.sample_trace(
            UInt{f82_tag},
            f82_services,
            point,
            0.37f,
            make_float2(0.29f, 0.71f),
            query);
        const auto f82_resampled = f82_surfaces.evaluate(
            UInt{f82_tag},
            f82_services,
            point,
            f82_sample.sample.wi,
            query);
        output.write(
            record::f82_sample_evaluation,
            make_float4(
                f82_sample.sample.evaluation.f,
                f82_sample.sample.evaluation.pdf));
        output.write(
            record::f82_resampled_evaluation,
            make_float4(f82_resampled.f, f82_resampled.pdf));

        const auto conductor_sample = conductor_surfaces.sample_trace(
            UInt{conductor_tag},
            conductor_services,
            point,
            0.61f,
            make_float2(0.43f, 0.83f),
            query);
        const auto conductor_resampled = conductor_surfaces.evaluate(
            UInt{conductor_tag},
            conductor_services,
            point,
            conductor_sample.sample.wi,
            query);
        output.write(
            record::conductor_sample_evaluation,
            make_float4(
                conductor_sample.sample.evaluation.f,
                conductor_sample.sample.evaluation.pdf));
        output.write(
            record::conductor_resampled_evaluation,
            make_float4(conductor_resampled.f, conductor_resampled.pdf));

        const auto oblique = normalize(make_float3(0.73f, 0.11f, 0.67f));
        const auto no_film = conductor_surfaces.evaluate(
            UInt{conductor_tag}, conductor_services, point, oblique, query);
        const auto f82_no_film = f82_surfaces.evaluate(
            UInt{f82_tag}, f82_services, point, oblique, query);
        const auto film = film_surfaces.evaluate(
            UInt{film_tag}, film_services, point, oblique, query);
        output.write(
            record::no_film_oblique,
            make_float4(no_film.f, no_film.pdf));
        output.write(
            record::f82_no_film_oblique,
            make_float4(f82_no_film.f, f82_no_film.pdf));
        output.write(
            record::film_oblique,
            make_float4(film.f, film.pdf));

        SurfaceClosureSet film_closures{1u};
        static_cast<void>(film_surfaces.collect_closures(
            UInt{film_tag}, film_services, point, true, true, film_closures));
        const auto film_closure = film_closures.entry(0u);
        output.write(
            record::film_identity,
            make_float4(
                cast<float>(film_closure.microfacet_fresnel),
                film_closure.thin_film_thickness,
                film_closure.thin_film_ior,
                cast<float>(film_closures.count())));

        const auto f82_film_evaluation = f82_film_surfaces.evaluate(
            UInt{f82_film_tag},
            f82_film_services,
            point,
            oblique,
            query);
        output.write(
            record::f82_film_oblique,
            make_float4(f82_film_evaluation.f, f82_film_evaluation.pdf));
        SurfaceClosureSet f82_film_closures{1u};
        static_cast<void>(f82_film_surfaces.collect_closures(
            UInt{f82_film_tag},
            f82_film_services,
            point,
            true,
            true,
            f82_film_closures));
        const auto f82_film_closure = f82_film_closures.entry(0u);
        output.write(
            record::f82_film_identity,
            make_float4(
                cast<float>(f82_film_closure.microfacet_fresnel),
                f82_film_closure.thin_film_thickness,
                f82_film_closure.thin_film_ior,
                cast<float>(f82_film_closures.count())));
    };

    if (backend == "fallback" &&
        !require_bounded_xir(
            "standalone_metallic_surface", test, 180000u)) {
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto f82_buffer =
        device.create_buffer<luisa::float4>(f82.parameters.size());
    auto conductor_buffer =
        device.create_buffer<luisa::float4>(conductor.parameters.size());
    auto film_buffer =
        device.create_buffer<luisa::float4>(conductor_film.parameters.size());
    auto f82_film_buffer =
        device.create_buffer<luisa::float4>(f82_film.parameters.size());
    auto output_buffer =
        device.create_buffer<luisa::float4>(record::count);
    auto shader = compile_named_kernel(
        device, "standalone_metallic_surface", test);
    std::array<luisa::float4, record::count> actual{};
    stream << f82_buffer.copy_from(luisa::span{f82.parameters})
           << conductor_buffer.copy_from(luisa::span{conductor.parameters})
           << film_buffer.copy_from(luisa::span{conductor_film.parameters})
           << f82_film_buffer.copy_from(luisa::span{f82_film.parameters})
           << shader(
                  f82_buffer,
                  conductor_buffer,
                  film_buffer,
                  f82_film_buffer,
                  output_buffer)
                  .dispatch(1u)
           << output_buffer.copy_to(luisa::span{actual})
           << synchronize();

    constexpr auto reflection_type =
        static_cast<float>(cycles_closure::type_microfacet_ggx);
    constexpr auto f82_fresnel = static_cast<float>(
        cycles_closure::MicrofacetFresnel::f82_tint);
    constexpr auto conductor_fresnel = static_cast<float>(
        cycles_closure::MicrofacetFresnel::conductor);
    const auto expected_b = luisa::float3{
        f82_b_channel(f82_color.x, f82_tint.x),
        f82_b_channel(f82_color.y, f82_tint.y),
        f82_b_channel(f82_color.z, f82_tint.z)};
    const auto aspect = std::sqrt(1.0f - 0.9f * anisotropy);
    const auto alpha = roughness * roughness;
    const auto expected_alpha_x = std::min(alpha / aspect, 1.0f);
    const auto expected_alpha_y = std::min(alpha * aspect, 1.0f);
    const auto expected_conductor_f0 = luisa::float3{
        conductor_f0_channel(conductor_ior.x, conductor_extinction.x),
        conductor_f0_channel(conductor_ior.y, conductor_extinction.y),
        conductor_f0_channel(conductor_ior.z, conductor_extinction.z)};

    constexpr float parameter_tolerance = 3.0e-5f;
    const auto identity_ok =
        approximately_equal(
            actual[record::f82_identity],
            luisa::float4{reflection_type, f82_fresnel, 1.0f, 1.0f},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::conductor_identity],
            luisa::float4{
                reflection_type, conductor_fresnel, 1.0f, 1.0f},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::film_identity],
            luisa::float4{
                conductor_fresnel, film_thickness, film_ior, 1.0f},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::f82_film_identity],
            luisa::float4{
                f82_fresnel, film_thickness, film_ior, 1.0f},
            parameter_tolerance);
    const auto parameters_ok =
        approximately_equal_rgb(
            actual[record::f82_parameters].xyz(),
            f82_color,
            parameter_tolerance) &&
        approximately_equal_rgb(
            actual[record::f82_b].xyz(),
            expected_b,
            parameter_tolerance) &&
        approximately_equal_rgb(
            actual[record::conductor_parameters].xyz(),
            conductor_ior,
            parameter_tolerance) &&
        approximately_equal_rgb(
            actual[record::conductor_extinction].xyz(),
            conductor_extinction,
            parameter_tolerance) &&
        approximately_equal(
            actual[record::f82_b].w,
            roughness,
            parameter_tolerance) &&
        approximately_equal(
            actual[record::conductor_extinction].w,
            roughness,
            parameter_tolerance);
    const auto microfacet_ok =
        approximately_equal(
            actual[record::f82_microfacet],
            luisa::float4{0.0f, 1.0f, 0.0f, expected_alpha_x},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::conductor_microfacet],
            luisa::float4{0.0f, 1.0f, 0.0f, expected_alpha_x},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::f82_microfacet_tail],
            luisa::float4{expected_alpha_y, 0.0f, 0.0f, 0.0f},
            parameter_tolerance) &&
        approximately_equal(
            actual[record::conductor_microfacet_tail],
            luisa::float4{expected_alpha_y, 0.0f, 0.0f, 0.0f},
            parameter_tolerance);
    const auto normal_fresnel_ok = proportional_rgb(
                                       actual[record::f82_normal_evaluation],
                                       f82_color,
                                       2.0e-4f) &&
                                   proportional_rgb(
                                       actual[record::conductor_normal_evaluation],
                                       expected_conductor_f0,
                                       2.0e-4f);
    const auto sample_eval_ok =
        approximately_equal(
            actual[record::f82_sample_evaluation],
            actual[record::f82_resampled_evaluation],
            8.0e-5f) &&
        approximately_equal(
            actual[record::conductor_sample_evaluation],
            actual[record::conductor_resampled_evaluation],
            8.0e-5f);
    const auto film_delta =
        std::abs(actual[record::film_oblique].x -
                 actual[record::no_film_oblique].x) +
        std::abs(actual[record::film_oblique].y -
                 actual[record::no_film_oblique].y) +
        std::abs(actual[record::film_oblique].z -
                 actual[record::no_film_oblique].z);
    const auto f82_film_delta =
        std::abs(actual[record::f82_film_oblique].x -
                 actual[record::f82_no_film_oblique].x) +
        std::abs(actual[record::f82_film_oblique].y -
                 actual[record::f82_no_film_oblique].y) +
        std::abs(actual[record::f82_film_oblique].z -
                 actual[record::f82_no_film_oblique].z);
    auto all_finite = true;
    for (const auto value : actual) {
        all_finite &= finite(value);
    }

    if (!identity_ok || !parameters_ok || !microfacet_ok ||
        !normal_fresnel_ok || !sample_eval_ok ||
        !(film_delta > 1.0e-5f) ||
        !(f82_film_delta > 1.0e-5f) || !all_finite) {
        std::cerr << "standalone Metallic physical regression failed on "
                  << backend << ": identity=" << identity_ok
                  << ", parameters=" << parameters_ok
                  << ", microfacet=" << microfacet_ok
                  << ", normal Fresnel=" << normal_fresnel_ok
                  << ", sample/eval=" << sample_eval_ok
                  << ", film delta=" << film_delta
                  << ", F82 film delta=" << f82_film_delta
                  << ", finite=" << all_finite
                  << "; F82 identity={"
                  << actual[record::f82_identity].x << ", "
                  << actual[record::f82_identity].y << ", "
                  << actual[record::f82_identity].z << ", "
                  << actual[record::f82_identity].w
                  << "}, conductor identity={"
                  << actual[record::conductor_identity].x << ", "
                  << actual[record::conductor_identity].y << ", "
                  << actual[record::conductor_identity].z << ", "
                  << actual[record::conductor_identity].w
                  << "}, film identity={"
                  << actual[record::film_identity].x << ", "
                  << actual[record::film_identity].y << ", "
                  << actual[record::film_identity].z << ", "
                  << actual[record::film_identity].w << "}\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
