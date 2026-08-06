#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::adapter;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr float classroom_roughness = 0.6116908789f;
constexpr luisa::float2 classroom_random{0.7217738628f, 0.6956095099f};
constexpr luisa::float3 classroom_incoming{0.2519240379f, -0.9676744938f,
                                           -0.0118478583f};
constexpr luisa::float3 classroom_beckmann_outgoing{
    -0.5096411705f, -0.8317917585f, -0.2199728042f};

[[nodiscard]] ShaderGraph make_glossy_graph(std::string_view distribution) {
  CyclesNormalizedShaderGraph source;
  source.nodes = {
      {.id = 10u,
       .type = "geometry",
       .variant = {},
       .label = "Geometry",
       .inputs = {},
       .properties = {}},
      {.id = 11u,
       .type = "glossy_bsdf",
       .variant = {},
       .label = "Standalone Glossy distribution regression",
       .inputs = {{.socket = "Color",
                   .source = std::nullopt,
                   .value = SocketValue::color({1.0f, 1.0f, 1.0f})},
                  {.socket = "Roughness",
                   .source = std::nullopt,
                   .value = SocketValue::floating(classroom_roughness)},
                  {.socket = "Normal",
                   .source = CyclesOutputRef{.node = 10u, .socket = "Normal"},
                   .value = std::nullopt}},
       .properties = {
           {"distribution", SocketValue::string(std::string{distribution})}}}};
  source.set_root(ShaderDomain::surface,
                  CyclesOutputRef{.node = 11u, .socket = "BSDF"});
  auto adapted =
      adapt_cycles_shader_graph(source, make_core_cycles_node_mappings());
  if (!adapted.ok()) {
    throw std::runtime_error{
        "failed to adapt normalized standalone Glossy graph"};
  }
  return std::move(*adapted.graph);
}

[[nodiscard]] bool rgb_equal(luisa::float4 actual, float expected,
                             float tolerance = 6.0e-5f) noexcept {
  return approximately_equal(actual.x, expected, tolerance) &&
         approximately_equal(actual.y, expected, tolerance) &&
         approximately_equal(actual.z, expected, tolerance);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  ShaderCompiler compiler{make_core_node_registry()};
  const auto beckmann_shader = compiler.compile(make_glossy_graph("BECKMANN"));
  const auto ggx_shader = compiler.compile(make_glossy_graph("GGX"));
  if (!beckmann_shader.ok() || !ggx_shader.ok()) {
    std::cerr << "failed to compile standalone Glossy graphs\n";
    return EXIT_FAILURE;
  }
  const auto beckmann_program =
      compile_surface_program(*beckmann_shader.program);
  const auto ggx_program = compile_surface_program(*ggx_shader.program);
  if (!beckmann_program.ok() || !ggx_program.ok()) {
    std::cerr << "failed to lower standalone Glossy programs\n";
    return EXIT_FAILURE;
  }
  const auto &beckmann_lowered =
      beckmann_program.program->closure_instructions().front();
  const auto &ggx_lowered = ggx_program.program->closure_instructions().front();
  if (beckmann_lowered.operation != ClosureOperation::glossy ||
      !beckmann_lowered.beckmann || beckmann_lowered.preserve_ggx_energy ||
      ggx_lowered.operation != ClosureOperation::glossy ||
      ggx_lowered.beckmann || ggx_lowered.preserve_ggx_energy) {
    std::cerr << "standalone Glossy lost its typed distribution\n";
    return EXIT_FAILURE;
  }

  const auto beckmann_values = parameter_data(*beckmann_program.program);
  const auto ggx_values = parameter_data(*ggx_program.program);
  SurfaceDispatch surfaces;
  const auto beckmann_tag =
      surfaces.create<GraphSurface>(beckmann_program.program);
  const auto ggx_tag = surfaces.create<GraphSurface>(ggx_program.program);

  constexpr std::uint32_t record_count = 10u;
  Kernel1D evaluate = [&](BufferFloat4 beckmann_parameter_buffer,
                          BufferFloat4 ggx_parameter_buffer,
                          BufferFloat4 output) noexcept {
    ParameterShaderServices beckmann_services{beckmann_parameter_buffer};
    ParameterShaderServices ggx_services{ggx_parameter_buffer};
    auto point = make_surface_point();
    const auto normal = make_float3(0.0f, -1.0f, 0.0f);
    point.geometric_normal = normal;
    point.shading_normal = normal;
    point.object_shading_normal = normal;
    point.undisplaced_shading_normal = normal;
    point.undisplaced_object_shading_normal = normal;
    point.incoming = make_float3(classroom_incoming);

    constexpr auto all_lobes = static_cast<std::uint32_t>(
        event_diffuse | event_glossy | event_transmission | event_transparent);
    const auto query = SurfaceQuery{
        .lobe_mask = all_lobes,
        .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = true,
        .refractive_caustics = true};

    const auto beckmann_trace = surfaces.closure_trace(
        UInt{beckmann_tag}, beckmann_services, point, 0u, true, true);
    const auto ggx_trace = surfaces.closure_trace(UInt{ggx_tag}, ggx_services,
                                                  point, 0u, true, true);
    output.write(0u, make_float4(cast<float>(beckmann_trace.count),
                                 cast<float>(beckmann_trace.type),
                                 beckmann_trace.sample_weight,
                                 select(0.0f, 1.0f, beckmann_trace.valid)));
    output.write(1u, make_float4(cast<float>(ggx_trace.count),
                                 cast<float>(ggx_trace.type),
                                 ggx_trace.sample_weight,
                                 select(0.0f, 1.0f, ggx_trace.valid)));

    const auto beckmann_sample = surfaces.sample_trace(
        UInt{beckmann_tag}, beckmann_services, point, 0.9328697920f,
        make_float2(classroom_random), query);
    output.write(2u,
                 make_float4(beckmann_sample.sample.wi,
                             select(0.0f, 1.0f, beckmann_sample.sample.valid)));
    output.write(3u, make_float4(beckmann_sample.sample.evaluation.f,
                                 beckmann_sample.sample.evaluation.pdf));
    output.write(
        4u, make_float4(cast<float>(beckmann_sample.sample.evaluation.events),
                        cast<float>(beckmann_sample.closure_type),
                        beckmann_sample.selection_rescaled,
                        select(0.0f, 1.0f, beckmann_sample.closure_valid)));

    const auto classroom_light_direction =
        make_float3(-0.2780451477f, -0.9296237826f, 0.2418480814f);
    const auto classroom_evaluation =
        surfaces.evaluate(UInt{beckmann_tag}, beckmann_services, point,
                          classroom_light_direction, query);
    output.write(5u,
                 make_float4(classroom_evaluation.f, classroom_evaluation.pdf));

    const auto oblique_direction = normalize(make_float3(0.92f, -0.25f, 0.30f));
    const auto oblique_evaluation = surfaces.evaluate(
        UInt{beckmann_tag}, beckmann_services, point, oblique_direction, query);
    output.write(6u, make_float4(oblique_evaluation.f, oblique_evaluation.pdf));

    SurfaceClosureSet collection{1u};
    static_cast<void>(surfaces.collect_closures(
        UInt{beckmann_tag}, beckmann_services, point, true, true, collection));
    const auto collected = collection.entry(0u);
    output.write(7u,
                 make_float4(select(0.0f, 1.0f, collected.beckmann),
                             cast<float>(collected.kind), collected.roughness,
                             cast<float>(collection.count())));

    const auto ggx_sample =
        surfaces.sample_trace(UInt{ggx_tag}, ggx_services, point, 0.9328697920f,
                              make_float2(classroom_random), query);
    output.write(8u, make_float4(ggx_sample.sample.wi,
                                 select(0.0f, 1.0f, ggx_sample.sample.valid)));
    output.write(9u, make_float4(ggx_sample.sample.evaluation.f,
                                 ggx_sample.sample.evaluation.pdf));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto beckmann_buffer =
      device.create_buffer<luisa::float4>(beckmann_values.size());
  auto ggx_buffer = device.create_buffer<luisa::float4>(ggx_values.size());
  auto output_buffer = device.create_buffer<luisa::float4>(record_count);
  auto kernel = device.compile(evaluate);
  std::array<luisa::float4, record_count> actual{};
  stream << beckmann_buffer.copy_from(luisa::span{beckmann_values})
         << ggx_buffer.copy_from(luisa::span{ggx_values})
         << kernel(beckmann_buffer, ggx_buffer, output_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr auto beckmann_type =
      static_cast<float>(cycles_closure::type_microfacet_beckmann);
  constexpr auto ggx_type =
      static_cast<float>(cycles_closure::type_microfacet_ggx);
  constexpr auto glossy_reflection =
      static_cast<float>(event_glossy | event_reflection);
  const auto trace_matches =
      approximately_equal(actual[0u],
                          luisa::float4{1.0f, beckmann_type, 1.0f, 1.0f}) &&
      approximately_equal(actual[1u],
                          luisa::float4{1.0f, ggx_type, 1.0f, 1.0f});
  const auto sample_matches =
      approximately_equal(actual[2u],
                          luisa::float4{classroom_beckmann_outgoing.x,
                                        classroom_beckmann_outgoing.y,
                                        classroom_beckmann_outgoing.z, 1.0f},
                          7.0e-5f) &&
      rgb_equal(actual[3u], 0.48469383f) &&
      approximately_equal(actual[3u].w, 0.48469383f, 6.0e-5f) &&
      approximately_equal(actual[4u].x, glossy_reflection) &&
      approximately_equal(actual[4u].y, beckmann_type) &&
      approximately_equal(actual[4u].w, 1.0f);
  // These constants are pinned from Cycles 29ccd5e2's Beckmann D and
  // Smith-Lambda implementation for the Classroom blackboard directions.
  // The oblique case deliberately exercises the nonzero Lambda branch.
  const auto evaluation_matches =
      rgb_equal(actual[5u], 0.54399043f) &&
      approximately_equal(actual[5u].w, 0.54399043f, 6.0e-5f) &&
      rgb_equal(actual[6u], 0.0018893486f) &&
      approximately_equal(actual[6u].w, 0.0020627747f, 6.0e-5f);
  const auto collection_matches =
      approximately_equal(actual[7u].x, 1.0f) &&
      approximately_equal(actual[7u].y,
                          static_cast<float>(SurfaceClosureKind::glossy)) &&
      approximately_equal(actual[7u].z, classroom_roughness) &&
      approximately_equal(actual[7u].w, 1.0f);
  const auto distribution_is_effective =
      approximately_equal(actual[8u].w, 1.0f) &&
      (length(actual[8u].xyz() - actual[2u].xyz()) > 0.25f);
  if (!trace_matches || !sample_matches || !evaluation_matches ||
      !collection_matches || !distribution_is_effective) {
    std::cerr << "standalone Beckmann Glossy regression failed on " << backend
              << '\n';
    for (auto index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": {" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "}\n";
    }
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
