#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_shader_shape_test_support.h"
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
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;

constexpr float classroom_roughness = 0.6116908789f;
constexpr luisa::float2 classroom_random{0.7217738628f, 0.6956095099f};
constexpr luisa::float3 classroom_incoming{0.2519240379f, -0.9676744938f,
                                           -0.0118478583f};
constexpr luisa::float3 classroom_beckmann_outgoing{
    -0.5096411705f, -0.8317917585f, -0.2199728042f};

namespace record {
constexpr std::uint32_t beckmann_trace = 0u;
constexpr std::uint32_t ggx_trace = 1u;
constexpr std::uint32_t beckmann_direction = 2u;
constexpr std::uint32_t beckmann_sample = 3u;
constexpr std::uint32_t beckmann_selection = 4u;
constexpr std::uint32_t beckmann_classroom_evaluation = 5u;
constexpr std::uint32_t beckmann_oblique_evaluation = 6u;
constexpr std::uint32_t beckmann_collection = 7u;
constexpr std::uint32_t ggx_direction = 8u;
constexpr std::uint32_t ggx_sample = 9u;
constexpr std::uint32_t average_roughness_squared = 10u;
constexpr std::uint32_t count = 11u;
} // namespace record

namespace roughness_record {
constexpr std::uint32_t beckmann_sample = 0u;
constexpr std::uint32_t beckmann_classroom_evaluation = 1u;
constexpr std::uint32_t beckmann_oblique_evaluation = 2u;
constexpr std::uint32_t ggx_sample = 3u;
constexpr std::uint32_t count = 4u;
} // namespace roughness_record

[[nodiscard]] SurfacePoint make_glossy_point() noexcept {
  auto point = make_surface_point();
  const auto normal = make_float3(0.0f, -1.0f, 0.0f);
  point.geometric_normal = normal;
  point.shading_normal = normal;
  point.object_shading_normal = normal;
  point.undisplaced_shading_normal = normal;
  point.undisplaced_object_shading_normal = normal;
  point.incoming = make_float3(classroom_incoming);
  return point;
}

[[nodiscard]] SurfaceQuery make_glossy_query() noexcept {
  constexpr auto all_lobes = static_cast<std::uint32_t>(
      event_diffuse | event_glossy | event_transmission | event_transparent);
  return {.lobe_mask = all_lobes,
          .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
          .glossy_filter_roughness = 0.0f,
          .reflective_caustics = true,
          .refractive_caustics = true};
}

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
  SurfaceDispatch beckmann_surfaces;
  const auto beckmann_tag =
      beckmann_surfaces.create<GraphSurface>(
          beckmann_program.program,
          merged_surface_closure_plan(
              *beckmann_program.program, beckmann_values));
  SurfaceDispatch ggx_surfaces;
  const auto ggx_tag = ggx_surfaces.create<GraphSurface>(
      ggx_program.program,
      merged_surface_closure_plan(*ggx_program.program, ggx_values));

  Kernel1D trace_beckmann = [&](BufferFloat4 parameters,
                                BufferFloat4 output) noexcept {
    ParameterShaderServices services{parameters};
    const auto value = beckmann_surfaces.closure_trace(
        UInt{beckmann_tag}, services, make_glossy_point(), 0u, true, true);
    output.write(record::beckmann_trace,
                 make_float4(cast<float>(value.count), cast<float>(value.type),
                             value.sample_weight,
                             select(0.0f, 1.0f, value.valid)));
  };

  Kernel1D trace_ggx = [&](BufferFloat4 parameters,
                           BufferFloat4 output) noexcept {
    ParameterShaderServices services{parameters};
    const auto value = ggx_surfaces.closure_trace(
        UInt{ggx_tag}, services, make_glossy_point(), 0u, true, true);
    output.write(record::ggx_trace,
                 make_float4(cast<float>(value.count), cast<float>(value.type),
                             value.sample_weight,
                             select(0.0f, 1.0f, value.valid)));
  };

  Kernel1D sample_beckmann = [&](BufferFloat4 parameters, BufferFloat4 output,
                                 BufferFloat roughness) noexcept {
    ParameterShaderServices services{parameters};
    const auto value = beckmann_surfaces.sample_trace(
        UInt{beckmann_tag}, services, make_glossy_point(), 0.9328697920f,
        make_float2(classroom_random), make_glossy_query());
    output.write(
        record::beckmann_direction,
        make_float4(value.sample.wi, select(0.0f, 1.0f, value.sample.valid)));
    output.write(
        record::beckmann_sample,
        make_float4(value.sample.evaluation.f, value.sample.evaluation.pdf));
    output.write(record::beckmann_selection,
                 make_float4(cast<float>(value.sample.evaluation.events),
                             cast<float>(value.closure_type),
                             value.selection_rescaled,
                             select(0.0f, 1.0f, value.closure_valid)));
    roughness.write(roughness_record::beckmann_sample,
                    value.sample.evaluation.average_roughness_squared);
  };

  Kernel1D evaluate_beckmann = [&](BufferFloat4 parameters, BufferFloat4 output,
                                   BufferFloat roughness) noexcept {
    ParameterShaderServices services{parameters};
    const auto case_index = dispatch_x();
    const auto classroom_direction =
        make_float3(-0.2780451477f, -0.9296237826f, 0.2418480814f);
    const auto oblique_direction = normalize(make_float3(0.92f, -0.25f, 0.30f));
    const auto direction =
        select(oblique_direction, classroom_direction, case_index == 0u);
    const auto value = beckmann_surfaces.evaluate(
        UInt{beckmann_tag}, services, make_glossy_point(), direction,
        make_glossy_query());
    output.write(record::beckmann_classroom_evaluation + case_index,
                 make_float4(value.f, value.pdf));
    roughness.write(roughness_record::beckmann_classroom_evaluation +
                        case_index,
                    value.average_roughness_squared);
  };

  Kernel1D collect_beckmann = [&](BufferFloat4 parameters,
                                  BufferFloat4 output) noexcept {
    ParameterShaderServices services{parameters};
    SurfaceClosureSet collection{1u};
    static_cast<void>(beckmann_surfaces.collect_closures(
        UInt{beckmann_tag}, services, make_glossy_point(), true, true,
        collection));
    const auto collected = collection.entry(0u);
    output.write(record::beckmann_collection,
                 make_float4(select(0.0f, 1.0f, collected.beckmann),
                             cast<float>(collected.closure_type),
                             collected.roughness,
                             cast<float>(collection.count())));
  };

  Kernel1D sample_ggx = [&](BufferFloat4 parameters, BufferFloat4 output,
                            BufferFloat roughness) noexcept {
    ParameterShaderServices services{parameters};
    const auto value = ggx_surfaces.sample_trace(
        UInt{ggx_tag}, services, make_glossy_point(), 0.9328697920f,
        make_float2(classroom_random), make_glossy_query());
    output.write(
        record::ggx_direction,
        make_float4(value.sample.wi, select(0.0f, 1.0f, value.sample.valid)));
    output.write(record::ggx_sample, make_float4(value.sample.evaluation.f,
                                                 value.sample.evaluation.pdf));
    roughness.write(roughness_record::ggx_sample,
                    value.sample.evaluation.average_roughness_squared);
  };

  Kernel1D pack_roughness = [](BufferFloat roughness,
                               BufferFloat4 output) noexcept {
    output.write(
        record::average_roughness_squared,
        make_float4(
            roughness.read(roughness_record::beckmann_sample),
            roughness.read(roughness_record::beckmann_classroom_evaluation),
            roughness.read(roughness_record::beckmann_oblique_evaluation),
            roughness.read(roughness_record::ggx_sample)));
  };

  if (backend == "fallback") {
    auto bounded = true;
    bounded &=
        require_bounded_xir("beckmann_glossy_trace", trace_beckmann, 3000u);
    bounded &= require_bounded_xir("ggx_glossy_trace", trace_ggx, 3000u);
    bounded &=
        require_bounded_xir("beckmann_glossy_sample", sample_beckmann, 30000u);
    bounded &= require_bounded_xir("beckmann_glossy_evaluate",
                                   evaluate_beckmann, 15000u);
    bounded &=
        require_bounded_xir("beckmann_glossy_collect", collect_beckmann, 3500u);
    bounded &= require_bounded_xir("ggx_glossy_sample", sample_ggx, 30000u);
    bounded &= require_bounded_xir("beckmann_glossy_pack_roughness",
                                   pack_roughness, 64u);
    if (!bounded) {
      return EXIT_FAILURE;
    }
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto beckmann_buffer =
      device.create_buffer<luisa::float4>(beckmann_values.size());
  auto ggx_buffer = device.create_buffer<luisa::float4>(ggx_values.size());
  auto output_buffer = device.create_buffer<luisa::float4>(record::count);
  auto roughness_buffer = device.create_buffer<float>(roughness_record::count);
  auto beckmann_trace_kernel =
      compile_named_kernel(device, "beckmann_glossy_trace", trace_beckmann);
  auto ggx_trace_kernel =
      compile_named_kernel(device, "ggx_glossy_trace", trace_ggx);
  auto beckmann_sample_kernel =
      compile_named_kernel(device, "beckmann_glossy_sample", sample_beckmann);
  auto beckmann_evaluate_kernel = compile_named_kernel(
      device, "beckmann_glossy_evaluate", evaluate_beckmann);
  auto beckmann_collect_kernel =
      compile_named_kernel(device, "beckmann_glossy_collect", collect_beckmann);
  auto ggx_sample_kernel =
      compile_named_kernel(device, "ggx_glossy_sample", sample_ggx);
  auto pack_roughness_kernel = compile_named_kernel(
      device, "beckmann_glossy_pack_roughness", pack_roughness);
  std::array<luisa::float4, record::count> actual{};
  stream << beckmann_buffer.copy_from(luisa::span{beckmann_values})
         << ggx_buffer.copy_from(luisa::span{ggx_values})
         << beckmann_trace_kernel(beckmann_buffer, output_buffer).dispatch(1u)
         << ggx_trace_kernel(ggx_buffer, output_buffer).dispatch(1u)
         << beckmann_sample_kernel(beckmann_buffer, output_buffer,
                                   roughness_buffer)
                .dispatch(1u)
         << beckmann_evaluate_kernel(beckmann_buffer, output_buffer,
                                     roughness_buffer)
                .dispatch(2u)
         << beckmann_collect_kernel(beckmann_buffer, output_buffer).dispatch(1u)
         << ggx_sample_kernel(ggx_buffer, output_buffer, roughness_buffer)
                .dispatch(1u)
         << pack_roughness_kernel(roughness_buffer, output_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr auto beckmann_type =
      static_cast<float>(cycles_closure::type_microfacet_beckmann);
  constexpr auto ggx_type =
      static_cast<float>(cycles_closure::type_microfacet_ggx);
  constexpr auto glossy_reflection =
      static_cast<float>(event_glossy | event_reflection);
  const auto trace_matches =
      approximately_equal(actual[record::beckmann_trace],
                          luisa::float4{1.0f, beckmann_type, 1.0f, 1.0f}) &&
      approximately_equal(actual[record::ggx_trace],
                          luisa::float4{1.0f, ggx_type, 1.0f, 1.0f});
  const auto sample_matches =
      approximately_equal(actual[record::beckmann_direction],
                          luisa::float4{classroom_beckmann_outgoing.x,
                                        classroom_beckmann_outgoing.y,
                                        classroom_beckmann_outgoing.z, 1.0f},
                          7.0e-5f) &&
      rgb_equal(actual[record::beckmann_sample], 0.48469383f) &&
      approximately_equal(actual[record::beckmann_sample].w, 0.48469383f,
                          6.0e-5f) &&
      approximately_equal(actual[record::beckmann_selection].x,
                          glossy_reflection) &&
      approximately_equal(actual[record::beckmann_selection].y,
                          beckmann_type) &&
      approximately_equal(actual[record::beckmann_selection].w, 1.0f);
  // These constants are pinned from Cycles 29ccd5e2's Beckmann D and
  // Smith-Lambda implementation for the Classroom blackboard directions.
  // The oblique case deliberately exercises the nonzero Lambda branch.
  const auto evaluation_matches =
      rgb_equal(actual[record::beckmann_classroom_evaluation], 0.54399043f) &&
      approximately_equal(actual[record::beckmann_classroom_evaluation].w,
                          0.54399043f, 6.0e-5f) &&
      rgb_equal(actual[record::beckmann_oblique_evaluation], 0.0018893486f) &&
      approximately_equal(actual[record::beckmann_oblique_evaluation].w,
                          0.0020627747f, 6.0e-5f);
  const auto collection_matches =
      approximately_equal(actual[record::beckmann_collection].x, 1.0f) &&
      approximately_equal(actual[record::beckmann_collection].y,
                          static_cast<float>(
                              cycles_closure::type_microfacet_beckmann)) &&
      approximately_equal(actual[record::beckmann_collection].z,
                          classroom_roughness) &&
      approximately_equal(actual[record::beckmann_collection].w, 1.0f);
  const auto distribution_is_effective =
      approximately_equal(actual[record::ggx_direction].w, 1.0f) &&
      (length(actual[record::ggx_direction].xyz() -
              actual[record::beckmann_direction].xyz()) > 0.25f);
  constexpr auto classroom_alpha = classroom_roughness * classroom_roughness;
  constexpr auto classroom_average_roughness_squared =
      classroom_alpha * classroom_alpha;
  const auto differential_roughness_matches =
      approximately_equal(actual[record::average_roughness_squared],
                          luisa::float4{classroom_average_roughness_squared});
  if (!trace_matches || !sample_matches || !evaluation_matches ||
      !collection_matches || !distribution_is_effective ||
      !differential_roughness_matches) {
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
