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
#include <string_view>

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

[[nodiscard]] ShaderGraph make_diffuse_glossy_graph() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse base");
  const auto glossy = graph.add_node(node_type::glossy_bsdf, "Reflective top");
  const auto root =
      graph.add_node(node_type::add_closure, "Original closure sum");
  const auto configured =
      graph.set_input(diffuse, "Color",
                      SocketValue::color({0.24f, 0.42f, 0.68f})) &&
      graph.set_input(diffuse, "Roughness", SocketValue::floating(0.0f)) &&
      graph.set_input(glossy, "Color",
                      SocketValue::color({0.71f, 0.53f, 0.29f})) &&
      graph.set_input(glossy, "Roughness", SocketValue::floating(0.31f)) &&
      graph.set_property(glossy, "Distribution", SocketValue::string("GGX")) &&
      graph.connect({.node = diffuse, .socket = "Closure"}, root, "A") &&
      graph.connect({.node = glossy, .socket = "Closure"}, root, "B");
  if (!configured) {
    throw std::runtime_error{"failed to configure standalone caustics graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = root, .socket = "Closure"});
  return graph;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  ShaderCompiler compiler{make_core_node_registry()};
  auto shader = compiler.compile(make_diffuse_glossy_graph());
  if (!shader.ok()) {
    throw std::runtime_error{"failed to compile standalone caustics graph"};
  }
  auto program = compile_surface_program(*shader.program);
  if (!program.ok()) {
    throw std::runtime_error{"failed to lower standalone caustics graph"};
  }

  SurfaceDispatch surfaces;
  const auto tag = surfaces.create<GraphSurface>(program.program);
  const auto parameters = parameter_data(*program.program);

  constexpr auto trace_records = 8u;
  constexpr auto collection_records = 4u;
  constexpr auto record_count = trace_records + collection_records;
  Kernel1D evaluate = [&](BufferFloat4 parameter_buffer,
                          BufferFloat4 output) noexcept {
    ParameterShaderServices services{parameter_buffer};
    const auto point = make_surface_point();
    const auto write_trace = [&](UInt record,
                                 const SurfaceClosureTrace &trace) noexcept {
      output.write(record,
                   make_float4(cast<float>(trace.count),
                               cast<float>(trace.type), trace.sample_weight,
                               select(0.0f, 1.0f, trace.valid)));
      output.write(record + 1u,
                   make_float4(trace.weight, cast<float>(trace.runtime_flags)));
    };

    write_trace(
        0u, surfaces.closure_trace(UInt{tag}, services, point, 0u, true, true));
    write_trace(
        2u, surfaces.closure_trace(UInt{tag}, services, point, 1u, true, true));
    write_trace(4u, surfaces.closure_trace(UInt{tag}, services, point, 0u,
                                           false, true));
    write_trace(6u, surfaces.closure_trace(UInt{tag}, services, point, 1u,
                                           false, true));

    SurfaceClosureSet enabled{4u};
    static_cast<void>(surfaces.collect_closures(UInt{tag}, services, point,
                                                true, true, enabled));
    SurfaceClosureSet disabled{4u};
    static_cast<void>(surfaces.collect_closures(UInt{tag}, services, point,
                                                false, true, disabled));
    const auto enabled_glossy = enabled.entry(1u);
    const auto disabled_diffuse = disabled.entry(0u);
    const auto disabled_absent = disabled.entry(1u);
    output.write(8u, make_float4(cast<float>(enabled.count()),
                                 cast<float>(enabled_glossy.kind),
                                 enabled_glossy.allocation_weight,
                                 enabled_glossy.sample_weight));
    output.write(9u, make_float4(cast<float>(disabled.count()),
                                 cast<float>(disabled_diffuse.kind),
                                 disabled_diffuse.allocation_weight,
                                 disabled_diffuse.sample_weight));
    output.write(10u, make_float4(cast<float>(disabled_absent.kind),
                                  disabled_absent.allocation_weight,
                                  disabled_absent.sample_weight, 0.0f));
    output.write(11u, make_float4(disabled_diffuse.weight, 0.0f));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto parameter_buffer =
      device.create_buffer<luisa::float4>(parameters.size());
  auto output_buffer = device.create_buffer<luisa::float4>(record_count);
  auto kernel = device.compile(evaluate);
  std::array<luisa::float4, record_count> actual{};
  stream << parameter_buffer.copy_from(luisa::span{parameters})
         << kernel(parameter_buffer, output_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr auto none = static_cast<float>(cycles_closure::type_none);
  constexpr auto diffuse = static_cast<float>(cycles_closure::type_diffuse);
  constexpr auto glossy =
      static_cast<float>(cycles_closure::type_microfacet_ggx);
  const auto traces_match =
      approximately_equal(actual[0u].x, 2.0f) &&
      approximately_equal(actual[2u].x, 2.0f) &&
      approximately_equal(actual[0u].y, diffuse) &&
      approximately_equal(actual[2u].y, glossy) &&
      approximately_equal(actual[0u].w, 1.0f) &&
      approximately_equal(actual[2u].w, 1.0f) &&
      approximately_equal(actual[4u].x, 1.0f) &&
      approximately_equal(actual[4u].y, diffuse) &&
      approximately_equal(actual[4u].w, 1.0f) &&
      approximately_equal(actual[6u], luisa::float4{1.0f, none, 0.0f, 0.0f});
  const auto collections_match =
      approximately_equal(actual[8u].x, 2.0f) &&
      approximately_equal(actual[8u].y,
                          static_cast<float>(SurfaceClosureKind::glossy)) &&
      actual[8u].z > 0.0f && actual[8u].w > 0.0f &&
      approximately_equal(actual[9u].x, 1.0f) &&
      approximately_equal(actual[9u].y,
                          static_cast<float>(SurfaceClosureKind::diffuse)) &&
      actual[9u].z > 0.0f && actual[9u].w > 0.0f &&
      approximately_equal(actual[10u], luisa::float4{}) &&
      approximately_equal(actual[11u],
                          luisa::float4{0.24f, 0.42f, 0.68f, 0.0f});
  if (!traces_match || !collections_match) {
    std::cerr << "standalone reflective-caustics allocation regression "
              << "failed on " << backend << '\n';
    for (auto index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": {" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "}\n";
    }
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
