#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
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
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr std::array<float, 16u> world_to_projector{
    2.0f, -1.0f, 0.25f, 0.0f, 0.5f, 3.0f,  0.0f, 0.0f,
    0.0f, 0.0f,  4.0f,  0.0f, 1.0f, -2.0f, 0.5f, 1.0f};

[[nodiscard]] ShaderGraph coordinate_graph(bool explicit_object) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate,
                     explicit_object ? "Explicit projector" : "Shading object");
  if (explicit_object) {
    Mat4f transform;
    transform.elements = world_to_projector;
    static_cast<void>(graph.set_property(coordinates, "ObjectUseTransform",
                                         SocketValue::boolean(true)));
    static_cast<void>(graph.set_property(coordinates, "ObjectWorldToObject",
                                         SocketValue::transform(transform)));
  }
  const auto conversion =
      graph.add_node(node_type::vector_to_color, "Coordinate vector to color");
  const auto emission =
      graph.add_node(node_type::emission, "Coordinate oracle emission");
  static_cast<void>(graph.connect({.node = coordinates, .socket = "Object"},
                                  conversion, "Vector"));
  static_cast<void>(graph.connect({.node = conversion, .socket = "Color"},
                                  emission, "Color"));
  static_cast<void>(
      graph.set_input(emission, "Strength", SocketValue::floating(1.0f)));
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] std::shared_ptr<const SurfaceProgram>
compile_coordinate_program(ShaderCompiler &compiler, bool explicit_object) {
  const auto shader = compiler.compile(coordinate_graph(explicit_object));
  if (!shader.ok()) {
    throw std::runtime_error{"failed to compile Texture Coordinate graph"};
  }
  auto lowered = compile_surface_program(*shader.program);
  if (!lowered.ok()) {
    throw std::runtime_error{"failed to lower Texture Coordinate graph"};
  }
  const auto expected_operation =
      explicit_object ? ValueOperation::object_position_with_transform
                      : ValueOperation::object_position;
  auto found = false;
  for (const auto &instruction : lowered.program->value_instructions()) {
    if (instruction.operation != expected_operation) {
      continue;
    }
    found = true;
    if (explicit_object &&
        (instruction.static_table.size() != world_to_projector.size() ||
         !std::equal(instruction.static_table.begin(),
                     instruction.static_table.end(),
                     world_to_projector.begin()))) {
      throw std::runtime_error{
          "explicit Object coordinate matrix changed during lowering"};
    }
  }
  if (!found) {
    throw std::runtime_error{
        explicit_object
            ? "explicit Object coordinates lost their affine transform"
            : "implicit Object coordinates changed operation"};
  }
  return std::move(lowered.program);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  ShaderCompiler compiler{make_core_node_registry()};
  const auto explicit_program = compile_coordinate_program(compiler, true);
  const auto implicit_program = compile_coordinate_program(compiler, false);

  SurfaceDispatch surfaces;
  const auto explicit_tag = surfaces.create<GraphSurface>(explicit_program);
  const auto implicit_tag = surfaces.create<GraphSurface>(implicit_program);
  const auto explicit_parameters = parameter_data(*explicit_program);
  const auto implicit_parameters = parameter_data(*implicit_program);

  Kernel1D evaluate = [&](BufferFloat4 explicit_parameter_buffer,
                          BufferFloat4 implicit_parameter_buffer,
                          BufferFloat4 result) noexcept {
    ParameterShaderServices explicit_services{explicit_parameter_buffer};
    ParameterShaderServices implicit_services{implicit_parameter_buffer};
    auto point = make_surface_point();
    point.position = make_float3(4.0f, 5.0f, 6.0f);
    point.object_position = make_float3(-0.3f, 0.4f, 0.8f);
    result.write(
        0u, make_float4(surfaces.emission(UInt{explicit_tag}, explicit_services,
                                          point, make_float3(0.0f, 0.0f, 1.0f),
                                          true),
                        0.0f));
    result.write(
        1u, make_float4(surfaces.emission(UInt{implicit_tag}, implicit_services,
                                          point, make_float3(0.0f, 0.0f, 1.0f),
                                          true),
                        0.0f));
  };

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto explicit_parameter_buffer =
      device.create_buffer<luisa::float4>(explicit_parameters.size());
  auto implicit_parameter_buffer =
      device.create_buffer<luisa::float4>(implicit_parameters.size());
  auto result_buffer = device.create_buffer<luisa::float4>(2u);
  auto kernel = device.compile(evaluate);
  std::array<luisa::float4, 2u> actual{};
  stream
      << explicit_parameter_buffer.copy_from(luisa::span{explicit_parameters})
      << implicit_parameter_buffer.copy_from(luisa::span{implicit_parameters})
      << kernel(explicit_parameter_buffer, implicit_parameter_buffer,
                result_buffer)
             .dispatch(1u)
      << result_buffer.copy_to(luisa::span{actual}) << synchronize();

  const std::array expected{luisa::float4{11.5f, 9.0f, 25.5f, 0.0f},
                            luisa::float4{-0.3f, 0.4f, 0.8f, 0.0f}};
  for (std::size_t index = 0u; index < actual.size(); ++index) {
    if (!approximately_equal(actual[index], expected[index])) {
      std::cerr << "Texture Coordinate Object result " << index
                << " mismatch on " << backend << ": got {" << actual[index].x
                << ", " << actual[index].y << ", " << actual[index].z << "}\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
