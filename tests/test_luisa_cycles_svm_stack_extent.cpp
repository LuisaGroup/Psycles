#include "cycles_svm_stack_extent_test_support.h"
#include "path_tracer_cycles_svm_light.h"
#include "path_tracer_cycles_svm_shadow.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace luisa::compute;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

ShaderGraph emission_graph(bool geometry_inputs) {
  ShaderGraph graph;
  const auto emission = graph.add_node(node_type::emission);
  if (geometry_inputs) {
    const auto geometry = graph.add_node(node_type::geometry);
    const auto position = graph.add_node(node_type::point_to_vector);
    const auto normal = graph.add_node(node_type::normal_to_vector);
    const auto math = graph.add_node(node_type::vector_math);
    const auto color = graph.add_node(node_type::vector_to_color);
    require(graph.connect({geometry, "Position"}, position, "Point") &&
                graph.connect({geometry, "Normal"}, normal, "Normal") &&
                graph.connect({position, "Vector"}, math, "A") &&
                graph.connect({normal, "Vector"}, math, "B") &&
                graph.set_property(math, "Operation",
                                   SocketValue::string("CROSS_PRODUCT")) &&
                graph.connect({math, "Vector"}, color, "Vector") &&
                graph.connect({color, "Color"}, emission, "Color"),
            "cannot connect geometry-dependent graph");
  }
  graph.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});
  return graph;
}

SceneSnapshot snapshot(bool second_shader) {
  SceneSnapshot scene;
  scene.materials.emplace(MaterialId{1u}, MaterialDesc{
      .name = "Constant emission", .shader = emission_graph(false),
      .cycles_shader_index = 5u});
  TriangleMeshDesc mesh;
  mesh.positions = {{-1.0f, -1.0f, -2.0f}, {1.0f, -1.0f, -2.0f},
                     {0.0f, 1.0f, -2.0f}};
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {MaterialId{1u}};
  mesh.triangle_material_slots = {0u};
  if (second_shader) {
    scene.materials.emplace(MaterialId{2u}, MaterialDesc{
        .name = "Geometry cross product", .shader = emission_graph(true),
        .cycles_shader_index = 8u});
    mesh.material_slots.emplace_back(MaterialId{2u});
    // This triangle is behind the camera. Its material must still contribute
    // to the static scene bound; no camera ray or pre-render is involved.
    mesh.positions.insert(mesh.positions.end(),
                          {{-1.0f, -1.0f, 2.0f}, {1.0f, -1.0f, 2.0f},
                           {0.0f, 1.0f, 2.0f}});
    mesh.triangles.push_back({3u, 4u, 5u});
    mesh.triangle_material_slots.emplace_back(1u);
  }
  scene.geometries.emplace(GeometryId{3u}, std::move(mesh));
  scene.instances.emplace(InstanceId{4u}, InstanceDesc{
      .geometry = GeometryId{3u}, .cycles_object_index = 0u});
  scene.cameras.emplace(CameraId{5u}, CameraDesc{});
  scene.active_camera = CameraId{5u};
  return scene;
}

std::size_t verify(LuisaPathTracerBackend &renderer, bool second_shader) {
  const auto compilation = renderer.compile_scene(snapshot(second_shader));
  for (const auto &diagnostic : compilation.diagnostics) {
    require(compilation.ok(), diagnostic.message);
  }
  require(compilation.ok(), "static stack scene failed to compile");
  const auto *compiled =
      dynamic_cast<const LuisaCompiledScene *>(compilation.scene.get());
  require(compiled != nullptr && compiled->data()->cycles_svm != nullptr,
          "scene compilation did not produce a native SVM image");
  const auto &scene = compiled->data();
  const auto &table = scene->cycles_svm->compilation.table;
  const auto extent = std::max<std::size_t>(1u, table.peak_stack_usage);
  require(extent < SVM_STACK_SIZE, "fixture should not require the full stack");
  if (second_shader) {
    require(scene->cycles_svm->material_shader_indices.contains(MaterialId{2u}) &&
                table.node_types_used[abi::NODE_VECTOR_MATH],
            "off-camera material was excluded from static analysis");
  }

  // Do not write peak_stack_usage, evaluate a shader, dispatch a render, or
  // read back stack usage. Record the real auxiliary entrypoints straight
  // from the freshly compiled scene and inspect their pre-optimization AST.
  const auto shadow = make_cycles_svm_shadow_surface_callable(scene);
  psycles::test_support::require_svm_stack_extent(shadow.function(), extent);
  const auto light = make_cycles_svm_light_emission_component(
      scene, scene->camera.projection, true, true);
  const Kernel1D<Buffer<luisa::float3>> light_kernel =
      [light](BufferFloat3 output) noexcept {
        Var<DirectLightTaskCall> task;
        Var<RenderKernelParameters> parameters;
        output.write(dispatch_x(), light->evaluate(task, parameters));
      };
  psycles::test_support::require_svm_stack_extent(
      light_kernel.function()->function(), extent);
  std::cout << "Compiled scene static stack extent: " << extent << '\n';
  return extent;
}
} // namespace

int main(int argc, char **argv) {
  try {
#if defined(_WIN32)
    _putenv_s("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1");
#else
    setenv("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1", 1);
#endif
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    LuisaPathTracerBackend renderer{Device{device.impl_shared()}};
    const auto simple = verify(renderer, false);
    const auto with_second = verify(renderer, true);
    require(with_second > simple,
            "array extent did not follow the larger statically used shader");
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
