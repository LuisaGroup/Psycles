#include "path_tracer_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <luisa/runtime/context.h>

namespace {
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

SceneSnapshot snapshot(bool folded_principled) {
  SceneSnapshot scene;
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf);
  const auto emission = graph.add_node(node_type::emission);
  const auto add = graph.add_node(node_type::add_closure);
  require(graph.connect({diffuse, "Closure"}, add, "A") &&
              graph.connect({emission, "Closure"}, add, "B"),
          "cannot connect mixed emission graph");
  auto root = OutputRef{add, "Closure"};
  if (folded_principled) {
    const auto principled = graph.add_node(node_type::principled_bsdf);
    const auto mix = graph.add_node(node_type::mix_closure);
    require(graph.connect(root, mix, "A") &&
                graph.connect({principled, "Closure"}, mix, "B") &&
                graph.set_input(mix, "Factor", SocketValue::floating(0.0f)),
            "cannot connect discarded Principled branch");
    root = {mix, "Closure"};
  }
  graph.set_root(ShaderDomain::surface, root);
  scene.materials.emplace(MaterialId{1u}, MaterialDesc{
      .name = "Diffuse plus Emission", .shader = std::move(graph),
      .cycles_shader_index = 5u});
  TriangleMeshDesc mesh;
  mesh.positions = {{-1.0f, -1.0f, -2.0f}, {1.0f, -1.0f, -2.0f},
                     {0.0f, 1.0f, -2.0f}};
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {MaterialId{1u}};
  mesh.triangle_material_slots = {0u};
  scene.geometries.emplace(GeometryId{2u}, std::move(mesh));
  scene.instances.emplace(InstanceId{3u}, InstanceDesc{
      .geometry = GeometryId{2u}, .cycles_object_index = 0u});
  scene.cameras.emplace(CameraId{4u}, CameraDesc{});
  scene.active_camera = CameraId{4u};
  return scene;
}
} // namespace

int main(int argc, char **argv) {
  try {
#if defined(_WIN32)
    _putenv_s("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1");
    _putenv_s("PSYCLES_DISABLE_SHADER_CACHE", "1");
#else
    setenv("PSYCLES_NATIVE_CYCLES_SVM_SURFACE", "1", 1);
    setenv("PSYCLES_DISABLE_SHADER_CACHE", "1", 1);
#endif
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    LuisaPathTracerBackend renderer{luisa::compute::Device{device.impl_shared()}};
    for (const auto folded : {false, true}) {
      auto result = renderer.compile_scene(snapshot(folded));
      for (const auto &diagnostic : result.diagnostics) {
        require(result.ok(), diagnostic.message);
      }
      require(result.ok(), "native closure-budget scene did not compile");
      const auto *scene = dynamic_cast<const LuisaCompiledScene *>(result.scene.get());
      require(scene != nullptr && scene->data()->native_cycles_svm_surface,
              "scene compilation did not use native SVM");
      const auto &data = *scene->data();
      require(data.cycles_svm->compilation.shader_metadata[5u].num_closures == 1u,
              "finalized Diffuse plus Emission graph must allocate one closure");
      // Original Cycles HIP observer: the graph contributes one, but the
      // always-referenced default Principled graph sets the scene budget to 12.
      // The old SurfaceProgram estimator gives 2, or 14 for the folded graph.
      require(data.volume_metadata.closure_allocation_budget == 12u,
              "native renderer retained the legacy closure allocation budget: " +
                  std::to_string(data.volume_metadata.closure_allocation_budget));
      require(data.volume_metadata.closure_allocation_budget ==
                  data.cycles_svm->compilation.max_closures,
              "native closure allocation is not owned by the Cycles graph image");
    }
    std::cout << "Native Cycles SVM scene closure budget tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
