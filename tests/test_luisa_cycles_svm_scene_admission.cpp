#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_surfaces.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/path_tracer.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

SceneSnapshot object_index_scene() {
  // Minimum Classroom admission failure: a valid native Object Info output
  // fed into ordinary Math. This test exercises the whole scene loader, not
  // just build_cycles_svm_runtime (which already accepted this graph).
  ShaderGraph graph;
  const auto info = graph.add_node(node_type::object_info, "Object Info");
  const auto math = graph.add_node(node_type::math, "Index plus offset");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_property(math, "Operation", SocketValue::string("ADD")),
          "cannot set Math operation");
  require(graph.set_input(math, "B", SocketValue::floating(0.25f)),
          "cannot set Math input");
  require(graph.connect({info, "ObjectIndex"}, math, "A"),
          "cannot connect Object Index");
  require(graph.connect({math, "Value"}, emission, "Strength"),
          "cannot connect emission strength");
  graph.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});

  SceneSnapshot snapshot;
  constexpr MaterialId material{1u};
  constexpr GeometryId geometry{2u};
  constexpr CameraId camera{4u};
  snapshot.materials.emplace(material, MaterialDesc{
      .name = "Object Index admission", .shader = std::move(graph)});
  TriangleMeshDesc mesh;
  mesh.name = "Admission triangle";
  mesh.positions = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f}};
  mesh.normals.values.assign(3u, psycles::Vec3f{0.0f, 0.0f, 1.0f});
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {material};
  mesh.triangle_material_slots = {0u};
  mesh.triangle_smooth = {1u};
  snapshot.geometries.emplace(geometry, std::move(mesh));
  snapshot.instances.emplace(InstanceId{3u}, InstanceDesc{
      .name = "Admission object", .geometry = geometry});
  snapshot.cameras.emplace(camera, CameraDesc{.name = "Admission camera"});
  snapshot.active_camera = camera;
  snapshot.world_sampling = WorldSampling::none;
  return snapshot;
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    LuisaPathTracerBackend renderer{context.create_device(backend), {}};
    const auto snapshot = object_index_scene();
    const auto result = renderer.compile_scene(snapshot);
    for (const auto &diagnostic : result.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    require(result.ok(), "native Object Index was rejected by the scene loader");
    const auto *compiled = dynamic_cast<const LuisaCompiledScene *>(result.scene.get());
    require(compiled != nullptr, "missing native compiled scene");
    const auto &scene = *compiled->data();
    require(scene.cycles_svm && scene.cycles_svm->geometry && scene.cycles_svm->objects,
            "native scene image was not finalized");
    require(scene.cycles_svm->compilation.table.node_types_used[abi::NODE_INFO_OB_INDEX],
            "native Object Index opcode was omitted");
    require(scene.materials.materials().empty() && scene.surfaces.size() == 0u &&
                !scene.surface_values,
            "surface admission still constructs legacy material programs");
    const auto consumers = make_surface_callables(compiled->data());
    require(consumers.population && !consumers.preparation.function_builder() &&
                !consumers.evaluate_light.function_builder() &&
                !consumers.constant_emission.function_builder() &&
                !consumers.emission.function_builder() &&
                !consumers.sample.function_builder() &&
                !consumers.closure_trace.function_builder() &&
                !consumers.sample_trace.function_builder() &&
                !consumers.bssrdf_normal.function_builder(),
            "native scene constructed an unused legacy material callable");
    std::cout << "Native scene admission passed on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
