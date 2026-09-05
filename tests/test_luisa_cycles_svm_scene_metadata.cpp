#include "path_tracer_analytic_light_scene.h"
#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_light_sampling_scene.h"
#include "path_tracer_mesh_light_scene.h"

#include <psycles/compiler/core_nodes.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;

constexpr MaterialId emitter{1u};
constexpr MaterialId dark{2u};
constexpr GeometryId geometry{3u};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

ShaderGraph emission_graph(Vec3f color, float strength) {
  ShaderGraph graph;
  const auto node = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(node, "Color", SocketValue::color(color)) &&
          graph.set_input(node, "Strength", SocketValue::floating(strength)),
      "cannot configure emission metadata graph");
  graph.set_root(ShaderDomain::surface, OutputRef{node, "Closure"});
  return graph;
}

SceneSnapshot make_snapshot() {
  SceneSnapshot snapshot;
  snapshot.materials.emplace(
      emitter, MaterialDesc{.name = "Native emitter",
                            .shader = emission_graph({0.25f, 0.5f, 1.0f}, 4.0f),
                            .emission_sampling = EmissionSampling::front,
                            .cycles_shader_index = 5u});
  snapshot.materials.emplace(
      dark, MaterialDesc{.name = "Native dark",
                         .shader = emission_graph({1.0f, 1.0f, 1.0f}, 0.0f),
                         .cycles_shader_index = 7u});
  TriangleMeshDesc mesh;
  mesh.positions = {{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}};
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {emitter};
  mesh.triangle_material_slots = {0u};
  snapshot.geometries.emplace(geometry, std::move(mesh));
  snapshot.instances.emplace(InstanceId{4u},
                             InstanceDesc{.name = "Emitter instance",
                                          .geometry = geometry,
                                          .cycles_object_index = 0u});
  snapshot.lights.emplace(LightId{5u}, LightDesc{.name = "Emitting point",
                                                 .type = LightType::point,
                                                 .shader = emitter,
                                                 .cycles_object_index = 1u});
  snapshot.lights.emplace(LightId{6u}, LightDesc{.name = "Dark point",
                                                 .type = LightType::point,
                                                 .shader = dark,
                                                 .cycles_object_index = 2u});
  return snapshot;
}

std::shared_ptr<LuisaSceneData> make_scene(Device &device,
                                           const SceneSnapshot &snapshot) {
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  scene->native_cycles_svm_surface = true;
  std::string diagnostic;
  scene->cycles_svm = build_cycles_svm_runtime(scene, snapshot, diagnostic);
  require(scene->cycles_svm != nullptr, diagnostic);
  scene->material_bindings.emplace(
      emitter, make_cycles_svm_material_binding(*scene, emitter, 0u, 13u, 2u));
  scene->material_bindings.emplace(
      dark, make_cycles_svm_material_binding(*scene, dark, 1u, 29u, 3u));
  return scene;
}

void install_stale_library(LuisaSceneData &scene,
                           const SceneSnapshot &snapshot) {
  auto stale = snapshot;
  stale.materials.at(emitter).shader = emission_graph({1.0f}, 0.0f);
  stale.materials.at(dark).shader = emission_graph({1.0f}, 8.0f);
  const ShaderCompiler compiler{make_core_node_registry()};
  require(scene.materials.update(stale, compiler).committed,
          "cannot construct stale legacy metadata counterexample");
}

void verify_lights(const SceneSnapshot &snapshot, const LuisaSceneData &scene) {
  const auto lights = AnalyticLightSceneComponent{}.build(snapshot, scene);
  require(lights.ok(), lights.diagnostic);
  require(lights.regular_count == 1u &&
              lights.device_lights.front().cycles_object_index == 1u,
          "native shader contribution was replaced by legacy light metadata");
  require(lights.regular_shader_emission_estimates.front() ==
              Vec3f{1.0f, 2.0f, 4.0f},
          "analytic light estimate does not match the native Cycles shader");
}

void verify_mesh(const SceneSnapshot &snapshot, const LuisaSceneData &scene) {
  require(collect_emission_sampling_materials(scene) ==
              std::set<MaterialId>{emitter},
          "native emitter membership depends on legacy material presence");
  std::array<GeometryUpload, 1u> uploads;
  uploads[0u].positions = {luisa::make_float3(0.0f, 0.0f, 0.0f),
                           luisa::make_float3(2.0f, 0.0f, 0.0f),
                           luisa::make_float3(0.0f, 2.0f, 0.0f)};
  const std::array triangles{EmissiveTriangleGpu{
      .instance_index = 0u,
      .geometry_index = 0u,
      .primitive_index = 0u,
      .emission_sampling = static_cast<std::uint32_t>(EmissionSampling::front),
      .cycles_primitive_index = 0u,
      .cycles_object_index = 0u}};
  const auto tree =
      MeshLightTreeSceneComponent{}.build(snapshot, scene, uploads, triangles);
  require(tree.ok(), tree.diagnostic);
  require(tree.subtrees.size() == 1u && tree.mesh_emitters.size() == 1u,
          "native mesh emitter disappeared from the light tree");
  // Cycles scene/light_tree.cpp: area *
  // average(abs(shader->emission_estimate)). This is the same external-source
  // analytic case as test_luisa_light_tree, now with no legacy program and then
  // a deliberately contradictory cache.
  const auto energy = tree.mesh_emitters.front().emitter.measure.energy;
  require(std::abs(energy - 2.0f * (1.0f + 2.0f + 4.0f) / 3.0f) < 1.0e-5f,
          "mesh-light energy does not match the native Cycles shader");
}

void verify_metadata(Device &device) {
  auto snapshot = make_snapshot();
  auto scene = make_scene(device, snapshot);
  const auto &table = scene->cycles_svm->compilation;
  require(table.shader_metadata.size() == table.table.shader_count &&
              table.shader_metadata.size() == 8u,
          "host shader metadata lost the dense Cycles index domain");
  require(!table.shader_metadata[6u].has_surface &&
              table.shader_metadata[6u].emission_estimate == Vec3f{},
          "an inert source hole acquired another shader's metadata");
  const auto &metadata = cycles_svm_material_metadata(*scene, emitter);
  require(metadata.has_surface && metadata.emission_is_constant &&
              !metadata.has_surface_spatial_varying &&
              metadata.emission_estimate == Vec3f{1.0f, 2.0f, 4.0f},
          "constant-emission host metadata differs from Cycles");
  const auto &binding = scene->material_bindings.at(emitter);
  require(
      binding.surface_tag == 0u && binding.parameter_block == 13u &&
          binding.material_identity == 2u && binding.cycles_shader_index == 5u,
      "native metadata changed renderer storage addresses or source identity");
  require(binding.emission_sampling == EmissionSampling::front &&
              (binding.flags & material_flag_may_emit) != 0u &&
              (binding.flags & material_flag_constant_emission) != 0u,
          "native material binding dropped emitter flags");

  // Cycles Shader::estimate_emission uses a strict AUTO threshold. Disabling
  // MIS does not remove endpoint emission, including the exact threshold.
  for (const auto strength : {0.0f, 0.5f, 0.75f}) {
    snapshot.materials.at(emitter).shader = emission_graph({1.0f}, strength);
    snapshot.materials.at(emitter).emission_sampling =
        EmissionSampling::automatic;
    scene = make_scene(device, snapshot);
    const auto &value = scene->material_bindings.at(emitter);
    require(value.emission_sampling == (strength > 0.5f
                                            ? EmissionSampling::front_back
                                            : EmissionSampling::none) &&
                ((value.flags & material_flag_may_emit) != 0u) ==
                    (strength != 0.0f),
            "AUTO emitter threshold diverged from endpoint contribution");
  }
  for (const auto sampling :
       {EmissionSampling::none, EmissionSampling::front, EmissionSampling::back,
        EmissionSampling::front_back}) {
    snapshot.materials.at(emitter).emission_sampling = sampling;
    scene = make_scene(device, snapshot);
    require(
        scene->material_bindings.at(emitter).emission_sampling == sampling,
        "native MIS front/back flags did not round-trip to geometry binding");
  }

  ShaderGraph transparent;
  const auto transparent_node =
      transparent.add_node(node_type::transparent_bsdf, "Transparent");
  transparent.set_root(ShaderDomain::surface,
                       OutputRef{transparent_node, "Closure"});
  snapshot.materials.at(emitter).shader = std::move(transparent);
  for (const auto enabled : {false, true}) {
    snapshot.materials.at(emitter).use_transparent_shadow = enabled;
    snapshot.materials.at(emitter).use_bump_map_correction = enabled;
    scene = make_scene(device, snapshot);
    const auto flags = scene->material_bindings.at(emitter).flags;
    require(
        ((flags & material_flag_has_transparent_shadow) != 0u) == enabled &&
            ((flags & material_flag_use_bump_map_correction) != 0u) ==
                enabled &&
            cycles_svm_material_metadata(*scene, emitter)
                .has_surface_transparent,
        "native shadow policy changed the underlying transparent closure fact");
  }

  ShaderGraph volume;
  const auto absorption =
      volume.add_node(node_type::volume_absorption, "Volume");
  volume.set_root(ShaderDomain::volume, OutputRef{absorption, "Volume"});
  snapshot.materials.at(emitter).shader = std::move(volume);
  snapshot.materials.at(emitter).use_transparent_shadow = false;
  for (const auto sampling :
       {VolumeSampling::distance, VolumeSampling::equiangular,
        VolumeSampling::multiple_importance}) {
    snapshot.materials.at(emitter).volume_sampling = sampling;
    scene = make_scene(device, snapshot);
    const auto &value = scene->material_bindings.at(emitter);
    require(value.volume_sampling == sampling &&
                (value.flags & material_flag_has_volume) != 0u &&
                (value.flags & material_flag_has_transparent_shadow) != 0u,
            "native volume binding lost sampling mode or shadow scheduling");
  }

  ShaderGraph camera;
  const auto data = camera.add_node(node_type::camera_data, "Camera Data");
  const auto emission = camera.add_node(node_type::emission, "Camera emission");
  require(camera.set_input(emission, "Color",
                           SocketValue::color({0.25f, 0.5f, 1.0f})) &&
              camera.connect({data, "View Distance"}, emission, "Strength"),
          "cannot construct dynamic Camera Data metadata probe");
  camera.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});
  snapshot.materials.at(emitter).shader = std::move(camera);
  scene = make_scene(device, snapshot);
  const auto &dynamic = cycles_svm_material_metadata(*scene, emitter);
  require(dynamic.has_surface && dynamic.has_surface_spatial_varying &&
              !dynamic.emission_is_constant &&
              dynamic.emission_estimate == Vec3f{0.25f, 0.5f, 1.0f},
          "Camera Data's native host facts require legacy lowering");
  require((scene->material_bindings.at(emitter).flags &
           material_flag_constant_emission) == 0u,
          "dynamic Camera Data incorrectly permits constant-emission shortcut");
  const auto lights = AnalyticLightSceneComponent{}.build(snapshot, *scene);
  require(
      lights.ok() && lights.regular_count == 1u &&
          lights.regular_shader_emission_estimates.front() ==
              dynamic.emission_estimate,
      "native-only Camera Data metadata did not reach analytic light packing");
}
} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    const auto snapshot = make_snapshot();
    bool passed = true;
    const auto check = [&](std::string_view name, auto &&test) {
      try {
        test();
        std::cout << name << ": passed\n";
      } catch (const std::exception &error) {
        std::cerr << name << ": " << error.what() << '\n';
        passed = false;
      }
    };
    check("analytic lights without legacy lowering",
          [&] { verify_lights(snapshot, *make_scene(device, snapshot)); });
    check("analytic lights ignore stale legacy metadata", [&] {
      auto scene = make_scene(device, snapshot);
      install_stale_library(*scene, snapshot);
      verify_lights(snapshot, *scene);
    });
    check("mesh light metadata without legacy lowering",
          [&] { verify_mesh(snapshot, *make_scene(device, snapshot)); });
    check("mesh light metadata ignores stale legacy metadata", [&] {
      auto scene = make_scene(device, snapshot);
      install_stale_library(*scene, snapshot);
      verify_mesh(snapshot, *scene);
    });
    check("native graph facts and geometry binding flags",
          [&] { verify_metadata(device); });
    check("missing native state is diagnosed without cache fallback", [&] {
      auto scene = make_scene(device, snapshot);
      install_stale_library(*scene, snapshot);
      scene->cycles_svm.reset();
      const auto lights = AnalyticLightSceneComponent{}.build(snapshot, *scene);
      require(!lights.ok() && !lights.diagnostic.empty(),
              "missing native shader table silently fell back to old metadata");
    });
    check("missing native material is diagnosed without cache fallback", [&] {
      auto scene = make_scene(device, snapshot);
      install_stale_library(*scene, snapshot);
      scene->cycles_svm->material_shader_indices.erase(emitter);
      const auto lights = AnalyticLightSceneComponent{}.build(snapshot, *scene);
      require(!lights.ok() && !lights.diagnostic.empty(),
              "missing native material silently fell back to old metadata");
    });
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
