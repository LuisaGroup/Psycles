#include "../src/luisa/path_tracer_scene_geometry.h"
#include "../src/luisa/path_tracer_subsurface_scene.h"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using psycles::contract::GeometryId;
using psycles::contract::InstanceDesc;
using psycles::contract::InstanceId;
using psycles::contract::MaterialId;
using psycles::contract::SceneSnapshot;
using psycles::contract::TriangleMeshDesc;
using psycles::luisa_backend::detail::build_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::collect_reachable_surface_materials;
using psycles::luisa_backend::detail::
    collect_triangle_instances_with_surface_materials;
using psycles::luisa_backend::detail::SubsurfaceSceneComponent;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_reachable_surface_materials() {
  SceneSnapshot scene;
  scene.geometries.emplace(
      GeometryId{1u},
      TriangleMeshDesc{
          .name = "instanced mesh",
          .triangles = {
              {0u, 1u, 2u},
              {0u, 1u, 2u},
              {0u, 1u, 2u}},
          .material_slots = {
              MaterialId{1u}, MaterialId{2u}, MaterialId{3u}},
          .triangle_material_slots = {0u, 2u, 99u}});
  scene.geometries.emplace(
      GeometryId{2u},
      TriangleMeshDesc{
          .name = "uninstanced mesh",
          .triangles = {{0u, 1u, 2u}},
          .material_slots = {MaterialId{5u}}});
  scene.geometries.emplace(
      GeometryId{4u},
      TriangleMeshDesc{
          .name = "empty mesh",
          .material_slots = {MaterialId{10u}}});
  scene.curve_geometries.emplace(
      GeometryId{3u},
      psycles::contract::CurveGeometryDesc{
          .name = "instanced curves",
          .keys = {
              {0.0f, 0.0f, 0.0f, 0.1f},
              {1.0f, 0.0f, 0.0f, 0.1f},
              {0.0f, 1.0f, 0.0f, 0.1f},
              {1.0f, 1.0f, 0.0f, 0.1f},
              {0.0f, 2.0f, 0.0f, 0.1f}},
          .curve_first_key = {0u, 2u, 4u},
          .material_slots = {
              MaterialId{6u}, MaterialId{7u}, MaterialId{9u}},
          .curve_material_slots = {0u, 1u, 2u}});
  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{
          .name = "mesh instance",
          .geometry = GeometryId{1u},
          .material_overrides = {MaterialId{4u}}});
  scene.instances.emplace(
      InstanceId{2u},
      InstanceDesc{
          .name = "curve instance",
          .geometry = GeometryId{3u},
          .material_overrides = {MaterialId{8u}}});
  scene.instances.emplace(
      InstanceId{3u},
      InstanceDesc{
          .name = "empty instance",
          .geometry = GeometryId{4u}});

  const auto reachable = collect_reachable_surface_materials(scene);
  require(
      reachable ==
          std::set<MaterialId>{
              MaterialId{3u}, MaterialId{4u},
              MaterialId{7u}, MaterialId{8u}},
      "surface material reachability diverged from primitive resolution");
}

void test_triangle_instances_with_surface_materials() {
  SceneSnapshot scene;
  scene.geometries.emplace(
      GeometryId{1u},
      TriangleMeshDesc{
          .name = "slot-resolved mesh",
          .triangles = {
              {0u, 1u, 2u}, {0u, 1u, 2u}, {0u, 1u, 2u}},
          .material_slots = {
              MaterialId{1u}, MaterialId{2u}, MaterialId{3u}},
          .triangle_material_slots = {0u, 2u, 99u}});
  scene.geometries.emplace(
      GeometryId{2u},
      TriangleMeshDesc{
          .name = "empty target mesh",
          .material_slots = {MaterialId{7u}}});
  scene.curve_geometries.emplace(
      GeometryId{3u},
      psycles::contract::CurveGeometryDesc{
          .name = "target curve",
          .keys = {{0.0f, 0.0f, 0.0f, 0.1f},
                   {1.0f, 0.0f, 0.0f, 0.1f}},
          .curve_first_key = {0u},
          .material_slots = {MaterialId{7u}},
          .curve_material_slots = {0u}});
  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "override target",
                   .geometry = GeometryId{1u},
                   .material_overrides = {MaterialId{7u}}});
  scene.instances.emplace(
      InstanceId{2u},
      InstanceDesc{.name = "clamped geometry target",
                   .geometry = GeometryId{1u},
                   .material_overrides = {MaterialId{8u}}});
  scene.instances.emplace(
      InstanceId{3u},
      InstanceDesc{.name = "unused override target",
                   .geometry = GeometryId{1u},
                   .material_overrides = {
                       MaterialId{8u}, MaterialId{7u}}});
  scene.instances.emplace(
      InstanceId{4u},
      InstanceDesc{.name = "empty target",
                   .geometry = GeometryId{2u}});
  scene.instances.emplace(
      InstanceId{5u},
      InstanceDesc{.name = "curve target",
                   .geometry = GeometryId{3u}});

  const auto instances = collect_triangle_instances_with_surface_materials(
      scene, {MaterialId{7u}});
  require(instances == std::vector<std::uint32_t>{0u},
          "BSSRDF object domain diverged from primitive material resolution");
  const auto plan = SubsurfaceSceneComponent{}.plan(
      scene, {MaterialId{7u}});
  require(plan.triangle_instance_count == 1u && plan.contains(0u) &&
              !plan.contains(1u) && !plan.contains(4u) &&
              !plan.contains(5u),
          "BSSRDF compact-domain mask is not a bijective primary-index image");
}

} // namespace

int main() {
  test_reachable_surface_materials();
  test_triangle_instances_with_surface_materials();
  std::cout << "Cycles instance material-domain tests passed\n";
  return EXIT_SUCCESS;
}
