#include "../src/luisa/path_tracer_instance_support.h"
#include "../src/luisa/path_tracer_internal.h"
#include "../src/luisa/path_tracer_scene_geometry.h"
#include "../src/luisa/path_tracer_subsurface_scene.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using psycles::contract::GeometryId;
using psycles::contract::InstanceDesc;
using psycles::contract::InstanceId;
using psycles::contract::MaterialId;
using psycles::contract::SceneSnapshot;
using psycles::contract::TriangleMeshDesc;
using psycles::luisa_backend::detail::classify_cycles_final_triangle_supports;
using psycles::luisa_backend::detail::build_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::collect_reachable_surface_materials;
using psycles::luisa_backend::detail::
    collect_triangle_instances_with_surface_materials;
using psycles::luisa_backend::detail::finalize_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::GeometryUpload;
using psycles::luisa_backend::detail::make_cycles_geometry_support_view;
using psycles::luisa_backend::detail::Triangle;
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

[[nodiscard]] GeometryUpload make_support() {
  GeometryUpload upload;
  upload.positions = {luisa::make_float3(0.0f, 0.0f, 0.0f),
                      luisa::make_float3(1.0f, 0.0f, 0.0f),
                      luisa::make_float3(0.0f, 1.0f, 0.0f)};
  upload.triangles = {Triangle{0u, 1u, 2u}};
  return upload;
}

void test_final_support_equivalence() {
  SceneSnapshot scene;
  for (auto id = 1u; id <= 4u; ++id) {
    scene.geometries.emplace(
        GeometryId{id},
        TriangleMeshDesc{.name = "support-" + std::to_string(id)});
  }
  std::vector<GeometryUpload> uploads;
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads.emplace_back(make_support());
  uploads[2u].positions[0u].z = std::nextafter(0.0f, 1.0f);
  uploads[3u].triangles[0u] = Triangle{0u, 2u, 1u};
  const std::map<GeometryId, std::uint32_t> indices{{GeometryId{1u}, 0u},
                                                    {GeometryId{2u}, 1u},
                                                    {GeometryId{3u}, 2u},
                                                    {GeometryId{4u}, 3u}};

  const auto classes =
      classify_cycles_final_triangle_supports(scene, indices, uploads);
  require(classes.ok(), "final support classification failed");
  require(classes.by_geometry.at(GeometryId{1u}) ==
              classes.by_geometry.at(GeometryId{2u}),
          "bitwise-equal final supports were split");
  require(classes.by_geometry.at(GeometryId{1u}) !=
              classes.by_geometry.at(GeometryId{3u}),
          "one-bit post-displacement position change was grouped");
  require(classes.by_geometry.at(GeometryId{1u}) !=
              classes.by_geometry.at(GeometryId{4u}),
          "triangle-index order change was grouped");

  // Simulate two source-identical meshes whose material/object contexts
  // produce different true-displacement results. Alias identity must follow
  // the final accelerator support, never the source mesh.
  uploads[1u].positions[1u].z = 0.125f;
  const auto displaced_classes =
      classify_cycles_final_triangle_supports(scene, indices, uploads);
  require(displaced_classes.ok(), "displaced support classification failed");
  require(displaced_classes.by_geometry.at(GeometryId{1u}) !=
              displaced_classes.by_geometry.at(GeometryId{2u}),
          "source-identical meshes remained aliased after displacement");
}

void test_exact_world_support_equivalence() {
  using psycles::contract::InstanceDesc;
  using psycles::contract::InstanceId;
  SceneSnapshot scene;
  std::vector<GeometryUpload> uploads;
  std::map<GeometryId, std::uint32_t> geometry_indices;
  for (auto id = 1u; id <= 3u; ++id) {
    scene.geometries.emplace(
        GeometryId{id},
        TriangleMeshDesc{.name = "barbershop-support-" +
                                 std::to_string(id)});
    geometry_indices.emplace(
        GeometryId{id}, static_cast<std::uint32_t>(uploads.size()));
    auto &upload = uploads.emplace_back();
    upload.positions = {
        luisa::make_float3(0.3648039698600769f,
                           0.5588208436965942f, 0.0f),
        luisa::make_float3(0.3118320405483246f,
                           0.5588208436965942f, 0.0f),
        luisa::make_float3(0.31214094161987305f,
                           0.2659621834754944f, 0.0f)};
    upload.triangles = {Triangle{0u, 1u, 2u}};
  }

  psycles::Mat4f object_three;
  object_three.elements = {
      -4.0158649738941676e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, -4.4516873742850294e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 3.258741617202759f,
      -0.0029841959476470947f, 1.0f};
  psycles::Mat4f object_seventy_four;
  object_seventy_four.elements = {
      6.936164709259174e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, 7.688912972980688e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 3.258741617202759f,
      -0.0029841959476470947f, 1.0f};
  auto distinct_world_support = object_seventy_four;
  distinct_world_support.elements[12u] = std::nextafter(
      distinct_world_support.elements[12u],
      std::numeric_limits<float>::infinity());

  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "Cycles object 3",
                   .geometry = GeometryId{1u},
                   .transform = object_three});
  scene.instances.emplace(
      InstanceId{2u},
      InstanceDesc{.name = "Cycles object 74",
                   .geometry = GeometryId{2u},
                   .transform = object_seventy_four});
  scene.instances.emplace(
      InstanceId{3u},
      InstanceDesc{.name = "one-ULP world support change",
                   .geometry = GeometryId{3u},
                   .transform = distinct_world_support});

  const auto support_classes =
      classify_cycles_final_triangle_supports(
          scene, geometry_indices, uploads);
  require(support_classes.ok(),
          "Barbershop support classification failed");
  std::map<
      GeometryId,
      psycles::luisa_backend::detail::CyclesGeometrySupportView>
      final_supports;
  for (const auto &[geometry_id, upload_index] : geometry_indices) {
    const auto &positions = uploads[upload_index].positions;
    const auto &triangles = uploads[upload_index].triangles;
    final_supports.emplace(
        geometry_id,
        make_cycles_geometry_support_view(
            std::span<const luisa::float3>{
                positions.data(), positions.size()},
            std::span<const Triangle>{
                triangles.data(), triangles.size()}));
  }
  auto plan = build_cycles_instance_intersection_plan(scene, {});
  psycles::luisa_backend::detail::CyclesPrimitiveCompletionPlan
      primitive_plan;
  require(finalize_cycles_instance_intersection_plan(
              scene, support_classes.by_geometry,
              final_supports, plan, primitive_plan),
          "Barbershop instance support finalization failed");
  require(object_three.elements[0u] !=
              object_seventy_four.elements[0u],
          "Barbershop transform fixture lost its distinct encodings");
  require(plan[0u].coincident_count == 2u &&
              plan[0u].coincident_next == 1u &&
              plan[1u].coincident_count == 2u &&
              plan[1u].coincident_next == 0u,
          "distinct transforms with exact finite world support were split");
  require(plan[2u].coincident_count == 1u &&
              plan[2u].coincident_next == 2u,
          "one-ULP world support change was grouped");
}

void test_sparse_primitive_world_support_completion() {
  using psycles::contract::InstanceDesc;
  using psycles::contract::InstanceId;
  SceneSnapshot scene;
  std::vector<GeometryUpload> uploads;
  std::map<GeometryId, std::uint32_t> geometry_indices;
  for (auto id = 1u; id <= 2u; ++id) {
    auto geometry = TriangleMeshDesc{
        .name = "Barbershop partial support " + std::to_string(id)};
    scene.geometries.emplace(GeometryId{id}, std::move(geometry));
    geometry_indices.emplace(
        GeometryId{id}, static_cast<std::uint32_t>(uploads.size()));
    auto &upload = uploads.emplace_back();
    upload.positions = {
        luisa::make_float3(-0.373418390750885f,
                           -0.5634331107139587f,
                           -0.005576633382588625f),
        luisa::make_float3(-0.373420774936676f,
                           -0.5620933771133423f,
                           -0.0088327182456851f),
        luisa::make_float3(-0.3707645535469055f,
                           -0.5614951848983765f,
                           -0.008238378912210464f),
        luisa::make_float3(0.36512258648872375f,
                           0.25676220655441284f, 0.0f),
        luisa::make_float3(0.3124595582485199f,
                           -0.03609641641378403f, 0.0f),
        luisa::make_float3(0.36543145775794983f,
                           -0.03609641641378403f, 0.0f),
        // Barbershop object 29 primitive 457046 / object 32 primitive
        // 504350. Their corresponding transformed vertices differ by one
        // ULP in y, but the closed triangles overlap and Cycles CPU/HIP both
        // accept object 29 at the source endpoint.
        luisa::make_float3(0x1.758f2cp-2f, 0x1.1e1dc4p-1f, 0.0f),
        luisa::make_float3(0x1.3f50e6p-2f, 0x1.1e1dc4p-1f, 0.0f),
        luisa::make_float3(0x1.3fa1ep-2f, 0x1.105864p-2f, 0.0f)};
    upload.triangles = {
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u},
        Triangle{6u, 7u, 8u}};
  }

  psycles::Mat4f object_twenty_nine;
  object_twenty_nine.elements = {
      -4.0158649738941676e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, -4.4516873742850294e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 0.6412966251373291f,
      -0.0029841959476470947f, 1.0f};
  psycles::Mat4f object_thirty_two;
  object_thirty_two.elements = {
      6.936164709259174e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, 7.688912972980688e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 0.6412966251373291f,
      -0.0029841959476470947f, 1.0f};
  auto nextafter_support = object_thirty_two;
  nextafter_support.elements[14u] = std::nextafter(
      nextafter_support.elements[14u],
      std::numeric_limits<float>::infinity());

  scene.instances.emplace(
      InstanceId{1u},
      InstanceDesc{.name = "Cycles object 29",
                   .geometry = GeometryId{1u},
                   .transform = object_twenty_nine});
  scene.instances.emplace(
      InstanceId{2u},
      InstanceDesc{.name = "Cycles object 32",
                   .geometry = GeometryId{2u},
                   .transform = object_thirty_two});
  scene.instances.emplace(
      InstanceId{3u},
      InstanceDesc{.name = "one-ULP primitive support change",
                   .geometry = GeometryId{2u},
                   .transform = nextafter_support});

  const auto support_classes =
      classify_cycles_final_triangle_supports(
          scene, geometry_indices, uploads);
  require(support_classes.ok(),
          "partial support classification failed");
  std::map<
      GeometryId,
      psycles::luisa_backend::detail::CyclesGeometrySupportView>
      final_supports;
  for (const auto &[geometry_id, upload_index] : geometry_indices) {
    const auto &upload = uploads[upload_index];
    final_supports.emplace(
        geometry_id,
        make_cycles_geometry_support_view(
            std::span<const luisa::float3>{
                upload.positions.data(), upload.positions.size()},
            std::span<const Triangle>{
                upload.triangles.data(), upload.triangles.size()}));
  }
  auto plan = build_cycles_instance_intersection_plan(scene, {});
  psycles::luisa_backend::detail::CyclesPrimitiveCompletionPlan
      primitive_plan;
  require(finalize_cycles_instance_intersection_plan(
              scene, support_classes.by_geometry,
              final_supports, plan, primitive_plan),
          "partial support finalization failed");
  require(plan[0u].coincident_count == 1u &&
              plan[1u].coincident_count == 1u,
          "partially equal instances were promoted to a whole class");
  const auto find_record =
      [&](std::size_t instance, std::uint32_t primitive)
      -> const psycles::luisa_backend::detail::
          CyclesPrimitiveCompletionRecord * {
    const auto first = plan[instance].primitive_completion_offset;
    const auto last = first +
                      plan[instance].primitive_completion_count;
    for (auto record = first; record < last; ++record) {
      if (primitive_plan.records[record].local_primitive == primitive) {
        return &primitive_plan.records[record];
      }
    }
    return nullptr;
  };
  const auto *object_29_exact = find_record(0u, 1u);
  const auto *object_32_exact = find_record(1u, 1u);
  require(object_29_exact != nullptr && object_32_exact != nullptr &&
              object_29_exact->instance_offset ==
                  object_32_exact->instance_offset &&
              object_29_exact->instance_count == 2u &&
              object_32_exact->instance_count == 2u,
          "exact partial primitive support was not grouped");
  require(primitive_plan.instances[object_29_exact->instance_offset] == 0u &&
              primitive_plan.instances[
                  object_29_exact->instance_offset + 1u] == 1u,
          "partial primitive aliases lost stable instance order");
  const auto *object_29_overlap = find_record(0u, 2u);
  const auto *object_32_overlap = find_record(1u, 2u);
  require(object_29_overlap != nullptr && object_32_overlap != nullptr &&
              object_29_overlap->instance_offset ==
                  object_32_overlap->instance_offset &&
              object_29_overlap->instance_count == 2u &&
              object_32_overlap->instance_count == 2u,
          "closed Barbershop primitive overlap was not completed");
  require(primitive_plan.instances[object_29_overlap->instance_offset] == 0u &&
              primitive_plan.instances[
                  object_29_overlap->instance_offset + 1u] == 1u,
          "overlapping primitive completion lost stable instance order");
  require(find_record(2u, 1u) == nullptr,
          "disjoint one-ULP planar support was completed");
  require(find_record(2u, 2u) == nullptr,
          "disjoint Barbershop planar support was completed");
}

void test_primitive_completion_is_not_an_equivalence_class() {
  using psycles::contract::InstanceDesc;
  using psycles::contract::InstanceId;
  SceneSnapshot scene;
  std::vector<GeometryUpload> uploads;
  std::map<GeometryId, std::uint32_t> geometry_indices;
  for (auto id = 1u; id <= 3u; ++id) {
    scene.geometries.emplace(
        GeometryId{id},
        TriangleMeshDesc{.name = "closed-overlap-chain-" +
                                 std::to_string(id)});
    geometry_indices.emplace(
        GeometryId{id}, static_cast<std::uint32_t>(uploads.size()));
    uploads.emplace_back(make_support());
  }
  for (auto id = 1u; id <= 3u; ++id) {
    auto transform = psycles::Mat4f{};
    transform.elements = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        static_cast<float>(id - 1u), 0.0f, 0.0f, 1.0f};
    scene.instances.emplace(
        InstanceId{id},
        InstanceDesc{.name = "closed-overlap-chain-instance-" +
                             std::to_string(id),
                     .geometry = GeometryId{id},
                     .transform = transform});
  }

  const auto support_classes =
      classify_cycles_final_triangle_supports(
          scene, geometry_indices, uploads);
  require(support_classes.ok(),
          "closed-overlap chain support classification failed");
  std::map<
      GeometryId,
      psycles::luisa_backend::detail::CyclesGeometrySupportView>
      final_supports;
  for (const auto &[geometry_id, upload_index] : geometry_indices) {
    const auto &upload = uploads[upload_index];
    final_supports.emplace(
        geometry_id,
        make_cycles_geometry_support_view(
            std::span<const luisa::float3>{
                upload.positions.data(), upload.positions.size()},
            std::span<const Triangle>{
                upload.triangles.data(), upload.triangles.size()}));
  }
  auto plan = build_cycles_instance_intersection_plan(scene, {});
  psycles::luisa_backend::detail::CyclesPrimitiveCompletionPlan
      primitive_plan;
  require(finalize_cycles_instance_intersection_plan(
              scene, support_classes.by_geometry,
              final_supports, plan, primitive_plan),
          "closed-overlap chain finalization failed");
  const auto completion =
      [&](std::size_t instance) -> std::span<const std::uint32_t> {
    require(plan[instance].primitive_completion_count == 1u,
            "closed-overlap chain lost its primitive record");
    const auto &record = primitive_plan.records[
        plan[instance].primitive_completion_offset];
    return std::span<const std::uint32_t>{
        primitive_plan.instances.data() + record.instance_offset,
        record.instance_count};
  };
  const auto a = completion(0u);
  const auto b = completion(1u);
  const auto c = completion(2u);
  require(a.size() == 2u && a[0u] == 0u && a[1u] == 1u,
          "left endpoint gained a transitive completion");
  require(b.size() == 3u && b[0u] == 0u && b[1u] == 1u && b[2u] == 2u,
          "middle endpoint lost a direct completion");
  require(c.size() == 2u && c[0u] == 1u && c[1u] == 2u,
          "right endpoint gained a transitive completion");
}

} // namespace

int main() {
  test_reachable_surface_materials();
  test_triangle_instances_with_surface_materials();
  test_final_support_equivalence();
  test_exact_world_support_equivalence();
  test_sparse_primitive_world_support_completion();
  test_primitive_completion_is_not_an_equivalence_class();
  std::cout << "Cycles final instance-support tests passed\n";
  return EXIT_SUCCESS;
}
