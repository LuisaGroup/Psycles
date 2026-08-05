#include "../src/luisa/path_tracer_instance_support.h"
#include "../src/luisa/path_tracer_internal.h"
#include "../src/luisa/path_tracer_scene_geometry.h"

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
using psycles::contract::SceneSnapshot;
using psycles::contract::TriangleMeshDesc;
using psycles::luisa_backend::detail::classify_cycles_final_triangle_supports;
using psycles::luisa_backend::detail::build_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::finalize_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::GeometryUpload;
using psycles::luisa_backend::detail::make_cycles_geometry_support_view;
using psycles::luisa_backend::detail::Triangle;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
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
  psycles::luisa_backend::detail::CyclesPrimitiveIntersectionPlan
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

void test_sparse_primitive_world_support_equivalence() {
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
                           -0.03609641641378403f, 0.0f)};
    upload.triangles = {
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};
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
  psycles::luisa_backend::detail::CyclesPrimitiveIntersectionPlan
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
          CyclesCoincidentPrimitiveRecord * {
    const auto first = plan[instance].coincident_primitive_offset;
    const auto last = first +
                      plan[instance].coincident_primitive_count;
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
  require(find_record(0u, 0u) == nullptr,
          "non-equal Barbershop primitive entered the sparse class");
  require(find_record(2u, 1u) == nullptr,
          "one-ULP target primitive support change was grouped");
}

} // namespace

int main() {
  test_final_support_equivalence();
  test_exact_world_support_equivalence();
  test_sparse_primitive_world_support_equivalence();
  std::cout << "Cycles final instance-support tests passed\n";
  return EXIT_SUCCESS;
}
