#include "cycles_shader_identity.h"
#include "path_kernel_curve_geometry.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_scene_traversal.h"
#include "path_tracer_scene_geometry.h"
#include "path_tracer_scene_upload.h"

#include <psycles/luisa/surface_ray.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::luisa_backend::surface_ray::invalid_primitive;

inline constexpr std::size_t record_count = 26u;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool equal_record(luisa::float4 actual,
                                luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
}

[[nodiscard]] bool test_completion_source_lookup_encodings() {
  constexpr std::array ordinary{InstanceGpu{.cycles_object_index = 7u},
                                InstanceGpu{.cycles_object_index = 19u}};
  const auto empty = make_cycles_completion_source_lookup(ordinary);
  if (!empty.ok() || !empty.dense_instances.empty() ||
      !empty.sparse_instances.empty()) {
    return false;
  }

  constexpr std::array dense_instances{
      InstanceGpu{.cycles_object_index = 7u, .coincident_count = 2u},
      InstanceGpu{.cycles_object_index = 19u}};
  const auto dense = make_cycles_completion_source_lookup(dense_instances);
  if (!dense.ok() || dense.dense_instances.size() != 8u ||
      !dense.sparse_instances.empty() || dense.dense_instances[7u] != 0u) {
    return false;
  }

  constexpr std::array sparse_instances{
      InstanceGpu{.cycles_object_index = 0x40000000u,
                  .primitive_completion_count = 1u},
      InstanceGpu{.cycles_object_index = 3u}};
  const auto sparse = make_cycles_completion_source_lookup(sparse_instances);
  if (!sparse.ok() || !sparse.dense_instances.empty() ||
      sparse.sparse_instances.size() != 1u ||
      sparse.sparse_instances[0u].x != 0x40000000u ||
      sparse.sparse_instances[0u].y != 0u) {
    return false;
  }

  constexpr std::array duplicate_instances{
      InstanceGpu{.cycles_object_index = 5u},
      InstanceGpu{.cycles_object_index = 5u}};
  return !make_cycles_completion_source_lookup(duplicate_instances).ok();
}

struct TraversalXirShape {
  std::size_t instructions{};
  std::size_t triangle_candidate_reads{};
  std::size_t procedural_candidate_reads{};
};

[[nodiscard]] TraversalXirShape traversal_xir_shape(
    const std::shared_ptr<LuisaSceneData> &scene,
    ScenePrimitiveStagePlan plan) {
  const auto traversal = make_scene_traversal_component(plan);
  Kernel1D shape = [scene, traversal](BufferUInt output) noexcept {
    const auto ray = make_ray(
        make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f);
    const auto hit = traversal->closest(
        scene, ray, 0xffu, ScenePrimitiveIdentity::invalid());
    output.write(0u, hit->inst);
  };
  auto module = luisa::compute::xir::ast_to_xir_translate(
      shape.function()->function(), {});
  TraversalXirShape result;
  for (auto *function : module->function_list()) {
    if (const auto *definition = function->definition()) {
      definition->traverse_instructions(
          [&](const luisa::compute::xir::Instruction *instruction) noexcept {
            ++result.instructions;
            if (instruction->isa<
                    luisa::compute::xir::RayQueryObjectReadInst>()) {
              const auto *read = static_cast<const luisa::compute::xir::
                  RayQueryObjectReadInst *>(instruction);
              result.triangle_candidate_reads +=
                  read->op() == luisa::compute::xir::
                                    RayQueryObjectReadOp::
                                        RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT
                      ? 1u
                      : 0u;
              result.procedural_candidate_reads +=
                  read->op() == luisa::compute::xir::
                                    RayQueryObjectReadOp::
                                        RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT
                      ? 1u
                      : 0u;
            }
          });
    }
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (!test_completion_source_lookup_encodings()) {
    std::cerr << "FAILED: completion-source lookup encodings\n";
    return EXIT_FAILURE;
  }
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();

  psycles::Mat4f bottle_transform;
  bottle_transform.elements[0u] = 0.5903866291046143f;
  bottle_transform.elements[5u] = 0.5903866291046143f;
  bottle_transform.elements[10u] = 0.5903866291046143f;
  bottle_transform.elements[12u] = 0.4168449342250824f;
  bottle_transform.elements[13u] = 8.169964790344238f;
  bottle_transform.elements[14u] = 1.4632248878479004f;
  psycles::Mat4f coincident_transform_source;
  coincident_transform_source.elements[12u] = 5.0f;
  const auto bottle_world_to_object =
      to_luisa(cycles_inverse_transform(bottle_transform));
  const auto coincident_world_to_object =
      to_luisa(cycles_inverse_transform(coincident_transform_source));
  psycles::Mat4f overlap_a_transform;
  overlap_a_transform.elements = {
      -4.013790899648484e-8f, -0.9182483553886414f, 0.0f, 0.0f,
      1.0184273719787598f, -4.4516873742850294e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      1.6824010610580444f, 4.567445755004883f,
      -0.0029841959476470947f, 1.0f};
  psycles::Mat4f overlap_b_transform;
  overlap_b_transform.elements = {
      6.932582152785471e-8f, -0.9182483553886414f, 0.0f, 0.0f,
      1.0184273719787598f, 7.688912972980688e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      1.6824010610580444f, 4.567445755004883f,
      -0.0029841959476470947f, 1.0f};
  const auto overlap_a_world_to_object =
      to_luisa(cycles_inverse_transform(overlap_a_transform));
  const auto overlap_b_world_to_object =
      to_luisa(cycles_inverse_transform(overlap_b_transform));
  psycles::Mat4f partial_a_transform;
  partial_a_transform.elements = {
      -4.0158649738941676e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, -4.4516873742850294e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 0.6412966251373291f,
      -0.0029841959476470947f, 1.0f};
  psycles::Mat4f partial_b_transform;
  partial_b_transform.elements = {
      6.936164709259174e-8f, 0.9187228083610535f, 0.0f, 0.0f,
      -1.0184273719787598f, 7.688912972980688e-8f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.9999998807907104f, 0.0f,
      3.0013208389282227f, 0.6412966251373291f,
      -0.0029841959476470947f, 1.0f};
  const auto partial_a_world_to_object =
      to_luisa(cycles_inverse_transform(partial_a_transform));
  const auto partial_b_world_to_object =
      to_luisa(cycles_inverse_transform(partial_b_transform));

  constexpr auto curve_bindless_base = geometry_bindless_stride;
  const std::array geometries{
      GeometryGpu{.bindless_base = 0u,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 100u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = curve_bindless_base,
                  .material_offset = 1u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 200u,
                  .cycles_segment_offset = 300u,
                  .primitive_kind = geometry_kind_curve,
                  .curve_subdivision_level = 2u},
      GeometryGpu{.bindless_base = 2u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 20474114u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 3u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 3396299u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 4u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 700000u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 5u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 94356u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 6u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 950264u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 7u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 457042u,
                  .primitive_kind = geometry_kind_triangle},
      GeometryGpu{.bindless_base = 8u * geometry_bindless_stride,
                  .material_offset = 0u,
                  .material_count = 1u,
                  .cycles_primitive_offset = 504346u,
                  .primitive_kind = geometry_kind_triangle}};
  const std::array instances{
      InstanceGpu{.geometry_index = 0u, .cycles_object_index = 11u},
      InstanceGpu{.geometry_index = 1u,
                  .override_offset = 0u,
                  .override_count = 1u,
                  .cycles_object_index = 22u},
      InstanceGpu{.geometry_index = 0u,
                  .cycles_object_index = 489u,
                  .coincident_next = 3u,
                  .coincident_count = 2u,
                  .cycles_world_to_object = coincident_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .cycles_object_index = 1936u,
                  .coincident_next = 2u,
                  .coincident_count = 2u,
                  .cycles_world_to_object = coincident_world_to_object},
      InstanceGpu{.geometry_index = 2u,
                  .cycles_object_index = 2131u,
                  .coincident_next = 5u,
                  .coincident_count = 2u,
                  .cycles_transform_applied = 1u,
                  .cycles_world_to_object = bottle_world_to_object},
      InstanceGpu{.geometry_index = 3u,
                  .cycles_object_index = 2372u,
                  .coincident_next = 4u,
                  .coincident_count = 2u,
                  .cycles_world_to_object = bottle_world_to_object},
      InstanceGpu{.geometry_index = 4u,
                  .cycles_object_index = 5011u,
                  .cycles_world_to_object = overlap_a_world_to_object},
      InstanceGpu{.geometry_index = 4u,
                  .cycles_object_index = 5066u,
                  .cycles_world_to_object = overlap_b_world_to_object},
      InstanceGpu{.geometry_index = 5u,
                  .cycles_object_index = 6u,
                  .coincident_next = 9u,
                  .coincident_count = 2u,
                  .cycles_transform_applied = 1u},
      InstanceGpu{.geometry_index = 6u,
                  .cycles_object_index = 71u,
                  .coincident_next = 8u,
                  .coincident_count = 2u,
                  .cycles_transform_applied = 1u},
      InstanceGpu{.geometry_index = 7u,
                  .cycles_object_index = 29u,
                  .primitive_completion_offset = 0u,
                  .primitive_completion_count = 2u,
                  .cycles_transform_applied = 1u,
                  .cycles_world_to_object = partial_a_world_to_object},
      InstanceGpu{.geometry_index = 8u,
                  .cycles_object_index = 32u,
                  .primitive_completion_offset = 2u,
                  .primitive_completion_count = 2u,
                  .cycles_transform_applied = 1u,
                  .cycles_world_to_object = partial_b_world_to_object}};
  constexpr std::array primitive_completions{
      PrimitiveCompletionGpu{.local_primitive = 1u,
                             .instance_offset = 0u,
                             .instance_count = 2u},
      PrimitiveCompletionGpu{.local_primitive = 4u,
                             .instance_offset = 0u,
                             .instance_count = 2u},
      PrimitiveCompletionGpu{.local_primitive = 1u,
                             .instance_offset = 0u,
                             .instance_count = 2u},
      PrimitiveCompletionGpu{.local_primitive = 4u,
                             .instance_offset = 0u,
                             .instance_count = 2u}};
  constexpr std::array primitive_completion_instances{10u, 11u};
  const auto completion_source_lookup =
      make_cycles_completion_source_lookup(instances);
  if (!completion_source_lookup.ok() ||
      completion_source_lookup.dense_instances.size() != 2373u ||
      !completion_source_lookup.sparse_instances.empty() ||
      completion_source_lookup.dense_instances[29u] != 10u ||
      completion_source_lookup.dense_instances[22u] != invalid_primitive) {
    std::cerr << "FAILED: dense completion-source lookup plan\n";
    return EXIT_FAILURE;
  }
  constexpr std::array geometry_materials{
      MaterialBindingGpu{.surface_tag = 41u,
                         .cycles_shader_index = 5u,
                         .material_identity = 105u},
      MaterialBindingGpu{.surface_tag = 55u,
                         .cycles_shader_index = 7u,
                         .material_identity = 107u}};
  constexpr std::array override_materials{
      MaterialBindingGpu{.surface_tag = 77u,
                         .cycles_shader_index = 9u,
                         .material_identity = 109u}};
  constexpr std::array vertices{luisa::float3{-2.0f, -2.0f, 4.0f},
                                luisa::float3{2.0f, -2.0f, 4.0f},
                                luisa::float3{0.0f, 2.0f, 4.0f}};
  constexpr std::array triangles{Triangle{0u, 1u, 2u}};
  constexpr std::array overlap_vertices{
      luisa::float3{-0.12768065929412842f,
                    -0.016480661928653717f,
                    -0.002021433785557747f},
      luisa::float3{-0.1807040125131607f,
                    0.0647093877196312f,
                    -0.002321503823623061f},
      luisa::float3{-0.18065254390239716f,
                    -0.016480661928653717f,
                    -0.002021433785557747f}};
  // Official Barbershop floor exact-support pairs. Object 71 primitives
  // 950264 and 950290 share their final triangles with object 6 primitives
  // 94356 and 94382. Cycles CPU and HIP exclude only object 71 and accept the
  // object 6 sibling at the closed t == 0 endpoint for both shadow and
  // continuation traversal.
  constexpr std::array barbershop_floor_vertices{
      luisa::float3{2.943859100341797f,
                    2.5070724487304688f,
                    -0.0029841959476470947f},
      luisa::float3{2.943913221359253f,
                    2.4047765731811523f,
                    -0.0029841959476470947f},
      luisa::float3{2.9934756755828857f,
                    2.4047765731811523f,
                    -0.0029841959476470947f},
      luisa::float3{3.338683843612671f,
                    2.5072803497314453f,
                    -0.0029841959476470947f},
      luisa::float3{3.289121150970459f,
                    2.5072803497314453f,
                    -0.0029841959476470947f},
      luisa::float3{3.28941011428833f,
                    2.2144217491149902f,
                    -0.0029841959476470947f}};
  constexpr auto barbershop_floor_triangles = []() noexcept {
    std::array<Triangle, 27u> result{};
    result[0u] = Triangle{0u, 1u, 2u};
    // Preserve the actual local primitive ordinal without introducing
    // unrelated fixture geometry that could become a traversal candidate.
    for (std::size_t i = 1u; i < 26u; ++i) {
      result[i] = Triangle{0u, 0u, 0u};
    }
    result[26u] = Triangle{3u, 4u, 5u};
    return result;
  }();
  constexpr std::array partial_support_vertices{
      luisa::float3{-0.373418390750885f,
                    -0.5634331107139587f,
                    -0.005576633382588625f},
      luisa::float3{-0.373420774936676f,
                    -0.5620933771133423f,
                    -0.0088327182456851f},
      luisa::float3{-0.3707645535469055f,
                    -0.5614951848983765f,
                    -0.008238378912210464f},
      luisa::float3{0.36512258648872375f,
                    0.25676220655441284f, 0.0f},
      luisa::float3{0.3124595582485199f,
                    -0.03609641641378403f, 0.0f},
      luisa::float3{0.36543145775794983f,
                    -0.03609641641378403f, 0.0f},
      // Barbershop object 29 primitive 457046 / object 32 primitive
      // 504350: corresponding world vertices differ by one ULP in y.
      luisa::float3{0x1.758f2cp-2f, 0x1.1e1dc4p-1f, 0.0f},
      luisa::float3{0x1.3f50e6p-2f, 0x1.1e1dc4p-1f, 0.0f},
      luisa::float3{0x1.3fa1ep-2f, 0x1.105864p-2f, 0.0f}};
  constexpr std::array partial_support_triangles{
      Triangle{0u, 1u, 2u}, Triangle{3u, 4u, 5u},
      Triangle{0u, 0u, 0u}, Triangle{0u, 0u, 0u},
      Triangle{6u, 7u, 8u}};
  std::array<luisa::float3, partial_support_vertices.size()>
      partial_a_world_vertices{};
  std::array<luisa::float3, partial_support_vertices.size()>
      partial_b_world_vertices{};
  for (std::size_t i = 0u; i < partial_support_vertices.size(); ++i) {
    const auto local = psycles::Vec3f{
        partial_support_vertices[i].x,
        partial_support_vertices[i].y,
        partial_support_vertices[i].z};
    const auto a = cycles_transform_point(partial_a_transform, local);
    const auto b = cycles_transform_point(partial_b_transform, local);
    partial_a_world_vertices[i] = make_float3(a.x, a.y, a.z);
    partial_b_world_vertices[i] = make_float3(b.x, b.y, b.z);
  }
  constexpr std::array bottle_vertices{
      luisa::float3{0.04980994760990143f,
                    -0.015110095962882042f,
                    0.0014796979958191514f},
      luisa::float3{0.047333888709545135f,
                    -0.0196063332259655f,
                    0.0005811812588945031f},
      luisa::float3{0.04902583360671997f,
                    -0.01487223245203495f,
                    0.0005811817827634513f}};
  std::array<luisa::float3, bottle_vertices.size()>
      bottle_world_vertices{};
  for (std::size_t i = 0u; i < bottle_vertices.size(); ++i) {
    const auto transformed = cycles_transform_point(
        bottle_transform,
        {bottle_vertices[i].x,
         bottle_vertices[i].y,
         bottle_vertices[i].z});
    bottle_world_vertices[i] =
        luisa::make_float3(transformed.x, transformed.y, transformed.z);
  }
  // The two segments are geometrically identical but have distinct Cycles
  // segment identities. Their device primitive order is deliberately opposite
  // to the Cycles order: Cycles packs the segment ordinal into prim_type, so
  // device primitive 0 must win the exact-distance tie on every backend.
  constexpr std::array curve_keys{luisa::float4{-2.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{-1.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{1.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{2.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{-2.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{-1.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{1.0f, 0.0f, 2.0f, 0.4f},
                                  luisa::float4{2.0f, 0.0f, 2.0f, 0.4f}};
  constexpr std::array curve_segments{
      CurveSegmentGpu{.key_before = 0u,
                      .key_begin = 1u,
                      .key_end = 2u,
                      .key_after = 3u,
                      .curve_index = 0u,
                      .cycles_curve_index = 200u,
                      .cycles_segment_index = 301u},
      CurveSegmentGpu{.key_before = 4u,
                      .key_begin = 5u,
                      .key_end = 6u,
                      .key_after = 7u,
                      .curve_index = 0u,
                      .cycles_curve_index = 200u,
                      .cycles_segment_index = 300u}};
  constexpr std::array curve_bounds{AABB{.packed_min = {-2.5f, -0.5f, 1.5f},
                                         .packed_max = {2.5f, 0.5f, 2.5f}},
                                    AABB{.packed_min = {-2.5f, -0.5f, 1.5f},
                                         .packed_max = {2.5f, 0.5f, 2.5f}}};
  constexpr std::array curve_material_slots{0u};
  constexpr std::array curve_intercepts{0.0f, 0.2f, 0.6f, 1.0f,
                                        0.0f, 0.2f, 0.6f, 1.0f};
  constexpr std::array curve_lengths{3.5f};
  constexpr std::array curve_randoms{0.25f};

  scene->geometry_buffer = device.create_buffer<GeometryGpu>(geometries.size());
  scene->instance_buffer = device.create_buffer<InstanceGpu>(instances.size());
  scene->primitive_completion_buffer =
      device.create_buffer<PrimitiveCompletionGpu>(
          primitive_completions.size());
  scene->primitive_completion_instance_buffer =
      device.create_buffer<luisa::uint>(primitive_completion_instances.size());
  scene->cycles_completion_source_dense_count = static_cast<std::uint32_t>(
      completion_source_lookup.dense_instances.size());
  scene->cycles_completion_source_dense_buffer =
      device.create_buffer<luisa::uint>(
          completion_source_lookup.dense_instances.size());
  scene->geometry_material_buffer =
      device.create_buffer<MaterialBindingGpu>(geometry_materials.size());
  scene->override_material_buffer =
      device.create_buffer<MaterialBindingGpu>(override_materials.size());
  auto vertex_buffer = device.create_buffer<luisa::float3>(vertices.size());
  auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
  auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
  auto bottle_vertex_buffer =
      device.create_buffer<luisa::float3>(bottle_vertices.size());
  auto bottle_world_vertex_buffer =
      device.create_buffer<luisa::float3>(bottle_world_vertices.size());
  auto bottle_triangle_buffer =
      device.create_buffer<Triangle>(triangles.size());
  auto bottle_mesh =
      device.create_mesh(bottle_vertex_buffer, bottle_triangle_buffer);
  auto overlap_vertex_buffer =
      device.create_buffer<luisa::float3>(overlap_vertices.size());
  auto overlap_triangle_buffer =
      device.create_buffer<Triangle>(triangles.size());
  auto overlap_mesh =
      device.create_mesh(overlap_vertex_buffer, overlap_triangle_buffer);
  auto barbershop_floor_vertex_buffer =
      device.create_buffer<luisa::float3>(barbershop_floor_vertices.size());
  auto barbershop_floor_triangle_buffer =
      device.create_buffer<Triangle>(barbershop_floor_triangles.size());
  auto barbershop_floor_mesh = device.create_mesh(
      barbershop_floor_vertex_buffer,
      barbershop_floor_triangle_buffer);
  auto partial_support_vertex_buffer =
      device.create_buffer<luisa::float3>(partial_support_vertices.size());
  auto partial_support_triangle_buffer =
      device.create_buffer<Triangle>(partial_support_triangles.size());
  auto partial_a_world_vertex_buffer =
      device.create_buffer<luisa::float3>(partial_a_world_vertices.size());
  auto partial_b_world_vertex_buffer =
      device.create_buffer<luisa::float3>(partial_b_world_vertices.size());
  auto partial_support_mesh = device.create_mesh(
      partial_support_vertex_buffer,
      partial_support_triangle_buffer);
  auto bounds_buffer = device.create_buffer<AABB>(curve_bounds.size());
  auto segment_buffer =
      device.create_buffer<CurveSegmentGpu>(curve_segments.size());
  auto key_buffer = device.create_buffer<luisa::float4>(curve_keys.size());
  auto curve_material_buffer =
      device.create_buffer<luisa::uint>(curve_material_slots.size());
  auto curve_intercept_buffer =
      device.create_buffer<float>(curve_intercepts.size());
  auto curve_length_buffer = device.create_buffer<float>(curve_lengths.size());
  auto curve_random_buffer = device.create_buffer<float>(curve_randoms.size());
  auto curves = device.create_procedural_primitive(bounds_buffer);

  scene->heap = device.create_bindless_array(9u * geometry_bindless_stride);
  scene->heap.emplace_on_update(0u, triangle_buffer);
  scene->heap.emplace_on_update(9u, vertex_buffer);
  scene->heap.emplace_on_update(curve_bindless_base, segment_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 1u, key_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 3u,
                                curve_intercept_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 4u,
                                curve_material_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 5u,
                                curve_length_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 6u,
                                curve_random_buffer);
  scene->heap.emplace_on_update(2u * geometry_bindless_stride,
                                bottle_triangle_buffer);
  scene->heap.emplace_on_update(2u * geometry_bindless_stride + 9u,
                                bottle_world_vertex_buffer);
  scene->heap.emplace_on_update(3u * geometry_bindless_stride,
                                bottle_triangle_buffer);
  scene->heap.emplace_on_update(3u * geometry_bindless_stride + 9u,
                                bottle_vertex_buffer);
  scene->heap.emplace_on_update(4u * geometry_bindless_stride,
                                overlap_triangle_buffer);
  scene->heap.emplace_on_update(4u * geometry_bindless_stride + 9u,
                                overlap_vertex_buffer);
  scene->heap.emplace_on_update(5u * geometry_bindless_stride,
                                barbershop_floor_triangle_buffer);
  scene->heap.emplace_on_update(5u * geometry_bindless_stride + 9u,
                                barbershop_floor_vertex_buffer);
  scene->heap.emplace_on_update(6u * geometry_bindless_stride,
                                barbershop_floor_triangle_buffer);
  scene->heap.emplace_on_update(6u * geometry_bindless_stride + 9u,
                                barbershop_floor_vertex_buffer);
  scene->heap.emplace_on_update(7u * geometry_bindless_stride,
                                partial_support_triangle_buffer);
  scene->heap.emplace_on_update(7u * geometry_bindless_stride + 9u,
                                partial_a_world_vertex_buffer);
  scene->heap.emplace_on_update(8u * geometry_bindless_stride,
                                partial_support_triangle_buffer);
  scene->heap.emplace_on_update(8u * geometry_bindless_stride + 9u,
                                partial_b_world_vertex_buffer);
  scene->accel = device.create_accel();
  scene->accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false, 0u);
  scene->accel.emplace_back(curves, make_float4x4(1.0f), 0xffu, 1u);
  const auto coincident_transform = translation(make_float3(5.0f, 0.0f, 0.0f));
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 2u);
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 3u);
  const auto bottle_luisa_transform = to_luisa(bottle_transform);
  scene->accel.emplace_back(
      bottle_mesh, bottle_luisa_transform, 0xffu, false, 4u);
  scene->accel.emplace_back(
      bottle_mesh, bottle_luisa_transform, 0xffu, false, 5u);
  scene->accel.emplace_back(
      overlap_mesh, to_luisa(overlap_a_transform), 0xffu, false, 6u);
  scene->accel.emplace_back(
      overlap_mesh, to_luisa(overlap_b_transform), 0xffu, false, 7u);
  scene->accel.emplace_back(
      barbershop_floor_mesh, make_float4x4(1.0f), 0xffu, false, 8u);
  scene->accel.emplace_back(
      barbershop_floor_mesh, make_float4x4(1.0f), 0xffu, false, 9u);
  scene->accel.emplace_back(
      partial_support_mesh, to_luisa(partial_a_transform),
      0xffu, false, 10u);
  scene->accel.emplace_back(
      partial_support_mesh, to_luisa(partial_b_transform),
      0xffu, false, 11u);

  const auto empty_shape = traversal_xir_shape(scene, {});
  const auto triangle_shape = traversal_xir_shape(
      scene, {.triangles = true});
  const auto curve_shape = traversal_xir_shape(
      scene, {.curves = true});
  const auto mixed_shape = traversal_xir_shape(
      scene, {.triangles = true, .curves = true});
  const auto report_shapes =
      std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr;
  if (report_shapes) {
    std::cerr << "scene traversal XIR: empty=" << empty_shape.instructions
              << ", triangles=" << triangle_shape.instructions
              << ", curves=" << curve_shape.instructions
              << ", mixed=" << mixed_shape.instructions << '\n';
  }
  if (empty_shape.triangle_candidate_reads != 0u ||
      empty_shape.procedural_candidate_reads != 0u ||
      triangle_shape.triangle_candidate_reads == 0u ||
      triangle_shape.procedural_candidate_reads != 0u ||
      curve_shape.triangle_candidate_reads != 0u ||
      curve_shape.procedural_candidate_reads == 0u ||
      mixed_shape.triangle_candidate_reads == 0u ||
      mixed_shape.procedural_candidate_reads == 0u ||
      !(empty_shape.instructions < triangle_shape.instructions &&
        empty_shape.instructions < curve_shape.instructions &&
        triangle_shape.instructions < mixed_shape.instructions &&
        curve_shape.instructions < mixed_shape.instructions)) {
    std::cerr << "FAILED: primitive capability did not bound scene-traversal "
                 "XIR\n";
    return EXIT_FAILURE;
  }

  auto output = device.create_buffer<luisa::float4>(record_count);
  const auto traversal = make_scene_traversal_component(
      {.triangles = true, .curves = true});
  const auto curve_primitive = make_curve_primitive_component();
  const auto curve_geometry = make_curve_geometry_component();
  Kernel1D evaluate = [scene, traversal, curve_primitive,
                       curve_geometry](BufferFloat4 records) noexcept {
    const UInt test = dispatch_x();
    UInt source_object = invalid_primitive;
    UInt source_primitive = invalid_primitive;
    UInt light_object = invalid_primitive;
    UInt light_primitive = invalid_primitive;
    source_object = select(source_object, 22u, (test == 1u) | (test == 2u));
    source_primitive = select(source_primitive, select(200u, 201u, test == 2u),
                              (test == 1u) | (test == 2u));
    source_object = select(source_object, 11u, test == 4u);
    source_primitive = select(source_primitive, 100u, test == 4u);
    source_object = select(source_object, 1u, test == 5u);
    source_primitive = select(source_primitive, 0u, test == 5u);
    const auto exclude_later_coincident =
        (test == 11u) | (test == 13u) | (test == 15u);
    source_object =
        select(source_object, 1936u, exclude_later_coincident);
    source_primitive =
        select(source_primitive, 100u, exclude_later_coincident);
    light_object = select(light_object, 22u, test == 3u);
    light_primitive = select(light_primitive, 200u, test == 3u);
    source_object = select(source_object, 2131u, test == 17u);
    source_primitive = select(source_primitive, 20474114u, test == 17u);
    light_object = select(light_object, 2131u, test == 18u);
    light_primitive = select(light_primitive, 20474114u, test == 18u);
    source_object = select(source_object, 2372u, test == 19u);
    source_primitive = select(source_primitive, 3396299u, test == 19u);
    const auto barbershop_endpoint_test =
        (test == 21u) | (test == 22u) | (test == 23u);
    source_object = select(source_object, 71u, barbershop_endpoint_test);
    source_primitive = select(
        source_primitive,
        select(950264u, 950290u, test >= 22u),
        barbershop_endpoint_test);
    const auto barbershop_exact_sparse_shadow_test = test == 24u;
    const auto barbershop_overlap_completion_shadow_test = test == 25u;
    const auto barbershop_sparse_shadow_test =
        barbershop_exact_sparse_shadow_test |
        barbershop_overlap_completion_shadow_test;
    source_object = select(
        source_object, 32u, barbershop_sparse_shadow_test);
    source_primitive = select(
        source_primitive,
        select(504347u, 504350u,
               barbershop_overlap_completion_shadow_test),
        barbershop_sparse_shadow_test);
    light_object = select(
        light_object, 165u, barbershop_sparse_shadow_test);
    const auto ray_x = select(0.0f, 5.0f, test >= 10u);
    const auto ray_z = select(0.0f, 4.0f,
                              (test == 12u) | (test == 13u));
    const auto ray_maximum = select(10.0f, 4.0f, test >= 14u);
    const auto bottle_test = (test >= 16u) & (test <= 19u);
    const auto overlap_test = test == 20u;
    const auto barbershop_continuation_test =
        (test == 22u) | (test == 23u);
    const auto barbershop_offset_rejection_test = test == 23u;
    const auto base_ray_origin = select(select(
        select(make_float3(ray_x, 0.0f, ray_z),
               make_float3(1.8301146030426025f,
                           9.144867897033691f,
                           1.5106611251831055f),
               bottle_test),
        make_float3(2.7035441398620605f,
                    8.901592254638672f,
                    1.234214186668396f),
        overlap_test),
        select(make_float3(2.974536657333374f,
                           2.4320390224456787f,
                           -0.0029841959476470947f),
               make_float3(3.2897191047668457f,
                           2.2167599201202393f,
                           select(-0.0029841959476470947f,
                                  -0.0029689371585845947f,
                                  barbershop_offset_rejection_test)),
               barbershop_continuation_test),
        barbershop_endpoint_test);
    const auto ray_origin = select(
        base_ray_origin,
        select(make_float3(2.8889687061309814f,
                           0.9575355648994446f,
                           -0.0029841959476470947f),
               make_float3(2.5468125343322754f,
                           0.9504910111427307f,
                           -0.0029841959476470947f),
               barbershop_overlap_completion_shadow_test),
        barbershop_sparse_shadow_test);
    const auto base_ray_direction = select(select(
        select(make_float3(0.0f, 0.0f, 1.0f),
               make_float3(-0.8146389126777649f,
                           -0.5793101787567139f,
                           -0.02762461081147194f),
               bottle_test),
        make_float3(-0.23053720593452454f,
                    -0.9327327013015747f,
                    -0.27724069356918335f),
        overlap_test),
        select(make_float3(-0.1057601049542427f,
                           -0.393018901348114f,
                           0.9134281277656555f),
               make_float3(-0.33513185381889343f,
                           0.7402154207229614f,
                           0.5828961133956909f),
               barbershop_continuation_test),
        barbershop_endpoint_test);
    const auto ray_direction = select(
        base_ray_direction,
        select(make_float3(-0.37712639570236206f,
                           0.0533241368830204f,
                           0.9246254563331604f),
               make_float3(0.09388570487499237f,
                           0.0796050876379013f,
                           0.9923954010009766f),
               barbershop_overlap_completion_shadow_test),
        barbershop_sparse_shadow_test);
    const auto base_ray_maximum = select(select(
        select(ray_maximum, 125.67607116699219f, bottle_test),
        101.64700317382812f,
        overlap_test),
        select(2.4996728897094727f,
               3.4028234663852886e+38f,
               barbershop_continuation_test),
        barbershop_endpoint_test);
    const auto ray = make_ray(
        ray_origin,
        ray_direction,
        0.0f,
        select(base_ray_maximum,
               select(2.347609281539917f,
                      2.1322386264801025f,
                      barbershop_overlap_completion_shadow_test),
               barbershop_sparse_shadow_test));
    Var<luisa::compute::CommittedHit> hit;
    $if(barbershop_continuation_test) {
      hit = traversal->closest(
          scene, ray, 0xffu,
          {.object = source_object, .primitive = source_primitive});
    }
    $else {
      hit = traversal->closest_shadow(
          scene, ray, 0xffu,
          {.object = source_object, .primitive = source_primitive},
          {.object = light_object, .primitive = light_primitive});
    };
    records.write(test,
                  make_float4(hit->committed_ray_t, cast<float>(hit->inst),
                              cast<float>(hit->prim),
                              select(0.0f, 1.0f, hit->is_procedural())));

    $if(overlap_test) {
      // This is the exact Barbershop floor configuration which exposed the
      // HIPRT divergence. A primary hardware hit is only a broad-phase
      // candidate: the Cycles predicate supplies both its distance and its
      // barycentrics. Reconstruct the surface from that result, then cast the
      // sampled-light shadow ray. Using the backend barycentrics here places
      // the origin on the wrong side of the near-overlapping sibling and
      // produces a false self-shadow.
      const auto instance = scene->instance_buffer->read(hit->inst);
      const auto geometry =
          scene->geometry_buffer->read(instance.geometry_index);
      const auto triangle =
          scene->heap->buffer<Triangle>(geometry.bindless_base).read(hit->prim);
      const auto positions = scene->heap->buffer<luisa::float3>(
          geometry.bindless_base + 9u);
      const auto p0 = positions.read(triangle.i0);
      const auto p1 = positions.read(triangle.i1);
      const auto p2 = positions.read(triangle.i2);
      const auto object_position =
          p0 + hit->bary.x * (p1 - p0) + hit->bary.y * (p2 - p0);
      const auto world_position =
          (scene->accel->instance_transform(hit->inst) *
           make_float4(object_position, 1.0f))
              .xyz();
      const auto shadow_ray = make_ray(
          world_position,
          make_float3(0.5150954127311707f,
                      0.6604911684989929f,
                      0.5462856888771057f),
          0.0f,
          3.6035547256469727f);
      const auto cycles_object = instance.cycles_object_index;
      const auto cycles_primitive =
          geometry.cycles_primitive_offset + hit->prim;
      const auto shadow_hit = traversal->closest_shadow(
          scene, shadow_ray, 0xffu,
          {.object = cycles_object, .primitive = cycles_primitive},
          ScenePrimitiveIdentity::invalid());
      records.write(
          test,
          make_float4(cast<float>(cycles_object),
                      cast<float>(cycles_primitive),
                      select(0.0f, 1.0f, shadow_hit->miss()),
                      hit->committed_ray_t));
    };

    $if(((test >= 10u) & (test <= 19u)) |
        ((test >= 21u) & (test <= 22u)) |
        (test == 24u) | (test == 25u)) {
      $if(!hit->miss()) {
        const auto instance = scene->instance_buffer->read(hit->inst);
        const auto geometry =
            scene->geometry_buffer->read(instance.geometry_index);
        records.write(test,
                      make_float4(hit->committed_ray_t,
                                  cast<float>(instance.cycles_object_index),
                                  cast<float>(geometry.cycles_primitive_offset +
                                             hit->prim),
                                  cast<float>(hit->inst)));
      };
    };

    $if(test == 6u) {
      const auto primitive = curve_primitive->emit(scene, 1u, 0u);
      records.write(
          test, make_float4(cast<float>(primitive.material_binding.surface_tag),
                            cast<float>(primitive.cycles_object_index),
                            cast<float>(primitive.cycles_primitive_index),
                            cast<float>(primitive.cycles_surface_shader &
                                        cycles_shader_identity::shader_mask)));
    };
    $if(test == 7u) {
      const auto geometry = curve_geometry->emit(scene, 1u, 0u, ray, 2.0f);
      records.write(test,
                    make_float4(geometry.intersection.u,
                                geometry.intersection.v, geometry.intercept,
                                geometry.length));
    };
    $if(test == 8u) {
      const auto geometry = curve_geometry->emit(scene, 1u, 0u, ray, 2.0f);
      records.write(test,
                    make_float4(geometry.thickness, geometry.random,
                                geometry.tangent_normal.z,
                                geometry.shading_normal.z));
    };
    $if(test == 9u) {
      const auto geometry = curve_geometry->emit(scene, 1u, 0u, ray, 2.0f);
      records.write(test,
                    make_float4(geometry.position.x, geometry.position.y,
                                geometry.position.z, geometry.dpdu.x));
    };
  };
  auto shader = device.compile(evaluate);

  std::array<luisa::float4, record_count> actual{};
  stream << scene->geometry_buffer.copy_from(luisa::span{geometries})
         << scene->instance_buffer.copy_from(luisa::span{instances})
         << scene->primitive_completion_buffer.copy_from(
                luisa::span{primitive_completions})
         << scene->primitive_completion_instance_buffer.copy_from(
                luisa::span{primitive_completion_instances})
         << scene->cycles_completion_source_dense_buffer.copy_from(
                luisa::span{completion_source_lookup.dense_instances})
         << scene->geometry_material_buffer.copy_from(
                luisa::span{geometry_materials})
         << scene->override_material_buffer.copy_from(
                luisa::span{override_materials})
         << vertex_buffer.copy_from(luisa::span{vertices})
         << triangle_buffer.copy_from(luisa::span{triangles})
         << bottle_vertex_buffer.copy_from(luisa::span{bottle_vertices})
         << bottle_world_vertex_buffer.copy_from(
                luisa::span{bottle_world_vertices})
         << bottle_triangle_buffer.copy_from(luisa::span{triangles})
         << overlap_vertex_buffer.copy_from(luisa::span{overlap_vertices})
         << overlap_triangle_buffer.copy_from(luisa::span{triangles})
         << barbershop_floor_vertex_buffer.copy_from(
                luisa::span{barbershop_floor_vertices})
         << barbershop_floor_triangle_buffer.copy_from(
                luisa::span{barbershop_floor_triangles})
         << partial_support_vertex_buffer.copy_from(
                luisa::span{partial_support_vertices})
         << partial_support_triangle_buffer.copy_from(
                luisa::span{partial_support_triangles})
         << partial_a_world_vertex_buffer.copy_from(
                luisa::span{partial_a_world_vertices})
         << partial_b_world_vertex_buffer.copy_from(
                luisa::span{partial_b_world_vertices})
         << bounds_buffer.copy_from(luisa::span{curve_bounds})
         << segment_buffer.copy_from(luisa::span{curve_segments})
         << key_buffer.copy_from(luisa::span{curve_keys})
         << curve_material_buffer.copy_from(luisa::span{curve_material_slots})
         << curve_intercept_buffer.copy_from(luisa::span{curve_intercepts})
         << curve_length_buffer.copy_from(luisa::span{curve_lengths})
         << curve_random_buffer.copy_from(luisa::span{curve_randoms})
         << scene->heap.update() << mesh.build() << bottle_mesh.build()
         << overlap_mesh.build() << barbershop_floor_mesh.build()
         << partial_support_mesh.build()
         << curves.build()
         << scene->accel.build() << shader(output).dispatch(record_count)
         << output.copy_to(luisa::span{actual}) << synchronize();

  constexpr std::array expected{luisa::float4{2.0f, 1.0f, 0.0f, 1.0f},
                                luisa::float4{4.0f, 0.0f, 0.0f, 0.0f},
                                luisa::float4{2.0f, 1.0f, 0.0f, 1.0f},
                                luisa::float4{4.0f, 0.0f, 0.0f, 0.0f},
                                luisa::float4{2.0f, 1.0f, 0.0f, 1.0f},
                                luisa::float4{2.0f, 1.0f, 0.0f, 1.0f},
                                luisa::float4{77.0f, 22.0f, 200.0f, 9.0f},
                                luisa::float4{0.5f, 0.0f, 0.4f, 3.5f},
                                luisa::float4{0.8f, 0.25f, -1.0f, -1.0f},
                                luisa::float4{0.0f, 0.0f, 2.0f, 2.25f},
                                luisa::float4{4.0f, 1936.0f, 100.0f, 3.0f},
                                luisa::float4{4.0f, 489.0f, 100.0f, 2.0f},
                                luisa::float4{0.0f, 1936.0f, 100.0f, 3.0f},
                                luisa::float4{0.0f, 489.0f, 100.0f, 2.0f},
                                luisa::float4{4.0f, 1936.0f, 100.0f, 3.0f},
                                luisa::float4{4.0f, 489.0f, 100.0f, 2.0f},
                                luisa::float4{1.6995198f, 2131.0f,
                                              20474114.0f, 4.0f},
                                luisa::float4{1.69952f, 2372.0f,
                                              3396299.0f, 5.0f},
                                luisa::float4{1.69952f, 2372.0f,
                                              3396299.0f, 5.0f},
                                luisa::float4{1.6995198f, 2131.0f,
                                              20474114.0f, 4.0f},
                                luisa::float4{5066.0f, 700000.0f, 1.0f,
                                              4.4699316f},
                                luisa::float4{0.0f, 6.0f, 94356.0f, 8.0f},
                                luisa::float4{0.0f, 6.0f, 94382.0f, 8.0f},
                                luisa::float4{3.4028234663852886e+38f,
                                              4294967295.0f,
                                              4294967295.0f, 0.0f},
                                luisa::float4{0.0f, 29.0f,
                                              457043.0f, 10.0f},
                                luisa::float4{0.0f, 29.0f,
                                              457046.0f, 10.0f}};
  for (auto index = std::size_t{0u}; index < expected.size(); ++index) {
    if (!equal_record(actual[index], expected[index])) {
      std::cerr << "scene traversal failed on " << backend << " at record "
                << index << ": got {" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << "}, expected {" << expected[index].x
                << ", " << expected[index].y << ", " << expected[index].z
                << ", " << expected[index].w << "}\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
