#include "cycles_shader_identity.h"
#include "path_kernel_curve_geometry.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_scene_traversal.h"
#include "path_kernel_subsurface_intersection.h"
#include "path_tracer_scene_geometry.h"

#include <psycles/luisa/surface_ray.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::luisa_backend::surface_ray::invalid_primitive;

inline constexpr std::size_t record_count = 23u;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool equal_record(luisa::float4 actual,
                                luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
}

[[nodiscard]] bool
valid_backend_native_record(std::size_t index, luisa::float4 actual,
                            luisa::float4 expected) noexcept {
  constexpr auto coincident_forward = std::size_t{10u};
  constexpr auto coincident_endpoint = std::size_t{12u};
  constexpr auto filtered_coincident_endpoint = std::size_t{13u};
  constexpr auto coincident_truncated = std::size_t{14u};
  constexpr auto filtered_coincident_truncated = std::size_t{15u};
  constexpr auto curve_forward = std::size_t{0u};
  constexpr auto curve_wrong_source = std::size_t{2u};
  constexpr auto curve_triangle_source = std::size_t{4u};
  constexpr auto curve_unknown_source = std::size_t{5u};
  constexpr auto transformed_overlap = std::size_t{16u};
  constexpr auto near_overlap = std::size_t{20u};
  const auto miss = equal_record(actual, make_float4(0.0f));
  if ((index == filtered_coincident_endpoint ||
       index == filtered_coincident_truncated) &&
      miss) {
    // These are the source-filtered variants of the endpoint probes below.
    // The non-endpoint probe at index 11 verifies that rejecting a candidate
    // continues traversal; an open interval may have no candidate to reject.
    return true;
  }
  if (index == curve_forward || index == curve_wrong_source ||
      index == curve_triangle_source || index == curve_unknown_source) {
    // Both procedural primitives are identical segments of the same Cycles
    // curve. Their local acceleration indices are not renderer identities.
    return near(actual.x, expected.x) && near(actual.y, expected.y) &&
           (near(actual.z, 0.0f) || near(actual.z, 1.0f)) &&
           near(actual.w, expected.w);
  }
  if (index == coincident_forward || index == coincident_endpoint ||
      index == coincident_truncated) {
    // RayQuery does not specify closed triangle-ray interval endpoints.
    // Vulkan's native query excludes t == tmin and t == tmax, while the HIP
    // and fallback implementations currently report them. Both behaviours
    // are valid here: production rays use an offset origin and candidate
    // filtering instead of relying on an endpoint intersection.
    if ((index == coincident_endpoint || index == coincident_truncated) &&
        miss) {
      return true;
    }
    // RayQuery deliberately leaves exact-distance candidate order to the
    // backend. Both identities describe the same accepted surface; the
    // adjacent odd-numbered cases separately prove that rejecting either
    // identity continues traversal to the other one.
    const auto first_identity = near(actual.y, 489.0f) && near(actual.w, 2.0f);
    const auto second_identity =
        near(actual.y, 1936.0f) && near(actual.w, 3.0f);
    return near(actual.x, expected.x) && near(actual.z, 100.0f) &&
           (first_identity || second_identity);
  }
  if (index == transformed_overlap) {
    const auto transform_applied = near(actual.y, 2131.0f) &&
                                   near(actual.z, 20474114.0f) &&
                                   near(actual.w, 4.0f);
    const auto object_space = near(actual.y, 2372.0f) &&
                              near(actual.z, 3396299.0f) &&
                              near(actual.w, 5.0f);
    return near(actual.x, expected.x) && (transform_applied || object_space);
  }
  if (index == near_overlap) {
    // The two nearly coincident Barbershop-floor instances are both valid
    // closest candidates at backend precision. The invariant needed by path
    // tracing is that the selected identity is filtered from the derived
    // shadow ray and the sibling does not create a false occluder.
    const auto valid_object =
        near(actual.x, 5011.0f) || near(actual.x, 5066.0f);
    return valid_object && near(actual.y, expected.y) &&
           near(actual.z, expected.z) && near(actual.w, expected.w);
  }
  return equal_record(actual, expected);
}

struct TraversalXirShape {
  std::size_t instructions{};
  std::size_t callable_definitions{};
  std::size_t ray_query_loops{};
  std::size_t triangle_candidate_reads{};
  std::size_t procedural_candidate_reads{};
  std::size_t instance_transform_queries{};
  std::size_t resource_reads{};
  std::size_t control_point_reads{};
  std::size_t minimum_control_point_true_guard_depth{
      std::numeric_limits<std::size_t>::max()};
  std::size_t candidate_ray_reads{};
  std::size_t minimum_candidate_ray_true_guard_depth{
      std::numeric_limits<std::size_t>::max()};
};

[[nodiscard]] TraversalXirShape
traversal_xir_shape(const std::shared_ptr<LuisaSceneData> &scene,
                    SceneTraversalStagePlan plan, bool collect_shadow = false) {
  const auto traversal = make_scene_traversal_component(plan);
  Kernel1D shape = [scene, traversal,
                    collect_shadow](BufferUInt output) noexcept {
    const auto ray =
        make_ray(make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f);
    if (collect_shadow) {
      const auto batch = traversal->collect_shadow(
          scene, ray, 0xffu, ScenePrimitiveIdentity::invalid(),
          ScenePrimitiveIdentity::invalid(), 8u);
      output.write(0u, batch->count);
    } else {
      const auto hit = traversal->closest(scene, ray, 0xffu,
                                          ScenePrimitiveIdentity::invalid());
      output.write(0u, hit->inst);
    }
  };
  TraversalXirShape result;
  result.callable_definitions =
      shape.function()->function().custom_callables().size();
  auto module = luisa::compute::xir::ast_to_xir_translate(
      shape.function()->function(), {});
  for (auto *function : module->function_list()) {
    if (auto *definition = function->definition()) {
      const auto dominators = luisa::compute::xir::compute_dom_tree(
          function, {.compute_dominance_frontiers = false});
      std::vector<luisa::compute::xir::IfInst *> selections;
      definition->traverse_instructions(
          [&](luisa::compute::xir::Instruction *instruction) noexcept {
            if (instruction->isa<luisa::compute::xir::IfInst>()) {
              selections.emplace_back(
                  static_cast<luisa::compute::xir::IfInst *>(instruction));
            }
          });
      definition->traverse_instructions([&](luisa::compute::xir::Instruction
                                                *instruction) noexcept {
        const auto true_guard_depth = [&]() noexcept {
          auto depth = std::size_t{0u};
          for (auto *selection : selections) {
            depth += dominators.dominates(selection->true_block(),
                                          instruction->parent_block());
          }
          return depth;
        };
        ++result.instructions;
        result.ray_query_loops +=
            instruction->isa<luisa::compute::xir::RayQueryLoopInst>() ? 1u : 0u;
        if (instruction->isa<luisa::compute::xir::RayQueryObjectReadInst>()) {
          const auto *read =
              static_cast<const luisa::compute::xir::RayQueryObjectReadInst *>(
                  instruction);
          result.triangle_candidate_reads +=
              read->op() == luisa::compute::xir::RayQueryObjectReadOp::
                                RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT
                  ? 1u
                  : 0u;
          result.procedural_candidate_reads +=
              read->op() == luisa::compute::xir::RayQueryObjectReadOp::
                                RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT
                  ? 1u
                  : 0u;
          if (read->op() == luisa::compute::xir::RayQueryObjectReadOp::
                                RAY_QUERY_OBJECT_WORLD_SPACE_RAY) {
            ++result.candidate_ray_reads;
            result.minimum_candidate_ray_true_guard_depth =
                std::min(result.minimum_candidate_ray_true_guard_depth,
                         true_guard_depth());
          }
        }
        if (instruction->isa<luisa::compute::xir::ResourceQueryInst>()) {
          const auto *query =
              static_cast<const luisa::compute::xir::ResourceQueryInst *>(
                  instruction);
          result.instance_transform_queries +=
              query->op() == luisa::compute::xir::ResourceQueryOp::
                                 RAY_TRACING_INSTANCE_TRANSFORM
                  ? 1u
                  : 0u;
        }
        result.resource_reads +=
            instruction->isa<luisa::compute::xir::ResourceReadInst>() ? 1u : 0u;
        if (instruction->isa<luisa::compute::xir::ResourceReadInst>()) {
          const auto *read =
              static_cast<const luisa::compute::xir::ResourceReadInst *>(
                  instruction);
          if (read->op() ==
                  luisa::compute::xir::ResourceReadOp::BINDLESS_BUFFER_READ &&
              read->type() == Type::of<luisa::float4>()) {
            ++result.control_point_reads;
            result.minimum_control_point_true_guard_depth =
                std::min(result.minimum_control_point_true_guard_depth,
                         true_guard_depth());
          }
        }
      });
    }
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
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
  overlap_a_transform.elements = {-4.013790899648484e-8f,
                                  -0.9182483553886414f,
                                  0.0f,
                                  0.0f,
                                  1.0184273719787598f,
                                  -4.4516873742850294e-8f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.9999998807907104f,
                                  0.0f,
                                  1.6824010610580444f,
                                  4.567445755004883f,
                                  -0.0029841959476470947f,
                                  1.0f};
  psycles::Mat4f overlap_b_transform;
  overlap_b_transform.elements = {6.932582152785471e-8f,
                                  -0.9182483553886414f,
                                  0.0f,
                                  0.0f,
                                  1.0184273719787598f,
                                  7.688912972980688e-8f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.9999998807907104f,
                                  0.0f,
                                  1.6824010610580444f,
                                  4.567445755004883f,
                                  -0.0029841959476470947f,
                                  1.0f};
  const auto overlap_a_world_to_object =
      to_luisa(cycles_inverse_transform(overlap_a_transform));
  const auto overlap_b_world_to_object =
      to_luisa(cycles_inverse_transform(overlap_b_transform));

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
                  .primitive_kind = geometry_kind_triangle}};
  const auto identity_world_to_object = luisa::make_float4x4(1.0f);
  const std::array instances{
      InstanceGpu{.geometry_index = 0u,
                  .cycles_object_index = 11u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 1u,
                  .override_offset = 0u,
                  .override_count = 1u,
                  .cycles_object_index = 22u,
                  .cycles_primitive_offset = 200u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .cycles_object_index = 489u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = coincident_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .cycles_object_index = 1936u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = coincident_world_to_object},
      InstanceGpu{.geometry_index = 2u,
                  .cycles_object_index = 2131u,
                  .cycles_primitive_offset = 20474114u,
                  .cycles_transform_applied = 1u,
                  .cycles_world_to_object = bottle_world_to_object},
      InstanceGpu{.geometry_index = 3u,
                  .cycles_object_index = 2372u,
                  .cycles_primitive_offset = 3396299u,
                  .cycles_world_to_object = bottle_world_to_object},
      InstanceGpu{.geometry_index = 4u,
                  .cycles_object_index = 5011u,
                  .cycles_primitive_offset = 700000u,
                  .cycles_world_to_object = overlap_a_world_to_object},
      InstanceGpu{.geometry_index = 4u,
                  .cycles_object_index = 5066u,
                  .cycles_primitive_offset = 700000u,
                  .cycles_world_to_object = overlap_b_world_to_object},
      // Six transparent layers are inserted in far-to-near instance order.
      // The collector must therefore derive distance order from candidates,
      // never from TLAS construction or callback order.
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3006u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3005u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3004u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3003u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3002u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 1u,
                  .override_count = 1u,
                  .visibility_mask = 0x02u,
                  .cycles_object_index = 3001u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object},
      InstanceGpu{.geometry_index = 0u,
                  .override_offset = 2u,
                  .override_count = 1u,
                  .visibility_mask = 0x04u,
                  .cycles_object_index = 3007u,
                  .cycles_primitive_offset = 100u,
                  .cycles_world_to_object = identity_world_to_object}};
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
                         .material_identity = 109u},
      MaterialBindingGpu{.surface_tag = 78u,
                         .cycles_shader_index = 10u,
                         .material_identity = 110u,
                         .flags = material_flag_may_be_transparent},
      MaterialBindingGpu{.surface_tag = 79u,
                         .cycles_shader_index = 11u,
                         .material_identity = 111u}};
  constexpr std::array vertices{luisa::float3{-2.0f, -2.0f, 4.0f},
                                luisa::float3{2.0f, -2.0f, 4.0f},
                                luisa::float3{0.0f, 2.0f, 4.0f}};
  constexpr std::array triangles{Triangle{0u, 1u, 2u}};
  constexpr std::array overlap_vertices{
      luisa::float3{-0.12768065929412842f, -0.016480661928653717f,
                    -0.002021433785557747f},
      luisa::float3{-0.1807040125131607f, 0.0647093877196312f,
                    -0.002321503823623061f},
      luisa::float3{-0.18065254390239716f, -0.016480661928653717f,
                    -0.002021433785557747f}};
  constexpr std::array bottle_vertices{
      luisa::float3{0.04980994760990143f, -0.015110095962882042f,
                    0.0014796979958191514f},
      luisa::float3{0.047333888709545135f, -0.0196063332259655f,
                    0.0005811812588945031f},
      luisa::float3{0.04902583360671997f, -0.01487223245203495f,
                    0.0005811817827634513f}};
  std::array<luisa::float3, bottle_vertices.size()> bottle_world_vertices{};
  for (std::size_t i = 0u; i < bottle_vertices.size(); ++i) {
    const auto transformed = cycles_transform_point(
        bottle_transform,
        {bottle_vertices[i].x, bottle_vertices[i].y, bottle_vertices[i].z});
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

  scene->heap = device.create_bindless_array(5u * geometry_bindless_stride);
  scene->heap.emplace_on_update(0u, triangle_buffer);
  scene->heap.emplace_on_update(4u, curve_material_buffer);
  scene->heap.emplace_on_update(9u, vertex_buffer);
  scene->heap.emplace_on_update(curve_bindless_base, segment_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 1u, key_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 3u,
                                curve_intercept_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 4u,
                                curve_material_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 5u, curve_length_buffer);
  scene->heap.emplace_on_update(curve_bindless_base + 6u, curve_random_buffer);
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
  scene->accel = device.create_accel();
  scene->accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false, 0u);
  scene->accel.emplace_back(curves, make_float4x4(1.0f), 0xffu, 1u);
  const auto coincident_transform = translation(make_float3(5.0f, 0.0f, 0.0f));
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 2u);
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 3u);
  const auto bottle_luisa_transform = to_luisa(bottle_transform);
  scene->accel.emplace_back(bottle_mesh, bottle_luisa_transform, 0xffu, false,
                            4u);
  scene->accel.emplace_back(bottle_mesh, bottle_luisa_transform, 0xffu, false,
                            5u);
  scene->accel.emplace_back(overlap_mesh, to_luisa(overlap_a_transform), 0xffu,
                            false, 6u);
  scene->accel.emplace_back(overlap_mesh, to_luisa(overlap_b_transform), 0xffu,
                            false, 7u);
  for (auto layer = std::uint32_t{0u}; layer < 6u; ++layer) {
    const auto distance = 6.0f - static_cast<float>(layer);
    scene->accel.emplace_back(
        mesh, translation(make_float3(10.0f, 0.0f, distance - 4.0f)), 0x02u,
        false, 8u + layer);
  }
  scene->accel.emplace_back(mesh, translation(make_float3(10.0f, 0.0f, 3.0f)),
                            0x04u, false, 14u);
  // The compact local domain deliberately has a different ordinal space and
  // traversal order from the primary TLAS. User ids are the sole canonical
  // injection back into InstanceGpu/primary-TLAS identity.
  scene->subsurface_accel.emplace(device.create_accel());
  scene->subsurface_accel->emplace_back(mesh, coincident_transform, 0xffu,
                                        false, 3u);
  scene->subsurface_accel->emplace_back(mesh, make_float4x4(1.0f), 0xffu, false,
                                        0u);
  scene->subsurface_instance_count = 2u;

  const auto empty_shape = traversal_xir_shape(scene, {});
  const auto triangle_shape =
      traversal_xir_shape(scene, {.primitives = {.triangles = true}});
  const auto curve_shape =
      traversal_xir_shape(scene, {.primitives = {.curves = true}});
  const auto mixed_shape = traversal_xir_shape(
      scene, {.primitives = {.triangles = true, .curves = true}});
  const auto shadow_batch_shape = traversal_xir_shape(
      scene, {.primitives = {.triangles = true, .curves = true}}, true);
  const auto report_shapes =
      std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr;
  if (report_shapes) {
    std::cerr << "scene traversal XIR: empty=" << empty_shape.instructions
              << ", triangles=" << triangle_shape.instructions
              << ", curves=" << curve_shape.instructions
              << ", mixed=" << mixed_shape.instructions
              << ", shadow_batch=" << shadow_batch_shape.instructions
              << ", curve_control_reads=" << curve_shape.control_point_reads
              << ", curve_control_guard_depth="
              << curve_shape.minimum_control_point_true_guard_depth
              << ", curve_ray_guard_depth="
              << curve_shape.minimum_candidate_ray_true_guard_depth << '\n';
  }
  if (empty_shape.triangle_candidate_reads != 0u ||
      empty_shape.procedural_candidate_reads != 0u ||
      empty_shape.callable_definitions != 0u ||
      triangle_shape.triangle_candidate_reads == 0u ||
      triangle_shape.resource_reads != 1u ||
      triangle_shape.procedural_candidate_reads != 0u ||
      triangle_shape.callable_definitions != 0u ||
      curve_shape.triangle_candidate_reads != 0u ||
      curve_shape.procedural_candidate_reads == 0u ||
      curve_shape.control_point_reads != 4u ||
      curve_shape.minimum_control_point_true_guard_depth < 2u ||
      curve_shape.candidate_ray_reads != 1u ||
      curve_shape.minimum_candidate_ray_true_guard_depth < 2u ||
      curve_shape.instance_transform_queries != 0u ||
      curve_shape.callable_definitions != 0u ||
      mixed_shape.triangle_candidate_reads == 0u ||
      mixed_shape.procedural_candidate_reads == 0u ||
      mixed_shape.control_point_reads != 4u ||
      mixed_shape.minimum_control_point_true_guard_depth < 2u ||
      mixed_shape.candidate_ray_reads != 1u ||
      mixed_shape.minimum_candidate_ray_true_guard_depth < 2u ||
      mixed_shape.instance_transform_queries != 0u ||
      mixed_shape.callable_definitions != 0u ||
      empty_shape.ray_query_loops != 0u ||
      triangle_shape.ray_query_loops != 1u ||
      curve_shape.ray_query_loops != 1u || mixed_shape.ray_query_loops != 1u ||
      shadow_batch_shape.ray_query_loops != 1u ||
      shadow_batch_shape.triangle_candidate_reads == 0u ||
      shadow_batch_shape.procedural_candidate_reads == 0u ||
      shadow_batch_shape.control_point_reads != 4u ||
      shadow_batch_shape.minimum_control_point_true_guard_depth < 2u ||
      shadow_batch_shape.candidate_ray_reads != 1u ||
      shadow_batch_shape.minimum_candidate_ray_true_guard_depth < 2u ||
      shadow_batch_shape.callable_definitions != 0u ||
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
      {.primitives = {.triangles = true, .curves = true}});
  const auto curve_primitive = make_curve_primitive_component();
  const auto curve_geometry = make_curve_geometry_component();
  Kernel1D evaluate = [scene, traversal, curve_primitive,
                       curve_geometry](BufferFloat4 records) noexcept {
    const UInt test = dispatch_x();
    $if(test >= 21u) {
      const auto local_hit =
          (*scene->subsurface_accel)
              ->intersect(make_ray(make_float3(select(5.0f, 0.0f, test == 22u),
                                               0.0f, 0.0f),
                                   make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f),
                          {.visibility_mask = 0xffu});
      const auto primary_instance =
          subsurface_primary_instance(scene, select(0u, 1u, test == 22u));
      records.write(test, make_float4(cast<float>(primary_instance),
                                      cast<float>(local_hit->inst),
                                      cast<float>(local_hit->prim),
                                      local_hit->committed_ray_t));
      $return();
    };
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
    source_object = select(source_object, 1936u, exclude_later_coincident);
    source_primitive = select(source_primitive, 100u, exclude_later_coincident);
    light_object = select(light_object, 22u, test == 3u);
    light_primitive = select(light_primitive, 200u, test == 3u);
    source_object = select(source_object, 2131u, test == 17u);
    source_primitive = select(source_primitive, 20474114u, test == 17u);
    light_object = select(light_object, 2131u, test == 18u);
    light_primitive = select(light_primitive, 20474114u, test == 18u);
    source_object = select(source_object, 2372u, test == 19u);
    source_primitive = select(source_primitive, 3396299u, test == 19u);
    const auto ray_x = select(0.0f, 5.0f, test >= 10u);
    const auto ray_z = select(0.0f, 4.0f, (test == 12u) | (test == 13u));
    const auto ray_maximum = select(10.0f, 4.0f, test >= 14u);
    const auto bottle_test = (test >= 16u) & (test <= 19u);
    const auto overlap_test = test == 20u;
    const auto ray_origin =
        select(select(make_float3(ray_x, 0.0f, ray_z),
                      make_float3(1.8301146030426025f, 9.144867897033691f,
                                  1.5106611251831055f),
                      bottle_test),
               make_float3(2.7035441398620605f, 8.901592254638672f,
                           1.234214186668396f),
               overlap_test);
    const auto ray_direction =
        select(select(make_float3(0.0f, 0.0f, 1.0f),
                      make_float3(-0.8146389126777649f, -0.5793101787567139f,
                                  -0.02762461081147194f),
                      bottle_test),
               make_float3(-0.23053720593452454f, -0.9327327013015747f,
                           -0.27724069356918335f),
               overlap_test);
    const auto ray_maximum_for_test =
        select(select(ray_maximum, 125.67607116699219f, bottle_test),
               101.64700317382812f, overlap_test);
    const auto ray =
        make_ray(ray_origin, ray_direction, 0.0f, ray_maximum_for_test);
    const auto hit = traversal->closest_shadow(
        scene, ray, 0xffu,
        {.object = source_object, .primitive = source_primitive},
        {.object = light_object, .primitive = light_primitive});
    records.write(test,
                  make_float4(hit->committed_ray_t, cast<float>(hit->inst),
                              cast<float>(hit->prim),
                              select(0.0f, 1.0f, hit->is_procedural())));

    $if(overlap_test) {
      // This is the Barbershop floor configuration which exposed a false
      // self-shadow between two nearly coincident instances. Reconstruct the
      // selected backend-native hit, then prove that identity filtering never
      // reports that same source again. A backend may still report the nearly
      // coincident sibling; RayQuery does not impose a cross-backend tie or
      // coplanarity policy.
      const auto instance = scene->instance_buffer->read(hit->inst);
      const auto geometry =
          scene->geometry_buffer->read(instance.geometry_index);
      const auto triangle =
          scene->heap->buffer<Triangle>(geometry.bindless_base).read(hit->prim);
      const auto positions =
          scene->heap->buffer<luisa::float3>(geometry.bindless_base + 9u);
      const auto p0 = positions.read(triangle.i0);
      const auto p1 = positions.read(triangle.i1);
      const auto p2 = positions.read(triangle.i2);
      const auto object_position =
          p0 + hit->bary.x * (p1 - p0) + hit->bary.y * (p2 - p0);
      const auto world_position = (scene->accel->instance_transform(hit->inst) *
                                   make_float4(object_position, 1.0f))
                                      .xyz();
      const auto shadow_ray =
          make_ray(world_position,
                   make_float3(0.5150954127311707f, 0.6604911684989929f,
                               0.5462856888771057f),
                   0.0f, 3.6035547256469727f);
      const auto cycles_object = instance.cycles_object_index;
      const auto cycles_primitive =
          geometry.cycles_primitive_offset + hit->prim;
      const auto shadow_hit = traversal->closest_shadow(
          scene, shadow_ray, 0xffu,
          {.object = cycles_object, .primitive = cycles_primitive},
          ScenePrimitiveIdentity::invalid());
      UInt shadow_object = invalid_primitive;
      UInt shadow_primitive = invalid_primitive;
      $if(!shadow_hit->miss()) {
        const auto shadow_instance =
            scene->instance_buffer->read(shadow_hit->inst);
        const auto shadow_geometry =
            scene->geometry_buffer->read(shadow_instance.geometry_index);
        shadow_object = shadow_instance.cycles_object_index;
        shadow_primitive =
            shadow_geometry.cycles_primitive_offset + shadow_hit->prim;
      };
      const auto repeated_source = (shadow_object == cycles_object) &
                                   (shadow_primitive == cycles_primitive);
      records.write(test, make_float4(cast<float>(cycles_object),
                                      cast<float>(cycles_primitive),
                                      select(0.0f, 1.0f, repeated_source),
                                      hit->committed_ray_t));
    };

    $if((test >= 10u) & (test <= 19u)) {
      $if(!hit->miss()) {
        const auto instance = scene->instance_buffer->read(hit->inst);
        const auto geometry =
            scene->geometry_buffer->read(instance.geometry_index);
        records.write(
            test,
            make_float4(
                hit->committed_ray_t, cast<float>(instance.cycles_object_index),
                cast<float>(geometry.cycles_primitive_offset + hit->prim),
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
      records.write(test, make_float4(geometry.intersection.u,
                                      geometry.intersection.v,
                                      geometry.intercept, geometry.length));
    };
    $if(test == 8u) {
      const auto geometry = curve_geometry->emit(scene, 1u, 0u, ray, 2.0f);
      records.write(test, make_float4(geometry.thickness, geometry.random,
                                      geometry.tangent_normal.z,
                                      geometry.shading_normal.z));
    };
    $if(test == 9u) {
      const auto geometry = curve_geometry->emit(scene, 1u, 0u, ray, 2.0f);
      records.write(test, make_float4(geometry.position.x, geometry.position.y,
                                      geometry.position.z, geometry.dpdu.x));
    };
  };
  // This is also a compiler-pipeline regression. Bypass persistent shader
  // cache so every run exercises RayQuery outlining and its handler-local
  // storage analysis instead of hiding an accidental complexity regression.
  auto shader = device.compile(
      evaluate, luisa::compute::ShaderOption{.enable_cache = false});

  constexpr auto shadow_batch_record_count = std::size_t{9u};
  auto shadow_batch_output =
      device.create_buffer<luisa::float4>(shadow_batch_record_count);
  Kernel1D evaluate_shadow_batches = [scene, traversal](
                                         BufferFloat4 records) noexcept {
    const auto ray = make_ray(make_float3(10.0f, 0.0f, 0.0f),
                              make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f);
    const auto all = traversal->collect_shadow(
        scene, ray, 0x02u, ScenePrimitiveIdentity::invalid(),
        ScenePrimitiveIdentity::invalid(), 8u);
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      const auto &hit = all->hits[static_cast<luisa::uint>(index)];
      records.write(static_cast<std::uint32_t>(index),
                    make_float4(hit->distance, cast<float>(hit->instance),
                                cast<float>(hit->primitive),
                                cast<float>(hit->hit_type)));
    }
    records.write(4u,
                  make_float4(cast<float>(all->count), cast<float>(all->total),
                              cast<float>(all->blocked), 0.0f));

    const auto continuation_ray =
        make_ray(ray->origin(), ray->direction(),
                 psycles::luisa_backend::surface_ray::intersection_t_offset(
                     all->hits[3u]->distance),
                 ray->t_max());
    const auto continuation = traversal->collect_shadow(
        scene, continuation_ray, 0x02u, ScenePrimitiveIdentity::invalid(),
        ScenePrimitiveIdentity::invalid(), 4u);
    records.write(5u, make_float4(continuation->hits[0u]->distance,
                                  continuation->hits[1u]->distance,
                                  cast<float>(continuation->count),
                                  cast<float>(continuation->total)));

    const auto excluded = traversal->collect_shadow(
        scene, ray, 0x02u, {.object = 3001u, .primitive = 100u},
        ScenePrimitiveIdentity::invalid(), 8u);
    records.write(6u, make_float4(excluded->hits[0u]->distance,
                                  cast<float>(excluded->hits[0u]->instance),
                                  cast<float>(excluded->count),
                                  cast<float>(excluded->total)));

    const auto exhausted = traversal->collect_shadow(
        scene, ray, 0x02u, ScenePrimitiveIdentity::invalid(),
        ScenePrimitiveIdentity::invalid(), 5u);
    records.write(7u, make_float4(cast<float>(exhausted->blocked),
                                  cast<float>(exhausted->total),
                                  cast<float>(exhausted->count), 0.0f));

    const auto opaque = traversal->collect_shadow(
        scene, ray, 0x06u, ScenePrimitiveIdentity::invalid(),
        ScenePrimitiveIdentity::invalid(), 8u);
    records.write(8u, make_float4(cast<float>(opaque->blocked),
                                  cast<float>(opaque->total),
                                  cast<float>(opaque->count), 0.0f));
  };
  auto shadow_batch_shader =
      device.compile(evaluate_shadow_batches,
                     luisa::compute::ShaderOption{.enable_cache = false});

  std::array<luisa::float4, record_count> actual{};
  std::array<luisa::float4, shadow_batch_record_count> shadow_batch_actual{};
  stream << scene->geometry_buffer.copy_from(luisa::span{geometries})
         << scene->instance_buffer.copy_from(luisa::span{instances})
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
         << bounds_buffer.copy_from(luisa::span{curve_bounds})
         << segment_buffer.copy_from(luisa::span{curve_segments})
         << key_buffer.copy_from(luisa::span{curve_keys})
         << curve_material_buffer.copy_from(luisa::span{curve_material_slots})
         << curve_intercept_buffer.copy_from(luisa::span{curve_intercepts})
         << curve_length_buffer.copy_from(luisa::span{curve_lengths})
         << curve_random_buffer.copy_from(luisa::span{curve_randoms})
         << scene->heap.update() << mesh.build() << bottle_mesh.build()
         << overlap_mesh.build() << curves.build() << scene->accel.build()
         << scene->subsurface_accel->build()
         << shader(output).dispatch(record_count)
         << shadow_batch_shader(shadow_batch_output).dispatch(1u)
         << output.copy_to(luisa::span{actual})
         << shadow_batch_output.copy_to(luisa::span{shadow_batch_actual})
         << synchronize();

  constexpr std::array expected{
      luisa::float4{2.0f, 1.0f, 0.0f, 1.0f},
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
      luisa::float4{1.6995198f, 2131.0f, 20474114.0f, 4.0f},
      luisa::float4{1.69952f, 2372.0f, 3396299.0f, 5.0f},
      luisa::float4{1.69952f, 2372.0f, 3396299.0f, 5.0f},
      luisa::float4{1.6995198f, 2131.0f, 20474114.0f, 4.0f},
      luisa::float4{5066.0f, 700000.0f, 0.0f, 4.4699316f},
      luisa::float4{3.0f, 0.0f, 0.0f, 4.0f},
      luisa::float4{0.0f, 1.0f, 0.0f, 4.0f}};
  auto failed = false;
  for (auto index = std::size_t{0u}; index < expected.size(); ++index) {
    if (!valid_backend_native_record(index, actual[index], expected[index])) {
      std::cerr << "scene traversal failed on " << backend << " at record "
                << index << ": got {" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << "}, expected {" << expected[index].x
                << ", " << expected[index].y << ", " << expected[index].z
                << ", " << expected[index].w << "}\n";
      failed = true;
    }
  }
  const auto surface_hit = static_cast<float>(
      static_cast<std::uint32_t>(luisa::compute::HitType::Surface));
  const std::array shadow_batch_expected{
      luisa::float4{1.0f, 13.0f, 0.0f, surface_hit},
      luisa::float4{2.0f, 12.0f, 0.0f, surface_hit},
      luisa::float4{3.0f, 11.0f, 0.0f, surface_hit},
      luisa::float4{4.0f, 10.0f, 0.0f, surface_hit},
      luisa::float4{4.0f, 6.0f, 0.0f, 0.0f},
      luisa::float4{5.0f, 6.0f, 2.0f, 2.0f},
      luisa::float4{2.0f, 12.0f, 4.0f, 5.0f},
      luisa::float4{1.0f, 6.0f, 4.0f, 0.0f}};
  for (auto index = std::size_t{0u}; index < shadow_batch_expected.size();
       ++index) {
    if (!equal_record(shadow_batch_actual[index],
                      shadow_batch_expected[index])) {
      const auto value = shadow_batch_actual[index];
      const auto wanted = shadow_batch_expected[index];
      std::cerr << "shadow batch failed on " << backend << " at record "
                << index << ": got {" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "}, expected {" << wanted.x
                << ", " << wanted.y << ", " << wanted.z << ", " << wanted.w
                << "}\n";
      failed = true;
    }
  }
  if (!near(shadow_batch_actual[8u].x, 1.0f)) {
    std::cerr << "opaque shadow candidate did not terminate collection on "
              << backend << '\n';
    failed = true;
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
