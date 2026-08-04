#include "cycles_shader_identity.h"
#include "path_kernel_curve_geometry.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_scene_traversal.h"

#include <psycles/luisa/surface_ray.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::luisa_backend::surface_ray::invalid_primitive;

inline constexpr std::size_t record_count = 16u;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool equal_record(luisa::float4 actual,
                                luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto scene = std::make_shared<LuisaSceneData>();

  constexpr auto curve_bindless_base = geometry_bindless_stride;
  constexpr std::array geometries{
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
                  .curve_subdivision_level = 2u}};
  constexpr std::array instances{
      InstanceGpu{.geometry_index = 0u, .cycles_object_index = 11u},
      InstanceGpu{.geometry_index = 1u,
                  .override_offset = 0u,
                  .override_count = 1u,
                  .cycles_object_index = 22u},
      InstanceGpu{.geometry_index = 0u, .cycles_object_index = 489u},
      InstanceGpu{.geometry_index = 0u, .cycles_object_index = 1936u}};
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

  scene->heap = device.create_bindless_array(2u * geometry_bindless_stride);
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
  scene->accel = device.create_accel();
  scene->accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false, 0u);
  scene->accel.emplace_back(curves, make_float4x4(1.0f), 0xffu, 1u);
  const auto coincident_transform = translation(make_float3(5.0f, 0.0f, 0.0f));
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 2u);
  scene->accel.emplace_back(mesh, coincident_transform, 0xffu, false, 3u);

  auto output = device.create_buffer<luisa::float4>(record_count);
  const auto traversal = make_scene_traversal_component();
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

    const auto ray_x = select(0.0f, 5.0f, test >= 10u);
    const auto ray_z = select(0.0f, 4.0f,
                              (test == 12u) | (test == 13u));
    const auto ray_maximum = select(10.0f, 4.0f, test >= 14u);
    const auto ray = make_ray(make_float3(ray_x, 0.0f, ray_z),
                              make_float3(0.0f, 0.0f, 1.0f), 0.0f,
                              ray_maximum);
    const auto hit = traversal->closest_shadow(
        scene, ray, 0xffu,
        {.object = source_object, .primitive = source_primitive},
        {.object = light_object, .primitive = light_primitive});
    records.write(test,
                  make_float4(hit->committed_ray_t, cast<float>(hit->inst),
                              cast<float>(hit->prim),
                              select(0.0f, 1.0f, hit->is_procedural())));

    $if(test >= 10u) {
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
         << scene->geometry_material_buffer.copy_from(
                luisa::span{geometry_materials})
         << scene->override_material_buffer.copy_from(
                luisa::span{override_materials})
         << vertex_buffer.copy_from(luisa::span{vertices})
         << triangle_buffer.copy_from(luisa::span{triangles})
         << bounds_buffer.copy_from(luisa::span{curve_bounds})
         << segment_buffer.copy_from(luisa::span{curve_segments})
         << key_buffer.copy_from(luisa::span{curve_keys})
         << curve_material_buffer.copy_from(luisa::span{curve_material_slots})
         << curve_intercept_buffer.copy_from(luisa::span{curve_intercepts})
         << curve_length_buffer.copy_from(luisa::span{curve_lengths})
         << curve_random_buffer.copy_from(luisa::span{curve_randoms})
         << scene->heap.update() << mesh.build() << curves.build()
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
                                luisa::float4{4.0f, 489.0f, 100.0f, 2.0f}};
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
