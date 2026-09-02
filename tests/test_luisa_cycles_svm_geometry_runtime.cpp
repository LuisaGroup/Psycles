#include "cycles_shader_identity.h"
#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_scene_geometry.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderGraph named_attribute_shader(std::string_view name) {
  ShaderGraph graph;
  const auto attribute = graph.add_node(node_type::attribute, "Attribute");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_property(attribute, "Attribute",
                             SocketValue::string(std::string{name})) &&
              graph.set_property(
                  attribute, "AttributeId",
                  SocketValue::unsigned_integer(attribute_id(name))) &&
              graph.connect({.node = attribute, .socket = "Color"}, emission,
                            "Color"),
          "could not build named-attribute material");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] SceneSnapshot make_scene() {
  constexpr MaterialId material_id{1u};
  constexpr MaterialId unused_material_id{5u};
  constexpr GeometryId geometry_id{2u};
  constexpr InstanceId instance_id{3u};
  SceneSnapshot scene;
  scene.revision = 1u;
  scene.materials.emplace(
      material_id,
      MaterialDesc{.name = "typed geometry runtime",
                   .shader = named_attribute_shader("later-by-shader-index"),
                   .cycles_shader_index = 7u});
  scene.materials.emplace(
      unused_material_id,
      MaterialDesc{.name = "unused Cycles geometry slot",
                   .shader = named_attribute_shader("earlier-by-shader-index"),
                   .cycles_shader_index = 2u});
  scene.geometries.emplace(
      geometry_id,
      TriangleMeshDesc{.name = "post-displacement triangle",
                       .positions = {{-10.0f, -10.0f, -10.0f},
                                     {-11.0f, -10.0f, -10.0f},
                                     {-10.0f, -11.0f, -10.0f}},
                       .triangles = {{0u, 1u, 2u}},
                       .material_slots = {material_id, unused_material_id},
                       .triangle_material_slots = {0u},
                       .cycles_primitive_offset = 3u});
  scene.instances.emplace(instance_id,
                          InstanceDesc{.name = "typed geometry runtime",
                                       .geometry = geometry_id,
                                       .cycles_object_index = 0u});
  scene.cycles_object_count = 1u;
  return scene;
}

[[nodiscard]] GeometryUpload make_finalized_upload() {
  GeometryUpload upload;
  upload.positions = {
      {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
  upload.cycles_intersection_positions = upload.positions;
  upload.normals = {{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  upload.triangles = {{0u, 2u, 1u}};
  upload.triangle_material_slots = {0u};
  upload.triangle_smooth = {0u};
  return upload;
}

template <typename T>
[[nodiscard]] std::vector<T> download(Stream &stream, Buffer<T> &buffer) {
  std::vector<T> result(buffer.size());
  stream << buffer.copy_to(luisa::span{result}) << synchronize();
  return result;
}

void verify_attribute_map(const std::vector<AttributeMap> &actual,
                          const std::vector<AttributeMap> &expected) {
  require(actual.size() == expected.size(),
          "attribute-map device extent changed");
  for (auto i = std::size_t{}; i < expected.size(); ++i) {
    require(actual[i].id == expected[i].id &&
                actual[i].offset == expected[i].offset &&
                actual[i].element == expected[i].element &&
                actual[i].type == expected[i].type &&
                actual[i].pad == expected[i].pad,
            "typed attribute-map upload changed");
  }
}

void verify_kernel_object_shader_view(Device &device, Stream &stream,
                                      Buffer<KernelObject> &object_buffer,
                                      const KernelObject &expected) {
  // A raw copy only proves the host-side buffer extent. Read one value from
  // every scalar-width/layout class through the DSL so this test also guards
  // KernelObject reflection and native backend field addressing.
  Kernel1D inspect = [](BufferVar<KernelObject> objects,
                        BufferFloat4 float_fields, BufferUInt4 integer_fields,
                        BufferULong wide_fields) noexcept {
    const auto object = objects.read(0u);
    float_fields.write(0u, make_float4(object.tfm.x.x, object.tfm.y.w,
                                       object.color.z, object.ao_distance));
    integer_fields.write(0u, make_uint4(cast<uint>(object.num_geom_steps),
                                        cast<uint>(object.numverts),
                                        object.visibility,
                                        cast<uint>(object.primitive_type)));
    wide_fields.write(0u, cast<luisa::ulong>(object.light_set_membership));
    wide_fields.write(1u, cast<luisa::ulong>(object.shadow_set_membership));
  };

  auto float_buffer = device.create_buffer<luisa::float4>(1u);
  auto integer_buffer = device.create_buffer<luisa::uint4>(1u);
  auto wide_buffer = device.create_buffer<luisa::ulong>(2u);
  auto shader = device.compile(inspect);
  auto probe = expected;
  probe.tfm.x.x = 1.25f;
  probe.tfm.y.w = -2.5f;
  probe.color.z = 0.75f;
  probe.ao_distance = 0.625f;
  probe.num_geom_steps = 0x4321u;
  probe.numverts = 0x01234567;
  probe.visibility = 0x89abcdefu;
  probe.primitive_type = 0x76543210;
  probe.light_set_membership = 0x1020304050607080ull;
  probe.shadow_set_membership = 0x8172635445362718ull;
  const std::array probe_records{probe};
  std::array<luisa::float4, 1u> float_fields{};
  std::array<luisa::uint4, 1u> integer_fields{};
  std::array<luisa::ulong, 2u> wide_fields{};
  stream << object_buffer.copy_from(luisa::span{probe_records})
         << shader(object_buffer, float_buffer, integer_buffer, wide_buffer)
                .dispatch(1u)
         << float_buffer.copy_to(luisa::span{float_fields})
         << integer_buffer.copy_to(luisa::span{integer_fields})
         << wide_buffer.copy_to(luisa::span{wide_fields}) << synchronize();

  require(float_fields[0u].x == probe.tfm.x.x &&
              float_fields[0u].y == probe.tfm.y.w &&
              float_fields[0u].z == probe.color.z &&
              float_fields[0u].w == probe.ao_distance,
          "device KernelObject float/nested-transform view changed");
  require(integer_fields[0u].x == probe.num_geom_steps &&
              integer_fields[0u].y ==
                  static_cast<std::uint32_t>(probe.numverts) &&
              integer_fields[0u].z == probe.visibility &&
              integer_fields[0u].w ==
                  static_cast<std::uint32_t>(probe.primitive_type),
          "device KernelObject 16/32-bit field view changed");
  require(wide_fields[0u] == probe.light_set_membership &&
              wide_fields[1u] == probe.shadow_set_membership,
          "device KernelObject 64-bit field view changed");
}

void test_runtime(Device &device) {
  constexpr GeometryId geometry_id{2u};
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  const auto snapshot = make_scene();

  ShaderCompiler shader_compiler{make_core_node_registry()};
  const auto reachability = build_scene_material_reachability(snapshot);
  const auto update = scene->materials.update(snapshot, shader_compiler,
                                              reachability.shader_materials);
  require(update.committed, "runtime material fixture did not compile");

  std::string diagnostic;
  scene->cycles_svm = build_cycles_svm_runtime(scene, snapshot, diagnostic);
  require(scene->cycles_svm != nullptr, diagnostic);
  require(scene->materials.find(MaterialId{5u}) == nullptr,
          "supplemental Cycles shader polluted legacy material reachability");
  const auto &named_attributes =
      scene->cycles_svm->compilation.named_attributes;
  require(named_attributes.size() == 2u &&
              named_attributes[0u].first == "earlier-by-shader-index" &&
              named_attributes[0u].second == ATTR_STD_NUM &&
              named_attributes[1u].first == "later-by-shader-index" &&
              named_attributes[1u].second == ATTR_STD_NUM + 1u,
          "Cycles shader-table resource identity ignored source shader order");

  const std::array uploads{make_finalized_upload()};
  const std::map<GeometryId, std::uint32_t> resources{{geometry_id, 0u}};
  const std::map<GeometryId, std::uint32_t> primitive_offsets{
      {geometry_id, 3u}};
  const std::map<GeometryId, std::uint32_t> no_curve_offsets;
  const auto intersection_plans =
      build_cycles_instance_intersection_plan(snapshot, {});

  const std::map<GeometryId, std::uint32_t> missing_offsets;
  require(!finalize_cycles_svm_scene_runtime(
              scene, snapshot, intersection_plans, uploads, resources,
              missing_offsets, no_curve_offsets, diagnostic) &&
              scene->cycles_svm->geometry == nullptr &&
              scene->cycles_svm->objects == nullptr,
          "rejected scene transaction installed a partial runtime");
  require(finalize_cycles_svm_scene_runtime(
              scene, snapshot, intersection_plans, uploads, resources,
              primitive_offsets, no_curve_offsets, diagnostic),
          diagnostic);
  auto *const installed_geometry = scene->cycles_svm->geometry.get();
  auto *const installed_objects = scene->cycles_svm->objects.get();
  require(installed_geometry != nullptr && installed_geometry->image.valid &&
              installed_objects != nullptr && installed_objects->image.valid,
          "valid geometry/object transaction was not installed");
  require(!finalize_cycles_svm_scene_runtime(
              scene, snapshot, intersection_plans, uploads, resources,
              primitive_offsets, no_curve_offsets, diagnostic) &&
              scene->cycles_svm->geometry.get() == installed_geometry &&
              scene->cycles_svm->objects.get() == installed_objects,
          "duplicate finalization replaced the installed transaction");

  auto stream = device.create_stream();
  upload_cycles_svm_scene_runtime(stream, *scene->cycles_svm);
  auto &geometry = *scene->cycles_svm->geometry;
  const auto attribute_map = download(stream, geometry.attribute_map_buffer);
  const auto attribute_float =
      download(stream, geometry.attribute_float_buffer);
  const auto attribute_float2 =
      download(stream, geometry.attribute_float2_buffer);
  const auto attribute_float3 =
      download(stream, geometry.attribute_float3_buffer);
  const auto attribute_float4 =
      download(stream, geometry.attribute_float4_buffer);
  const auto attribute_uchar4 =
      download(stream, geometry.attribute_uchar4_buffer);
  const auto attribute_normal =
      download(stream, geometry.attribute_normal_buffer);
  const auto triangle_vertices =
      download(stream, geometry.triangle_vertex_buffer);
  const auto curve_keys = download(stream, geometry.curve_key_buffer);
  const auto points = download(stream, geometry.point_buffer);
  const auto triangle_indices =
      download(stream, geometry.triangle_index_buffer);
  const auto triangle_shaders =
      download(stream, geometry.triangle_shader_buffer);
  const auto curves = download(stream, geometry.curve_buffer);
  auto &objects = *scene->cycles_svm->objects;
  const auto object_records = download(stream, objects.object_buffer);
  const auto object_flags = download(stream, objects.object_flag_buffer);

  const auto &image = geometry.image;
  verify_attribute_map(attribute_map, image.attributes.attribute_map);
  require(triangle_vertices.size() == 3u && triangle_vertices[0u].x == 1.0f &&
              triangle_vertices[2u].z == 9.0f,
          "typed triangle-vertex upload did not use finalized positions");
  require(attribute_normal.size() == 3u &&
              attribute_normal[0u].value ==
                  image.attributes.attributes_normal[0u].value &&
              attribute_normal[2u].value ==
                  image.attributes.attributes_normal[2u].value,
          "typed packed-normal upload changed");
  require(triangle_indices.size() == 4u && triangle_indices[3u].x == 0u &&
              triangle_indices[3u].y == 2u && triangle_indices[3u].z == 1u,
          "global typed triangle-index upload changed");
  require(triangle_shaders.size() == 4u &&
              triangle_shaders[3u] ==
                  cycles_shader_identity::surface(7u, false),
          "global typed tri_shader upload changed");

  require(
      attribute_float.size() == 1u && attribute_float[0u] == 0.0f &&
          attribute_float2.size() == 1u && attribute_float2[0u].x == 0.0f &&
          attribute_float2[0u].y == 0.0f && attribute_float3.size() == 1u &&
          attribute_float3[0u].x == 0.0f && attribute_float3[0u].y == 0.0f &&
          attribute_float3[0u].z == 0.0f && attribute_float4.size() == 1u &&
          attribute_float4[0u].x == 0.0f && attribute_float4[0u].y == 0.0f &&
          attribute_float4[0u].z == 0.0f && attribute_float4[0u].w == 0.0f &&
          attribute_uchar4.size() == 1u && attribute_uchar4[0u].x == 0u &&
          attribute_uchar4[0u].y == 0u && attribute_uchar4[0u].z == 0u &&
          attribute_uchar4[0u].w == 0u && curve_keys.size() == 1u &&
          curve_keys[0u].x == 0.0f && points.size() == 1u &&
          points[0u].x == 0.0f && curves.size() == 1u &&
          curves[0u].shader_id == 0 && curves[0u].first_key == 0 &&
          curves[0u].num_keys == 0 && curves[0u].type == 0,
      "empty semantic arrays did not upload unreachable zero sentinels");

  require(object_records.size() == objects.image.objects.size() &&
              object_flags == objects.image.object_flags &&
              std::memcmp(object_records.data(), objects.image.objects.data(),
                          object_records.size() * sizeof(KernelObject)) == 0,
          "typed KernelObject/object_flag upload changed the host image");
  require(object_records.size() == 1u && object_records[0u].numverts == 3 &&
              object_records[0u].numprims == 1 &&
              object_records[0u].primitive_type == PRIMITIVE_TRIANGLE &&
              (object_flags[0u] & SD_OBJECT_TRANSFORM_APPLIED) != 0u,
          "runtime object image did not retain finalized mesh state");
  verify_kernel_object_shader_view(device, stream, objects.object_buffer,
                                   objects.image.objects[0u]);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  test_runtime(device);
  return EXIT_SUCCESS;
}
