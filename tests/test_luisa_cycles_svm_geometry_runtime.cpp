#include "path_tracer_cycles_svm_scene.h"
#include "path_tracer_scene_geometry.h"

#include <psycles/compiler/core_nodes.h>

#include <array>
#include <cstdint>
#include <cstdlib>
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

  const std::map<GeometryId, std::uint32_t> missing_offsets;
  require(!finalize_cycles_svm_geometry_runtime(scene, snapshot, uploads,
                                                resources, missing_offsets,
                                                no_curve_offsets, diagnostic) &&
              scene->cycles_svm->geometry == nullptr,
          "rejected geometry transaction installed a partial runtime");
  require(finalize_cycles_svm_geometry_runtime(scene, snapshot, uploads,
                                               resources, primitive_offsets,
                                               no_curve_offsets, diagnostic),
          diagnostic);
  auto *const installed = scene->cycles_svm->geometry.get();
  require(installed != nullptr && installed->image.valid,
          "valid geometry transaction was not installed");
  require(!finalize_cycles_svm_geometry_runtime(scene, snapshot, uploads,
                                                resources, primitive_offsets,
                                                no_curve_offsets, diagnostic) &&
              scene->cycles_svm->geometry.get() == installed,
          "duplicate finalization replaced the installed transaction");

  auto stream = device.create_stream();
  upload_cycles_svm_geometry_runtime(stream, *scene->cycles_svm);
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
  const auto curves = download(stream, geometry.curve_buffer);

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
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  test_runtime(device);
  return EXIT_SUCCESS;
}
