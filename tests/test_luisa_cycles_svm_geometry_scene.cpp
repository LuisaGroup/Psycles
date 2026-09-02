#include "cycles_shader_identity.h"
#include "path_tracer_cycles_svm_geometry.h"
#include "path_tracer_internal.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::luisa_backend::detail;

constexpr auto first_named_id = static_cast<std::uint64_t>(ATTR_STD_NUM);
constexpr auto mesh_uv_id = first_named_id;
constexpr auto mesh_color_id = first_named_id + 1u;
constexpr auto mesh_tangent_id = first_named_id + 2u;
constexpr auto mesh_tangent_sign_id = first_named_id + 3u;
constexpr auto curve_uv_id = first_named_id + 4u;
constexpr auto missing_light_attribute_id = first_named_id + 5u;

constexpr MaterialId mesh_material{10u};
constexpr MaterialId curve_material{11u};
constexpr MaterialId light_material{12u};
constexpr MaterialId world_material{13u};
constexpr GeometryId mesh_id{20u};
constexpr GeometryId curve_id{30u};
constexpr InstanceId mesh_instance_id{40u};
constexpr InstanceId curve_instance_id{41u};
constexpr LightId light_id{50u};

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

[[nodiscard]] const AttributeMap &
entry(const CyclesSvmGeometrySceneImage &image, std::uint32_t geometry_index,
      std::uint64_t id) {
  auto offset = static_cast<std::size_t>(
      image.attributes.geometries.at(geometry_index).attribute_map_offset);
  while (offset < image.attributes.attribute_map.size()) {
    const auto &candidate = image.attributes.attribute_map[offset];
    if (candidate.id == id) {
      return candidate;
    }
    require(candidate.id != static_cast<std::uint64_t>(ATTR_STD_NONE),
            "requested attribute is absent from the map");
    offset += static_cast<std::size_t>(ATTR_PRIM_TYPES);
  }
  require(false, "attribute map walk escaped the host image");
  return image.attributes.attribute_map.front();
}

struct Fixture {
  SceneSnapshot snapshot;
  CompiledShaderTable compilation;
  std::map<MaterialId, std::uint32_t> material_shader_indices;
  std::vector<GeometryUpload> uploads;
  std::map<GeometryId, std::uint32_t> resource_geometry_indices;
  std::map<GeometryId, std::uint32_t> triangle_primitive_offsets;
  std::map<GeometryId, std::uint32_t> curve_primitive_offsets;

  [[nodiscard]] CyclesSvmGeometrySceneImage build() const {
    const auto identities = plan_object_identities(snapshot);
    require(identities.valid, identities.diagnostic);
    return build_cycles_svm_geometry_scene_image(
        snapshot, compilation, material_shader_indices, identities, uploads,
        resource_geometry_indices, triangle_primitive_offsets,
        curve_primitive_offsets);
  }
};

[[nodiscard]] Fixture make_fixture() {
  Fixture fixture;
  fixture.compilation.table.valid = true;
  fixture.compilation.table.shader_count = 4u;
  fixture.compilation.shader_attribute_ids_in_request_order = {
      {mesh_color_id, static_cast<std::uint64_t>(ATTR_STD_VERTEX_COLOR),
       mesh_uv_id, mesh_tangent_id, mesh_tangent_sign_id,
       static_cast<std::uint64_t>(ATTR_STD_UV),
       static_cast<std::uint64_t>(ATTR_STD_UV_TANGENT),
       static_cast<std::uint64_t>(ATTR_STD_UV_TANGENT_SIGN),
       static_cast<std::uint64_t>(ATTR_STD_GENERATED),
       static_cast<std::uint64_t>(ATTR_STD_GENERATED_TRANSFORM),
       static_cast<std::uint64_t>(ATTR_STD_RANDOM_PER_ISLAND)},
      {curve_uv_id, static_cast<std::uint64_t>(ATTR_STD_CURVE_INTERCEPT),
       static_cast<std::uint64_t>(ATTR_STD_CURVE_LENGTH),
       static_cast<std::uint64_t>(ATTR_STD_CURVE_RANDOM)},
      {missing_light_attribute_id},
      {}};
  fixture.compilation.named_attributes = {
      {"UVMap", mesh_uv_id},
      {"Color", mesh_color_id},
      {"UVMap.tangent", mesh_tangent_id},
      {"UVMap.tangent_sign", mesh_tangent_sign_id},
      {"StrandUV", curve_uv_id},
      {"light-only-missing", missing_light_attribute_id}};
  fixture.material_shader_indices = {{mesh_material, 0u},
                                     {curve_material, 1u},
                                     {light_material, 2u},
                                     {world_material, 3u}};

  TriangleMeshDesc mesh;
  mesh.name = "post-displacement mesh";
  mesh.positions = {{-100.0f, -100.0f, -100.0f},
                    {-101.0f, -100.0f, -100.0f},
                    {-100.0f, -101.0f, -100.0f}};
  mesh.triangles = {{0u, 1u, 2u}};
  mesh.material_slots = {mesh_material};
  mesh.default_color_attribute = "Color";
  mesh.uv_layers.emplace(
      "UVMap", MeshAttribute<Vec2f>{
                   .domain = MeshAttributeDomain::corner,
                   .values = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}}});
  mesh.cycles_byte_color_attributes.emplace(
      "Color",
      MeshAttribute<std::array<std::uint8_t, 4u>>{
          .domain = MeshAttributeDomain::corner,
          .values = {
              {{1u, 2u, 3u, 4u}}, {{5u, 6u, 7u, 8u}}, {{9u, 10u, 11u, 12u}}}});
  fixture.snapshot.geometries.emplace(mesh_id, mesh);

  CurveGeometryDesc curve;
  curve.name = "hair";
  curve.shape = CurveShape::thick;
  curve.keys = {{2.0f, 3.0f, 4.0f, 0.1f}, {5.0f, 6.0f, 7.0f, 0.2f}};
  curve.curve_first_key = {0u};
  curve.material_slots = {curve_material};
  curve.curve_material_slots = {0u};
  curve.default_uv_layer = "StrandUV";
  curve.uv_layers.emplace("StrandUV", std::vector<Vec2f>{{0.25f, 0.75f}});
  curve.intercept = {0.0f, 1.0f};
  curve.length = {3.5f};
  curve.random = {0.625f};
  fixture.snapshot.curve_geometries.emplace(curve_id, curve);

  fixture.snapshot.instances.emplace(curve_instance_id,
                                     InstanceDesc{.name = "hair object",
                                                  .geometry = curve_id,
                                                  .cycles_object_index = 1u});
  fixture.snapshot.instances.emplace(mesh_instance_id,
                                     InstanceDesc{.name = "mesh object",
                                                  .geometry = mesh_id,
                                                  .cycles_object_index = 4u});
  fixture.snapshot.lights.emplace(light_id,
                                  LightDesc{.name = "analytic light",
                                            .shader = light_material,
                                            .cycles_object_index = 2u});
  fixture.snapshot.world_shader = world_material;
  fixture.snapshot.cycles_object_count = 8u;
  fixture.snapshot.cycles_background_object_index = 7u;

  GeometryUpload upload;
  upload.attribute_domains = geometry_uv_corner | geometry_uv_tangent_corner;
  upload.default_uv_available = true;
  upload.positions = {
      {10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}};
  upload.normals = {{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  upload.uv = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  upload.uv_tangents = {{1.0f, 0.0f, 0.0f, -1.0f},
                        {1.0f, 0.0f, 0.0f, 1.0f},
                        {1.0f, 0.0f, 0.0f, -1.0f}};
  upload.generated = {
      {0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
  upload.triangles = {{0u, 2u, 1u}};
  upload.triangle_random_per_island = {0.375f};
  upload.attributes.emplace_back(
      AttributeUpload{.id = uv_tangent_attribute_id("UVMap"),
                      .domain = pack_attribute_layout(attribute_domain_corner),
                      .values = {{0.0f, 1.0f, 0.0f, -1.0f},
                                 {0.0f, 1.0f, 0.0f, 1.0f},
                                 {0.0f, 1.0f, 0.0f, -1.0f}}});
  fixture.uploads.emplace_back(std::move(upload));
  fixture.resource_geometry_indices.emplace(mesh_id, 0u);
  fixture.triangle_primitive_offsets.emplace(mesh_id, 5u);
  fixture.curve_primitive_offsets.emplace(curve_id, 9u);
  return fixture;
}

void test_exact_post_displacement_scene_image() {
  const auto fixture = make_fixture();
  const auto image = fixture.build();
  require(image.valid, image.diagnostic);

  const auto curve_geometry = image.attribute_geometry_indices.at(curve_id);
  const auto mesh_geometry = image.attribute_geometry_indices.at(mesh_id);
  require(curve_geometry == 0u &&
              image.light_attribute_geometry_indices.at(light_id) == 1u &&
              mesh_geometry == 2u &&
              image.background_attribute_geometry_index == 3u,
          "Cycles object-derived geometry order changed");

  const auto mesh_map =
      image.attributes.geometries[mesh_geometry].attribute_map_offset;
  const std::array expected_mesh_request_order{
      mesh_color_id,
      static_cast<std::uint64_t>(ATTR_STD_VERTEX_COLOR),
      mesh_uv_id,
      mesh_tangent_id,
      mesh_tangent_sign_id,
      static_cast<std::uint64_t>(ATTR_STD_UV),
      static_cast<std::uint64_t>(ATTR_STD_UV_TANGENT),
      static_cast<std::uint64_t>(ATTR_STD_UV_TANGENT_SIGN),
      static_cast<std::uint64_t>(ATTR_STD_GENERATED),
      static_cast<std::uint64_t>(ATTR_STD_GENERATED_TRANSFORM),
      static_cast<std::uint64_t>(ATTR_STD_RANDOM_PER_ISLAND)};
  for (auto i = std::size_t{}; i < expected_mesh_request_order.size(); ++i) {
    require(image.attributes
                    .attribute_map[mesh_map + i * static_cast<std::size_t>(
                                                      ATTR_PRIM_TYPES)]
                    .id == expected_mesh_request_order[i],
            "shader request order changed while packing geometry");
  }

  require(
      image.attributes.tri_verts.size() == 3u &&
          image.attributes.tri_verts[0u].x == 10.0f &&
          image.attributes.tri_verts[2u].z == 18.0f,
      "geometry image used source positions instead of finalized positions");
  require(image.attributes.geometries[mesh_geometry].normal_offset == 0 &&
              image.attributes.attributes_normal[0u].value ==
                  pack_geometry_normal({0.0f, 0.0f, 1.0f}).value,
          "finalized normal storage changed");
  require(entry(image, mesh_geometry, mesh_color_id).element ==
                  ATTR_ELEMENT_CORNER_BYTE &&
              image.attributes.attributes_uchar4.size() == 6u &&
              image.attributes.attributes_uchar4[0u].x == 1u,
          "named/default byte color did not retain Cycles storage");
  require(entry(image, mesh_geometry, mesh_tangent_id).type ==
                  NODE_ATTR_FLOAT3 &&
              entry(image, mesh_geometry, mesh_tangent_sign_id).type ==
                  NODE_ATTR_FLOAT &&
              image.attributes.attributes_float3.front().y == 1.0f,
          "named tangent vector/sign storage changed");
  require(image.triangle_vertex_indices.size() == 6u &&
              image.triangle_vertex_indices[5u].x == 0u &&
              image.triangle_vertex_indices[5u].y == 2u &&
              image.triangle_vertex_indices[5u].z == 1u,
          "global triangle table did not use finalized local indices");

  require(image.attributes.curve_keys.size() == 2u &&
              image.attributes.curve_keys[0u].x == 2.0f &&
              image.attributes.curve_keys[1u].w == 0.2f,
          "curve position/radius storage changed");
  require(image.curves.size() == 10u && image.curves[9u].first_key == 0 &&
              image.curves[9u].num_keys == 2 &&
              image.curves[9u].type == PRIMITIVE_CURVE_THICK &&
              std::bit_cast<std::uint32_t>(image.curves[9u].shader_id) ==
                  cycles_shader_identity::surface(1u, false),
          "global KernelCurve image changed");
  require(entry(image, curve_geometry, ATTR_STD_CURVE_INTERCEPT).offset == 0 &&
              entry(image, curve_geometry, ATTR_STD_CURVE_LENGTH).offset ==
                  -7 &&
              entry(image, curve_geometry, ATTR_STD_CURVE_RANDOM).offset == -6,
          "curve-domain offset correction changed");
  require(entry(image, 1u, missing_light_attribute_id).element ==
              ATTR_ELEMENT_NONE,
          "missing analytic-light attribute was fabricated");
}

void test_invalid_relations_reject_the_whole_transaction() {
  {
    auto fixture = make_fixture();
    fixture.uploads[0u].attributes.clear();
    const auto image = fixture.build();
    require(!image.valid &&
                image.diagnostic.find("post-displacement source is absent") !=
                    std::string::npos,
            "requested named tangent without a finalized source was accepted");
  }
  {
    auto fixture = make_fixture();
    constexpr GeometryId second_mesh_id{21u};
    auto second_mesh = fixture.snapshot.geometries.at(mesh_id);
    second_mesh.name = "overlapping mesh";
    fixture.snapshot.geometries.emplace(second_mesh_id, std::move(second_mesh));
    fixture.snapshot.instances.emplace(
        InstanceId{42u}, InstanceDesc{.name = "overlapping object",
                                      .geometry = second_mesh_id,
                                      .cycles_object_index = 3u});
    fixture.uploads.emplace_back(fixture.uploads.front());
    fixture.resource_geometry_indices.emplace(second_mesh_id, 1u);
    fixture.triangle_primitive_offsets.emplace(second_mesh_id, 5u);
    const auto image = fixture.build();
    require(!image.valid &&
                image.diagnostic.find("intervals overlap") != std::string::npos,
            "overlapping Cycles primitive intervals were accepted");
  }
  {
    auto fixture = make_fixture();
    fixture.snapshot.curve_geometries.at(curve_id).curve_first_key = {3u};
    const auto image = fixture.build();
    require(!image.valid &&
                image.diagnostic.find("key interval") != std::string::npos,
            "out-of-domain KernelCurve key interval was accepted");
  }
  {
    auto fixture = make_fixture();
    fixture.compilation.table.valid = false;
    require(!fixture.build().valid,
            "invalid shader-table type state entered a geometry image");
  }
}

} // namespace

int main() {
  test_exact_post_displacement_scene_image();
  test_invalid_relations_reject_the_whole_transaction();
  std::cout << "Luisa Cycles SVM geometry scene tests passed\n";
  return 0;
}
