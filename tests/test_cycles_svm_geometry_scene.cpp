#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler::cycles_svm;

constexpr auto first_named_id = static_cast<std::uint64_t>(ATTR_STD_NUM);
constexpr auto uv_name_id = first_named_id;
constexpr auto byte_color_id = first_named_id + 1u;
constexpr auto absent_name_id = first_named_id + 2u;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

[[nodiscard]] const AttributeMap &
entry(const GeometryAttributeTableImage &image, std::uint32_t map_offset,
      std::uint64_t id) {
  auto offset = static_cast<std::size_t>(map_offset);
  while (offset < image.attribute_map.size()) {
    const auto &candidate = image.attribute_map[offset];
    if (candidate.id == id) {
      return candidate;
    }
    require(candidate.id != static_cast<std::uint64_t>(ATTR_STD_NONE),
            "attribute fixture omitted a requested map entry");
    offset += static_cast<std::size_t>(ATTR_PRIM_TYPES);
  }
  require(false, "attribute fixture map walk escaped its image");
  return image.attribute_map.front();
}

[[nodiscard]] GeometryAttributeSource mesh_position() {
  return {.standard = ATTR_STD_POSITION,
          .element = ATTR_ELEMENT_VERTEX,
          .type = NODE_ATTR_FLOAT3,
          .payload = std::vector<packed_float3>{
              {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}};
}

[[nodiscard]] GeometryAttributeInput mesh_input() {
  GeometryAttributeInput mesh;
  mesh.name = "oracle mesh";
  mesh.kind = GeometryAttributeKind::mesh;
  mesh.primitive_offset = 5u;
  mesh.vertex_count = 3u;
  mesh.primitive_count = 2u;
  mesh.corner_count = 6u;
  mesh.requested_attributes = {
      uv_name_id,
      byte_color_id,
      static_cast<std::uint64_t>(ATTR_STD_UV),
      static_cast<std::uint64_t>(ATTR_STD_RANDOM_PER_ISLAND),
      absent_name_id,
      static_cast<std::uint64_t>(ATTR_STD_GENERATED_TRANSFORM)};
  mesh.attributes.emplace_back(mesh_position());
  mesh.attributes.emplace_back(
      GeometryAttributeSource{.standard = ATTR_STD_VERTEX_NORMAL,
                              .element = ATTR_ELEMENT_VERTEX_NORMAL,
                              .type = NODE_ATTR_FLOAT3,
                              .payload = std::vector<packed_normal>{
                                  pack_geometry_normal({1.0f, 0.0f, 0.0f}),
                                  pack_geometry_normal({0.0f, 1.0f, 0.0f}),
                                  pack_geometry_normal({0.0f, 0.0f, 1.0f})}});
  mesh.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_CORNER_NORMAL,
      .element = ATTR_ELEMENT_CORNER_NORMAL,
      .type = NODE_ATTR_FLOAT3,
      .payload = std::vector<packed_normal>(
          6u, pack_geometry_normal({0.0f, 0.0f, 1.0f}))});
  mesh.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_UV,
      .named_id = uv_name_id,
      .element = ATTR_ELEMENT_CORNER,
      .type = NODE_ATTR_FLOAT2,
      .payload = std::vector<packed_float2>{{0.0f, 0.0f},
                                            {1.0f, 0.0f},
                                            {0.0f, 1.0f},
                                            {1.0f, 0.0f},
                                            {1.0f, 1.0f},
                                            {0.0f, 1.0f}}});
  mesh.attributes.emplace_back(GeometryAttributeSource{
      .named_id = byte_color_id,
      .element = ATTR_ELEMENT_CORNER_BYTE,
      .type = NODE_ATTR_RGBA,
      .payload = std::vector<uchar4>{{1u, 2u, 3u, 4u},
                                     {5u, 6u, 7u, 8u},
                                     {9u, 10u, 11u, 12u},
                                     {13u, 14u, 15u, 16u},
                                     {17u, 18u, 19u, 20u},
                                     {21u, 22u, 23u, 24u}}});
  mesh.attributes.emplace_back(
      GeometryAttributeSource{.standard = ATTR_STD_RANDOM_PER_ISLAND,
                              .element = ATTR_ELEMENT_FACE,
                              .type = NODE_ATTR_FLOAT,
                              .payload = std::vector<float>{0.25f, 0.75f}});
  mesh.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_GENERATED_TRANSFORM,
      .element = ATTR_ELEMENT_MESH,
      .type = NODE_ATTR_MATRIX,
      .payload =
          std::vector<PackedTransform>{{.x = {1.0f, 2.0f, 3.0f, 4.0f},
                                        .y = {5.0f, 6.0f, 7.0f, 8.0f},
                                        .z = {9.0f, 10.0f, 11.0f, 12.0f}}}});
  return mesh;
}

[[nodiscard]] GeometryAttributeInput hair_input() {
  GeometryAttributeInput hair;
  hair.name = "oracle hair";
  hair.kind = GeometryAttributeKind::hair;
  hair.primitive_offset = 20u;
  hair.curve_count = 2u;
  hair.key_count = 4u;
  hair.requested_attributes = {
      static_cast<std::uint64_t>(ATTR_STD_RADIUS),
      static_cast<std::uint64_t>(ATTR_STD_UV),
      static_cast<std::uint64_t>(ATTR_STD_CURVE_INTERCEPT),
      static_cast<std::uint64_t>(ATTR_STD_CURVE_LENGTH)};
  hair.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_POSITION,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT3,
      .motion_steps = 2u,
      .payload = std::vector<packed_float3>{{0.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 1.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {1.0f, 0.0f, 1.0f},
                                            {0.1f, 0.0f, 0.0f},
                                            {0.1f, 0.0f, 1.0f},
                                            {1.1f, 0.0f, 0.0f},
                                            {1.1f, 0.0f, 1.0f}}});
  hair.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_RADIUS,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT,
      .payload = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}});
  hair.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_UV,
      .element = ATTR_ELEMENT_CURVE,
      .type = NODE_ATTR_FLOAT2,
      .payload = std::vector<packed_float2>{{0.2f, 0.3f}, {0.7f, 0.8f}}});
  hair.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_CURVE_INTERCEPT,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT,
      .payload = std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f}});
  hair.attributes.emplace_back(
      GeometryAttributeSource{.standard = ATTR_STD_CURVE_LENGTH,
                              .element = ATTR_ELEMENT_CURVE,
                              .type = NODE_ATTR_FLOAT,
                              .payload = std::vector<float>{1.0f, 2.0f}});
  hair.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_SHADOW_TRANSPARENCY,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT,
      .payload = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}});
  return hair;
}

void test_mesh_map_and_typed_arrays_match_cycles_algebra() {
  const auto image = build_geometry_attribute_table({mesh_input()});
  require(image.valid, "valid mesh attribute image was rejected");
  require(image.attribute_map.size() == 20u,
          "mesh map did not use (requests + terminator) * 2 entries");
  for (auto offset = std::size_t{1u}; offset < image.attribute_map.size();
       offset += 2u) {
    require(image.attribute_map[offset].id == 0u &&
                image.attribute_map[offset].element == 0u &&
                image.attribute_map[offset].offset == 0,
            "unused subdivision map lane was not zero initialized");
  }

  const auto map = image.geometries[0u].attribute_map_offset;
  require(entry(image, map, uv_name_id).offset == -15 &&
              entry(image, map, ATTR_STD_UV).offset == -9,
          "named/standard aliases were not independently packed");
  require(image.attributes_float2.size() == 12u,
          "aliased UV payload was not copied once per Cycles request");
  require(entry(image, map, byte_color_id).offset == -15 &&
              entry(image, map, byte_color_id).element ==
                  ATTR_ELEMENT_CORNER_BYTE &&
              entry(image, map, byte_color_id).type == NODE_ATTR_RGBA &&
              image.attributes_uchar4.size() == 6u,
          "corner byte-color storage or primitive correction changed");
  require(entry(image, map, ATTR_STD_RANDOM_PER_ISLAND).offset == -5 &&
              image.attributes_float == std::vector<float>({0.25f, 0.75f}),
          "face attribute primitive correction changed");
  require(entry(image, map, absent_name_id).element == ATTR_ELEMENT_NONE &&
              entry(image, map, absent_name_id).offset == 0 &&
              entry(image, map, absent_name_id).type == NODE_ATTR_FLOAT,
          "missing request did not retain Cycles' default descriptor");
  require(image.attributes_float4.size() == 3u &&
              image.attributes_float4[0u].w == 4.0f &&
              image.attributes_float4[2u].z == 11.0f,
          "matrix attribute was not flattened into three float4 rows");
  require(image.tri_verts.size() == 3u &&
              image.geometries[0u].position_offset == 0,
          "mesh positions did not use the dedicated tri_verts array");
  require(image.attributes_normal.size() == 9u &&
              entry(image, map, ATTR_STD_VERTEX_NORMAL).offset == 0 &&
              entry(image, map, ATTR_STD_CORNER_NORMAL).offset == -12 &&
              image.geometries[0u].normal_offset == -12,
          "corner-normal cache precedence or packed-normal storage changed");
  const auto &terminator = image.attribute_map[18u];
  require(terminator.id == ATTR_STD_NONE && terminator.element == 0u &&
              terminator.offset == 0 && terminator.type == 0u,
          "geometry map terminator changed");
}

void test_hair_position_radius_and_curve_corrections_match_cycles() {
  const auto image =
      build_geometry_attribute_table({mesh_input(), hair_input()});
  require(image.valid, "valid mesh/hair attribute image was rejected");
  require(image.geometries.size() == 2u &&
              image.geometries[1u].attribute_map_offset == 20u,
          "second geometry map offset changed");
  const auto map = image.geometries[1u].attribute_map_offset;
  require(entry(image, map, ATTR_STD_RADIUS).element == ATTR_ELEMENT_NONE &&
              entry(image, map, ATTR_STD_RADIUS).offset == 0,
          "hair radius request was exposed separately from packed position");
  require(image.curve_keys.size() == 8u && image.curve_keys[0u].w == 0.1f &&
              image.curve_keys[4u].x == 0.1f &&
              image.curve_keys[4u].w == 0.1f &&
              image.geometries[1u].position_offset == 0,
          "hair position/radius motion packing changed");
  require(entry(image, map, ATTR_STD_UV).offset == -8,
          "curve-domain primitive correction changed");
  require(entry(image, map, ATTR_STD_CURVE_INTERCEPT).offset == 2 &&
              entry(image, map, ATTR_STD_CURVE_LENGTH).offset == -14,
          "curve-key/curve float table offsets changed");
  require(entry(image, map, ATTR_STD_SHADOW_TRANSPARENCY).element ==
                  ATTR_ELEMENT_CURVE_KEY &&
              image.attributes_float.size() == 12u,
          "mandatory existing hair shadow transparency was omitted");
  require(image.geometries[1u].normal_offset == ATTR_STD_NOT_FOUND,
          "hair geometry manufactured a normal cache offset");
}

void test_native_packed_normal_oracle() {
  // Frozen from Cycles 5.2.1 util/types_normal.h for exact axis inputs. These
  // cases pin the 2x16-bit octahedral storage, not a decoded float tolerance.
  require(pack_geometry_normal({0.0f, 0.0f, 1.0f}).value == 0x80008000u &&
              pack_geometry_normal({1.0f, 0.0f, 0.0f}).value == 0x8000ffffu &&
              pack_geometry_normal({0.0f, 1.0f, 0.0f}).value == 0xffff8000u &&
              pack_geometry_normal({0.0f, 0.0f, -1.0f}).value == 0xffffffffu,
          "Cycles packed-normal octahedral encoding changed");
}

void test_invalid_type_states_are_rejected() {
  auto missing_radius = hair_input();
  missing_radius.attributes.erase(missing_radius.attributes.begin() + 1);
  require(!build_geometry_attribute_table({missing_radius}).valid,
          "hair position without radius entered an uploadable image");

  auto duplicate = mesh_input();
  duplicate.attributes.emplace_back(mesh_position());
  require(!build_geometry_attribute_table({duplicate}).valid,
          "ambiguous standard attribute source was accepted");

  auto short_uv = mesh_input();
  auto &payload =
      std::get<std::vector<packed_float2>>(short_uv.attributes[3u].payload);
  payload.pop_back();
  require(!build_geometry_attribute_table({short_uv}).valid,
          "attribute payload with a non-total domain was accepted");
}

} // namespace

int main() {
  test_mesh_map_and_typed_arrays_match_cycles_algebra();
  test_hair_position_radius_and_curve_corrections_match_cycles();
  test_native_packed_normal_oracle();
  test_invalid_type_states_are_rejected();
  std::cout << "Cycles SVM geometry attribute scene tests passed\n";
  return 0;
}
