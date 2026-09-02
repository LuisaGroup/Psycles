#include "path_tracer_cycles_svm_geometry.h"

#include "path_tracer_internal.h"
#include "path_tracer_scene_geometry.h"

#include "cycles_shader_identity.h"
#include "path_tracer_generated_coordinates.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using namespace compiler::cycles_svm;

enum class OrderedGeometryKind : std::uint8_t {
  mesh,
  curve,
  light,
  background,
};

struct OrderedGeometry {
  std::uint64_t order{};
  OrderedGeometryKind kind{};
  std::uint64_t id{};
};

struct StaticMeshTransform {
  const contract::InstanceDesc *instance{};
  const CyclesInstanceIntersectionPlan *plan{};
};

[[nodiscard]] CyclesSvmGeometrySceneImage reject(std::string diagnostic) {
  CyclesSvmGeometrySceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] packed_float3 pack(luisa::float3 value) noexcept {
  return {value.x, value.y, value.z};
}

[[nodiscard]] packed_float3 pack(Vec3f value) noexcept {
  return {value.x, value.y, value.z};
}

[[nodiscard]] packed_float2 pack(luisa::float2 value) noexcept {
  return {value.x, value.y};
}

[[nodiscard]] packed_float2 pack(Vec2f value) noexcept {
  return {value.x, value.y};
}

[[nodiscard]] packed_float4 pack(Vec4f value) noexcept {
  return {value.x, value.y, value.z, value.w};
}

[[nodiscard]] PackedTransform pack_transform(const Mat4f &matrix) noexcept {
  const auto &e = matrix.elements;
  return {.x = {e[0u], e[4u], e[8u], e[12u]},
          .y = {e[1u], e[5u], e[9u], e[13u]},
          .z = {e[2u], e[6u], e[10u], e[14u]}};
}

[[nodiscard]] packed_float3 transform_static_normal(
    const Mat4f &world_to_object, luisa::float3 normal) noexcept {
  // Cycles' Mesh::apply_transform multiplies normals by the transpose of the
  // object inverse. Mat4f is column-major, so the columns of world_to_object
  // become the rows of the normal transform here.
  const auto &e = world_to_object.elements;
  const auto transformed = Vec3f{
      std::fma(normal.x, e[0u],
               std::fma(normal.y, e[1u], normal.z * e[2u])),
      std::fma(normal.x, e[4u],
               std::fma(normal.y, e[5u], normal.z * e[6u])),
      std::fma(normal.x, e[8u],
               std::fma(normal.y, e[9u], normal.z * e[10u]))};
  const auto length_squared = transformed.x * transformed.x +
                              transformed.y * transformed.y +
                              transformed.z * transformed.z;
  if (!(length_squared > 0.0f) || !std::isfinite(length_squared)) {
    return pack(transformed);
  }
  const auto inverse_length = 1.0f / std::sqrt(length_squared);
  return {transformed.x * inverse_length, transformed.y * inverse_length,
          transformed.z * inverse_length};
}

[[nodiscard]] bool same_position(luisa::float3 actual,
                                 Vec3f expected) noexcept {
  return actual.x == expected.x && actual.y == expected.y &&
         actual.z == expected.z;
}

[[nodiscard]] AttributeElement
mesh_element(contract::MeshAttributeDomain domain) noexcept {
  using enum contract::MeshAttributeDomain;
  switch (domain) {
  case point:
    return ATTR_ELEMENT_VERTEX;
  case corner:
    return ATTR_ELEMENT_CORNER;
  case face:
    return ATTR_ELEMENT_FACE;
  }
  return ATTR_ELEMENT_NONE;
}

[[nodiscard]] AttributeElement upload_element(std::uint32_t domain) noexcept {
  switch (domain & attribute_domain_mask) {
  case attribute_domain_point:
    return ATTR_ELEMENT_VERTEX;
  case attribute_domain_corner:
    return ATTR_ELEMENT_CORNER;
  case attribute_domain_face:
    return ATTR_ELEMENT_FACE;
  default:
    return ATTR_ELEMENT_NONE;
  }
}

template <typename Destination, typename Source, typename Convert>
[[nodiscard]] std::vector<Destination> convert_values(const Source &source,
                                                      Convert &&convert) {
  std::vector<Destination> result;
  result.reserve(source.size());
  for (const auto &value : source) {
    result.emplace_back(convert(value));
  }
  return result;
}

[[nodiscard]] bool ends_with(std::string_view value,
                             std::string_view suffix) noexcept {
  return value.size() >= suffix.size() && value.ends_with(suffix);
}

[[nodiscard]] const AttributeUpload *
find_upload_attribute(const GeometryUpload &upload, std::uint64_t id) noexcept {
  const auto iter = std::ranges::find_if(
      upload.attributes,
      [id](const AttributeUpload &attribute) { return attribute.id == id; });
  return iter == upload.attributes.end() ? nullptr : &*iter;
}

void append_tangent_sources(GeometryAttributeInput &input,
                            AttributeStandard vector_standard,
                            AttributeStandard sign_standard,
                            std::span<const luisa::float4> values) {
  input.attributes.emplace_back(GeometryAttributeSource{
      .standard = vector_standard,
      .named_id = std::nullopt,
      .element = ATTR_ELEMENT_CORNER,
      .type = NODE_ATTR_FLOAT3,
      .payload = convert_values<packed_float3>(values, [](luisa::float4 value) {
        return packed_float3{value.x, value.y, value.z};
      })});
  input.attributes.emplace_back(GeometryAttributeSource{
      .standard = sign_standard,
      .named_id = std::nullopt,
      .element = ATTR_ELEMENT_CORNER,
      .type = NODE_ATTR_FLOAT,
      .payload = convert_values<float>(
          values, [](luisa::float4 value) { return value.w; })});
}

[[nodiscard]] bool requests_attribute(const GeometryAttributeInput &input,
                                      std::uint64_t id) noexcept {
  return std::ranges::find(input.requested_attributes, id) !=
         input.requested_attributes.end();
}

[[nodiscard]] bool append_shader_requests(
    std::vector<std::uint64_t> &requests, std::set<std::uint64_t> &seen,
    contract::MaterialId material,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const CompiledShaderTable &compilation, std::string &diagnostic) {
  const auto shader = material_shader_indices.find(material);
  if (shader == material_shader_indices.end() ||
      shader->second >=
          compilation.shader_attribute_ids_in_request_order.size()) {
    diagnostic = "Cycles geometry references unavailable material " +
                 std::to_string(material.value);
    return false;
  }
  for (const auto id :
       compilation.shader_attribute_ids_in_request_order[shader->second]) {
    if (seen.emplace(id).second) {
      requests.emplace_back(id);
    }
  }
  return true;
}

template <typename Geometry>
[[nodiscard]] bool resolve_geometry_shader_slots(
    const contract::SceneSnapshot &snapshot,
    contract::GeometryId geometry_id, const Geometry &geometry,
    std::vector<contract::MaterialId> &resolved,
    std::string &diagnostic) {
  const auto effective_slots = [&](const contract::InstanceDesc &instance) {
    auto slots = geometry.material_slots;
    if (slots.size() < instance.material_overrides.size()) {
      slots.resize(instance.material_overrides.size());
    }
    for (auto i = std::size_t{}; i < instance.material_overrides.size(); ++i) {
      slots[i] = instance.material_overrides[i];
    }
    return slots;
  };

  bool has_user = false;
  for (const auto &[instance_id, instance] : snapshot.instances) {
    static_cast<void>(instance_id);
    if (instance.geometry != geometry_id) {
      continue;
    }
    auto slots = effective_slots(instance);
    if (!has_user) {
      resolved = std::move(slots);
      has_user = true;
      continue;
    }
    if (slots != resolved) {
      diagnostic = "Cycles geometry '" + geometry.name +
                   "' has object users with distinct used-shader arrays; "
                   "the source geometry must be split before native "
                   "tri_shader packing";
      return false;
    }
  }
  if (!has_user) {
    resolved = geometry.material_slots;
  }
  if (resolved.empty()) {
    diagnostic = "Cycles geometry '" + geometry.name +
                 "' has no default or object-resolved shader slot";
    return false;
  }
  return true;
}

[[nodiscard]] bool resolve_shader_index(
    contract::MaterialId material,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const CompiledShaderTable &compilation, std::uint32_t &shader,
    std::string &diagnostic) {
  const auto iter = material_shader_indices.find(material);
  if (iter == material_shader_indices.end() ||
      iter->second >= compilation.table.shader_count) {
    diagnostic = "Cycles primitive references unavailable material " +
                 std::to_string(material.value);
    return false;
  }
  if ((iter->second & ~cycles_shader_identity::shader_mask) != 0u) {
    diagnostic = "Cycles material " + std::to_string(material.value) +
                 " has a shader index that overlaps shader decoration bits";
    return false;
  }
  shader = iter->second;
  return true;
}

template <typename Geometry>
[[nodiscard]] bool collect_geometry_requests(
    const contract::SceneSnapshot &snapshot, contract::GeometryId geometry_id,
    const Geometry &geometry,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const CompiledShaderTable &compilation,
    const ObjectIdentityPlan &object_identities,
    std::vector<std::uint64_t> &requests, std::string &diagnostic) {
  std::set<std::uint64_t> seen;
  for (const auto material : geometry.material_slots) {
    if (!append_shader_requests(requests, seen, material,
                                material_shader_indices, compilation,
                                diagnostic)) {
      return false;
    }
  }
  std::vector<std::pair<std::uint32_t, const contract::InstanceDesc *>> users;
  for (const auto &[instance_id, instance] : snapshot.instances) {
    if (instance.geometry == geometry_id) {
      const auto object = object_identities.instance_indices.find(instance_id);
      if (object == object_identities.instance_indices.end()) {
        diagnostic = "Cycles geometry cannot resolve one of its object users";
        return false;
      }
      users.emplace_back(object->second, &instance);
    }
  }
  std::ranges::sort(users, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (const auto &[object_index, instance] : users) {
    static_cast<void>(object_index);
    for (const auto material : instance->material_overrides) {
      if (!append_shader_requests(requests, seen, material,
                                  material_shader_indices, compilation,
                                  diagnostic)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool append_mesh_named_sources(
    GeometryAttributeInput &input, const contract::TriangleMeshDesc &geometry,
    const GeometryUpload &upload,
    const std::map<std::string, std::uint64_t, std::less<>> &named_ids,
    std::string &diagnostic) {
  for (const auto &[name, id] : named_ids) {
    if (!requests_attribute(input, id)) {
      continue;
    }
    if (const auto uv = geometry.uv_layers.find(name);
        uv != geometry.uv_layers.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .named_id = id,
          .element = mesh_element(uv->second.domain),
          .type = NODE_ATTR_FLOAT2,
          .payload = convert_values<packed_float2>(
              uv->second.values, [](Vec2f value) { return pack(value); })});
      continue;
    }
    if (const auto bytes = geometry.cycles_byte_color_attributes.find(name);
        bytes != geometry.cycles_byte_color_attributes.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .named_id = id,
          .element = ATTR_ELEMENT_CORNER_BYTE,
          .type = NODE_ATTR_RGBA,
          .payload = convert_values<uchar4>(
              bytes->second.values, [](const std::array<std::uint8_t, 4u> &v) {
                return uchar4{v[0u], v[1u], v[2u], v[3u]};
              })});
      continue;
    }
    if (const auto color = geometry.color_attributes.find(name);
        color != geometry.color_attributes.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .named_id = id,
          .element = mesh_element(color->second.domain),
          .type = NODE_ATTR_RGBA,
          .payload = convert_values<packed_float4>(
              color->second.values, [](Vec4f value) { return pack(value); })});
      continue;
    }

    struct TangentSuffix {
      std::string_view suffix;
      bool undisplaced;
      bool sign;
    };
    static constexpr auto suffixes =
        std::array{TangentSuffix{".undisplaced_tangent_sign", true, true},
                   TangentSuffix{".undisplaced_tangent", true, false},
                   TangentSuffix{".tangent_sign", false, true},
                   TangentSuffix{".tangent", false, false}};
    for (const auto suffix : suffixes) {
      if (!ends_with(name, suffix.suffix)) {
        continue;
      }
      const auto base = name.substr(0u, name.size() - suffix.suffix.size());
      if (!geometry.uv_layers.contains(base)) {
        break;
      }
      const auto legacy_id =
          suffix.undisplaced
              ? contract::uv_undisplaced_tangent_attribute_id(base)
              : contract::uv_tangent_attribute_id(base);
      const auto *attribute = find_upload_attribute(upload, legacy_id);
      if (attribute == nullptr || attribute->values.empty()) {
        diagnostic =
            "Cycles named tangent '" + name +
            "' was requested but its post-displacement source is absent";
        return false;
      }
      if (upload_element(attribute->domain) != ATTR_ELEMENT_CORNER) {
        diagnostic = "Cycles named tangent '" + name +
                     "' is not stored on the corner domain";
        return false;
      }
      if (suffix.sign) {
        input.attributes.emplace_back(GeometryAttributeSource{
            .named_id = id,
            .element = ATTR_ELEMENT_CORNER,
            .type = NODE_ATTR_FLOAT,
            .payload = convert_values<float>(
                attribute->values,
                [](luisa::float4 value) { return value.w; })});
      } else {
        input.attributes.emplace_back(GeometryAttributeSource{
            .named_id = id,
            .element = ATTR_ELEMENT_CORNER,
            .type = NODE_ATTR_FLOAT3,
            .payload = convert_values<packed_float3>(
                attribute->values, [](luisa::float4 value) {
                  return packed_float3{value.x, value.y, value.z};
                })});
      }
      break;
    }
  }
  return true;
}

[[nodiscard]] GeometryAttributeInput make_mesh_input(
    const contract::SceneSnapshot &snapshot, contract::GeometryId geometry_id,
    const contract::TriangleMeshDesc &geometry, const GeometryUpload &upload,
    std::uint32_t primitive_offset,
    const std::map<std::string, std::uint64_t, std::less<>> &named_ids,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const CompiledShaderTable &compilation,
    const ObjectIdentityPlan &object_identities,
    const StaticMeshTransform *static_transform, std::string &diagnostic) {
  GeometryAttributeInput input;
  input.name = geometry.name;
  input.kind = GeometryAttributeKind::mesh;
  input.primitive_offset = primitive_offset;
  input.vertex_count = upload.positions.size();
  input.primitive_count = upload.triangles.size();
  input.corner_count = upload.triangles.size() * 3u;
  if (!collect_geometry_requests(
          snapshot, geometry_id, geometry, material_shader_indices, compilation,
          object_identities, input.requested_attributes, diagnostic)) {
    return {};
  }

  if (static_transform != nullptr) {
    if (upload.cycles_intersection_positions.size() !=
        upload.positions.size()) {
      diagnostic = "Cycles statically transformed mesh '" + geometry.name +
                   "' has no complete world-space intersection image";
      return {};
    }
    if (!upload.undisplaced_positions.empty() ||
        !upload.undisplaced_normals.empty() ||
        !upload.undisplaced_uv_tangents.empty()) {
      diagnostic = "Cycles statically transformed mesh '" + geometry.name +
                   "' also carries displacement-only attributes";
      return {};
    }
    for (auto i = std::size_t{}; i < upload.positions.size(); ++i) {
      const auto expected = cycles_transform_point(
          static_transform->instance->transform,
          Vec3f{upload.positions[i].x, upload.positions[i].y,
                upload.positions[i].z});
      if (!same_position(upload.cycles_intersection_positions[i], expected)) {
        diagnostic = "Cycles statically transformed mesh '" + geometry.name +
                     "' disagrees with its accelerator position image";
        return {};
      }
    }
  } else if (!upload.cycles_intersection_positions.empty()) {
    diagnostic = "Cycles object-space mesh '" + geometry.name +
                 "' unexpectedly carries a transformed intersection image";
    return {};
  }

  input.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_POSITION,
      .named_id = std::nullopt,
      .element = ATTR_ELEMENT_VERTEX,
      .type = NODE_ATTR_FLOAT3,
      .payload = convert_values<packed_float3>(
          upload.positions, [static_transform](luisa::float3 value) {
            if (static_transform == nullptr) {
              return pack(value);
            }
            return pack(cycles_transform_point(
                static_transform->instance->transform,
                Vec3f{value.x, value.y, value.z}));
          })});

  const auto corner_normals =
      (upload.attribute_domains & geometry_normal_corner) != 0u;
  input.attributes.emplace_back(GeometryAttributeSource{
      .standard =
          corner_normals ? ATTR_STD_CORNER_NORMAL : ATTR_STD_VERTEX_NORMAL,
      .named_id = std::nullopt,
      .element = corner_normals ? ATTR_ELEMENT_CORNER_NORMAL
                                : ATTR_ELEMENT_VERTEX_NORMAL,
      .type = NODE_ATTR_FLOAT3,
      .payload = convert_values<packed_normal>(
          upload.normals, [static_transform](luisa::float3 value) {
            const auto normal = static_transform == nullptr
                                    ? pack(value)
                                    : transform_static_normal(
                                          static_transform->plan
                                              ->world_to_object,
                                          value);
            return pack_geometry_normal(normal);
          })});

  if (upload.default_uv_available) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_UV,
        .named_id = std::nullopt,
        .element = (upload.attribute_domains & geometry_uv_corner) != 0u
                       ? ATTR_ELEMENT_CORNER
                       : ATTR_ELEMENT_VERTEX,
        .type = NODE_ATTR_FLOAT2,
        .payload = convert_values<packed_float2>(
            upload.uv, [](luisa::float2 value) { return pack(value); })});
  }
  if (!upload.uv_tangents.empty()) {
    append_tangent_sources(input, ATTR_STD_UV_TANGENT, ATTR_STD_UV_TANGENT_SIGN,
                           upload.uv_tangents);
  }
  if (!upload.undisplaced_uv_tangents.empty()) {
    append_tangent_sources(input, ATTR_STD_UV_TANGENT_UNDISPLACED,
                           ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED,
                           upload.undisplaced_uv_tangents);
  }

  if (!upload.generated.empty()) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_GENERATED,
        .named_id = std::nullopt,
        .element = (upload.attribute_domains & geometry_generated_corner) != 0u
                       ? ATTR_ELEMENT_CORNER
                       : ATTR_ELEMENT_VERTEX,
        .type = NODE_ATTR_FLOAT3,
        .payload = convert_values<packed_float3>(
            upload.generated,
            [](luisa::float3 value) { return pack(value); })});
  }
  if (requests_attribute(input, ATTR_STD_GENERATED_TRANSFORM)) {
    const auto generated_transform = geometry.generated_transform.value_or(
        make_generated_coordinate_mapping(geometry).object_to_generated);
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_GENERATED_TRANSFORM,
        .named_id = std::nullopt,
        .element = ATTR_ELEMENT_MESH,
        .type = NODE_ATTR_MATRIX,
        .payload =
            std::vector<PackedTransform>{pack_transform(generated_transform)}});
  }
  if (!upload.triangle_random_per_island.empty()) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_RANDOM_PER_ISLAND,
        .named_id = std::nullopt,
        .element = ATTR_ELEMENT_FACE,
        .type = NODE_ATTR_FLOAT,
        .payload =
            std::vector<float>{upload.triangle_random_per_island.begin(),
                               upload.triangle_random_per_island.end()}});
  }

  if (!upload.undisplaced_positions.empty()) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_POSITION_UNDISPLACED,
        .named_id = std::nullopt,
        .element = ATTR_ELEMENT_VERTEX,
        .type = NODE_ATTR_FLOAT3,
        .payload = convert_values<packed_float3>(
            upload.undisplaced_positions,
            [](luisa::float3 value) { return pack(value); })});
  }
  if (!upload.undisplaced_normals.empty()) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_NORMAL_UNDISPLACED,
        .named_id = std::nullopt,
        .element = corner_normals ? ATTR_ELEMENT_CORNER_NORMAL
                                  : ATTR_ELEMENT_VERTEX_NORMAL,
        .type = NODE_ATTR_FLOAT3,
        .payload = convert_values<packed_normal>(
            upload.undisplaced_normals, [](luisa::float3 value) {
              return pack_geometry_normal(pack(value));
            })});
  }
  if (geometry.default_color_attribute) {
    const auto &name = *geometry.default_color_attribute;
    if (const auto bytes = geometry.cycles_byte_color_attributes.find(name);
        bytes != geometry.cycles_byte_color_attributes.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .standard = ATTR_STD_VERTEX_COLOR,
          .named_id = std::nullopt,
          .element = ATTR_ELEMENT_CORNER_BYTE,
          .type = NODE_ATTR_RGBA,
          .payload = convert_values<uchar4>(
              bytes->second.values, [](const std::array<std::uint8_t, 4u> &v) {
                return uchar4{v[0u], v[1u], v[2u], v[3u]};
              })});
    } else if (const auto color = geometry.color_attributes.find(name);
               color != geometry.color_attributes.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .standard = ATTR_STD_VERTEX_COLOR,
          .named_id = std::nullopt,
          .element = mesh_element(color->second.domain),
          .type = NODE_ATTR_RGBA,
          .payload = convert_values<packed_float4>(
              color->second.values, [](Vec4f value) { return pack(value); })});
    }
  }

  const auto pointiness =
      find_upload_attribute(upload, contract::cycles_pointiness_attribute_id);
  if (pointiness != nullptr) {
    input.attributes.emplace_back(GeometryAttributeSource{
        .standard = ATTR_STD_POINTINESS,
        .named_id = std::nullopt,
        .element = upload_element(pointiness->domain),
        .type = NODE_ATTR_FLOAT,
        .payload = convert_values<float>(
            pointiness->values, [](luisa::float4 value) { return value.x; })});
  }
  if (!append_mesh_named_sources(input, geometry, upload, named_ids,
                                 diagnostic)) {
    return {};
  }
  return input;
}

[[nodiscard]] GeometryAttributeInput make_curve_input(
    const contract::SceneSnapshot &snapshot, contract::GeometryId geometry_id,
    const contract::CurveGeometryDesc &geometry, std::uint32_t curve_offset,
    const std::map<std::string, std::uint64_t, std::less<>> &named_ids,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const CompiledShaderTable &compilation,
    const ObjectIdentityPlan &object_identities, std::string &diagnostic) {
  GeometryAttributeInput input;
  input.name = geometry.name;
  input.kind = GeometryAttributeKind::hair;
  input.primitive_offset = curve_offset;
  input.curve_count = geometry.curve_first_key.size();
  input.key_count = geometry.keys.size();
  if (!collect_geometry_requests(
          snapshot, geometry_id, geometry, material_shader_indices, compilation,
          object_identities, input.requested_attributes, diagnostic)) {
    return {};
  }
  input.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_POSITION,
      .named_id = std::nullopt,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT3,
      .payload = convert_values<packed_float3>(geometry.keys, [](Vec4f value) {
        return packed_float3{value.x, value.y, value.z};
      })});
  input.attributes.emplace_back(GeometryAttributeSource{
      .standard = ATTR_STD_RADIUS,
      .named_id = std::nullopt,
      .element = ATTR_ELEMENT_CURVE_KEY,
      .type = NODE_ATTR_FLOAT,
      .payload = convert_values<float>(geometry.keys,
                                       [](Vec4f value) { return value.w; })});
  if (geometry.default_uv_layer) {
    const auto uv = geometry.uv_layers.find(*geometry.default_uv_layer);
    if (uv != geometry.uv_layers.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .standard = ATTR_STD_UV,
          .named_id = std::nullopt,
          .element = ATTR_ELEMENT_CURVE,
          .type = NODE_ATTR_FLOAT2,
          .payload = convert_values<packed_float2>(
              uv->second, [](Vec2f value) { return pack(value); })});
    }
  }
  if (!geometry.intercept.empty()) {
    input.attributes.emplace_back(
        GeometryAttributeSource{.standard = ATTR_STD_CURVE_INTERCEPT,
                                .named_id = std::nullopt,
                                .element = ATTR_ELEMENT_CURVE_KEY,
                                .type = NODE_ATTR_FLOAT,
                                .payload = geometry.intercept});
  }
  if (!geometry.length.empty()) {
    input.attributes.emplace_back(
        GeometryAttributeSource{.standard = ATTR_STD_CURVE_LENGTH,
                                .named_id = std::nullopt,
                                .element = ATTR_ELEMENT_CURVE,
                                .type = NODE_ATTR_FLOAT,
                                .payload = geometry.length});
  }
  if (!geometry.random.empty()) {
    input.attributes.emplace_back(
        GeometryAttributeSource{.standard = ATTR_STD_CURVE_RANDOM,
                                .named_id = std::nullopt,
                                .element = ATTR_ELEMENT_CURVE,
                                .type = NODE_ATTR_FLOAT,
                                .payload = geometry.random});
  }
  for (const auto &[name, id] : named_ids) {
    if (!requests_attribute(input, id)) {
      continue;
    }
    if (const auto uv = geometry.uv_layers.find(name);
        uv != geometry.uv_layers.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .named_id = id,
          .element = ATTR_ELEMENT_CURVE,
          .type = NODE_ATTR_FLOAT2,
          .payload = convert_values<packed_float2>(
              uv->second, [](Vec2f value) { return pack(value); })});
    } else if (const auto color = geometry.color_attributes.find(name);
               color != geometry.color_attributes.end()) {
      input.attributes.emplace_back(GeometryAttributeSource{
          .named_id = id,
          .element = ATTR_ELEMENT_CURVE,
          .type = NODE_ATTR_RGBA,
          .payload = convert_values<packed_float4>(
              color->second, [](Vec4f value) { return pack(value); })});
    }
  }
  return input;
}

[[nodiscard]] bool checked_resize(std::size_t offset, std::size_t count,
                                  std::size_t &extent) noexcept {
  if (count > std::numeric_limits<std::size_t>::max() - offset) {
    return false;
  }
  extent = std::max(extent, offset + count);
  return true;
}

} // namespace

CyclesSvmGeometrySceneImage build_cycles_svm_geometry_scene_image(
    const contract::SceneSnapshot &snapshot,
    const CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const ObjectIdentityPlan &object_identities,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets) {
  if (!compilation.table.valid) {
    return reject("Cycles geometry image received an invalid shader table");
  }
  if (!object_identities.valid) {
    return reject("Cycles geometry image received an invalid object domain");
  }
  if (intersection_plans.size() != snapshot.instances.size()) {
    return reject("Cycles geometry image received an intersection plan with "
                  "the wrong instance extent");
  }
  std::map<std::string, std::uint64_t, std::less<>> named_ids;
  for (const auto &[name, id] : compilation.named_attributes) {
    if (!named_ids.emplace(name, id).second) {
      return reject("Cycles named attribute table contains duplicate name '" +
                    name + "'");
    }
  }

  std::map<contract::GeometryId, std::uint32_t> first_object;
  std::map<contract::GeometryId, std::size_t> geometry_users;
  std::map<contract::GeometryId, StaticMeshTransform> static_mesh_transforms;
  auto plan_index = std::size_t{};
  for (const auto &[instance_id, instance] : snapshot.instances) {
    const auto &intersection_plan = intersection_plans[plan_index++];
    if (intersection_plan.instance != instance_id) {
      return reject("Cycles geometry image received an intersection plan in "
                    "the wrong instance order");
    }
    if (intersection_plan.world_to_object !=
        cycles_inverse_transform(instance.transform)) {
      return reject("Cycles geometry image received an intersection plan "
                    "whose inverse transform is inconsistent");
    }
    const auto geometry_kind_count =
        static_cast<std::uint32_t>(
            snapshot.geometries.contains(instance.geometry)) +
        static_cast<std::uint32_t>(
            snapshot.curve_geometries.contains(instance.geometry));
    if (geometry_kind_count != 1u) {
      return reject("Cycles geometry image requires each instance to resolve "
                    "exactly one geometry kind");
    }
    ++geometry_users[instance.geometry];
    if (intersection_plan.transform_applied) {
      if (!snapshot.geometries.contains(instance.geometry)) {
        return reject("Cycles intersection plan applies a static transform to "
                      "a non-mesh geometry");
      }
      if (!static_mesh_transforms
               .emplace(instance.geometry,
                        StaticMeshTransform{.instance = &instance,
                                            .plan = &intersection_plan})
               .second) {
        return reject("Cycles intersection plan applies one mesh transform "
                      "more than once");
      }
    }
    const auto object = object_identities.instance_indices.find(instance_id);
    if (object == object_identities.instance_indices.end()) {
      return reject("Cycles geometry image cannot resolve an instance object");
    }
    auto [iter, inserted] =
        first_object.emplace(instance.geometry, object->second);
    if (!inserted) {
      iter->second = std::min(iter->second, object->second);
    }
  }
  for (const auto &[geometry, transform] : static_mesh_transforms) {
    static_cast<void>(transform);
    if (geometry_users.at(geometry) != 1u) {
      return reject("Cycles intersection plan applies a transform to shared "
                    "mesh geometry");
    }
  }

  std::vector<OrderedGeometry> order;
  order.reserve(snapshot.geometries.size() + snapshot.curve_geometries.size() +
                snapshot.lights.size() +
                (object_identities.background_index ? 1u : 0u));
  auto unowned_order =
      static_cast<std::uint64_t>(object_identities.object_count);
  const auto append_geometry = [&](contract::GeometryId id,
                                   OrderedGeometryKind kind) {
    const auto owner = first_object.find(id);
    order.emplace_back(OrderedGeometry{
        .order = owner == first_object.end() ? unowned_order++ : owner->second,
        .kind = kind,
        .id = id.value});
  };
  for (const auto &[id, geometry] : snapshot.geometries) {
    static_cast<void>(geometry);
    append_geometry(id, OrderedGeometryKind::mesh);
  }
  for (const auto &[id, geometry] : snapshot.curve_geometries) {
    static_cast<void>(geometry);
    append_geometry(id, OrderedGeometryKind::curve);
  }
  for (const auto &[id, light] : snapshot.lights) {
    static_cast<void>(light);
    const auto object = object_identities.light_indices.find(id);
    if (object == object_identities.light_indices.end()) {
      return reject("Cycles geometry image cannot resolve a light object");
    }
    order.emplace_back(OrderedGeometry{.order = object->second,
                                       .kind = OrderedGeometryKind::light,
                                       .id = id.value});
  }
  if (object_identities.background_index) {
    order.emplace_back(
        OrderedGeometry{.order = *object_identities.background_index,
                        .kind = OrderedGeometryKind::background,
                        .id = 0u});
  }
  std::ranges::sort(order,
                    [](const OrderedGeometry &lhs, const OrderedGeometry &rhs) {
                      return std::tuple{lhs.order, lhs.kind, lhs.id} <
                             std::tuple{rhs.order, rhs.kind, rhs.id};
                    });

  std::vector<GeometryAttributeInput> inputs;
  inputs.reserve(order.size());
  CyclesSvmGeometrySceneImage result;
  for (const auto item : order) {
    if (inputs.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      return reject("Cycles geometry image exceeds the uint32 geometry domain");
    }
    const auto image_index = static_cast<std::uint32_t>(inputs.size());
    std::string diagnostic;
    switch (item.kind) {
    case OrderedGeometryKind::mesh: {
      const contract::GeometryId id{item.id};
      const auto resource = resource_geometry_indices.find(id);
      const auto primitive = triangle_primitive_offsets.find(id);
      if (resource == resource_geometry_indices.end() ||
          resource->second >= mesh_uploads.size() ||
          primitive == triangle_primitive_offsets.end()) {
        return reject("Cycles mesh geometry " + std::to_string(id.value) +
                      " has no finalized post-displacement upload");
      }
      const auto static_transform_iter = static_mesh_transforms.find(id);
      const auto *const static_transform =
          static_transform_iter == static_mesh_transforms.end()
              ? nullptr
              : &static_transform_iter->second;
      auto input = make_mesh_input(
          snapshot, id, snapshot.geometries.at(id),
          mesh_uploads[resource->second], primitive->second, named_ids,
          material_shader_indices, compilation, object_identities,
          static_transform, diagnostic);
      if (!diagnostic.empty()) {
        return reject(std::move(diagnostic));
      }
      result.attribute_geometry_indices.emplace(id, image_index);
      inputs.emplace_back(std::move(input));
      break;
    }
    case OrderedGeometryKind::curve: {
      const contract::GeometryId id{item.id};
      const auto primitive = curve_primitive_offsets.find(id);
      if (primitive == curve_primitive_offsets.end()) {
        return reject("Cycles curve geometry " + std::to_string(id.value) +
                      " has no finalized primitive offset");
      }
      auto input = make_curve_input(
          snapshot, id, snapshot.curve_geometries.at(id), primitive->second,
          named_ids, material_shader_indices, compilation, object_identities,
          diagnostic);
      if (!diagnostic.empty()) {
        return reject(std::move(diagnostic));
      }
      result.attribute_geometry_indices.emplace(id, image_index);
      inputs.emplace_back(std::move(input));
      break;
    }
    case OrderedGeometryKind::light: {
      const contract::LightId id{item.id};
      const auto &light = snapshot.lights.at(id);
      GeometryAttributeInput input;
      input.name = light.name;
      input.kind = GeometryAttributeKind::light;
      if (light.shader) {
        std::set<std::uint64_t> seen;
        if (!append_shader_requests(input.requested_attributes, seen,
                                    *light.shader, material_shader_indices,
                                    compilation, diagnostic)) {
          return reject(std::move(diagnostic));
        }
      }
      result.light_attribute_geometry_indices.emplace(id, image_index);
      inputs.emplace_back(std::move(input));
      break;
    }
    case OrderedGeometryKind::background: {
      GeometryAttributeInput input;
      input.name = "Cycles background light";
      input.kind = GeometryAttributeKind::light;
      if (snapshot.world_shader) {
        std::set<std::uint64_t> seen;
        if (!append_shader_requests(
                input.requested_attributes, seen, *snapshot.world_shader,
                material_shader_indices, compilation, diagnostic)) {
          return reject(std::move(diagnostic));
        }
      }
      result.background_attribute_geometry_index = image_index;
      inputs.emplace_back(std::move(input));
      break;
    }
    }
  }

  result.attributes = build_geometry_attribute_table(inputs);
  if (!result.attributes.valid) {
    return reject(result.attributes.diagnostic);
  }

  std::size_t triangle_extent{};
  std::vector<bool> occupied_triangles;
  for (const auto &[id, geometry] : snapshot.geometries) {
    const auto offset_iter = triangle_primitive_offsets.find(id);
    if (offset_iter == triangle_primitive_offsets.end()) {
      return reject("Cycles mesh geometry " + std::to_string(id.value) +
                    " has no primitive offset");
    }
    const auto offset = offset_iter->second;
    if (!checked_resize(offset, geometry.triangles.size(), triangle_extent)) {
      return reject("Cycles triangle vertex-index extent overflows size_t");
    }
  }
  result.triangle_vertex_indices.resize(triangle_extent);
  result.triangle_shaders.resize(triangle_extent);
  occupied_triangles.resize(triangle_extent, false);
  for (const auto &[id, geometry] : snapshot.geometries) {
    const auto offset = triangle_primitive_offsets.find(id)->second;
    const auto resource = resource_geometry_indices.find(id);
    if (resource == resource_geometry_indices.end() ||
        resource->second >= mesh_uploads.size()) {
      return reject("Cycles mesh geometry " + std::to_string(id.value) +
                    " has no finalized upload while packing triangles");
    }
    const auto &upload = mesh_uploads[resource->second];
    if (upload.triangles.size() != geometry.triangles.size()) {
      return reject(
          "Cycles mesh finalized topology differs from its primitive interval");
    }
    std::vector<contract::MaterialId> shader_slots;
    std::string diagnostic;
    if (!resolve_geometry_shader_slots(snapshot, id, geometry, shader_slots,
                                       diagnostic)) {
      return reject(std::move(diagnostic));
    }
    if (!upload.triangle_material_slots.empty() &&
        upload.triangle_material_slots.size() != upload.triangles.size()) {
      return reject(
          "Cycles finalized triangle material image has the wrong extent");
    }
    if (!upload.triangle_smooth.empty() &&
        upload.triangle_smooth.size() != upload.triangles.size()) {
      return reject("Cycles finalized triangle smooth image has the wrong "
                    "extent");
    }
    if (geometry.triangle_material_slots.size() == upload.triangles.size() &&
        !upload.triangle_material_slots.empty() &&
        !std::ranges::equal(geometry.triangle_material_slots,
                            upload.triangle_material_slots)) {
      return reject("Cycles finalized triangle material image differs from "
                    "its source geometry");
    }
    if (geometry.triangle_smooth.size() == upload.triangles.size() &&
        !upload.triangle_smooth.empty() &&
        !std::ranges::equal(geometry.triangle_smooth,
                            upload.triangle_smooth)) {
      return reject("Cycles finalized triangle smooth image differs from its "
                    "source geometry");
    }
    const auto has_corner_normals =
        (upload.attribute_domains & geometry_normal_corner) != 0u;
    for (auto i = std::size_t{}; i < upload.triangles.size(); ++i) {
      const auto destination = static_cast<std::size_t>(offset) + i;
      if (occupied_triangles[destination]) {
        return reject("Cycles triangle primitive intervals overlap");
      }
      occupied_triangles[destination] = true;
      const auto &triangle = upload.triangles[i];
      if (triangle.i0 >= upload.positions.size() ||
          triangle.i1 >= upload.positions.size() ||
          triangle.i2 >= upload.positions.size()) {
        return reject(
            "Cycles mesh triangle references a vertex outside its geometry");
      }
      result.triangle_vertex_indices[destination] = {triangle.i0, triangle.i1,
                                                     triangle.i2};
      const auto source_slot = !upload.triangle_material_slots.empty()
                                   ? upload.triangle_material_slots[i]
                               : i < geometry.triangle_material_slots.size()
                                   ? geometry.triangle_material_slots[i]
                                   : 0u;
      const auto slot = std::min<std::size_t>(source_slot,
                                              shader_slots.size() - 1u);
      std::uint32_t shader{};
      if (!resolve_shader_index(shader_slots[slot], material_shader_indices,
                                compilation, shader, diagnostic)) {
        return reject(std::move(diagnostic));
      }
      // Mesh::pack_shaders ignores authored flatness when corner normals are
      // present: the corner-normal attribute already carries that topology,
      // and every triangle is decorated smooth so the kernel interpolates it.
      const auto smooth =
          has_corner_normals ||
          (!upload.triangle_smooth.empty()
               ? upload.triangle_smooth[i] != 0u
               : i < geometry.triangle_smooth.size() &&
                     geometry.triangle_smooth[i] != 0u);
      result.triangle_shaders[destination] =
          cycles_shader_identity::surface(shader, smooth);
    }
  }

  std::size_t curve_extent{};
  std::vector<bool> occupied_curves;
  for (const auto &[id, geometry] : snapshot.curve_geometries) {
    const auto offset_iter = curve_primitive_offsets.find(id);
    if (offset_iter == curve_primitive_offsets.end()) {
      return reject("Cycles curve geometry " + std::to_string(id.value) +
                    " has no primitive offset");
    }
    const auto offset = offset_iter->second;
    if (!checked_resize(offset, geometry.curve_first_key.size(),
                        curve_extent)) {
      return reject("Cycles curve extent overflows size_t");
    }
  }
  result.curves.resize(curve_extent);
  occupied_curves.resize(curve_extent, false);
  for (const auto &[id, geometry] : snapshot.curve_geometries) {
    const auto offset = curve_primitive_offsets.find(id)->second;
    if (geometry.keys.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      return reject("Cycles curve key table is not representable as uint32");
    }
    std::vector<contract::MaterialId> shader_slots;
    std::string diagnostic;
    if (!resolve_geometry_shader_slots(snapshot, id, geometry, shader_slots,
                                       diagnostic)) {
      return reject(std::move(diagnostic));
    }
    for (auto curve = std::size_t{}; curve < geometry.curve_first_key.size();
         ++curve) {
      const auto destination = static_cast<std::size_t>(offset) + curve;
      if (occupied_curves[destination]) {
        return reject("Cycles curve primitive intervals overlap");
      }
      occupied_curves[destination] = true;
      const auto first = geometry.curve_first_key[curve];
      const auto end = curve + 1u < geometry.curve_first_key.size()
                           ? geometry.curve_first_key[curve + 1u]
                           : static_cast<std::uint32_t>(geometry.keys.size());
      if (first > geometry.keys.size() || end > geometry.keys.size() ||
          first > static_cast<std::uint32_t>(
                      std::numeric_limits<std::int32_t>::max()) ||
          end < first ||
          end - first > static_cast<std::uint32_t>(
                            std::numeric_limits<std::int32_t>::max())) {
        return reject("Cycles curve key interval is not representable");
      }
      const auto slot = curve < geometry.curve_material_slots.size()
                            ? geometry.curve_material_slots[curve]
                            : 0u;
      const auto material = shader_slots[std::min<std::size_t>(
          slot, shader_slots.size() - 1u)];
      std::uint32_t shader{};
      if (!resolve_shader_index(material, material_shader_indices, compilation,
                                shader, diagnostic)) {
        return reject(std::move(diagnostic));
      }
      result.curves[destination] = KernelCurve{
          .shader_id = std::bit_cast<std::int32_t>(
              cycles_shader_identity::surface(shader, false)),
          .first_key = static_cast<std::int32_t>(first),
          .num_keys = static_cast<std::int32_t>(end - first),
          .type =
              cycles_svm_curve_primitive_type(geometry.shape)};
    }
  }
  result.valid = true;
  return result;
}

} // namespace psycles::luisa_backend::detail
