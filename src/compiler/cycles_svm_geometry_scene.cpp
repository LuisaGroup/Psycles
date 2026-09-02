#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] GeometryAttributeTableImage reject(std::string diagnostic) {
  GeometryAttributeTableImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] std::string label(const GeometryAttributeInput &geometry) {
  return geometry.name.empty() ? "Cycles geometry"
                               : "Cycles geometry '" + geometry.name + "'";
}

[[nodiscard]] std::uint32_t base_element(AttributeElement element) noexcept {
  constexpr auto qualifiers = static_cast<std::uint32_t>(
      ATTR_ELEMENT_IS_MOTION | ATTR_ELEMENT_IS_NORMAL | ATTR_ELEMENT_IS_BYTE);
  return static_cast<std::uint32_t>(element) & ~qualifiers;
}

[[nodiscard]] std::optional<std::size_t>
element_count(const GeometryAttributeInput &geometry,
              AttributeElement element) noexcept {
  switch (base_element(element)) {
  case ATTR_ELEMENT_OBJECT:
  case ATTR_ELEMENT_MESH:
  case ATTR_ELEMENT_VOXEL:
    return 1u;
  case ATTR_ELEMENT_VERTEX:
    if (geometry.kind == GeometryAttributeKind::mesh ||
        geometry.kind == GeometryAttributeKind::volume ||
        geometry.kind == GeometryAttributeKind::pointcloud) {
      return geometry.vertex_count;
    }
    return std::nullopt;
  case ATTR_ELEMENT_FACE:
    if (geometry.kind == GeometryAttributeKind::mesh ||
        geometry.kind == GeometryAttributeKind::volume) {
      return geometry.primitive_count;
    }
    return std::nullopt;
  case ATTR_ELEMENT_CORNER:
    if (geometry.kind == GeometryAttributeKind::mesh) {
      return geometry.corner_count;
    }
    return std::nullopt;
  case ATTR_ELEMENT_CURVE:
    if (geometry.kind == GeometryAttributeKind::hair) {
      return geometry.curve_count;
    }
    return std::nullopt;
  case ATTR_ELEMENT_CURVE_KEY:
    if (geometry.kind == GeometryAttributeKind::hair) {
      return geometry.key_count;
    }
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] bool checked_product(std::size_t lhs, std::uint32_t rhs,
                                   std::size_t &result) noexcept {
  if (rhs != 0u && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    return false;
  }
  result = lhs * static_cast<std::size_t>(rhs);
  return true;
}

template <typename T>
[[nodiscard]] bool payload_has_size(const GeometryAttributeSource &source,
                                    std::size_t expected) noexcept {
  const auto *values = std::get_if<std::vector<T>>(&source.payload);
  return values != nullptr && values->size() == expected;
}

[[nodiscard]] bool validate_payload(const GeometryAttributeInput &geometry,
                                    const GeometryAttributeSource &source,
                                    std::string &diagnostic) {
  if (source.motion_steps == 0u) {
    diagnostic = label(geometry) + " has an attribute with zero motion steps";
    return false;
  }
  const auto count = element_count(geometry, source.element);
  if (!count) {
    diagnostic = label(geometry) +
                 " has an attribute element outside its primitive domain";
    return false;
  }
  std::size_t expected{};
  if (!checked_product(*count, source.motion_steps, expected)) {
    diagnostic = label(geometry) + " attribute element count overflows size_t";
    return false;
  }

  const auto byte = (static_cast<std::uint32_t>(source.element) &
                     static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) != 0u;
  const auto normal =
      (static_cast<std::uint32_t>(source.element) &
       static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) != 0u;
  const auto voxel = base_element(source.element) == ATTR_ELEMENT_VOXEL;
  if (byte && normal) {
    diagnostic = label(geometry) +
                 " has an attribute marked as both byte and normal storage";
    return false;
  }
  bool valid = false;
  if (voxel) {
    valid = std::holds_alternative<VoxelAttributeHandle>(source.payload);
  } else if (byte) {
    valid = source.type == NODE_ATTR_RGBA &&
            payload_has_size<uchar4>(source, expected);
  } else if (normal) {
    valid = source.type == NODE_ATTR_FLOAT3 &&
            payload_has_size<packed_normal>(source, expected);
  } else {
    switch (source.type) {
    case NODE_ATTR_FLOAT:
      valid = payload_has_size<float>(source, expected);
      break;
    case NODE_ATTR_FLOAT2:
      valid = payload_has_size<packed_float2>(source, expected);
      break;
    case NODE_ATTR_FLOAT3:
      valid = payload_has_size<packed_float3>(source, expected);
      break;
    case NODE_ATTR_FLOAT4:
    case NODE_ATTR_RGBA:
      valid = payload_has_size<packed_float4>(source, expected);
      break;
    case NODE_ATTR_MATRIX:
      valid = payload_has_size<PackedTransform>(source, expected);
      break;
    }
  }
  if (!valid) {
    diagnostic = label(geometry) +
                 " has an attribute payload inconsistent with its Cycles "
                 "type, element, or motion extent";
  }
  return valid;
}

[[nodiscard]] bool is_standard_request(std::uint64_t id) noexcept {
  return id > static_cast<std::uint64_t>(ATTR_STD_NONE) &&
         id < static_cast<std::uint64_t>(ATTR_STD_NUM);
}

[[nodiscard]] const GeometryAttributeSource *
find_source(const GeometryAttributeInput &geometry, std::uint64_t id) noexcept {
  if (is_standard_request(id)) {
    const auto standard = static_cast<AttributeStandard>(id);
    for (const auto &source : geometry.attributes) {
      if (source.standard == standard) {
        return &source;
      }
    }
    return nullptr;
  }
  for (const auto &source : geometry.attributes) {
    if (source.named_id && *source.named_id == id) {
      return &source;
    }
  }
  return nullptr;
}

[[nodiscard]] bool
validate_source_identities(const GeometryAttributeInput &geometry,
                           std::string &diagnostic) {
  std::set<AttributeStandard> standards;
  std::set<std::uint64_t> named;
  for (const auto &source : geometry.attributes) {
    if (source.standard != ATTR_STD_NONE) {
      const auto standard = static_cast<std::int64_t>(source.standard);
      if (standard <= ATTR_STD_NONE || standard >= ATTR_STD_NUM) {
        diagnostic = label(geometry) +
                     " has an out-of-domain standard attribute identity";
        return false;
      }
      if (!standards.emplace(source.standard).second) {
        diagnostic =
            label(geometry) + " has duplicate standard attribute sources";
        return false;
      }
    }
    if (source.named_id) {
      if (*source.named_id < static_cast<std::uint64_t>(ATTR_STD_NUM)) {
        diagnostic = label(geometry) +
                     " has a named attribute in the standard-ID domain";
        return false;
      }
      if (!named.emplace(*source.named_id).second) {
        diagnostic = label(geometry) + " has duplicate named attribute sources";
        return false;
      }
    }
    if (source.standard == ATTR_STD_NONE && !source.named_id) {
      diagnostic = label(geometry) + " has an unaddressable attribute source";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool append_offset(std::size_t size, std::int64_t correction,
                                 std::int32_t &offset) noexcept {
  const auto corrected = static_cast<std::uint64_t>(size) <=
                                 static_cast<std::uint64_t>(
                                     std::numeric_limits<std::int64_t>::max())
                             ? static_cast<std::int64_t>(size) - correction
                             : std::numeric_limits<std::int64_t>::max();
  if (corrected < std::numeric_limits<std::int32_t>::min() ||
      corrected > std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  offset = static_cast<std::int32_t>(corrected);
  return true;
}

[[nodiscard]] std::int64_t
offset_correction(const GeometryAttributeInput &geometry,
                  AttributeElement element) noexcept {
  switch (base_element(element)) {
  case ATTR_ELEMENT_FACE:
    return static_cast<std::int64_t>(geometry.primitive_offset);
  case ATTR_ELEMENT_CORNER:
    return 3ll * static_cast<std::int64_t>(geometry.primitive_offset);
  case ATTR_ELEMENT_CURVE:
    return static_cast<std::int64_t>(geometry.primitive_offset);
  case ATTR_ELEMENT_VERTEX:
    return geometry.kind == GeometryAttributeKind::pointcloud
               ? static_cast<std::int64_t>(geometry.primitive_offset)
               : 0ll;
  default:
    return 0ll;
  }
}

template <typename T>
void append_vector(std::vector<T> &destination,
                   const GeometryAttributeSource &source) {
  const auto &values = std::get<std::vector<T>>(source.payload);
  destination.insert(destination.end(), values.begin(), values.end());
}

[[nodiscard]] bool append_position_radius(
    GeometryAttributeTableImage &image, const GeometryAttributeInput &geometry,
    const GeometryAttributeSource &position, AttributeDescriptor &descriptor,
    std::string &diagnostic) {
  const auto *radius =
      find_source(geometry, static_cast<std::uint64_t>(ATTR_STD_RADIUS));
  if (radius == nullptr || !validate_payload(geometry, *radius, diagnostic)) {
    if (radius == nullptr) {
      diagnostic =
          label(geometry) + " has Cycles position keys without radius storage";
    }
    return false;
  }
  const auto *positions =
      std::get_if<std::vector<packed_float3>>(&position.payload);
  const auto *radii = std::get_if<std::vector<float>>(&radius->payload);
  if (positions == nullptr || radii == nullptr ||
      (radius->motion_steps != 1u &&
       radius->motion_steps != position.motion_steps)) {
    diagnostic = label(geometry) +
                 " has incompatible Cycles position/radius motion storage";
    return false;
  }
  const auto count = geometry.kind == GeometryAttributeKind::hair
                         ? geometry.key_count
                         : geometry.vertex_count;
  auto &destination = geometry.kind == GeometryAttributeKind::hair
                          ? image.curve_keys
                          : image.points;
  if (!append_offset(destination.size(),
                     geometry.kind == GeometryAttributeKind::pointcloud
                         ? geometry.primitive_offset
                         : 0u,
                     descriptor.offset)) {
    diagnostic =
        label(geometry) + " position/radius table offset exceeds int32";
    return false;
  }
  destination.reserve(destination.size() + positions->size());
  for (auto step = std::uint32_t{}; step < position.motion_steps; ++step) {
    const auto radius_step = radius->motion_steps == 1u ? 0u : step;
    for (auto index = std::size_t{}; index < count; ++index) {
      const auto &p =
          (*positions)[static_cast<std::size_t>(step) * count + index];
      const auto r =
          (*radii)[static_cast<std::size_t>(radius_step) * count + index];
      destination.emplace_back(packed_float4{p.x, p.y, p.z, r});
    }
  }
  descriptor.element = position.element;
  descriptor.type = NODE_ATTR_FLOAT4;
  return true;
}

[[nodiscard]] bool append_attribute(GeometryAttributeTableImage &image,
                                    const GeometryAttributeInput &geometry,
                                    const GeometryAttributeSource &source,
                                    AttributeDescriptor &descriptor,
                                    std::string &diagnostic) {
  if (!validate_payload(geometry, source, diagnostic)) {
    return false;
  }
  if ((geometry.kind == GeometryAttributeKind::hair ||
       geometry.kind == GeometryAttributeKind::pointcloud) &&
      source.standard == ATTR_STD_POSITION) {
    return append_position_radius(image, geometry, source, descriptor,
                                  diagnostic);
  }
  if ((geometry.kind == GeometryAttributeKind::hair ||
       geometry.kind == GeometryAttributeKind::pointcloud) &&
      source.standard == ATTR_STD_RADIUS) {
    descriptor.element = ATTR_ELEMENT_NONE;
    descriptor.type = NODE_ATTR_FLOAT;
    descriptor.offset = 0;
    return true;
  }

  descriptor.element = source.element;
  descriptor.type = source.type;
  const auto correction = offset_correction(geometry, source.element);
  const auto byte = (static_cast<std::uint32_t>(source.element) &
                     static_cast<std::uint32_t>(ATTR_ELEMENT_IS_BYTE)) != 0u;
  const auto normal =
      (static_cast<std::uint32_t>(source.element) &
       static_cast<std::uint32_t>(ATTR_ELEMENT_IS_NORMAL)) != 0u;
  const auto voxel = base_element(source.element) == ATTR_ELEMENT_VOXEL;

  if (voxel) {
    const auto slot = std::get<VoxelAttributeHandle>(source.payload).slot;
    if (slot >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
      diagnostic = label(geometry) + " voxel slot exceeds int32";
      return false;
    }
    descriptor.offset = static_cast<std::int32_t>(slot);
    return true;
  }
  if (byte) {
    if (!append_offset(image.attributes_uchar4.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " byte attribute offset exceeds int32";
      return false;
    }
    append_vector(image.attributes_uchar4, source);
    return true;
  }
  if (normal) {
    if (!append_offset(image.attributes_normal.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " normal attribute offset exceeds int32";
      return false;
    }
    append_vector(image.attributes_normal, source);
    return true;
  }

  switch (source.type) {
  case NODE_ATTR_FLOAT:
    if (!append_offset(image.attributes_float.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " float attribute offset exceeds int32";
      return false;
    }
    append_vector(image.attributes_float, source);
    return true;
  case NODE_ATTR_FLOAT2:
    if (!append_offset(image.attributes_float2.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " float2 attribute offset exceeds int32";
      return false;
    }
    append_vector(image.attributes_float2, source);
    return true;
  case NODE_ATTR_FLOAT3: {
    auto &destination = source.standard == ATTR_STD_POSITION
                            ? image.tri_verts
                            : image.attributes_float3;
    if (!append_offset(destination.size(), correction, descriptor.offset)) {
      diagnostic = label(geometry) + " float3 attribute offset exceeds int32";
      return false;
    }
    append_vector(destination, source);
    return true;
  }
  case NODE_ATTR_FLOAT4:
  case NODE_ATTR_RGBA:
    if (!append_offset(image.attributes_float4.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " float4 attribute offset exceeds int32";
      return false;
    }
    append_vector(image.attributes_float4, source);
    return true;
  case NODE_ATTR_MATRIX: {
    if (!append_offset(image.attributes_float4.size(), correction,
                       descriptor.offset)) {
      diagnostic = label(geometry) + " matrix attribute offset exceeds int32";
      return false;
    }
    const auto &values = std::get<std::vector<PackedTransform>>(source.payload);
    image.attributes_float4.reserve(image.attributes_float4.size() +
                                    values.size() * 3u);
    for (const auto &value : values) {
      image.attributes_float4.emplace_back(value.x);
      image.attributes_float4.emplace_back(value.y);
      image.attributes_float4.emplace_back(value.z);
    }
    return true;
  }
  }
  diagnostic = label(geometry) + " has an unknown Cycles attribute type";
  return false;
}

void append_unique_request(std::vector<std::uint64_t> &requests,
                           std::set<std::uint64_t> &seen, std::uint64_t id) {
  if (seen.emplace(id).second) {
    requests.emplace_back(id);
  }
}

[[nodiscard]] std::vector<std::uint64_t>
collect_requests(const GeometryAttributeInput &geometry,
                 std::string &diagnostic) {
  std::vector<std::uint64_t> result;
  std::set<std::uint64_t> seen;
  result.reserve(geometry.requested_attributes.size() + 4u);
  for (const auto id : geometry.requested_attributes) {
    if (id == static_cast<std::uint64_t>(ATTR_STD_NONE)) {
      diagnostic = label(geometry) + " requests ATTR_STD_NONE";
      return {};
    }
    append_unique_request(result, seen, id);
  }
  for (const auto &source : geometry.attributes) {
    switch (source.standard) {
    case ATTR_STD_POSITION:
    case ATTR_STD_VERTEX_NORMAL:
    case ATTR_STD_CORNER_NORMAL:
    case ATTR_STD_SHADOW_TRANSPARENCY:
      append_unique_request(result, seen,
                            static_cast<std::uint64_t>(source.standard));
      break;
    default:
      break;
    }
  }
  return result;
}

[[nodiscard]] std::int32_t find_map_offset(const std::vector<AttributeMap> &map,
                                           std::uint32_t begin,
                                           std::uint64_t id) noexcept {
  auto offset = begin;
  while (offset < map.size() && map[offset].id != id) {
    if (map[offset].id == static_cast<std::uint64_t>(ATTR_STD_NONE)) {
      if (map[offset].element == 0u) {
        return ATTR_STD_NOT_FOUND;
      }
      const auto link = map[offset].offset;
      if (link < 0) {
        return ATTR_STD_NOT_FOUND;
      }
      offset = static_cast<std::uint32_t>(link);
    } else {
      if (offset > std::numeric_limits<std::uint32_t>::max() -
                       static_cast<std::uint32_t>(ATTR_PRIM_TYPES)) {
        return ATTR_STD_NOT_FOUND;
      }
      offset += static_cast<std::uint32_t>(ATTR_PRIM_TYPES);
    }
  }
  if (offset >= map.size() || map[offset].element == ATTR_ELEMENT_NONE) {
    return ATTR_STD_NOT_FOUND;
  }
  return map[offset].offset;
}

} // namespace

packed_normal pack_geometry_normal(packed_float3 normal) noexcept {
  const auto inv_l1 = 1.0f / (std::fabs(normal.x) + std::fabs(normal.y) +
                              std::fabs(normal.z) + 1.0e-6f);
  auto vx = normal.x * inv_l1;
  auto vy = normal.y * inv_l1;
  const auto wrap_x = (1.0f - std::fabs(vy)) * std::copysign(1.0f, vx);
  const auto wrap_y = (1.0f - std::fabs(vx)) * std::copysign(1.0f, vy);
  if (normal.z < 0.0f) {
    vx = wrap_x;
    vy = wrap_y;
  }
  const auto encode = [](float value) noexcept {
    constexpr auto maximum = (1u << 16u) - 1u;
    constexpr auto half = static_cast<float>(maximum) / 2.0f;
    return static_cast<std::uint32_t>(std::clamp(
        value * half + (half + 0.5f), 0.0f, static_cast<float>(maximum)));
  };
  return {.value = encode(vx) | (encode(vy) << 16u)};
}

GeometryAttributeTableImage build_geometry_attribute_table(
    const std::vector<GeometryAttributeInput> &geometries) {
  GeometryAttributeTableImage result;
  result.geometries.reserve(geometries.size());

  for (const auto &geometry : geometries) {
    std::string diagnostic;
    if ((geometry.kind == GeometryAttributeKind::mesh ||
         geometry.kind == GeometryAttributeKind::volume) &&
        (geometry.primitive_count >
             std::numeric_limits<std::size_t>::max() / 3u ||
         geometry.corner_count != geometry.primitive_count * 3u)) {
      return reject(label(geometry) +
                    " does not have three corners per triangle primitive");
    }
    if (!validate_source_identities(geometry, diagnostic)) {
      return reject(std::move(diagnostic));
    }
    auto requests = collect_requests(geometry, diagnostic);
    if (!diagnostic.empty()) {
      return reject(std::move(diagnostic));
    }
    const auto map_offset = result.attribute_map.size();
    if (requests.size() == std::numeric_limits<std::size_t>::max()) {
      return reject(label(geometry) +
                    " attribute request count overflows size_t");
    }
    std::size_t map_entries{};
    if (!checked_product(requests.size() + 1u,
                         static_cast<std::uint32_t>(ATTR_PRIM_TYPES),
                         map_entries)) {
      return reject(label(geometry) + " attribute map size overflows size_t");
    }
    if (map_offset > std::numeric_limits<std::uint32_t>::max() ||
        map_entries > std::numeric_limits<std::uint32_t>::max() - map_offset) {
      return reject(label(geometry) + " attribute map exceeds uint32");
    }
    result.attribute_map.resize(map_offset + map_entries);
    auto cursor = map_offset;
    for (const auto id : requests) {
      AttributeDescriptor descriptor{
          .element = ATTR_ELEMENT_NONE, .type = NODE_ATTR_FLOAT, .offset = 0};
      if (const auto *source = find_source(geometry, id)) {
        if (!append_attribute(result, geometry, *source, descriptor,
                              diagnostic)) {
          return reject(std::move(diagnostic));
        }
      }
      result.attribute_map[cursor] = AttributeMap{
          .id = id,
          .offset = descriptor.offset,
          .element = static_cast<std::uint16_t>(descriptor.element),
          .type = static_cast<std::uint8_t>(descriptor.type),
          .pad = 0u};
      cursor += static_cast<std::size_t>(ATTR_PRIM_TYPES);
    }
    // resize() value-initialized both terminator lanes. This is exactly the
    // unchained Cycles geometry terminator: id NONE, element false, offset 0.
    const auto begin = static_cast<std::uint32_t>(map_offset);
    auto normal =
        find_map_offset(result.attribute_map, begin, ATTR_STD_CORNER_NORMAL);
    if (normal == ATTR_STD_NOT_FOUND) {
      normal =
          find_map_offset(result.attribute_map, begin, ATTR_STD_VERTEX_NORMAL);
    }
    result.geometries.emplace_back(FinalizedGeometryAttribute{
        .attribute_map_offset = begin,
        .position_offset =
            find_map_offset(result.attribute_map, begin, ATTR_STD_POSITION),
        .normal_offset = normal});
  }
  result.valid = true;
  return result;
}

} // namespace psycles::compiler::cycles_svm
