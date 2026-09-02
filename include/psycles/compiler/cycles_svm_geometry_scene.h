#pragma once

#include <psycles/compiler/cycles_svm_types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace psycles::compiler::cycles_svm {

// Exact source payload categories accepted by Cycles' AttributeTableBuilder.
// The payload remains typed until it reaches the matching DeviceScene array;
// no float4-erased intermediate representation exists at this boundary.
struct VoxelAttributeHandle {
  std::uint32_t slot{};
};

using GeometryAttributePayload =
    std::variant<std::vector<float>, std::vector<packed_float2>,
                 std::vector<packed_float3>, std::vector<packed_float4>,
                 std::vector<uchar4>, std::vector<packed_normal>,
                 std::vector<PackedTransform>, VoxelAttributeHandle>;

struct GeometryAttributeSource {
  // Standard and named identities are independent aliases of one Cycles
  // Attribute. A source may carry both (for example the active UV layer).
  AttributeStandard standard{ATTR_STD_NONE};
  std::optional<std::uint64_t> named_id;
  AttributeElement element{ATTR_ELEMENT_NONE};
  NodeAttributeType type{NODE_ATTR_FLOAT};
  // Center storage followed by Cycles' non-center motion buffers. A value of
  // one denotes the ordinary static attribute.
  std::uint32_t motion_steps{1u};
  GeometryAttributePayload payload{std::vector<float>{}};
};

enum class GeometryAttributeKind : std::uint8_t {
  mesh,
  hair,
  pointcloud,
  volume,
  light,
};

// One finalized Geometry input. `requested_attributes` is the already-unioned
// image of global and used-shader requests. The builder adds the four Cycles
// mandatory existing attributes (P, vertex/corner N, shadow transparency)
// itself, so callers cannot accidentally omit traversal-critical storage.
struct GeometryAttributeInput {
  std::string name;
  GeometryAttributeKind kind{GeometryAttributeKind::mesh};
  std::uint32_t primitive_offset{};
  std::size_t vertex_count{};
  std::size_t primitive_count{};
  std::size_t corner_count{};
  std::size_t curve_count{};
  std::size_t key_count{};
  std::vector<std::uint64_t> requested_attributes;
  std::vector<GeometryAttributeSource> attributes;
};

// Geometry-owned values consumed by ObjectManager::device_update_geom_offsets.
// Every offset is final: ATTR_STD_NOT_FOUND is a valid result for empty or
// non-applicable geometry and is distinct from the type-state used before this
// image exists.
struct FinalizedGeometryAttribute {
  std::uint32_t attribute_map_offset{};
  std::int32_t position_offset{ATTR_STD_NOT_FOUND};
  std::int32_t normal_offset{ATTR_STD_NOT_FOUND};
};

// Isomorphic host image of the Cycles DeviceScene attribute storage. Attribute
// maps use the two-lane ATTR_PRIM_TYPES stride; the second lane remains the
// zeroed SUBD descriptor until subdivision support supplies it explicitly.
struct GeometryAttributeTableImage {
  bool valid{};
  std::string diagnostic;
  std::vector<AttributeMap> attribute_map;
  std::vector<float> attributes_float;
  std::vector<packed_float2> attributes_float2;
  std::vector<packed_float3> attributes_float3;
  std::vector<packed_float4> attributes_float4;
  std::vector<uchar4> attributes_uchar4;
  std::vector<packed_normal> attributes_normal;
  std::vector<packed_float3> tri_verts;
  std::vector<packed_float4> curve_keys;
  std::vector<packed_float4> points;
  std::vector<FinalizedGeometryAttribute> geometries;
};

[[nodiscard]] packed_normal pack_geometry_normal(packed_float3 normal) noexcept;

[[nodiscard]] GeometryAttributeTableImage build_geometry_attribute_table(
    const std::vector<GeometryAttributeInput> &geometries);

} // namespace psycles::compiler::cycles_svm
