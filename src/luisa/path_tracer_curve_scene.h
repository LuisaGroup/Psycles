#pragma once

#include "path_tracer_attribute_residency.h"
#include "path_tracer_internal.h"

#include <map>
#include <optional>
#include <string>

namespace psycles::luisa_backend::detail {

// Union of the exact per-segment Catmull--Rom intervals used to build the
// procedural-primitive AABBs. Object/volume classification must consume this
// same image: bounding only the authored keys is not conservative because a
// cubic segment may overshoot its endpoints.
struct CurveGeometryBounds {
  bool valid{};
  Vec3f minimum{};
  Vec3f maximum{};
};

[[nodiscard]] CurveGeometryBounds
build_curve_geometry_bounds(const contract::CurveGeometryDesc &geometry);

struct CurveGeometryUpload {
  luisa::vector<luisa::float4> keys;
  luisa::vector<CurveSegmentGpu> segments;
  luisa::vector<luisa::compute::AABB> bounds;
  luisa::vector<luisa::uint> material_slots;
  std::optional<std::string> default_uv_layer;
  std::map<std::string, luisa::vector<luisa::float2>, std::less<>> uv_layers;
  std::map<std::string, luisa::vector<luisa::float4>, std::less<>>
      color_attributes;
  luisa::vector<float> intercept;
  luisa::vector<float> length;
  luisa::vector<float> random;
};

[[nodiscard]] CurveGeometryUpload
build_curve_geometry_upload(const contract::CurveGeometryDesc &geometry,
                            std::uint32_t cycles_curve_offset,
                            std::uint32_t cycles_segment_offset);

struct CurveSceneUploadResult {
  std::string diagnostic;
  std::map<contract::GeometryId, std::uint32_t> resource_indices;
  // Exact intervals chosen by the single production resolver. Downstream
  // Cycles DeviceScene packing consumes these values; it must not reconstruct
  // a second, potentially divergent curve address space.
  std::map<contract::GeometryId, std::uint32_t> cycles_curve_offsets;
  std::map<contract::GeometryId, std::uint32_t> cycles_segment_offsets;

  [[nodiscard]] bool ok() const noexcept { return diagnostic.empty(); }
};

class CurveSceneUploadComponent {

public:
  [[nodiscard]] CurveSceneUploadResult
  upload(const std::shared_ptr<LuisaSceneData> &data,
         const SceneSnapshot &snapshot, Stream &stream,
         std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
         luisa::vector<GeometryGpu> &geometry_gpu,
         luisa::vector<MaterialBindingGpu> &geometry_materials,
         luisa::vector<AttributeBindingGpu> &attribute_bindings,
         luisa::vector<AttributeRangeGpu> &attribute_ranges,
         const SceneAttributeResidencyPlan &attribute_residency,
         std::uint32_t &next_attribute_slot) const;
};

} // namespace psycles::luisa_backend::detail
