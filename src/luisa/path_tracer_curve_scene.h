#pragma once

#include "path_tracer_internal.h"

#include <map>
#include <string>

namespace psycles::luisa_backend::detail {

struct CurveGeometryUpload {
  luisa::vector<luisa::float4> keys;
  luisa::vector<CurveSegmentGpu> segments;
  luisa::vector<luisa::compute::AABB> bounds;
  luisa::vector<luisa::uint> material_slots;
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
         luisa::vector<AttributeRangeGpu> &attribute_ranges) const;
};

} // namespace psycles::luisa_backend::detail
