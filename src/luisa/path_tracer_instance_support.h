#pragma once

#include <psycles/contract/scene.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {

struct GeometryUpload;

struct CyclesFinalTriangleSupportClasses {
  std::map<contract::GeometryId, std::uint32_t> by_geometry;
  std::string diagnostic;

  [[nodiscard]] bool ok() const noexcept { return diagnostic.empty(); }
};

// Assigns the same class exactly when the final accelerator position and
// triangle-index arrays are bitwise equal. Shading attributes deliberately do
// not participate in this geometric equivalence relation.
[[nodiscard]] CyclesFinalTriangleSupportClasses
classify_cycles_final_triangle_supports(
    const contract::SceneSnapshot &scene,
    const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    const std::vector<GeometryUpload> &uploads);

} // namespace psycles::luisa_backend::detail
