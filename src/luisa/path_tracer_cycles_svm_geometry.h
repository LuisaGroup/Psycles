#pragma once

#include "path_tracer_internal.h"

#include <psycles/compiler/cycles_svm_geometry_scene.h>

#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {

// One transactional host image of the Cycles geometry-owned DeviceScene
// tables. `attribute_geometry_indices` is the proof-carrying correspondence
// between scene identities and GeometryAttributeTableImage::geometries;
// object finalization must use this map rather than assume Psycles resource
// order is Cycles geometry order.
struct CyclesSvmGeometrySceneImage {
  bool valid{};
  std::string diagnostic;
  compiler::cycles_svm::GeometryAttributeTableImage attributes;
  std::map<contract::GeometryId, std::uint32_t> attribute_geometry_indices;
  std::map<contract::LightId, std::uint32_t> light_attribute_geometry_indices;
  std::optional<std::uint32_t> background_attribute_geometry_index;
  std::vector<compiler::cycles_svm::packed_uint3> triangle_vertex_indices;
  std::vector<compiler::cycles_svm::KernelCurve> curves;
};

// Build after displacement has finalized `mesh_uploads`. The source object
// domain determines geometry order, shader request vectors are merged with
// Cycles' first-insertion-wins set algebra, and all typed arrays are committed
// only if every interval and source relation is valid.
[[nodiscard]] CyclesSvmGeometrySceneImage build_cycles_svm_geometry_scene_image(
    const contract::SceneSnapshot &snapshot,
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    const std::map<contract::MaterialId, std::uint32_t>
        &material_shader_indices,
    const compiler::cycles_svm::ObjectIdentityPlan &object_identities,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices,
    const std::map<contract::GeometryId, std::uint32_t>
        &triangle_primitive_offsets,
    const std::map<contract::GeometryId, std::uint32_t>
        &curve_primitive_offsets);

} // namespace psycles::luisa_backend::detail
