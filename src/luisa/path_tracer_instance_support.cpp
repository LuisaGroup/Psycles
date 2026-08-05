#include "path_tracer_instance_support.h"

#include "path_tracer_internal.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

void hash_word(std::uint64_t &hash, std::uint32_t word) noexcept {
  constexpr auto prime = std::uint64_t{1099511628211ull};
  for (auto byte = 0u; byte < 4u; ++byte) {
    hash ^= (word >> (byte * 8u)) & 0xffu;
    hash *= prime;
  }
}

void hash_size(std::uint64_t &hash, std::size_t size) noexcept {
  const auto value = static_cast<std::uint64_t>(size);
  hash_word(hash, static_cast<std::uint32_t>(value));
  hash_word(hash, static_cast<std::uint32_t>(value >> 32u));
}

[[nodiscard]] std::uint64_t
support_hash(const GeometryUpload &upload) noexcept {
  auto hash = std::uint64_t{14695981039346656037ull};
  hash_size(hash, upload.positions.size());
  hash_size(hash, upload.triangles.size());
  for (const auto point : upload.positions) {
    hash_word(hash, std::bit_cast<std::uint32_t>(point.x));
    hash_word(hash, std::bit_cast<std::uint32_t>(point.y));
    hash_word(hash, std::bit_cast<std::uint32_t>(point.z));
  }
  for (const auto triangle : upload.triangles) {
    hash_word(hash, triangle.i0);
    hash_word(hash, triangle.i1);
    hash_word(hash, triangle.i2);
  }
  return hash;
}

[[nodiscard]] bool same_bits(float a, float b) noexcept {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

[[nodiscard]] bool same_support_bits(const GeometryUpload &a,
                                     const GeometryUpload &b) noexcept {
  if (a.positions.size() != b.positions.size() ||
      a.triangles.size() != b.triangles.size()) {
    return false;
  }
  for (std::size_t i = 0u; i < a.positions.size(); ++i) {
    if (!same_bits(a.positions[i].x, b.positions[i].x) ||
        !same_bits(a.positions[i].y, b.positions[i].y) ||
        !same_bits(a.positions[i].z, b.positions[i].z)) {
      return false;
    }
  }
  for (std::size_t i = 0u; i < a.triangles.size(); ++i) {
    if (a.triangles[i].i0 != b.triangles[i].i0 ||
        a.triangles[i].i1 != b.triangles[i].i1 ||
        a.triangles[i].i2 != b.triangles[i].i2) {
      return false;
    }
  }
  return true;
}

} // namespace

CyclesFinalTriangleSupportClasses classify_cycles_final_triangle_supports(
    const contract::SceneSnapshot &scene,
    const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    const std::vector<GeometryUpload> &uploads) {
  CyclesFinalTriangleSupportClasses result;
  struct SupportClass {
    std::uint32_t upload_index{};
    std::uint32_t class_index{};
  };
  std::vector<SupportClass> classes;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
  for (const auto &[geometry_id, geometry] : scene.geometries) {
    const auto index = geometry_indices.find(geometry_id);
    if (index == geometry_indices.end() || index->second >= uploads.size()) {
      result.diagnostic = "geometry '" + geometry.name +
                          "' has no final triangle support upload";
      return result;
    }
    const auto &upload = uploads[index->second];
    const auto hash = support_hash(upload);
    auto &candidates = buckets[hash];
    const SupportClass *matching = nullptr;
    for (const auto candidate : candidates) {
      const auto &support_class = classes[candidate];
      if (same_support_bits(uploads[support_class.upload_index], upload)) {
        matching = &support_class;
        break;
      }
    }
    if (matching == nullptr) {
      const auto class_index = static_cast<std::uint32_t>(classes.size());
      candidates.emplace_back(classes.size());
      classes.emplace_back(SupportClass{.upload_index = index->second,
                                        .class_index = class_index});
      matching = &classes.back();
    }
    result.by_geometry.emplace(geometry_id, matching->class_index);
  }
  return result;
}

bool finalize_cycles_final_instance_supports(
    const contract::SceneSnapshot &scene,
    const CyclesFinalTriangleSupportClasses &support_classes,
    const std::map<contract::GeometryId, std::uint32_t> &geometry_indices,
    const std::vector<GeometryUpload> &uploads,
    std::span<CyclesInstanceIntersectionPlan> instance_plan,
    CyclesPrimitiveCompletionPlan &primitive_plan) {
  if (!support_classes.ok()) {
    return false;
  }
  std::map<contract::GeometryId, CyclesGeometrySupportView> supports;
  for (const auto &[geometry_id, upload_index] : geometry_indices) {
    if (upload_index >= uploads.size()) {
      // Curves live in a separate upload domain but share the scene's stable
      // geometry index map. They have no triangle support to adapt here.
      continue;
    }
    const auto &upload = uploads[upload_index];
    supports.emplace(
        geometry_id,
        make_cycles_geometry_support_view(
            std::span{upload.positions}, std::span{upload.triangles}));
  }
  return finalize_cycles_instance_intersection_plan(
      scene, support_classes.by_geometry, supports,
      instance_plan, primitive_plan);
}

} // namespace psycles::luisa_backend::detail
