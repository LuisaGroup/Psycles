#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/cycles_svm_geometry_scene.h>
#include <psycles/compiler/cycles_svm_object_scene.h>
#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>

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
  std::map<contract::LightId, std::uint32_t>
      light_attribute_geometry_indices;
  std::optional<std::uint32_t> background_attribute_geometry_index;
  std::vector<compiler::cycles_svm::packed_uint3> triangle_vertex_indices;
  // Cycles DeviceScene::tri_shader, in the same sparse global primitive
  // domain as triangle_vertex_indices. Every live entry contains the raw
  // Scene::shaders index plus SHADER_CAST_SHADOW/SMOOTH_NORMAL decoration.
  std::vector<std::uint32_t> triangle_shaders;
  std::vector<compiler::cycles_svm::KernelCurve> curves;
};

// Dense, hole-preserving projection of Cycles' scene->objects and
// DeviceScene::object_flag arrays. Unsupported source objects remain zero
// records at their exact source indices; represented objects are finalized
// only against the post-displacement geometry image above.
struct CyclesSvmObjectSceneImage {
  bool valid{};
  std::string diagnostic;
  std::vector<compiler::cycles_svm::KernelObject> objects;
  std::vector<std::uint32_t> object_flags;
};

[[nodiscard]] constexpr std::int32_t cycles_svm_curve_primitive_type(
    contract::CurveShape shape) noexcept {
  using enum contract::CurveShape;
  using namespace compiler::cycles_svm;
  switch (shape) {
  case ribbon:
    return PRIMITIVE_CURVE_RIBBON;
  case thick:
    return PRIMITIVE_CURVE_THICK;
  case thick_linear:
    return PRIMITIVE_CURVE_THICK_LINEAR;
  }
  return PRIMITIVE_NONE;
}

// Device projection of one Cycles ImageManager handle. The source texture
// identity and its immutable sampler are deliberately separate: equal source
// images may occupy several SVM handles without duplicating image storage.
// Packing the two two-bit sampler coordinates keeps the table at eight bytes
// per handle and, unlike a scene-sized shader switch, bounds generated code by
// the fixed Cycles sampler algebra.
struct CyclesSvmImageBindingGpu {
  luisa::uint texture_slot{};
  luisa::uint sampler{};
};

// Isomorphic projection of Cycles 5.2.1 KernelParticle. The raw exporter
// system ID is intentionally absent: it has completed its sole job once the
// host prefix algebra resolves object particle indices.
struct alignas(16) CyclesSvmParticleGpu {
  luisa::uint index{};
  float age{};
  float lifetime{};
  float size{};
  luisa::float4 rotation{};
  luisa::float4 location{};
  luisa::float4 velocity{};
  luisa::float4 angular_velocity{};
};

[[nodiscard]] constexpr CyclesSvmParticleGpu make_cycles_svm_particle(
    const contract::CyclesParticleSource &source) noexcept {
  return {
      .index = source.source_index,
      .age = source.age,
      .lifetime = source.lifetime,
      .size = source.size,
      .rotation = {source.rotation.x, source.rotation.y, source.rotation.z,
                   source.rotation.w},
      .location = {source.location.x, source.location.y, source.location.z,
                   0.0f},
      .velocity = {source.velocity.x, source.velocity.y, source.velocity.z,
                   0.0f},
      .angular_velocity = {source.angular_velocity.x,
                           source.angular_velocity.y,
                           source.angular_velocity.z, 0.0f}};
}

inline constexpr std::uint32_t cycles_svm_image_interpolation_mask = 0x3u;
inline constexpr std::uint32_t cycles_svm_image_extension_shift = 2u;
inline constexpr std::uint32_t cycles_svm_image_extension_mask =
    0x3u << cycles_svm_image_extension_shift;

[[nodiscard]] constexpr std::uint32_t cycles_svm_image_sampling_family(
    compiler::cycles_svm::ImageInterpolation interpolation) noexcept {
  using enum compiler::cycles_svm::ImageInterpolation;
  switch (interpolation) {
  case closest:
    return 0u;
  case linear:
    return 1u;
  case cubic:
  case smart:
    return 2u;
  }
  return 2u;
}

[[nodiscard]] constexpr std::uint32_t cycles_svm_image_sampling_extension(
    compiler::cycles_svm::ImageExtension extension) noexcept {
  using enum compiler::cycles_svm::ImageExtension;
  switch (extension) {
  case repeat:
    return 0u;
  case clip:
    return 1u;
  case extend:
    return 2u;
  case mirror:
    return 3u;
  }
  return 0u;
}

[[nodiscard]] constexpr std::uint32_t cycles_svm_image_sampler(
    compiler::cycles_svm::ImageInterpolation interpolation,
    compiler::cycles_svm::ImageExtension extension) noexcept {
  return cycles_svm_image_sampling_family(interpolation) |
         (cycles_svm_image_sampling_extension(extension)
          << cycles_svm_image_extension_shift);
}

[[nodiscard]] constexpr CyclesSvmImageBindingGpu make_cycles_svm_image_binding(
    std::uint32_t texture_slot,
    compiler::cycles_svm::ImageInterpolation interpolation,
    compiler::cycles_svm::ImageExtension extension) noexcept {
  return {.texture_slot = texture_slot,
          .sampler = cycles_svm_image_sampler(interpolation, extension)};
}

[[nodiscard]] constexpr bool
cycles_svm_image_binding_contract_holds() noexcept {
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  return sizeof(CyclesSvmImageBindingGpu) == 2u * sizeof(std::uint32_t) &&
         cycles_svm_image_sampling_family(ImageInterpolation::linear) == 1u &&
         cycles_svm_image_sampling_family(ImageInterpolation::closest) == 0u &&
         cycles_svm_image_sampling_family(ImageInterpolation::cubic) == 2u &&
         cycles_svm_image_sampling_family(ImageInterpolation::smart) == 2u &&
         cycles_svm_image_sampling_extension(ImageExtension::repeat) == 0u &&
         cycles_svm_image_sampling_extension(ImageExtension::clip) == 1u &&
         cycles_svm_image_sampling_extension(ImageExtension::extend) == 2u &&
         cycles_svm_image_sampling_extension(ImageExtension::mirror) == 3u;
}

static_assert(cycles_svm_image_binding_contract_holds());
static_assert(alignof(CyclesSvmParticleGpu) == 16u);
static_assert(sizeof(CyclesSvmParticleGpu) == 80u);
static_assert(offsetof(CyclesSvmParticleGpu, index) == 0u);
static_assert(offsetof(CyclesSvmParticleGpu, age) == 4u);
static_assert(offsetof(CyclesSvmParticleGpu, lifetime) == 8u);
static_assert(offsetof(CyclesSvmParticleGpu, size) == 12u);
static_assert(offsetof(CyclesSvmParticleGpu, rotation) == 16u);
static_assert(offsetof(CyclesSvmParticleGpu, location) == 32u);
static_assert(offsetof(CyclesSvmParticleGpu, velocity) == 48u);
static_assert(offsetof(CyclesSvmParticleGpu, angular_velocity) == 64u);

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::CyclesSvmImageBindingGpu,
             texture_slot, sampler){};
LUISA_STRUCT(psycles::luisa_backend::detail::CyclesSvmParticleGpu,
             index, age, lifetime, size, rotation, location, velocity,
             angular_velocity){};
