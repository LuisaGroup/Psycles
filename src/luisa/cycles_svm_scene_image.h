#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend::detail {

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
