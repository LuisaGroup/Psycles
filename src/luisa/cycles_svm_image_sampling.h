#pragma once

#include "cycles_texture_sampling.h"

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include <cstdint>

namespace psycles::luisa_backend::cycles_svm::detail {

// Cycles SVM stores util/types_image.h values in ImageManager handles, while
// the established Psycles native sampler helper accepts the surface-program
// canonical order (closest, linear, cubic) and (repeat, clip, extend, mirror).
// This is the single boundary permutation between those two contracts.
[[nodiscard]] constexpr std::uint32_t image_sampling_family(
    compiler::cycles_svm::ImageInterpolation interpolation) noexcept {
  using enum compiler::cycles_svm::ImageInterpolation;
  switch (interpolation) {
    case linear:
      return 1u;
    case closest:
      return 0u;
    case cubic:
    case smart:
      return 2u;
  }
  return 2u;
}

[[nodiscard]] constexpr std::uint32_t image_sampling_extension(
    compiler::cycles_svm::ImageExtension extension) noexcept {
  using enum compiler::cycles_svm::ImageExtension;
  switch (extension) {
    case repeat:
      return 0u;
    case extend:
      return 2u;
    case clip:
      return 1u;
    case mirror:
      return 3u;
  }
  return 0u;
}

[[nodiscard]] constexpr bool image_sampling_adapter_contract_holds() noexcept {
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  return image_sampling_family(ImageInterpolation::linear) == 1u &&
         image_sampling_family(ImageInterpolation::closest) == 0u &&
         image_sampling_family(ImageInterpolation::cubic) == 2u &&
         image_sampling_family(ImageInterpolation::smart) == 2u &&
         image_sampling_extension(ImageExtension::repeat) == 0u &&
         image_sampling_extension(ImageExtension::extend) == 2u &&
         image_sampling_extension(ImageExtension::clip) == 1u &&
         image_sampling_extension(ImageExtension::mirror) == 3u;
}

static_assert(image_sampling_adapter_contract_holds());

template<typename TextureHeap>
[[nodiscard]] luisa::compute::Float4 sample_image_2d(
    const TextureHeap &textures,
    luisa::compute::Expr<std::uint32_t> resource_handle,
    const Dual2 &uv,
    compiler::cycles_svm::ImageInterpolation interpolation,
    compiler::cycles_svm::ImageExtension extension) noexcept {
  auto sample_uv = uv.val;
  // Blender UVs are bottom-up; Psycles uploads decoded rows top-down. Cycles'
  // own device image layer performs the equivalent storage-orientation
  // adaptation below the SVM handler.
  sample_uv.y = 1.0f - sample_uv.y;
  return ::psycles::luisa_backend::detail::sample_cycles_texture_2d(
      textures, resource_handle, sample_uv,
      image_sampling_family(interpolation),
      image_sampling_extension(extension));
}

} // namespace psycles::luisa_backend::cycles_svm::detail
