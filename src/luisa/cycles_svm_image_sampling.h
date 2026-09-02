#pragma once

#include "cycles_texture_sampling.h"
#include "cycles_svm_scene_image.h"

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cstdint>

namespace psycles::luisa_backend::cycles_svm::detail {

// Cycles SVM stores util/types_image.h values in ImageManager handles, while
// the established Psycles native sampler helper accepts the surface-program
// canonical order (closest, linear, cubic) and (repeat, clip, extend, mirror).
// This is the single boundary permutation between those two contracts.
[[nodiscard]] constexpr std::uint32_t image_sampling_family(
    compiler::cycles_svm::ImageInterpolation interpolation) noexcept {
  return ::psycles::luisa_backend::detail::
      cycles_svm_image_sampling_family(interpolation);
}

[[nodiscard]] constexpr std::uint32_t image_sampling_extension(
    compiler::cycles_svm::ImageExtension extension) noexcept {
  return ::psycles::luisa_backend::detail::
      cycles_svm_image_sampling_extension(extension);
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

// Resolve a runtime Cycles ImageManager handle without generating one shader
// branch per scene image. The device table selects texture storage and the
// two finite sampler coordinates; the host loop below emits exactly the fixed
// 3 x 4 Cycles sampler product for every scene size.
template<typename TextureHeap, typename ImageBindingBuffer>
[[nodiscard]] luisa::compute::Float4 sample_scene_image_2d(
    const TextureHeap &textures,
    const ImageBindingBuffer &bindings,
    luisa::compute::Expr<std::int32_t> image_id,
    const Dual2 &uv) noexcept {
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  namespace scene_detail = ::psycles::luisa_backend::detail;

  const auto binding = bindings->read(image_id.cast<std::uint32_t>());
  const luisa::compute::UInt interpolation =
      binding.sampler & scene_detail::cycles_svm_image_interpolation_mask;
  const luisa::compute::UInt extension =
      (binding.sampler & scene_detail::cycles_svm_image_extension_mask) >>
      scene_detail::cycles_svm_image_extension_shift;
  luisa::compute::Float4 result = luisa::compute::make_float4(0.0f);
  constexpr std::array interpolation_domain{
      ImageInterpolation::closest,
      ImageInterpolation::linear,
      ImageInterpolation::cubic};
  constexpr std::array extension_domain{
      ImageExtension::repeat,
      ImageExtension::clip,
      ImageExtension::extend,
      ImageExtension::mirror};
  for (const auto static_interpolation : interpolation_domain) {
    for (const auto static_extension : extension_domain) {
      const auto interpolation_code =
          image_sampling_family(static_interpolation);
      const auto extension_code =
          image_sampling_extension(static_extension);
      $if((interpolation == interpolation_code) &
          (extension == extension_code)) {
        result = sample_image_2d(
            textures, binding.texture_slot, uv,
            static_interpolation, static_extension);
      };
    }
  }
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
