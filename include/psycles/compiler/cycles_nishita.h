#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/core/math.h>

#include <cstddef>
#include <vector>

namespace psycles::compiler::cycles_svm {

inline constexpr std::size_t nishita_texture_width = 512u;
inline constexpr std::size_t nishita_texture_height = 256u;
inline constexpr std::size_t nishita_texture_channels = 4u;

struct NishitaSunData {
  Vec3f pixel_bottom;
  Vec3f pixel_top;
  float earth_intersection_angle{};
};

// Exact host precomputation used by Cycles 5.2.1 SkyTextureNode. Sun pixels
// depend on the authored positive diameter even when the node later disables
// the disc by storing -1 in the SVM payload.
[[nodiscard]] NishitaSunData precompute_nishita_sun(
    const NishitaImageBinding &image, float angular_diameter);

// Cycles SkyLoader radiance converted to Psycles' top-down FLOAT4 upload
// convention. The shared native sampler maps Blender's bottom-up texture
// coordinates back to these rows, so the SVM handler retains Cycles' UVs.
// The returned vector contains width * height * 4 interleaved floats.
[[nodiscard]] std::vector<float>
precompute_nishita_texture(const NishitaImageBinding &image);

} // namespace psycles::compiler::cycles_svm
