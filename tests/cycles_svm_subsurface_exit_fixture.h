#pragma once

#include <psycles/compiler/cycles_svm_types.h>

#include <array>

namespace psycles::test_support {
namespace sss_exit_abi = compiler::cycles_svm;

// Inputs only. All expected state and BSDF results come from the original
// Cycles subsurface_shader_data_setup / surface_shader functions on HIP.
struct SubsurfaceExitInput {
  unsigned flags;
  unsigned normal_case;
  unsigned light_flags;
  std::array<float, 3u> random;
};

inline constexpr auto subsurface_exit_inputs = [] {
  std::array<SubsurfaceExitInput, 144u> result{};
  constexpr std::array flags{
      0u,
      unsigned(sss_exit_abi::SD_HAS_BSSRDF_BUMP),
      unsigned(sss_exit_abi::SD_HAS_BSSRDF_BUMP |
               sss_exit_abi::SD_USE_BUMP_MAP_CORRECTION),
      unsigned(sss_exit_abi::SD_HAS_BSSRDF_BUMP |
               sss_exit_abi::SD_USE_BUMP_MAP_CORRECTION |
               sss_exit_abi::SD_BACKFACING | sss_exit_abi::SD_TRANSPARENT)};
  // ShaderFlag belongs to the packed shader ID (not ShaderDataFlag).
  constexpr std::array lights{0u, 1u << 28u, (1u << 28u) | (1u << 27u)};
  constexpr std::array randoms{
      std::array{0.388530254f, 0.472640067f, 0.562300205f},
      std::array{0.02f, 0.97f, 0.625f},
      std::array{0.77f, 0.12f, 0.125f}};
  auto i = 0u;
  for (auto flag : flags) {
    for (auto normal = 0u; normal < 4u; ++normal) {
      for (auto light : lights) {
        for (auto random : randoms) {
          result[i++] = {flag, normal, light, random};
        }
      }
    }
  }
  return result;
}();
} // namespace psycles::test_support
