#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace psycles::luisa_backend::detail {

inline constexpr std::uint32_t vt_max_mip_levels = 16u;

[[nodiscard]] constexpr std::uint32_t
vt_mip_extent(std::uint32_t base_extent, std::uint32_t mip_level) noexcept {
  if (base_extent == 0u) {
    return 0u;
  }
  if (mip_level >= 31u) {
    return 1u;
  }
  return std::max(1u, base_extent >> mip_level);
}

[[nodiscard]] constexpr std::uint32_t
vt_compute_mip_count(std::uint32_t width, std::uint32_t height) noexcept {
  if (width == 0u || height == 0u) {
    return 1u;
  }
  std::uint32_t extent = std::max(width, height);
  std::uint32_t count = 1u;
  while (extent > 1u && count < vt_max_mip_levels) {
    extent >>= 1u;
    ++count;
  }
  return count;
}

struct VtUnorm8MipLevel {
  std::uint32_t width{};
  std::uint32_t height{};
  std::vector<std::uint8_t> rgba;
};

struct VtFloat4MipLevel {
  std::uint32_t width{};
  std::uint32_t height{};
  std::vector<float> rgba;
};

// Generates mip levels 1 .. vt_compute_mip_count(width, height) - 1 using an
// area-preserving separable box filter (exact 3-tap weights on odd NPOT axes).
// For sRGB inputs, RGB is linearized via IEC 61966-2-1 before filtering and
// re-encoded per stored unorm8 mip level while preserving linear accumulation
// across successive downsamples. Alpha is always filtered linearly.
[[nodiscard]] std::vector<VtUnorm8MipLevel> vt_build_unorm8_tail_mips(
    std::span<const std::uint8_t> base_rgba,
    std::uint32_t width,
    std::uint32_t height,
    bool is_srgb);

[[nodiscard]] std::vector<VtFloat4MipLevel> vt_build_float4_tail_mips(
    std::span<const float> base_rgba,
    std::uint32_t width,
    std::uint32_t height);

} // namespace psycles::luisa_backend::detail
