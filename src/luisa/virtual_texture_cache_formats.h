#pragma once

#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include <cstdint>
#include <span>
#include <string_view>

#include <luisa/runtime/rhi/pixel.h>

namespace psycles::luisa_backend::detail {

// Concrete GPU tile storage format assigned per (source, extension) virtual
// texture range. Values 0..3 match the legacy BYTE4 .vtpages compression tags.
enum class VtTileFormat : std::uint8_t {
  rgba8 = 0u,
  bc7 = 1u,
  bc3 = 2u,
  bc1 = 3u,
  bc4 = 4u,
  bc5 = 5u,
  bc6h = 6u,
  short4 = 7u,
  short1 = 8u,
  half4 = 9u,
  half1 = 10u,
  float4 = 11u,
  float1 = 12u,
};

inline constexpr std::size_t kVtTileFormatCount = 13u;

enum class VtChannelSwizzle : std::uint32_t {
  rgba = 0u,
  rrr1 = 1u,
  rrrg = 2u,
};

[[nodiscard]] constexpr std::string_view vt_tile_format_name(
    VtTileFormat fmt) noexcept {
  switch (fmt) {
    case VtTileFormat::rgba8:
      return "rgba8";
    case VtTileFormat::bc7:
      return "bc7";
    case VtTileFormat::bc3:
      return "bc3";
    case VtTileFormat::bc1:
      return "bc1";
    case VtTileFormat::bc4:
      return "bc4";
    case VtTileFormat::bc5:
      return "bc5";
    case VtTileFormat::bc6h:
      return "bc6h";
    case VtTileFormat::short4:
      return "short4";
    case VtTileFormat::short1:
      return "short1";
    case VtTileFormat::half4:
      return "half4";
    case VtTileFormat::half1:
      return "half1";
    case VtTileFormat::float4:
      return "float4";
    case VtTileFormat::float1:
      return "float1";
  }
  return "rgba8";
}

[[nodiscard]] constexpr luisa::compute::PixelStorage vt_tile_format_storage(
    VtTileFormat fmt) noexcept {
  switch (fmt) {
    case VtTileFormat::rgba8:
      return luisa::compute::PixelStorage::BYTE4;
    case VtTileFormat::bc7:
      return luisa::compute::PixelStorage::BC7;
    case VtTileFormat::bc3:
      return luisa::compute::PixelStorage::BC3;
    case VtTileFormat::bc1:
      return luisa::compute::PixelStorage::BC1;
    case VtTileFormat::bc4:
      return luisa::compute::PixelStorage::BC4;
    case VtTileFormat::bc5:
      return luisa::compute::PixelStorage::BC5;
    case VtTileFormat::bc6h:
      return luisa::compute::PixelStorage::BC6;
    case VtTileFormat::short4:
      return luisa::compute::PixelStorage::SHORT4;
    case VtTileFormat::short1:
      return luisa::compute::PixelStorage::SHORT1;
    case VtTileFormat::half4:
      return luisa::compute::PixelStorage::HALF4;
    case VtTileFormat::half1:
      return luisa::compute::PixelStorage::HALF1;
    case VtTileFormat::float4:
      return luisa::compute::PixelStorage::FLOAT4;
    case VtTileFormat::float1:
      return luisa::compute::PixelStorage::FLOAT1;
  }
  return luisa::compute::PixelStorage::BYTE4;
}

[[nodiscard]] constexpr VtChannelSwizzle vt_tile_format_swizzle(
    VtTileFormat fmt) noexcept {
  switch (fmt) {
    case VtTileFormat::bc4:
    case VtTileFormat::short1:
    case VtTileFormat::half1:
    case VtTileFormat::float1:
      return VtChannelSwizzle::rrr1;
    case VtTileFormat::bc5:
      return VtChannelSwizzle::rrrg;
    default:
      return VtChannelSwizzle::rgba;
  }
}

[[nodiscard]] constexpr bool vt_tile_format_is_high_precision(
    VtTileFormat fmt) noexcept {
  switch (fmt) {
    case VtTileFormat::bc6h:
    case VtTileFormat::short4:
    case VtTileFormat::short1:
    case VtTileFormat::half4:
    case VtTileFormat::half1:
    case VtTileFormat::float4:
    case VtTileFormat::float1:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] constexpr bool vt_tile_format_is_block_compressed(
    VtTileFormat fmt) noexcept {
  switch (fmt) {
    case VtTileFormat::bc7:
    case VtTileFormat::bc3:
    case VtTileFormat::bc1:
    case VtTileFormat::bc4:
    case VtTileFormat::bc5:
    case VtTileFormat::bc6h:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] constexpr std::uint32_t vt_tile_format_bytes(
    std::uint32_t physical_tile_size,
    VtTileFormat fmt) noexcept {
  const std::uint32_t texels = physical_tile_size * physical_tile_size;
  const std::uint32_t blocks_per_axis = (physical_tile_size + 3u) / 4u;
  const std::uint32_t blocks = blocks_per_axis * blocks_per_axis;
  switch (fmt) {
    case VtTileFormat::rgba8:
      return texels * 4u;
    case VtTileFormat::bc7:
    case VtTileFormat::bc3:
    case VtTileFormat::bc5:
    case VtTileFormat::bc6h:
      return blocks * 16u;
    case VtTileFormat::bc1:
    case VtTileFormat::bc4:
      return blocks * 8u;
    case VtTileFormat::short4:
    case VtTileFormat::half4:
      return texels * 8u;
    case VtTileFormat::short1:
    case VtTileFormat::half1:
      return texels * 2u;
    case VtTileFormat::float4:
      return texels * 16u;
    case VtTileFormat::float1:
      return texels * 4u;
  }
  return texels * 4u;
}

[[nodiscard]] constexpr std::uint8_t vt_disk_compression_tag(
    VtTileFormat fmt) noexcept {
  return (fmt == VtTileFormat::float4) ? 0u : static_cast<std::uint8_t>(fmt);
}

[[nodiscard]] constexpr VtTileFormat vt_tile_format_from_disk_key(
    std::uint8_t storage,
    std::uint8_t compression) noexcept {
  if (compression == 0u &&
      storage == static_cast<std::uint8_t>(io::DecodedImageStorage::float32)) {
    return VtTileFormat::float4;
  }
  if (static_cast<std::size_t>(compression) < kVtTileFormatCount) {
    return static_cast<VtTileFormat>(compression);
  }
  return storage == static_cast<std::uint8_t>(io::DecodedImageStorage::float32)
             ? VtTileFormat::float4
             : VtTileFormat::rgba8;
}

[[nodiscard]] VtTileFormat vt_select_tile_format(
    LuisaVirtualTextureCompressionMode mode,
    const io::ProbedImageInfo &probed,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space,
    std::uint32_t extension,
    bool supports_batch_copy) noexcept;

void vt_fill_zero_tile(
    std::uint32_t physical_tile_size,
    VtTileFormat fmt,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc6h_tile(
    const float *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_decode_bc6h_block(
    const std::uint8_t src[16],
    float dst_rgba[64]) noexcept;

void vt_pack_tile_formatted(
    const std::uint8_t *unorm8_pixels,
    const float *float_pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t tile_x,
    std::uint32_t tile_y,
    std::uint32_t extension,
    std::uint32_t physical_tile_size,
    std::uint32_t logical_tile_size,
    std::uint32_t border,
    VtTileFormat fmt,
    std::span<std::uint8_t> dst_bytes) noexcept;

[[nodiscard]] std::uint16_t vt_float_to_half_bits(float v) noexcept;
[[nodiscard]] float vt_half_bits_to_float(std::uint16_t h) noexcept;

} // namespace psycles::luisa_backend::detail
