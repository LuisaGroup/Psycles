#pragma once

#include <psycles/luisa/path_tracer.h>

#include <cstdint>
#include <span>

#include <luisa/runtime/rhi/pixel.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] constexpr luisa::compute::PixelStorage vt_byte4_pixel_storage(
    LuisaVirtualTextureCompressionMode mode) noexcept {
  switch (mode) {
    case LuisaVirtualTextureCompressionMode::off:
    case LuisaVirtualTextureCompressionMode::auto_hq:
    case LuisaVirtualTextureCompressionMode::auto_compact:
      return luisa::compute::PixelStorage::BYTE4;
    case LuisaVirtualTextureCompressionMode::bc7:
      return luisa::compute::PixelStorage::BC7;
    case LuisaVirtualTextureCompressionMode::bc3:
      return luisa::compute::PixelStorage::BC3;
    case LuisaVirtualTextureCompressionMode::bc1:
      return luisa::compute::PixelStorage::BC1;
  }
  return luisa::compute::PixelStorage::BYTE4;
}

[[nodiscard]] constexpr std::uint32_t vt_byte4_tile_bytes(
    std::uint32_t physical_tile_size,
    LuisaVirtualTextureCompressionMode mode) noexcept {
  const std::uint32_t blocks_per_axis = (physical_tile_size + 3u) / 4u;
  const std::uint32_t block_count = blocks_per_axis * blocks_per_axis;
  switch (mode) {
    case LuisaVirtualTextureCompressionMode::off:
    case LuisaVirtualTextureCompressionMode::auto_hq:
    case LuisaVirtualTextureCompressionMode::auto_compact:
      return physical_tile_size * physical_tile_size * 4u;
    case LuisaVirtualTextureCompressionMode::bc7:
    case LuisaVirtualTextureCompressionMode::bc3:
      return block_count * 16u;
    case LuisaVirtualTextureCompressionMode::bc1:
      return block_count * 8u;
  }
  return physical_tile_size * physical_tile_size * 4u;
}

void vt_fill_zero_byte4_tile(
    std::uint32_t physical_tile_size,
    LuisaVirtualTextureCompressionMode mode,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc1_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc3_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc4_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc5_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_encode_bc7_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept;

void vt_pack_byte4_tile_compressed(
    const std::uint8_t *pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t tile_x,
    std::uint32_t tile_y,
    std::uint32_t extension,
    std::uint32_t physical_tile_size,
    std::uint32_t logical_tile_size,
    std::uint32_t border,
    LuisaVirtualTextureCompressionMode compression,
    std::span<std::uint8_t> dst_bytes) noexcept;

void vt_decode_bc1_block(
    const std::uint8_t src[8],
    std::uint8_t dst_rgba[64]) noexcept;

void vt_decode_bc3_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept;

void vt_decode_bc4_block(
    const std::uint8_t src[8],
    std::uint8_t dst_rgba[64]) noexcept;

void vt_decode_bc5_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept;

void vt_decode_bc7_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept;

} // namespace psycles::luisa_backend::detail
