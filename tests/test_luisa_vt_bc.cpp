#include "cycles_svm_scene_image.h"
#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_cache_bc.h"
#include "virtual_texture_cache_formats.h"
#include "virtual_texture_cache_gpu_bc.h"
#include "virtual_texture_cache_prepack.h"

namespace psycles::vt_bc_test {
void test_tex_compress_ext_direct(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream);
void test_gpu_bc_compressor(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream);
void run_gpu_bc_benchmark(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream);
} // namespace psycles::vt_bc_test

#include <psycles/contract/scene.h>
#include <psycles/luisa/path_tracer.h>

#if defined(PSYCLES_WITH_OPENIMAGEIO)
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>
#endif

#include <luisa/backends/ext/batch_copy_cmd.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;
using namespace luisa::compute::coro;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

[[nodiscard]] double compute_psnr(
    const std::uint8_t *ref,
    const std::uint8_t *act,
    std::size_t count,
    bool rgb_only = false) noexcept {
  double sum_sq = 0.0;
  std::size_t n = 0u;
  for (std::size_t i = 0u; i < count; i += 4u) {
    const int ch_end = rgb_only ? 3 : 4;
    for (int ch = 0; ch < ch_end; ++ch) {
      const double d =
          static_cast<double>(ref[i + ch]) - static_cast<double>(act[i + ch]);
      sum_sq += d * d;
      ++n;
    }
  }
  const double mse = sum_sq / static_cast<double>(std::max<std::size_t>(n, 1u));
  if (mse <= 1e-12) {
    return 99.0;
  }
  return 10.0 * std::log10((255.0 * 255.0) / mse);
}

#if defined(PSYCLES_WITH_OPENIMAGEIO)
template <typename T>
[[nodiscard]] std::vector<std::uint8_t>
encode_oiio_mem(std::string_view ext_name, std::uint32_t width,
                std::uint32_t height, int channels, OIIO::TypeDesc fmt,
                const T *data) {
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height),
                       channels, fmt};
  const std::string fname = "vt_bc_test." + std::string{ext_name};
  auto out = OIIO::ImageOutput::create(fname, &mem_io);
  require(out && out->supports("ioproxy"), "OIIO ioproxy unsupported");
  require(out->open(fname, spec), "OIIO open mem failed");
  require(out->write_image(fmt, data), "OIIO write mem failed");
  require(out->close(), "OIIO close mem failed");
  return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
}
#endif

[[nodiscard]] std::vector<std::uint8_t>
encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                 const std::vector<std::uint8_t> &rgba8) {
  require(rgba8.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA8 buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::UINT8};
  if (auto out = OIIO::ImageOutput::create("vt_bc_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_bc_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
#endif
  std::ostringstream hdr;
  hdr << "P6\n" << width << ' ' << height << "\n255\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  for (std::size_t i = 0u; i < rgba8.size(); i += 4u) {
    bytes.push_back(rgba8[i]);
    bytes.push_back(rgba8[i + 1u]);
    bytes.push_back(rgba8[i + 2u]);
  }
  return bytes;
}

[[nodiscard]] std::vector<std::uint8_t>
encode_exr_rgba32f(std::uint32_t width, std::uint32_t height,
                   const std::vector<float> &rgba32f) {
  require(rgba32f.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA32F buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::FLOAT};
  if (auto out = OIIO::ImageOutput::create("vt_bc_test.exr", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_bc_test.exr", spec), "OIIO open EXR mem failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write EXR mem failed");
    require(out->close(), "OIIO close EXR mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
#endif
  std::ostringstream hdr;
  hdr << "PF\n" << width << ' ' << height << "\n-1.0\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  for (std::uint32_t row = height; row > 0u; --row) {
    const std::size_t y = row - 1u;
    for (std::uint32_t x = 0u; x < width; ++x) {
      const float *px = &rgba32f[(y * width + x) * 4u];
      const auto *raw = reinterpret_cast<const std::uint8_t *>(px);
      bytes.insert(bytes.end(), raw, raw + 3u * sizeof(float));
    }
  }
  return bytes;
}

// Builds a synthetic 64x64 RGBA8 test tile containing solid zero, gradient,
// varying-alpha cutout, and sharp high-contrast opaque 2-region blocks.
[[nodiscard]] std::vector<std::uint8_t> make_synthetic_tile_rgba(std::uint32_t P) {
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(P) * P * 4u, 0u);
  for (std::uint32_t y = 0u; y < P; ++y) {
    for (std::uint32_t x = 0u; x < P; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * P + x) * 4u;
      const std::uint32_t bx = x / 4u;
      const std::uint32_t by = y / 4u;
      if (by == 0u && bx == 0u) {
        // Transparent black solid block.
        continue;
      }
      if (by == 0u && bx == 1u) {
        // Mixed-LSB solid block.
        rgba[off + 0u] = 100u;
        rgba[off + 1u] = 101u;
        rgba[off + 2u] = 102u;
        rgba[off + 3u] = 255u;
      } else if (by < 4u) {
        // Smooth RGBA gradient + independent alpha.
        rgba[off + 0u] = static_cast<std::uint8_t>((x * 255u) / P);
        rgba[off + 1u] = static_cast<std::uint8_t>((y * 255u) / (P / 4u));
        rgba[off + 2u] = static_cast<std::uint8_t>(((x + y) * 199u) / P);
        rgba[off + 3u] = static_cast<std::uint8_t>((x * 240u) / P + 15u);
      } else if (by < 8u) {
        // Sharp opaque 2-region edge (vertical / diagonal / horizontal).
        const bool side = ((x % 8u) < 4u) && ((x + y) % 7u < 4u);
        rgba[off + 0u] = side ? 240u : 20u;
        rgba[off + 1u] = side ? 24u : 228u;
        rgba[off + 2u] = side ? 16u : 244u;
        rgba[off + 3u] = 255u;
      } else {
        // Smooth opaque texture pattern (triangle wave avoids synthetic 255->0 wrap cliffs).
        auto tri_wave = [](std::uint32_t val) noexcept -> std::uint8_t {
          const std::uint32_t m = val & 511u;
          return static_cast<std::uint8_t>(m < 256u ? m : (511u - m));
        };
        rgba[off + 0u] = tri_wave(x * 7u + y * 3u + 17u);
        rgba[off + 1u] = tri_wave(x * 3u + y * 11u + 61u);
        rgba[off + 2u] = tri_wave(x * 5u + y * 5u + 131u);
        rgba[off + 3u] = 255u;
      }
    }
  }
  return rgba;
}

void write_bc7_mode1_probe_block(
    std::uint32_t part,
    std::uint8_t dst[16]) noexcept {
  // Craft a known valid Mode 1 block with distinct RGB in subsets 0 and 1,
  // choosing indices that specifically test anchor0 (idx[0] < 4) and anchor2.
  static constexpr std::array<std::uint8_t, 16u> kAnchor2{
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
  std::uint64_t lo = 0ull;
  std::uint64_t hi = 0ull;
  std::uint32_t bp = 0u;
  auto push = [&](std::uint32_t val, std::uint32_t bits) noexcept {
    const std::uint64_t m = static_cast<std::uint64_t>(val) & ((1ull << bits) - 1ull);
    if (bp < 64u) {
      lo |= (m << bp);
      if (bp + bits > 64u) {
        hi |= (m >> (64u - bp));
      }
    } else {
      hi |= (m << (bp - 64u));
    }
    bp += bits;
  };
  push(0x02u, 2u);
  push(part, 6u);
  // R0,R1,R2,R3; G0,G1,G2,G3; B0,B1,B2,B3 (6 bits each)
  const std::array<std::uint32_t, 12u> ep6{
      4u, 60u, 8u, 56u,  // R
      60u, 8u, 4u, 4u,   // G
      8u, 8u, 60u, 8u    // B
  };
  for (std::uint32_t v : ep6) {
    push(v, 6u);
  }
  push(0u, 1u); // P0
  push(1u, 1u); // P1
  const std::uint32_t anchor2 = kAnchor2[part];
  for (std::uint32_t i = 0u; i < 16u; ++i) {
    const std::uint32_t bits = (i == 0u || i == anchor2) ? 2u : 3u;
    const std::uint32_t idx =
        (bits == 2u) ? (i == 0u ? 1u : 2u) : (static_cast<std::uint32_t>(i * 3u + 1u) & 7u);
    push(idx, bits);
  }
  std::memcpy(dst, &lo, 8u);
  std::memcpy(dst + 8u, &hi, 8u);
}

void write_bc7_mode5_probe_block(
    std::uint32_t rot,
    std::uint8_t dst[16]) noexcept {
  std::uint64_t lo = 0ull;
  std::uint64_t hi = 0ull;
  std::uint32_t bp = 0u;
  auto push = [&](std::uint32_t val, std::uint32_t bits) noexcept {
    const std::uint64_t m = static_cast<std::uint64_t>(val) & ((1ull << bits) - 1ull);
    if (bp < 64u) {
      lo |= (m << bp);
      if (bp + bits > 64u) {
        hi |= (m >> (64u - bp));
      }
    } else {
      hi |= (m << (bp - 64u));
    }
    bp += bits;
  };
  push(0x20u, 6u);
  push(rot, 2u);
  // 7-bit RGB pairs
  push(10u, 7u);
  push(110u, 7u);
  push(90u, 7u);
  push(20u, 7u);
  push(30u, 7u);
  push(100u, 7u);
  // 8-bit A pair
  push(15u, 8u);
  push(240u, 8u);
  for (std::uint32_t i = 0u; i < 16u; ++i) {
    push(i == 0u ? 1u : ((i + 1u) & 3u), i == 0u ? 1u : 2u);
  }
  for (std::uint32_t i = 0u; i < 16u; ++i) {
    push(i == 0u ? 0u : ((i * 3u) & 3u), i == 0u ? 1u : 2u);
  }
  std::memcpy(dst, &lo, 8u);
  std::memcpy(dst + 8u, &hi, 8u);
}

void write_bc7_mode7_probe_block(
    std::uint32_t part,
    std::uint8_t dst[16]) noexcept {
  static constexpr std::array<std::uint8_t, 16u> kAnchor2{
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
  std::uint64_t lo = 0ull;
  std::uint64_t hi = 0ull;
  std::uint32_t bp = 0u;
  auto push = [&](std::uint32_t val, std::uint32_t bits) noexcept {
    const std::uint64_t m = static_cast<std::uint64_t>(val) & ((1ull << bits) - 1ull);
    if (bp < 64u) {
      lo |= (m << bp);
      if (bp + bits > 64u) {
        hi |= (m >> (64u - bp));
      }
    } else {
      hi |= (m << (bp - 64u));
    }
    bp += bits;
  };
  push(0x80u, 8u);
  push(part, 6u);
  // 5-bit R0..R3, G0..G3, B0..B3, A0..A3
  const std::array<std::uint32_t, 16u> ep5{
      2u, 29u, 5u, 27u,   // R
      30u, 3u, 4u, 20u,   // G
      4u, 6u, 31u, 2u,    // B
      0u, 28u, 15u, 31u   // A
  };
  for (std::uint32_t v : ep5) {
    push(v, 5u);
  }
  push(0u, 1u); // P0
  push(1u, 1u); // P1
  push(1u, 1u); // P2
  push(0u, 1u); // P3
  const std::uint32_t anchor2 = kAnchor2[part];
  for (std::uint32_t i = 0u; i < 16u; ++i) {
    const std::uint32_t bits = (i == 0u || i == anchor2) ? 1u : 2u;
    const std::uint32_t idx =
        (bits == 1u) ? (i == 0u ? 1u : 0u) : (static_cast<std::uint32_t>(i * 3u + 1u) & 3u);
    push(idx, bits);
  }
  std::memcpy(dst, &lo, 8u);
  std::memcpy(dst + 8u, &hi, 8u);
}

void test_bc_encoder_decoder_quality() {
  std::cout << "[VT BC Test 1] CPU BC1/BC3/BC7 encoder + decoder quality...\n";
  constexpr std::uint32_t P = 64u;
  std::vector<std::uint8_t> zero_buf(vt_byte4_tile_bytes(P, LuisaVirtualTextureCompressionMode::bc7), 0u);
  vt_fill_zero_byte4_tile(P, LuisaVirtualTextureCompressionMode::bc7, zero_buf);
  for (std::size_t b = 0u; b < zero_buf.size(); b += 16u) {
    require(zero_buf[b] == 0x40u, "BC7 zero tile byte 0 must be valid Mode 6 (0x40)");
    std::uint8_t dec[64]{};
    vt_decode_bc7_block(zero_buf.data() + b, dec);
    for (std::uint8_t v : dec) {
      require(v == 0u, "BC7 Mode 6 zero block decoded non-zero");
    }
  }

  const auto rgba = make_synthetic_tile_rgba(P);
  const std::size_t bc7_bytes = vt_byte4_tile_bytes(P, LuisaVirtualTextureCompressionMode::bc7);
  const std::size_t bc3_bytes = vt_byte4_tile_bytes(P, LuisaVirtualTextureCompressionMode::bc3);
  const std::size_t bc1_bytes = vt_byte4_tile_bytes(P, LuisaVirtualTextureCompressionMode::bc1);
  require(bc7_bytes == 4096u && bc3_bytes == 4096u && bc1_bytes == 2048u,
          "Unexpected 64x64 BC tile byte sizes");

  std::vector<std::uint8_t> enc_bc7(bc7_bytes);
  std::vector<std::uint8_t> enc_bc3(bc3_bytes);
  std::vector<std::uint8_t> enc_bc1(bc1_bytes);
  vt_encode_bc7_tile(rgba.data(), P, enc_bc7);
  vt_encode_bc3_tile(rgba.data(), P, enc_bc3);
  vt_encode_bc1_tile(rgba.data(), P, enc_bc1);

  auto untile_decode = [&](const std::vector<std::uint8_t> &enc,
                           LuisaVirtualTextureCompressionMode mode) {
    std::vector<std::uint8_t> out(rgba.size(), 0u);
    const std::uint32_t bpa = P / 4u;
    const std::size_t bsz = (mode == LuisaVirtualTextureCompressionMode::bc1) ? 8u : 16u;
    std::uint8_t blk[64]{};
    for (std::uint32_t by = 0u; by < bpa; ++by) {
      for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
        const std::uint8_t *src_b = enc.data() + (by * bpa + bx) * bsz;
        if (mode == LuisaVirtualTextureCompressionMode::bc7) {
          vt_decode_bc7_block(src_b, blk);
        } else if (mode == LuisaVirtualTextureCompressionMode::bc3) {
          vt_decode_bc3_block(src_b, blk);
        } else {
          vt_decode_bc1_block(src_b, blk);
        }
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          const std::size_t dst_off =
              (static_cast<std::size_t>(by * 4u + r) * P + bx * 4u) * 4u;
          std::memcpy(out.data() + dst_off, blk + r * 16u, 16u);
        }
      }
    }
    return out;
  };

  const auto dec7 = untile_decode(enc_bc7, LuisaVirtualTextureCompressionMode::bc7);
  const auto dec3 = untile_decode(enc_bc3, LuisaVirtualTextureCompressionMode::bc3);
  const auto dec1 = untile_decode(enc_bc1, LuisaVirtualTextureCompressionMode::bc1);

  // Verify mixed-LSB solid block at (bx=1, by=0).
  int solid_max_err = 0;
  for (std::uint32_t y = 0u; y < 4u; ++y) {
    for (std::uint32_t x = 4u; x < 8u; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * P + x) * 4u;
      for (int c = 0; c < 4; ++c) {
        solid_max_err = std::max(solid_max_err,
                                 std::abs(static_cast<int>(dec7[off + c]) -
                                          static_cast<int>(rgba[off + c])));
      }
    }
  }
  require(solid_max_err <= 1, "BC7 mixed-LSB solid max_err > 1");

  const double psnr7 = compute_psnr(rgba.data(), dec7.data(), rgba.size(), false);
  const double psnr3 = compute_psnr(rgba.data(), dec3.data(), rgba.size(), false);
  const std::size_t opaque_half_off = (P / 2u) * P * 4u;
  const double psnr1_rgb = compute_psnr(
      rgba.data() + opaque_half_off, dec1.data() + opaque_half_off,
      rgba.size() - opaque_half_off, true);
  std::cout << "  BC7 PSNR=" << psnr7 << " dB, BC3 PSNR=" << psnr3
            << " dB, BC1 opaque RGB PSNR=" << psnr1_rgb
            << " dB (solid_max_err=" << solid_max_err << ")\n";
  require(psnr7 >= 38.0, "BC7 PSNR below 38 dB threshold");
  require(psnr3 >= 30.0, "BC3 PSNR below 30 dB threshold");
  require(psnr1_rgb >= 30.0, "BC1 RGB PSNR below 30 dB threshold");
}

void test_bc_gpu_hw_decode_parity(Device &device, Stream &stream) {
  std::cout << "[VT BC Test 2] Vulkan hardware BC7/BC3/BC1 decode parity (Modes 1, 5, 6, 7)...\n";
  if (device.impl()->query("vk_batch_copy_cmd") != "1") {
    std::cout << "  SKIPPED: backend does not support vk_batch_copy_cmd\n";
    return;
  }

  for (const std::uint32_t P : {64u, 128u}) {
    const auto rgba = make_synthetic_tile_rgba(P);
    const std::uint32_t bpa = P / 4u;

    for (const auto mode : {LuisaVirtualTextureCompressionMode::bc7,
                            LuisaVirtualTextureCompressionMode::bc3,
                            LuisaVirtualTextureCompressionMode::bc1}) {
      const std::uint32_t tile_bytes = vt_byte4_tile_bytes(P, mode);
      const std::size_t bsz =
          (mode == LuisaVirtualTextureCompressionMode::bc1) ? 8u : 16u;
      std::vector<std::uint8_t> enc(tile_bytes);
      if (mode == LuisaVirtualTextureCompressionMode::bc7) {
        vt_encode_bc7_tile(rgba.data(), P, enc);
        if (P == 64u) {
          // Overwrite blocks 0..15 with Mode 1 partitions 0..15, blocks 16..19 with
          // Mode 5 rot=0..3, and blocks 20..35 with Mode 7 partitions 0..15 to test
          // Vulkan HW spec bit-for-bit.
          for (std::uint32_t p = 0u; p < 16u; ++p) {
            write_bc7_mode1_probe_block(p, enc.data() + p * 16u);
          }
          for (std::uint32_t rot = 0u; rot < 4u; ++rot) {
            write_bc7_mode5_probe_block(rot, enc.data() + (16u + rot) * 16u);
          }
          for (std::uint32_t p = 0u; p < 16u; ++p) {
            write_bc7_mode7_probe_block(p, enc.data() + (20u + p) * 16u);
          }
        }
      } else if (mode == LuisaVirtualTextureCompressionMode::bc3) {
        vt_encode_bc3_tile(rgba.data(), P, enc);
      } else {
        vt_encode_bc1_tile(rgba.data(), P, enc);
      }

      // Compute CPU reference decode per texel.
      std::vector<std::uint8_t> cpu_ref(static_cast<std::size_t>(P) * P * 4u, 0u);
      std::uint8_t blk[64]{};
      for (std::uint32_t by = 0u; by < bpa; ++by) {
        for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
          const std::uint8_t *src_b = enc.data() + (by * bpa + bx) * bsz;
          if (mode == LuisaVirtualTextureCompressionMode::bc7) {
            vt_decode_bc7_block(src_b, blk);
          } else if (mode == LuisaVirtualTextureCompressionMode::bc3) {
            vt_decode_bc3_block(src_b, blk);
          } else {
            vt_decode_bc1_block(src_b, blk);
          }
          for (std::uint32_t r = 0u; r < 4u; ++r) {
            const std::size_t dst_off =
                (static_cast<std::size_t>(by * 4u + r) * P + bx * 4u) * 4u;
            std::memcpy(cpu_ref.data() + dst_off, blk + r * 16u, 16u);
          }
        }
      }

      const PixelStorage storage = vt_byte4_pixel_storage(mode);
      auto tex = device.create_image<float>(storage, P, P);
      auto heap = device.create_bindless_array(2u);
      heap.emplace_on_update(0u, tex, Sampler::point_edge());

      const std::uint32_t u32_count = tile_bytes / 4u;
      auto staging = device.create_buffer<luisa::uint>(u32_count);
      auto out_buf = device.create_buffer<luisa::float4>(P * P);
      std::vector<luisa::uint> enc_u32(u32_count);
      std::memcpy(enc_u32.data(), enc.data(), tile_bytes);

      auto sample_kernel = device.compile<1>(
          [P](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
            const auto tid = dispatch_x();
            $if(tid >= P * P) { $return(); };
            const auto x = tid % P;
            const auto y = tid / P;
            const float inv = 1.0f / static_cast<float>(P);
            const auto u = (cast<float>(x) + 0.5f) * inv;
            const auto v = (cast<float>(y) + 0.5f) * inv;
            dst.write(tid, h.tex2d(0u).sample(make_float2(u, v)));
          });

      luisa::vector<BufferToTextureCopyRegion> regions{
          BufferToTextureCopyRegion{
              .buffer_offset = 0ull,
              .buffer_row_length = 0u,
              .buffer_image_height = 0u,
              .mip_level = 0u,
              .texture_offset = luisa::make_uint3(0u, 0u, 0u),
              .texture_extent = luisa::make_uint3(P, P, 1u)}};

      CommandList cmd;
      cmd << heap.update()
          << staging.copy_from(luisa::span{enc_u32})
          << luisa::make_unique<BatchBufferToTextureCopyCommand>(
                 staging.handle(), 0ull, tile_bytes, tex.handle(), storage,
                 std::move(regions))
          << sample_kernel(heap, out_buf).dispatch(P * P);
      std::vector<luisa::float4> gpu_px(P * P);
      stream << cmd.commit() << out_buf.copy_to(luisa::span{gpu_px})
             << synchronize();

      double max_diff = 0.0;
      std::size_t mismatches = 0u;
      for (std::uint32_t i = 0u; i < P * P; ++i) {
        for (int ch = 0; ch < 4; ++ch) {
          const float g = (&gpu_px[i].x)[ch];
          require(std::isfinite(g), "Non-finite GPU BC sample");
          const double ref_f =
              static_cast<double>(cpu_ref[static_cast<std::size_t>(i) * 4u + ch]) /
              255.0;
          const double diff = std::abs(static_cast<double>(g) - ref_f);
          max_diff = std::max(max_diff, diff);
          if (diff > 1.5 / 255.0) {
            if (mismatches < 16u) {
              const std::uint32_t x = i % P;
              const std::uint32_t y = i / P;
              std::cerr << "  HW PARITY MISMATCH mode="
                        << luisa_virtual_texture_compression_mode_name(mode)
                        << " P=" << P << " bx=" << (x / 4u) << " by=" << (y / 4u)
                        << " lx=" << (x % 4u) << " ly=" << (y % 4u) << " ch=" << ch
                        << " got=" << (g * 255.0) << " ref=" << (ref_f * 255.0)
                        << '\n';
            }
            ++mismatches;
          }
        }
      }
      std::cout << "  mode=" << luisa_virtual_texture_compression_mode_name(mode)
                << " P=" << P << " max_diff_lsb=" << (max_diff * 255.0) << '\n';
      require(mismatches == 0u,
              "Vulkan hardware BC decode parity failed for " +
                  std::string{luisa_virtual_texture_compression_mode_name(mode)});
    }
  }
}

void test_vt_cache_bc_end_to_end(Device &device, Stream &stream) {
  std::cout << "[VT BC Test 3] VirtualTextureCache BC end-to-end (NPOT + FLOAT4 + prepack/disk + async eviction)...\n";
  if (device.impl()->query("vk_batch_copy_cmd") != "1") {
    std::cout << "  SKIPPED: backend does not support vk_batch_copy_cmd\n";
    return;
  }

  constexpr std::uint32_t w = 127u;
  constexpr std::uint32_t h = 73u;
  std::vector<std::uint8_t> b4_rgba(static_cast<std::size_t>(w) * h * 4u);
  auto tri_wave_npot = [](std::uint32_t val) noexcept -> std::uint8_t {
    const std::uint32_t m = val & 511u;
    return static_cast<std::uint8_t>(m < 256u ? m : (511u - m));
  };
  for (std::uint32_t y = 0u; y < h; ++y) {
    for (std::uint32_t x = 0u; x < w; ++x) {
      const std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4u;
      b4_rgba[idx + 0u] = tri_wave_npot(x * 5u + y * 2u + 20u);
      b4_rgba[idx + 1u] = tri_wave_npot(x * 2u + y * 6u + 40u);
      b4_rgba[idx + 2u] = tri_wave_npot(x * 3u + y * 4u + 80u);
      b4_rgba[idx + 3u] = static_cast<std::uint8_t>(200u + ((x + y) % 55u));
    }
  }
  const auto png_bytes = encode_png_rgba8(w, h, b4_rgba);
  const auto png_probe = probe_scene_image_info(png_bytes, "bc_npot.png");
  require(png_probe.has_value(), "NPOT PNG probe failed");

  constexpr std::uint32_t fw = 61u;
  constexpr std::uint32_t fh = 59u;
  std::vector<float> f4_pix(static_cast<std::size_t>(fw) * fh * 4u);
  for (std::uint32_t y = 0u; y < fh; ++y) {
    for (std::uint32_t x = 0u; x < fw; ++x) {
      const std::size_t idx = (static_cast<std::size_t>(y) * fw + x) * 4u;
      f4_pix[idx + 0u] = 0.05f * static_cast<float>(x) + 1.125f;
      f4_pix[idx + 1u] = 0.03f * static_cast<float>(y);
      f4_pix[idx + 2u] = 0.01f * static_cast<float>((x * y) % 97u);
      f4_pix[idx + 3u] = 0.75f;
    }
  }
  const auto exr_bytes = encode_exr_rgba32f(fw, fh, f4_pix);
  const auto exr_probe = probe_scene_image_info(exr_bytes, "bc_hdr.exr");
  require(exr_probe.has_value() && exr_probe->storage == io::DecodedImageStorage::float32,
          "EXR float32 probe failed");

  const std::vector<VtProbedSceneImage> probed{
      VtProbedSceneImage{.id = ImageId{1u},
                         .name = "bc_npot.png",
                         .encoded_data = png_bytes,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *png_probe},
      VtProbedSceneImage{.id = ImageId{2u},
                         .name = "bc_hdr.exr",
                         .encoded_data = exr_bytes,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *exr_probe}};

  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  constexpr std::array exts{ImageExtension::repeat, ImageExtension::clip,
                            ImageExtension::extend, ImageExtension::mirror};
  constexpr std::array interps{ImageInterpolation::closest,
                               ImageInterpolation::linear,
                               ImageInterpolation::cubic};
  std::vector<ImageBinding> handles;
  std::vector<CyclesSvmImageBindingGpu> gpu_bindings;
  for (const auto ext : exts) {
    for (const auto interp : interps) {
      const auto idx = static_cast<std::uint32_t>(handles.size());
      handles.push_back(ImageBinding{.resource_id = 1u,
                                     .interpolation = interp,
                                     .extension = ext});
      gpu_bindings.push_back(make_cycles_svm_image_binding(idx, interp, ext));
    }
  }
  // Binding 12 = FLOAT4 texture.
  handles.push_back(ImageBinding{.resource_id = 2u,
                                 .interpolation = ImageInterpolation::linear,
                                 .extension = ImageExtension::repeat});
  gpu_bindings.push_back(make_cycles_svm_image_binding(
      12u, ImageInterpolation::linear, ImageExtension::repeat));

  std::vector<luisa::float2> uvs;
  for (std::uint32_t y : {0u, 1u, 35u, 59u, 60u, 61u, 72u}) {
    for (std::uint32_t x : {0u, 1u, 59u, 60u, 61u, 120u, 123u, 124u, 126u}) {
      uvs.push_back(luisa::make_float2((static_cast<float>(x) + 0.5f) / w,
                                       (static_cast<float>(y) + 0.5f) / h));
    }
  }
  for (const float u : {-0.25f, 0.0f, 0.47f, 0.99f, 1.35f}) {
    for (const float v : {-0.25f, 0.0f, 0.82f, 1.0f, 1.25f}) {
      uvs.push_back(luisa::make_float2(u, v));
    }
  }
  const std::uint32_t u_count = static_cast<std::uint32_t>(uvs.size());
  const std::uint32_t b_count = static_cast<std::uint32_t>(handles.size());
  const std::uint32_t total = u_count * b_count;

  BindlessArray dummy_direct_heap = device.create_bindless_array(16u);
  auto uv_buf = device.create_buffer<luisa::float2>(u_count);
  auto out_buf = device.create_buffer<luisa::float4>(total);
  auto miss_count = device.create_buffer<luisa::uint>(1u);
  stream << uv_buf.copy_from(luisa::span{uvs}) << synchronize();

  struct TempDirGuard {
    std::filesystem::path dir;
    explicit TempDirGuard(std::filesystem::path d) : dir{std::move(d)} {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
    ~TempDirGuard() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  };
  const TempDirGuard disk_guard{
      std::filesystem::temp_directory_path() / "psycles_vt_bc_disk_test"};

  auto run_sync_matrix = [&](const LuisaVirtualTextureCacheOptions &opts) {
    auto vt = VirtualTextureCache::create(device, stream, opts, probed, handles,
                                          gpu_bindings);
    auto k = device.compile<1>(
        [&](BufferVar<luisa::float2> in_uvs, BufferVar<luisa::float4> out,
            BufferVar<luisa::uint> misses, UInt nuv) noexcept {
          const auto uid = dispatch_x();
          $if(uid >= nuv) { $return(); };
          const auto uv = in_uvs.read(uid);
          for (std::uint32_t b = 0u; b < 13u; ++b) {
            UInt miss = vt_invalid_page;
            const auto col =
                vt->sample_2d_const(static_cast<std::int32_t>(b), uv, &miss);
            $if(miss != vt_invalid_page) { misses.atomic(0u).fetch_add(1u); };
            out.write(b * nuv + uid, col);
          }
        });
    std::uint32_t zero = 0u;
    std::uint32_t hm = 0u;
    bool converged = false;
    for (int it = 0; it < 16; ++it) {
      stream << miss_count.copy_from(&zero)
             << k(uv_buf, out_buf, miss_count, u_count).dispatch(u_count)
             << miss_count.copy_to(&hm) << synchronize();
      if (hm == 0u) {
        converged = true;
        break;
      }
      vt->service_bitset_misses(stream);
    }
    require(converged, "VT BC matrix did not converge");
    std::vector<luisa::float4> px(total);
    stream << out_buf.copy_to(luisa::span{px}) << synchronize();
    return std::pair{vt->stats(), std::move(px)};
  };

  for (const std::uint32_t P : {64u, 128u}) {
    const LuisaVirtualTextureCacheOptions base_opts{
        .enabled = true,
        .gpu_cache_budget_bytes = 32ull * 1024ull * 1024ull,
        .physical_tile_size = P,
        .prepack_mode = LuisaVirtualTexturePrepackMode::disk,
        .prepack_disk_dir = disk_guard.dir.string()};
    const auto [ref_stats, ref_px] = run_sync_matrix(base_opts);

    for (const auto mode : {LuisaVirtualTextureCompressionMode::bc7,
                            LuisaVirtualTextureCompressionMode::bc3,
                            LuisaVirtualTextureCompressionMode::bc1}) {
      auto opts = base_opts;
      opts.compression_mode = mode;
      const auto [cold_stats, cold_px] = run_sync_matrix(opts);
      const auto [warm_stats, warm_px] = run_sync_matrix(opts);
      require(warm_stats.prepack_disk_reads > 0u &&
                  warm_stats.images_decoded == 0u,
              "Expected disk warm reads with 0 decodes for BC");
      require(std::memcmp(cold_px.data(), warm_px.data(),
                          cold_px.size() * sizeof(luisa::float4)) == 0,
              "Disk warm BC pixels differ from cold BC pixels");

      double b4_sum_sq = 0.0;
      std::size_t b4_cnt = 0u;
      double f4_max_abs = 0.0;
      const int ch_end = (mode == LuisaVirtualTextureCompressionMode::bc1) ? 3 : 4;
      for (std::uint32_t b = 0u; b < 12u; ++b) {
        for (std::uint32_t u = 0u; u < u_count; ++u) {
          const std::size_t i = static_cast<std::size_t>(b) * u_count + u;
          for (int c = 0; c < 4; ++c) {
            const float v = (&cold_px[i].x)[c];
            require(std::isfinite(v), "Non-finite BC VT sample");
            if (c < ch_end) {
              const double d = static_cast<double>(v) -
                               static_cast<double>((&ref_px[i].x)[c]);
              b4_sum_sq += d * d;
              ++b4_cnt;
            }
          }
        }
      }
      for (std::uint32_t u = 0u; u < u_count; ++u) {
        const std::size_t i = 12u * u_count + u;
        for (int c = 0; c < 4; ++c) {
          const float v = (&cold_px[i].x)[c];
          require(std::isfinite(v), "Non-finite FLOAT4 VT sample under BC");
          f4_max_abs = std::max(
              f4_max_abs, std::abs(static_cast<double>(v) -
                                   static_cast<double>((&ref_px[i].x)[c])));
        }
      }
      const double b4_rmse = std::sqrt(b4_sum_sq / static_cast<double>(b4_cnt));
      std::cout << "  P=" << P
                << " mode=" << luisa_virtual_texture_compression_mode_name(mode)
                << " b4_rmse=" << b4_rmse << " f4_max_abs=" << f4_max_abs
                << " upload_bytes=" << cold_stats.gpu_upload_bytes
                << " atlas_bytes=" << cold_stats.gpu_allocated_atlas_bytes << '\n';
      require(f4_max_abs <= 1e-4, "FLOAT4 should remain lossless under BC");
      if (mode == LuisaVirtualTextureCompressionMode::bc7) {
        require(b4_rmse < 0.015, "BC7 VT RMSE >= 0.015");
      } else {
        require(b4_rmse < 0.035, "BC3/BC1 VT RGB RMSE >= 0.035");
      }
      require(cold_stats.gpu_upload_bytes < ref_stats.gpu_upload_bytes,
              "Expected compressed upload bytes < uncompressed");
      require(cold_stats.gpu_allocated_atlas_bytes <
                  ref_stats.gpu_allocated_atlas_bytes,
              "Expected compressed atlas bytes < uncompressed");
    }
  }

  // Tight-budget WavefrontCoroScheduler async eviction + Copy Queue parity check.
  {
    std::vector<VtProbedSceneImage> big_probed;
    std::vector<ImageBinding> big_handles;
    std::vector<CyclesSvmImageBindingGpu> big_bindings;
    constexpr std::uint32_t bw = 256u;
    constexpr std::uint32_t bh = 256u;
    for (std::uint32_t img = 0u; img < 8u; ++img) {
      std::vector<std::uint8_t> pix(static_cast<std::size_t>(bw) * bh * 4u);
      for (std::uint32_t y = 0u; y < bh; ++y) {
        for (std::uint32_t x = 0u; x < bw; ++x) {
          const std::size_t i = (static_cast<std::size_t>(y) * bw + x) * 4u;
          pix[i + 0u] = static_cast<std::uint8_t>((x * 3u + img * 29u) & 0xffu);
          pix[i + 1u] = static_cast<std::uint8_t>((y * 5u + img * 17u) & 0xffu);
          pix[i + 2u] = static_cast<std::uint8_t>(((x + y) * 2u + img * 41u) & 0xffu);
          pix[i + 3u] = 255u;
        }
      }
      const auto pbytes = encode_png_rgba8(bw, bh, pix);
      const auto pr = probe_scene_image_info(pbytes, "big.png");
      require(pr.has_value(), "big png probe failed");
      big_probed.push_back(VtProbedSceneImage{
          .id = ImageId{100u + img},
          .name = "big_" + std::to_string(img) + ".png",
          .encoded_data = pbytes,
          .alpha_type = ImageAlphaType::straight,
          .color_space = ImageColorSpace::data,
          .probed = *pr});
      big_handles.push_back(ImageBinding{.resource_id = 100u + img,
                                         .interpolation = ImageInterpolation::linear,
                                         .extension = ImageExtension::repeat});
      big_bindings.push_back(make_cycles_svm_image_binding(
          img, ImageInterpolation::linear, ImageExtension::repeat));
    }

    auto run_coro_evict = [&](LuisaVirtualTextureCompressionMode mode,
                              bool async_worker,
                              LuisaVirtualTextureCopyQueueMode cq_mode) {
      const LuisaVirtualTextureCacheOptions opts{
          .enabled = true,
          .gpu_cache_budget_bytes = 256ull * 1024ull,
          .physical_tile_size = 64u,
          .prepack_mode = LuisaVirtualTexturePrepackMode::memory,
          .host_prepacked_cache_budget_bytes = 16ull * 1024ull * 1024ull,
          .prepack_warm_on_init = true,
          .enable_async_worker = async_worker,
          .copy_queue_mode = cq_mode,
          .compression_mode = mode};
      auto vt = VirtualTextureCache::create(device, stream, opts, big_probed,
                                            big_handles, big_bindings);
      auto vt_wait_resident = [&](Expr<luisa::uint> page_id) noexcept {
        UInt missed_page = page_id;
        $loop {
          $suspend("vt_wait",
                   coro_stage(luisa::string{vt_extension_schema})
                       .read(luisa::string{vt_extension_binding}, missed_page)
                       .blocking_async()
                       .async_domain(luisa::string{vt_extension_schema}));
          $if(vt->is_resident(missed_page)) { $break; };
        };
      };
      constexpr std::uint32_t kThreads = 2048u;
      Coroutine<void(Buffer<luisa::float4>)> coro{
          [&](BufferFloat4 out_buf) {
            const auto tid = dispatch_x();
            const UInt b = tid & 7u;
            const UInt tile_x = (tid >> 3u) % 5u;
            const UInt tile_y = (tid >> 6u) % 5u;
            const Float u = (cast<float>(tile_x * 60u + 15u) + 0.5f) / 256.0f;
            const Float v = (cast<float>(tile_y * 60u + 15u) + 0.5f) / 256.0f;
            Float4 col = make_float4(0.0f);
            $loop {
              UInt miss = vt_invalid_page;
              col = vt->sample_2d_dynamic(Int(b), make_float2(u, v),
                                          dummy_direct_heap, &miss);
              $if(miss == vt_invalid_page) { $break; };
              vt_wait_resident(miss);
            };
            out_buf.write(tid, col);
          }};
      WavefrontCoroSchedulerConfig cfg;
      cfg.thread_count = 1024u;
      cfg.execution_block_size = 64u;
      cfg.global_memory_soa = true;
      cfg.frame_buffer_compaction = true;
      cfg.largest_continuation_first = true;
      cfg.incremental_continuation_counts = true;
      cfg.fused_continuation_counts = true;
      WavefrontCoroScheduler<Buffer<luisa::float4>> sched{device, coro, cfg};
      sched.register_extension_handler(stream,
                                       vt->make_wavefront_extension_handler());
      auto out_buf = device.create_buffer<luisa::float4>(kThreads);
      stream << sched(out_buf).dispatch(kThreads);
      std::vector<luisa::float4> host_out(kThreads);
      stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();
      return std::pair{vt->stats(), std::move(host_out)};
    };

    const auto [s_off, p_off] = run_coro_evict(
        LuisaVirtualTextureCompressionMode::off, false,
        LuisaVirtualTextureCopyQueueMode::off);
    const auto [s_bc7_aw, p_bc7_aw] = run_coro_evict(
        LuisaVirtualTextureCompressionMode::bc7, true,
        LuisaVirtualTextureCopyQueueMode::off);
    const auto [s_bc7_cq, p_bc7_cq] = run_coro_evict(
        LuisaVirtualTextureCompressionMode::bc7, true,
        LuisaVirtualTextureCopyQueueMode::staged_shader);
    require(s_bc7_aw.byte4_cache_capacity >= s_off.byte4_cache_capacity * 4u,
            "Expected >= 4x cache capacity under BC7");
    require(s_bc7_aw.tiles_evicted > 0ull && s_bc7_aw.async_batches > 0ull,
            "Expected BC7 async eviction batches > 0");
    require(s_bc7_cq.copy_queue_batches > 0ull,
            "Expected BC7 Copy Queue batches > 0");
    double sum_sq = 0.0;
    for (std::size_t i = 0u; i < p_off.size(); ++i) {
      for (int c = 0; c < 4; ++c) {
        const float a = (&p_bc7_aw[i].x)[c];
        const float b = (&p_bc7_cq[i].x)[c];
        require(std::isfinite(a) && std::isfinite(b),
                "Non-finite async BC7 eviction output");
        require(std::abs(a - b) <= 1e-4f,
                "BC7 cq=off vs cq=staged_shader async mismatch");
        const double d =
            static_cast<double>(a) - static_cast<double>((&p_off[i].x)[c]);
        sum_sq += d * d;
      }
    }
    const double rmse =
        std::sqrt(sum_sq / static_cast<double>(p_off.size() * 4u));
    std::cout << "  async eviction: off_cap=" << s_off.byte4_cache_capacity
              << " bc7_cap=" << s_bc7_aw.byte4_cache_capacity
              << " evictions=" << s_bc7_aw.tiles_evicted << " rmse=" << rmse
              << '\n';
    require(rmse < 0.015, "Async BC7 eviction RMSE >= 0.015");
  }
}

void test_multi_format_hw_decode_parity(Device &device, Stream &stream) {
  std::cout << "[VT BC Test 5] Multi-format GPU HW decode + swizzle parity (BC4/BC5/BC6H/SHORT/HALF/FLOAT1)...\n";
  if (device.impl()->query("vk_batch_copy_cmd") != "1") {
    std::cout << "  SKIPPED: backend does not support vk_batch_copy_cmd\n";
    return;
  }

  constexpr std::uint32_t P = 64u;
  constexpr std::uint32_t T = P - 2u * vt_border_size;
  constexpr std::uint32_t texels = P * P;
  constexpr std::uint32_t bpa = P / 4u;

  auto check_gpu_tile = [&](VtTileFormat fmt,
                            const std::vector<std::uint8_t> &tile_bytes,
                            const std::vector<luisa::float4> &cpu_ref,
                            double max_abs_tol, double max_rel_tol) {
    const std::uint32_t expected_bytes = vt_tile_format_bytes(P, fmt);
    require(tile_bytes.size() == expected_bytes, "Tile byte size mismatch");
    const PixelStorage storage = vt_tile_format_storage(fmt);
    const std::uint32_t swizzle_u32 =
        static_cast<std::uint32_t>(vt_tile_format_swizzle(fmt));

    auto tex = device.create_image<float>(storage, P, P);
    auto heap = device.create_bindless_array(2u);
    heap.emplace_on_update(0u, tex, Sampler::point_edge());

    const std::uint32_t u32_count = (expected_bytes + 3u) / 4u;
    auto staging = device.create_buffer<luisa::uint>(u32_count);
    auto out_buf = device.create_buffer<luisa::float4>(texels);
    std::vector<luisa::uint> enc_u32(u32_count, 0u);
    std::memcpy(enc_u32.data(), tile_bytes.data(), expected_bytes);

    auto sample_kernel = device.compile<1>(
        [P, swizzle_u32](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
          const auto tid = dispatch_x();
          $if(tid >= P * P) { $return(); };
          const auto x = tid % P;
          const auto y = tid / P;
          const float inv = 1.0f / static_cast<float>(P);
          const auto u = (cast<float>(x) + 0.5f) * inv;
          const auto v = (cast<float>(y) + 0.5f) * inv;
          Float4 c = h.tex2d(0u).sample(make_float2(u, v));
          if (swizzle_u32 == static_cast<std::uint32_t>(VtChannelSwizzle::rrr1)) {
            c = make_float4(c.x, c.x, c.x, 1.0f);
          } else if (swizzle_u32 ==
                     static_cast<std::uint32_t>(VtChannelSwizzle::rrrg)) {
            c = make_float4(c.x, c.x, c.x, c.y);
          }
          dst.write(tid, c);
        });

    luisa::vector<BufferToTextureCopyRegion> regions{
        BufferToTextureCopyRegion{
            .buffer_offset = 0ull,
            .buffer_row_length = 0u,
            .buffer_image_height = 0u,
            .mip_level = 0u,
            .texture_offset = luisa::make_uint3(0u, 0u, 0u),
            .texture_extent = luisa::make_uint3(P, P, 1u)}};
    CommandList cmd;
    cmd << heap.update()
        << staging.copy_from(luisa::span{enc_u32})
        << luisa::make_unique<BatchBufferToTextureCopyCommand>(
               staging.handle(), 0ull, expected_bytes, tex.handle(), storage,
               std::move(regions))
        << sample_kernel(heap, out_buf).dispatch(texels);
    std::vector<luisa::float4> gpu_px(texels);
    stream << cmd.commit() << out_buf.copy_to(luisa::span{gpu_px})
           << synchronize();

    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t mismatches = 0u;
    for (std::uint32_t i = 0u; i < texels; ++i) {
      for (int ch = 0; ch < 4; ++ch) {
        const float g = (&gpu_px[i].x)[ch];
        const float r = (&cpu_ref[i].x)[ch];
        require(std::isfinite(g), "Non-finite multi-format GPU sample");
        const double abs_d = std::abs(static_cast<double>(g) - r);
        const double rel_d = abs_d / std::max(std::abs(static_cast<double>(r)), 1e-3);
        max_abs = std::max(max_abs, abs_d);
        max_rel = std::max(max_rel, rel_d);
        const bool bad = (max_rel_tol > 0.0)
                             ? (abs_d > max_abs_tol && rel_d > max_rel_tol)
                             : (abs_d > max_abs_tol);
        if (bad) {
          if (mismatches < 8u) {
            std::cerr << "  MISMATCH fmt=" << vt_tile_format_name(fmt)
                      << " i=" << i << " ch=" << ch << " got=" << g
                      << " ref=" << r << '\n';
          }
          ++mismatches;
        }
      }
    }
    std::cout << "  fmt=" << vt_tile_format_name(fmt)
              << " max_abs=" << max_abs << " max_rel=" << max_rel << '\n';
    require(mismatches == 0u,
            "Multi-format HW decode parity failed for " +
                std::string{vt_tile_format_name(fmt)});
  };

  const auto rgba_u8 = make_synthetic_tile_rgba(P);

  // BC4: single channel R -> (R, R, R, 1)
  {
    std::vector<std::uint8_t> enc(vt_tile_format_bytes(P, VtTileFormat::bc4));
    vt_encode_bc4_tile(rgba_u8.data(), P, enc);
    std::vector<luisa::float4> cpu_ref(texels);
    std::uint8_t blk[64]{};
    for (std::uint32_t by = 0u; by < bpa; ++by) {
      for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
        vt_decode_bc4_block(enc.data() + (by * bpa + bx) * 8u, blk);
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::size_t dst_i =
                static_cast<std::size_t>(by * 4u + r) * P + bx * 4u + c;
            const std::size_t b_i = (r * 4u + c) * 4u;
            const float v = static_cast<float>(blk[b_i]) / 255.0f;
            cpu_ref[dst_i] = luisa::make_float4(v, v, v, 1.0f);
          }
        }
      }
    }
    check_gpu_tile(VtTileFormat::bc4, enc, cpu_ref, 1.5 / 255.0, 0.0);
  }

  // BC5: (R, A) packed into RG -> rrrg swizzle -> (R, R, R, A)
  {
    std::vector<std::uint8_t> enc(vt_tile_format_bytes(P, VtTileFormat::bc5));
    vt_encode_bc5_tile(rgba_u8.data(), P, enc);
    std::vector<luisa::float4> cpu_ref(texels);
    std::uint8_t blk[64]{};
    for (std::uint32_t by = 0u; by < bpa; ++by) {
      for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
        vt_decode_bc5_block(enc.data() + (by * bpa + bx) * 16u, blk);
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::size_t dst_i =
                static_cast<std::size_t>(by * 4u + r) * P + bx * 4u + c;
            const std::size_t b_i = (r * 4u + c) * 4u;
            const float rv = static_cast<float>(blk[b_i + 0u]) / 255.0f;
            const float av = static_cast<float>(blk[b_i + 1u]) / 255.0f;
            cpu_ref[dst_i] = luisa::make_float4(rv, rv, rv, av);
          }
        }
      }
    }
    check_gpu_tile(VtTileFormat::bc5, enc, cpu_ref, 1.5 / 255.0, 0.0);
  }

  // Float source tiles (positive HDR for BC6H, signed [-2.5, +12.5] for HALF/FLOAT, [0, 1] for SHORT)
  std::vector<float> hdr_f32(static_cast<std::size_t>(P) * P * 4u);
  std::vector<float> signed_f32(static_cast<std::size_t>(P) * P * 4u);
  std::vector<float> unit_f32(static_cast<std::size_t>(P) * P * 4u);
  for (std::uint32_t y = 0u; y < P; ++y) {
    for (std::uint32_t x = 0u; x < P; ++x) {
      const std::size_t o = (static_cast<std::size_t>(y) * P + x) * 4u;
      const std::uint32_t bx = x / 4u;
      const std::uint32_t by = y / 4u;
      const float u = static_cast<float>(x % 4u) * 0.2f +
                      static_cast<float>(y % 4u) * 0.13f;
      hdr_f32[o + 0u] = 0.05f + 0.55f * static_cast<float>(bx) + (0.2f + 0.1f * by) * u;
      hdr_f32[o + 1u] = 0.03f + 0.35f * static_cast<float>(by) + (0.15f + 0.05f * bx) * u;
      hdr_f32[o + 2u] = 0.02f + 0.25f * static_cast<float>((bx + by) & 7u) + 0.12f * u;
      hdr_f32[o + 3u] = 1.0f;

      signed_f32[o + 0u] = -2.5f + 0.25f * static_cast<float>(x);
      signed_f32[o + 1u] = 1.5f - 0.06f * static_cast<float>(y);
      signed_f32[o + 2u] = 0.125f * static_cast<float>((x + y) % 17u);
      signed_f32[o + 3u] = 0.75f;

      unit_f32[o + 0u] = static_cast<float>(x) / static_cast<float>(P - 1u);
      unit_f32[o + 1u] = static_cast<float>(y) / static_cast<float>(P - 1u);
      unit_f32[o + 2u] =
          static_cast<float>((x + y) % 64u) / 63.0f;
      unit_f32[o + 3u] = 0.625f;
    }
  }

  // BC6H Mode 11: Vulkan HW decode parity vs vt_decode_bc6h_block
  {
    std::vector<std::uint8_t> enc(vt_tile_format_bytes(P, VtTileFormat::bc6h));
    vt_encode_bc6h_tile(hdr_f32.data(), P, enc);
    std::vector<luisa::float4> cpu_ref(texels);
    float blk[64]{};
    double hdr_sum_rel_sq = 0.0;
    for (std::uint32_t by = 0u; by < bpa; ++by) {
      for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
        vt_decode_bc6h_block(enc.data() + (by * bpa + bx) * 16u, blk);
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::size_t dst_i =
                static_cast<std::size_t>(by * 4u + r) * P + bx * 4u + c;
            const std::size_t b_i = (r * 4u + c) * 4u;
            cpu_ref[dst_i] = luisa::make_float4(blk[b_i + 0u], blk[b_i + 1u],
                                                blk[b_i + 2u], 1.0f);
            for (int ch = 0; ch < 3; ++ch) {
              const float ref_v = std::max(hdr_f32[dst_i * 4u + ch], 1e-2f);
              const double rel = (blk[b_i + ch] - hdr_f32[dst_i * 4u + ch]) / ref_v;
              hdr_sum_rel_sq += rel * rel;
            }
          }
        }
      }
    }
    check_gpu_tile(VtTileFormat::bc6h, enc, cpu_ref, 5e-3, 5e-3);
    const double rel_rmse =
        std::sqrt(hdr_sum_rel_sq / static_cast<double>(texels * 3u));
    std::cout << "  bc6h hdr rel_rmse=" << rel_rmse << '\n';
    require(rel_rmse < 0.03, "BC6H HDR relative RMSE >= 0.03");
  }

  // Pack full-tile helper using vt_pack_tile_formatted at (P x P) interior without wrapping artifacts:
  // Use image width=T, height=T starting at tile (0,0) with border=2 (or width=T extend) and compare CPU decode!
  // Better yet, construct an exact (P x P) source via extend over (T x T) where centered texels match 1:1.
  std::vector<float> src_unit_T(static_cast<std::size_t>(T) * T * 4u);
  std::vector<float> src_signed_T(static_cast<std::size_t>(T) * T * 4u);
  for (std::uint32_t y = 0u; y < T; ++y) {
    for (std::uint32_t x = 0u; x < T; ++x) {
      const std::size_t src_o = (static_cast<std::size_t>(y) * T + x) * 4u;
      const std::size_t p_o =
          (static_cast<std::size_t>(y + vt_border_size) * P +
           (x + vt_border_size)) *
          4u;
      std::memcpy(src_unit_T.data() + src_o, unit_f32.data() + p_o, 16u);
      std::memcpy(src_signed_T.data() + src_o, signed_f32.data() + p_o, 16u);
    }
  }
  auto clamp_coord = [&](std::uint32_t v) noexcept -> std::uint32_t {
    if (v < vt_border_size) {
      return 0u;
    }
    const std::uint32_t inner = v - vt_border_size;
    return std::min(inner, T - 1u);
  };

  for (const auto fmt : {VtTileFormat::short4, VtTileFormat::short1,
                         VtTileFormat::half4, VtTileFormat::half1,
                         VtTileFormat::float1}) {
    const bool use_unit =
        (fmt == VtTileFormat::short4 || fmt == VtTileFormat::short1);
    const auto &src_vec = use_unit ? src_unit_T : src_signed_T;
    std::vector<std::uint8_t> enc(vt_tile_format_bytes(P, fmt));
    vt_pack_tile_formatted(nullptr, src_vec.data(), T, T, 0u, 0u,
                           2u /*extend*/, P, T, vt_border_size, fmt, enc);
    std::vector<luisa::float4> ref(texels);
    for (std::uint32_t y = 0u; y < P; ++y) {
      for (std::uint32_t x = 0u; x < P; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * P + x;
        const std::uint32_t sx = clamp_coord(x);
        const std::uint32_t sy = clamp_coord(y);
        const std::size_t so = (static_cast<std::size_t>(sy) * T + sx) * 4u;
        auto q_half = [](float v) {
          return vt_half_bits_to_float(vt_float_to_half_bits(v));
        };
        if (fmt == VtTileFormat::short4) {
          ref[i] = luisa::make_float4(src_vec[so + 0u], src_vec[so + 1u],
                                      src_vec[so + 2u], src_vec[so + 3u]);
        } else if (fmt == VtTileFormat::short1) {
          const float v = src_vec[so + 0u];
          ref[i] = luisa::make_float4(v, v, v, 1.0f);
        } else if (fmt == VtTileFormat::half4) {
          ref[i] = luisa::make_float4(
              q_half(src_vec[so + 0u]), q_half(src_vec[so + 1u]),
              q_half(src_vec[so + 2u]), q_half(src_vec[so + 3u]));
        } else if (fmt == VtTileFormat::half1) {
          const float v = q_half(src_vec[so + 0u]);
          ref[i] = luisa::make_float4(v, v, v, 1.0f);
        } else {
          const float v = src_vec[so + 0u];
          ref[i] = luisa::make_float4(v, v, v, 1.0f);
        }
      }
    }
    check_gpu_tile(fmt, enc, ref, 1e-4, 0.0);
  }
}

void test_vt_cache_auto_multi_pool(Device &device, Stream &stream) {
  std::cout << "[VT BC Test 6] VirtualTextureCache multi-pool auto / auto_compact end-to-end...\n";
  if (device.impl()->query("vk_batch_copy_cmd") != "1") {
    std::cout << "  SKIPPED: backend does not support vk_batch_copy_cmd\n";
    return;
  }
#if !defined(PSYCLES_WITH_OPENIMAGEIO)
  std::cout << "  SKIPPED: OpenImageIO disabled\n";
  return;
#else
  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;

  constexpr std::uint32_t W = 64u;
  constexpr std::uint32_t H = 48u;
  constexpr std::size_t npix = static_cast<std::size_t>(W) * H;
  auto tri = [](std::uint32_t v) noexcept -> std::uint8_t {
    const std::uint32_t m = v & 511u;
    return static_cast<std::uint8_t>(m < 256u ? m : (511u - m));
  };

  std::vector<std::uint8_t> gray1(npix);
  std::vector<std::uint8_t> ga2(npix * 2u);
  std::vector<std::uint8_t> rgb3(npix * 3u);
  std::vector<std::uint8_t> rgba4(npix * 4u);
  std::vector<std::uint16_t> u16_4(npix * 4u);
  std::vector<std::uint16_t> bump_h4(npix * 4u);
  std::vector<std::uint16_t> hdr_h3(npix * 3u);

  for (std::uint32_t y = 0u; y < H; ++y) {
    for (std::uint32_t x = 0u; x < W; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * W + x;
      const std::uint8_t g = tri(x * 5u + y * 3u + 19u);
      gray1[i] = g;
      ga2[i * 2u + 0u] = g;
      ga2[i * 2u + 1u] = static_cast<std::uint8_t>(100u + ((x + y) % 150u));
      rgb3[i * 3u + 0u] = tri(x * 7u + 11u);
      rgb3[i * 3u + 1u] = tri(y * 9u + 31u);
      rgb3[i * 3u + 2u] = tri((x + y) * 4u + 71u);
      rgba4[i * 4u + 0u] = rgb3[i * 3u + 0u];
      rgba4[i * 4u + 1u] = rgb3[i * 3u + 1u];
      rgba4[i * 4u + 2u] = rgb3[i * 3u + 2u];
      rgba4[i * 4u + 3u] = static_cast<std::uint8_t>(40u + ((x * 3u) % 210u));

      u16_4[i * 4u + 0u] = static_cast<std::uint16_t>(x * 900u + y * 100u);
      u16_4[i * 4u + 1u] = static_cast<std::uint16_t>(y * 1200u);
      u16_4[i * 4u + 2u] = static_cast<std::uint16_t>((x + y) * 500u);
      u16_4[i * 4u + 3u] = 65535u;

      const float bx = -1.25f + 2.5f * (static_cast<float>(x) / 63.0f);
      const float by = -0.75f + 1.5f * (static_cast<float>(y) / 47.0f);
      bump_h4[i * 4u + 0u] = vt_float_to_half_bits(bx);
      bump_h4[i * 4u + 1u] = vt_float_to_half_bits(by);
      bump_h4[i * 4u + 2u] = vt_float_to_half_bits(0.25f);
      bump_h4[i * 4u + 3u] = vt_float_to_half_bits(1.0f);

      hdr_h3[i * 3u + 0u] =
          vt_float_to_half_bits(0.1f + 0.12f * static_cast<float>(x));
      hdr_h3[i * 3u + 1u] =
          vt_float_to_half_bits(0.2f + 0.08f * static_cast<float>(y));
      hdr_h3[i * 3u + 2u] = vt_float_to_half_bits(
          0.05f + 0.05f * static_cast<float>((x + y) % 19u));
    }
  }

  const auto b_gray =
      encode_oiio_mem("png", W, H, 1, OIIO::TypeDesc::UINT8, gray1.data());
  const auto b_ga =
      encode_oiio_mem("png", W, H, 2, OIIO::TypeDesc::UINT8, ga2.data());
  const auto b_rgb =
      encode_oiio_mem("png", W, H, 3, OIIO::TypeDesc::UINT8, rgb3.data());
  const auto b_rgba =
      encode_oiio_mem("png", W, H, 4, OIIO::TypeDesc::UINT8, rgba4.data());
  const auto b_u16 =
      encode_oiio_mem("png", W, H, 4, OIIO::TypeDesc::UINT16, u16_4.data());
  const auto b_bump =
      encode_oiio_mem("exr", W, H, 4, OIIO::TypeDesc::HALF, bump_h4.data());
  const auto b_hdr =
      encode_oiio_mem("exr", W, H, 3, OIIO::TypeDesc::HALF, hdr_h3.data());

  auto make_probed = [&](std::uint64_t id, std::string name,
                         const std::vector<std::uint8_t> &bytes,
                         ImageAlphaType alpha, ImageColorSpace cs,
                         std::uint8_t exp_ch, std::uint8_t exp_bd,
                         bool exp_fp, bool exp_a) {
    const auto p = probe_scene_image_info(bytes, name);
    require(p.has_value(), "Probe failed for " + name);
    require(p->channels == exp_ch && p->bit_depth == exp_bd &&
                p->is_floating_point == exp_fp && p->has_alpha == exp_a,
            "Probed metadata mismatch for " + name);
    return VtProbedSceneImage{.id = ImageId{id},
                              .name = std::move(name),
                              .encoded_data = bytes,
                              .alpha_type = alpha,
                              .color_space = cs,
                              .probed = *p};
  };

  const std::vector<VtProbedSceneImage> probed{
      make_probed(1u, "gray.png", b_gray, ImageAlphaType::ignore,
                  ImageColorSpace::data, 1u, 8u, false, false),
      make_probed(2u, "ga.png", b_ga, ImageAlphaType::straight,
                  ImageColorSpace::data, 2u, 8u, false, true),
      make_probed(3u, "rgb.png", b_rgb, ImageAlphaType::ignore,
                  ImageColorSpace::srgb, 3u, 8u, false, false),
      make_probed(4u, "rgba.png", b_rgba, ImageAlphaType::straight,
                  ImageColorSpace::srgb, 4u, 8u, false, true),
      make_probed(5u, "u16.png", b_u16, ImageAlphaType::ignore,
                  ImageColorSpace::data, 4u, 16u, false, true),
      make_probed(6u, "bump.exr", b_bump, ImageAlphaType::premultiplied,
                  ImageColorSpace::data, 4u, 16u, true, true),
      make_probed(7u, "hdr.exr", b_hdr, ImageAlphaType::ignore,
                  ImageColorSpace::linear, 3u, 16u, true, false)};

  // Bindings: 0=gray REPEAT(bc4), 1=gray CLIP(bc5), 2=ga REPEAT(bc5),
  // 3=rgb REPEAT(bc7/bc1), 4=rgba REPEAT(bc7/bc3), 5=u16 REPEAT(short4),
  // 6=bump REPEAT(half4 negative), 7=hdr REPEAT(bc6h)
  const std::vector<ImageBinding> handles{
      {.resource_id = 1u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 1u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::clip},
      {.resource_id = 2u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 3u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 4u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 5u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 6u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 7u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat}};
  std::vector<CyclesSvmImageBindingGpu> bindings;
  for (std::uint32_t i = 0u; i < handles.size(); ++i) {
    bindings.push_back(make_cycles_svm_image_binding(
        i, handles[i].interpolation, handles[i].extension));
  }

  std::vector<luisa::float2> uvs{
      luisa::make_float2(0.15f, 0.25f), luisa::make_float2(0.45f, 0.65f),
      luisa::make_float2(0.75f, 0.35f), luisa::make_float2(0.92f, 0.88f),
      luisa::make_float2(-0.25f, 0.5f), luisa::make_float2(1.25f, 0.5f)};
  const std::uint32_t u_count = static_cast<std::uint32_t>(uvs.size());
  const std::uint32_t b_count = static_cast<std::uint32_t>(handles.size());
  auto uv_buf = device.create_buffer<luisa::float2>(u_count);
  auto out_buf = device.create_buffer<luisa::float4>(u_count * b_count);
  stream << uv_buf.copy_from(luisa::span{uvs}) << synchronize();
  BindlessArray dummy_heap = device.create_bindless_array(8u);

  const std::filesystem::path disk_dir =
      std::filesystem::temp_directory_path() / "psycles_vt_auto_disk_test";
  std::error_code ec;
  std::filesystem::remove_all(disk_dir, ec);

  auto run_mode = [&](LuisaVirtualTextureCompressionMode mode,
                      LuisaVirtualTexturePrepackMode pp_mode) {
    const LuisaVirtualTextureCacheOptions opts{
        .enabled = true,
        .gpu_cache_budget_bytes = 32ull * 1024ull * 1024ull,
        .physical_tile_size = 64u,
        .prepack_mode = pp_mode,
        .prepack_disk_dir = disk_dir.string(),
        .compression_mode = mode};
    auto vt = VirtualTextureCache::create(device, stream, opts, probed, handles,
                                          bindings);
    auto miss_cnt = device.create_buffer<luisa::uint>(1u);
    auto k = device.compile<1>(
        [&](BufferVar<luisa::float2> in_uv, BufferVar<luisa::float4> dst,
            BufferVar<luisa::uint> misses, UInt nuv) noexcept {
          const auto uid = dispatch_x();
          $if(uid >= nuv) { $return(); };
          const auto uv = in_uv.read(uid);
          for (std::uint32_t b = 0u; b < 8u; ++b) {
            UInt miss = vt_invalid_page;
            const auto c = vt->sample_2d_const(static_cast<std::int32_t>(b), uv,
                                               &miss);
            $if(miss != vt_invalid_page) { misses.atomic(0u).fetch_add(1u); };
            dst.write(b * nuv + uid, c);
          }
        });
    std::uint32_t z = 0u, hm = 0u;
    for (int it = 0; it < 8; ++it) {
      stream << miss_cnt.copy_from(&z)
             << k(uv_buf, out_buf, miss_cnt, u_count).dispatch(u_count)
             << miss_cnt.copy_to(&hm) << synchronize();
      if (hm == 0u) {
        break;
      }
      vt->service_bitset_misses(stream);
    }
    require(hm == 0u, "Multi-pool VT did not converge");
    std::vector<luisa::float4> px(u_count * b_count);
    stream << out_buf.copy_to(luisa::span{px}) << synchronize();
    return std::pair{vt->stats(), std::move(px)};
  };

  const auto [s_off, p_off] =
      run_mode(LuisaVirtualTextureCompressionMode::off,
               LuisaVirtualTexturePrepackMode::off);
  const auto [s_hq, p_hq] =
      run_mode(LuisaVirtualTextureCompressionMode::auto_hq,
               LuisaVirtualTexturePrepackMode::disk);
  const auto [s_hq_warm, p_hq_warm] =
      run_mode(LuisaVirtualTextureCompressionMode::auto_hq,
               LuisaVirtualTexturePrepackMode::disk);
  const auto [s_cp, p_cp] =
      run_mode(LuisaVirtualTextureCompressionMode::auto_compact,
               LuisaVirtualTexturePrepackMode::off);

  std::filesystem::remove_all(disk_dir, ec);

  std::cout << "  auto_hq pools=" << s_hq.pool_summary << '\n';
  std::cout << "  auto_compact pools=" << s_cp.pool_summary << '\n';
  for (std::string_view req_pool :
       {"bc4:", "bc5:", "bc7:", "short4:", "half4:", "bc6h:"}) {
    require(s_hq.pool_summary.find(req_pool) != std::string::npos,
            std::string{"Missing expected pool in auto_hq: "} +
                std::string{req_pool});
  }
  for (std::string_view req_pool :
       {"bc4:", "bc5:", "bc3:", "bc1:", "short4:", "half4:", "bc6h:"}) {
    require(s_cp.pool_summary.find(req_pool) != std::string::npos,
            std::string{"Missing expected pool in auto_compact: "} +
                std::string{req_pool});
  }
  require(s_hq.prepack_disk_writes > 0u && s_hq_warm.prepack_disk_reads > 0u &&
              s_hq_warm.images_decoded == 0u,
          "Expected auto_hq disk prepack write and warm read with 0 decodes");
  require(std::memcmp(p_hq.data(), p_hq_warm.data(),
                      p_hq.size() * sizeof(luisa::float4)) == 0,
          "auto_hq disk warm output differs from cold output");

  // Check CLIP outside UVs [-0.25, 1.25] on binding 1 (bc5) == (0,0,0,0)
  for (const auto &px_vec : {std::cref(p_hq), std::cref(p_cp)}) {
    for (std::uint32_t u : {4u, 5u}) {
      const auto c = px_vec.get()[1u * u_count + u];
      require(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f && c.w == 0.0f,
              "CLIP out-of-bounds must remain (0,0,0,0) under BC5");
    }
    // Check negative bump (binding 6 -> half4) and 16-bit PNG (binding 5 -> short4) remain lossless
    bool saw_negative_bump = false;
    for (std::uint32_t u = 0u; u < u_count; ++u) {
      const auto c5 = px_vec.get()[5u * u_count + u];
      const auto r5 = p_off[5u * u_count + u];
      const auto c6 = px_vec.get()[6u * u_count + u];
      const auto r6 = p_off[6u * u_count + u];
      if (r6.x < -0.1f || r6.y < -0.1f) {
        saw_negative_bump = true;
      }
      for (int ch = 0; ch < 4; ++ch) {
        require(std::abs((&c5.x)[ch] - (&r5.x)[ch]) < 1e-4f,
                "SHORT4 PNG precision loss exceeded 1e-4");
        require(std::abs((&c6.x)[ch] - (&r6.x)[ch]) < 1e-3f,
                "HALF4 bump precision loss exceeded 1e-3");
      }
    }
    require(saw_negative_bump, "Expected negative bump values in test");
  }
  require(s_hq.gpu_allocated_atlas_bytes < s_off.gpu_allocated_atlas_bytes,
          "Expected auto_hq atlas bytes < off");
  require(s_cp.gpu_allocated_atlas_bytes < s_hq.gpu_allocated_atlas_bytes,
          "Expected auto_compact atlas bytes < auto_hq");
#endif
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    bool bench_mode = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string_view{argv[i]} == "--bench") {
        bench_mode = true;
      }
    }
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    if (bench_mode) {
      psycles::vt_bc_test::run_gpu_bc_benchmark(device, stream);
      return EXIT_SUCCESS;
    }
    test_bc_encoder_decoder_quality();
    std::cout << "Running Psycles VT BC tests on backend: " << backend << '\n';
    test_bc_gpu_hw_decode_parity(device, stream);
    test_multi_format_hw_decode_parity(device, stream);
    test_vt_cache_bc_end_to_end(device, stream);
    test_vt_cache_auto_multi_pool(device, stream);
    psycles::vt_bc_test::test_tex_compress_ext_direct(device, stream);
    psycles::vt_bc_test::test_gpu_bc_compressor(device, stream);
    std::cout << "ALL VT BC COMPRESSION TESTS PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
