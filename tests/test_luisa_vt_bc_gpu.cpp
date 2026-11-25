#include "virtual_texture_cache_bc.h"
#include "virtual_texture_cache_formats.h"
#include "virtual_texture_cache_gpu_bc.h"

#include <psycles/core/parallel.h>

#include <luisa/backends/ext/tex_compress_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace psycles::vt_bc_test {
namespace {

using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

[[nodiscard]] double compute_psnr_rgba(
    const std::uint8_t *ref,
    const std::uint8_t *act,
    std::size_t byte_count,
    bool rgb_only = false) noexcept {
  double sum_sq = 0.0;
  std::size_t n = 0u;
  for (std::size_t i = 0u; i < byte_count; i += 4u) {
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

[[nodiscard]] std::vector<std::uint8_t> make_mixed_rgba_tile(std::uint32_t P) {
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(P) * P * 4u, 0u);
  for (std::uint32_t y = 0u; y < P; ++y) {
    for (std::uint32_t x = 0u; x < P; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * P + x) * 4u;
      const std::uint32_t bx = x / 4u;
      const std::uint32_t by = y / 4u;
      if (by == 0u && bx == 0u) {
        continue;
      }
      if (by == 0u && bx == 1u) {
        rgba[off + 0u] = 100u;
        rgba[off + 1u] = 101u;
        rgba[off + 2u] = 102u;
        rgba[off + 3u] = 255u;
      } else if (by < 4u) {
        rgba[off + 0u] = static_cast<std::uint8_t>((x * 255u) / P);
        rgba[off + 1u] = static_cast<std::uint8_t>((y * 255u) / (P / 4u));
        rgba[off + 2u] = static_cast<std::uint8_t>(((x + y) * 199u) / P);
        rgba[off + 3u] = static_cast<std::uint8_t>((x * 240u) / P + 15u);
      } else if (by < 8u) {
        const bool side = ((x % 8u) < 4u) && ((x + y) % 7u < 4u);
        rgba[off + 0u] = side ? 240u : 20u;
        rgba[off + 1u] = side ? 24u : 228u;
        rgba[off + 2u] = side ? 16u : 244u;
        rgba[off + 3u] = 255u;
      } else {
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

[[nodiscard]] std::vector<std::uint8_t> make_cutout_rgba_tile(std::uint32_t P) {
  std::vector<std::uint8_t> cutout(static_cast<std::size_t>(P) * P * 4u, 0u);
  for (std::uint32_t y = 0u; y < P; ++y) {
    for (std::uint32_t x = 0u; x < P; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * P + x) * 4u;
      const bool keep = ((x / 4u + y / 4u) & 1u) == 0u;
      cutout[off + 0u] = keep ? 220u : 20u;
      cutout[off + 1u] = keep ? 40u : 190u;
      cutout[off + 2u] = keep ? 30u : 230u;
      cutout[off + 3u] = keep ? 255u : 0u;
    }
  }
  return cutout;
}

[[nodiscard]] std::vector<float> make_hdr_tile(std::uint32_t P) {
  std::vector<float> hdr(static_cast<std::size_t>(P) * P * 4u, 1.0f);
  for (std::uint32_t y = 0u; y < P; ++y) {
    for (std::uint32_t x = 0u; x < P; ++x) {
      const std::size_t off = (static_cast<std::size_t>(y) * P + x) * 4u;
      hdr[off + 0u] = 0.05f + 12.0f * (static_cast<float>(x) / P);
      hdr[off + 1u] = 0.10f + 4.5f * (static_cast<float>(y) / P);
      hdr[off + 2u] = 0.25f + 2.0f * (static_cast<float>((x + y) % P) / P);
      hdr[off + 3u] = 1.0f;
    }
  }
  return hdr;
}

[[nodiscard]] bool supports_hw_bc_sampling(const Device & /*device*/) noexcept {
  return true;
}

[[nodiscard]] std::vector<std::uint8_t> sw_decode_legacy_bc_rgba8(
    VtTileFormat fmt,
    std::uint32_t P,
    const std::uint8_t *enc);

[[nodiscard]] std::vector<luisa::float4> hw_decode_image(
    Device &device,
    Stream &stream,
    PixelStorage storage,
    std::uint32_t W,
    std::uint32_t H,
    const void *enc) {
  auto tex = device.create_image<float>(storage, W, H);
  auto heap = device.create_bindless_array(2u);
  heap.emplace_on_update(0u, tex, Sampler::point_edge());
  auto out_buf = device.create_buffer<luisa::float4>(static_cast<std::size_t>(W) * H);
  auto k = device.compile<1>(
      [W, H](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
        const auto tid = dispatch_x();
        $if(tid >= W * H) { $return(); };
        const auto y = tid / W;
        const auto x = tid - y * W;
        const auto uv = make_float2(
            (cast<float>(x) + 0.5f) / static_cast<float>(W),
            (cast<float>(y) + 0.5f) / static_cast<float>(H));
        dst.write(tid, h.tex2d(0u).sample(uv));
      });
  std::vector<luisa::float4> px(static_cast<std::size_t>(W) * H);
  stream << tex.copy_from(enc) << heap.update()
         << k(heap, out_buf).dispatch(W * H)
         << out_buf.copy_to(luisa::span{px}) << synchronize();
  return px;
}

[[nodiscard]] std::vector<std::uint8_t> f4_to_rgba8(
    const std::vector<luisa::float4> &px) {
  std::vector<std::uint8_t> u8(px.size() * 4u, 0u);
  for (std::size_t i = 0u; i < px.size(); ++i) {
    const float vals[4]{px[i].x, px[i].y, px[i].z, px[i].w};
    for (int c = 0; c < 4; ++c) {
      u8[i * 4u + c] = static_cast<std::uint8_t>(std::clamp(
          static_cast<int>(std::lround(std::clamp(vals[c], 0.0f, 1.0f) * 255.0f)),
          0, 255));
    }
  }
  return u8;
}

[[nodiscard]] int max_channel_diff(
    const std::vector<std::uint8_t> &a,
    const std::vector<std::uint8_t> &b,
    int ch_idx = -1) noexcept {
  int max_diff = 0;
  for (std::size_t i = 0u; i < a.size(); i += 4u) {
    if (ch_idx >= 0) {
      max_diff = std::max(
          max_diff, std::abs(static_cast<int>(a[i + ch_idx]) -
                             static_cast<int>(b[i + ch_idx])));
    } else {
      for (int c = 0; c < 4; ++c) {
        max_diff = std::max(
            max_diff, std::abs(static_cast<int>(a[i + c]) -
                               static_cast<int>(b[i + c])));
      }
    }
  }
  return max_diff;
}

void verify_bc7_compressor_mode(
    Device &device,
    Stream &stream,
    VtGpuBcCompressor &bc,
    std::uint32_t P) {
  constexpr std::uint32_t kTiles = 4u;
  const auto src0 = make_mixed_rgba_tile(P);
  const auto cutout = make_cutout_rgba_tile(P);
  const std::size_t bc7_bytes = vt_tile_format_bytes(P, VtTileFormat::bc7);
  std::vector<std::uint8_t> bc7_out(kTiles * bc7_bytes, 0u);
  std::vector<VtGpuBcTileJob> jobs;
  jobs.reserve(kTiles);
  for (std::uint32_t t = 0u; t < kTiles; ++t) {
    jobs.push_back(VtGpuBcTileJob{
        .unorm8_pixels = (t == 1u) ? cutout.data() : src0.data(),
        .float_pixels = nullptr,
        .width = P,
        .height = P,
        .tile_x = 0u,
        .tile_y = 0u,
        .extension = 0u,
        .physical_tile_size = P,
        .logical_tile_size = P,
        .border = 0u,
        .format = VtTileFormat::bc7,
        .dst = bc7_out.data() + t * bc7_bytes,
        .dst_bytes = bc7_bytes});
  }
  require(bc.compress_batch(jobs), "GPU BC7 compress_batch failed");

  const bool hw_bc = supports_hw_bc_sampling(device);
  const auto dec_syn = hw_bc
      ? f4_to_rgba8(hw_decode_image(device, stream, PixelStorage::BC7, P, P, bc7_out.data()))
      : sw_decode_legacy_bc_rgba8(VtTileFormat::bc7, P, bc7_out.data());
  const auto dec_cut = hw_bc
      ? f4_to_rgba8(hw_decode_image(device, stream, PixelStorage::BC7, P, P, bc7_out.data() + bc7_bytes))
      : sw_decode_legacy_bc_rgba8(VtTileFormat::bc7, P, bc7_out.data() + bc7_bytes);
  const double psnr_syn =
      compute_psnr_rgba(src0.data(), dec_syn.data(), src0.size(), false);
  const double psnr_cut =
      compute_psnr_rgba(cutout.data(), dec_cut.data(), cutout.size(), false);
  const int a_diff = max_channel_diff(src0, dec_syn, 3);
  const int cut_a_diff = max_channel_diff(cutout, dec_cut, 3);
  std::cout << "  GPU BC7 (" << bc.backend_name() << "/" << bc.quality_name()
            << ") syn_PSNR=" << psnr_syn
            << " dB, cutout_PSNR=" << psnr_cut
            << " dB, alpha_max_diff=" << a_diff
            << ", cutout_alpha_max_diff=" << cut_a_diff << '\n';
  require(psnr_syn >= 38.0, "GPU BC7 synthetic PSNR below 38 dB");
  require(psnr_cut >= 38.0, "GPU BC7 cutout PSNR below 38 dB");
  require(a_diff <= 10, "GPU BC7 alpha max diff exceeded 10/255");
  require(cut_a_diff <= 4, "GPU BC7 cutout alpha max diff exceeded 4/255");

  if (hw_bc && bc.backend() == VtGpuBcCompressor::Backend::dsl) {
    const auto sw_syn =
        sw_decode_legacy_bc_rgba8(VtTileFormat::bc7, P, bc7_out.data());
    const double sw_psnr =
        compute_psnr_rgba(src0.data(), sw_syn.data(), src0.size(), false);
    require(sw_psnr >= 38.0, "GPU BC7 DSL software decode PSNR below 38 dB");
  }

  // Exhaustive 16-partition sharp-edge test (opaque Mode 1 + cutout Mode 7).
  static constexpr std::array<std::uint32_t, 16u> kPartMasks{
      0xccccu, 0x8888u, 0xeeeeu, 0xecc8u, 0xc880u, 0xfeecu, 0xfec8u, 0xec80u,
      0xc800u, 0xffecu, 0xfe80u, 0xe800u, 0xffe8u, 0xff00u, 0xfff0u, 0xf000u};
  std::vector<std::uint8_t> part_opaque(src0.size(), 0u);
  std::vector<std::uint8_t> part_cutout(src0.size(), 0u);
  for (std::uint32_t by = 0u; by < P / 4u; ++by) {
    for (std::uint32_t bx = 0u; bx < P / 4u; ++bx) {
      const std::uint32_t mask = kPartMasks[(by * (P / 4u) + bx) & 15u];
      for (std::uint32_t py = 0u; py < 4u; ++py) {
        for (std::uint32_t px = 0u; px < 4u; ++px) {
          const std::uint32_t bit = (mask >> (py * 4u + px)) & 1u;
          const std::size_t off =
              (static_cast<std::size_t>(by * 4u + py) * P + bx * 4u + px) * 4u;
          part_opaque[off + 0u] = bit ? 20u : 240u;
          part_opaque[off + 1u] = bit ? 228u : 24u;
          part_opaque[off + 2u] = bit ? 244u : 16u;
          part_opaque[off + 3u] = 255u;
          part_cutout[off + 0u] = bit ? 20u : 240u;
          part_cutout[off + 1u] = bit ? 228u : 24u;
          part_cutout[off + 2u] = bit ? 244u : 16u;
          part_cutout[off + 3u] = bit ? 0u : 255u;
        }
      }
    }
  }
  std::vector<std::uint8_t> part_out(2u * bc7_bytes, 0u);
  std::array<VtGpuBcTileJob, 2u> part_jobs{
      VtGpuBcTileJob{.unorm8_pixels = part_opaque.data(),
                     .width = P,
                     .height = P,
                     .physical_tile_size = P,
                     .logical_tile_size = P,
                     .format = VtTileFormat::bc7,
                     .dst = part_out.data(),
                     .dst_bytes = bc7_bytes},
      VtGpuBcTileJob{.unorm8_pixels = part_cutout.data(),
                     .width = P,
                     .height = P,
                     .physical_tile_size = P,
                     .logical_tile_size = P,
                     .format = VtTileFormat::bc7,
                     .dst = part_out.data() + bc7_bytes,
                     .dst_bytes = bc7_bytes}};
  require(bc.compress_batch(part_jobs), "GPU BC7 partition compress_batch failed");
  const auto dec_po = hw_bc
      ? f4_to_rgba8(hw_decode_image(device, stream, PixelStorage::BC7, P, P, part_out.data()))
      : sw_decode_legacy_bc_rgba8(VtTileFormat::bc7, P, part_out.data());
  const auto dec_pc = hw_bc
      ? f4_to_rgba8(hw_decode_image(device, stream, PixelStorage::BC7, P, P, part_out.data() + bc7_bytes))
      : sw_decode_legacy_bc_rgba8(VtTileFormat::bc7, P, part_out.data() + bc7_bytes);
  const int max_po = max_channel_diff(part_opaque, dec_po);
  const int max_pc = max_channel_diff(part_cutout, dec_pc);
  std::cout << "  GPU BC7 16-partition (" << bc.backend_name() << "/"
            << bc.quality_name()
            << ") max_err(opaque/cutout)=" << max_po << "/" << max_pc << '\n';
  require(max_po <= 8, "GPU BC7 16-partition opaque max error exceeded 8/255");
  require(max_pc <= 8, "GPU BC7 16-partition cutout max error exceeded 8/255");
}

void verify_bc6h_compressor_mode(
    Device &device,
    Stream &stream,
    VtGpuBcCompressor &bc,
    std::uint32_t P) {
  constexpr std::uint32_t kTiles = 4u;
  const auto hdr_src = make_hdr_tile(P);
  const std::size_t bc6_bytes = vt_tile_format_bytes(P, VtTileFormat::bc6h);
  std::vector<std::uint8_t> bc6_out(kTiles * bc6_bytes, 0u);
  std::vector<VtGpuBcTileJob> jobs;
  for (std::uint32_t t = 0u; t < kTiles; ++t) {
    jobs.push_back(VtGpuBcTileJob{
        .float_pixels = hdr_src.data(),
        .width = P,
        .height = P,
        .physical_tile_size = P,
        .logical_tile_size = P,
        .format = VtTileFormat::bc6h,
        .dst = bc6_out.data() + t * bc6_bytes,
        .dst_bytes = bc6_bytes});
  }
  require(bc.compress_batch(jobs), "GPU BC6H compress_batch failed");
  if (!supports_hw_bc_sampling(device)) {
    const bool any_nonzero = std::any_of(
        bc6_out.begin(), bc6_out.end(),
        [](std::uint8_t v) noexcept { return v != 0u; });
    require(any_nonzero, "GPU BC6H produced all-zero output");
    std::cout << "  GPU BC6H (" << bc.backend_name() << "/" << bc.quality_name()
              << ") hw_decode=skipped (" << device.backend_name() << ")\n";
    return;
  }
  const auto dec =
      hw_decode_image(device, stream, PixelStorage::BC6, P, P, bc6_out.data());
  double rel_sum = 0.0;
  std::size_t n = 0u;
  for (std::size_t i = 0u; i < static_cast<std::size_t>(P) * P; ++i) {
    const float dv[3]{dec[i].x, dec[i].y, dec[i].z};
    for (int ch = 0; ch < 3; ++ch) {
      require(std::isfinite(dv[ch]) && dv[ch] >= 0.0f,
              "GPU BC6H decoded non-finite or negative value");
      rel_sum += std::abs(dv[ch] - hdr_src[i * 4u + ch]) /
                 std::max(hdr_src[i * 4u + ch], 0.1f);
      ++n;
    }
  }
  const double mean_rel = rel_sum / static_cast<double>(n);
  std::cout << "  GPU BC6H (" << bc.backend_name() << "/" << bc.quality_name()
            << ") mean_rel_err=" << mean_rel << '\n';
  require(mean_rel < 0.05, "GPU BC6H mean relative error >= 5%");
}

[[nodiscard]] std::vector<std::uint8_t> sw_decode_legacy_bc_rgba8(
    VtTileFormat fmt,
    std::uint32_t P,
    const std::uint8_t *enc) {
  const std::uint32_t bpa = P / 4u;
  const std::size_t blk_bytes =
      (fmt == VtTileFormat::bc1 || fmt == VtTileFormat::bc4) ? 8u : 16u;
  std::uint8_t blk[64]{};
  std::vector<std::uint8_t> sw(static_cast<std::size_t>(P) * P * 4u, 0u);
  for (std::uint32_t by = 0u; by < bpa; ++by) {
    for (std::uint32_t bx = 0u; bx < bpa; ++bx) {
      const std::uint8_t *src_blk = enc + (by * bpa + bx) * blk_bytes;
      switch (fmt) {
        case VtTileFormat::bc1: vt_decode_bc1_block(src_blk, blk); break;
        case VtTileFormat::bc3: vt_decode_bc3_block(src_blk, blk); break;
        case VtTileFormat::bc4: vt_decode_bc4_block(src_blk, blk); break;
        case VtTileFormat::bc5: vt_decode_bc5_block(src_blk, blk); break;
        case VtTileFormat::bc7: vt_decode_bc7_block(src_blk, blk); break;
        default: break;
      }
      for (std::uint32_t r = 0u; r < 4u; ++r) {
        std::memcpy(
            sw.data() + (static_cast<std::size_t>(by * 4u + r) * P + bx * 4u) * 4u,
            blk + r * 16u, 16u);
      }
    }
  }
  return sw;
}

void verify_dsl_legacy_bc_formats(
    Device &device,
    Stream &stream,
    VtGpuBcCompressor &bc,
    std::uint32_t P) {
  auto opaque_src = make_mixed_rgba_tile(P);
  for (std::size_t i = 3u; i < opaque_src.size(); i += 4u) {
    opaque_src[i] = 255u;
  }
  const auto rgba_src = make_mixed_rgba_tile(P);
  const bool hw_legacy_bc = supports_hw_bc_sampling(device);

  auto run_fmt = [&](VtTileFormat fmt, const std::uint8_t *src_rgba) {
    const std::size_t tbytes = vt_tile_format_bytes(P, fmt);
    std::vector<std::uint8_t> out(2u * tbytes, 0u);
    std::array<VtGpuBcTileJob, 2u> jobs{
        VtGpuBcTileJob{.unorm8_pixels = src_rgba,
                       .width = P,
                       .height = P,
                       .physical_tile_size = P,
                       .logical_tile_size = P,
                       .format = fmt,
                       .dst = out.data(),
                       .dst_bytes = tbytes},
        VtGpuBcTileJob{.unorm8_pixels = src_rgba,
                       .width = P,
                       .height = P,
                       .physical_tile_size = P,
                       .logical_tile_size = P,
                       .format = fmt,
                       .dst = out.data() + tbytes,
                       .dst_bytes = tbytes}};
    require(bc.compress_batch(jobs), "GPU legacy BC compress_batch failed");
    return out;
  };

  // BC1
  {
    const auto out = run_fmt(VtTileFormat::bc1, opaque_src.data());
    const auto sw = sw_decode_legacy_bc_rgba8(VtTileFormat::bc1, P, out.data());
    const double sw_psnr =
        compute_psnr_rgba(opaque_src.data(), sw.data(), opaque_src.size(), true);
    if (hw_legacy_bc) {
      const auto dec = f4_to_rgba8(
          hw_decode_image(device, stream, PixelStorage::BC1, P, P, out.data()));
      const double hw_psnr =
          compute_psnr_rgba(opaque_src.data(), dec.data(), opaque_src.size(), true);
      std::cout << "  GPU BC1 (dsl) hw_PSNR=" << hw_psnr
                << " dB, sw_PSNR=" << sw_psnr << " dB\n";
      require(hw_psnr >= 30.0 && sw_psnr >= 30.0, "GPU BC1 PSNR below 30 dB");
    } else {
      std::cout << "  GPU BC1 (dsl) hw_PSNR=skipped, sw_PSNR=" << sw_psnr
                << " dB\n";
      require(sw_psnr >= 30.0, "GPU BC1 SW PSNR below 30 dB");
    }
  }

  // BC3
  {
    const auto out = run_fmt(VtTileFormat::bc3, rgba_src.data());
    const auto sw = sw_decode_legacy_bc_rgba8(VtTileFormat::bc3, P, out.data());
    const double sw_psnr =
        compute_psnr_rgba(rgba_src.data(), sw.data(), rgba_src.size(), false);
    const int sw_a_diff = max_channel_diff(rgba_src, sw, 3);
    if (hw_legacy_bc) {
      const auto dec = f4_to_rgba8(
          hw_decode_image(device, stream, PixelStorage::BC3, P, P, out.data()));
      const double hw_psnr =
          compute_psnr_rgba(rgba_src.data(), dec.data(), rgba_src.size(), false);
      const int a_diff = max_channel_diff(rgba_src, dec, 3);
      std::cout << "  GPU BC3 (dsl) hw_PSNR=" << hw_psnr
                << " dB, sw_PSNR=" << sw_psnr << " dB, alpha_max_diff=" << a_diff
                << '\n';
      require(hw_psnr >= 30.0 && sw_psnr >= 30.0, "GPU BC3 PSNR below 30 dB");
      require(a_diff <= 8 && sw_a_diff <= 8,
              "GPU BC3 alpha max diff exceeded 8/255");
    } else {
      std::cout << "  GPU BC3 (dsl) hw_PSNR=skipped, sw_PSNR=" << sw_psnr
                << " dB, alpha_max_diff=" << sw_a_diff << '\n';
      require(sw_psnr >= 30.0, "GPU BC3 SW PSNR below 30 dB");
      require(sw_a_diff <= 8, "GPU BC3 SW alpha max diff exceeded 8/255");
    }
  }

  // BC4
  {
    const auto out = run_fmt(VtTileFormat::bc4, rgba_src.data());
    const auto sw = sw_decode_legacy_bc_rgba8(VtTileFormat::bc4, P, out.data());
    double sum_sq_sw = 0.0;
    for (std::size_t i = 0u; i < rgba_src.size(); i += 4u) {
      const double d1 = static_cast<double>(rgba_src[i]) - sw[i];
      sum_sq_sw += d1 * d1;
    }
    const double n_px = static_cast<double>(P) * P;
    const double sw_psnr =
        sum_sq_sw <= 1e-12 ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / (sum_sq_sw / n_px));
    if (hw_legacy_bc) {
      const auto dec = f4_to_rgba8(
          hw_decode_image(device, stream, PixelStorage::BC4, P, P, out.data()));
      double sum_sq_hw = 0.0;
      for (std::size_t i = 0u; i < rgba_src.size(); i += 4u) {
        const double d0 = static_cast<double>(rgba_src[i]) - dec[i];
        sum_sq_hw += d0 * d0;
      }
      const double hw_psnr =
          sum_sq_hw <= 1e-12 ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / (sum_sq_hw / n_px));
      std::cout << "  GPU BC4 (dsl) hw_PSNR=" << hw_psnr
                << " dB, sw_PSNR=" << sw_psnr << " dB\n";
      require(hw_psnr >= 38.0 && sw_psnr >= 38.0, "GPU BC4 PSNR below 38 dB");
    } else {
      std::cout << "  GPU BC4 (dsl) hw_PSNR=skipped, sw_PSNR=" << sw_psnr
                << " dB\n";
      require(sw_psnr >= 38.0, "GPU BC4 SW PSNR below 38 dB");
    }
  }

  // BC5 (encodes R=ch0 and A=ch3 into HW X and Y channels)
  {
    const auto out = run_fmt(VtTileFormat::bc5, rgba_src.data());
    const auto sw = sw_decode_legacy_bc_rgba8(VtTileFormat::bc5, P, out.data());
    double sum_sq_sw = 0.0;
    for (std::size_t i = 0u; i < rgba_src.size(); i += 4u) {
      const double dr1 = static_cast<double>(rgba_src[i + 0u]) - sw[i + 0u];
      const double da1 = static_cast<double>(rgba_src[i + 3u]) - sw[i + 1u];
      sum_sq_sw += dr1 * dr1 + da1 * da1;
    }
    const double n_samples = static_cast<double>(P) * P * 2.0;
    const double sw_psnr = sum_sq_sw <= 1e-12
                               ? 99.0
                               : 10.0 * std::log10((255.0 * 255.0) / (sum_sq_sw / n_samples));
    if (hw_legacy_bc) {
      const auto dec = f4_to_rgba8(
          hw_decode_image(device, stream, PixelStorage::BC5, P, P, out.data()));
      double sum_sq_hw = 0.0;
      for (std::size_t i = 0u; i < rgba_src.size(); i += 4u) {
        const double dr0 = static_cast<double>(rgba_src[i + 0u]) - dec[i + 0u];
        const double da0 = static_cast<double>(rgba_src[i + 3u]) - dec[i + 1u];
        sum_sq_hw += dr0 * dr0 + da0 * da0;
      }
      const double hw_psnr = sum_sq_hw <= 1e-12
                                 ? 99.0
                                 : 10.0 * std::log10((255.0 * 255.0) / (sum_sq_hw / n_samples));
      std::cout << "  GPU BC5 (dsl) hw_PSNR=" << hw_psnr
                << " dB, sw_PSNR=" << sw_psnr << " dB\n";
      require(hw_psnr >= 38.0 && sw_psnr >= 38.0, "GPU BC5 PSNR below 38 dB");
    } else {
      std::cout << "  GPU BC5 (dsl) hw_PSNR=skipped, sw_PSNR=" << sw_psnr
                << " dB\n";
      require(sw_psnr >= 38.0, "GPU BC5 SW PSNR below 38 dB");
    }
  }
}

[[nodiscard]] double median_of(std::vector<double> vals) {
  std::sort(vals.begin(), vals.end());
  return vals[vals.size() / 2u];
}

} // namespace

void test_tex_compress_ext_direct(Device &device, Stream &stream) {
  auto *ext = device.extension<TexCompressExt>();
  if (ext == nullptr) {
    std::cout << "[VT BC Test 4b] TexCompressExt unavailable on "
              << device.backend_name() << "; skipping direct ext tests.\n";
    return;
  }
  std::cout << "[VT BC Test 4b] Direct TexCompressExt checks on "
            << device.backend_name() << "...\n";
  require(ext->check_builtin_shader() == TexCompressExt::Result::Success,
          "TexCompressExt::check_builtin_shader failed");

  constexpr std::uint32_t P = 64u;
  const auto src_syn = make_mixed_rgba_tile(P);
  const auto src_cut = make_cutout_rgba_tile(P);
  const std::size_t blk_u32s = (P / 4u) * (P / 4u) * 4u;

  auto to_f4 = [](const std::vector<std::uint8_t> &u8) {
    std::vector<luisa::float4> f4(u8.size() / 4u);
    for (std::size_t i = 0u; i < f4.size(); ++i) {
      f4[i] = luisa::make_float4(
          u8[i * 4u + 0u] / 255.0f, u8[i * 4u + 1u] / 255.0f,
          u8[i * 4u + 2u] / 255.0f, u8[i * 4u + 3u] / 255.0f);
    }
    return f4;
  };

  // BYTE4 and FLOAT4 input storage + alpha preservation.
  for (bool use_float4 : {false, true}) {
    const PixelStorage storage =
        use_float4 ? PixelStorage::FLOAT4 : PixelStorage::BYTE4;
    auto tex_syn = device.create_image<float>(storage, P, P);
    auto tex_cut = device.create_image<float>(storage, P, P);
    auto out_syn = device.create_buffer<luisa::uint>(blk_u32s);
    auto out_cut = device.create_buffer<luisa::uint>(blk_u32s);
    const auto f4_syn = to_f4(src_syn);
    const auto f4_cut = to_f4(src_cut);
    stream << tex_syn.copy_from(use_float4 ? static_cast<const void *>(f4_syn.data())
                                           : static_cast<const void *>(src_syn.data()))
           << tex_cut.copy_from(use_float4 ? static_cast<const void *>(f4_cut.data())
                                           : static_cast<const void *>(src_cut.data()))
           << synchronize();
    require(ext->compress_bc7(stream, tex_syn.view(0u), out_syn, 1.0f) ==
                TexCompressExt::Result::Success,
            "TexCompressExt::compress_bc7(syn) failed");
    require(ext->compress_bc7(stream, tex_cut.view(0u), out_cut, 1.0f) ==
                TexCompressExt::Result::Success,
            "TexCompressExt::compress_bc7(cut) failed");
    std::vector<luisa::uint> host_syn(blk_u32s), host_cut(blk_u32s);
    stream << out_syn.copy_to(luisa::span{host_syn})
           << out_cut.copy_to(luisa::span{host_cut}) << synchronize();
    const auto dec_syn = f4_to_rgba8(
        hw_decode_image(device, stream, PixelStorage::BC7, P, P, host_syn.data()));
    const auto dec_cut = f4_to_rgba8(
        hw_decode_image(device, stream, PixelStorage::BC7, P, P, host_cut.data()));
    const double psnr_syn =
        compute_psnr_rgba(src_syn.data(), dec_syn.data(), src_syn.size(), false);
    const double psnr_cut =
        compute_psnr_rgba(src_cut.data(), dec_cut.data(), src_cut.size(), false);
    const int a_diff = max_channel_diff(src_syn, dec_syn, 3);
    const int cut_a_diff = max_channel_diff(src_cut, dec_cut, 3);
    std::cout << "  TexCompressExt BC7(" << (use_float4 ? "FLOAT4" : "BYTE4")
              << ") syn_PSNR=" << psnr_syn << " dB, cutout_PSNR=" << psnr_cut
              << " dB, alpha_max_diff=" << a_diff << "/" << cut_a_diff << '\n';
    require(psnr_syn >= 38.0 && psnr_cut >= 38.0,
            "TexCompressExt BC7 PSNR below 38 dB");
    require(a_diff <= 10 && cut_a_diff <= 4,
            "TexCompressExt BC7 alpha error too high");
  }

  // Non-multiple-of-4 size (13x11).
  {
    constexpr std::uint32_t nw = 13u, nh = 11u;
    std::vector<std::uint8_t> npot(nw * nh * 4u, 0u);
    for (std::uint32_t y = 0u; y < nh; ++y) {
      for (std::uint32_t x = 0u; x < nw; ++x) {
        const std::size_t off = (y * nw + x) * 4u;
        npot[off + 0u] = static_cast<std::uint8_t>(x * 20u);
        npot[off + 1u] = static_cast<std::uint8_t>(y * 23u);
        npot[off + 2u] = static_cast<std::uint8_t>((x + y) * 10u);
        npot[off + 3u] = static_cast<std::uint8_t>(255u - x * 12u);
      }
    }
    const std::size_t npot_u32s = ((nw + 3u) / 4u) * ((nh + 3u) / 4u) * 4u;
    auto ntex = device.create_image<float>(PixelStorage::BYTE4, nw, nh);
    auto nbuf = device.create_buffer<luisa::uint>(npot_u32s);
    stream << ntex.copy_from(npot.data()) << synchronize();
    require(ext->compress_bc7(stream, ntex.view(0u), nbuf, 1.0f) ==
                TexCompressExt::Result::Success,
            "TexCompressExt NPOT compress_bc7 failed");
    std::vector<luisa::uint> nout(npot_u32s);
    stream << nbuf.copy_to(luisa::span{nout}) << synchronize();
    const auto ndec = f4_to_rgba8(
        hw_decode_image(device, stream, PixelStorage::BC7, nw, nh, nout.data()));
    int interior_diff = 0;
    int edge_diff = 0;
    for (std::uint32_t y = 0u; y < nh; ++y) {
      for (std::uint32_t x = 0u; x < nw; ++x) {
        const std::size_t off = (y * nw + x) * 4u;
        for (std::size_t c = 0u; c < 4u; ++c) {
          const int d = std::abs(static_cast<int>(npot[off + c]) -
                                 static_cast<int>(ndec[off + c]));
          if (x < 12u && y < 8u) {
            interior_diff = std::max(interior_diff, d);
          } else {
            edge_diff = std::max(edge_diff, d);
          }
        }
      }
    }
    const int ndiff = std::max(interior_diff, edge_diff);
    const double npsnr =
        compute_psnr_rgba(npot.data(), ndec.data(), npot.size(), false);
    std::cout << "  TexCompressExt BC7 NPOT(13x11) PSNR=" << npsnr
              << " dB, max_diff=" << ndiff << " (interior=" << interior_diff
              << ", edge=" << edge_diff << ")\n";
    require(npsnr >= 30.0 && ndiff <= 24 && edge_diff <= interior_diff + 2,
            "TexCompressExt NPOT 13x11 quality check failed");
  }

  // Non-zero output offset_bytes + sentinel preservation for BC7 and BC6H.
  {
    constexpr std::uint32_t kPadU32 = 16u;
    constexpr luisa::uint kSentinel = 0xdeadbeefu;
    std::vector<luisa::uint> init_vals(kPadU32 + blk_u32s + kPadU32, kSentinel);
    auto dst_buf = device.create_buffer<luisa::uint>(init_vals.size());
    auto tex_b4 = device.create_image<float>(PixelStorage::BYTE4, P, P);
    const auto hdr = make_hdr_tile(P);
    auto tex_f4 = device.create_image<float>(PixelStorage::FLOAT4, P, P);
    stream << dst_buf.copy_from(init_vals.data())
           << tex_b4.copy_from(src_syn.data())
           << tex_f4.copy_from(hdr.data()) << synchronize();

    require(ext->compress_bc7(stream, tex_b4.view(0u),
                              dst_buf.view(kPadU32, blk_u32s), 1.0f) ==
                TexCompressExt::Result::Success,
            "TexCompressExt offset BC7 failed");
    std::vector<luisa::uint> out(init_vals.size(), 0u);
    stream << dst_buf.copy_to(luisa::span{out}) << synchronize();
    for (std::uint32_t i = 0u; i < kPadU32; ++i) {
      require(out[i] == kSentinel && out[kPadU32 + blk_u32s + i] == kSentinel,
              "TexCompressExt offset BC7 overwrote sentinel region");
    }
    const auto dec7 = f4_to_rgba8(hw_decode_image(
        device, stream, PixelStorage::BC7, P, P, out.data() + kPadU32));
    require(compute_psnr_rgba(src_syn.data(), dec7.data(), src_syn.size(), false) >= 38.0,
            "TexCompressExt offset BC7 PSNR below 38 dB");

    stream << dst_buf.copy_from(init_vals.data()) << synchronize();
    require(ext->compress_bc6h(stream, tex_f4.view(0u),
                               dst_buf.view(kPadU32, blk_u32s)) ==
                TexCompressExt::Result::Success,
            "TexCompressExt offset BC6H failed");
    stream << dst_buf.copy_to(luisa::span{out}) << synchronize();
    for (std::uint32_t i = 0u; i < kPadU32; ++i) {
      require(out[i] == kSentinel && out[kPadU32 + blk_u32s + i] == kSentinel,
              "TexCompressExt offset BC6H overwrote sentinel region");
    }
    const auto dec6 = hw_decode_image(
        device, stream, PixelStorage::BC6, P, P, out.data() + kPadU32);
    double rel_sum = 0.0;
    std::size_t n = 0u;
    for (std::size_t i = 0u; i < static_cast<std::size_t>(P) * P; ++i) {
      const float dv[3]{dec6[i].x, dec6[i].y, dec6[i].z};
      for (int c = 0; c < 3; ++c) {
        require(std::isfinite(dv[c]) && dv[c] >= 0.0f, "TexCompressExt BC6H invalid");
        rel_sum += std::abs(dv[c] - hdr[i * 4u + c]) / std::max(hdr[i * 4u + c], 0.1f);
        ++n;
      }
    }
    const double mean_rel = rel_sum / static_cast<double>(n);
    std::cout << "  TexCompressExt offset BC6H mean_rel_err=" << mean_rel << '\n';
    require(mean_rel < 0.05, "TexCompressExt offset BC6H rel err >= 5%");
  }
}

void test_bc_sampling_all_formats(Device &device, Stream &stream) {
  std::cout << "[VT BC Sampling] All BC1..BC7 + BC7_SRGB, NPOT(13x11), mip, "
               "bilinear, direct/bindless on backend: "
            << device.backend_name() << "...\n";

  // Helper: crop top-left (bx_n x by_n) blocks from a (16x16) block array.
  auto crop_blocks = [](const std::uint8_t *src, std::uint32_t src_bpr,
                        std::uint32_t dst_bpr, std::uint32_t dst_rows,
                        std::size_t blk_bytes) {
    std::vector<std::uint8_t> out(
        static_cast<std::size_t>(dst_bpr) * dst_rows * blk_bytes, 0u);
    for (std::uint32_t by = 0u; by < dst_rows; ++by) {
      std::memcpy(
          out.data() + static_cast<std::size_t>(by) * dst_bpr * blk_bytes,
          src + static_cast<std::size_t>(by) * src_bpr * blk_bytes,
          static_cast<std::size_t>(dst_bpr) * blk_bytes);
    }
    return out;
  };

  auto decode_bc_u8_grid = [](VtTileFormat fmt, std::uint32_t W, std::uint32_t H,
                              const std::uint8_t *enc) {
    const std::uint32_t bpr = (W + 3u) / 4u;
    const std::uint32_t rows = (H + 3u) / 4u;
    const std::size_t blk_bytes =
        (fmt == VtTileFormat::bc1 || fmt == VtTileFormat::bc4) ? 8u : 16u;
    std::vector<luisa::float4> ref(static_cast<std::size_t>(W) * H);
    std::uint8_t blk[64]{};
    for (std::uint32_t by = 0u; by < rows; ++by) {
      for (std::uint32_t bx = 0u; bx < bpr; ++bx) {
        const std::uint8_t *sb =
            enc + (static_cast<std::size_t>(by) * bpr + bx) * blk_bytes;
        switch (fmt) {
          case VtTileFormat::bc1: vt_decode_bc1_block(sb, blk); break;
          case VtTileFormat::bc3: vt_decode_bc3_block(sb, blk); break;
          case VtTileFormat::bc4: vt_decode_bc4_block(sb, blk); break;
          case VtTileFormat::bc5: vt_decode_bc5_block(sb, blk); break;
          case VtTileFormat::bc7: vt_decode_bc7_block(sb, blk); break;
          default: break;
        }
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          const std::uint32_t py = by * 4u + r;
          if (py >= H) { continue; }
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::uint32_t px = bx * 4u + c;
            if (px >= W) { continue; }
            const std::size_t bi = (r * 4u + c) * 4u;
            const std::size_t pi = static_cast<std::size_t>(py) * W + px;
            constexpr float inv255 = 1.0f / 255.0f;
            if (fmt == VtTileFormat::bc4) {
              ref[pi] = luisa::make_float4(blk[bi] * inv255, 0.0f, 0.0f, 1.0f);
            } else if (fmt == VtTileFormat::bc5) {
              ref[pi] = luisa::make_float4(blk[bi] * inv255, blk[bi + 1u] * inv255, 0.0f, 1.0f);
            } else {
              ref[pi] = luisa::make_float4(
                  blk[bi + 0u] * inv255, blk[bi + 1u] * inv255,
                  blk[bi + 2u] * inv255, blk[bi + 3u] * inv255);
            }
          }
        }
      }
    }
    return ref;
  };

  auto check_f4_parity = [](const std::vector<luisa::float4> &ref,
                            const std::vector<luisa::float4> &act,
                            float tol, std::string_view label) {
    require(ref.size() == act.size(), std::string{label} + " size mismatch");
    float max_err = 0.0f;
    for (std::size_t i = 0u; i < ref.size(); ++i) {
      const float r[4]{ref[i].x, ref[i].y, ref[i].z, ref[i].w};
      const float a[4]{act[i].x, act[i].y, act[i].z, act[i].w};
      for (int c = 0; c < 4; ++c) {
        require(std::isfinite(a[c]), std::string{label} + " non-finite output");
        max_err = std::max(max_err, std::abs(r[c] - a[c]));
      }
    }
    if (max_err > tol) {
      std::cerr << "  " << label << " max_err=" << max_err << " tol=" << tol << '\n';
    }
    require(max_err <= tol, std::string{label} + " exceeded tolerance");
  };

  auto sample_dyn = [&](PixelStorage storage, std::uint32_t W, std::uint32_t H,
                        const void *enc, SamplerFilter filter) {
    auto tex = device.create_image<float>(storage, W, H);
    auto heap = device.create_bindless_array(2u);
    heap.emplace_on_update(0u, tex, Sampler::linear_point_edge());
    auto out_buf = device.create_buffer<luisa::float4>(static_cast<std::size_t>(W) * H);
    auto k = device.compile<1>(
        [W, H, filter](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
          const auto tid = dispatch_x();
          $if(tid >= W * H) { $return(); };
          const auto y = tid / W;
          const auto x = tid - y * W;
          const auto uv = make_float2(
              (cast<float>(x) + 0.5f) / static_cast<float>(W),
              (cast<float>(y) + 0.5f) / static_cast<float>(H));
          dst.write(tid, h.tex2d(0u).sample(uv, filter, SamplerAddress::EDGE));
        });
    std::vector<luisa::float4> px(static_cast<std::size_t>(W) * H);
    stream << tex.copy_from(enc) << heap.update()
           << k(heap, out_buf).dispatch(W * H)
           << out_buf.copy_to(luisa::span{px}) << synchronize();
    return px;
  };

  auto read_direct = [&](PixelStorage storage, std::uint32_t W, std::uint32_t H,
                         const void *enc) {
    auto tex = device.create_image<float>(storage, W, H);
    auto out_buf = device.create_buffer<luisa::float4>(static_cast<std::size_t>(W) * H);
    auto k = device.compile<1>(
        [W, H](ImageVar<float> img, BufferVar<luisa::float4> dst) noexcept {
          const auto tid = dispatch_x();
          $if(tid >= W * H) { $return(); };
          const auto y = tid / W;
          const auto x = tid - y * W;
          dst.write(tid, img.read(make_uint2(x, y)));
        });
    std::vector<luisa::float4> px(static_cast<std::size_t>(W) * H);
    stream << tex.copy_from(enc)
           << k(tex, out_buf).dispatch(W * H)
           << out_buf.copy_to(luisa::span{px}) << synchronize();
    return px;
  };

  auto read_bindless = [&](PixelStorage storage, std::uint32_t W, std::uint32_t H,
                           const void *enc) {
    auto tex = device.create_image<float>(storage, W, H);
    auto heap = device.create_bindless_array(2u);
    heap.emplace_on_update(0u, tex, Sampler::point_edge());
    auto out_buf = device.create_buffer<luisa::float4>(static_cast<std::size_t>(W) * H);
    auto k = device.compile<1>(
        [W, H](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
          const auto tid = dispatch_x();
          $if(tid >= W * H) { $return(); };
          const auto y = tid / W;
          const auto x = tid - y * W;
          dst.write(tid, h.tex2d(0u).read(make_uint2(x, y)));
        });
    std::vector<luisa::float4> px(static_cast<std::size_t>(W) * H);
    stream << tex.copy_from(enc) << heap.update()
           << k(heap, out_buf).dispatch(W * H)
           << out_buf.copy_to(luisa::span{px}) << synchronize();
    return px;
  };

  constexpr std::uint32_t W = 13u, H = 11u;
  constexpr std::uint32_t BPR = (W + 3u) / 4u, BROWS = (H + 3u) / 4u;
  constexpr std::uint32_t P16 = 16u;
  const auto mix16 = make_mixed_rgba_tile(P16);
  auto opaque16 = mix16;
  for (std::size_t i = 3u; i < opaque16.size(); i += 4u) { opaque16[i] = 255u; }

  // 1. BC1 NPOT (13x11)
  {
    std::vector<std::uint8_t> enc16(vt_tile_format_bytes(P16, VtTileFormat::bc1));
    vt_encode_bc1_tile(opaque16.data(), P16, enc16);
    const auto enc = crop_blocks(enc16.data(), 4u, BPR, BROWS, 8u);
    const auto ref = decode_bc_u8_grid(VtTileFormat::bc1, W, H, enc.data());
    check_f4_parity(ref, hw_decode_image(device, stream, PixelStorage::BC1, W, H, enc.data()), 1.5f / 255.0f, "NPOT BC1 bindless sample");
    check_f4_parity(ref, sample_dyn(PixelStorage::BC1, W, H, enc.data(), SamplerFilter::POINT), 1.5f / 255.0f, "NPOT BC1 dynamic sampler");
  }

  // 2. BC2 NPOT (13x11): construct explicit 4-bit alpha + 4-color BC1 RGB blocks
  {
    std::vector<std::uint8_t> enc(static_cast<std::size_t>(BPR) * BROWS * 16u, 0u);
    std::vector<luisa::float4> ref(static_cast<std::size_t>(W) * H);
    for (std::uint32_t by = 0u; by < BROWS; ++by) {
      for (std::uint32_t bx = 0u; bx < BPR; ++bx) {
        std::uint8_t *blk = enc.data() + (static_cast<std::size_t>(by) * BPR + bx) * 16u;
        const std::uint8_t r0 = static_cast<std::uint8_t>((bx * 7u + by * 5u + 3u) & 31u);
        const std::uint8_t g0 = static_cast<std::uint8_t>((bx * 9u + by * 11u + 7u) & 63u);
        const std::uint8_t b0 = static_cast<std::uint8_t>((bx * 13u + by * 3u + 11u) & 31u);
        const std::uint8_t r1 = static_cast<std::uint8_t>(31u - r0);
        const std::uint8_t g1 = static_cast<std::uint8_t>(63u - g0);
        const std::uint8_t b1 = static_cast<std::uint8_t>(31u - b0);
        const std::uint16_t c0 = static_cast<std::uint16_t>((r0 << 11u) | (g0 << 5u) | b0);
        const std::uint16_t c1 = static_cast<std::uint16_t>((r1 << 11u) | (g1 << 5u) | b1);
        const std::uint32_t rgb_bits = 0xE4E4E4E4u;
        for (std::uint32_t p = 0u; p < 16u; ++p) {
          const std::uint8_t a4 = static_cast<std::uint8_t>((p + bx + by * 3u) & 15u);
          blk[p >> 1u] |= static_cast<std::uint8_t>(a4 << ((p & 1u) * 4u));
        }
        blk[8u] = static_cast<std::uint8_t>(c0 & 0xffu);
        blk[9u] = static_cast<std::uint8_t>(c0 >> 8u);
        blk[10u] = static_cast<std::uint8_t>(c1 & 0xffu);
        blk[11u] = static_cast<std::uint8_t>(c1 >> 8u);
        std::memcpy(blk + 12u, &rgb_bits, 4u);
        auto q5 = [](std::uint8_t v) { return static_cast<std::uint32_t>((v << 3u) | (v >> 2u)); };
        auto q6 = [](std::uint8_t v) { return static_cast<std::uint32_t>((v << 2u) | (v >> 4u)); };
        const std::uint32_t er0 = q5(r0), eg0 = q6(g0), eb0 = q5(b0);
        const std::uint32_t er1 = q5(r1), eg1 = q6(g1), eb1 = q5(b1);
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          const std::uint32_t py = by * 4u + r;
          if (py >= H) { continue; }
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::uint32_t px = bx * 4u + c;
            if (px >= W) { continue; }
            const std::uint32_t p = r * 4u + c;
            const std::uint32_t idx = (rgb_bits >> (2u * p)) & 3u;
            auto interp = [idx](std::uint32_t v0, std::uint32_t v1) {
              if (idx == 0u) return v0;
              if (idx == 1u) return v1;
              if (idx == 2u) return (2u * v0 + v1 + 1u) / 3u;
              return (v0 + 2u * v1 + 1u) / 3u;
            };
            const std::uint8_t a4 = (blk[p >> 1u] >> ((p & 1u) * 4u)) & 0xfu;
            ref[static_cast<std::size_t>(py) * W + px] = luisa::make_float4(
                static_cast<float>(interp(er0, er1)) / 255.0f,
                static_cast<float>(interp(eg0, eg1)) / 255.0f,
                static_cast<float>(interp(eb0, eb1)) / 255.0f,
                static_cast<float>(a4) / 15.0f);
          }
        }
      }
    }
    check_f4_parity(ref, hw_decode_image(device, stream, PixelStorage::BC2, W, H, enc.data()), 3.0f / 255.0f, "NPOT BC2 bindless sample");
  }

  // 3. BC3, BC4, BC5, BC7, BC7_SRGB NPOT (13x11)
  for (const auto [fmt, ps, name] : {
           std::tuple{VtTileFormat::bc3, PixelStorage::BC3, std::string_view{"NPOT BC3"}},
           std::tuple{VtTileFormat::bc4, PixelStorage::BC4, std::string_view{"NPOT BC4"}},
           std::tuple{VtTileFormat::bc5, PixelStorage::BC5, std::string_view{"NPOT BC5"}},
           std::tuple{VtTileFormat::bc7, PixelStorage::BC7, std::string_view{"NPOT BC7"}},
       }) {
    const std::size_t blk_bytes = (fmt == VtTileFormat::bc4) ? 8u : 16u;
    std::vector<std::uint8_t> enc16(vt_tile_format_bytes(P16, fmt));
    if (fmt == VtTileFormat::bc3) vt_encode_bc3_tile(mix16.data(), P16, enc16);
    else if (fmt == VtTileFormat::bc4) vt_encode_bc4_tile(mix16.data(), P16, enc16);
    else if (fmt == VtTileFormat::bc5) vt_encode_bc5_tile(mix16.data(), P16, enc16);
    else vt_encode_bc7_tile(mix16.data(), P16, enc16);
    const auto enc = crop_blocks(enc16.data(), 4u, BPR, BROWS, blk_bytes);
    const auto ref = decode_bc_u8_grid(fmt, W, H, enc.data());
    check_f4_parity(ref, hw_decode_image(device, stream, ps, W, H, enc.data()), 2.0f / 255.0f, std::string{name} + " bindless sample");
    if (fmt == VtTileFormat::bc4 || fmt == VtTileFormat::bc5 || fmt == VtTileFormat::bc7) {
      check_f4_parity(ref, sample_dyn(ps, W, H, enc.data(), SamplerFilter::POINT), 2.0f / 255.0f, std::string{name} + " dynamic sampler");
    }
    if (fmt == VtTileFormat::bc4 || fmt == VtTileFormat::bc7) {
      check_f4_parity(ref, read_direct(ps, W, H, enc.data()), 2.0f / 255.0f, std::string{name} + " direct read");
      check_f4_parity(ref, read_bindless(ps, W, H, enc.data()), 2.0f / 255.0f, std::string{name} + " bindless read");
    }
    if (fmt == VtTileFormat::bc7) {
      // BC7_SRGB parity
      auto srgb_ref = ref;
      auto to_lin = [](float c) noexcept {
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
      };
      for (auto &px : srgb_ref) {
        px.x = to_lin(px.x);
        px.y = to_lin(px.y);
        px.z = to_lin(px.z);
      }
      check_f4_parity(srgb_ref, hw_decode_image(device, stream, PixelStorage::BC7_SRGB, W, H, enc.data()), 2.0f / 255.0f, "NPOT BC7_SRGB bindless sample");
      check_f4_parity(srgb_ref, sample_dyn(PixelStorage::BC7_SRGB, W, H, enc.data(), SamplerFilter::POINT), 2.0f / 255.0f, "NPOT BC7_SRGB dynamic sampler");

      // Bilinear check between (0,0), (1,0), (0,1), (1,1) at uv = (1/W, 1/H)
      auto tex_lin = device.create_image<float>(PixelStorage::BC7, W, H);
      auto heap_lin = device.create_bindless_array(2u);
      heap_lin.emplace_on_update(0u, tex_lin, Sampler::linear_point_edge());
      auto out_lin = device.create_buffer<luisa::float4>(1u);
      auto k_lin = device.compile<1>(
          [W, H](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
            dst.write(0u, h.tex2d(0u).sample(make_float2(1.0f / static_cast<float>(W), 1.0f / static_cast<float>(H))));
          });
      luisa::float4 got_lin{};
      stream << tex_lin.copy_from(enc.data()) << heap_lin.update()
             << k_lin(heap_lin, out_lin).dispatch(1u)
             << out_lin.copy_to(&got_lin) << synchronize();
      const auto &p00 = ref[0u];
      const auto &p10 = ref[1u];
      const auto &p01 = ref[W];
      const auto &p11 = ref[W + 1u];
      const luisa::float4 exp_lin = (p00 + p10 + p01 + p11) * 0.25f;
      check_f4_parity({exp_lin}, {got_lin}, 4.0f / 255.0f, "NPOT BC7 bilinear sample");
    }
  }

  // 4. BC6H NPOT (13x11) exact decode/read/sample parity against CPU BC6H reference
  {
    std::vector<float> hdr16(static_cast<std::size_t>(P16) * P16 * 4u, 1.0f);
    for (std::uint32_t y = 0u; y < P16; ++y) {
      for (std::uint32_t x = 0u; x < P16; ++x) {
        const std::size_t o = (static_cast<std::size_t>(y) * P16 + x) * 4u;
        hdr16[o + 0u] = 0.25f + 0.50f * static_cast<float>(x);
        hdr16[o + 1u] = 0.25f + 0.35f * static_cast<float>(y);
        hdr16[o + 2u] = 0.25f + 0.20f * static_cast<float>(x + y);
        hdr16[o + 3u] = 1.0f;
      }
    }
    std::vector<std::uint8_t> enc16(vt_tile_format_bytes(P16, VtTileFormat::bc6h));
    vt_encode_bc6h_tile(hdr16.data(), P16, enc16);
    const auto enc = crop_blocks(enc16.data(), 4u, BPR, BROWS, 16u);
    std::vector<luisa::float4> ref(static_cast<std::size_t>(W) * H);
    float blk_f[64]{};
    for (std::uint32_t by = 0u; by < BROWS; ++by) {
      for (std::uint32_t bx = 0u; bx < BPR; ++bx) {
        const std::uint8_t *sb =
            enc.data() + (static_cast<std::size_t>(by) * BPR + bx) * 16u;
        vt_decode_bc6h_block(sb, blk_f);
        for (std::uint32_t r = 0u; r < 4u; ++r) {
          const std::uint32_t py = by * 4u + r;
          if (py >= H) { continue; }
          for (std::uint32_t c = 0u; c < 4u; ++c) {
            const std::uint32_t px = bx * 4u + c;
            if (px >= W) { continue; }
            const std::size_t bi = (r * 4u + c) * 4u;
            ref[static_cast<std::size_t>(py) * W + px] =
                luisa::make_float4(blk_f[bi], blk_f[bi + 1u], blk_f[bi + 2u], 1.0f);
          }
        }
      }
    }
    check_f4_parity(ref, hw_decode_image(device, stream, PixelStorage::BC6, W, H, enc.data()), 1e-3f, "NPOT BC6H bindless sample");
    check_f4_parity(ref, sample_dyn(PixelStorage::BC6, W, H, enc.data(), SamplerFilter::POINT), 1e-3f, "NPOT BC6H dynamic sampler");
    check_f4_parity(ref, read_direct(PixelStorage::BC6, W, H, enc.data()), 1e-3f, "NPOT BC6H direct read");
    check_f4_parity(ref, read_bindless(PixelStorage::BC6, W, H, enc.data()), 1e-3f, "NPOT BC6H bindless read");
  }

  // 5. Multi-mip (32x32, 4 mips) on BC7 and BC1
  for (const auto [fmt, ps, name] : {
           std::tuple{VtTileFormat::bc7, PixelStorage::BC7, std::string_view{"Mip4 BC7"}},
           std::tuple{VtTileFormat::bc1, PixelStorage::BC1, std::string_view{"Mip4 BC1"}}}) {
    auto tex = device.create_image<float>(ps, 32u, 32u, 4u);
    auto heap = device.create_bindless_array(2u);
    heap.emplace_on_update(0u, tex, Sampler::point_edge());
    std::vector<std::vector<luisa::float4>> mip_refs(4u);
    CommandList cl;
    std::vector<std::vector<std::uint8_t>> mip_encs(4u);
    for (std::uint32_t m = 0u; m < 4u; ++m) {
      const std::uint32_t ext = 32u >> m;
      std::vector<std::uint8_t> rgba(static_cast<std::size_t>(ext) * ext * 4u);
      for (std::uint32_t y = 0u; y < ext; ++y) {
        for (std::uint32_t x = 0u; x < ext; ++x) {
          const std::size_t o = (static_cast<std::size_t>(y) * ext + x) * 4u;
          rgba[o + 0u] = static_cast<std::uint8_t>((m * 53u + x * 7u) & 255u);
          rgba[o + 1u] = static_cast<std::uint8_t>((m * 97u + y * 11u) & 255u);
          rgba[o + 2u] = static_cast<std::uint8_t>((200u - m * 41u) & 255u);
          rgba[o + 3u] = (fmt == VtTileFormat::bc1) ? 255u : static_cast<std::uint8_t>((240u - m * 37u) & 255u);
        }
      }
      mip_encs[m].resize(vt_tile_format_bytes(ext, fmt));
      if (fmt == VtTileFormat::bc7) vt_encode_bc7_tile(rgba.data(), ext, mip_encs[m]);
      else vt_encode_bc1_tile(rgba.data(), ext, mip_encs[m]);
      mip_refs[m] = decode_bc_u8_grid(fmt, ext, ext, mip_encs[m].data());
      cl << tex.view(m).copy_from(mip_encs[m].data());
    }
    cl << heap.update();
    stream << cl.commit();
    for (std::uint32_t m = 0u; m < 4u; ++m) {
      const std::uint32_t ext = 32u >> m;
      auto out_buf = device.create_buffer<luisa::float4>(static_cast<std::size_t>(ext) * ext);
      auto k = device.compile<1>(
          [ext, m](BindlessVar h, BufferVar<luisa::float4> dst) noexcept {
            const auto tid = dispatch_x();
            $if(tid >= ext * ext) { $return(); };
            const auto y = tid / ext;
            const auto x = tid - y * ext;
            const auto uv = make_float2(
                (cast<float>(x) + 0.5f) / static_cast<float>(ext),
                (cast<float>(y) + 0.5f) / static_cast<float>(ext));
            dst.write(tid, h.tex2d(0u).sample(uv, static_cast<float>(m)));
          });
      std::vector<luisa::float4> got(static_cast<std::size_t>(ext) * ext);
      stream << k(heap, out_buf).dispatch(ext * ext)
             << out_buf.copy_to(luisa::span{got}) << synchronize();
      check_f4_parity(mip_refs[m], got, 2.0f / 255.0f, std::string{name} + " mip=" + std::to_string(m));
    }
  }
}

void test_gpu_bc_compressor(Device &device, Stream &stream) {
  test_bc_sampling_all_formats(device, stream);
  const std::string_view backend{device.backend_name()};
  std::cout << "[VT BC Test 5] GPU BC1..BC7 compressor on backend: " << backend
            << "...\n";
  VtGpuBcCompressor default_bc{device};
  require(default_bc.available(), "Expected VtGpuBcCompressor to be available");
  require(std::string_view{default_bc.backend_name()} == "dsl",
          "Expected default auto backend to be pure Luisa DSL");

  constexpr std::uint32_t P = 64u;
  VtGpuBcCompressor dsl_bc{device, VtGpuBcMode::force_dsl, VtGpuBcQuality::balanced};
  require(dsl_bc.available() && std::string_view{dsl_bc.backend_name()} == "dsl",
          "Expected force_dsl backend to be dsl");
  require(dsl_bc.quality() == VtGpuBcQuality::balanced,
          "Expected default force_dsl quality to be balanced");
  verify_bc7_compressor_mode(device, stream, dsl_bc, P);
  verify_bc6h_compressor_mode(device, stream, dsl_bc, P);
  verify_dsl_legacy_bc_formats(device, stream, dsl_bc, P);
  require(dsl_bc.tiles_encoded() == 18u,
          "Expected tiles_encoded == 18 across BC7 + partition + BC6H + BC1/3/4/5");

  for (const VtGpuBcQuality q : {VtGpuBcQuality::fast, VtGpuBcQuality::high}) {
    dsl_bc.set_quality(q);
    require(dsl_bc.quality() == q, "Expected set_quality to update quality");
    verify_bc7_compressor_mode(device, stream, dsl_bc, P);
    verify_bc6h_compressor_mode(device, stream, dsl_bc, P);
  }
  require(dsl_bc.tiles_encoded() == 38u,
          "Expected tiles_encoded == 38 after fast and high quality passes");
  dsl_bc.set_quality(VtGpuBcQuality::balanced);

  if (device.extension<TexCompressExt>() != nullptr) {
    VtGpuBcCompressor ext_bc{device, VtGpuBcMode::force_ext};
    require(ext_bc.available() &&
                std::string_view{ext_bc.backend_name()} == "tex_compress_ext",
            "Expected force_ext backend to be tex_compress_ext");
    verify_bc7_compressor_mode(device, stream, ext_bc, P);
    verify_bc6h_compressor_mode(device, stream, ext_bc, P);
    require(ext_bc.tiles_encoded() == 10u,
            "Expected ext tiles_encoded == 10 across BC7 + partition + BC6H");
  }
}

void run_gpu_bc_benchmark(Device &device, Stream &stream) {
  constexpr std::uint32_t P = 128u;
  constexpr std::array<std::uint32_t, 3u> kBatches{16u, 64u, 256u};
  constexpr int kWarmup = 2;
  constexpr int kIters = 5;

  const auto mix_rgba = make_mixed_rgba_tile(P);
  const auto hdr_rgba = make_hdr_tile(P);
  auto opaque_rgba = mix_rgba;
  for (std::size_t i = 3u; i < opaque_rgba.size(); i += 4u) {
    opaque_rgba[i] = 255u;
  }

  VtGpuBcCompressor dsl_bc{device, VtGpuBcMode::force_dsl};
  const std::array<VtTileFormat, 6u> all_fmts{
      VtTileFormat::bc7, VtTileFormat::bc6h, VtTileFormat::bc1,
      VtTileFormat::bc3, VtTileFormat::bc4, VtTileFormat::bc5};
  dsl_bc.warmup(all_fmts);

  const bool has_ext = (device.extension<TexCompressExt>() != nullptr);
  VtGpuBcCompressor ext_bc{device, VtGpuBcMode::force_ext};
  if (has_ext) {
    const std::array<VtTileFormat, 2u> ext_fmts{
        VtTileFormat::bc7, VtTileFormat::bc6h};
    ext_bc.warmup(ext_fmts);
  }

  auto measure_quality = [&](VtTileFormat fmt, const std::uint8_t *enc) {
    const bool hw_bc = supports_hw_bc_sampling(device);
    if (fmt == VtTileFormat::bc6h) {
      if (!hw_bc) {
        return 0.0;
      }
      const auto dec =
          hw_decode_image(device, stream, PixelStorage::BC6, P, P, enc);
      double rel_sum = 0.0;
      std::size_t n = 0u;
      for (std::size_t i = 0u; i < static_cast<std::size_t>(P) * P; ++i) {
        const float dv[3]{dec[i].x, dec[i].y, dec[i].z};
        for (int c = 0; c < 3; ++c) {
          rel_sum += std::abs(dv[c] - hdr_rgba[i * 4u + c]) /
                     std::max(hdr_rgba[i * 4u + c], 0.1f);
          ++n;
        }
      }
      return rel_sum / static_cast<double>(n);
    }
    PixelStorage ps = PixelStorage::BC7;
    const std::uint8_t *ref = mix_rgba.data();
    bool rgb_only = false;
    if (fmt == VtTileFormat::bc1) {
      ps = PixelStorage::BC1;
      ref = opaque_rgba.data();
      rgb_only = true;
    } else if (fmt == VtTileFormat::bc3) {
      ps = PixelStorage::BC3;
    } else if (fmt == VtTileFormat::bc4) {
      ps = PixelStorage::BC4;
    } else if (fmt == VtTileFormat::bc5) {
      ps = PixelStorage::BC5;
    }
    const auto dec =
        !hw_bc
            ? sw_decode_legacy_bc_rgba8(fmt, P, enc)
            : f4_to_rgba8(hw_decode_image(device, stream, ps, P, P, enc));
    if (fmt == VtTileFormat::bc4) {
      double s = 0.0;
      for (std::size_t i = 0u; i < mix_rgba.size(); i += 4u) {
        const double d = static_cast<double>(ref[i]) - dec[i];
        s += d * d;
      }
      const double mse = s / (static_cast<double>(P) * P);
      return mse <= 1e-12 ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / mse);
    }
    if (fmt == VtTileFormat::bc5) {
      double s = 0.0;
      for (std::size_t i = 0u; i < mix_rgba.size(); i += 4u) {
        const double dr = static_cast<double>(ref[i + 0u]) - dec[i + 0u];
        const double da = static_cast<double>(ref[i + 3u]) - dec[i + 1u];
        s += dr * dr + da * da;
      }
      const double mse = s / (static_cast<double>(P) * P * 2.0);
      return mse <= 1e-12 ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / mse);
    }
    return compute_psnr_rgba(ref, dec.data(), mix_rgba.size(), rgb_only);
  };

  auto fmt_name = [](VtTileFormat fmt) -> const char * {
    switch (fmt) {
      case VtTileFormat::bc1: return "bc1";
      case VtTileFormat::bc3: return "bc3";
      case VtTileFormat::bc4: return "bc4";
      case VtTileFormat::bc5: return "bc5";
      case VtTileFormat::bc6h: return "bc6h";
      case VtTileFormat::bc7: return "bc7";
      default: return "unknown";
    }
  };

  std::cout << std::fixed << std::setprecision(3);
  for (const VtTileFormat fmt : all_fmts) {
    const std::size_t tbytes = vt_tile_format_bytes(P, fmt);
    const std::uint8_t *u8_ptr =
        (fmt == VtTileFormat::bc1)
            ? opaque_rgba.data()
            : ((fmt == VtTileFormat::bc6h) ? nullptr : mix_rgba.data());
    const float *f_ptr =
        (fmt == VtTileFormat::bc6h) ? hdr_rgba.data() : nullptr;

    for (const std::uint32_t N : kBatches) {
      const double mpix = (static_cast<double>(N) * P * P) / 1e6;
      std::vector<std::uint8_t> dst(static_cast<std::size_t>(N) * tbytes, 0u);
      std::vector<VtGpuBcTileJob> jobs(N);
      for (std::uint32_t i = 0u; i < N; ++i) {
        jobs[i] = VtGpuBcTileJob{
            .unorm8_pixels = u8_ptr,
            .float_pixels = f_ptr,
            .width = P,
            .height = P,
            .physical_tile_size = P,
            .logical_tile_size = P,
            .format = fmt,
            .dst = dst.data() + static_cast<std::size_t>(i) * tbytes,
            .dst_bytes = tbytes};
      }

      // 1. CPU parallel
      {
        auto run_cpu = [&]() {
          psycles::parallel_for(N, [&](std::size_t idx) noexcept {
            vt_pack_tile_formatted(
                u8_ptr, f_ptr, P, P, 0u, 0u, 0u, P, P, 0u, fmt,
                std::span<std::uint8_t>{dst.data() + idx * tbytes, tbytes});
          });
        };
        for (int w = 0; w < kWarmup; ++w) run_cpu();
        std::vector<double> times;
        for (int it = 0; it < kIters; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          run_cpu();
          times.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count());
        }
        const double med = median_of(times);
        const double qual = measure_quality(fmt, dst.data());
        std::cout << "BENCH backend=" << device.backend_name()
                  << " fmt=" << fmt_name(fmt)
                  << " method=cpu_parallel tiles=" << N
                  << " total_ms=" << med << " gpu_ms=0.000"
                  << " mpix_s=" << (mpix / (med * 1e-3))
                  << " gpu_mpix_s=0.000 quality=" << qual << std::endl;
      }

      // 2. GPU DSL (all 3 quality levels for BC7 and BC6H)
      if (fmt == VtTileFormat::bc7 || fmt == VtTileFormat::bc6h) {
        for (const VtGpuBcQuality q :
             {VtGpuBcQuality::fast, VtGpuBcQuality::balanced, VtGpuBcQuality::high}) {
          dsl_bc.set_quality(q);
          for (int w = 0; w < kWarmup; ++w) {
            require(dsl_bc.compress_batch(jobs), "DSL bench compress_batch failed");
          }
          std::vector<double> totals, gpus;
          for (int it = 0; it < kIters; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            require(dsl_bc.compress_batch(jobs), "DSL bench compress_batch failed");
            totals.push_back(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count());
            gpus.push_back(dsl_bc.last_gpu_kernel_ms());
          }
          const double med_tot = median_of(totals);
          const double med_gpu = median_of(gpus);
          const double qual = measure_quality(fmt, dst.data());
          std::cout << "BENCH backend=" << device.backend_name()
                    << " fmt=" << fmt_name(fmt)
                    << " method=gpu_dsl_" << dsl_bc.quality_name()
                    << " tiles=" << N
                    << " total_ms=" << med_tot << " gpu_ms=" << med_gpu
                    << " mpix_s=" << (mpix / (med_tot * 1e-3))
                    << " gpu_mpix_s=" << (mpix / (std::max(med_gpu, 1e-6) * 1e-3))
                    << " quality=" << qual << std::endl;
        }
        dsl_bc.set_quality(VtGpuBcQuality::balanced);
      } else {
        for (int w = 0; w < kWarmup; ++w) {
          require(dsl_bc.compress_batch(jobs), "DSL bench compress_batch failed");
        }
        std::vector<double> totals, gpus;
        for (int it = 0; it < kIters; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          require(dsl_bc.compress_batch(jobs), "DSL bench compress_batch failed");
          totals.push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count());
          gpus.push_back(dsl_bc.last_gpu_kernel_ms());
        }
        const double med_tot = median_of(totals);
        const double med_gpu = median_of(gpus);
        const double qual = measure_quality(fmt, dst.data());
        std::cout << "BENCH backend=" << device.backend_name()
                  << " fmt=" << fmt_name(fmt)
                  << " method=gpu_dsl tiles=" << N
                  << " total_ms=" << med_tot << " gpu_ms=" << med_gpu
                  << " mpix_s=" << (mpix / (med_tot * 1e-3))
                  << " gpu_mpix_s=" << (mpix / (std::max(med_gpu, 1e-6) * 1e-3))
                  << " quality=" << qual << std::endl;
      }

      // 3. GPU TexCompressExt (BC7 / BC6H)
      if (has_ext && (fmt == VtTileFormat::bc7 || fmt == VtTileFormat::bc6h)) {
        for (int w = 0; w < kWarmup; ++w) {
          require(ext_bc.compress_batch(jobs), "EXT bench compress_batch failed");
        }
        std::vector<double> totals, gpus;
        for (int it = 0; it < kIters; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          require(ext_bc.compress_batch(jobs), "EXT bench compress_batch failed");
          totals.push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count());
          gpus.push_back(ext_bc.last_gpu_kernel_ms());
        }
        const double med_tot = median_of(totals);
        const double med_gpu = median_of(gpus);
        const double qual = measure_quality(fmt, dst.data());
        std::cout << "BENCH backend=" << device.backend_name()
                  << " fmt=" << fmt_name(fmt)
                  << " method=gpu_ext tiles=" << N
                  << " total_ms=" << med_tot << " gpu_ms=" << med_gpu
                  << " mpix_s=" << (mpix / (med_tot * 1e-3))
                  << " gpu_mpix_s=" << (mpix / (std::max(med_gpu, 1e-6) * 1e-3))
                  << " quality=" << qual << std::endl;
      }
    }
  }
}

} // namespace psycles::vt_bc_test
