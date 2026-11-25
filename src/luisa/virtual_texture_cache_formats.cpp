#include "virtual_texture_cache_formats.h"

#include "virtual_texture_cache_bc.h"
#include "virtual_texture_cache_prepack.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <luisa/core/basic_types.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::array<std::uint8_t, 16u> kBc7ZeroBlock{
    0x40u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
inline constexpr std::array<std::uint8_t, 16u> kBc6hZeroBlock{
    0x03u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

inline constexpr std::array<int, 16u> kBc6Weight4{
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

struct Bc6BitWriter {
  std::uint64_t lo{0ull};
  std::uint64_t hi{0ull};
  std::uint32_t bit_pos{0u};

  void write(std::uint32_t val, std::uint32_t bits) noexcept {
    const std::uint64_t mask = (bits >= 64u) ? ~0ull : ((1ull << bits) - 1ull);
    const std::uint64_t masked = static_cast<std::uint64_t>(val) & mask;
    if (bit_pos < 64u) {
      lo |= (masked << bit_pos);
      if (bit_pos + bits > 64u) {
        hi |= (masked >> (64u - bit_pos));
      }
    } else {
      hi |= (masked << (bit_pos - 64u));
    }
    bit_pos += bits;
  }

  void finish(std::uint8_t dst[16]) const noexcept {
    std::memcpy(dst, &lo, 8u);
    std::memcpy(dst + 8u, &hi, 8u);
  }
};

struct Bc6BitReader {
  std::uint64_t lo{0ull};
  std::uint64_t hi{0ull};
  std::uint32_t bit_pos{0u};

  explicit Bc6BitReader(const std::uint8_t src[16]) noexcept {
    std::memcpy(&lo, src, 8u);
    std::memcpy(&hi, src + 8u, 8u);
  }

  [[nodiscard]] std::uint32_t read(std::uint32_t bits) noexcept {
    std::uint64_t v = 0ull;
    if (bit_pos < 64u) {
      v = lo >> bit_pos;
      if (bit_pos + bits > 64u) {
        v |= hi << (64u - bit_pos);
      }
    } else {
      v = hi >> (bit_pos - 64u);
    }
    bit_pos += bits;
    return static_cast<std::uint32_t>(v & ((1ull << bits) - 1ull));
  }
};

[[nodiscard]] inline std::uint16_t float_to_uf16(float v) noexcept {
  if (!(v > 0.0f)) {
    return 0u;
  }
  if (v >= 65504.0f) {
    return 0x7bffu;
  }
  return vt_float_to_half_bits(v);
}

[[nodiscard]] inline int bc6h_unquantize_10u(int q) noexcept {
  if (q <= 0) {
    return 0;
  }
  if (q >= 1023) {
    return 0xffff;
  }
  return ((q << 16) + 0x8000) >> 10;
}

[[nodiscard]] inline int bc6h_finish_unquantize_u(int comp) noexcept {
  return (comp * 31) >> 6;
}

void encode_bc6h_mode11_block(
    const float src[64],
    std::uint8_t dst[16]) noexcept {
  std::array<std::uint16_t, 48u> h16{};
  std::array<float, 48u> hf{};
  std::array<float, 3u> mean{};
  std::array<float, 3u> cmin{31743.0f, 31743.0f, 31743.0f};
  std::array<float, 3u> cmax{};
  for (int i = 0; i < 16; ++i) {
    for (int ch = 0; ch < 3; ++ch) {
      const std::uint16_t hv = float_to_uf16(src[i * 4 + ch]);
      h16[i * 3 + ch] = hv;
      const float f = static_cast<float>(hv);
      hf[i * 3 + ch] = f;
      mean[ch] += f;
      cmin[ch] = std::min(cmin[ch], f);
      cmax[ch] = std::max(cmax[ch], f);
    }
  }

  std::array<int, 3u> ep0{};
  std::array<int, 3u> ep1{};
  std::array<std::uint8_t, 16u> idx{};

  if (cmin[0] == cmax[0] && cmin[1] == cmax[1] && cmin[2] == cmax[2]) {
    for (int ch = 0; ch < 3; ++ch) {
      const int target = static_cast<int>(h16[ch]);
      const int q = std::clamp(
          static_cast<int>(std::lround(static_cast<float>(target) * (1023.0f / 31743.0f))),
          0, 1023);
      ep0[ch] = q;
      ep1[ch] = q;
    }
    Bc6BitWriter w;
    w.write(0x03u, 5u);
    for (int ch = 0; ch < 3; ++ch) {
      w.write(static_cast<std::uint32_t>(ep0[ch]), 10u);
    }
    for (int ch = 0; ch < 3; ++ch) {
      w.write(static_cast<std::uint32_t>(ep1[ch]), 10u);
    }
    w.write(0u, 3u);
    for (int i = 1; i < 16; ++i) {
      w.write(0u, 4u);
    }
    w.finish(dst);
    return;
  }

  for (float &m : mean) {
    m *= (1.0f / 16.0f);
  }
  float cov[3][3]{};
  for (int i = 0; i < 16; ++i) {
    const std::array<float, 3u> d{
        hf[i * 3 + 0] - mean[0], hf[i * 3 + 1] - mean[1], hf[i * 3 + 2] - mean[2]};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        cov[r][c] += d[r] * d[c];
      }
    }
  }
  std::array<float, 3u> axis{
      cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]};
  const float norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  std::array<float, 3u> e0{mean};
  std::array<float, 3u> e1{mean};
  if (norm2 > 1e-6f) {
    const float inv0 = 1.0f / std::sqrt(norm2);
    for (float &a : axis) {
      a *= inv0;
    }
    for (int it = 0; it < 4; ++it) {
      std::array<float, 3u> nxt{};
      for (int r = 0; r < 3; ++r) {
        nxt[r] = cov[r][0] * axis[0] + cov[r][1] * axis[1] + cov[r][2] * axis[2];
      }
      const float n2 = nxt[0] * nxt[0] + nxt[1] * nxt[1] + nxt[2] * nxt[2];
      if (n2 <= 1e-8f) {
        break;
      }
      const float inv = 1.0f / std::sqrt(n2);
      for (int ch = 0; ch < 3; ++ch) {
        axis[ch] = nxt[ch] * inv;
      }
    }
    float tmin = 1e30f;
    float tmax = -1e30f;
    for (int i = 0; i < 16; ++i) {
      const float p = (hf[i * 3 + 0] - mean[0]) * axis[0] +
                      (hf[i * 3 + 1] - mean[1]) * axis[1] +
                      (hf[i * 3 + 2] - mean[2]) * axis[2];
      tmin = std::min(tmin, p);
      tmax = std::max(tmax, p);
    }
    for (int ch = 0; ch < 3; ++ch) {
      e0[ch] = std::clamp(mean[ch] + tmin * axis[ch], 0.0f, 31743.0f);
      e1[ch] = std::clamp(mean[ch] + tmax * axis[ch], 0.0f, 31743.0f);
    }
  }

  auto assign_float = [&]() noexcept {
    const std::array<float, 3u> diff{
        e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]};
    const float len2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    if (len2 <= 1e-6f) {
      idx.fill(0u);
      return;
    }
    const float inv_len2 = 1.0f / len2;
    for (int i = 0; i < 16; ++i) {
      const float dot = (hf[i * 3 + 0] - e0[0]) * diff[0] +
                        (hf[i * 3 + 1] - e0[1]) * diff[1] +
                        (hf[i * 3 + 2] - e0[2]) * diff[2];
      const float t = std::clamp(dot * inv_len2, 0.0f, 1.0f);
      int best_k = std::clamp(static_cast<int>(std::lround(t * 15.0f)), 0, 15);
      const int k_lo = std::max(0, best_k - 1);
      const int k_hi = std::min(15, best_k + 1);
      float best_d = 1e30f;
      for (int k = k_lo; k <= k_hi; ++k) {
        const float w = static_cast<float>(kBc6Weight4[k]) * (1.0f / 64.0f);
        float d2 = 0.0f;
        for (int ch = 0; ch < 3; ++ch) {
          const float dc = (e0[ch] + w * diff[ch]) - hf[i * 3 + ch];
          d2 += dc * dc;
        }
        if (d2 < best_d) {
          best_d = d2;
          best_k = k;
        }
      }
      idx[i] = static_cast<std::uint8_t>(best_k);
    }
  };

  assign_float();
  for (int it = 0; it < 2; ++it) {
    double a2 = 0.0;
    double ab = 0.0;
    double b2 = 0.0;
    std::array<double, 3u> rhs0{};
    std::array<double, 3u> rhs1{};
    for (int i = 0; i < 16; ++i) {
      const double b = static_cast<double>(kBc6Weight4[idx[i]]) * (1.0 / 64.0);
      const double a = 1.0 - b;
      a2 += a * a;
      ab += a * b;
      b2 += b * b;
      for (int ch = 0; ch < 3; ++ch) {
        const double p = static_cast<double>(hf[i * 3 + ch]);
        rhs0[ch] += a * p;
        rhs1[ch] += b * p;
      }
    }
    const double det = a2 * b2 - ab * ab;
    if (det > 1e-6) {
      const double inv_det = 1.0 / det;
      for (int ch = 0; ch < 3; ++ch) {
        e0[ch] = static_cast<float>(
            std::clamp((b2 * rhs0[ch] - ab * rhs1[ch]) * inv_det, 0.0, 31743.0));
        e1[ch] = static_cast<float>(
            std::clamp((-ab * rhs0[ch] + a2 * rhs1[ch]) * inv_det, 0.0, 31743.0));
      }
      assign_float();
    } else {
      break;
    }
  }

  std::array<int, 3u> ue0{};
  std::array<int, 3u> ue1{};
  for (int ch = 0; ch < 3; ++ch) {
    ep0[ch] = std::clamp(
        static_cast<int>(std::lround(e0[ch] * (1023.0f / 31743.0f))), 0, 1023);
    ep1[ch] = std::clamp(
        static_cast<int>(std::lround(e1[ch] * (1023.0f / 31743.0f))), 0, 1023);
    ue0[ch] = bc6h_unquantize_10u(ep0[ch]);
    ue1[ch] = bc6h_unquantize_10u(ep1[ch]);
  }
  std::array<std::array<int, 3u>, 16u> pal{};
  for (int k = 0; k < 16; ++k) {
    const int w = kBc6Weight4[k];
    for (int ch = 0; ch < 3; ++ch) {
      const int interp = (ue0[ch] * (64 - w) + ue1[ch] * w + 32) >> 6;
      pal[k][ch] = bc6h_finish_unquantize_u(interp);
    }
  }
  for (int i = 0; i < 16; ++i) {
    const int hr = static_cast<int>(h16[i * 3 + 0]);
    const int hg = static_cast<int>(h16[i * 3 + 1]);
    const int hb = static_cast<int>(h16[i * 3 + 2]);
    int best_k = 0;
    std::int64_t best_e = INT64_MAX;
    for (int k = 0; k < 16; ++k) {
      const std::int64_t dr = pal[k][0] - hr;
      const std::int64_t dg = pal[k][1] - hg;
      const std::int64_t db = pal[k][2] - hb;
      const std::int64_t e = dr * dr + dg * dg + db * db;
      if (e < best_e) {
        best_e = e;
        best_k = k;
      }
    }
    idx[i] = static_cast<std::uint8_t>(best_k);
  }
  if (idx[0] >= 8u) {
    std::swap(ep0, ep1);
    for (auto &x : idx) {
      x = static_cast<std::uint8_t>(15u - x);
    }
  }
  Bc6BitWriter w;
  w.write(0x03u, 5u);
  for (int ch = 0; ch < 3; ++ch) {
    w.write(static_cast<std::uint32_t>(ep0[ch]), 10u);
  }
  for (int ch = 0; ch < 3; ++ch) {
    w.write(static_cast<std::uint32_t>(ep1[ch]), 10u);
  }
  w.write(idx[0], 3u);
  for (int i = 1; i < 16; ++i) {
    w.write(idx[i], 4u);
  }
  w.finish(dst);
}

[[nodiscard]] std::uint16_t quantize_unorm16(float v) noexcept {
  const float clamped = std::clamp(v, 0.0f, 1.0f);
  return static_cast<std::uint16_t>(
      std::clamp(static_cast<int>(std::lround(clamped * 65535.0f)), 0, 65535));
}

} // namespace

std::uint16_t vt_float_to_half_bits(float v) noexcept {
  std::uint32_t f = 0u;
  std::memcpy(&f, &v, sizeof(f));
  const std::uint32_t sign = (f >> 16u) & 0x8000u;
  const std::uint32_t exp = (f >> 23u) & 0xffu;
  const std::uint32_t frac = f & 0x7fffffu;
  if (exp == 0xffu) {
    if (frac != 0u) {
      return static_cast<std::uint16_t>(sign | 0x7e00u | (frac >> 13u) | 1u);
    }
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  const int new_exp = static_cast<int>(exp) - 127 + 15;
  if (new_exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (new_exp <= 0) {
    if (new_exp < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t mant = frac | 0x800000u;
    const int shift = 14 - new_exp;
    std::uint32_t sub = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t half_way = 1u << (shift - 1);
    if (rem > half_way || (rem == half_way && (sub & 1u) != 0u)) {
      ++sub;
    }
    return static_cast<std::uint16_t>(sign | sub);
  }
  std::uint32_t out = sign | (static_cast<std::uint32_t>(new_exp) << 10u) |
                      (frac >> 13u);
  const std::uint32_t rem = frac & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (out & 1u) != 0u)) {
    ++out;
  }
  return static_cast<std::uint16_t>(out);
}

float vt_half_bits_to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16u;
  const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10u) & 0x1fu;
  const std::uint32_t frac = static_cast<std::uint32_t>(h) & 0x3ffu;
  std::uint32_t f = 0u;
  if (exp == 0u) {
    if (frac == 0u) {
      f = sign;
    } else {
      std::uint32_t e = 127u - 14u;
      std::uint32_t m = frac;
      while ((m & 0x400u) == 0u) {
        m <<= 1u;
        --e;
      }
      m &= 0x3ffu;
      f = sign | (e << 23u) | (m << 13u);
    }
  } else if (exp == 0x1fu) {
    f = sign | 0x7f800000u | (frac << 13u);
  } else {
    f = sign | ((exp - 15u + 127u) << 23u) | (frac << 13u);
  }
  float out = 0.0f;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

VtTileFormat vt_select_tile_format(
    LuisaVirtualTextureCompressionMode mode,
    const io::ProbedImageInfo &probed,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space,
    std::uint32_t extension,
    bool supports_batch_copy) noexcept {
  const bool is_float =
      (probed.storage == io::DecodedImageStorage::float32);
  if (!supports_batch_copy ||
      mode == LuisaVirtualTextureCompressionMode::off) {
    return is_float ? VtTileFormat::float4 : VtTileFormat::rgba8;
  }
  switch (mode) {
    case LuisaVirtualTextureCompressionMode::off:
      return is_float ? VtTileFormat::float4 : VtTileFormat::rgba8;
    case LuisaVirtualTextureCompressionMode::bc7:
      return is_float ? VtTileFormat::float4 : VtTileFormat::bc7;
    case LuisaVirtualTextureCompressionMode::bc3:
      return is_float ? VtTileFormat::float4 : VtTileFormat::bc3;
    case LuisaVirtualTextureCompressionMode::bc1:
      return is_float ? VtTileFormat::float4 : VtTileFormat::bc1;
    case LuisaVirtualTextureCompressionMode::auto_hq:
    case LuisaVirtualTextureCompressionMode::auto_compact:
      break;
  }

  const bool compact =
      (mode == LuisaVirtualTextureCompressionMode::auto_compact);
  const bool is_clip = (extension == 1u);
  const bool eff_alpha =
      is_clip ||
      (probed.has_alpha && alpha_type != contract::ImageAlphaType::ignore);

  if (!is_float) {
    if (probed.channels == 1u) {
      return eff_alpha ? VtTileFormat::bc5 : VtTileFormat::bc4;
    }
    if (probed.channels == 2u) {
      return eff_alpha ? VtTileFormat::bc5 : VtTileFormat::bc4;
    }
    if (compact) {
      return eff_alpha ? VtTileFormat::bc3 : VtTileFormat::bc1;
    }
    return VtTileFormat::bc7;
  }

  if (!probed.is_floating_point && probed.bit_depth <= 16u) {
    return (probed.channels == 1u && !eff_alpha) ? VtTileFormat::short1
                                                 : VtTileFormat::short4;
  }
  if (probed.is_floating_point && !eff_alpha &&
      color_space != contract::ImageColorSpace::data &&
      probed.channels >= 3u) {
    return VtTileFormat::bc6h;
  }
  if (probed.is_floating_point && probed.bit_depth <= 16u) {
    return (probed.channels == 1u && !eff_alpha) ? VtTileFormat::half1
                                                 : VtTileFormat::half4;
  }
  return (probed.channels == 1u && !eff_alpha) ? VtTileFormat::float1
                                               : VtTileFormat::float4;
}

void vt_fill_zero_tile(
    std::uint32_t physical_tile_size,
    VtTileFormat fmt,
    std::span<std::uint8_t> dst) noexcept {
  const std::uint32_t needed = vt_tile_format_bytes(physical_tile_size, fmt);
  if (dst.size() < needed) {
    return;
  }
  if (fmt == VtTileFormat::bc7) {
    for (std::size_t off = 0u; off < needed; off += 16u) {
      std::memcpy(dst.data() + off, kBc7ZeroBlock.data(), 16u);
    }
    return;
  }
  if (fmt == VtTileFormat::bc6h) {
    for (std::size_t off = 0u; off < needed; off += 16u) {
      std::memcpy(dst.data() + off, kBc6hZeroBlock.data(), 16u);
    }
    return;
  }
  std::memset(dst.data(), 0, needed);
}

void vt_encode_bc6h_tile(
    const float *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept {
  if (rgba == nullptr || (tile_size % 4u) != 0u) {
    return;
  }
  const std::uint32_t bx_count = tile_size / 4u;
  const std::size_t needed = static_cast<std::size_t>(bx_count) * bx_count * 16u;
  if (dst.size() < needed) {
    return;
  }
  float block[64];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        std::memcpy(block + row * 16u, rgba + src_off, 16u * sizeof(float));
      }
      encode_bc6h_mode11_block(block, out);
      out += 16u;
    }
  }
}

void vt_decode_bc6h_block(
    const std::uint8_t src[16],
    float dst_rgba[64]) noexcept {
  std::memset(dst_rgba, 0, 64u * sizeof(float));
  for (int i = 0; i < 16; ++i) {
    dst_rgba[i * 4 + 3] = 1.0f;
  }
  Bc6BitReader r{src};
  const std::uint32_t mode = r.read(5u);
  if (mode != 0x03u) {
    return;
  }
  std::array<int, 3u> ep0{};
  std::array<int, 3u> ep1{};
  for (int ch = 0; ch < 3; ++ch) {
    ep0[ch] = bc6h_unquantize_10u(static_cast<int>(r.read(10u)));
  }
  for (int ch = 0; ch < 3; ++ch) {
    ep1[ch] = bc6h_unquantize_10u(static_cast<int>(r.read(10u)));
  }
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t idx = r.read(i == 0 ? 3u : 4u);
    const int w = kBc6Weight4[idx];
    for (int ch = 0; ch < 3; ++ch) {
      const int interp = (ep0[ch] * (64 - w) + ep1[ch] * w + 32) >> 6;
      const int hf = bc6h_finish_unquantize_u(interp);
      dst_rgba[i * 4 + ch] =
          vt_half_bits_to_float(static_cast<std::uint16_t>(hf & 0x7fff));
    }
    dst_rgba[i * 4 + 3] = 1.0f;
  }
}

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
    std::span<std::uint8_t> dst_bytes) noexcept {
  const std::size_t needed = vt_tile_format_bytes(physical_tile_size, fmt);
  if (dst_bytes.size() < needed) {
    return;
  }
  const std::size_t texels =
      static_cast<std::size_t>(physical_tile_size) * physical_tile_size;
  if (fmt == VtTileFormat::rgba8) {
    vt_pack_byte4_tile(
        unorm8_pixels, width, height, tile_x, tile_y, extension,
        physical_tile_size, logical_tile_size, border,
        std::span<luisa::uint>{reinterpret_cast<luisa::uint *>(dst_bytes.data()),
                               texels});
    return;
  }
  if (fmt == VtTileFormat::float4) {
    vt_pack_float4_tile(
        float_pixels, width, height, tile_x, tile_y, extension,
        physical_tile_size, logical_tile_size, border,
        std::span<luisa::float4>{
            reinterpret_cast<luisa::float4 *>(dst_bytes.data()), texels});
    return;
  }
  if (physical_tile_size > 128u || (physical_tile_size % 4u) != 0u ||
      width == 0u || height == 0u ||
      (vt_tile_format_is_high_precision(fmt) && float_pixels == nullptr) ||
      (!vt_tile_format_is_high_precision(fmt) && unorm8_pixels == nullptr)) {
    vt_fill_zero_tile(physical_tile_size, fmt, dst_bytes);
    return;
  }

  if (!vt_tile_format_is_high_precision(fmt)) {
    thread_local std::array<luisa::uint, 128u * 128u> raw_u8{};
    vt_pack_byte4_tile(
        unorm8_pixels, width, height, tile_x, tile_y, extension,
        physical_tile_size, logical_tile_size, border,
        std::span<luisa::uint>{raw_u8.data(), texels});
    const auto *rgba = reinterpret_cast<const std::uint8_t *>(raw_u8.data());
    switch (fmt) {
      case VtTileFormat::bc7:
        vt_encode_bc7_tile(rgba, physical_tile_size, dst_bytes);
        break;
      case VtTileFormat::bc3:
        vt_encode_bc3_tile(rgba, physical_tile_size, dst_bytes);
        break;
      case VtTileFormat::bc1:
        vt_encode_bc1_tile(rgba, physical_tile_size, dst_bytes);
        break;
      case VtTileFormat::bc4:
        vt_encode_bc4_tile(rgba, physical_tile_size, dst_bytes);
        break;
      case VtTileFormat::bc5:
        vt_encode_bc5_tile(rgba, physical_tile_size, dst_bytes);
        break;
      default:
        break;
    }
    return;
  }

  thread_local std::array<luisa::float4, 128u * 128u> raw_f32{};
  vt_pack_float4_tile(
      float_pixels, width, height, tile_x, tile_y, extension,
      physical_tile_size, logical_tile_size, border,
      std::span<luisa::float4>{raw_f32.data(), texels});
  const auto *src_f = reinterpret_cast<const float *>(raw_f32.data());
  switch (fmt) {
    case VtTileFormat::bc6h:
      vt_encode_bc6h_tile(src_f, physical_tile_size, dst_bytes);
      break;
    case VtTileFormat::short4: {
      auto *dst16 = reinterpret_cast<std::uint16_t *>(dst_bytes.data());
      for (std::size_t i = 0u; i < texels * 4u; ++i) {
        dst16[i] = quantize_unorm16(src_f[i]);
      }
      break;
    }
    case VtTileFormat::short1: {
      auto *dst16 = reinterpret_cast<std::uint16_t *>(dst_bytes.data());
      for (std::size_t i = 0u; i < texels; ++i) {
        dst16[i] = quantize_unorm16(src_f[i * 4u]);
      }
      break;
    }
    case VtTileFormat::half4: {
      auto *dst16 = reinterpret_cast<std::uint16_t *>(dst_bytes.data());
      for (std::size_t i = 0u; i < texels * 4u; ++i) {
        dst16[i] = vt_float_to_half_bits(src_f[i]);
      }
      break;
    }
    case VtTileFormat::half1: {
      auto *dst16 = reinterpret_cast<std::uint16_t *>(dst_bytes.data());
      for (std::size_t i = 0u; i < texels; ++i) {
        dst16[i] = vt_float_to_half_bits(src_f[i * 4u]);
      }
      break;
    }
    case VtTileFormat::float1: {
      auto *dst32 = reinterpret_cast<float *>(dst_bytes.data());
      for (std::size_t i = 0u; i < texels; ++i) {
        dst32[i] = src_f[i * 4u];
      }
      break;
    }
    default:
      break;
  }
}

} // namespace psycles::luisa_backend::detail
