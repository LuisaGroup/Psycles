#include "virtual_texture_cache_bc.h"

#include "virtual_texture_cache_prepack.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <luisa/core/basic_types.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4127 4244 4245 4305 4456 4996)
#endif
#define STB_DXT_STATIC
#define STB_DXT_IMPLEMENTATION
#include <stb/stb_dxt.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::array<std::uint8_t, 16u> kBc7Mode6ZeroBlock{
    0x40u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

inline constexpr std::array<int, 4u> kWeight2{0, 21, 43, 64};
inline constexpr std::array<int, 8u> kWeight3{0, 9, 18, 27, 37, 46, 55, 64};
inline constexpr std::array<int, 16u> kWeight4{
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// Verified 2-subset BC7 partitions p in 0..15 (every entry has subset[15] == 1
// and anchor2 == 15) plus 17 (top-right diagonal) and 18 (bottom-left corner).
inline constexpr std::size_t kMode1PartitionCount = 16u;
inline constexpr std::array<std::uint8_t, kMode1PartitionCount> kMode1Partitions{
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u};

inline constexpr std::array<std::array<std::uint8_t, 16u>, 64u> kPartition2{{
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, // 0
    {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, // 1
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1}, // 2
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1}, // 3
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1}, // 4
    {0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}, // 5
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}, // 6
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1}, // 7
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1}, // 8
    {0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 9
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1}, // 10
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1}, // 11
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 12
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, // 13
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 14
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, // 15
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1}, // 16
    {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, // 17
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0}, // 18
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0}, // 19
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, // 20
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0}, // 21
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}, // 22
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1}, // 23
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0}, // 24
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}, // 25
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, // 26
    {0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0}, // 27
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0}, // 28
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, // 29
    {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0}, // 30
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0}, // 31
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, // 32
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}, // 33
    {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}, // 34
    {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, // 35
    {0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}, // 36
    {0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0}, // 37
    {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1}, // 38
    {0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1}, // 39
    {0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0}, // 40
    {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0}, // 41
    {0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0}, // 42
    {0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0}, // 43
    {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}, // 44
    {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}, // 45
    {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1}, // 46
    {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}, // 47
    {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0}, // 48
    {0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0}, // 49
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0}, // 50
    {0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0}, // 51
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1}, // 52
    {0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1}, // 53
    {0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0}, // 54
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0}, // 55
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}, // 56
    {0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1}, // 57
    {0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1}, // 58
    {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1}, // 59
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, // 60
    {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, // 61
    {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0}, // 62
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1}, // 63
}};

inline constexpr std::array<std::uint8_t, 64u> kAnchor2{
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2,
    15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2,  2,  2,  15, 15, 6,
    6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15};

inline constexpr std::array<std::uint32_t, 64u> kPartition3Masks{
    0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u,
    0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u,
    0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u,
    0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u,
    0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u,
    0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u,
    0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u,
    0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u,
    0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u,
    0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u,
    0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u,
    0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u,
    0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u,
    0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u,
    0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u,
    0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u};

inline constexpr std::array<std::uint8_t, 64u> kAnchor3_1{
    3,  3,  15, 15, 8,  3,  15, 15, 8,  8,  6,  6,  6,  5,  3,  3,
    3,  3,  8,  15, 3,  3,  6,  10, 5,  8,  8,  6,  8,  5,  15, 15,
    8,  15, 3,  5,  6,  10, 8,  15, 15, 3,  15, 5,  15, 15, 15, 15,
    3,  15, 5,  5,  5,  8,  5,  10, 5,  10, 8,  13, 15, 12, 3,  3};

inline constexpr std::array<std::uint8_t, 64u> kAnchor3_2{
    15, 8,  8,  3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,
    15, 8,  15, 3,  15, 8,  15, 8,  3,  15, 6,  10, 15, 15, 10, 8,
    15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15, 3,  6,  6,  8,
    15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8};

struct Bc7BitWriter {
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

struct Bc7BitReader {
  std::uint64_t lo{0ull};
  std::uint64_t hi{0ull};
  std::uint32_t bit_pos{0u};

  explicit Bc7BitReader(const std::uint8_t src[16]) noexcept {
    std::memcpy(&lo, src, 8u);
    std::memcpy(&hi, src + 8u, 8u);
  }

  [[nodiscard]] std::uint32_t read(std::uint32_t bits) noexcept {
    if (bits == 0u) {
      return 0u;
    }
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

[[nodiscard]] inline std::uint8_t bc7_interp_u8(int e0, int e1, int w) noexcept {
  return static_cast<std::uint8_t>(((64 - w) * e0 + w * e1 + 32) >> 6);
}

[[nodiscard]] inline int bc7_unquantize(int val, int bits) noexcept {
  const int v = val << (8 - bits);
  return v | (v >> bits);
}

[[nodiscard]] inline std::uint32_t bc7_quant_floor(float v, int step, int parity) noexcept {
  const int vi = std::clamp(static_cast<int>(std::lround((v - static_cast<float>(parity)) * 0.5f)),
                            0, step - 1);
  return static_cast<std::uint32_t>(vi);
}

[[nodiscard]] inline std::uint8_t clamp_u8_f(float v) noexcept {
  return static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(v)), 0, 255));
}

struct Bc7BlockCandidate {
  std::uint64_t sse{UINT64_MAX};
  std::uint8_t bytes[16]{};
};

void emit_bc7_mode6(
    const std::array<std::uint8_t, 4u> &c0,
    const std::array<std::uint8_t, 4u> &c1,
    std::uint32_t p0,
    std::uint32_t p1,
    std::array<std::uint8_t, 16u> idx,
    std::uint8_t out[16]) noexcept {
  std::array<std::uint8_t, 4u> e0 = c0;
  std::array<std::uint8_t, 4u> e1 = c1;
  if (idx[0] >= 8u) {
    std::swap(e0, e1);
    std::swap(p0, p1);
    for (auto &x : idx) {
      x = static_cast<std::uint8_t>(15u - x);
    }
  }
  Bc7BitWriter w;
  w.write(0x40u, 7u);
  for (int ch = 0; ch < 4; ++ch) {
    w.write(e0[ch], 7u);
    w.write(e1[ch], 7u);
  }
  w.write(p0, 1u);
  w.write(p1, 1u);
  w.write(idx[0], 3u);
  for (std::size_t i = 1u; i < 16u; ++i) {
    w.write(idx[i], 4u);
  }
  w.finish(out);
}

[[nodiscard]] Bc7BlockCandidate encode_bc7_mode6_solid(
    const std::array<std::uint8_t, 4u> &px) noexcept {
  Bc7BlockCandidate cand;
  const std::uint32_t lsb_or = (px[0] | px[1] | px[2] | px[3]) & 1u;
  const std::uint32_t lsb_and = (px[0] & px[1] & px[2] & px[3]) & 1u;
  if (lsb_or == lsb_and) {
    std::array<std::uint8_t, 4u> c{
        static_cast<std::uint8_t>(px[0] >> 1u),
        static_cast<std::uint8_t>(px[1] >> 1u),
        static_cast<std::uint8_t>(px[2] >> 1u),
        static_cast<std::uint8_t>(px[3] >> 1u)};
    std::array<std::uint8_t, 16u> idx{};
    cand.sse = 0ull;
    emit_bc7_mode6(c, c, lsb_and, lsb_and, idx, cand.bytes);
    return cand;
  }
  // Mixed channel parities: idx = 7 (weights 34:30) solves every channel with
  // e0 parity p0 and e1 parity p1 to <= 1 LSB total (typically exact 0).
  constexpr std::size_t kSolidIdx = 7u;
  constexpr int w = kWeight4[kSolidIdx];
  for (std::uint32_t p0 = 0u; p0 <= 1u; ++p0) {
    const std::uint32_t p1 = 1u - p0;
    std::array<std::uint8_t, 4u> best_c0{};
    std::array<std::uint8_t, 4u> best_c1{};
    std::uint64_t block_sse1 = 0ull;
    for (int ch = 0; ch < 4; ++ch) {
      const int target = px[ch];
      const int base0 = std::clamp((target - static_cast<int>(p0)) >> 1, 0, 127);
      const int base1 = std::clamp((target - static_cast<int>(p1)) >> 1, 0, 127);
      int ch_best_err = INT32_MAX;
      int ch_c0 = base0;
      int ch_c1 = base1;
      for (int d0 = -1; d0 <= 1; ++d0) {
        const int tc0 = std::clamp(base0 + d0, 0, 127);
        const int e0 = (tc0 << 1) | static_cast<int>(p0);
        for (int d1 = -1; d1 <= 1; ++d1) {
          const int tc1 = std::clamp(base1 + d1, 0, 127);
          const int e1 = (tc1 << 1) | static_cast<int>(p1);
          const int dec = bc7_interp_u8(e0, e1, w);
          const int diff = dec - target;
          const int err = diff * diff;
          if (err < ch_best_err) {
            ch_best_err = err;
            ch_c0 = tc0;
            ch_c1 = tc1;
            if (err == 0) {
              break;
            }
          }
        }
        if (ch_best_err == 0) {
          break;
        }
      }
      best_c0[ch] = static_cast<std::uint8_t>(ch_c0);
      best_c1[ch] = static_cast<std::uint8_t>(ch_c1);
      block_sse1 += static_cast<std::uint64_t>(ch_best_err);
    }
    const std::uint64_t sse = block_sse1 * 16ull;
    if (sse < cand.sse) {
      std::array<std::uint8_t, 16u> idx{};
      idx.fill(static_cast<std::uint8_t>(kSolidIdx));
      cand.sse = sse;
      emit_bc7_mode6(best_c0, best_c1, p0, p1, idx, cand.bytes);
      if (sse == 0ull) {
        return cand;
      }
    }
  }
  return cand;
}

template <int Channels, int NumWeights>
void solve_least_squares_endpoints(
    const float *pixels,
    const std::uint8_t *indices,
    const int *weights,
    std::array<float, Channels> &e0,
    std::array<float, Channels> &e1) noexcept {
  double a2 = 0.0;
  double ab = 0.0;
  double b2 = 0.0;
  std::array<double, Channels> rhs0{};
  std::array<double, Channels> rhs1{};
  for (int i = 0; i < 16; ++i) {
    const double b = static_cast<double>(weights[indices[i]]) * (1.0 / 64.0);
    const double a = 1.0 - b;
    a2 += a * a;
    ab += a * b;
    b2 += b * b;
    for (int ch = 0; ch < Channels; ++ch) {
      const double p = static_cast<double>(pixels[i * Channels + ch]);
      rhs0[ch] += a * p;
      rhs1[ch] += b * p;
    }
  }
  const double det = a2 * b2 - ab * ab;
  if (det <= 1e-6) {
    std::array<float, Channels> sum{};
    for (int i = 0; i < 16; ++i) {
      for (int ch = 0; ch < Channels; ++ch) {
        sum[ch] += pixels[i * Channels + ch];
      }
    }
    for (int ch = 0; ch < Channels; ++ch) {
      const float avg = std::clamp(sum[ch] * (1.0f / 16.0f), 0.0f, 255.0f);
      e0[ch] = avg;
      e1[ch] = avg;
    }
    return;
  }
  const double inv_det = 1.0 / det;
  for (int ch = 0; ch < Channels; ++ch) {
    const double x0 = (b2 * rhs0[ch] - ab * rhs1[ch]) * inv_det;
    const double x1 = (-ab * rhs0[ch] + a2 * rhs1[ch]) * inv_det;
    e0[ch] = static_cast<float>(std::clamp(x0, 0.0, 255.0));
    e1[ch] = static_cast<float>(std::clamp(x1, 0.0, 255.0));
  }
}

[[nodiscard]] Bc7BlockCandidate encode_bc7_mode6(
    const std::uint8_t src[64]) noexcept {
  std::array<float, 64u> pf{};
  std::array<float, 4u> mean{};
  std::array<float, 4u> min_c{255.0f, 255.0f, 255.0f, 255.0f};
  std::array<float, 4u> max_c{0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 16; ++i) {
    for (int ch = 0; ch < 4; ++ch) {
      const float v = static_cast<float>(src[i * 4 + ch]);
      pf[i * 4 + ch] = v;
      mean[ch] += v;
      min_c[ch] = std::min(min_c[ch], v);
      max_c[ch] = std::max(max_c[ch], v);
    }
  }
  if (min_c[0] == max_c[0] && min_c[1] == max_c[1] &&
      min_c[2] == max_c[2] && min_c[3] == max_c[3]) {
    return encode_bc7_mode6_solid({src[0], src[1], src[2], src[3]});
  }
  for (int ch = 0; ch < 4; ++ch) {
    mean[ch] *= (1.0f / 16.0f);
  }

  float cov[4][4]{};
  for (int i = 0; i < 16; ++i) {
    std::array<float, 4u> d{};
    for (int ch = 0; ch < 4; ++ch) {
      d[ch] = pf[i * 4 + ch] - mean[ch];
    }
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        cov[r][c] += d[r] * d[c];
      }
    }
  }
  std::array<float, 4u> axis{};
  float norm2 = 0.0f;
  for (int ch = 0; ch < 4; ++ch) {
    axis[ch] = max_c[ch] - min_c[ch];
    norm2 += axis[ch] * axis[ch];
  }
  if (norm2 > 1e-6f) {
    const float inv_n = 1.0f / std::sqrt(norm2);
    for (float &a : axis) {
      a *= inv_n;
    }
    for (int it = 0; it < 4; ++it) {
      std::array<float, 4u> nxt{};
      for (int r = 0; r < 4; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < 4; ++c) {
          acc += cov[r][c] * axis[c];
        }
        nxt[r] = acc;
      }
      const float n2 = nxt[0] * nxt[0] + nxt[1] * nxt[1] +
                       nxt[2] * nxt[2] + nxt[3] * nxt[3];
      if (n2 <= 1e-8f) {
        break;
      }
      const float inv = 1.0f / std::sqrt(n2);
      for (int ch = 0; ch < 4; ++ch) {
        axis[ch] = nxt[ch] * inv;
      }
    }
  } else {
    axis = {1.0f, 0.0f, 0.0f, 0.0f};
  }

  float t_min = 1e30f;
  float t_max = -1e30f;
  for (int i = 0; i < 16; ++i) {
    float proj = 0.0f;
    for (int ch = 0; ch < 4; ++ch) {
      proj += (pf[i * 4 + ch] - mean[ch]) * axis[ch];
    }
    t_min = std::min(t_min, proj);
    t_max = std::max(t_max, proj);
  }
  std::array<float, 4u> e0{};
  std::array<float, 4u> e1{};
  for (int ch = 0; ch < 4; ++ch) {
    e0[ch] = std::clamp(mean[ch] + t_min * axis[ch], 0.0f, 255.0f);
    e1[ch] = std::clamp(mean[ch] + t_max * axis[ch], 0.0f, 255.0f);
  }

  std::array<std::uint8_t, 16u> idx{};
  auto assign_float_indices = [&]() noexcept {
    std::array<float, 4u> diff{
        e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2], e1[3] - e0[3]};
    const float len2 = diff[0] * diff[0] + diff[1] * diff[1] +
                       diff[2] * diff[2] + diff[3] * diff[3];
    if (len2 <= 1e-6f) {
      idx.fill(0u);
      return;
    }
    const float inv_len2 = 1.0f / len2;
    for (int i = 0; i < 16; ++i) {
      float dot = 0.0f;
      for (int ch = 0; ch < 4; ++ch) {
        dot += (pf[i * 4 + ch] - e0[ch]) * diff[ch];
      }
      const float t = std::clamp(dot * inv_len2, 0.0f, 1.0f);
      int best_k = std::clamp(static_cast<int>(std::lround(t * 15.0f)), 0, 15);
      // Refine in the small 3-index neighborhood by actual 4D Euclidean distance.
      const int k_lo = std::max(0, best_k - 1);
      const int k_hi = std::min(15, best_k + 1);
      float best_d = 1e30f;
      for (int k = k_lo; k <= k_hi; ++k) {
        const float w = static_cast<float>(kWeight4[k]) * (1.0f / 64.0f);
        float d2 = 0.0f;
        for (int ch = 0; ch < 4; ++ch) {
          const float dc = (e0[ch] + w * diff[ch]) - pf[i * 4 + ch];
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

  assign_float_indices();
  for (int it = 0; it < 2; ++it) {
    solve_least_squares_endpoints<4, 16>(pf.data(), idx.data(), kWeight4.data(), e0, e1);
    assign_float_indices();
  }

  Bc7BlockCandidate best{};
  for (std::uint32_t p0 = 0u; p0 <= 1u; ++p0) {
    std::array<std::uint8_t, 4u> c0{};
    std::array<int, 4u> qe0{};
    for (int ch = 0; ch < 4; ++ch) {
      c0[ch] = static_cast<std::uint8_t>(
          bc7_quant_floor(e0[ch], 128, static_cast<int>(p0)));
      qe0[ch] = (static_cast<int>(c0[ch]) << 1) | static_cast<int>(p0);
    }
    for (std::uint32_t p1 = 0u; p1 <= 1u; ++p1) {
      std::array<std::uint8_t, 4u> c1{};
      std::array<int, 4u> qe1{};
      for (int ch = 0; ch < 4; ++ch) {
        c1[ch] = static_cast<std::uint8_t>(
            bc7_quant_floor(e1[ch], 128, static_cast<int>(p1)));
        qe1[ch] = (static_cast<int>(c1[ch]) << 1) | static_cast<int>(p1);
      }
      std::array<std::array<int, 4u>, 16u> pal{};
      for (int k = 0; k < 16; ++k) {
        const int w = kWeight4[k];
        for (int ch = 0; ch < 4; ++ch) {
          pal[k][ch] = bc7_interp_u8(qe0[ch], qe1[ch], w);
        }
      }
      std::array<std::uint8_t, 16u> trial_idx{};
      std::uint64_t sse = 0ull;
      for (int i = 0; i < 16; ++i) {
        const int r = src[i * 4 + 0];
        const int g = src[i * 4 + 1];
        const int b = src[i * 4 + 2];
        const int a = src[i * 4 + 3];
        int best_k = 0;
        int best_e = INT32_MAX;
        for (int k = 0; k < 16; ++k) {
          const int dr = pal[k][0] - r;
          const int dg = pal[k][1] - g;
          const int db = pal[k][2] - b;
          const int da = pal[k][3] - a;
          const int e = dr * dr + dg * dg + db * db + da * da;
          if (e < best_e) {
            best_e = e;
            best_k = k;
          }
        }
        trial_idx[i] = static_cast<std::uint8_t>(best_k);
        sse += static_cast<std::uint64_t>(best_e);
      }
      if (sse < best.sse) {
        best.sse = sse;
        emit_bc7_mode6(c0, c1, p0, p1, trial_idx, best.bytes);
      }
    }
  }
  return best;
}

[[nodiscard]] Bc7BlockCandidate encode_bc7_mode5_rot(
    const std::uint8_t src[64],
    std::uint32_t rot) noexcept {
  // Reorder channels so coupled[0..2] are RGB-slot and indep is A-slot under `rot`.
  std::array<float, 48u> c_pix{};
  std::array<float, 16u> a_pix{};
  std::array<std::uint8_t, 48u> c_u8{};
  std::array<std::uint8_t, 16u> a_u8{};
  std::array<float, 3u> c_min{255.0f, 255.0f, 255.0f};
  std::array<float, 3u> c_max{0.0f, 0.0f, 0.0f};
  std::array<float, 3u> c_mean{};
  int a_min = 255;
  int a_max = 0;

  const auto map_channels = [rot](int r, int g, int b, int a, int out[4]) noexcept {
    out[0] = r;
    out[1] = g;
    out[2] = b;
    out[3] = a;
    if (rot == 1u) {
      std::swap(out[0], out[3]);
    } else if (rot == 2u) {
      std::swap(out[1], out[3]);
    } else if (rot == 3u) {
      std::swap(out[2], out[3]);
    }
  };

  for (int i = 0; i < 16; ++i) {
    int ch[4]{};
    map_channels(src[i * 4 + 0], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3], ch);
    for (int k = 0; k < 3; ++k) {
      const float v = static_cast<float>(ch[k]);
      c_u8[i * 3 + k] = static_cast<std::uint8_t>(ch[k]);
      c_pix[i * 3 + k] = v;
      c_min[k] = std::min(c_min[k], v);
      c_max[k] = std::max(c_max[k], v);
      c_mean[k] += v;
    }
    a_u8[i] = static_cast<std::uint8_t>(ch[3]);
    a_pix[i] = static_cast<float>(ch[3]);
    a_min = std::min(a_min, ch[3]);
    a_max = std::max(a_max, ch[3]);
  }
  for (float &m : c_mean) {
    m *= (1.0f / 16.0f);
  }

  // Fit 1D independent channel (exact 8-bit endpoints).
  std::array<float, 1u> ae0{static_cast<float>(a_min)};
  std::array<float, 1u> ae1{static_cast<float>(a_max)};
  std::array<std::uint8_t, 16u> a_idx{};
  auto assign_a = [&](int q0, int q1) noexcept {
    std::array<int, 4u> apal{};
    for (int k = 0; k < 4; ++k) {
      apal[k] = bc7_interp_u8(q0, q1, kWeight2[k]);
    }
    std::uint64_t a_sse = 0ull;
    for (int i = 0; i < 16; ++i) {
      const int av = a_u8[i];
      int best_k = 0;
      int best_e = INT32_MAX;
      for (int k = 0; k < 4; ++k) {
        const int d = apal[k] - av;
        const int e = d * d;
        if (e < best_e) {
          best_e = e;
          best_k = k;
        }
      }
      a_idx[i] = static_cast<std::uint8_t>(best_k);
      a_sse += static_cast<std::uint64_t>(best_e);
    }
    return a_sse;
  };
  (void)assign_a(a_min, a_max);
  if (a_max > a_min) {
    solve_least_squares_endpoints<1, 4>(a_pix.data(), a_idx.data(), kWeight2.data(), ae0, ae1);
  }
  std::uint8_t qa0 = clamp_u8_f(ae0[0]);
  std::uint8_t qa1 = clamp_u8_f(ae1[0]);
  const std::uint64_t a_sse = assign_a(qa0, qa1);

  // Fit 3D coupled channels with PCA + 1 LS step.
  float cov3[3][3]{};
  for (int i = 0; i < 16; ++i) {
    const float d0 = c_pix[i * 3 + 0] - c_mean[0];
    const float d1 = c_pix[i * 3 + 1] - c_mean[1];
    const float d2 = c_pix[i * 3 + 2] - c_mean[2];
    const std::array<float, 3u> d{d0, d1, d2};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        cov3[r][c] += d[r] * d[c];
      }
    }
  }
  std::array<float, 3u> axis{
      c_max[0] - c_min[0], c_max[1] - c_min[1], c_max[2] - c_min[2]};
  const float norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  std::array<float, 3u> ce0{};
  std::array<float, 3u> ce1{};
  if (norm2 > 1e-6f) {
    const float inv0 = 1.0f / std::sqrt(norm2);
    for (float &a : axis) {
      a *= inv0;
    }
    for (int it = 0; it < 3; ++it) {
      std::array<float, 3u> nxt{};
      for (int r = 0; r < 3; ++r) {
        nxt[r] = cov3[r][0] * axis[0] + cov3[r][1] * axis[1] + cov3[r][2] * axis[2];
      }
      const float n2 = nxt[0] * nxt[0] + nxt[1] * nxt[1] + nxt[2] * nxt[2];
      if (n2 <= 1e-8f) {
        break;
      }
      const float inv = 1.0f / std::sqrt(n2);
      for (int k = 0; k < 3; ++k) {
        axis[k] = nxt[k] * inv;
      }
    }
    float tmin = 1e30f;
    float tmax = -1e30f;
    for (int i = 0; i < 16; ++i) {
      const float p = (c_pix[i * 3 + 0] - c_mean[0]) * axis[0] +
                      (c_pix[i * 3 + 1] - c_mean[1]) * axis[1] +
                      (c_pix[i * 3 + 2] - c_mean[2]) * axis[2];
      tmin = std::min(tmin, p);
      tmax = std::max(tmax, p);
    }
    for (int k = 0; k < 3; ++k) {
      ce0[k] = std::clamp(c_mean[k] + tmin * axis[k], 0.0f, 255.0f);
      ce1[k] = std::clamp(c_mean[k] + tmax * axis[k], 0.0f, 255.0f);
    }
  } else {
    ce0 = c_mean;
    ce1 = c_mean;
  }

  std::array<std::uint8_t, 16u> c_idx{};
  auto assign_c_float = [&]() noexcept {
    const std::array<float, 3u> diff{
        ce1[0] - ce0[0], ce1[1] - ce0[1], ce1[2] - ce0[2]};
    const float len2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    if (len2 <= 1e-6f) {
      c_idx.fill(0u);
      return;
    }
    const float inv_len2 = 1.0f / len2;
    for (int i = 0; i < 16; ++i) {
      const float dot = (c_pix[i * 3 + 0] - ce0[0]) * diff[0] +
                        (c_pix[i * 3 + 1] - ce0[1]) * diff[1] +
                        (c_pix[i * 3 + 2] - ce0[2]) * diff[2];
      const float t = std::clamp(dot * inv_len2, 0.0f, 1.0f);
      c_idx[i] = static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(t * 3.0f)), 0, 3));
    }
  };
  assign_c_float();
  solve_least_squares_endpoints<3, 4>(c_pix.data(), c_idx.data(), kWeight2.data(), ce0, ce1);

  std::array<std::uint8_t, 3u> q0_7{};
  std::array<std::uint8_t, 3u> q1_7{};
  std::array<int, 3u> qe0{};
  std::array<int, 3u> qe1{};
  for (int k = 0; k < 3; ++k) {
    const int v0 = std::clamp(static_cast<int>(std::lround(ce0[k] * (127.0f / 255.0f))), 0, 127);
    const int v1 = std::clamp(static_cast<int>(std::lround(ce1[k] * (127.0f / 255.0f))), 0, 127);
    q0_7[k] = static_cast<std::uint8_t>(v0);
    q1_7[k] = static_cast<std::uint8_t>(v1);
    qe0[k] = bc7_unquantize(v0, 7);
    qe1[k] = bc7_unquantize(v1, 7);
  }
  std::array<std::array<int, 3u>, 4u> cpal{};
  for (int k = 0; k < 4; ++k) {
    for (int ch = 0; ch < 3; ++ch) {
      cpal[k][ch] = bc7_interp_u8(qe0[ch], qe1[ch], kWeight2[k]);
    }
  }
  std::uint64_t c_sse = 0ull;
  for (int i = 0; i < 16; ++i) {
    int best_k = 0;
    int best_e = INT32_MAX;
    for (int k = 0; k < 4; ++k) {
      const int d0 = cpal[k][0] - c_u8[i * 3 + 0];
      const int d1 = cpal[k][1] - c_u8[i * 3 + 1];
      const int d2 = cpal[k][2] - c_u8[i * 3 + 2];
      const int e = d0 * d0 + d1 * d1 + d2 * d2;
      if (e < best_e) {
        best_e = e;
        best_k = k;
      }
    }
    c_idx[i] = static_cast<std::uint8_t>(best_k);
    c_sse += static_cast<std::uint64_t>(best_e);
  }

  // Enforce independent anchor MSB==0 for color and alpha.
  if (c_idx[0] >= 2u) {
    std::swap(q0_7, q1_7);
    for (auto &x : c_idx) {
      x = static_cast<std::uint8_t>(3u - x);
    }
  }
  if (a_idx[0] >= 2u) {
    std::swap(qa0, qa1);
    for (auto &x : a_idx) {
      x = static_cast<std::uint8_t>(3u - x);
    }
  }

  Bc7BlockCandidate out;
  out.sse = c_sse + a_sse;
  Bc7BitWriter w;
  w.write(0x20u, 6u);
  w.write(rot, 2u);
  for (int ch = 0; ch < 3; ++ch) {
    w.write(q0_7[ch], 7u);
    w.write(q1_7[ch], 7u);
  }
  w.write(qa0, 8u);
  w.write(qa1, 8u);
  w.write(c_idx[0], 1u);
  for (std::size_t i = 1u; i < 16u; ++i) {
    w.write(c_idx[i], 2u);
  }
  w.write(a_idx[0], 1u);
  for (std::size_t i = 1u; i < 16u; ++i) {
    w.write(a_idx[i], 2u);
  }
  w.finish(out.bytes);
  return out;
}

[[nodiscard]] Bc7BlockCandidate encode_bc7_mode1(
    const std::uint8_t src[64]) noexcept {
  Bc7BlockCandidate best{};
  std::array<float, 48u> rgb{};
  for (int i = 0; i < 16; ++i) {
    rgb[i * 3 + 0] = static_cast<float>(src[i * 4 + 0]);
    rgb[i * 3 + 1] = static_cast<float>(src[i * 4 + 1]);
    rgb[i * 3 + 2] = static_cast<float>(src[i * 4 + 2]);
  }

  for (const std::uint8_t part : kMode1Partitions) {
    const auto &pmask = kPartition2[part];
    const std::uint8_t anchor2 = kAnchor2[part];

    std::array<std::uint8_t, 6u> ep6{};
    std::array<std::uint8_t, 6u> ep6_hi{};
    std::array<std::uint32_t, 2u> best_p{};
    std::array<std::uint8_t, 16u> idx{};
    std::uint64_t part_sse = 0ull;

    for (std::uint8_t sub = 0u; sub < 2u; ++sub) {
      std::array<float, 3u> mean{};
      std::array<float, 3u> cmin{255.0f, 255.0f, 255.0f};
      std::array<float, 3u> cmax{0.0f, 0.0f, 0.0f};
      int cnt = 0;
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] != sub) {
          continue;
        }
        ++cnt;
        for (int ch = 0; ch < 3; ++ch) {
          const float v = rgb[i * 3 + ch];
          mean[ch] += v;
          cmin[ch] = std::min(cmin[ch], v);
          cmax[ch] = std::max(cmax[ch], v);
        }
      }
      if (cnt == 0) {
        continue;
      }
      for (float &m : mean) {
        m /= static_cast<float>(cnt);
      }
      // 3D principal axis inside subset using covariance power iteration.
      float cov3[3][3]{};
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] != sub) {
          continue;
        }
        const std::array<float, 3u> d{
            rgb[i * 3 + 0] - mean[0],
            rgb[i * 3 + 1] - mean[1],
            rgb[i * 3 + 2] - mean[2]};
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            cov3[r][c] += d[r] * d[c];
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
        for (int it = 0; it < 2; ++it) {
          std::array<float, 3u> nxt{};
          for (int r = 0; r < 3; ++r) {
            nxt[r] = cov3[r][0] * axis[0] + cov3[r][1] * axis[1] + cov3[r][2] * axis[2];
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
          if (pmask[i] != sub) {
            continue;
          }
          const float t = (rgb[i * 3 + 0] - mean[0]) * axis[0] +
                          (rgb[i * 3 + 1] - mean[1]) * axis[1] +
                          (rgb[i * 3 + 2] - mean[2]) * axis[2];
          tmin = std::min(tmin, t);
          tmax = std::max(tmax, t);
        }
        for (int ch = 0; ch < 3; ++ch) {
          e0[ch] = std::clamp(mean[ch] + tmin * axis[ch], 0.0f, 255.0f);
          e1[ch] = std::clamp(mean[ch] + tmax * axis[ch], 0.0f, 255.0f);
        }
      }

      // One subset least-squares refinement with 3-bit weights.
      const std::array<float, 3u> diff{
          e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]};
      const float len2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
      if (len2 > 1e-6f) {
        const float inv_len2 = 1.0f / len2;
        double a2 = 0.0;
        double ab = 0.0;
        double b2 = 0.0;
        std::array<double, 3u> rhs0{};
        std::array<double, 3u> rhs1{};
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] != sub) {
            continue;
          }
          const float dot = (rgb[i * 3 + 0] - e0[0]) * diff[0] +
                            (rgb[i * 3 + 1] - e0[1]) * diff[1] +
                            (rgb[i * 3 + 2] - e0[2]) * diff[2];
          const int k = std::clamp(static_cast<int>(std::lround(
                                       std::clamp(dot * inv_len2, 0.0f, 1.0f) * 7.0f)),
                                   0, 7);
          const double b = static_cast<double>(kWeight3[k]) * (1.0 / 64.0);
          const double a = 1.0 - b;
          a2 += a * a;
          ab += a * b;
          b2 += b * b;
          for (int ch = 0; ch < 3; ++ch) {
            rhs0[ch] += a * static_cast<double>(rgb[i * 3 + ch]);
            rhs1[ch] += b * static_cast<double>(rgb[i * 3 + ch]);
          }
        }
        const double det = a2 * b2 - ab * ab;
        if (det > 1e-6) {
          const double inv_det = 1.0 / det;
          for (int ch = 0; ch < 3; ++ch) {
            e0[ch] = static_cast<float>(
                std::clamp((b2 * rhs0[ch] - ab * rhs1[ch]) * inv_det, 0.0, 255.0));
            e1[ch] = static_cast<float>(
                std::clamp((-ab * rhs0[ch] + a2 * rhs1[ch]) * inv_det, 0.0, 255.0));
          }
        }
      }

      // Evaluate shared P-bit in {0, 1} for this subset independently.
      std::uint64_t sub_best_sse = UINT64_MAX;
      std::array<std::uint8_t, 3u> sub_c0{};
      std::array<std::uint8_t, 3u> sub_c1{};
      std::uint32_t sub_p = 0u;
      std::array<std::uint8_t, 16u> sub_idx{};
      for (std::uint32_t p = 0u; p <= 1u; ++p) {
        std::array<std::uint8_t, 3u> tc0{};
        std::array<std::uint8_t, 3u> tc1{};
        std::array<int, 3u> qe0{};
        std::array<int, 3u> qe1{};
        for (int ch = 0; ch < 3; ++ch) {
          // Unquantized 7-bit step is ~4; best 6-bit value for shared parity p.
          const int c0_6 = std::clamp(
              static_cast<int>(std::lround((e0[ch] * (127.0f / 255.0f) - static_cast<float>(p)) * 0.5f)),
              0, 63);
          const int c1_6 = std::clamp(
              static_cast<int>(std::lround((e1[ch] * (127.0f / 255.0f) - static_cast<float>(p)) * 0.5f)),
              0, 63);
          tc0[ch] = static_cast<std::uint8_t>(c0_6);
          tc1[ch] = static_cast<std::uint8_t>(c1_6);
          qe0[ch] = bc7_unquantize((c0_6 << 1) | static_cast<int>(p), 7);
          qe1[ch] = bc7_unquantize((c1_6 << 1) | static_cast<int>(p), 7);
        }
        std::array<std::array<int, 3u>, 8u> pal{};
        for (int k = 0; k < 8; ++k) {
          for (int ch = 0; ch < 3; ++ch) {
            pal[k][ch] = bc7_interp_u8(qe0[ch], qe1[ch], kWeight3[k]);
          }
        }
        std::uint64_t sub_sse = 0ull;
        std::array<std::uint8_t, 16u> trial_sub_idx{};
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] != sub) {
            continue;
          }
          const int r = src[i * 4 + 0];
          const int g = src[i * 4 + 1];
          const int b = src[i * 4 + 2];
          int bk = 0;
          int be = INT32_MAX;
          for (int k = 0; k < 8; ++k) {
            const int dr = pal[k][0] - r;
            const int dg = pal[k][1] - g;
            const int db = pal[k][2] - b;
            const int e = dr * dr + dg * dg + db * db;
            if (e < be) {
              be = e;
              bk = k;
            }
          }
          trial_sub_idx[i] = static_cast<std::uint8_t>(bk);
          sub_sse += static_cast<std::uint64_t>(be);
        }
        if (sub_sse < sub_best_sse) {
          sub_best_sse = sub_sse;
          sub_c0 = tc0;
          sub_c1 = tc1;
          sub_p = p;
          sub_idx = trial_sub_idx;
        }
      }

      const int anchor_i = (sub == 0u) ? 0 : static_cast<int>(anchor2);
      if (sub_idx[anchor_i] >= 4u) {
        std::swap(sub_c0, sub_c1);
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] == sub) {
            sub_idx[i] = static_cast<std::uint8_t>(7u - sub_idx[i]);
          }
        }
      }
      best_p[sub] = sub_p;
      if (sub == 0u) {
        ep6[0] = sub_c0[0];
        ep6[1] = sub_c1[0];
        ep6[2] = sub_c0[1];
        ep6[3] = sub_c1[1];
        ep6[4] = sub_c0[2];
        ep6[5] = sub_c1[2];
      } else {
        ep6_hi[0] = sub_c0[0];
        ep6_hi[1] = sub_c1[0];
        ep6_hi[2] = sub_c0[1];
        ep6_hi[3] = sub_c1[1];
        ep6_hi[4] = sub_c0[2];
        ep6_hi[5] = sub_c1[2];
      }
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] == sub) {
          idx[i] = sub_idx[i];
        }
      }
      part_sse += sub_best_sse;
    }

    if (part_sse < best.sse) {
      best.sse = part_sse;
      Bc7BitWriter w;
      w.write(0x02u, 2u);
      w.write(part, 6u);
      // Order: R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3 (6 bits each).
      for (int ch = 0; ch < 3; ++ch) {
        w.write(ep6[ch * 2 + 0], 6u);
        w.write(ep6[ch * 2 + 1], 6u);
        w.write(ep6_hi[ch * 2 + 0], 6u);
        w.write(ep6_hi[ch * 2 + 1], 6u);
      }
      w.write(best_p[0], 1u);
      w.write(best_p[1], 1u);
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t bits =
            (i == 0 || i == static_cast<int>(anchor2)) ? 2u : 3u;
        w.write(idx[i], bits);
      }
      w.finish(best.bytes);
    }
  }
  return best;
}

[[nodiscard]] Bc7BlockCandidate encode_bc7_mode7(
    const std::uint8_t src[64]) noexcept {
  Bc7BlockCandidate best{};
  std::array<float, 64u> rgba{};
  for (int i = 0; i < 16; ++i) {
    for (int ch = 0; ch < 4; ++ch) {
      rgba[i * 4 + ch] = static_cast<float>(src[i * 4 + ch]);
    }
  }

  for (const std::uint8_t part : kMode1Partitions) {
    const auto &pmask = kPartition2[part];
    const std::uint8_t anchor2 = kAnchor2[part];

    std::array<std::uint8_t, 8u> ep5{};
    std::array<std::uint8_t, 8u> ep5_hi{};
    std::array<std::uint32_t, 4u> best_p{};
    std::array<std::uint8_t, 16u> idx{};
    std::uint64_t part_sse = 0ull;

    for (std::uint8_t sub = 0u; sub < 2u; ++sub) {
      std::array<float, 4u> mean{};
      std::array<float, 4u> cmin{255.0f, 255.0f, 255.0f, 255.0f};
      std::array<float, 4u> cmax{0.0f, 0.0f, 0.0f, 0.0f};
      int cnt = 0;
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] != sub) {
          continue;
        }
        ++cnt;
        for (int ch = 0; ch < 4; ++ch) {
          const float v = rgba[i * 4 + ch];
          mean[ch] += v;
          cmin[ch] = std::min(cmin[ch], v);
          cmax[ch] = std::max(cmax[ch], v);
        }
      }
      if (cnt == 0) {
        continue;
      }
      for (float &m : mean) {
        m /= static_cast<float>(cnt);
      }

      float cov[4][4]{};
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] != sub) {
          continue;
        }
        const std::array<float, 4u> d{
            rgba[i * 4 + 0] - mean[0], rgba[i * 4 + 1] - mean[1],
            rgba[i * 4 + 2] - mean[2], rgba[i * 4 + 3] - mean[3]};
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            cov[r][c] += d[r] * d[c];
          }
        }
      }
      std::array<float, 4u> axis{
          cmax[0] - cmin[0], cmax[1] - cmin[1],
          cmax[2] - cmin[2], cmax[3] - cmin[3]};
      const float norm2 = axis[0] * axis[0] + axis[1] * axis[1] +
                          axis[2] * axis[2] + axis[3] * axis[3];
      std::array<float, 4u> e0{mean};
      std::array<float, 4u> e1{mean};
      if (norm2 > 1e-6f) {
        const float inv0 = 1.0f / std::sqrt(norm2);
        for (float &a : axis) {
          a *= inv0;
        }
        for (int it = 0; it < 2; ++it) {
          std::array<float, 4u> nxt{};
          for (int r = 0; r < 4; ++r) {
            nxt[r] = cov[r][0] * axis[0] + cov[r][1] * axis[1] +
                     cov[r][2] * axis[2] + cov[r][3] * axis[3];
          }
          const float n2 = nxt[0] * nxt[0] + nxt[1] * nxt[1] +
                           nxt[2] * nxt[2] + nxt[3] * nxt[3];
          if (n2 <= 1e-8f) {
            break;
          }
          const float inv = 1.0f / std::sqrt(n2);
          for (int ch = 0; ch < 4; ++ch) {
            axis[ch] = nxt[ch] * inv;
          }
        }
        float tmin = 1e30f;
        float tmax = -1e30f;
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] != sub) {
            continue;
          }
          const float t = (rgba[i * 4 + 0] - mean[0]) * axis[0] +
                          (rgba[i * 4 + 1] - mean[1]) * axis[1] +
                          (rgba[i * 4 + 2] - mean[2]) * axis[2] +
                          (rgba[i * 4 + 3] - mean[3]) * axis[3];
          tmin = std::min(tmin, t);
          tmax = std::max(tmax, t);
        }
        for (int ch = 0; ch < 4; ++ch) {
          e0[ch] = std::clamp(mean[ch] + tmin * axis[ch], 0.0f, 255.0f);
          e1[ch] = std::clamp(mean[ch] + tmax * axis[ch], 0.0f, 255.0f);
        }
      }

      // One subset least-squares refinement with 2-bit weights.
      const std::array<float, 4u> diff{
          e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2], e1[3] - e0[3]};
      const float len2 = diff[0] * diff[0] + diff[1] * diff[1] +
                         diff[2] * diff[2] + diff[3] * diff[3];
      if (len2 > 1e-6f) {
        const float inv_len2 = 1.0f / len2;
        double a2 = 0.0;
        double ab = 0.0;
        double b2 = 0.0;
        std::array<double, 4u> rhs0{};
        std::array<double, 4u> rhs1{};
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] != sub) {
            continue;
          }
          const float dot = (rgba[i * 4 + 0] - e0[0]) * diff[0] +
                            (rgba[i * 4 + 1] - e0[1]) * diff[1] +
                            (rgba[i * 4 + 2] - e0[2]) * diff[2] +
                            (rgba[i * 4 + 3] - e0[3]) * diff[3];
          const int k = std::clamp(
              static_cast<int>(std::lround(
                  std::clamp(dot * inv_len2, 0.0f, 1.0f) * 3.0f)),
              0, 3);
          const double b = static_cast<double>(kWeight2[k]) * (1.0 / 64.0);
          const double a = 1.0 - b;
          a2 += a * a;
          ab += a * b;
          b2 += b * b;
          for (int ch = 0; ch < 4; ++ch) {
            rhs0[ch] += a * static_cast<double>(rgba[i * 4 + ch]);
            rhs1[ch] += b * static_cast<double>(rgba[i * 4 + ch]);
          }
        }
        const double det = a2 * b2 - ab * ab;
        if (det > 1e-6) {
          const double inv_det = 1.0 / det;
          for (int ch = 0; ch < 4; ++ch) {
            e0[ch] = static_cast<float>(
                std::clamp((b2 * rhs0[ch] - ab * rhs1[ch]) * inv_det, 0.0, 255.0));
            e1[ch] = static_cast<float>(
                std::clamp((-ab * rhs0[ch] + a2 * rhs1[ch]) * inv_det, 0.0, 255.0));
          }
        }
      }

      // Evaluate all 4 unique endpoint P-bit pairs (p0, p1) in {0, 1}^2.
      std::uint64_t sub_best_sse = UINT64_MAX;
      std::array<std::uint8_t, 4u> sub_c0{};
      std::array<std::uint8_t, 4u> sub_c1{};
      std::uint32_t sub_p0 = 0u;
      std::uint32_t sub_p1 = 0u;
      std::array<std::uint8_t, 16u> sub_idx{};
      for (std::uint32_t p0 = 0u; p0 <= 1u; ++p0) {
        std::array<std::uint8_t, 4u> tc0{};
        std::array<int, 4u> qe0{};
        for (int ch = 0; ch < 4; ++ch) {
          const int c0_5 = std::clamp(
              static_cast<int>(std::lround(
                  (e0[ch] * (63.0f / 255.0f) - static_cast<float>(p0)) * 0.5f)),
              0, 31);
          tc0[ch] = static_cast<std::uint8_t>(c0_5);
          qe0[ch] = bc7_unquantize((c0_5 << 1) | static_cast<int>(p0), 6);
        }
        for (std::uint32_t p1 = 0u; p1 <= 1u; ++p1) {
          std::array<std::uint8_t, 4u> tc1{};
          std::array<int, 4u> qe1{};
          for (int ch = 0; ch < 4; ++ch) {
            const int c1_5 = std::clamp(
                static_cast<int>(std::lround(
                    (e1[ch] * (63.0f / 255.0f) - static_cast<float>(p1)) * 0.5f)),
                0, 31);
            tc1[ch] = static_cast<std::uint8_t>(c1_5);
            qe1[ch] = bc7_unquantize((c1_5 << 1) | static_cast<int>(p1), 6);
          }
          std::array<std::array<int, 4u>, 4u> pal{};
          for (int k = 0; k < 4; ++k) {
            for (int ch = 0; ch < 4; ++ch) {
              pal[k][ch] = bc7_interp_u8(qe0[ch], qe1[ch], kWeight2[k]);
            }
          }
          std::uint64_t sub_sse = 0ull;
          std::array<std::uint8_t, 16u> trial_sub_idx{};
          for (int i = 0; i < 16; ++i) {
            if (pmask[i] != sub) {
              continue;
            }
            const int r = src[i * 4 + 0];
            const int g = src[i * 4 + 1];
            const int b = src[i * 4 + 2];
            const int a = src[i * 4 + 3];
            int bk = 0;
            int be = INT32_MAX;
            for (int k = 0; k < 4; ++k) {
              const int dr = pal[k][0] - r;
              const int dg = pal[k][1] - g;
              const int db = pal[k][2] - b;
              const int da = pal[k][3] - a;
              const int e = dr * dr + dg * dg + db * db + da * da;
              if (e < be) {
                be = e;
                bk = k;
              }
            }
            trial_sub_idx[i] = static_cast<std::uint8_t>(bk);
            sub_sse += static_cast<std::uint64_t>(be);
          }
          if (sub_sse < sub_best_sse) {
            sub_best_sse = sub_sse;
            sub_c0 = tc0;
            sub_c1 = tc1;
            sub_p0 = p0;
            sub_p1 = p1;
            sub_idx = trial_sub_idx;
          }
        }
      }

      const int anchor_i = (sub == 0u) ? 0 : static_cast<int>(anchor2);
      if (sub_idx[anchor_i] >= 2u) {
        std::swap(sub_c0, sub_c1);
        std::swap(sub_p0, sub_p1);
        for (int i = 0; i < 16; ++i) {
          if (pmask[i] == sub) {
            sub_idx[i] = static_cast<std::uint8_t>(3u - sub_idx[i]);
          }
        }
      }
      best_p[sub * 2u + 0u] = sub_p0;
      best_p[sub * 2u + 1u] = sub_p1;
      auto &dst_ep = (sub == 0u) ? ep5 : ep5_hi;
      for (int ch = 0; ch < 4; ++ch) {
        dst_ep[ch * 2 + 0] = sub_c0[ch];
        dst_ep[ch * 2 + 1] = sub_c1[ch];
      }
      for (int i = 0; i < 16; ++i) {
        if (pmask[i] == sub) {
          idx[i] = sub_idx[i];
        }
      }
      part_sse += sub_best_sse;
    }

    if (part_sse < best.sse) {
      best.sse = part_sse;
      Bc7BitWriter w;
      w.write(0x80u, 8u);
      w.write(part, 6u);
      // Channel-major 5-bit endpoints: R0..R3, G0..G3, B0..B3, A0..A3.
      for (int ch = 0; ch < 4; ++ch) {
        w.write(ep5[ch * 2 + 0], 5u);
        w.write(ep5[ch * 2 + 1], 5u);
        w.write(ep5_hi[ch * 2 + 0], 5u);
        w.write(ep5_hi[ch * 2 + 1], 5u);
      }
      for (int e = 0; e < 4; ++e) {
        w.write(best_p[e], 1u);
      }
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t bits =
            (i == 0 || i == static_cast<int>(anchor2)) ? 1u : 2u;
        w.write(idx[i], bits);
      }
      w.finish(best.bytes);
    }
  }
  return best;
}

void encode_bc7_block(const std::uint8_t src[64], std::uint8_t dst[16]) noexcept {
  int a_min = 255;
  int a_max = 0;
  for (int i = 0; i < 16; ++i) {
    const int a = src[i * 4 + 3];
    a_min = std::min(a_min, a);
    a_max = std::max(a_max, a);
  }
  Bc7BlockCandidate best = encode_bc7_mode6(src);
  if (best.sse == 0ull) {
    std::memcpy(dst, best.bytes, 16u);
    return;
  }
  if (a_min != a_max) {
    const Bc7BlockCandidate m5 = encode_bc7_mode5_rot(src, 0u);
    if (m5.sse < best.sse) {
      best = m5;
    }
  } else if (best.sse > 48ull * 16ull) {
    for (std::uint32_t rot = 1u; rot <= 3u; ++rot) {
      const Bc7BlockCandidate m5 = encode_bc7_mode5_rot(src, rot);
      if (m5.sse < best.sse) {
        best = m5;
      }
    }
  }
  if (a_min == 255 && best.sse > 64ull * 16ull) {
    const Bc7BlockCandidate m1 = encode_bc7_mode1(src);
    if (m1.sse < best.sse) {
      best = m1;
    }
  } else if (a_min < 255 && best.sse > 64ull * 16ull) {
    const Bc7BlockCandidate m7 = encode_bc7_mode7(src);
    if (m7.sse < best.sse) {
      best = m7;
    }
  }
  std::memcpy(dst, best.bytes, 16u);
}

[[nodiscard]] std::array<std::uint8_t, 3u> unpack_rgb565(std::uint16_t c) noexcept {
  const std::uint32_t r5 = (c >> 11u) & 0x1fu;
  const std::uint32_t g6 = (c >> 5u) & 0x3fu;
  const std::uint32_t b5 = c & 0x1fu;
  return {
      static_cast<std::uint8_t>((r5 << 3u) | (r5 >> 2u)),
      static_cast<std::uint8_t>((g6 << 2u) | (g6 >> 4u)),
      static_cast<std::uint8_t>((b5 << 3u) | (b5 >> 2u))};
}

void decode_bc1_color_block(
    const std::uint8_t src[8],
    bool force_four_color,
    std::uint8_t dst_rgba[64]) noexcept {
  const std::uint16_t c0 =
      static_cast<std::uint16_t>(src[0]) |
      (static_cast<std::uint16_t>(src[1]) << 8u);
  const std::uint16_t c1 =
      static_cast<std::uint16_t>(src[2]) |
      (static_cast<std::uint16_t>(src[3]) << 8u);
  std::uint32_t indices =
      static_cast<std::uint32_t>(src[4]) |
      (static_cast<std::uint32_t>(src[5]) << 8u) |
      (static_cast<std::uint32_t>(src[6]) << 16u) |
      (static_cast<std::uint32_t>(src[7]) << 24u);

  const auto rgb0 = unpack_rgb565(c0);
  const auto rgb1 = unpack_rgb565(c1);
  std::array<std::array<std::uint8_t, 4u>, 4u> pal{};
  for (int ch = 0; ch < 3; ++ch) {
    pal[0][ch] = rgb0[ch];
    pal[1][ch] = rgb1[ch];
  }
  pal[0][3] = 255u;
  pal[1][3] = 255u;
  pal[2][3] = 255u;
  pal[3][3] = 255u;
  if (force_four_color || c0 > c1) {
    for (int ch = 0; ch < 3; ++ch) {
      pal[2][ch] = static_cast<std::uint8_t>((2u * rgb0[ch] + rgb1[ch] + 1u) / 3u);
      pal[3][ch] = static_cast<std::uint8_t>((rgb0[ch] + 2u * rgb1[ch] + 1u) / 3u);
    }
  } else {
    for (int ch = 0; ch < 3; ++ch) {
      pal[2][ch] = static_cast<std::uint8_t>((rgb0[ch] + rgb1[ch]) / 2u);
      pal[3][ch] = 0u;
    }
  }
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t code = indices & 0x3u;
    indices >>= 2u;
    std::memcpy(dst_rgba + i * 4, pal[code].data(), 4u);
  }
}

void decode_bc4_channel_block(
    const std::uint8_t src[8],
    std::uint8_t *dst_first,
    std::size_t dst_stride) noexcept {
  const std::uint32_t a0 = src[0];
  const std::uint32_t a1 = src[1];
  std::array<std::uint8_t, 8u> pal{};
  pal[0] = static_cast<std::uint8_t>(a0);
  pal[1] = static_cast<std::uint8_t>(a1);
  if (a0 > a1) {
    for (std::uint32_t i = 1u; i <= 6u; ++i) {
      pal[i + 1u] = static_cast<std::uint8_t>(((7u - i) * a0 + i * a1 + 3u) / 7u);
    }
  } else {
    for (std::uint32_t i = 1u; i <= 4u; ++i) {
      pal[i + 1u] = static_cast<std::uint8_t>(((5u - i) * a0 + i * a1 + 2u) / 5u);
    }
    pal[6] = 0u;
    pal[7] = 255u;
  }
  std::uint64_t abits = 0ull;
  for (int b = 0; b < 6; ++b) {
    abits |= static_cast<std::uint64_t>(src[2 + b]) << (b * 8);
  }
  for (std::size_t i = 0u; i < 16u; ++i) {
    dst_first[i * dst_stride] = pal[abits & 0x7ull];
    abits >>= 3u;
  }
}

} // namespace

void vt_fill_zero_byte4_tile(
    std::uint32_t physical_tile_size,
    LuisaVirtualTextureCompressionMode mode,
    std::span<std::uint8_t> dst) noexcept {
  const std::uint32_t needed = vt_byte4_tile_bytes(physical_tile_size, mode);
  if (dst.size() < needed) {
    return;
  }
  if (mode == LuisaVirtualTextureCompressionMode::bc7) {
    for (std::size_t off = 0u; off < needed; off += 16u) {
      std::memcpy(dst.data() + off, kBc7Mode6ZeroBlock.data(), 16u);
    }
    return;
  }
  std::memset(dst.data(), 0, needed);
}

void vt_encode_bc1_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept {
  if (rgba == nullptr || (tile_size % 4u) != 0u) {
    return;
  }
  const std::uint32_t bx_count = tile_size / 4u;
  const std::size_t needed = static_cast<std::size_t>(bx_count) * bx_count * 8u;
  if (dst.size() < needed) {
    return;
  }
  std::uint8_t block[64];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        std::memcpy(block + row * 16u, rgba + src_off, 16u);
      }
      // stb_compress_dxt_block(..., alpha=0) requires constant alpha across the 4x4 block.
      for (std::uint32_t p = 0u; p < 16u; ++p) {
        block[p * 4u + 3u] = 255u;
      }
      stb_compress_dxt_block(out, block, 0, STB_DXT_HIGHQUAL);
      out += 8u;
    }
  }
}

void vt_encode_bc3_tile(
    const std::uint8_t *rgba,
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
  std::uint8_t block[64];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        std::memcpy(block + row * 16u, rgba + src_off, 16u);
      }
      stb_compress_dxt_block(out, block, 1, STB_DXT_HIGHQUAL);
      out += 16u;
    }
  }
}

void vt_encode_bc4_tile(
    const std::uint8_t *rgba,
    std::uint32_t tile_size,
    std::span<std::uint8_t> dst) noexcept {
  if (rgba == nullptr || (tile_size % 4u) != 0u) {
    return;
  }
  const std::uint32_t bx_count = tile_size / 4u;
  const std::size_t needed = static_cast<std::size_t>(bx_count) * bx_count * 8u;
  if (dst.size() < needed) {
    return;
  }
  std::uint8_t block[16];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        for (std::uint32_t col = 0u; col < 4u; ++col) {
          block[row * 4u + col] = rgba[src_off + col * 4u];
        }
      }
      stb_compress_bc4_block(out, block);
      out += 8u;
    }
  }
}

void vt_encode_bc5_tile(
    const std::uint8_t *rgba,
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
  std::uint8_t block[32];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        for (std::uint32_t col = 0u; col < 4u; ++col) {
          const std::size_t dst_p = (row * 4u + col) * 2u;
          block[dst_p + 0u] = rgba[src_off + col * 4u + 0u];
          block[dst_p + 1u] = rgba[src_off + col * 4u + 3u];
        }
      }
      stb_compress_bc5_block(out, block);
      out += 16u;
    }
  }
}

void vt_encode_bc7_tile(
    const std::uint8_t *rgba,
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
  std::uint8_t block[64];
  std::uint8_t *out = dst.data();
  for (std::uint32_t by = 0u; by < bx_count; ++by) {
    for (std::uint32_t bx = 0u; bx < bx_count; ++bx) {
      for (std::uint32_t row = 0u; row < 4u; ++row) {
        const std::size_t src_off =
            (static_cast<std::size_t>(by * 4u + row) * tile_size + bx * 4u) * 4u;
        std::memcpy(block + row * 16u, rgba + src_off, 16u);
      }
      encode_bc7_block(block, out);
      out += 16u;
    }
  }
}

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
    std::span<std::uint8_t> dst_bytes) noexcept {
  if (compression == LuisaVirtualTextureCompressionMode::off) {
    const std::size_t texels =
        static_cast<std::size_t>(physical_tile_size) * physical_tile_size;
    if (dst_bytes.size() < texels * sizeof(luisa::uint)) {
      return;
    }
    vt_pack_byte4_tile(
        pixels, width, height, tile_x, tile_y, extension, physical_tile_size,
        logical_tile_size, border,
        std::span<luisa::uint>{reinterpret_cast<luisa::uint *>(dst_bytes.data()),
                               texels});
    return;
  }
  if (pixels == nullptr || width == 0u || height == 0u ||
      physical_tile_size > 128u || (physical_tile_size % 4u) != 0u) {
    vt_fill_zero_byte4_tile(physical_tile_size, compression, dst_bytes);
    return;
  }
  thread_local std::array<luisa::uint, 128u * 128u> raw_tile{};
  const std::size_t texels =
      static_cast<std::size_t>(physical_tile_size) * physical_tile_size;
  vt_pack_byte4_tile(
      pixels, width, height, tile_x, tile_y, extension, physical_tile_size,
      logical_tile_size, border,
      std::span<luisa::uint>{raw_tile.data(), texels});
  const auto *rgba = reinterpret_cast<const std::uint8_t *>(raw_tile.data());
  switch (compression) {
    case LuisaVirtualTextureCompressionMode::bc7:
      vt_encode_bc7_tile(rgba, physical_tile_size, dst_bytes);
      break;
    case LuisaVirtualTextureCompressionMode::bc3:
      vt_encode_bc3_tile(rgba, physical_tile_size, dst_bytes);
      break;
    case LuisaVirtualTextureCompressionMode::bc1:
      vt_encode_bc1_tile(rgba, physical_tile_size, dst_bytes);
      break;
    case LuisaVirtualTextureCompressionMode::off:
    case LuisaVirtualTextureCompressionMode::auto_hq:
    case LuisaVirtualTextureCompressionMode::auto_compact:
      break;
  }
}

void vt_decode_bc1_block(
    const std::uint8_t src[8],
    std::uint8_t dst_rgba[64]) noexcept {
  decode_bc1_color_block(src, false, dst_rgba);
}

void vt_decode_bc3_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept {
  decode_bc1_color_block(src + 8u, true, dst_rgba);
  decode_bc4_channel_block(src, dst_rgba + 3u, 4u);
}

void vt_decode_bc4_block(
    const std::uint8_t src[8],
    std::uint8_t dst_rgba[64]) noexcept {
  decode_bc4_channel_block(src, dst_rgba, 4u);
  for (std::size_t i = 0u; i < 16u; ++i) {
    const std::uint8_t v = dst_rgba[i * 4u];
    dst_rgba[i * 4u + 1u] = v;
    dst_rgba[i * 4u + 2u] = v;
    dst_rgba[i * 4u + 3u] = 255u;
  }
}

void vt_decode_bc5_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept {
  decode_bc4_channel_block(src, dst_rgba + 0u, 4u);
  decode_bc4_channel_block(src + 8u, dst_rgba + 1u, 4u);
  for (std::size_t i = 0u; i < 16u; ++i) {
    dst_rgba[i * 4u + 2u] = 0u;
    dst_rgba[i * 4u + 3u] = 255u;
  }
}

void vt_decode_bc7_block(
    const std::uint8_t src[16],
    std::uint8_t dst_rgba[64]) noexcept {
  std::memset(dst_rgba, 0, 64u);
  const std::uint8_t b0 = src[0];
  if (b0 == 0u) {
    return;
  }
  int mode = 0;
  while (((b0 >> mode) & 1u) == 0u && mode < 8) {
    ++mode;
  }
  if (mode >= 8) {
    return;
  }

  Bc7BitReader r{src};
  static_cast<void>(r.read(static_cast<std::uint32_t>(mode + 1)));

  if (mode == 6) {
    std::array<std::array<int, 4u>, 2u> ep{};
    for (int ch = 0; ch < 4; ++ch) {
      ep[0][ch] = static_cast<int>(r.read(7u));
      ep[1][ch] = static_cast<int>(r.read(7u));
    }
    const int p0 = static_cast<int>(r.read(1u));
    const int p1 = static_cast<int>(r.read(1u));
    for (int ch = 0; ch < 4; ++ch) {
      ep[0][ch] = (ep[0][ch] << 1) | p0;
      ep[1][ch] = (ep[1][ch] << 1) | p1;
    }
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t idx = r.read(i == 0 ? 3u : 4u);
      const int w = kWeight4[idx];
      for (int ch = 0; ch < 4; ++ch) {
        dst_rgba[i * 4 + ch] = bc7_interp_u8(ep[0][ch], ep[1][ch], w);
      }
    }
    return;
  }

  if (mode == 5) {
    const std::uint32_t rot = r.read(2u);
    std::array<std::array<int, 3u>, 2u> cep{};
    for (int ch = 0; ch < 3; ++ch) {
      cep[0][ch] = bc7_unquantize(static_cast<int>(r.read(7u)), 7);
      cep[1][ch] = bc7_unquantize(static_cast<int>(r.read(7u)), 7);
    }
    const int a0 = static_cast<int>(r.read(8u));
    const int a1 = static_cast<int>(r.read(8u));
    std::array<std::uint32_t, 16u> cidx{};
    std::array<std::uint32_t, 16u> aidx{};
    for (int i = 0; i < 16; ++i) {
      cidx[i] = r.read(i == 0 ? 1u : 2u);
    }
    for (int i = 0; i < 16; ++i) {
      aidx[i] = r.read(i == 0 ? 1u : 2u);
    }
    for (int i = 0; i < 16; ++i) {
      std::array<std::uint8_t, 4u> px{};
      const int cw = kWeight2[cidx[i]];
      for (int ch = 0; ch < 3; ++ch) {
        px[ch] = bc7_interp_u8(cep[0][ch], cep[1][ch], cw);
      }
      px[3] = bc7_interp_u8(a0, a1, kWeight2[aidx[i]]);
      if (rot == 1u) {
        std::swap(px[0], px[3]);
      } else if (rot == 2u) {
        std::swap(px[1], px[3]);
      } else if (rot == 3u) {
        std::swap(px[2], px[3]);
      }
      std::memcpy(dst_rgba + i * 4, px.data(), 4u);
    }
    return;
  }

  if (mode == 1) {
    const std::uint32_t part = r.read(6u);
    const auto &pmask = kPartition2[part];
    const std::uint8_t anchor2 = kAnchor2[part];
    std::array<std::array<int, 3u>, 4u> ep{};
    for (int ch = 0; ch < 3; ++ch) {
      for (int e = 0; e < 4; ++e) {
        ep[e][ch] = static_cast<int>(r.read(6u));
      }
    }
    const int p0 = static_cast<int>(r.read(1u));
    const int p1 = static_cast<int>(r.read(1u));
    for (int e = 0; e < 4; ++e) {
      const int p = (e < 2) ? p0 : p1;
      for (int ch = 0; ch < 3; ++ch) {
        ep[e][ch] = bc7_unquantize((ep[e][ch] << 1) | p, 7);
      }
    }
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t bits =
          (i == 0 || i == static_cast<int>(anchor2)) ? 2u : 3u;
      const std::uint32_t idx = r.read(bits);
      const int sub = pmask[i];
      const int w = kWeight3[idx];
      for (int ch = 0; ch < 3; ++ch) {
        dst_rgba[i * 4 + ch] =
            bc7_interp_u8(ep[sub * 2 + 0][ch], ep[sub * 2 + 1][ch], w);
      }
      dst_rgba[i * 4 + 3] = 255u;
    }
    return;
  }

  if (mode == 7) {
    const std::uint32_t part = r.read(6u);
    const auto &pmask = kPartition2[part];
    const std::uint8_t anchor2 = kAnchor2[part];
    std::array<std::array<int, 4u>, 4u> ep{};
    for (int ch = 0; ch < 4; ++ch) {
      for (int e = 0; e < 4; ++e) {
        ep[e][ch] = static_cast<int>(r.read(5u));
      }
    }
    std::array<int, 4u> p{};
    for (int e = 0; e < 4; ++e) {
      p[e] = static_cast<int>(r.read(1u));
    }
    for (int e = 0; e < 4; ++e) {
      for (int ch = 0; ch < 4; ++ch) {
        ep[e][ch] = bc7_unquantize((ep[e][ch] << 1) | p[e], 6);
      }
    }
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t bits =
          (i == 0 || i == static_cast<int>(anchor2)) ? 1u : 2u;
      const std::uint32_t idx = r.read(bits);
      const int sub = pmask[i];
      const int w = kWeight2[idx];
      for (int ch = 0; ch < 4; ++ch) {
        dst_rgba[i * 4 + ch] =
            bc7_interp_u8(ep[sub * 2 + 0][ch], ep[sub * 2 + 1][ch], w);
      }
    }
    return;
  }

  if (mode == 0 || mode == 2) {
    const std::uint32_t part = r.read(mode == 0 ? 4u : 6u);
    const std::uint32_t pmask = kPartition3Masks[part];
    const int a1 = static_cast<int>(kAnchor3_1[part]);
    const int a2 = static_cast<int>(kAnchor3_2[part]);
    std::array<std::array<int, 3u>, 6u> ep{};
    const std::uint32_t cbits = (mode == 0) ? 4u : 5u;
    for (int ch = 0; ch < 3; ++ch) {
      for (int e = 0; e < 6; ++e) {
        ep[e][ch] = static_cast<int>(r.read(cbits));
      }
    }
    for (int e = 0; e < 6; ++e) {
      if (mode == 0) {
        const int p = static_cast<int>(r.read(1u));
        for (int ch = 0; ch < 3; ++ch) {
          ep[e][ch] = bc7_unquantize((ep[e][ch] << 1) | p, 5);
        }
      } else {
        for (int ch = 0; ch < 3; ++ch) {
          ep[e][ch] = bc7_unquantize(ep[e][ch], 5);
        }
      }
    }
    const std::uint32_t max_ibits = (mode == 0) ? 3u : 2u;
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t bits =
          (i == 0 || i == a1 || i == a2) ? (max_ibits - 1u) : max_ibits;
      const std::uint32_t idx = r.read(bits);
      const int sub = static_cast<int>((pmask >> (i * 2)) & 3u);
      const int w = (mode == 0) ? kWeight3[idx] : kWeight2[idx];
      for (int ch = 0; ch < 3; ++ch) {
        dst_rgba[i * 4 + ch] =
            bc7_interp_u8(ep[sub * 2 + 0][ch], ep[sub * 2 + 1][ch], w);
      }
      dst_rgba[i * 4 + 3] = 255u;
    }
    return;
  }

  if (mode == 3) {
    const std::uint32_t part = r.read(6u);
    const auto &pmask = kPartition2[part];
    const int anchor2 = static_cast<int>(kAnchor2[part]);
    std::array<std::array<int, 3u>, 4u> ep{};
    for (int ch = 0; ch < 3; ++ch) {
      for (int e = 0; e < 4; ++e) {
        ep[e][ch] = static_cast<int>(r.read(7u));
      }
    }
    for (int e = 0; e < 4; ++e) {
      const int p = static_cast<int>(r.read(1u));
      for (int ch = 0; ch < 3; ++ch) {
        ep[e][ch] = (ep[e][ch] << 1) | p;
      }
    }
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t bits = (i == 0 || i == anchor2) ? 1u : 2u;
      const std::uint32_t idx = r.read(bits);
      const int sub = pmask[i];
      const int w = kWeight2[idx];
      for (int ch = 0; ch < 3; ++ch) {
        dst_rgba[i * 4 + ch] =
            bc7_interp_u8(ep[sub * 2 + 0][ch], ep[sub * 2 + 1][ch], w);
      }
      dst_rgba[i * 4 + 3] = 255u;
    }
    return;
  }

  if (mode == 4) {
    const std::uint32_t rot = r.read(2u);
    const std::uint32_t idx_mode = r.read(1u);
    std::array<std::array<int, 3u>, 2u> cep{};
    for (int ch = 0; ch < 3; ++ch) {
      cep[0][ch] = bc7_unquantize(static_cast<int>(r.read(5u)), 5);
      cep[1][ch] = bc7_unquantize(static_cast<int>(r.read(5u)), 5);
    }
    const int a0 = bc7_unquantize(static_cast<int>(r.read(6u)), 6);
    const int a1 = bc7_unquantize(static_cast<int>(r.read(6u)), 6);
    std::array<std::uint32_t, 16u> idx2{};
    std::array<std::uint32_t, 16u> idx3{};
    for (int i = 0; i < 16; ++i) {
      idx2[i] = r.read(i == 0 ? 1u : 2u);
    }
    for (int i = 0; i < 16; ++i) {
      idx3[i] = r.read(i == 0 ? 2u : 3u);
    }
    for (int i = 0; i < 16; ++i) {
      std::array<std::uint8_t, 4u> px{};
      const int cw = (idx_mode == 0u) ? kWeight2[idx2[i]] : kWeight3[idx3[i]];
      const int aw = (idx_mode == 0u) ? kWeight3[idx3[i]] : kWeight2[idx2[i]];
      for (int ch = 0; ch < 3; ++ch) {
        px[ch] = bc7_interp_u8(cep[0][ch], cep[1][ch], cw);
      }
      px[3] = bc7_interp_u8(a0, a1, aw);
      if (rot >= 1u && rot <= 3u) {
        std::swap(px[rot - 1u], px[3]);
      }
      std::memcpy(dst_rgba + i * 4, px.data(), 4u);
    }
    return;
  }
}

} // namespace psycles::luisa_backend::detail
