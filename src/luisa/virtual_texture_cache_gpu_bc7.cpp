#include "virtual_texture_cache_gpu_bc.h"

#include <array>
#include <cstdint>

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa;
using namespace luisa::compute;

inline constexpr std::array<int, 4u> kWeight2{0, 21, 43, 64};
inline constexpr std::array<int, 8u> kWeight3{0, 9, 18, 27, 37, 46, 55, 64};
inline constexpr std::array<int, 16u> kWeight4{
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// Full 64-entry BC7 2-subset partition masks & second anchor indices (identical
// to DirectXTex bc7_header). Pixel 0 is always subset 0 anchor.
inline constexpr std::array<std::uint32_t, 64u> kPart2Masks{
    0xccccu, 0x8888u, 0xeeeeu, 0xecc8u, 0xc880u, 0xfeecu, 0xfec8u, 0xec80u,
    0xc800u, 0xffecu, 0xfe80u, 0xe800u, 0xffe8u, 0xff00u, 0xfff0u, 0xf000u,
    0xf710u, 0x008eu, 0x7100u, 0x08ceu, 0x008cu, 0x7310u, 0x3100u, 0x8cceu,
    0x088cu, 0x3110u, 0x6666u, 0x366cu, 0x17e8u, 0x0ff0u, 0x718eu, 0x399cu,
    0xaaaau, 0xf0f0u, 0x5a5au, 0x33ccu, 0x3c3cu, 0x55aau, 0x9696u, 0xa55au,
    0x73ceu, 0x13c8u, 0x324cu, 0x3bdcu, 0x6996u, 0xc33cu, 0x9966u, 0x0660u,
    0x0272u, 0x04e4u, 0x4e40u, 0x2720u, 0xc936u, 0x936cu, 0x39c6u, 0x639cu,
    0x9336u, 0x9cc6u, 0x817eu, 0xe718u, 0xccf0u, 0x0fccu, 0x7744u, 0xee22u};

inline constexpr std::array<std::uint32_t, 64u> kPart2Anchor{
    15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
    15u, 15u, 15u, 15u, 2u,  8u,  2u,  2u,  8u,  8u,  15u, 2u,  8u,
    2u,  2u,  8u,  8u,  2u,  2u,  15u, 15u, 6u,  8u,  2u,  8u,  15u,
    15u, 2u,  8u,  2u,  2u,  2u,  15u, 15u, 6u,  6u,  2u,  6u,  8u,
    15u, 15u, 2u,  2u,  15u, 15u, 15u, 15u, 15u, 2u,  2u,  15u};

// Full 64-entry BC7 3-subset partition masks (2b per pixel) & anchors 1 and 2.
inline constexpr std::array<std::uint32_t, 64u> kPart3Masks{
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

inline constexpr std::array<std::uint32_t, 64u> kPart3Anchor1{
    3u,  3u,  15u, 15u, 8u,  3u,  15u, 15u, 8u,  8u,  6u,  6u,  6u,
    5u,  3u,  3u,  3u,  3u,  8u,  15u, 3u,  3u,  6u,  10u, 5u,  8u,
    8u,  6u,  8u,  5u,  15u, 15u, 8u,  15u, 3u,  5u,  6u,  10u, 8u,
    15u, 15u, 3u,  15u, 5u,  15u, 15u, 15u, 15u, 3u,  15u, 5u,  5u,
    5u,  8u,  5u,  10u, 5u,  10u, 8u,  13u, 15u, 12u, 3u,  3u};

inline constexpr std::array<std::uint32_t, 64u> kPart3Anchor2{
    15u, 8u,  8u,  3u,  15u, 15u, 3u,  8u,  15u, 15u, 15u, 15u, 15u,
    15u, 15u, 8u,  15u, 8u,  15u, 3u,  15u, 8u,  15u, 8u,  3u,  15u,
    6u,  10u, 15u, 15u, 10u, 8u,  15u, 3u,  15u, 10u, 10u, 8u,  9u,
    10u, 6u,  15u, 8u,  15u, 3u,  6u,  6u,  8u,  15u, 3u,  15u, 15u,
    15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 3u,  15u, 15u, 8u};

struct Bc7DslBit128 {
  UInt w0{0u};
  UInt w1{0u};
  UInt w2{0u};
  UInt w3{0u};
  UInt pos{0u};

  void write(UInt val, UInt bits) noexcept {
    const UInt mask = (1u << bits) - 1u;
    const UInt v = val & mask;
    const UInt word = pos >> 5u;
    const UInt sh = pos & 31u;
    const UInt lo_part = v << sh;
    $if(word == 0u) {
      w0 |= lo_part;
      $if(sh + bits > 32u) { w1 |= (v >> (32u - sh)); };
    } $elif(word == 1u) {
      w1 |= lo_part;
      $if(sh + bits > 32u) { w2 |= (v >> (32u - sh)); };
    } $elif(word == 2u) {
      w2 |= lo_part;
      $if(sh + bits > 32u) { w3 |= (v >> (32u - sh)); };
    } $else {
      w3 |= lo_part;
    };
    pos += bits;
  }
};

[[nodiscard]] Int bc7_interp_dsl(Int e0, Int e1, Int w) noexcept {
  return ((64 - w) * e0 + w * e1 + 32) >> 6;
}

void pca_ls_rgb_subset(
    const ArrayFloat<16> &fr,
    const ArrayFloat<16> &fg,
    const ArrayFloat<16> &fb,
    UInt mask,
    bool is_3sub,
    UInt target_sub,
    const ArrayInt<8> &w_arr,
    int max_idx,
    int pca_iters,
    int ls_iters,
    Float3 &out_e0,
    Float3 &out_e1) noexcept {
  Float3 s_mean = make_float3(0.0f);
  Float3 s_min = make_float3(255.0f);
  Float3 s_max = make_float3(0.0f);
  Int cnt = 0;
  $for(i, 16u) {
    const UInt sub =
        is_3sub ? ((mask >> (cast<uint>(i) * 2u)) & 3u)
                : ((mask >> cast<uint>(i)) & 1u);
    $if(sub == target_sub) {
      cnt += 1;
      const Float3 px = make_float3(fr[i], fg[i], fb[i]);
      s_mean += px;
      s_min = min(s_min, px);
      s_max = max(s_max, px);
    };
  };
  $if(cnt <= 0) {
    out_e0 = make_float3(0.0f);
    out_e1 = make_float3(0.0f);
    return;
  };
  s_mean /= cast<float>(cnt);
  out_e0 = s_mean;
  out_e1 = s_mean;

  Float sc00 = 0.0f, sc01 = 0.0f, sc02 = 0.0f;
  Float sc11 = 0.0f, sc12 = 0.0f, sc22 = 0.0f;
  $for(i, 16u) {
    const UInt sub =
        is_3sub ? ((mask >> (cast<uint>(i) * 2u)) & 3u)
                : ((mask >> cast<uint>(i)) & 1u);
    $if(sub == target_sub) {
      const Float d0 = fr[i] - s_mean.x;
      const Float d1 = fg[i] - s_mean.y;
      const Float d2 = fb[i] - s_mean.z;
      sc00 += d0 * d0;
      sc01 += d0 * d1;
      sc02 += d0 * d2;
      sc11 += d1 * d1;
      sc12 += d1 * d2;
      sc22 += d2 * d2;
    };
  };
  Float3 s_axis = s_max - s_min;
  const Float s_n2 = dot(s_axis, s_axis);
  $if(s_n2 > 1e-6f) {
    s_axis /= sqrt(s_n2);
    for (int it = 0; it < pca_iters; ++it) {
      const Float3 nxt = make_float3(
          sc00 * s_axis.x + sc01 * s_axis.y + sc02 * s_axis.z,
          sc01 * s_axis.x + sc11 * s_axis.y + sc12 * s_axis.z,
          sc02 * s_axis.x + sc12 * s_axis.y + sc22 * s_axis.z);
      const Float n2 = dot(nxt, nxt);
      $if(n2 > 1e-8f) { s_axis = nxt / sqrt(n2); };
    }
    Float stmin = 1e30f;
    Float stmax = -1e30f;
    $for(i, 16u) {
      const UInt sub =
          is_3sub ? ((mask >> (cast<uint>(i) * 2u)) & 3u)
                  : ((mask >> cast<uint>(i)) & 1u);
      $if(sub == target_sub) {
        const Float p = dot(make_float3(fr[i], fg[i], fb[i]) - s_mean, s_axis);
        stmin = min(stmin, p);
        stmax = max(stmax, p);
      };
    };
    out_e0 = clamp(s_mean + stmin * s_axis, 0.0f, 255.0f);
    out_e1 = clamp(s_mean + stmax * s_axis, 0.0f, 255.0f);
  };

  const float max_f = static_cast<float>(max_idx);
  for (int it = 0; it < ls_iters; ++it) {
    const Float3 sdiff = out_e1 - out_e0;
    const Float slen2 = dot(sdiff, sdiff);
    $if(slen2 > 1e-6f) {
      const Float inv_l2 = 1.0f / slen2;
      Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
      Float3 rhs0 = make_float3(0.0f);
      Float3 rhs1 = make_float3(0.0f);
      $for(i, 16u) {
        const UInt sub =
            is_3sub ? ((mask >> (cast<uint>(i) * 2u)) & 3u)
                    : ((mask >> cast<uint>(i)) & 1u);
        $if(sub == target_sub) {
          const Float3 px = make_float3(fr[i], fg[i], fb[i]);
          const Float t = clamp(dot(px - out_e0, sdiff) * inv_l2, 0.0f, 1.0f);
          const Int k = clamp(cast<int>(round(t * max_f)), 0, max_idx);
          const Float b = cast<float>(w_arr[k]) * (1.0f / 64.0f);
          const Float a = 1.0f - b;
          a2 += a * a;
          ab += a * b;
          b2 += b * b;
          rhs0 += a * px;
          rhs1 += b * px;
        };
      };
      const Float det = a2 * b2 - ab * ab;
      $if(det > 1e-6f) {
        const Float inv_d = 1.0f / det;
        out_e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
        out_e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
      };
    };
  }
}

void pca_ls_rgba_subset(
    const ArrayFloat<16> &fr,
    const ArrayFloat<16> &fg,
    const ArrayFloat<16> &fb,
    const ArrayFloat<16> &fa,
    UInt pmask,
    UInt target_sub,
    const ArrayInt<4> &w2_arr,
    int pca_iters,
    int ls_iters,
    Float4 &out_e0,
    Float4 &out_e1) noexcept {
  Float4 s_mean = make_float4(0.0f);
  Float4 s_min = make_float4(255.0f);
  Float4 s_max = make_float4(0.0f);
  Int cnt = 0;
  $for(i, 16u) {
    $if(((pmask >> cast<uint>(i)) & 1u) == target_sub) {
      cnt += 1;
      const Float4 px = make_float4(fr[i], fg[i], fb[i], fa[i]);
      s_mean += px;
      s_min = min(s_min, px);
      s_max = max(s_max, px);
    };
  };
  $if(cnt <= 0) {
    out_e0 = make_float4(0.0f);
    out_e1 = make_float4(0.0f);
    return;
  };
  s_mean /= cast<float>(cnt);
  out_e0 = s_mean;
  out_e1 = s_mean;

  Float sc00 = 0.0f, sc01 = 0.0f, sc02 = 0.0f, sc03 = 0.0f;
  Float sc11 = 0.0f, sc12 = 0.0f, sc13 = 0.0f;
  Float sc22 = 0.0f, sc23 = 0.0f, sc33 = 0.0f;
  $for(i, 16u) {
    $if(((pmask >> cast<uint>(i)) & 1u) == target_sub) {
      const Float d0 = fr[i] - s_mean.x;
      const Float d1 = fg[i] - s_mean.y;
      const Float d2 = fb[i] - s_mean.z;
      const Float d3 = fa[i] - s_mean.w;
      sc00 += d0 * d0; sc01 += d0 * d1; sc02 += d0 * d2; sc03 += d0 * d3;
      sc11 += d1 * d1; sc12 += d1 * d2; sc13 += d1 * d3;
      sc22 += d2 * d2; sc23 += d2 * d3; sc33 += d3 * d3;
    };
  };
  Float4 s_axis = s_max - s_min;
  const Float s_n2 = dot(s_axis, s_axis);
  $if(s_n2 > 1e-6f) {
    s_axis /= sqrt(s_n2);
    for (int it = 0; it < pca_iters; ++it) {
      const Float4 nxt = make_float4(
          sc00 * s_axis.x + sc01 * s_axis.y + sc02 * s_axis.z + sc03 * s_axis.w,
          sc01 * s_axis.x + sc11 * s_axis.y + sc12 * s_axis.z + sc13 * s_axis.w,
          sc02 * s_axis.x + sc12 * s_axis.y + sc22 * s_axis.z + sc23 * s_axis.w,
          sc03 * s_axis.x + sc13 * s_axis.y + sc23 * s_axis.z + sc33 * s_axis.w);
      const Float n2 = dot(nxt, nxt);
      $if(n2 > 1e-8f) { s_axis = nxt / sqrt(n2); };
    }
    Float stmin = 1e30f;
    Float stmax = -1e30f;
    $for(i, 16u) {
      $if(((pmask >> cast<uint>(i)) & 1u) == target_sub) {
        const Float p = dot(
            make_float4(fr[i], fg[i], fb[i], fa[i]) - s_mean, s_axis);
        stmin = min(stmin, p);
        stmax = max(stmax, p);
      };
    };
    out_e0 = clamp(s_mean + stmin * s_axis, 0.0f, 255.0f);
    out_e1 = clamp(s_mean + stmax * s_axis, 0.0f, 255.0f);
  };

  for (int it = 0; it < ls_iters; ++it) {
    const Float4 sdiff = out_e1 - out_e0;
    const Float slen2 = dot(sdiff, sdiff);
    $if(slen2 > 1e-6f) {
      const Float inv_l2 = 1.0f / slen2;
      Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
      Float4 rhs0 = make_float4(0.0f);
      Float4 rhs1 = make_float4(0.0f);
      $for(i, 16u) {
        $if(((pmask >> cast<uint>(i)) & 1u) == target_sub) {
          const Float4 px = make_float4(fr[i], fg[i], fb[i], fa[i]);
          const Float t = clamp(dot(px - out_e0, sdiff) * inv_l2, 0.0f, 1.0f);
          const Int k = clamp(cast<int>(round(t * 3.0f)), 0, 3);
          const Float b = cast<float>(w2_arr[k]) * (1.0f / 64.0f);
          const Float a = 1.0f - b;
          a2 += a * a; ab += a * b; b2 += b * b;
          rhs0 += a * px; rhs1 += b * px;
        };
      };
      const Float det = a2 * b2 - ab * ab;
      $if(det > 1e-6f) {
        const Float inv_d = 1.0f / det;
        out_e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
        out_e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
      };
    };
  }
}

void screen_top_part2_dsl(
    const ArrayInt<16> &pr,
    const ArrayInt<16> &pg,
    const ArrayInt<16> &pb,
    const ArrayInt<16> &pa,
    Bool include_alpha,
    const ArrayUInt<64> &part_masks,
    UInt part_limit,
    ArrayUInt<4> &top_p) noexcept {
  ArrayUInt<4> top_s;
  for (int k = 0; k < 4; ++k) {
    top_p[k] = 0u;
    top_s[k] = 0xffffffffu;
  }
  $for(p, part_limit) {
    const UInt pmask = part_masks[p];
    Int mn0r = 255, mn0g = 255, mn0b = 255, mn0a = 255;
    Int mx0r = 0, mx0g = 0, mx0b = 0, mx0a = 0;
    Int mn1r = 255, mn1g = 255, mn1b = 255, mn1a = 255;
    Int mx1r = 0, mx1g = 0, mx1b = 0, mx1a = 0;
    Int c0 = 0, c1 = 0;
    $for(i, 16u) {
      const UInt bit = (pmask >> cast<uint>(i)) & 1u;
      $if(bit == 0u) {
        c0 += 1;
        mn0r = min(mn0r, pr[i]); mx0r = max(mx0r, pr[i]);
        mn0g = min(mn0g, pg[i]); mx0g = max(mx0g, pg[i]);
        mn0b = min(mn0b, pb[i]); mx0b = max(mx0b, pb[i]);
        $if(include_alpha) {
          mn0a = min(mn0a, pa[i]); mx0a = max(mx0a, pa[i]);
        };
      } $else {
        c1 += 1;
        mn1r = min(mn1r, pr[i]); mx1r = max(mx1r, pr[i]);
        mn1g = min(mn1g, pg[i]); mx1g = max(mx1g, pg[i]);
        mn1b = min(mn1b, pb[i]); mx1b = max(mx1b, pb[i]);
        $if(include_alpha) {
          mn1a = min(mn1a, pa[i]); mx1a = max(mx1a, pa[i]);
        };
      };
    };
    const Int dr0 = max(mx0r - mn0r, 0), dg0 = max(mx0g - mn0g, 0);
    const Int db0 = max(mx0b - mn0b, 0), da0 = ite(include_alpha, max(mx0a - mn0a, 0), 0);
    const Int dr1 = max(mx1r - mn1r, 0), dg1 = max(mx1g - mn1g, 0);
    const Int db1 = max(mx1b - mn1b, 0), da1 = ite(include_alpha, max(mx1a - mn1a, 0), 0);
    const UInt score =
        cast<uint>(c0 * (dr0 * dr0 + dg0 * dg0 + db0 * db0 + da0 * da0) +
                   c1 * (dr1 * dr1 + dg1 * dg1 + db1 * db1 + da1 * da1));
    UInt cp = cast<uint>(p);
    UInt cs = score;
    for (int r = 0; r < 4; ++r) {
      $if(cs < top_s[r]) {
        const UInt ts = top_s[r];
        const UInt tp = top_p[r];
        top_s[r] = cs;
        top_p[r] = cp;
        cs = ts;
        cp = tp;
      };
    }
  };
}

} // namespace

VtGpuBcBlockShader compile_vt_gpu_bc7_shader(luisa::compute::Device &device) {
  return device.compile<1>(
      [](BufferVar<luisa::uint> in_tiles, BufferVar<luisa::uint> out_blocks,
         UInt P, UInt total_blocks, UInt quality) noexcept {
        const UInt gid = dispatch_x();
        $if(gid >= total_blocks) { $return(); };

        const UInt bx_count = P >> 2u;
        const UInt blocks_per_tile = bx_count * bx_count;
        const UInt tile_idx = gid / blocks_per_tile;
        const UInt local_blk = gid - tile_idx * blocks_per_tile;
        const UInt blk_y = local_blk / bx_count;
        const UInt blk_x = local_blk - blk_y * bx_count;
        const UInt tile_base = tile_idx * P * P;
        const UInt base_x = blk_x * 4u;
        const UInt base_y = blk_y * 4u;

        ArrayInt<16> pr, pg, pb, pa;
        Int min_r = 255, max_r = 0;
        Int min_g = 255, max_g = 0;
        Int min_b = 255, max_b = 0;
        Int min_a = 255, max_a = 0;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const UInt packed =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            const Int r = cast<int>(packed & 0xffu);
            const Int g = cast<int>((packed >> 8u) & 0xffu);
            const Int b = cast<int>((packed >> 16u) & 0xffu);
            const Int a = cast<int>((packed >> 24u) & 0xffu);
            pr[i] = r;
            pg[i] = g;
            pb[i] = b;
            pa[i] = a;
            min_r = min(min_r, r);
            max_r = max(max_r, r);
            min_g = min(min_g, g);
            max_g = max(max_g, g);
            min_b = min(min_b, b);
            max_b = max(max_b, b);
            min_a = min(min_a, a);
            max_a = max(max_a, a);
          }
        }

        const Bool all_same = (min_r == max_r) & (min_g == max_g) &
                              (min_b == max_b) & (min_a == max_a);
        $if(all_same) {
          const UInt lsb_or = cast<uint>(min_r | min_g | min_b | min_a) & 1u;
          const UInt lsb_and = cast<uint>(min_r & min_g & min_b & min_a) & 1u;
          $if(lsb_or == lsb_and) {
            Bc7DslBit128 bw;
            bw.write(0x40u, 7u);
            bw.write(cast<uint>(min_r) >> 1u, 7u);
            bw.write(cast<uint>(min_r) >> 1u, 7u);
            bw.write(cast<uint>(min_g) >> 1u, 7u);
            bw.write(cast<uint>(min_g) >> 1u, 7u);
            bw.write(cast<uint>(min_b) >> 1u, 7u);
            bw.write(cast<uint>(min_b) >> 1u, 7u);
            bw.write(cast<uint>(min_a) >> 1u, 7u);
            bw.write(cast<uint>(min_a) >> 1u, 7u);
            bw.write(lsb_and, 1u);
            bw.write(lsb_and, 1u);
            out_blocks.write(gid * 4u + 0u, bw.w0);
            out_blocks.write(gid * 4u + 1u, bw.w1);
            out_blocks.write(gid * 4u + 2u, bw.w2);
            out_blocks.write(gid * 4u + 3u, bw.w3);
            $return();
          };
          UInt best_sse_s = 0xffffffffu;
          ArrayUInt<4> best_c0, best_c1;
          UInt best_p0 = 0u, best_p1 = 1u;
          for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
            best_c0[ch] = 0u;
            best_c1[ch] = 0u;
          }
          ArrayInt<4> target_ch;
          target_ch[0] = min_r;
          target_ch[1] = min_g;
          target_ch[2] = min_b;
          target_ch[3] = min_a;
          for (std::uint32_t p0 = 0u; p0 <= 1u; ++p0) {
            const std::uint32_t p1 = 1u - p0;
            ArrayUInt<4> tc0, tc1;
            UInt sse1 = 0u;
            for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
              const Int t = target_ch[ch];
              const Int base0 = clamp((t - static_cast<int>(p0)) >> 1, 0, 127);
              const Int base1 = clamp((t - static_cast<int>(p1)) >> 1, 0, 127);
              Int ch_best = 0x7fffffff;
              Int ch_c0 = base0;
              Int ch_c1 = base1;
              for (int d0 = -1; d0 <= 1; ++d0) {
                const Int c0v = clamp(base0 + d0, 0, 127);
                const Int e0 = (c0v << 1) | static_cast<int>(p0);
                for (int d1 = -1; d1 <= 1; ++d1) {
                  const Int c1v = clamp(base1 + d1, 0, 127);
                  const Int e1 = (c1v << 1) | static_cast<int>(p1);
                  const Int dec = bc7_interp_dsl(e0, e1, kWeight4[7]);
                  const Int diff = dec - t;
                  const Int err = diff * diff;
                  $if(err < ch_best) {
                    ch_best = err;
                    ch_c0 = c0v;
                    ch_c1 = c1v;
                  };
                }
              }
              tc0[ch] = cast<uint>(ch_c0);
              tc1[ch] = cast<uint>(ch_c1);
              sse1 += cast<uint>(ch_best);
            }
            const UInt total_sse = sse1 * 16u;
            $if(total_sse < best_sse_s) {
              best_sse_s = total_sse;
              best_p0 = p0;
              best_p1 = p1;
              for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
                best_c0[ch] = tc0[ch];
                best_c1[ch] = tc1[ch];
              }
            };
          }
          Bc7DslBit128 bw;
          bw.write(0x40u, 7u);
          for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
            bw.write(best_c0[ch], 7u);
            bw.write(best_c1[ch], 7u);
          }
          bw.write(best_p0, 1u);
          bw.write(best_p1, 1u);
          bw.write(7u, 3u);
          for (std::uint32_t i = 1u; i < 16u; ++i) {
            bw.write(7u, 4u);
          }
          out_blocks.write(gid * 4u + 0u, bw.w0);
          out_blocks.write(gid * 4u + 1u, bw.w1);
          out_blocks.write(gid * 4u + 2u, bw.w2);
          out_blocks.write(gid * 4u + 3u, bw.w3);
          $return();
        };

        // ---- Mode 6: 1 subset 7.7.7.7 + p0,p1 + 4-bit indices ----
        ArrayFloat<16> fr, fg, fb, fa;
        Float4 mean4 = make_float4(0.0f);
        $for(i, 16u) {
          const Float r = cast<float>(pr[i]);
          const Float g = cast<float>(pg[i]);
          const Float b = cast<float>(pb[i]);
          const Float a = cast<float>(pa[i]);
          fr[i] = r;
          fg[i] = g;
          fb[i] = b;
          fa[i] = a;
          mean4 += make_float4(r, g, b, a);
        };
        mean4 *= (1.0f / 16.0f);

        Float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
        Float c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
        Float c22 = 0.0f, c23 = 0.0f, c33 = 0.0f;
        $for(i, 16u) {
          const Float d0 = fr[i] - mean4.x;
          const Float d1 = fg[i] - mean4.y;
          const Float d2 = fb[i] - mean4.z;
          const Float d3 = fa[i] - mean4.w;
          c00 += d0 * d0; c01 += d0 * d1; c02 += d0 * d2; c03 += d0 * d3;
          c11 += d1 * d1; c12 += d1 * d2; c13 += d1 * d3;
          c22 += d2 * d2; c23 += d2 * d3; c33 += d3 * d3;
        };
        Float4 axis = make_float4(cast<float>(max_r - min_r),
                                  cast<float>(max_g - min_g),
                                  cast<float>(max_b - min_b),
                                  cast<float>(max_a - min_a));
        const Float norm2 = dot(axis, axis);
        $if(norm2 > 1e-6f) {
          axis /= sqrt(norm2);
          for (int it = 0; it < 4; ++it) {
            const Float4 nxt = make_float4(
                c00 * axis.x + c01 * axis.y + c02 * axis.z + c03 * axis.w,
                c01 * axis.x + c11 * axis.y + c12 * axis.z + c13 * axis.w,
                c02 * axis.x + c12 * axis.y + c22 * axis.z + c23 * axis.w,
                c03 * axis.x + c13 * axis.y + c23 * axis.z + c33 * axis.w);
            const Float n2 = dot(nxt, nxt);
            $if(n2 > 1e-8f) { axis = nxt / sqrt(n2); };
          }
        } $else {
          axis = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        };

        Float t_min = 1e30f;
        Float t_max = -1e30f;
        $for(i, 16u) {
          const Float4 d =
              make_float4(fr[i], fg[i], fb[i], fa[i]) - mean4;
          const Float proj = dot(d, axis);
          t_min = min(t_min, proj);
          t_max = max(t_max, proj);
        };
        Float4 e0 = clamp(mean4 + t_min * axis, 0.0f, 255.0f);
        Float4 e1 = clamp(mean4 + t_max * axis, 0.0f, 255.0f);

        ArrayInt<16> w4_arr;
        for (std::uint32_t k = 0u; k < 16u; ++k) {
          w4_arr[k] = kWeight4[k];
        }
        ArrayInt<8> w3_arr;
        for (std::uint32_t k = 0u; k < 8u; ++k) {
          w3_arr[k] = kWeight3[k];
        }
        ArrayInt<4> w2_arr;
        for (std::uint32_t k = 0u; k < 4u; ++k) {
          w2_arr[k] = kWeight2[k];
        }

        for (int it = 0; it < 2; ++it) {
          const Float4 diff = e1 - e0;
          const Float len2 = dot(diff, diff);
          $if(len2 > 1e-6f) {
            const Float inv_len2 = 1.0f / len2;
            Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
            Float4 rhs0 = make_float4(0.0f);
            Float4 rhs1 = make_float4(0.0f);
            $for(i, 16u) {
              const Float4 px = make_float4(fr[i], fg[i], fb[i], fa[i]);
              const Float t = clamp(dot(px - e0, diff) * inv_len2, 0.0f, 1.0f);
              const Int k_est = clamp(cast<int>(round(t * 15.0f)), 0, 15);
              Int best_k = k_est;
              Float best_d = 1e30f;
              for (int off = -1; off <= 1; ++off) {
                const Int k = clamp(k_est + off, 0, 15);
                const Float wf = cast<float>(w4_arr[k]) * (1.0f / 64.0f);
                const Float4 dc = (e0 + wf * diff) - px;
                const Float d2 = dot(dc, dc);
                $if(d2 < best_d) {
                  best_d = d2;
                  best_k = k;
                };
              }
              const Float b = cast<float>(w4_arr[best_k]) * (1.0f / 64.0f);
              const Float a = 1.0f - b;
              a2 += a * a; ab += a * b; b2 += b * b;
              rhs0 += a * px; rhs1 += b * px;
            };
            const Float det = a2 * b2 - ab * ab;
            $if(det > 1e-6f) {
              const Float inv_det = 1.0f / det;
              e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_det, 0.0f, 255.0f);
              e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_det, 0.0f, 255.0f);
            };
          };
        }

        UInt best_sse = 0xffffffffu;
        Bc7DslBit128 best_bits;
        ArrayUInt<4> m6_c0, m6_c1;
        UInt m6_p0 = 0u, m6_p1 = 0u;
        ArrayUInt<16> m6_idx;
        $for(ch, 4u) { m6_c0[ch] = 0u; m6_c1[ch] = 0u; };
        $for(i, 16u) { m6_idx[i] = 0u; };

        ArrayFloat<4> e0_arr, e1_arr;
        e0_arr[0] = e0.x; e0_arr[1] = e0.y; e0_arr[2] = e0.z; e0_arr[3] = e0.w;
        e1_arr[0] = e1.x; e1_arr[1] = e1.y; e1_arr[2] = e1.z; e1_arr[3] = e1.w;

        $for(p0, 2u) {
          ArrayInt<4> c0_arr, qe0_arr;
          const Float pf0 = cast<float>(p0);
          $for(ch, 4u) {
            c0_arr[ch] = clamp(cast<int>(round((e0_arr[ch] - pf0) * 0.5f)), 0, 127);
            qe0_arr[ch] = (c0_arr[ch] << 1) | cast<int>(p0);
          };
          $for(p1, 2u) {
            ArrayInt<4> c1_arr, qe1_arr;
            const Float pf1 = cast<float>(p1);
            $for(ch, 4u) {
              c1_arr[ch] = clamp(cast<int>(round((e1_arr[ch] - pf1) * 0.5f)), 0, 127);
              qe1_arr[ch] = (c1_arr[ch] << 1) | cast<int>(p1);
            };
            ArrayInt<16> pal_r, pal_g, pal_b, pal_a;
            $for(k, 16u) {
              const Int w = w4_arr[k];
              pal_r[k] = bc7_interp_dsl(qe0_arr[0], qe1_arr[0], w);
              pal_g[k] = bc7_interp_dsl(qe0_arr[1], qe1_arr[1], w);
              pal_b[k] = bc7_interp_dsl(qe0_arr[2], qe1_arr[2], w);
              pal_a[k] = bc7_interp_dsl(qe0_arr[3], qe1_arr[3], w);
            };
            UInt sse = 0u;
            ArrayUInt<16> trial_idx;
            $for(i, 16u) {
              const Int r = pr[i], g = pg[i], b = pb[i], a = pa[i];
              UInt bk = 0u;
              Int be = 0x7fffffff;
              $for(k, 16u) {
                const Int dr = pal_r[k] - r;
                const Int dg = pal_g[k] - g;
                const Int db = pal_b[k] - b;
                const Int da = pal_a[k] - a;
                const Int e = dr * dr + dg * dg + db * db + da * da;
                $if(e < be) { be = e; bk = cast<uint>(k); };
              };
              trial_idx[i] = bk;
              sse += cast<uint>(be);
            };
            $if(sse < best_sse) {
              best_sse = sse;
              m6_p0 = cast<uint>(p0);
              m6_p1 = cast<uint>(p1);
              $for(ch, 4u) {
                m6_c0[ch] = cast<uint>(c0_arr[ch]);
                m6_c1[ch] = cast<uint>(c1_arr[ch]);
              };
              $for(i, 16u) { m6_idx[i] = trial_idx[i]; };
            };
          };
        };
        $if(m6_idx[0] >= 8u) {
          for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
            const UInt tmp = m6_c0[ch];
            m6_c0[ch] = m6_c1[ch];
            m6_c1[ch] = tmp;
          }
          const UInt tp = m6_p0; m6_p0 = m6_p1; m6_p1 = tp;
          $for(i, 16u) { m6_idx[i] = 15u - m6_idx[i]; };
        };
        best_bits.write(0x40u, 7u);
        for (std::uint32_t ch = 0u; ch < 4u; ++ch) {
          best_bits.write(m6_c0[ch], 7u);
          best_bits.write(m6_c1[ch], 7u);
        }
        best_bits.write(m6_p0, 1u);
        best_bits.write(m6_p1, 1u);
        best_bits.write(m6_idx[0], 3u);
        for (std::uint32_t i = 1u; i < 16u; ++i) {
          best_bits.write(m6_idx[i], 4u);
        }

        const Bool all_opaque = (min_a == 255) & (max_a == 255);
        const Bool has_alpha_var = (min_a != max_a);

        // ---- Mode 5 & Mode 4 (1-subset decorrelated RGB + scalar A) ----
        // Reuse rotated channel decomposition cleanly across Mode 5 and Mode 4.
        {
          const Bool try_m4 = (quality >= 1u) & (best_sse > 64u);
          const UInt m5_rot_thresh = ite(quality >= 2u, 128u, 768u);
          const Bool eval_all_rot =
              has_alpha_var | (best_sse > m5_rot_thresh) | (try_m4 & (best_sse > 256u));
          const UInt rot_count = ite(eval_all_rot, 4u, 1u);

          $for(rot, rot_count) {
            ArrayInt<16> ic0, ic1, ic2, iind;
            ArrayFloat<16> fc0, fc1, fc2, find_arr;
            Float3 c_mean = make_float3(0.0f);
            Float3 c_min = make_float3(255.0f);
            Float3 c_max = make_float3(0.0f);
            Int a_min_r = 255, a_max_r = 0;
            $for(i, 16u) {
              Int c0v = pr[i], c1v = pg[i], c2v = pb[i], indv = pa[i];
              $if(rot == 1u) {
                const Int t = c0v; c0v = indv; indv = t;
              } $elif(rot == 2u) {
                const Int t = c1v; c1v = indv; indv = t;
              } $elif(rot == 3u) {
                const Int t = c2v; c2v = indv; indv = t;
              };
              ic0[i] = c0v; ic1[i] = c1v; ic2[i] = c2v; iind[i] = indv;
              const Float3 cf = make_float3(
                  cast<float>(c0v), cast<float>(c1v), cast<float>(c2v));
              fc0[i] = cf.x; fc1[i] = cf.y; fc2[i] = cf.z;
              find_arr[i] = cast<float>(indv);
              c_mean += cf;
              c_min = min(c_min, cf); c_max = max(c_max, cf);
              a_min_r = min(a_min_r, indv); a_max_r = max(a_max_r, indv);
            };
            c_mean *= (1.0f / 16.0f);

            // PCA axis + min/max span on the 3 color channels
            Float rc00 = 0.0f, rc01 = 0.0f, rc02 = 0.0f;
            Float rc11 = 0.0f, rc12 = 0.0f, rc22 = 0.0f;
            $for(i, 16u) {
              const Float d0 = fc0[i] - c_mean.x;
              const Float d1 = fc1[i] - c_mean.y;
              const Float d2 = fc2[i] - c_mean.z;
              rc00 += d0 * d0; rc01 += d0 * d1; rc02 += d0 * d2;
              rc11 += d1 * d1; rc12 += d1 * d2; rc22 += d2 * d2;
            };
            Float3 c_axis = c_max - c_min;
            Float3 base_ce0 = c_mean, base_ce1 = c_mean;
            const Float cn2 = dot(c_axis, c_axis);
            $if(cn2 > 1e-6f) {
              c_axis /= sqrt(cn2);
              for (int it = 0; it < 3; ++it) {
                const Float3 nxt = make_float3(
                    rc00 * c_axis.x + rc01 * c_axis.y + rc02 * c_axis.z,
                    rc01 * c_axis.x + rc11 * c_axis.y + rc12 * c_axis.z,
                    rc02 * c_axis.x + rc12 * c_axis.y + rc22 * c_axis.z);
                const Float n2 = dot(nxt, nxt);
                $if(n2 > 1e-8f) { c_axis = nxt / sqrt(n2); };
              }
              Float ctmin = 1e30f, ctmax = -1e30f;
              $for(i, 16u) {
                const Float p =
                    dot(make_float3(fc0[i], fc1[i], fc2[i]) - c_mean, c_axis);
                ctmin = min(ctmin, p); ctmax = max(ctmax, p);
              };
              base_ce0 = clamp(c_mean + ctmin * c_axis, 0.0f, 255.0f);
              base_ce1 = clamp(c_mean + ctmax * c_axis, 0.0f, 255.0f);
            };

            auto refine_3ch_ls = [&](Float3 &ce0, Float3 &ce1,
                                     const ArrayInt<8> &wa, int max_i) {
              const Float3 cdiff = ce1 - ce0;
              const Float clen2 = dot(cdiff, cdiff);
              $if(clen2 > 1e-6f) {
                const Float inv_cl2 = 1.0f / clen2;
                Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
                Float3 rhs0 = make_float3(0.0f), rhs1 = make_float3(0.0f);
                $for(i, 16u) {
                  const Float3 px = make_float3(fc0[i], fc1[i], fc2[i]);
                  const Float t = clamp(dot(px - ce0, cdiff) * inv_cl2, 0.0f, 1.0f);
                  const Int k = clamp(
                      cast<int>(round(t * static_cast<float>(max_i))), 0, max_i);
                  const Float b = cast<float>(wa[k]) * (1.0f / 64.0f);
                  const Float a = 1.0f - b;
                  a2 += a * a; ab += a * b; b2 += b * b;
                  rhs0 += a * px; rhs1 += b * px;
                };
                const Float det = a2 * b2 - ab * ab;
                $if(det > 1e-6f) {
                  const Float inv_d = 1.0f / det;
                  ce0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
                  ce1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
                };
              };
            };

            auto eval_scalar_2b = [&](Float &ae0, Float &ae1) {
              $if(a_max_r > a_min_r) {
                const Float inv_ds = 1.0f / cast<float>(a_max_r - a_min_r);
                Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f, rhs0 = 0.0f, rhs1 = 0.0f;
                $for(i, 16u) {
                  const Int k = clamp(
                      cast<int>(round((find_arr[i] - cast<float>(a_min_r)) *
                                      inv_ds * 3.0f)),
                      0, 3);
                  const Float b = cast<float>(w2_arr[k]) * (1.0f / 64.0f);
                  const Float a = 1.0f - b;
                  a2 += a * a; ab += a * b; b2 += b * b;
                  rhs0 += a * find_arr[i]; rhs1 += b * find_arr[i];
                };
                const Float det = a2 * b2 - ab * ab;
                $if(det > 1e-6f) {
                  const Float inv_d = 1.0f / det;
                  ae0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
                  ae1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
                };
              };
            };

            // ---- Mode 5 (7.7.7 + 8A, 2b color + 2b alpha) ----
            {
              Float ae0 = cast<float>(a_min_r), ae1 = cast<float>(a_max_r);
              eval_scalar_2b(ae0, ae1);
              Int qa0_i = clamp(cast<int>(round(ae0)), 0, 255);
              Int qa1_i = clamp(cast<int>(round(ae1)), 0, 255);
              ArrayInt<4> apal;
              $for(k, 4u) { apal[k] = bc7_interp_dsl(qa0_i, qa1_i, w2_arr[k]); };
              UInt a_sse = 0u;
              ArrayUInt<16> a_idx;
              $for(i, 16u) {
                const Int av = iind[i];
                UInt bk = 0u; Int be = 0x7fffffff;
                $for(k, 4u) {
                  const Int d = apal[k] - av;
                  const Int e = d * d;
                  $if(e < be) { be = e; bk = cast<uint>(k); };
                };
                a_idx[i] = bk; a_sse += cast<uint>(be);
              };

              Float3 m5_e0 = base_ce0, m5_e1 = base_ce1;
              {
                const Float3 cdiff = m5_e1 - m5_e0;
                const Float clen2 = dot(cdiff, cdiff);
                $if(clen2 > 1e-6f) {
                  const Float inv_cl2 = 1.0f / clen2;
                  Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
                  Float3 rhs0 = make_float3(0.0f), rhs1 = make_float3(0.0f);
                  $for(i, 16u) {
                    const Float3 px = make_float3(fc0[i], fc1[i], fc2[i]);
                    const Int k = clamp(
                        cast<int>(round(clamp(dot(px - m5_e0, cdiff) * inv_cl2, 0.0f, 1.0f) * 3.0f)),
                        0, 3);
                    const Float b = cast<float>(w2_arr[k]) * (1.0f / 64.0f);
                    const Float a = 1.0f - b;
                    a2 += a * a; ab += a * b; b2 += b * b;
                    rhs0 += a * px; rhs1 += b * px;
                  };
                  const Float det = a2 * b2 - ab * ab;
                  $if(det > 1e-6f) {
                    const Float inv_d = 1.0f / det;
                    m5_e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
                    m5_e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
                  };
                };
              }
              ArrayUInt<3> q0_7, q1_7;
              ArrayInt<3> qe0, qe1;
              q0_7[0] = cast<uint>(clamp(cast<int>(round(m5_e0.x * (127.0f / 255.0f))), 0, 127));
              q0_7[1] = cast<uint>(clamp(cast<int>(round(m5_e0.y * (127.0f / 255.0f))), 0, 127));
              q0_7[2] = cast<uint>(clamp(cast<int>(round(m5_e0.z * (127.0f / 255.0f))), 0, 127));
              q1_7[0] = cast<uint>(clamp(cast<int>(round(m5_e1.x * (127.0f / 255.0f))), 0, 127));
              q1_7[1] = cast<uint>(clamp(cast<int>(round(m5_e1.y * (127.0f / 255.0f))), 0, 127));
              q1_7[2] = cast<uint>(clamp(cast<int>(round(m5_e1.z * (127.0f / 255.0f))), 0, 127));
              $for(ch, 3u) {
                const Int v0 = cast<int>(q0_7[ch]);
                const Int v1 = cast<int>(q1_7[ch]);
                qe0[ch] = (v0 << 1) | (v0 >> 6);
                qe1[ch] = (v1 << 1) | (v1 >> 6);
              };
              ArrayInt<4> cpal_r, cpal_g, cpal_b;
              $for(k, 4u) {
                const Int w = w2_arr[k];
                cpal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                cpal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                cpal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
              };
              UInt c_sse = 0u;
              ArrayUInt<16> c_idx;
              $for(i, 16u) {
                const Int r = ic0[i], g = ic1[i], b = ic2[i];
                UInt bk = 0u; Int be = 0x7fffffff;
                $for(k, 4u) {
                  const Int dr = cpal_r[k] - r;
                  const Int dg = cpal_g[k] - g;
                  const Int db = cpal_b[k] - b;
                  const Int e = dr * dr + dg * dg + db * db;
                  $if(e < be) { be = e; bk = cast<uint>(k); };
                };
                c_idx[i] = bk; c_sse += cast<uint>(be);
              };
              const UInt m5_sse = c_sse + a_sse;
              $if(m5_sse < best_sse) {
                best_sse = m5_sse;
                $if(c_idx[0] >= 2u) {
                  $for(ch, 3u) {
                    const UInt t = q0_7[ch]; q0_7[ch] = q1_7[ch]; q1_7[ch] = t;
                  };
                  $for(i, 16u) { c_idx[i] = 3u - c_idx[i]; };
                };
                UInt qa0 = cast<uint>(qa0_i), qa1 = cast<uint>(qa1_i);
                $if(a_idx[0] >= 2u) {
                  const UInt t = qa0; qa0 = qa1; qa1 = t;
                  $for(i, 16u) { a_idx[i] = 3u - a_idx[i]; };
                };
                Bc7DslBit128 m5_bits;
                m5_bits.write(0x20u, 6u);
                m5_bits.write(rot, 2u);
                for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                  m5_bits.write(q0_7[ch], 7u);
                  m5_bits.write(q1_7[ch], 7u);
                }
                m5_bits.write(qa0, 8u);
                m5_bits.write(qa1, 8u);
                m5_bits.write(c_idx[0], 1u);
                for (std::uint32_t i = 1u; i < 16u; ++i) {
                  m5_bits.write(c_idx[i], 2u);
                }
                m5_bits.write(a_idx[0], 1u);
                for (std::uint32_t i = 1u; i < 16u; ++i) {
                  m5_bits.write(a_idx[i], 2u);
                }
                best_bits = m5_bits;
              };
            }

            // ---- Mode 4: 1-subset 5.5.5 + 6A with idxMode 0 (2bC/3bA) & 1 (3bC/2bA) ----
            $if(try_m4 & ((rot == 0u) | has_alpha_var | (best_sse > 256u))) {
              for (std::uint32_t idx_mode = 0u; idx_mode < 2u; ++idx_mode) {
                Float3 ce0 = base_ce0, ce1 = base_ce1;
                if (idx_mode == 1u) {
                  refine_3ch_ls(ce0, ce1, w3_arr, 7);
                } else {
                  const Float3 cdiff = ce1 - ce0;
                  const Float clen2 = dot(cdiff, cdiff);
                  $if(clen2 > 1e-6f) {
                    const Float inv_cl2 = 1.0f / clen2;
                    Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
                    Float3 rhs0 = make_float3(0.0f), rhs1 = make_float3(0.0f);
                    $for(i, 16u) {
                      const Float3 px = make_float3(fc0[i], fc1[i], fc2[i]);
                      const Int k = clamp(
                          cast<int>(round(clamp(dot(px - ce0, cdiff) * inv_cl2, 0.0f, 1.0f) * 3.0f)),
                          0, 3);
                      const Float b = cast<float>(w2_arr[k]) * (1.0f / 64.0f);
                      const Float a = 1.0f - b;
                      a2 += a * a; ab += a * b; b2 += b * b;
                      rhs0 += a * px; rhs1 += b * px;
                    };
                    const Float det = a2 * b2 - ab * ab;
                    $if(det > 1e-6f) {
                      const Float inv_d = 1.0f / det;
                      ce0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
                      ce1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
                    };
                  };
                }
                ArrayUInt<3> q0_5, q1_5;
                ArrayInt<3> qe0, qe1;
                q0_5[0] = cast<uint>(clamp(cast<int>(round(ce0.x * (31.0f / 255.0f))), 0, 31));
                q0_5[1] = cast<uint>(clamp(cast<int>(round(ce0.y * (31.0f / 255.0f))), 0, 31));
                q0_5[2] = cast<uint>(clamp(cast<int>(round(ce0.z * (31.0f / 255.0f))), 0, 31));
                q1_5[0] = cast<uint>(clamp(cast<int>(round(ce1.x * (31.0f / 255.0f))), 0, 31));
                q1_5[1] = cast<uint>(clamp(cast<int>(round(ce1.y * (31.0f / 255.0f))), 0, 31));
                q1_5[2] = cast<uint>(clamp(cast<int>(round(ce1.z * (31.0f / 255.0f))), 0, 31));
                $for(ch, 3u) {
                  const Int v0 = cast<int>(q0_5[ch]);
                  const Int v1 = cast<int>(q1_5[ch]);
                  qe0[ch] = (v0 << 3) | (v0 >> 2);
                  qe1[ch] = (v1 << 3) | (v1 >> 2);
                };

                const std::uint32_t c_levels = (idx_mode == 1u) ? 8u : 4u;
                const std::uint32_t a_levels = (idx_mode == 0u) ? 8u : 4u;

                ArrayInt<8> cpal_r, cpal_g, cpal_b;
                $for(k, c_levels) {
                  const Int w = (idx_mode == 1u) ? w3_arr[k] : w2_arr[k];
                  cpal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                  cpal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                  cpal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
                };
                UInt c_sse = 0u;
                ArrayUInt<16> c_idx;
                $for(i, 16u) {
                  const Int r = ic0[i], g = ic1[i], b = ic2[i];
                  UInt bk = 0u; Int be = 0x7fffffff;
                  $for(k, c_levels) {
                    const Int dr = cpal_r[k] - r;
                    const Int dg = cpal_g[k] - g;
                    const Int db = cpal_b[k] - b;
                    const Int e = dr * dr + dg * dg + db * db;
                    $if(e < be) { be = e; bk = cast<uint>(k); };
                  };
                  c_idx[i] = bk; c_sse += cast<uint>(be);
                };

                Float ae0 = cast<float>(a_min_r), ae1 = cast<float>(a_max_r);
                $if(a_max_r > a_min_r) {
                  const Float inv_ds = 1.0f / cast<float>(a_max_r - a_min_r);
                  const float a_maxf = static_cast<float>(a_levels - 1u);
                  Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f, rhs0 = 0.0f, rhs1 = 0.0f;
                  $for(i, 16u) {
                    const Int k = clamp(
                        cast<int>(round((find_arr[i] - cast<float>(a_min_r)) *
                                        inv_ds * a_maxf)),
                        0, static_cast<int>(a_levels - 1u));
                    const Int w = (idx_mode == 0u) ? w3_arr[k] : w2_arr[k];
                    const Float b = cast<float>(w) * (1.0f / 64.0f);
                    const Float a = 1.0f - b;
                    a2 += a * a; ab += a * b; b2 += b * b;
                    rhs0 += a * find_arr[i]; rhs1 += b * find_arr[i];
                  };
                  const Float det = a2 * b2 - ab * ab;
                  $if(det > 1e-6f) {
                    const Float inv_d = 1.0f / det;
                    ae0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
                    ae1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
                  };
                };
                UInt qa0_6 = cast<uint>(clamp(cast<int>(round(ae0 * (63.0f / 255.0f))), 0, 63));
                UInt qa1_6 = cast<uint>(clamp(cast<int>(round(ae1 * (63.0f / 255.0f))), 0, 63));
                const Int qae0 = (cast<int>(qa0_6) << 2) | (cast<int>(qa0_6) >> 4);
                const Int qae1 = (cast<int>(qa1_6) << 2) | (cast<int>(qa1_6) >> 4);
                ArrayInt<8> apal;
                $for(k, a_levels) {
                  const Int w = (idx_mode == 0u) ? w3_arr[k] : w2_arr[k];
                  apal[k] = bc7_interp_dsl(qae0, qae1, w);
                };
                UInt a_sse = 0u;
                ArrayUInt<16> a_idx;
                $for(i, 16u) {
                  const Int av = iind[i];
                  UInt bk = 0u; Int be = 0x7fffffff;
                  $for(k, a_levels) {
                    const Int d = apal[k] - av;
                    const Int e = d * d;
                    $if(e < be) { be = e; bk = cast<uint>(k); };
                  };
                  a_idx[i] = bk; a_sse += cast<uint>(be);
                };

                const UInt m4_sse = c_sse + a_sse;
                $if(m4_sse < best_sse) {
                  best_sse = m4_sse;
                  $if(c_idx[0] >= (c_levels / 2u)) {
                    $for(ch, 3u) {
                      const UInt t = q0_5[ch]; q0_5[ch] = q1_5[ch]; q1_5[ch] = t;
                    };
                    $for(i, 16u) { c_idx[i] = (c_levels - 1u) - c_idx[i]; };
                  };
                  $if(a_idx[0] >= (a_levels / 2u)) {
                    const UInt t = qa0_6; qa0_6 = qa1_6; qa1_6 = t;
                    $for(i, 16u) { a_idx[i] = (a_levels - 1u) - a_idx[i]; };
                  };
                  Bc7DslBit128 m4_bits;
                  m4_bits.write(0x10u, 5u);
                  m4_bits.write(rot, 2u);
                  m4_bits.write(idx_mode, 1u);
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    m4_bits.write(q0_5[ch], 5u);
                    m4_bits.write(q1_5[ch], 5u);
                  }
                  m4_bits.write(qa0_6, 6u);
                  m4_bits.write(qa1_6, 6u);
                  // 31 bits of 2-bit indices first, then 47 bits of 3-bit indices.
                  if (idx_mode == 0u) {
                    m4_bits.write(c_idx[0], 1u);
                    for (std::uint32_t i = 1u; i < 16u; ++i) {
                      m4_bits.write(c_idx[i], 2u);
                    }
                    m4_bits.write(a_idx[0], 2u);
                    for (std::uint32_t i = 1u; i < 16u; ++i) {
                      m4_bits.write(a_idx[i], 3u);
                    }
                  } else {
                    m4_bits.write(a_idx[0], 1u);
                    for (std::uint32_t i = 1u; i < 16u; ++i) {
                      m4_bits.write(a_idx[i], 2u);
                    }
                    m4_bits.write(c_idx[0], 2u);
                    for (std::uint32_t i = 1u; i < 16u; ++i) {
                      m4_bits.write(c_idx[i], 3u);
                    }
                  }
                  best_bits = m4_bits;
                };
              }
            };
          };
        }

        // ---- 2-Subset Modes 1, 3, 7 with fast 16/64-partition screening ----
        ArrayUInt<64> part2_masks, part2_anchors;
        for (std::uint32_t p = 0u; p < 64u; ++p) {
          part2_masks[p] = kPart2Masks[p];
          part2_anchors[p] = kPart2Anchor[p];
        }
        const UInt gate_2sub =
            ite(quality >= 2u, 64u, ite(quality == 1u, 256u, 1024u));
        const UInt part2_limit = ite(quality >= 1u, 64u, 16u);
        const UInt topk_2sub =
            ite(quality >= 2u, 4u, ite(quality == 1u, 2u, 1u));

        $if(best_sse > gate_2sub) {
          ArrayUInt<4> top_p;
          screen_top_part2_dsl(
              pr, pg, pb, pa, !all_opaque, part2_masks, part2_limit, top_p);

          $if(all_opaque) {
            // Mode 1 (6.6.6+P, 3b idx) and Mode 3 (7.7.7+4P, 2b idx) on top-K.
            $for(ti, topk_2sub) {
              const UInt part = top_p[ti];
              const UInt pmask = part2_masks[part];
              const UInt anchor2 = part2_anchors[part];

              ArrayFloat<3> s0e0, s0e1, s1e0, s1e1;
              Float3 f0a = make_float3(0.0f), f0b = make_float3(0.0f);
              Float3 f1a = make_float3(0.0f), f1b = make_float3(0.0f);
              pca_ls_rgb_subset(fr, fg, fb, pmask, false, 0u, w3_arr, 7, 2, 1, f0a, f0b);
              pca_ls_rgb_subset(fr, fg, fb, pmask, false, 1u, w3_arr, 7, 2, 1, f1a, f1b);
              s0e0[0] = f0a.x; s0e0[1] = f0a.y; s0e0[2] = f0a.z;
              s0e1[0] = f0b.x; s0e1[1] = f0b.y; s0e1[2] = f0b.z;
              s1e0[0] = f1a.x; s1e0[1] = f1a.y; s1e0[2] = f1a.z;
              s1e1[0] = f1b.x; s1e1[1] = f1b.y; s1e1[2] = f1b.z;

              // Evaluate Mode 1: 6.6.6 + 1 shared P-bit per subset, 3-bit indices.
              {
                ArrayUInt<12> cur_ep;
                ArrayUInt<2> cur_p;
                ArrayUInt<16> cur_idx;
                UInt m1_sse = 0u;
                $for(k, 12u) { cur_ep[k] = 0u; };
                cur_p[0] = 0u; cur_p[1] = 0u;
                $for(i, 16u) { cur_idx[i] = 0u; };

                $for(sub, 2u) {
                  const UInt sub_u = cast<uint>(sub);
                  const UInt anc = ite(sub_u == 0u, 0u, anchor2);
                  UInt sub_best = 0xffffffffu;
                  ArrayUInt<3> sub_c0, sub_c1;
                  UInt sub_p = 0u;
                  ArrayUInt<16> sub_idx;
                  $for(ch, 3u) { sub_c0[ch] = 0u; sub_c1[ch] = 0u; };
                  $for(i, 16u) { sub_idx[i] = 0u; };

                  $for(p, 2u) {
                    ArrayInt<3> tc0, tc1, qe0, qe1;
                    const Float pf = cast<float>(p);
                    $for(ch, 3u) {
                      const Float sv0 = ite(sub_u == 0u, s0e0[ch], s1e0[ch]);
                      const Float sv1 = ite(sub_u == 0u, s0e1[ch], s1e1[ch]);
                      tc0[ch] = clamp(cast<int>(round((sv0 * (127.0f / 255.0f) - pf) * 0.5f)), 0, 63);
                      tc1[ch] = clamp(cast<int>(round((sv1 * (127.0f / 255.0f) - pf) * 0.5f)), 0, 63);
                      const Int v0 = (tc0[ch] << 1) | cast<int>(p);
                      const Int v1 = (tc1[ch] << 1) | cast<int>(p);
                      qe0[ch] = (v0 << 1) | (v0 >> 6);
                      qe1[ch] = (v1 << 1) | (v1 >> 6);
                    };
                    ArrayInt<8> pal_r, pal_g, pal_b;
                    $for(k, 8u) {
                      const Int w = w3_arr[k];
                      pal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                      pal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                      pal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
                    };
                    UInt ss = 0u;
                    ArrayUInt<16> tidx;
                    $for(i, 16u) {
                      tidx[i] = 0u;
                      $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                        const Int r = pr[i], g = pg[i], b = pb[i];
                        UInt bk = 0u; Int be = 0x7fffffff;
                        $for(k, 8u) {
                          const Int dr = pal_r[k] - r, dg = pal_g[k] - g, db = pal_b[k] - b;
                          const Int e = dr * dr + dg * dg + db * db;
                          $if(e < be) { be = e; bk = cast<uint>(k); };
                        };
                        tidx[i] = bk; ss += cast<uint>(be);
                      };
                    };
                    $if(ss < sub_best) {
                      sub_best = ss; sub_p = cast<uint>(p);
                      $for(ch, 3u) {
                        sub_c0[ch] = cast<uint>(tc0[ch]);
                        sub_c1[ch] = cast<uint>(tc1[ch]);
                      };
                      $for(i, 16u) { sub_idx[i] = tidx[i]; };
                    };
                  };
                  $if(sub_idx[anc] >= 4u) {
                    $for(ch, 3u) {
                      const UInt t = sub_c0[ch]; sub_c0[ch] = sub_c1[ch]; sub_c1[ch] = t;
                    };
                    $for(i, 16u) {
                      $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                        sub_idx[i] = 7u - sub_idx[i];
                      };
                    };
                  };
                  cur_p[sub_u] = sub_p;
                  $for(ch, 3u) {
                    cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 0u] = sub_c0[ch];
                    cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 1u] = sub_c1[ch];
                  };
                  $for(i, 16u) {
                    $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) { cur_idx[i] = sub_idx[i]; };
                  };
                  m1_sse += sub_best;
                };
                $if(m1_sse < best_sse) {
                  best_sse = m1_sse;
                  Bc7DslBit128 cand;
                  cand.write(0x02u, 2u);
                  cand.write(part, 6u);
                  for (std::uint32_t k = 0u; k < 12u; ++k) cand.write(cur_ep[k], 6u);
                  cand.write(cur_p[0], 1u); cand.write(cur_p[1], 1u);
                  for (std::uint32_t i = 0u; i < 16u; ++i) {
                    const UInt b = ite((i == 0u) | (anchor2 == i), 2u, 3u);
                    cand.write(cur_idx[i], b);
                  }
                  best_bits = cand;
                };
              }

              // Mode 3 (balanced & high): 2-subset 7.7.7 + 4 P-bits, 2b indices.
              $if(quality >= 1u) {
                ArrayUInt<12> cur_ep;
                ArrayUInt<4> cur_p;
                ArrayUInt<16> cur_idx;
                UInt m3_sse = 0u;
                $for(k, 12u) { cur_ep[k] = 0u; };
                $for(k, 4u) { cur_p[k] = 0u; };
                $for(i, 16u) { cur_idx[i] = 0u; };

                $for(sub, 2u) {
                  const UInt sub_u = cast<uint>(sub);
                  const UInt anc = ite(sub_u == 0u, 0u, anchor2);
                  UInt sub_best = 0xffffffffu;
                  ArrayUInt<3> sub_c0, sub_c1;
                  UInt sub_p0 = 0u, sub_p1 = 0u;
                  ArrayUInt<16> sub_idx;
                  $for(ch, 3u) { sub_c0[ch] = 0u; sub_c1[ch] = 0u; };
                  $for(i, 16u) { sub_idx[i] = 0u; };

                  $for(p0, 2u) {
                    ArrayInt<3> tc0, qe0;
                    const Float pf0 = cast<float>(p0);
                    $for(ch, 3u) {
                      const Float sv0 = ite(sub_u == 0u, s0e0[ch], s1e0[ch]);
                      tc0[ch] = clamp(cast<int>(round((sv0 - pf0) * 0.5f)), 0, 127);
                      qe0[ch] = (tc0[ch] << 1) | cast<int>(p0);
                    };
                    $for(p1, 2u) {
                      ArrayInt<3> tc1, qe1;
                      const Float pf1 = cast<float>(p1);
                      $for(ch, 3u) {
                        const Float sv1 = ite(sub_u == 0u, s0e1[ch], s1e1[ch]);
                        tc1[ch] = clamp(cast<int>(round((sv1 - pf1) * 0.5f)), 0, 127);
                        qe1[ch] = (tc1[ch] << 1) | cast<int>(p1);
                      };
                      ArrayInt<4> pal_r, pal_g, pal_b;
                      $for(k, 4u) {
                        const Int w = w2_arr[k];
                        pal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                        pal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                        pal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
                      };
                      UInt ss = 0u;
                      ArrayUInt<16> tidx;
                      $for(i, 16u) {
                        tidx[i] = 0u;
                        $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                          const Int r = pr[i], g = pg[i], b = pb[i];
                          UInt bk = 0u; Int be = 0x7fffffff;
                          $for(k, 4u) {
                            const Int dr = pal_r[k] - r, dg = pal_g[k] - g, db = pal_b[k] - b;
                            const Int e = dr * dr + dg * dg + db * db;
                            $if(e < be) { be = e; bk = cast<uint>(k); };
                          };
                          tidx[i] = bk; ss += cast<uint>(be);
                        };
                      };
                      $if(ss < sub_best) {
                        sub_best = ss;
                        sub_p0 = cast<uint>(p0); sub_p1 = cast<uint>(p1);
                        $for(ch, 3u) {
                          sub_c0[ch] = cast<uint>(tc0[ch]);
                          sub_c1[ch] = cast<uint>(tc1[ch]);
                        };
                        $for(i, 16u) { sub_idx[i] = tidx[i]; };
                      };
                    };
                  };
                  $if(sub_idx[anc] >= 2u) {
                    $for(ch, 3u) {
                      const UInt t = sub_c0[ch]; sub_c0[ch] = sub_c1[ch]; sub_c1[ch] = t;
                    };
                    const UInt tp = sub_p0; sub_p0 = sub_p1; sub_p1 = tp;
                    $for(i, 16u) {
                      $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                        sub_idx[i] = 3u - sub_idx[i];
                      };
                    };
                  };
                  cur_p[sub_u * 2u + 0u] = sub_p0;
                  cur_p[sub_u * 2u + 1u] = sub_p1;
                  $for(ch, 3u) {
                    cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 0u] = sub_c0[ch];
                    cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 1u] = sub_c1[ch];
                  };
                  $for(i, 16u) {
                    $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) { cur_idx[i] = sub_idx[i]; };
                  };
                  m3_sse += sub_best;
                };
                $if(m3_sse < best_sse) {
                  best_sse = m3_sse;
                  Bc7DslBit128 cand;
                  cand.write(0x08u, 4u);
                  cand.write(part, 6u);
                  for (std::uint32_t k = 0u; k < 12u; ++k) cand.write(cur_ep[k], 7u);
                  for (std::uint32_t e = 0u; e < 4u; ++e) cand.write(cur_p[e], 1u);
                  for (std::uint32_t i = 0u; i < 16u; ++i) {
                    const UInt b = ite((i == 0u) | (anchor2 == i), 1u, 2u);
                    cand.write(cur_idx[i], b);
                  }
                  best_bits = cand;
                };
              };
            };
          } $else {
            // Mode 7: 2-subset RGBA 5.5.5.5 + 4 P-bits, 2b indices.
            $for(ti, topk_2sub) {
              const UInt part = top_p[ti];
              const UInt pmask = part2_masks[part];
              const UInt anchor2 = part2_anchors[part];
              ArrayUInt<16> cur_ep;
              ArrayUInt<4> cur_p;
              ArrayUInt<16> cur_idx;
              UInt part_sse = 0u;
              $for(k, 16u) { cur_ep[k] = 0u; };
              $for(k, 4u) { cur_p[k] = 0u; };
              $for(i, 16u) { cur_idx[i] = 0u; };

              $for(sub, 2u) {
                const UInt sub_u = cast<uint>(sub);
                const UInt anc = ite(sub_u == 0u, 0u, anchor2);
                Float4 se0 = make_float4(0.0f), se1 = make_float4(0.0f);
                pca_ls_rgba_subset(fr, fg, fb, fa, pmask, sub_u, w2_arr, 2, 1, se0, se1);

                UInt sub_best = 0xffffffffu;
                ArrayUInt<4> sub_c0, sub_c1;
                UInt sub_p0 = 0u, sub_p1 = 0u;
                ArrayUInt<16> sub_idx;
                $for(ch, 4u) { sub_c0[ch] = 0u; sub_c1[ch] = 0u; };
                $for(i, 16u) { sub_idx[i] = 0u; };

                ArrayFloat<4> se0_arr, se1_arr;
                se0_arr[0] = se0.x; se0_arr[1] = se0.y; se0_arr[2] = se0.z; se0_arr[3] = se0.w;
                se1_arr[0] = se1.x; se1_arr[1] = se1.y; se1_arr[2] = se1.z; se1_arr[3] = se1.w;

                $for(p0, 2u) {
                  ArrayInt<4> tc0, qe0;
                  const Float pf0 = cast<float>(p0);
                  $for(ch, 4u) {
                    tc0[ch] = clamp(cast<int>(round((se0_arr[ch] * (63.0f / 255.0f) - pf0) * 0.5f)), 0, 31);
                    const Int v0 = (tc0[ch] << 1) | cast<int>(p0);
                    qe0[ch] = (v0 << 2) | (v0 >> 4);
                  };
                  $for(p1, 2u) {
                    ArrayInt<4> tc1, qe1;
                    const Float pf1 = cast<float>(p1);
                    $for(ch, 4u) {
                      tc1[ch] = clamp(cast<int>(round((se1_arr[ch] * (63.0f / 255.0f) - pf1) * 0.5f)), 0, 31);
                      const Int v1 = (tc1[ch] << 1) | cast<int>(p1);
                      qe1[ch] = (v1 << 2) | (v1 >> 4);
                    };
                    ArrayInt<4> pal_r, pal_g, pal_b, pal_a;
                    $for(k, 4u) {
                      const Int w = w2_arr[k];
                      pal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                      pal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                      pal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
                      pal_a[k] = bc7_interp_dsl(qe0[3], qe1[3], w);
                    };
                    UInt ss = 0u;
                    ArrayUInt<16> tidx;
                    $for(i, 16u) {
                      tidx[i] = 0u;
                      $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                        const Int r = pr[i], g = pg[i], b = pb[i], a = pa[i];
                        UInt bk = 0u; Int be = 0x7fffffff;
                        $for(k, 4u) {
                          const Int dr = pal_r[k] - r, dg = pal_g[k] - g,
                                    db = pal_b[k] - b, da = pal_a[k] - a;
                          const Int e = dr * dr + dg * dg + db * db + da * da;
                          $if(e < be) { be = e; bk = cast<uint>(k); };
                        };
                        tidx[i] = bk; ss += cast<uint>(be);
                      };
                    };
                    $if(ss < sub_best) {
                      sub_best = ss;
                      sub_p0 = cast<uint>(p0); sub_p1 = cast<uint>(p1);
                      $for(ch, 4u) {
                        sub_c0[ch] = cast<uint>(tc0[ch]);
                        sub_c1[ch] = cast<uint>(tc1[ch]);
                      };
                      $for(i, 16u) { sub_idx[i] = tidx[i]; };
                    };
                  };
                };
                $if(sub_idx[anc] >= 2u) {
                  $for(ch, 4u) {
                    const UInt t = sub_c0[ch]; sub_c0[ch] = sub_c1[ch]; sub_c1[ch] = t;
                  };
                  const UInt tp = sub_p0; sub_p0 = sub_p1; sub_p1 = tp;
                  $for(i, 16u) {
                    $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) {
                      sub_idx[i] = 3u - sub_idx[i];
                    };
                  };
                };
                cur_p[sub_u * 2u + 0u] = sub_p0;
                cur_p[sub_u * 2u + 1u] = sub_p1;
                $for(ch, 4u) {
                  cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 0u] = sub_c0[ch];
                  cur_ep[cast<uint>(ch) * 4u + sub_u * 2u + 1u] = sub_c1[ch];
                };
                $for(i, 16u) {
                  $if(((pmask >> cast<uint>(i)) & 1u) == sub_u) { cur_idx[i] = sub_idx[i]; };
                };
                part_sse += sub_best;
              };
              $if(part_sse < best_sse) {
                best_sse = part_sse;
                Bc7DslBit128 cand;
                cand.write(0x80u, 8u);
                cand.write(part, 6u);
                for (std::uint32_t k = 0u; k < 16u; ++k) cand.write(cur_ep[k], 5u);
                for (std::uint32_t e = 0u; e < 4u; ++e) cand.write(cur_p[e], 1u);
                for (std::uint32_t i = 0u; i < 16u; ++i) {
                  const UInt b = ite((i == 0u) | (anchor2 == i), 1u, 2u);
                  cand.write(cur_idx[i], b);
                }
                best_bits = cand;
              };
            };
          };
        };

        // ---- 3-Subset Modes 0 & 2 (high quality only, opaque blocks) ----
        $if((quality >= 2u) & all_opaque & (best_sse > 256u)) {
          ArrayUInt<64> p3_masks, p3_a1, p3_a2;
          for (std::uint32_t p = 0u; p < 64u; ++p) {
            p3_masks[p] = kPart3Masks[p];
            p3_a1[p] = kPart3Anchor1[p];
            p3_a2[p] = kPart3Anchor2[p];
          }
          // Screen 64 3-subset partitions for top-2 overall & track best 0..15 for Mode 0.
          ArrayUInt<2> top_m2, top_m0;
          ArrayUInt<2> top_m2_s, top_m0_s;
          for (int k = 0; k < 2; ++k) {
            top_m2[k] = 0u; top_m2_s[k] = 0xffffffffu;
            top_m0[k] = 0u; top_m0_s[k] = 0xffffffffu;
          }
          $for(p, 64u) {
            const UInt pm3 = p3_masks[p];
            ArrayInt<3> mnr, mng, mnb, mxr, mxg, mxb, cnt;
            for (int s = 0; s < 3; ++s) {
              mnr[s] = 255; mng[s] = 255; mnb[s] = 255;
              mxr[s] = 0; mxg[s] = 0; mxb[s] = 0; cnt[s] = 0;
            }
            $for(i, 16u) {
              const UInt sub = (pm3 >> (cast<uint>(i) * 2u)) & 3u;
              $if(sub < 3u) {
                cnt[sub] += 1;
                mnr[sub] = min(mnr[sub], pr[i]); mxr[sub] = max(mxr[sub], pr[i]);
                mng[sub] = min(mng[sub], pg[i]); mxg[sub] = max(mxg[sub], pg[i]);
                mnb[sub] = min(mnb[sub], pb[i]); mxb[sub] = max(mxb[sub], pb[i]);
              };
            };
            UInt sc = 0u;
            for (int s = 0; s < 3; ++s) {
              const Int dr = max(mxr[s] - mnr[s], 0);
              const Int dg = max(mxg[s] - mng[s], 0);
              const Int db = max(mxb[s] - mnb[s], 0);
              sc += cast<uint>(cnt[s] * (dr * dr + dg * dg + db * db));
            }
            {
              UInt cp = cast<uint>(p), cs = sc;
              for (int r = 0; r < 2; ++r) {
                $if(cs < top_m2_s[r]) {
                  const UInt ts = top_m2_s[r], tp = top_m2[r];
                  top_m2_s[r] = cs; top_m2[r] = cp; cs = ts; cp = tp;
                };
              }
            }
            $if(p < 16u) {
              UInt cp = cast<uint>(p), cs = sc;
              for (int r = 0; r < 2; ++r) {
                $if(cs < top_m0_s[r]) {
                  const UInt ts = top_m0_s[r], tp = top_m0[r];
                  top_m0_s[r] = cs; top_m0[r] = cp; cs = ts; cp = tp;
                };
              }
            };
          };

          // Mode 2: 3-subset 5.5.5, 2b idx (top-2 of 64 partitions).
          $for(ti, 2u) {
            const UInt part = top_m2[ti];
            const UInt pm3 = p3_masks[part];
            const UInt a1 = p3_a1[part], a2 = p3_a2[part];
            ArrayUInt<18> ep;
            ArrayUInt<16> idx;
            UInt m2_sse = 0u;
            $for(k, 18u) { ep[k] = 0u; };
            $for(i, 16u) { idx[i] = 0u; };

            $for(sub, 3u) {
              const UInt sub_u = cast<uint>(sub);
              const UInt anc = ite(sub_u == 0u, 0u, ite(sub_u == 1u, a1, a2));
              Float3 se0 = make_float3(0.0f), se1 = make_float3(0.0f);
              pca_ls_rgb_subset(fr, fg, fb, pm3, true, sub_u, w3_arr, 3, 2, 0, se0, se1);
              // Refine for 2b weights accurately:
              {
                const Float3 sd = se1 - se0;
                const Float sl2 = dot(sd, sd);
                $if(sl2 > 1e-6f) {
                  const Float il2 = 1.0f / sl2;
                  Float x2 = 0.0f, xy = 0.0f, y2 = 0.0f;
                  Float3 r0 = make_float3(0.0f), r1 = make_float3(0.0f);
                  $for(i, 16u) {
                    $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) {
                      const Float3 px = make_float3(fr[i], fg[i], fb[i]);
                      const Int k = clamp(cast<int>(round(clamp(dot(px - se0, sd) * il2, 0.0f, 1.0f) * 3.0f)), 0, 3);
                      const Float b = cast<float>(w2_arr[k]) * (1.0f / 64.0f);
                      const Float a = 1.0f - b;
                      x2 += a * a; xy += a * b; y2 += b * b;
                      r0 += a * px; r1 += b * px;
                    };
                  };
                  const Float det = x2 * y2 - xy * xy;
                  $if(det > 1e-6f) {
                    const Float id = 1.0f / det;
                    se0 = clamp((y2 * r0 - xy * r1) * id, 0.0f, 255.0f);
                    se1 = clamp((-xy * r0 + x2 * r1) * id, 0.0f, 255.0f);
                  };
                };
              }
              ArrayUInt<3> c0, c1;
              ArrayInt<3> q0, q1;
              c0[0] = cast<uint>(clamp(cast<int>(round(se0.x * (31.0f / 255.0f))), 0, 31));
              c0[1] = cast<uint>(clamp(cast<int>(round(se0.y * (31.0f / 255.0f))), 0, 31));
              c0[2] = cast<uint>(clamp(cast<int>(round(se0.z * (31.0f / 255.0f))), 0, 31));
              c1[0] = cast<uint>(clamp(cast<int>(round(se1.x * (31.0f / 255.0f))), 0, 31));
              c1[1] = cast<uint>(clamp(cast<int>(round(se1.y * (31.0f / 255.0f))), 0, 31));
              c1[2] = cast<uint>(clamp(cast<int>(round(se1.z * (31.0f / 255.0f))), 0, 31));
              $for(ch, 3u) {
                const Int v0 = cast<int>(c0[ch]), v1 = cast<int>(c1[ch]);
                q0[ch] = (v0 << 3) | (v0 >> 2);
                q1[ch] = (v1 << 3) | (v1 >> 2);
              };
              ArrayInt<4> pal_r, pal_g, pal_b;
              $for(k, 4u) {
                const Int w = w2_arr[k];
                pal_r[k] = bc7_interp_dsl(q0[0], q1[0], w);
                pal_g[k] = bc7_interp_dsl(q0[1], q1[1], w);
                pal_b[k] = bc7_interp_dsl(q0[2], q1[2], w);
              };
              ArrayUInt<16> sidx;
              UInt ss = 0u;
              $for(i, 16u) {
                sidx[i] = 0u;
                $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) {
                  const Int r = pr[i], g = pg[i], b = pb[i];
                  UInt bk = 0u; Int be = 0x7fffffff;
                  $for(k, 4u) {
                    const Int dr = pal_r[k] - r, dg = pal_g[k] - g, db = pal_b[k] - b;
                    const Int e = dr * dr + dg * dg + db * db;
                    $if(e < be) { be = e; bk = cast<uint>(k); };
                  };
                  sidx[i] = bk; ss += cast<uint>(be);
                };
              };
              $if(sidx[anc] >= 2u) {
                $for(ch, 3u) { const UInt t = c0[ch]; c0[ch] = c1[ch]; c1[ch] = t; };
                $for(i, 16u) {
                  $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) {
                    sidx[i] = 3u - sidx[i];
                  };
                };
              };
              $for(ch, 3u) {
                ep[cast<uint>(ch) * 6u + sub_u * 2u + 0u] = c0[ch];
                ep[cast<uint>(ch) * 6u + sub_u * 2u + 1u] = c1[ch];
              };
              $for(i, 16u) {
                $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) { idx[i] = sidx[i]; };
              };
              m2_sse += ss;
            };
            $if(m2_sse < best_sse) {
              best_sse = m2_sse;
              Bc7DslBit128 cand;
              cand.write(0x04u, 3u);
              cand.write(part, 6u);
              for (std::uint32_t k = 0u; k < 18u; ++k) cand.write(ep[k], 5u);
              for (std::uint32_t i = 0u; i < 16u; ++i) {
                const UInt b = ite((i == 0u) | (a1 == i) | (a2 == i), 1u, 2u);
                cand.write(idx[i], b);
              }
              best_bits = cand;
            };
          };

          // Mode 0: 3-subset 4.4.4 + 6 P-bits, 3b idx (top-2 of 0..15).
          $for(ti, 2u) {
            const UInt part = top_m0[ti];
            const UInt pm3 = p3_masks[part];
            const UInt a1 = p3_a1[part], a2 = p3_a2[part];
            ArrayUInt<18> ep;
            ArrayUInt<6> pb_bits;
            ArrayUInt<16> idx;
            UInt m0_sse = 0u;
            $for(k, 18u) { ep[k] = 0u; };
            $for(k, 6u) { pb_bits[k] = 0u; };
            $for(i, 16u) { idx[i] = 0u; };

            $for(sub, 3u) {
              const UInt sub_u = cast<uint>(sub);
              const UInt anc = ite(sub_u == 0u, 0u, ite(sub_u == 1u, a1, a2));
              Float3 se0 = make_float3(0.0f), se1 = make_float3(0.0f);
              pca_ls_rgb_subset(fr, fg, fb, pm3, true, sub_u, w3_arr, 7, 2, 1, se0, se1);

              UInt sub_best = 0xffffffffu;
              ArrayUInt<3> sub_c0, sub_c1;
              UInt sp0 = 0u, sp1 = 0u;
              ArrayUInt<16> sub_idx;
              $for(ch, 3u) { sub_c0[ch] = 0u; sub_c1[ch] = 0u; };
              $for(i, 16u) { sub_idx[i] = 0u; };
              ArrayFloat<3> s0a, s1a;
              s0a[0] = se0.x; s0a[1] = se0.y; s0a[2] = se0.z;
              s1a[0] = se1.x; s1a[1] = se1.y; s1a[2] = se1.z;

              $for(p0, 2u) {
                ArrayInt<3> tc0, qe0;
                const Float pf0 = cast<float>(p0);
                $for(ch, 3u) {
                  tc0[ch] = clamp(cast<int>(round((s0a[ch] * (31.0f / 255.0f) - pf0) * 0.5f)), 0, 15);
                  const Int v0 = (tc0[ch] << 1) | cast<int>(p0);
                  qe0[ch] = (v0 << 3) | (v0 >> 2);
                };
                $for(p1, 2u) {
                  ArrayInt<3> tc1, qe1;
                  const Float pf1 = cast<float>(p1);
                  $for(ch, 3u) {
                    tc1[ch] = clamp(cast<int>(round((s1a[ch] * (31.0f / 255.0f) - pf1) * 0.5f)), 0, 15);
                    const Int v1 = (tc1[ch] << 1) | cast<int>(p1);
                    qe1[ch] = (v1 << 3) | (v1 >> 2);
                  };
                  ArrayInt<8> pal_r, pal_g, pal_b;
                  $for(k, 8u) {
                    const Int w = w3_arr[k];
                    pal_r[k] = bc7_interp_dsl(qe0[0], qe1[0], w);
                    pal_g[k] = bc7_interp_dsl(qe0[1], qe1[1], w);
                    pal_b[k] = bc7_interp_dsl(qe0[2], qe1[2], w);
                  };
                  UInt ss = 0u;
                  ArrayUInt<16> tidx;
                  $for(i, 16u) {
                    tidx[i] = 0u;
                    $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) {
                      const Int r = pr[i], g = pg[i], b = pb[i];
                      UInt bk = 0u; Int be = 0x7fffffff;
                      $for(k, 8u) {
                        const Int dr = pal_r[k] - r, dg = pal_g[k] - g, db = pal_b[k] - b;
                        const Int e = dr * dr + dg * dg + db * db;
                        $if(e < be) { be = e; bk = cast<uint>(k); };
                      };
                      tidx[i] = bk; ss += cast<uint>(be);
                    };
                  };
                  $if(ss < sub_best) {
                    sub_best = ss; sp0 = cast<uint>(p0); sp1 = cast<uint>(p1);
                    $for(ch, 3u) { sub_c0[ch] = cast<uint>(tc0[ch]); sub_c1[ch] = cast<uint>(tc1[ch]); };
                    $for(i, 16u) { sub_idx[i] = tidx[i]; };
                  };
                };
              };
              $if(sub_idx[anc] >= 4u) {
                $for(ch, 3u) { const UInt t = sub_c0[ch]; sub_c0[ch] = sub_c1[ch]; sub_c1[ch] = t; };
                const UInt tp = sp0; sp0 = sp1; sp1 = tp;
                $for(i, 16u) {
                  $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) {
                    sub_idx[i] = 7u - sub_idx[i];
                  };
                };
              };
              pb_bits[sub_u * 2u + 0u] = sp0;
              pb_bits[sub_u * 2u + 1u] = sp1;
              $for(ch, 3u) {
                ep[cast<uint>(ch) * 6u + sub_u * 2u + 0u] = sub_c0[ch];
                ep[cast<uint>(ch) * 6u + sub_u * 2u + 1u] = sub_c1[ch];
              };
              $for(i, 16u) {
                $if(((pm3 >> (cast<uint>(i) * 2u)) & 3u) == sub_u) { idx[i] = sub_idx[i]; };
              };
              m0_sse += sub_best;
            };
            $if(m0_sse < best_sse) {
              best_sse = m0_sse;
              Bc7DslBit128 cand;
              cand.write(0x01u, 1u);
              cand.write(part, 4u);
              for (std::uint32_t k = 0u; k < 18u; ++k) cand.write(ep[k], 4u);
              for (std::uint32_t e = 0u; e < 6u; ++e) cand.write(pb_bits[e], 1u);
              for (std::uint32_t i = 0u; i < 16u; ++i) {
                const UInt b = ite((i == 0u) | (a1 == i) | (a2 == i), 2u, 3u);
                cand.write(idx[i], b);
              }
              best_bits = cand;
            };
          };
        };

        out_blocks.write(gid * 4u + 0u, best_bits.w0);
        out_blocks.write(gid * 4u + 1u, best_bits.w1);
        out_blocks.write(gid * 4u + 2u, best_bits.w2);
        out_blocks.write(gid * 4u + 3u, best_bits.w3);
      });
}

} // namespace psycles::luisa_backend::detail
