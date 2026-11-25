#include "virtual_texture_cache_gpu_bc.h"

#include <array>
#include <cstdint>

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa;
using namespace luisa::compute;

inline constexpr std::array<int, 8u> kBc6hWeight3{
    0, 9, 18, 27, 37, 46, 55, 64};
inline constexpr std::array<int, 16u> kBc6hWeight4{
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// 32 BC6H 2-subset partitions and second anchor indices (identical to
// DirectXTex bc6_header). Pixel 0 is always subset 0; anchor2 is subset 1.
inline constexpr std::array<std::uint32_t, 32u> kBc6hPart2Masks{
    0xccccu, 0x8888u, 0xeeeeu, 0xecc8u, 0xc880u, 0xfeecu, 0xfec8u, 0xec80u,
    0xc800u, 0xffecu, 0xfe80u, 0xe800u, 0xffe8u, 0xff00u, 0xfff0u, 0xf000u,
    0xf710u, 0x008eu, 0x7100u, 0x08ceu, 0x008cu, 0x7310u, 0x3100u, 0x8cceu,
    0x088cu, 0x3110u, 0x6666u, 0x366cu, 0x17e8u, 0x0ff0u, 0x718eu, 0x399cu};

inline constexpr std::array<std::uint32_t, 32u> kBc6hPart2Anchor{
    15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
    15u, 15u, 15u, 15u, 15u, 15u, 2u,  8u,  2u,  2u,  8u,
    8u,  15u, 2u,  8u,  2u,  2u,  8u,  8u,  2u,  2u};

struct Bc6hDslBit128 {
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

  void write_at(UInt val, UInt bits, UInt bit_offset) noexcept {
    pos = bit_offset;
    write(val, bits);
  }
};

[[nodiscard]] UInt float_to_uf16_dsl(Float v) noexcept {
  UInt out = 0u;
  $if(v > 0.0f) {
    $if(v >= 65504.0f) {
      out = 0x7bffu;
    } $else {
      const UInt f = as<UInt>(v);
      const UInt exp = (f >> 23u) & 0xffu;
      const UInt frac = f & 0x7fffffu;
      const Int new_exp = cast<int>(exp) - 127 + 15;
      $if(new_exp >= 31) {
        out = 0x7bffu;
      } $elif(new_exp <= 0) {
        $if(new_exp >= -10) {
          const UInt mant = frac | 0x800000u;
          const UInt shift = cast<uint>(14 - new_exp);
          UInt sub = mant >> shift;
          const UInt rem = mant & ((1u << shift) - 1u);
          const UInt half_way = 1u << (shift - 1u);
          $if((rem > half_way) | ((rem == half_way) & ((sub & 1u) != 0u))) {
            sub += 1u;
          };
          out = sub;
        };
      } $else {
        const UInt rounded_frac = frac + 0x1000u;
        $if((rounded_frac & 0x800000u) != 0u) {
          const Int e_inc = new_exp + 1;
          out = ite(e_inc >= 31, 0x7bffu, cast<uint>(e_inc) << 10u);
        } $else {
          out = (cast<uint>(new_exp) << 10u) | (rounded_frac >> 13u);
        };
      };
    };
  };
  return out;
}

[[nodiscard]] Int bc6h_quantize_uf16_dsl(Int ph_val, int prec) noexcept {
  if (prec >= 15) {
    return clamp(ph_val, 0, 65535);
  }
  const int max_q = (1 << prec) - 1;
  Int q = ((ph_val << prec) + (1 << 15)) >> 16;
  q = ite(ph_val <= 0, 0, q);
  q = ite(ph_val >= 65535, max_q, q);
  return clamp(q, 0, max_q);
}

[[nodiscard]] Int bc6h_unquantize_uf16_dsl(Int q, int prec) noexcept {
  if (prec >= 15) {
    return q;
  }
  const int max_q = (1 << prec) - 1;
  Int u = ((q << 16) + 32768) >> prec;
  u = ite(q <= 0, 0, u);
  return ite(q >= max_q, 65535, u);
}

void bc6h_pca_ls_endpoints_dsl(
    const ArrayFloat<16> &fr,
    const ArrayFloat<16> &fg,
    const ArrayFloat<16> &fb,
    UInt pmask,
    UInt target_sub,
    bool use_mask,
    const ArrayInt<16> &w_arr,
    int max_k,
    int pca_iters,
    int ls_iters,
    Float3 &out_e0,
    Float3 &out_e1) noexcept {
  Float3 mean = make_float3(0.0f);
  Float3 cmin = make_float3(31743.0f);
  Float3 cmax = make_float3(0.0f);
  Int count = 0;
  $for(i, 16u) {
    const Bool match =
        use_mask ? (((pmask >> cast<uint>(i)) & 1u) == target_sub) : Bool{true};
    $if(match) {
      count += 1;
      const Float3 px = make_float3(fr[i], fg[i], fb[i]);
      mean += px;
      cmin = min(cmin, px);
      cmax = max(cmax, px);
    };
  };
  $if(count <= 0) {
    out_e0 = make_float3(0.0f);
    out_e1 = make_float3(0.0f);
    return;
  };
  mean /= cast<float>(count);
  out_e0 = mean;
  out_e1 = mean;

  Float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f;
  Float c11 = 0.0f, c12 = 0.0f, c22 = 0.0f;
  $for(i, 16u) {
    const Bool match =
        use_mask ? (((pmask >> cast<uint>(i)) & 1u) == target_sub) : Bool{true};
    $if(match) {
      const Float d0 = fr[i] - mean.x;
      const Float d1 = fg[i] - mean.y;
      const Float d2 = fb[i] - mean.z;
      c00 += d0 * d0;
      c01 += d0 * d1;
      c02 += d0 * d2;
      c11 += d1 * d1;
      c12 += d1 * d2;
      c22 += d2 * d2;
    };
  };
  Float3 axis = cmax - cmin;
  const Float norm2 = dot(axis, axis);
  $if(norm2 > 1e-6f) {
    axis /= sqrt(norm2);
    for (int it = 0; it < pca_iters; ++it) {
      const Float3 nxt = make_float3(
          c00 * axis.x + c01 * axis.y + c02 * axis.z,
          c01 * axis.x + c11 * axis.y + c12 * axis.z,
          c02 * axis.x + c12 * axis.y + c22 * axis.z);
      const Float n2 = dot(nxt, nxt);
      $if(n2 > 1e-8f) { axis = nxt / sqrt(n2); };
    }
    Float tmin = 1e30f;
    Float tmax = -1e30f;
    $for(i, 16u) {
      const Bool match =
          use_mask ? (((pmask >> cast<uint>(i)) & 1u) == target_sub) : Bool{true};
      $if(match) {
        const Float p = dot(make_float3(fr[i], fg[i], fb[i]) - mean, axis);
        tmin = min(tmin, p);
        tmax = max(tmax, p);
      };
    };
    out_e0 = clamp(mean + tmin * axis, 0.0f, 31743.0f);
    out_e1 = clamp(mean + tmax * axis, 0.0f, 31743.0f);
  };

  const float max_kf = static_cast<float>(max_k);
  for (int it = 0; it < ls_iters; ++it) {
    const Float3 diff = out_e1 - out_e0;
    const Float len2 = dot(diff, diff);
    $if(len2 > 1e-6f) {
      const Float inv_l2 = 1.0f / len2;
      Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
      Float3 rhs0 = make_float3(0.0f);
      Float3 rhs1 = make_float3(0.0f);
      $for(i, 16u) {
        const Bool match =
            use_mask ? (((pmask >> cast<uint>(i)) & 1u) == target_sub) : Bool{true};
        $if(match) {
          const Float3 px = make_float3(fr[i], fg[i], fb[i]);
          const Float t = clamp(dot(px - out_e0, diff) * inv_l2, 0.0f, 1.0f);
          const Int k = clamp(cast<int>(round(t * max_kf)), 0, max_k);
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
        out_e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 31743.0f);
        out_e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 31743.0f);
      };
    };
  }
}

void bc6h_pack_1sub_header(
    Bc6hDslBit128 &bw,
    UInt mode_bits,
    int prec_base,
    int prec_delta,
    bool transformed,
    const ArrayUInt<3> &e0,
    const ArrayUInt<3> &e1) noexcept {
  bw.w0 = 0u;
  bw.w1 = 0u;
  bw.w2 = 0u;
  bw.w3 = 0u;
  bw.pos = 0u;
  bw.write(mode_bits, 5u);
  if (!transformed) {
    // Mode 11: direct 10.10.10 + 10.10.10
    for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
      bw.write(e0[ch] & 0x3ffu, 10u);
    }
    for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
      bw.write(e1[ch] & 0x3ffu, 10u);
    }
    return;
  }
  const UInt dmask = (1u << static_cast<std::uint32_t>(prec_delta)) - 1u;
  // Base low 10 bits first (R0, G0, B0) -> bits 5..34.
  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
    bw.write(e0[ch] & 0x3ffu, 10u);
  }
  // For each channel: delta low bits + extra high bits of e0 in exact HLSL order.
  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
    bw.write(e1[ch] & dmask, static_cast<std::uint32_t>(prec_delta));
    if (prec_base == 11) {
      bw.write((e0[ch] >> 10u) & 1u, 1u);
    } else if (prec_base == 12) {
      bw.write((e0[ch] >> 11u) & 1u, 1u);
      bw.write((e0[ch] >> 10u) & 1u, 1u);
    } else if (prec_base == 16) {
      for (int bit = 15; bit >= 10; --bit) {
        bw.write((e0[ch] >> static_cast<std::uint32_t>(bit)) & 1u, 1u);
      }
    }
  }
}

void bc6h_pack_2sub_header(
    Bc6hDslBit128 &bw,
    UInt mode_id, // 1, 2, or 10
    UInt part,
    const ArrayUInt<3> &e00,
    const ArrayUInt<3> &e01,
    const ArrayUInt<3> &e10,
    const ArrayUInt<3> &e11) noexcept {
  UInt bx = 0u, by = 0u, bz = 0u;
  $if(mode_id == 1u) {
    // Mode 1: 0x00 (2b), prec(10,5,5,5), e01/e10/e11 are 5-bit deltas
    bx |= ((e10[1] >> 4u) & 1u) << 2u;
    bx |= ((e10[2] >> 4u) & 1u) << 3u;
    bx |= ((e11[2] >> 4u) & 1u) << 4u;
    bx |= (e00[0] & 0x3ffu) << 5u;
    bx |= (e00[1] & 0x3ffu) << 15u;
    bx |= (e00[2] & 0x7fu) << 25u;

    by |= (e00[2] >> 7u) & 0x7u;
    by |= (e01[0] & 0x1fu) << 3u;
    by |= ((e11[1] >> 4u) & 1u) << 8u;
    by |= (e10[1] & 0xfu) << 9u;
    by |= (e01[1] & 0x1fu) << 13u;
    by |= (e11[2] & 1u) << 18u;
    by |= (e11[1] & 0xfu) << 19u;
    by |= (e01[2] & 0x1fu) << 23u;
    by |= ((e11[2] >> 1u) & 1u) << 28u;
    by |= (e10[2] & 0x7u) << 29u;

    bz |= (e10[2] >> 3u) & 1u;
    bz |= (e10[0] & 0x1fu) << 1u;
    bz |= ((e11[2] >> 2u) & 1u) << 6u;
    bz |= (e11[0] & 0x1fu) << 7u;
    bz |= ((e11[2] >> 3u) & 1u) << 12u;
    bz |= (part & 0x1fu) << 13u;
  } $elif(mode_id == 2u) {
    // Mode 2: 0x01 (2b), prec(7,6,6,6)
    bx |= 0x01u;
    bx |= ((e10[1] >> 5u) & 1u) << 2u;
    bx |= ((e11[1] >> 4u) & 1u) << 3u;
    bx |= ((e11[1] >> 5u) & 1u) << 4u;
    bx |= (e00[0] & 0x7fu) << 5u;
    bx |= (e11[2] & 0x3u) << 12u;
    bx |= ((e10[2] >> 4u) & 1u) << 14u;
    bx |= (e00[1] & 0x7fu) << 15u;
    bx |= ((e10[2] >> 5u) & 1u) << 22u;
    bx |= ((e11[2] >> 2u) & 1u) << 23u;
    bx |= ((e10[1] >> 4u) & 1u) << 24u;
    bx |= (e00[2] & 0x7fu) << 25u;

    by |= (e11[2] >> 3u) & 1u;
    by |= ((e11[2] >> 5u) & 1u) << 1u;
    by |= ((e11[2] >> 4u) & 1u) << 2u;
    by |= (e01[0] & 0x3fu) << 3u;
    by |= (e10[1] & 0xfu) << 9u;
    by |= (e01[1] & 0x3fu) << 13u;
    by |= (e11[1] & 0xfu) << 19u;
    by |= (e01[2] & 0x3fu) << 23u;
    by |= (e10[2] & 0x7u) << 29u;

    bz |= (e10[2] >> 3u) & 1u;
    bz |= (e10[0] & 0x3fu) << 1u;
    bz |= (e11[0] & 0x3fu) << 7u;
    bz |= (part & 0x1fu) << 13u;
  } $else {
    // Mode 10: 0x1E (5b), prec(6,6,6,6), untransformed
    bx |= 0x1eu;
    bx |= (e00[0] & 0x3fu) << 5u;
    bx |= ((e11[1] >> 4u) & 1u) << 11u;
    bx |= (e11[2] & 0x3u) << 12u;
    bx |= ((e10[2] >> 4u) & 1u) << 14u;
    bx |= (e00[1] & 0x3fu) << 15u;
    bx |= ((e10[1] >> 5u) & 1u) << 21u;
    bx |= ((e10[2] >> 5u) & 1u) << 22u;
    bx |= ((e11[2] >> 2u) & 1u) << 23u;
    bx |= ((e10[1] >> 4u) & 1u) << 24u;
    bx |= (e00[2] & 0x3fu) << 25u;
    bx |= ((e11[1] >> 5u) & 1u) << 31u;

    by |= (e11[2] >> 3u) & 1u;
    by |= ((e11[2] >> 5u) & 1u) << 1u;
    by |= ((e11[2] >> 4u) & 1u) << 2u;
    by |= (e01[0] & 0x3fu) << 3u;
    by |= (e10[1] & 0xfu) << 9u;
    by |= (e01[1] & 0x3fu) << 13u;
    by |= (e11[1] & 0xfu) << 19u;
    by |= (e01[2] & 0x3fu) << 23u;
    by |= (e10[2] & 0x7u) << 29u;

    bz |= (e10[2] >> 3u) & 1u;
    bz |= (e10[0] & 0x3fu) << 1u;
    bz |= (e11[0] & 0x3fu) << 7u;
    bz |= (part & 0x1fu) << 13u;
  };
  bw.w0 = bx;
  bw.w1 = by;
  bw.w2 = bz;
  bw.w3 = 0u;
  bw.pos = 82u;
}

} // namespace

VtGpuBc6hBlockShader compile_vt_gpu_bc6h_shader(luisa::compute::Device &device) {
  return device.compile<1>(
      [](BufferVar<luisa::float4> in_tiles, BufferVar<luisa::uint> out_blocks,
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

        ArrayInt<16> hr, hg, hb;
        ArrayFloat<16> fr, fg, fb, lum;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const Float4 rgba =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            const UInt u0 = float_to_uf16_dsl(rgba.x);
            const UInt u1 = float_to_uf16_dsl(rgba.y);
            const UInt u2 = float_to_uf16_dsl(rgba.z);
            hr[i] = cast<int>(u0);
            hg[i] = cast<int>(u1);
            hb[i] = cast<int>(u2);
            fr[i] = cast<float>(u0);
            fg[i] = cast<float>(u1);
            fb[i] = cast<float>(u2);
            lum[i] = 0.2126f * fr[i] + 0.7152f * fg[i] + 0.0722f * fb[i];
          }
        }

        ArrayInt<16> w4_arr;
        for (std::uint32_t k = 0u; k < 16u; ++k) {
          w4_arr[k] = kBc6hWeight4[k];
        }
        ArrayInt<16> w3_arr;
        for (std::uint32_t k = 0u; k < 8u; ++k) {
          w3_arr[k] = kBc6hWeight3[k];
        }
        for (std::uint32_t k = 8u; k < 16u; ++k) {
          w3_arr[k] = 64;
        }

        // ---- 1-Subset: Modes 11 (10b), 12 (11+9d), 13 (12+8d), 14 (16+4d) ----
        Float3 e0 = make_float3(0.0f);
        Float3 e1 = make_float3(0.0f);
        bc6h_pca_ls_endpoints_dsl(
            fr, fg, fb, 0u, 0u, false, w4_arr, 15, 3, 2, e0, e1);

        // Anchor 0 fixup before delta quantization: orient e0 closer to px[0].
        const Float3 px0 = make_float3(fr[0], fg[0], fb[0]);
        const Float3 d_p0 = px0 - e0;
        const Float3 d_p1 = px0 - e1;
        $if(dot(d_p0, d_p0) > dot(d_p1, d_p1)) {
          const Float3 t = e0;
          e0 = e1;
          e1 = t;
        };

        // Pre-scaled [0..65535] endpoint candidates matching start_quantize().
        ArrayInt<3> p0_ph, p1_ph;
        p0_ph[0] = clamp(cast<int>(round(e0.x * (65535.0f / 31743.0f))), 0, 65535);
        p0_ph[1] = clamp(cast<int>(round(e0.y * (65535.0f / 31743.0f))), 0, 65535);
        p0_ph[2] = clamp(cast<int>(round(e0.z * (65535.0f / 31743.0f))), 0, 65535);
        p1_ph[0] = clamp(cast<int>(round(e1.x * (65535.0f / 31743.0f))), 0, 65535);
        p1_ph[1] = clamp(cast<int>(round(e1.y * (65535.0f / 31743.0f))), 0, 65535);
        p1_ph[2] = clamp(cast<int>(round(e1.z * (65535.0f / 31743.0f))), 0, 65535);

        Float best_err = 1e30f;
        Bc6hDslBit128 best_bits;

        auto eval_1sub_mode = [&](UInt mode_bits, int prec_base, int prec_delta,
                                  bool transformed) {
          ArrayInt<3> q0, q1, d1;
          Bool bad = false;
          for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
            q0[ch] = bc6h_quantize_uf16_dsl(p0_ph[ch], prec_base);
            q1[ch] = bc6h_quantize_uf16_dsl(p1_ph[ch], prec_base);
          }
          if (transformed) {
            const int d_pos_max = (1 << (prec_delta - 1)) - 1;
            const int d_neg_min = -(1 << (prec_delta - 1));
            for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
              d1[ch] = q1[ch] - q0[ch];
              $if((d1[ch] < d_neg_min) | (d1[ch] > d_pos_max)) { bad = true; };
            }
          } else {
            for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
              d1[ch] = q1[ch];
            }
          }
          $if(!bad) {
            ArrayInt<3> u0, u1;
            for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
              u0[ch] = bc6h_unquantize_uf16_dsl(q0[ch], prec_base);
              u1[ch] = bc6h_unquantize_uf16_dsl(q1[ch], prec_base);
            }
            ArrayInt<16> pal_r, pal_g, pal_b;
            $for(k, 16u) {
              const Int w = w4_arr[k];
              pal_r[k] = (((u0[0] * (64 - w) + u1[0] * w + 32) >> 6) * 31) >> 6;
              pal_g[k] = (((u0[1] * (64 - w) + u1[1] * w + 32) >> 6) * 31) >> 6;
              pal_b[k] = (((u0[2] * (64 - w) + u1[2] * w + 32) >> 6) * 31) >> 6;
            };
            ArrayUInt<16> idx;
            Float err = 0.0f;
            $for(i, 16u) {
              const Float r = fr[i], g = fg[i], b = fb[i];
              UInt bk = 0u;
              Float be = 1e30f;
              $for(k, 16u) {
                const Float dr = cast<float>(pal_r[k]) - r;
                const Float dg = cast<float>(pal_g[k]) - g;
                const Float db = cast<float>(pal_b[k]) - b;
                const Float de = dr * dr + dg * dg + db * db;
                $if(de < be) {
                  be = de;
                  bk = cast<uint>(k);
                };
              };
              idx[i] = bk;
              err += be;
            };
            // Ensure anchor bit idx[0] < 8; re-encode swapped endpoints safely.
            Bool valid_anchor = true;
            $if(idx[0] >= 8u) {
              if (transformed) {
                const int d_pos_max = (1 << (prec_delta - 1)) - 1;
                const int d_neg_min = -(1 << (prec_delta - 1));
                Bool swap_ok = true;
                for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                  const Int nd = -d1[ch];
                  $if((nd < d_neg_min) | (nd > d_pos_max)) { swap_ok = false; };
                }
                $if(swap_ok) {
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    q0[ch] = q1[ch];
                    d1[ch] = -d1[ch];
                  }
                  $for(i, 16u) { idx[i] = 15u - idx[i]; };
                } $else {
                  valid_anchor = false;
                };
              } else {
                for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                  const Int t = q0[ch];
                  q0[ch] = d1[ch];
                  d1[ch] = t;
                }
                $for(i, 16u) { idx[i] = 15u - idx[i]; };
              }
            };
            $if(valid_anchor & (err < best_err)) {
              best_err = err;
              ArrayUInt<3> enc0, enc1;
              for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                enc0[ch] = cast<uint>(q0[ch]);
                enc1[ch] = cast<uint>(d1[ch]);
              }
              Bc6hDslBit128 cand;
              bc6h_pack_1sub_header(
                  cand, mode_bits, prec_base, prec_delta, transformed, enc0, enc1);
              cand.write(idx[0], 3u);
              for (std::uint32_t i = 1u; i < 16u; ++i) {
                cand.write(idx[i], 4u);
              }
              best_bits = cand;
            };
          };
        };

        // Always evaluate all 4 1-subset modes (Mode 11 guarantees fallback).
        eval_1sub_mode(0x03u, 10, 10, false);
        eval_1sub_mode(0x07u, 11, 9, true);
        eval_1sub_mode(0x0bu, 12, 8, true);
        eval_1sub_mode(0x0fu, 16, 4, true);

        // ---- 2-Subset Modes 1, 2, 10 (balanced & high, screened across 32 parts) ----
        const Float gate_2sub = ite(quality >= 2u, 4.0f, 16.0f);
        $if((quality >= 1u) & (best_err > gate_2sub)) {
          ArrayUInt<32> masks2;
          ArrayUInt<32> anchors2;
          for (std::uint32_t p = 0u; p < 32u; ++p) {
            masks2[p] = kBc6hPart2Masks[p];
            anchors2[p] = kBc6hPart2Anchor[p];
          }
          // Screen all 32 partitions using luminance-extremal endpoints (as in
          // bc6_trymode_le10cs), picking top K (K=2 balanced, K=4 high).
          const Int top_k = ite(quality >= 2u, 4, 2);
          ArrayUInt<4> top_p;
          ArrayFloat<4> top_s;
          for (int k = 0; k < 4; ++k) {
            top_p[k] = 0u;
            top_s[k] = 1e30f;
          }
          $for(p, 32u) {
            const UInt pmask = masks2[p];
            Float lum_lo0 = 1e30f, lum_hi0 = -1e30f;
            Float lum_lo1 = 1e30f, lum_hi1 = -1e30f;
            Float3 e00 = make_float3(0.0f), e01 = make_float3(0.0f);
            Float3 e10 = make_float3(0.0f), e11 = make_float3(0.0f);
            $for(i, 16u) {
              const UInt sub = (pmask >> cast<uint>(i)) & 1u;
              const Float lv = lum[i];
              const Float3 px = make_float3(fr[i], fg[i], fb[i]);
              $if(sub == 0u) {
                $if(lv < lum_lo0) { lum_lo0 = lv; e00 = px; };
                $if(lv > lum_hi0) { lum_hi0 = lv; e01 = px; };
              } $else {
                $if(lv < lum_lo1) { lum_lo1 = lv; e10 = px; };
                $if(lv > lum_hi1) { lum_hi1 = lv; e11 = px; };
              };
            };
            const Float3 sp0 = e01 - e00;
            const Float3 sp1 = e11 - e10;
            const Float len0 = max(dot(sp0, sp0), 1e-6f);
            const Float len1 = max(dot(sp1, sp1), 1e-6f);
            Float proxy_err = 0.0f;
            $for(i, 16u) {
              const UInt sub = (pmask >> cast<uint>(i)) & 1u;
              const Float3 px = make_float3(fr[i], fg[i], fb[i]);
              $if(sub == 0u) {
                const Float t = clamp(dot(px - e00, sp0) / len0, 0.0f, 1.0f);
                const Float3 proj = e00 + t * sp0 - px;
                proxy_err += dot(proj, proj);
              } $else {
                const Float t = clamp(dot(px - e10, sp1) / len1, 0.0f, 1.0f);
                const Float3 proj = e10 + t * sp1 - px;
                proxy_err += dot(proj, proj);
              };
            };
            // Insert into sorted top-4 array.
            UInt cur_p = cast<uint>(p);
            Float cur_s = proxy_err;
            for (int r = 0; r < 4; ++r) {
              $if(cur_s < top_s[r]) {
                const Float tmp_s = top_s[r];
                const UInt tmp_p = top_p[r];
                top_s[r] = cur_s;
                top_p[r] = cur_p;
                cur_s = tmp_s;
                cur_p = tmp_p;
              };
            }
          };

          const int ls_iters_2sub = 1;
          $for(ti, top_k) {
            const UInt part = top_p[ti];
            const UInt pmask = masks2[part];
            const UInt anchor2 = anchors2[part];

            Float3 e00 = make_float3(0.0f), e01 = make_float3(0.0f);
            Float3 e10 = make_float3(0.0f), e11 = make_float3(0.0f);
            bc6h_pca_ls_endpoints_dsl(
                fr, fg, fb, pmask, 0u, true, w3_arr, 7, 2, ls_iters_2sub, e00, e01);
            bc6h_pca_ls_endpoints_dsl(
                fr, fg, fb, pmask, 1u, true, w3_arr, 7, 2, ls_iters_2sub, e10, e11);

            // Anchor fixup before quantization: orient subset 0 to px[0], subset 1 to px[anchor2].
            {
              const Float3 p_a0 = make_float3(fr[0], fg[0], fb[0]);
              const Float3 d0a = p_a0 - e00;
              const Float3 d0b = p_a0 - e01;
              $if(dot(d0a, d0a) > dot(d0b, d0b)) {
                const Float3 t = e00;
                e00 = e01;
                e01 = t;
              };
              const Float3 p_a2 =
                  make_float3(fr[anchor2], fg[anchor2], fb[anchor2]);
              const Float3 d1a = p_a2 - e10;
              const Float3 d1b = p_a2 - e11;
              $if(dot(d1a, d1a) > dot(d1b, d1b)) {
                const Float3 t = e10;
                e10 = e11;
                e11 = t;
              };
            }

            ArrayInt<3> p00, p01, p10, p11;
            p00[0] = clamp(cast<int>(round(e00.x * (65535.0f / 31743.0f))), 0, 65535);
            p00[1] = clamp(cast<int>(round(e00.y * (65535.0f / 31743.0f))), 0, 65535);
            p00[2] = clamp(cast<int>(round(e00.z * (65535.0f / 31743.0f))), 0, 65535);
            p01[0] = clamp(cast<int>(round(e01.x * (65535.0f / 31743.0f))), 0, 65535);
            p01[1] = clamp(cast<int>(round(e01.y * (65535.0f / 31743.0f))), 0, 65535);
            p01[2] = clamp(cast<int>(round(e01.z * (65535.0f / 31743.0f))), 0, 65535);
            p10[0] = clamp(cast<int>(round(e10.x * (65535.0f / 31743.0f))), 0, 65535);
            p10[1] = clamp(cast<int>(round(e10.y * (65535.0f / 31743.0f))), 0, 65535);
            p10[2] = clamp(cast<int>(round(e10.z * (65535.0f / 31743.0f))), 0, 65535);
            p11[0] = clamp(cast<int>(round(e11.x * (65535.0f / 31743.0f))), 0, 65535);
            p11[1] = clamp(cast<int>(round(e11.y * (65535.0f / 31743.0f))), 0, 65535);
            p11[2] = clamp(cast<int>(round(e11.z * (65535.0f / 31743.0f))), 0, 65535);

            auto eval_2sub_mode = [&](UInt mode_id, int prec_base,
                                      int prec_delta, bool transformed) {
              ArrayInt<3> q00, q01, q10, q11, d01, d10, d11;
              Bool bad = false;
              for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                q00[ch] = bc6h_quantize_uf16_dsl(p00[ch], prec_base);
                q01[ch] = bc6h_quantize_uf16_dsl(p01[ch], prec_base);
                q10[ch] = bc6h_quantize_uf16_dsl(p10[ch], prec_base);
                q11[ch] = bc6h_quantize_uf16_dsl(p11[ch], prec_base);
              }
              auto check_deltas = [&]() noexcept -> Bool {
                if (!transformed) {
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    d01[ch] = q01[ch];
                    d10[ch] = q10[ch];
                    d11[ch] = q11[ch];
                  }
                  return true;
                }
                const int d_pos_max = (1 << (prec_delta - 1)) - 1;
                const int d_neg_min = -(1 << (prec_delta - 1));
                Bool ok = true;
                for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                  d01[ch] = q01[ch] - q00[ch];
                  d10[ch] = q10[ch] - q00[ch];
                  d11[ch] = q11[ch] - q00[ch];
                  $if((d01[ch] < d_neg_min) | (d01[ch] > d_pos_max) |
                      (d10[ch] < d_neg_min) | (d10[ch] > d_pos_max) |
                      (d11[ch] < d_neg_min) | (d11[ch] > d_pos_max)) {
                    ok = false;
                  };
                }
                return ok;
              };
              bad = !check_deltas();
              $if(!bad) {
                ArrayInt<3> u00, u01, u10, u11;
                for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                  u00[ch] = bc6h_unquantize_uf16_dsl(q00[ch], prec_base);
                  u01[ch] = bc6h_unquantize_uf16_dsl(q01[ch], prec_base);
                  u10[ch] = bc6h_unquantize_uf16_dsl(q10[ch], prec_base);
                  u11[ch] = bc6h_unquantize_uf16_dsl(q11[ch], prec_base);
                }
                ArrayInt<8> p0r, p0g, p0b, p1r, p1g, p1b;
                $for(k, 8u) {
                  const Int w = w3_arr[k];
                  p0r[k] = (((u00[0] * (64 - w) + u01[0] * w + 32) >> 6) * 31) >> 6;
                  p0g[k] = (((u00[1] * (64 - w) + u01[1] * w + 32) >> 6) * 31) >> 6;
                  p0b[k] = (((u00[2] * (64 - w) + u01[2] * w + 32) >> 6) * 31) >> 6;
                  p1r[k] = (((u10[0] * (64 - w) + u11[0] * w + 32) >> 6) * 31) >> 6;
                  p1g[k] = (((u10[1] * (64 - w) + u11[1] * w + 32) >> 6) * 31) >> 6;
                  p1b[k] = (((u10[2] * (64 - w) + u11[2] * w + 32) >> 6) * 31) >> 6;
                };
                ArrayUInt<16> idx;
                Float err = 0.0f;
                $for(i, 16u) {
                  const UInt sub = (pmask >> cast<uint>(i)) & 1u;
                  const Float r = fr[i], g = fg[i], b = fb[i];
                  UInt bk = 0u;
                  Float be = 1e30f;
                  $if(sub == 0u) {
                    $for(k, 8u) {
                      const Float dr = cast<float>(p0r[k]) - r;
                      const Float dg = cast<float>(p0g[k]) - g;
                      const Float db = cast<float>(p0b[k]) - b;
                      const Float de = dr * dr + dg * dg + db * db;
                      $if(de < be) {
                        be = de;
                        bk = cast<uint>(k);
                      };
                    };
                  } $else {
                    $for(k, 8u) {
                      const Float dr = cast<float>(p1r[k]) - r;
                      const Float dg = cast<float>(p1g[k]) - g;
                      const Float db = cast<float>(p1b[k]) - b;
                      const Float de = dr * dr + dg * dg + db * db;
                      $if(de < be) {
                        be = de;
                        bk = cast<uint>(k);
                      };
                    };
                  };
                  idx[i] = bk;
                  err += be;
                };

                Bool anchor_ok = true;
                $if(idx[0] >= 4u) {
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    const Int t = q00[ch];
                    q00[ch] = q01[ch];
                    q01[ch] = t;
                  }
                  $for(i, 16u) {
                    $if(((pmask >> cast<uint>(i)) & 1u) == 0u) {
                      idx[i] = 7u - idx[i];
                    };
                  };
                };
                $if(idx[anchor2] >= 4u) {
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    const Int t = q10[ch];
                    q10[ch] = q11[ch];
                    q11[ch] = t;
                  }
                  $for(i, 16u) {
                    $if(((pmask >> cast<uint>(i)) & 1u) == 1u) {
                      idx[i] = 7u - idx[i];
                    };
                  };
                };
                anchor_ok = check_deltas();
                $if(anchor_ok & (err < best_err)) {
                  best_err = err;
                  ArrayUInt<3> enc00, enc01, enc10, enc11;
                  for (std::uint32_t ch = 0u; ch < 3u; ++ch) {
                    enc00[ch] = cast<uint>(q00[ch]);
                    enc01[ch] = cast<uint>(d01[ch]);
                    enc10[ch] = cast<uint>(d10[ch]);
                    enc11[ch] = cast<uint>(d11[ch]);
                  }
                  Bc6hDslBit128 cand;
                  bc6h_pack_2sub_header(
                      cand, mode_id, part, enc00, enc01, enc10, enc11);
                  for (std::uint32_t i = 0u; i < 16u; ++i) {
                    const UInt bits = ite((i == 0u) | (anchor2 == i), 2u, 3u);
                    cand.write(idx[i], bits);
                  }
                  best_bits = cand;
                };
              };
            };

            eval_2sub_mode(1u, 10, 5, true);
            eval_2sub_mode(2u, 7, 6, true);
            eval_2sub_mode(10u, 6, 6, false);
          };
        };

        out_blocks.write(gid * 4u + 0u, best_bits.w0);
        out_blocks.write(gid * 4u + 1u, best_bits.w1);
        out_blocks.write(gid * 4u + 2u, best_bits.w2);
        out_blocks.write(gid * 4u + 3u, best_bits.w3);
      });
}

} // namespace psycles::luisa_backend::detail
