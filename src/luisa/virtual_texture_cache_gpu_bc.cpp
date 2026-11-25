#include "virtual_texture_cache_gpu_bc.h"

#include "virtual_texture_cache_prepack.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <luisa/backends/ext/tex_compress_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa;
using namespace luisa::compute;

inline constexpr std::uint32_t kMaxGpuBcBatchTiles = 256u;
inline constexpr std::uint32_t kMaxExtStripHeight = 16384u;

[[nodiscard]] VtGpuBcMode parse_gpu_bc_mode_from_env() noexcept {
  const char *env = std::getenv("PSYCLES_VT_GPU_BC");
  if (env == nullptr || *env == '\0') {
    return VtGpuBcMode::auto_select;
  }
  const std::string_view sv{env};
  if (sv == "off" || sv == "OFF" || sv == "0" || sv == "false" || sv == "FALSE") {
    return VtGpuBcMode::off;
  }
  if (sv == "ext" || sv == "EXT") {
    return VtGpuBcMode::force_ext;
  }
  if (sv == "dsl" || sv == "DSL") {
    return VtGpuBcMode::force_dsl;
  }
  return VtGpuBcMode::auto_select;
}

[[nodiscard]] VtGpuBcQuality parse_gpu_bc_quality_from_env() noexcept {
  const char *env = std::getenv("PSYCLES_VT_GPU_BC_QUALITY");
  if (env == nullptr || *env == '\0') {
    return VtGpuBcQuality::balanced;
  }
  const std::string_view sv{env};
  if (sv == "fast" || sv == "FAST" || sv == "low" || sv == "0") {
    return VtGpuBcQuality::fast;
  }
  if (sv == "high" || sv == "HIGH" || sv == "hq" || sv == "slow" || sv == "2") {
    return VtGpuBcQuality::high;
  }
  return VtGpuBcQuality::balanced;
}

[[nodiscard]] std::uint32_t next_pow2_u32(std::uint32_t v) noexcept {
  std::uint32_t p = 1u;
  while (p < v && p < kMaxGpuBcBatchTiles) {
    p <<= 1u;
  }
  return p;
}

[[nodiscard]] double elapsed_ms_since(
    std::chrono::steady_clock::time_point t0) noexcept {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

void compute_block_origin_dsl(
    UInt P,
    UInt gid,
    UInt &tile_base,
    UInt &base_x,
    UInt &base_y) noexcept {
  const UInt bx_count = P >> 2u;
  const UInt blocks_per_tile = bx_count * bx_count;
  const UInt tile_idx = gid / blocks_per_tile;
  const UInt local_blk = gid - tile_idx * blocks_per_tile;
  const UInt blk_y = local_blk / bx_count;
  const UInt blk_x = local_blk - blk_y * bx_count;
  tile_base = tile_idx * P * P;
  base_x = blk_x * 4u;
  base_y = blk_y * 4u;
}

// Encodes 16 single-channel texels in [0, 255] as an 8-byte BC4 block (UInt2).
[[nodiscard]] UInt2 encode_bc4_channel_dsl(const ArrayInt<16> &vals) noexcept {
  Int min_v = 255;
  Int max_v = 0;
  $for(i, 16u) {
    min_v = min(min_v, vals[i]);
    max_v = max(max_v, vals[i]);
  };

  UInt2 result = make_uint2(0u, 0u);
  $if(min_v == max_v) {
    const UInt c = cast<uint>(min_v);
    result = make_uint2(c | (c << 8u), 0u);
  } $else {
    auto pack_indices = [](const ArrayUInt<16> &idx, UInt a0,
                           UInt a1) noexcept -> UInt2 {
      UInt lo16 = 0u;
      UInt hi32 = 0u;
      for (std::uint32_t i = 0u; i < 16u; ++i) {
        const std::uint32_t bit = i * 3u;
        const UInt k = idx[i] & 0x7u;
        if (bit < 16u) {
          lo16 |= (k << bit);
          if (bit + 3u > 16u) {
            hi32 |= (k >> (16u - bit));
          }
        } else {
          hi32 |= (k << (bit - 16u));
        }
      }
      return make_uint2(a0 | (a1 << 8u) | ((lo16 & 0xffffu) << 16u), hi32);
    };

    // Mode A: 8 interpolated values (a0 > a1).
    Int best_a0 = max_v;
    Int best_a1 = min_v;
    ArrayInt<8> pal_a;
    pal_a[0] = best_a0;
    pal_a[1] = best_a1;
    for (int i = 1; i <= 6; ++i) {
      pal_a[i + 1] = ((7 - i) * best_a0 + i * best_a1 + 3) / 7;
    }
    ArrayUInt<16> best_idx;
    UInt best_sse = 0u;
    $for(i, 16u) {
      const Int v = vals[i];
      UInt bk = 0u;
      Int be = 0x7fffffff;
      $for(k, 8u) {
        const Int d = pal_a[k] - v;
        const Int e = d * d;
        $if(e < be) {
          be = e;
          bk = cast<uint>(k);
        };
      };
      best_idx[i] = bk;
      best_sse += cast<uint>(be);
    };

    // Mode B: 6 interpolated values + exact {0, 255} (b0 <= b1) for cutouts.
    $if((min_v == 0) | (max_v == 255)) {
      Int in_min = 255;
      Int in_max = 0;
      $for(i, 16u) {
        const Int v = vals[i];
        $if((v > 0) & (v < 255)) {
          in_min = min(in_min, v);
          in_max = max(in_max, v);
        };
      };
      $if(in_min <= in_max) {
        const Int b0 = in_min;
        const Int b1 = in_max;
        ArrayInt<8> pal_b;
        pal_b[0] = b0;
        pal_b[1] = b1;
        for (int i = 1; i <= 4; ++i) {
          pal_b[i + 1] = ((5 - i) * b0 + i * b1 + 2) / 5;
        }
        pal_b[6] = 0;
        pal_b[7] = 255;
        ArrayUInt<16> idx_b;
        UInt sse_b = 0u;
        $for(i, 16u) {
          const Int v = vals[i];
          UInt bk = 0u;
          Int be = 0x7fffffff;
          $for(k, 8u) {
            const Int d = pal_b[k] - v;
            const Int e = d * d;
            $if(e < be) {
              be = e;
              bk = cast<uint>(k);
            };
          };
          idx_b[i] = bk;
          sse_b += cast<uint>(be);
        };
        $if(sse_b < best_sse) {
          best_sse = sse_b;
          best_a0 = b0;
          best_a1 = b1;
          $for(i, 16u) { best_idx[i] = idx_b[i]; };
        };
      };
    };

    result = pack_indices(best_idx, cast<uint>(best_a0), cast<uint>(best_a1));
  };
  return result;
}

// Encodes 16 RGB texels in [0, 255] as an 8-byte opaque 4-color BC1 block.
[[nodiscard]] UInt2 encode_bc1_rgb_dsl(
    const ArrayInt<16> &pr,
    const ArrayInt<16> &pg,
    const ArrayInt<16> &pb) noexcept {
  Int min_r = 255, max_r = 0;
  Int min_g = 255, max_g = 0;
  Int min_b = 255, max_b = 0;
  ArrayFloat<16> fr, fg, fb;
  Float3 mean = make_float3(0.0f);
  $for(i, 16u) {
    const Int r = pr[i], g = pg[i], b = pb[i];
    min_r = min(min_r, r);
    max_r = max(max_r, r);
    min_g = min(min_g, g);
    max_g = max(max_g, g);
    min_b = min(min_b, b);
    max_b = max(max_b, b);
    const Float rf = cast<float>(r);
    const Float gf = cast<float>(g);
    const Float bf = cast<float>(b);
    fr[i] = rf;
    fg[i] = gf;
    fb[i] = bf;
    mean += make_float3(rf, gf, bf);
  };
  mean *= (1.0f / 16.0f);

  auto to_rgb565 = [](Int r, Int g, Int b) noexcept -> UInt {
    const UInt r5 = cast<uint>(clamp((r * 31 + 127) / 255, 0, 31));
    const UInt g6 = cast<uint>(clamp((g * 63 + 127) / 255, 0, 63));
    const UInt b5 = cast<uint>(clamp((b * 31 + 127) / 255, 0, 31));
    return (r5 << 11u) | (g6 << 5u) | b5;
  };
  auto unpack_rgb565_r = [](UInt c) noexcept -> Int {
    const Int v = cast<int>((c >> 11u) & 0x1fu);
    return (v << 3) | (v >> 2);
  };
  auto unpack_rgb565_g = [](UInt c) noexcept -> Int {
    const Int v = cast<int>((c >> 5u) & 0x3fu);
    return (v << 2) | (v >> 4);
  };
  auto unpack_rgb565_b = [](UInt c) noexcept -> Int {
    const Int v = cast<int>(c & 0x1fu);
    return (v << 3) | (v >> 2);
  };

  UInt2 out = make_uint2(0u, 0u);
  $if((min_r == max_r) & (min_g == max_g) & (min_b == max_b)) {
    const UInt c = to_rgb565(min_r, min_g, min_b);
    out = make_uint2(c | (c << 16u), 0u);
  } $else {
    Float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f;
    Float c11 = 0.0f, c12 = 0.0f, c22 = 0.0f;
    $for(i, 16u) {
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
    Float3 axis = make_float3(cast<float>(max_r - min_r),
                              cast<float>(max_g - min_g),
                              cast<float>(max_b - min_b));
    Float3 e0 = mean;
    Float3 e1 = mean;
    const Float norm2 = dot(axis, axis);
    $if(norm2 > 1e-6f) {
      axis /= sqrt(norm2);
      for (int it = 0; it < 4; ++it) {
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
        const Float p = dot(make_float3(fr[i], fg[i], fb[i]) - mean, axis);
        tmin = min(tmin, p);
        tmax = max(tmax, p);
      };
      e0 = clamp(mean + tmin * axis, 0.0f, 255.0f);
      e1 = clamp(mean + tmax * axis, 0.0f, 255.0f);
    };

    // 2x 2-bit least-squares refinement matching BC1 4-color weights {0, 1/3, 2/3, 1}.
    ArrayFloat<4> wf;
    wf[0] = 0.0f;
    wf[1] = 1.0f;
    wf[2] = 1.0f / 3.0f;
    wf[3] = 2.0f / 3.0f;
    ArrayFloat<4> lin_w;
    lin_w[0] = 0.0f;
    lin_w[1] = 1.0f / 3.0f;
    lin_w[2] = 2.0f / 3.0f;
    lin_w[3] = 1.0f;

    for (int it = 0; it < 2; ++it) {
      const Float3 diff = e1 - e0;
      const Float len2 = dot(diff, diff);
      $if(len2 > 1e-6f) {
        const Float inv_l2 = 1.0f / len2;
        Float a2 = 0.0f, ab = 0.0f, b2 = 0.0f;
        Float3 rhs0 = make_float3(0.0f);
        Float3 rhs1 = make_float3(0.0f);
        $for(i, 16u) {
          const Float3 px = make_float3(fr[i], fg[i], fb[i]);
          const Float t = clamp(dot(px - e0, diff) * inv_l2, 0.0f, 1.0f);
          const Int k = clamp(cast<int>(round(t * 3.0f)), 0, 3);
          const Float b = lin_w[k];
          const Float a = 1.0f - b;
          a2 += a * a;
          ab += a * b;
          b2 += b * b;
          rhs0 += a * px;
          rhs1 += b * px;
        };
        const Float det = a2 * b2 - ab * ab;
        $if(det > 1e-6f) {
          const Float inv_d = 1.0f / det;
          e0 = clamp((b2 * rhs0 - ab * rhs1) * inv_d, 0.0f, 255.0f);
          e1 = clamp((-ab * rhs0 + a2 * rhs1) * inv_d, 0.0f, 255.0f);
        };
      };
    }

    UInt c0 = to_rgb565(cast<int>(round(e0.x)), cast<int>(round(e0.y)),
                        cast<int>(round(e0.z)));
    UInt c1 = to_rgb565(cast<int>(round(e1.x)), cast<int>(round(e1.y)),
                        cast<int>(round(e1.z)));
    $if(c0 < c1) {
      const UInt t = c0;
      c0 = c1;
      c1 = t;
    };
    $if(c0 == c1) {
      out = make_uint2(c0 | (c0 << 16u), 0u);
    } $else {
      ArrayInt<4> pal_r, pal_g, pal_b;
      pal_r[0] = unpack_rgb565_r(c0);
      pal_g[0] = unpack_rgb565_g(c0);
      pal_b[0] = unpack_rgb565_b(c0);
      pal_r[1] = unpack_rgb565_r(c1);
      pal_g[1] = unpack_rgb565_g(c1);
      pal_b[1] = unpack_rgb565_b(c1);
      pal_r[2] = (2 * pal_r[0] + pal_r[1] + 1) / 3;
      pal_g[2] = (2 * pal_g[0] + pal_g[1] + 1) / 3;
      pal_b[2] = (2 * pal_b[0] + pal_b[1] + 1) / 3;
      pal_r[3] = (pal_r[0] + 2 * pal_r[1] + 1) / 3;
      pal_g[3] = (pal_g[0] + 2 * pal_g[1] + 1) / 3;
      pal_b[3] = (pal_b[0] + 2 * pal_b[1] + 1) / 3;

      UInt indices = 0u;
      $for(i, 16u) {
        const Int r = pr[i], g = pg[i], b = pb[i];
        UInt bk = 0u;
        Int be = 0x7fffffff;
        $for(k, 4u) {
          const Int dr = pal_r[k] - r;
          const Int dg = pal_g[k] - g;
          const Int db = pal_b[k] - b;
          const Int e = dr * dr + dg * dg + db * db;
          $if(e < be) {
            be = e;
            bk = cast<uint>(k);
          };
        };
        indices |= (bk << (cast<uint>(i) * 2u));
      };
      out = make_uint2(c0 | (c1 << 16u), indices);
    };
  };
  return out;
}

} // namespace

VtGpuBcCompressor::VtGpuBcCompressor(Device &device) noexcept
    : _device{device},
      _mode{parse_gpu_bc_mode_from_env()},
      _quality{parse_gpu_bc_quality_from_env()} {
  init_backend();
}

VtGpuBcCompressor::VtGpuBcCompressor(
    Device &device,
    VtGpuBcQuality quality_override) noexcept
    : _device{device},
      _mode{parse_gpu_bc_mode_from_env()},
      _quality{quality_override} {
  init_backend();
}

VtGpuBcCompressor::VtGpuBcCompressor(
    Device &device,
    VtGpuBcMode mode_override,
    VtGpuBcQuality quality_override) noexcept
    : _device{device}, _mode{mode_override}, _quality{quality_override} {
  init_backend();
}

void VtGpuBcCompressor::init_backend() noexcept {
  if (_mode == VtGpuBcMode::off) {
    _backend = Backend::none;
    _min_batch_tiles = UINT32_MAX;
    return;
  }
  if (_mode == VtGpuBcMode::force_ext) {
    _tex_compress_ext = _device.extension<TexCompressExt>();
    if (_tex_compress_ext != nullptr) {
      _backend = Backend::tex_compress_ext;
      _min_batch_tiles = 1u;
      return;
    }
    _backend = Backend::none;
    _min_batch_tiles = UINT32_MAX;
    return;
  }
  _backend = Backend::dsl;
  _min_batch_tiles = 1u;
}

const char *VtGpuBcCompressor::backend_name() const noexcept {
  switch (_backend) {
    case Backend::tex_compress_ext:
      return "tex_compress_ext";
    case Backend::dsl:
      return "dsl";
    case Backend::none:
    default:
      return "off";
  }
}

std::uint64_t VtGpuBcCompressor::tiles_encoded() const noexcept {
  std::lock_guard lock{_mutex};
  return _tiles_encoded;
}

double VtGpuBcCompressor::encode_ms() const noexcept {
  std::lock_guard lock{_mutex};
  return _encode_ms;
}

double VtGpuBcCompressor::gpu_kernel_ms() const noexcept {
  std::lock_guard lock{_mutex};
  return _gpu_kernel_ms;
}

double VtGpuBcCompressor::last_gpu_kernel_ms() const noexcept {
  std::lock_guard lock{_mutex};
  return _last_gpu_kernel_ms;
}

void VtGpuBcCompressor::reset_stats() noexcept {
  std::lock_guard lock{_mutex};
  _tiles_encoded = 0ull;
  _encode_ms = 0.0;
  _gpu_kernel_ms = 0.0;
  _last_gpu_kernel_ms = 0.0;
}

void VtGpuBcCompressor::warmup(std::span<const VtTileFormat> formats) noexcept {
  if (_backend == Backend::none || formats.empty()) {
    return;
  }
  std::lock_guard lock{_mutex};
  bool checked_ext = false;
  for (const VtTileFormat fmt : formats) {
    if (!supports_format(fmt)) {
      continue;
    }
    if (_backend == Backend::tex_compress_ext &&
        (fmt == VtTileFormat::bc7 || fmt == VtTileFormat::bc6h)) {
      if (!checked_ext && _tex_compress_ext != nullptr) {
        static_cast<void>(_tex_compress_ext->check_builtin_shader());
        checked_ext = true;
      }
      continue;
    }
    ensure_stream_locked();
    ensure_shader_locked(fmt);
  }
}

void VtGpuBcCompressor::warmup(bool need_bc7, bool need_bc6h) noexcept {
  std::array<VtTileFormat, 2u> fmts{};
  std::size_t count = 0u;
  if (need_bc7) {
    fmts[count++] = VtTileFormat::bc7;
  }
  if (need_bc6h) {
    fmts[count++] = VtTileFormat::bc6h;
  }
  warmup(std::span<const VtTileFormat>{fmts.data(), count});
}

void VtGpuBcCompressor::ensure_stream_locked() {
  if (!_compute_stream) {
    _compute_stream = _device.create_stream(StreamTag::COMPUTE);
  }
}

void VtGpuBcCompressor::ensure_shader_locked(VtTileFormat fmt) {
  switch (fmt) {
    case VtTileFormat::bc1:
      ensure_bc1_shader_locked();
      break;
    case VtTileFormat::bc3:
      ensure_bc3_shader_locked();
      break;
    case VtTileFormat::bc4:
      ensure_bc4_shader_locked();
      break;
    case VtTileFormat::bc5:
      ensure_bc5_shader_locked();
      break;
    case VtTileFormat::bc6h:
      ensure_bc6h_shader_locked();
      break;
    case VtTileFormat::bc7:
      ensure_bc7_shader_locked();
      break;
    default:
      break;
  }
}

void VtGpuBcCompressor::ensure_bc1_shader_locked() {
  if (_has_bc1_shader) {
    return;
  }
  _bc1_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> in_tiles, BufferVar<luisa::uint> out_blocks,
         UInt P, UInt total_blocks, UInt /*quality*/) noexcept {
        const UInt gid = dispatch_x();
        $if(gid >= total_blocks) { $return(); };
        UInt tile_base = 0u, base_x = 0u, base_y = 0u;
        compute_block_origin_dsl(P, gid, tile_base, base_x, base_y);
        ArrayInt<16> pr, pg, pb;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const UInt packed =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            pr[i] = cast<int>(packed & 0xffu);
            pg[i] = cast<int>((packed >> 8u) & 0xffu);
            pb[i] = cast<int>((packed >> 16u) & 0xffu);
          }
        }
        const UInt2 blk = encode_bc1_rgb_dsl(pr, pg, pb);
        out_blocks.write(gid * 2u + 0u, blk.x);
        out_blocks.write(gid * 2u + 1u, blk.y);
      });
  _has_bc1_shader = true;
}

void VtGpuBcCompressor::ensure_bc3_shader_locked() {
  if (_has_bc3_shader) {
    return;
  }
  _bc3_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> in_tiles, BufferVar<luisa::uint> out_blocks,
         UInt P, UInt total_blocks, UInt /*quality*/) noexcept {
        const UInt gid = dispatch_x();
        $if(gid >= total_blocks) { $return(); };
        UInt tile_base = 0u, base_x = 0u, base_y = 0u;
        compute_block_origin_dsl(P, gid, tile_base, base_x, base_y);
        ArrayInt<16> pr, pg, pb, pa;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const UInt packed =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            pr[i] = cast<int>(packed & 0xffu);
            pg[i] = cast<int>((packed >> 8u) & 0xffu);
            pb[i] = cast<int>((packed >> 16u) & 0xffu);
            pa[i] = cast<int>((packed >> 24u) & 0xffu);
          }
        }
        const UInt2 a_blk = encode_bc4_channel_dsl(pa);
        const UInt2 rgb_blk = encode_bc1_rgb_dsl(pr, pg, pb);
        out_blocks.write(gid * 4u + 0u, a_blk.x);
        out_blocks.write(gid * 4u + 1u, a_blk.y);
        out_blocks.write(gid * 4u + 2u, rgb_blk.x);
        out_blocks.write(gid * 4u + 3u, rgb_blk.y);
      });
  _has_bc3_shader = true;
}

void VtGpuBcCompressor::ensure_bc4_shader_locked() {
  if (_has_bc4_shader) {
    return;
  }
  _bc4_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> in_tiles, BufferVar<luisa::uint> out_blocks,
         UInt P, UInt total_blocks, UInt /*quality*/) noexcept {
        const UInt gid = dispatch_x();
        $if(gid >= total_blocks) { $return(); };
        UInt tile_base = 0u, base_x = 0u, base_y = 0u;
        compute_block_origin_dsl(P, gid, tile_base, base_x, base_y);
        ArrayInt<16> pr;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const UInt packed =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            pr[i] = cast<int>(packed & 0xffu);
          }
        }
        const UInt2 blk = encode_bc4_channel_dsl(pr);
        out_blocks.write(gid * 2u + 0u, blk.x);
        out_blocks.write(gid * 2u + 1u, blk.y);
      });
  _has_bc4_shader = true;
}

void VtGpuBcCompressor::ensure_bc5_shader_locked() {
  if (_has_bc5_shader) {
    return;
  }
  _bc5_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> in_tiles, BufferVar<luisa::uint> out_blocks,
         UInt P, UInt total_blocks, UInt /*quality*/) noexcept {
        const UInt gid = dispatch_x();
        $if(gid >= total_blocks) { $return(); };
        UInt tile_base = 0u, base_x = 0u, base_y = 0u;
        compute_block_origin_dsl(P, gid, tile_base, base_x, base_y);
        // Normal maps are swizzled by Cycles / vt_encode_bc5_tile into R (ch0)
        // and A (ch3).
        ArrayInt<16> pr, pa;
        for (std::uint32_t py = 0u; py < 4u; ++py) {
          for (std::uint32_t px = 0u; px < 4u; ++px) {
            const std::uint32_t i = py * 4u + px;
            const UInt packed =
                in_tiles.read(tile_base + (base_y + py) * P + (base_x + px));
            pr[i] = cast<int>(packed & 0xffu);
            pa[i] = cast<int>((packed >> 24u) & 0xffu);
          }
        }
        const UInt2 blk0 = encode_bc4_channel_dsl(pr);
        const UInt2 blk1 = encode_bc4_channel_dsl(pa);
        out_blocks.write(gid * 4u + 0u, blk0.x);
        out_blocks.write(gid * 4u + 1u, blk0.y);
        out_blocks.write(gid * 4u + 2u, blk1.x);
        out_blocks.write(gid * 4u + 3u, blk1.y);
      });
  _has_bc5_shader = true;
}

void VtGpuBcCompressor::ensure_bc7_shader_locked() {
  if (_has_bc7_shader) {
    return;
  }
  _bc7_shader = compile_vt_gpu_bc7_shader(_device);
  _has_bc7_shader = true;
}

void VtGpuBcCompressor::ensure_bc6h_shader_locked() {
  if (_has_bc6h_shader) {
    return;
  }
  _bc6h_shader = compile_vt_gpu_bc6h_shader(_device);
  _has_bc6h_shader = true;
}

VtGpuBcCompressor::ExtBucket *VtGpuBcCompressor::acquire_ext_bucket_locked(
    std::uint32_t tile_size,
    std::uint32_t tile_count,
    bool is_bc6h) {
  const std::uint32_t max_tiles =
      std::max<std::uint32_t>(1u, kMaxExtStripHeight / tile_size);
  const std::uint32_t bucket = std::min(next_pow2_u32(tile_count), max_tiles);
  for (auto &b : _ext_buckets) {
    if (b.tile_size == tile_size && b.bucket_tiles == bucket &&
        b.is_bc6h == is_bc6h) {
      return &b;
    }
  }
  const std::uint32_t blocks_per_tile = (tile_size >> 2u) * (tile_size >> 2u);
  ExtBucket nb;
  nb.tile_size = tile_size;
  nb.bucket_tiles = bucket;
  nb.is_bc6h = is_bc6h;
  nb.src_image = _device.create_image<float>(
      is_bc6h ? PixelStorage::FLOAT4 : PixelStorage::BYTE4,
      tile_size, bucket * tile_size, 1u);
  nb.dst_buffer = _device.create_buffer<luisa::uint>(
      static_cast<std::size_t>(bucket) * blocks_per_tile * 4u);
  if (!nb.src_image || !nb.dst_buffer) {
    return nullptr;
  }
  _ext_buckets.push_back(std::move(nb));
  return &_ext_buckets.back();
}

void VtGpuBcCompressor::ensure_dsl_buffers_locked(
    std::uint32_t tile_size,
    std::uint32_t tile_count,
    bool need_bc6h) {
  const std::uint32_t cap = next_pow2_u32(tile_count);
  const std::uint32_t blocks_per_tile = (tile_size >> 2u) * (tile_size >> 2u);
  const std::size_t texels_total =
      static_cast<std::size_t>(cap) * tile_size * tile_size;
  const std::size_t blocks_u32_total =
      static_cast<std::size_t>(cap) * blocks_per_tile * 4u;

  if (_dsl_buffers.tile_size != tile_size) {
    _dsl_buffers.tile_size = tile_size;
    _dsl_buffers.capacity_tiles = 0u;
  }
  if (_dsl_buffers.capacity_tiles < cap) {
    _dsl_buffers.capacity_tiles = cap;
  }
  if (!_dsl_buffers.out_blocks_u32 ||
      _dsl_buffers.out_blocks_u32.size() < blocks_u32_total) {
    _dsl_buffers.out_blocks_u32 =
        _device.create_buffer<luisa::uint>(blocks_u32_total);
  }
  if (!need_bc6h &&
      (!_dsl_buffers.ldr_in_u32 ||
       _dsl_buffers.ldr_in_u32.size() < texels_total)) {
    _dsl_buffers.ldr_in_u32 = _device.create_buffer<luisa::uint>(texels_total);
  }
  if (need_bc6h &&
      (!_dsl_buffers.bc6h_in_f4 ||
       _dsl_buffers.bc6h_in_f4.size() < texels_total)) {
    _dsl_buffers.bc6h_in_f4 =
        _device.create_buffer<luisa::float4>(texels_total);
  }
}

bool VtGpuBcCompressor::compress_group_ext_locked(
    std::span<const VtGpuBcTileJob *const> group_jobs,
    VtTileFormat fmt,
    std::uint32_t P) {
  if (_tex_compress_ext == nullptr) {
    return false;
  }
  const std::uint32_t N = static_cast<std::uint32_t>(group_jobs.size());
  const std::size_t texels = static_cast<std::size_t>(P) * P;
  const std::size_t blocks_per_tile =
      static_cast<std::size_t>(P >> 2u) * (P >> 2u);
  const std::size_t tile_u32s = blocks_per_tile * 4u;
  const std::size_t tile_bytes = blocks_per_tile * 16u;
  const bool is_bc6h = (fmt == VtTileFormat::bc6h);

  ExtBucket *bucket = acquire_ext_bucket_locked(P, N, is_bc6h);
  if (bucket == nullptr) {
    return false;
  }
  const std::size_t host_count =
      static_cast<std::size_t>(bucket->bucket_tiles) * texels;

  if (!is_bc6h) {
    _host_in_u32.resize(host_count);
    if (N < bucket->bucket_tiles) {
      std::memset(_host_in_u32.data() + N * texels, 0,
                  (bucket->bucket_tiles - N) * texels * sizeof(luisa::uint));
    }
    auto pack_tile_u32 = [&](std::size_t t) noexcept {
      const auto &job = *group_jobs[t];
      luisa::uint *dst_u32 = _host_in_u32.data() + t * texels;
      if (job.unorm8_pixels != nullptr) {
        vt_pack_byte4_tile(
            job.unorm8_pixels, job.width, job.height, job.tile_x, job.tile_y,
            job.extension, P, job.logical_tile_size, job.border,
            std::span<luisa::uint>{dst_u32, texels});
      } else {
        std::fill_n(dst_u32, texels, 0u);
      }
    };
    if (N >= 8u) {
      psycles::parallel_for(N, pack_tile_u32);
    } else {
      for (std::uint32_t t = 0u; t < N; ++t) {
        pack_tile_u32(t);
      }
    }
  } else {
    _host_in_f4.resize(host_count);
    if (N < bucket->bucket_tiles) {
      std::memset(_host_in_f4.data() + N * texels, 0,
                  (bucket->bucket_tiles - N) * texels * sizeof(luisa::float4));
    }
    auto pack_tile_f4 = [&](std::size_t t) noexcept {
      const auto &job = *group_jobs[t];
      luisa::float4 *dst_f4 = _host_in_f4.data() + t * texels;
      vt_pack_float4_tile(
          job.float_pixels, job.width, job.height, job.tile_x, job.tile_y,
          job.extension, P, job.logical_tile_size, job.border,
          std::span<luisa::float4>{dst_f4, texels});
    };
    if (N >= 8u) {
      psycles::parallel_for(N, pack_tile_f4);
    } else {
      for (std::uint32_t t = 0u; t < N; ++t) {
        pack_tile_f4(t);
      }
    }
  }

  ensure_stream_locked();
  _host_out_u32.resize(static_cast<std::size_t>(N) * tile_u32s);
  const auto gpu_t0 = std::chrono::steady_clock::now();
  _compute_stream << bucket->src_image.copy_from(
      is_bc6h ? static_cast<const void *>(_host_in_f4.data())
              : static_cast<const void *>(_host_in_u32.data()));
  const auto res =
      is_bc6h
          ? _tex_compress_ext->compress_bc6h(
                _compute_stream, bucket->src_image.view(0u),
                bucket->dst_buffer.view())
          : _tex_compress_ext->compress_bc7(
                _compute_stream, bucket->src_image.view(0u),
                bucket->dst_buffer.view(), 1.0f);
  if (res != TexCompressExt::Result::Success) {
    _compute_stream << synchronize();
    return false;
  }
  _compute_stream
      << bucket->dst_buffer.view(0u, static_cast<std::size_t>(N) * tile_u32s)
             .copy_to(_host_out_u32.data())
      << synchronize();
  const double gpu_ms = elapsed_ms_since(gpu_t0);
  _gpu_kernel_ms += gpu_ms;
  _last_gpu_kernel_ms += gpu_ms;

  for (std::uint32_t t = 0u; t < N; ++t) {
    std::memcpy(group_jobs[t]->dst,
                _host_out_u32.data() + static_cast<std::size_t>(t) * tile_u32s,
                tile_bytes);
  }
  return true;
}

bool VtGpuBcCompressor::compress_group_dsl_locked(
    std::span<const VtGpuBcTileJob *const> group_jobs,
    VtTileFormat fmt,
    std::uint32_t P) {
  const std::uint32_t N = static_cast<std::uint32_t>(group_jobs.size());
  const std::size_t texels = static_cast<std::size_t>(P) * P;
  const std::uint32_t bx_count = P >> 2u;
  const std::uint32_t blocks_per_tile = bx_count * bx_count;
  const std::uint32_t total_blocks = N * blocks_per_tile;
  const std::uint32_t u32s_per_block =
      (fmt == VtTileFormat::bc1 || fmt == VtTileFormat::bc4) ? 2u : 4u;
  const std::size_t tile_u32s =
      static_cast<std::size_t>(blocks_per_tile) * u32s_per_block;
  const std::size_t tile_bytes = tile_u32s * sizeof(luisa::uint);
  const bool is_bc6h = (fmt == VtTileFormat::bc6h);

  ensure_stream_locked();
  ensure_dsl_buffers_locked(P, N, is_bc6h);
  ensure_shader_locked(fmt);
  _host_out_u32.resize(static_cast<std::size_t>(N) * tile_u32s);

  if (!is_bc6h) {
    _host_in_u32.resize(static_cast<std::size_t>(N) * texels);
    auto pack_tile_u32 = [&](std::size_t t) noexcept {
      const auto &job = *group_jobs[t];
      luisa::uint *dst_u32 = _host_in_u32.data() + t * texels;
      if (job.unorm8_pixels != nullptr) {
        vt_pack_byte4_tile(
            job.unorm8_pixels, job.width, job.height, job.tile_x, job.tile_y,
            job.extension, P, job.logical_tile_size, job.border,
            std::span<luisa::uint>{dst_u32, texels});
      } else {
        std::fill_n(dst_u32, texels, 0u);
      }
    };
    if (N >= 8u) {
      psycles::parallel_for(N, pack_tile_u32);
    } else {
      for (std::uint32_t t = 0u; t < N; ++t) {
        pack_tile_u32(t);
      }
    }
    const std::size_t in_u32_count = static_cast<std::size_t>(N) * texels;
    auto &shader = [&]() -> VtGpuBcBlockShader & {
      switch (fmt) {
        case VtTileFormat::bc1:
          return _bc1_shader;
        case VtTileFormat::bc3:
          return _bc3_shader;
        case VtTileFormat::bc4:
          return _bc4_shader;
        case VtTileFormat::bc5:
          return _bc5_shader;
        case VtTileFormat::bc7:
        default:
          return _bc7_shader;
      }
    }();
    const auto gpu_t0 = std::chrono::steady_clock::now();
    _compute_stream
        << _dsl_buffers.ldr_in_u32.view(0u, in_u32_count)
               .copy_from(_host_in_u32.data())
        << shader(_dsl_buffers.ldr_in_u32, _dsl_buffers.out_blocks_u32, P,
                  total_blocks, static_cast<luisa::uint>(_quality))
               .dispatch(total_blocks)
        << _dsl_buffers.out_blocks_u32
               .view(0u, static_cast<std::size_t>(N) * tile_u32s)
               .copy_to(_host_out_u32.data())
        << synchronize();
    const double gpu_ms = elapsed_ms_since(gpu_t0);
    _gpu_kernel_ms += gpu_ms;
    _last_gpu_kernel_ms += gpu_ms;
  } else {
    _host_in_f4.resize(static_cast<std::size_t>(N) * texels);
    auto pack_tile_f4 = [&](std::size_t t) noexcept {
      const auto &job = *group_jobs[t];
      luisa::float4 *dst_f4 = _host_in_f4.data() + t * texels;
      vt_pack_float4_tile(
          job.float_pixels, job.width, job.height, job.tile_x, job.tile_y,
          job.extension, P, job.logical_tile_size, job.border,
          std::span<luisa::float4>{dst_f4, texels});
    };
    if (N >= 8u) {
      psycles::parallel_for(N, pack_tile_f4);
    } else {
      for (std::uint32_t t = 0u; t < N; ++t) {
        pack_tile_f4(t);
      }
    }
    const std::size_t in_f4_count = static_cast<std::size_t>(N) * texels;
    const auto gpu_t0 = std::chrono::steady_clock::now();
    _compute_stream
        << _dsl_buffers.bc6h_in_f4.view(0u, in_f4_count)
               .copy_from(_host_in_f4.data())
        << _bc6h_shader(_dsl_buffers.bc6h_in_f4, _dsl_buffers.out_blocks_u32, P,
                        total_blocks, static_cast<luisa::uint>(_quality))
               .dispatch(total_blocks)
        << _dsl_buffers.out_blocks_u32
               .view(0u, static_cast<std::size_t>(N) * tile_u32s)
               .copy_to(_host_out_u32.data())
        << synchronize();
    const double gpu_ms = elapsed_ms_since(gpu_t0);
    _gpu_kernel_ms += gpu_ms;
    _last_gpu_kernel_ms += gpu_ms;
  }

  for (std::uint32_t t = 0u; t < N; ++t) {
    std::memcpy(group_jobs[t]->dst,
                _host_out_u32.data() + static_cast<std::size_t>(t) * tile_u32s,
                tile_bytes);
  }
  return true;
}

bool VtGpuBcCompressor::compress_group_locked(
    std::span<const VtGpuBcTileJob *const> group_jobs,
    VtTileFormat fmt,
    std::uint32_t P) {
  if (_backend == Backend::tex_compress_ext &&
      (fmt == VtTileFormat::bc7 || fmt == VtTileFormat::bc6h)) {
    if (compress_group_ext_locked(group_jobs, fmt, P)) {
      return true;
    }
    if (_mode == VtGpuBcMode::auto_select) {
      return compress_group_dsl_locked(group_jobs, fmt, P);
    }
    return false;
  }
  if (_backend != Backend::none) {
    return compress_group_dsl_locked(group_jobs, fmt, P);
  }
  return false;
}

bool VtGpuBcCompressor::compress_batch(
    std::span<const VtGpuBcTileJob> jobs) noexcept {
  if (!available() || jobs.size() < _min_batch_tiles) {
    return false;
  }
  for (const auto &j : jobs) {
    if (!supports_format(j.format) || j.dst == nullptr ||
        j.physical_tile_size < 4u || j.physical_tile_size > 128u ||
        (j.physical_tile_size % 4u) != 0u ||
        j.width == 0u || j.height == 0u ||
        j.dst_bytes < vt_tile_format_bytes(j.physical_tile_size, j.format)) {
      return false;
    }
    if (j.format == VtTileFormat::bc6h) {
      if (j.float_pixels == nullptr) {
        return false;
      }
    } else if (j.unorm8_pixels == nullptr) {
      return false;
    }
  }

  std::lock_guard lock{_mutex};
  _last_gpu_kernel_ms = 0.0;
  const auto t0 = std::chrono::steady_clock::now();

  constexpr std::array<VtTileFormat, 6u> kFormats{
      VtTileFormat::bc7, VtTileFormat::bc6h, VtTileFormat::bc1,
      VtTileFormat::bc3, VtTileFormat::bc4, VtTileFormat::bc5};
  std::array<std::vector<const VtGpuBcTileJob *>, 6u> by_fmt{};
  for (const auto &j : jobs) {
    for (std::size_t fi = 0u; fi < kFormats.size(); ++fi) {
      if (j.format == kFormats[fi]) {
        by_fmt[fi].push_back(&j);
        break;
      }
    }
  }

  auto dispatch_by_size = [&](std::span<const VtGpuBcTileJob *const> list,
                              VtTileFormat fmt) -> bool {
    if (list.empty()) {
      return true;
    }
    auto max_chunk_for = [&](std::uint32_t p_tile) noexcept -> std::size_t {
      if (_backend == Backend::tex_compress_ext &&
          (fmt == VtTileFormat::bc7 || fmt == VtTileFormat::bc6h)) {
        return std::min<std::size_t>(
            kMaxGpuBcBatchTiles,
            std::max<std::uint32_t>(1u, kMaxExtStripHeight / p_tile));
      }
      return kMaxGpuBcBatchTiles;
    };
    const std::uint32_t P0 = list.front()->physical_tile_size;
    bool uniform_p = true;
    for (const auto *j : list) {
      if (j->physical_tile_size != P0) {
        uniform_p = false;
        break;
      }
    }
    if (uniform_p) {
      const std::size_t chunk_cap = max_chunk_for(P0);
      for (std::size_t off = 0u; off < list.size(); off += chunk_cap) {
        const std::size_t n =
            std::min<std::size_t>(chunk_cap, list.size() - off);
        if (!compress_group_locked(list.subspan(off, n), fmt, P0)) {
          return false;
        }
      }
      return true;
    }
    std::vector<std::uint32_t> sizes;
    for (const auto *j : list) {
      if (std::find(sizes.begin(), sizes.end(), j->physical_tile_size) ==
          sizes.end()) {
        sizes.push_back(j->physical_tile_size);
      }
    }
    std::vector<const VtGpuBcTileJob *> sub;
    for (const std::uint32_t p : sizes) {
      sub.clear();
      for (const auto *j : list) {
        if (j->physical_tile_size == p) {
          sub.push_back(j);
        }
      }
      const std::size_t chunk_cap = max_chunk_for(p);
      for (std::size_t off = 0u; off < sub.size(); off += chunk_cap) {
        const std::size_t n =
            std::min<std::size_t>(chunk_cap, sub.size() - off);
        if (!compress_group_locked(
                std::span<const VtGpuBcTileJob *const>{sub}.subspan(off, n), fmt, p)) {
          return false;
        }
      }
    }
    return true;
  };

  for (std::size_t fi = 0u; fi < kFormats.size(); ++fi) {
    if (!dispatch_by_size(by_fmt[fi], kFormats[fi])) {
      return false;
    }
  }

  _tiles_encoded += jobs.size();
  _encode_ms += elapsed_ms_since(t0);
  return true;
}

} // namespace psycles::luisa_backend::detail
