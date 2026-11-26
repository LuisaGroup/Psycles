#include "virtual_texture_cache_mipmap.h"

#include "virtual_texture_cache.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] const std::array<float, 256u> &srgb_to_linear_lut() noexcept {
  static const auto lut = []() {
    std::array<float, 256u> table{};
    for (std::size_t i = 0u; i < 256u; ++i) {
      const float s = static_cast<float>(i) / 255.0f;
      table[i] = (s <= 0.04045f)
                     ? (s / 12.92f)
                     : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }
    return table;
  }();
  return lut;
}

[[nodiscard]] std::uint8_t linear_to_unorm8(float v) noexcept {
  const float c = std::clamp(v, 0.0f, 1.0f) * 255.0f;
  return static_cast<std::uint8_t>(std::clamp(std::lround(c), 0l, 255l));
}

// Piecewise float-bit LUT for c in [2^-9, 1.0) (IEEE-754 biased exponents
// 118..126 with top 10 mantissa bits = 9,216 entries). Avoids per-pixel std::pow.
[[nodiscard]] const std::array<std::uint8_t, 9216u> &
linear_to_srgb_lut() noexcept {
  static const auto lut = []() {
    std::array<std::uint8_t, 9216u> table{};
    for (std::uint32_t exp_off = 0u; exp_off < 9u; ++exp_off) {
      const std::uint32_t exp = 118u + exp_off;
      for (std::uint32_t mant = 0u; mant < 1024u; ++mant) {
        const std::uint32_t bits =
            (exp << 23u) | (mant << 13u) | (1u << 12u);
        float c = 0.0f;
        std::memcpy(&c, &bits, sizeof(c));
        const float s = (c <= 0.0031308f)
                            ? (c * 12.92f)
                            : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
        table[(exp_off << 10u) | mant] = linear_to_unorm8(s);
      }
    }
    return table;
  }();
  return lut;
}

[[nodiscard]] std::uint8_t linear_to_srgb_u8(float v) noexcept {
  if (!(v > 0.0f)) {
    return 0u;
  }
  if (v >= 1.0f) {
    return 255u;
  }
  if (v <= 0.0031308f) {
    return linear_to_unorm8(v * 12.92f);
  }
  std::uint32_t bits = 0u;
  std::memcpy(&bits, &v, sizeof(bits));
  const std::uint32_t exp = (bits >> 23u) & 0xffu;
  if (exp >= 118u && exp <= 126u) {
    const std::uint32_t idx = ((exp - 118u) << 10u) | ((bits >> 13u) & 0x3ffu);
    return linear_to_srgb_lut()[idx];
  }
  const float s = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
  return linear_to_unorm8(s);
}

// Downsamples a 1D row or column slice via area-preserving box filtering.
template<std::size_t N>
void downsample_line_float(const float *src,
                           std::size_t src_stride_floats,
                           std::uint32_t src_len,
                           float *dst,
                           std::size_t dst_stride_floats,
                           std::uint32_t dst_len) noexcept {
  if (dst_len == 0u || src_len == 0u) {
    return;
  }
  if (src_len == dst_len) {
    for (std::uint32_t i = 0u; i < dst_len; ++i) {
      const float *s = src + static_cast<std::size_t>(i) * src_stride_floats;
      float *d = dst + static_cast<std::size_t>(i) * dst_stride_floats;
      for (std::size_t c = 0u; c < N; ++c) {
        d[c] = s[c];
      }
    }
    return;
  }
  if ((src_len & 1u) == 0u && src_len == dst_len * 2u) {
    for (std::uint32_t i = 0u; i < dst_len; ++i) {
      const float *s0 =
          src + static_cast<std::size_t>(i * 2u) * src_stride_floats;
      const float *s1 = s0 + src_stride_floats;
      float *d = dst + static_cast<std::size_t>(i) * dst_stride_floats;
      for (std::size_t c = 0u; c < N; ++c) {
        d[c] = 0.5f * (s0[c] + s1[c]);
      }
    }
    return;
  }
  // Exact 3-tap box coverage weights for odd NPOT length (2n + 1 -> n).
  const float inv = 1.0f / static_cast<float>(src_len);
  for (std::uint32_t i = 0u; i < dst_len; ++i) {
    const float w0 = static_cast<float>(dst_len - i) * inv;
    const float w1 = static_cast<float>(dst_len) * inv;
    const float w2 = static_cast<float>(i + 1u) * inv;
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const float *s0 = src + base * src_stride_floats;
    const float *s1 = s0 + src_stride_floats;
    const float *s2 =
        src + std::min<std::size_t>(base + 2u, src_len - 1u) * src_stride_floats;
    float *d = dst + static_cast<std::size_t>(i) * dst_stride_floats;
    for (std::size_t c = 0u; c < N; ++c) {
      d[c] = w0 * s0[c] + w1 * s1[c] + w2 * s2[c];
    }
  }
}

template<std::size_t N>
void downsample_image_float(const float *src,
                            std::uint32_t src_w,
                            std::uint32_t src_h,
                            std::uint32_t dst_w,
                            std::uint32_t dst_h,
                            std::vector<float> &horiz_scratch,
                            std::vector<float> &dst) {
  horiz_scratch.resize(static_cast<std::size_t>(dst_w) * src_h * N);
  auto do_horiz_row = [&](std::size_t y) {
    downsample_line_float<N>(
        src + y * src_w * N, N, src_w,
        horiz_scratch.data() + y * dst_w * N, N, dst_w);
  };
  if (src_h >= 64u) {
    psycles::parallel_for(static_cast<std::size_t>(src_h), do_horiz_row);
  } else {
    for (std::uint32_t y = 0u; y < src_h; ++y) {
      do_horiz_row(y);
    }
  }

  dst.resize(static_cast<std::size_t>(dst_w) * dst_h * N);
  const std::size_t row_floats = static_cast<std::size_t>(dst_w) * N;
  if (src_h == dst_h) {
    std::memcpy(dst.data(), horiz_scratch.data(), dst.size() * sizeof(float));
    return;
  }
  if ((src_h & 1u) == 0u && src_h == dst_h * 2u) {
    auto do_vert_even_row = [&](std::size_t y) {
      const float *r0 = horiz_scratch.data() + (y * 2u) * row_floats;
      const float *r1 = r0 + row_floats;
      float *d = dst.data() + y * row_floats;
      for (std::size_t i = 0u; i < row_floats; ++i) {
        d[i] = 0.5f * (r0[i] + r1[i]);
      }
    };
    if (dst_h >= 64u) {
      psycles::parallel_for(static_cast<std::size_t>(dst_h), do_vert_even_row);
    } else {
      for (std::uint32_t y = 0u; y < dst_h; ++y) {
        do_vert_even_row(y);
      }
    }
    return;
  }
  const float inv = 1.0f / static_cast<float>(src_h);
  auto do_vert_odd_row = [&](std::size_t y) {
    const float w0 =
        static_cast<float>(dst_h - static_cast<std::uint32_t>(y)) * inv;
    const float w1 = static_cast<float>(dst_h) * inv;
    const float w2 = static_cast<float>(static_cast<std::uint32_t>(y) + 1u) * inv;
    const std::size_t base = y * 2u;
    const float *s0 = horiz_scratch.data() + base * row_floats;
    const float *s1 = s0 + row_floats;
    const float *s2 =
        horiz_scratch.data() +
        std::min<std::size_t>(base + 2u, src_h - 1u) * row_floats;
    float *d = dst.data() + y * row_floats;
    for (std::size_t i = 0u; i < row_floats; ++i) {
      d[i] = w0 * s0[i] + w1 * s1[i] + w2 * s2[i];
    }
  };
  if (dst_h >= 64u) {
    psycles::parallel_for(static_cast<std::size_t>(dst_h), do_vert_odd_row);
  } else {
    for (std::uint32_t y = 0u; y < dst_h; ++y) {
      do_vert_odd_row(y);
    }
  }
}

void downsample_image_channels(const float *src,
                               std::uint32_t src_w,
                               std::uint32_t src_h,
                               std::uint32_t dst_w,
                               std::uint32_t dst_h,
                               std::uint32_t channels,
                               std::vector<float> &horiz_scratch,
                               std::vector<float> &dst) {
  switch (channels) {
  case 1u:
    downsample_image_float<1u>(src, src_w, src_h, dst_w, dst_h, horiz_scratch,
                               dst);
    return;
  case 2u:
    downsample_image_float<2u>(src, src_w, src_h, dst_w, dst_h, horiz_scratch,
                               dst);
    return;
  default:
    downsample_image_float<4u>(src, src_w, src_h, dst_w, dst_h, horiz_scratch,
                               dst);
    return;
  }
}

void append_u64_array(std::string &out,
                      std::string_view key,
                      std::span<const std::uint64_t> values) {
  out.append(",\"");
  out.append(key);
  out.append("\":[");
  for (std::size_t i = 0u; i < values.size(); ++i) {
    if (i > 0u) {
      out.push_back(',');
    }
    char num[32];
    const int n = std::snprintf(
        num, sizeof(num), "%llu", static_cast<unsigned long long>(values[i]));
    if (n > 0) {
      out.append(num, static_cast<std::size_t>(n));
    }
  }
  out.push_back(']');
}

} // namespace

std::vector<VtUnorm8MipLevel> vt_build_unorm8_channel_tail_mips(
    std::span<const std::uint8_t> base_pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t channels,
    bool is_srgb) {
  std::vector<VtUnorm8MipLevel> out;
  if (channels != 1u && channels != 2u && channels != 4u) {
    return out;
  }
  const std::uint32_t mip_count = vt_compute_mip_count(width, height);
  if (mip_count <= 1u ||
      base_pixels.size() < static_cast<std::size_t>(width) * height * channels) {
    return out;
  }
  out.reserve(mip_count - 1u);

  const std::size_t base_pixels_count = static_cast<std::size_t>(width) * height;
  std::vector<float> cur_linear(base_pixels_count * channels);
  constexpr float inv255 = 1.0f / 255.0f;
  const auto &lut = srgb_to_linear_lut();
  const std::uint32_t color_channels = channels == 4u ? 3u : 1u;
  auto decode_base_row = [&](std::size_t y) {
    const std::size_t row_off = y * width * channels;
    for (std::uint32_t x = 0u; x < width; ++x) {
      const std::size_t off = row_off + static_cast<std::size_t>(x) * channels;
      for (std::uint32_t c = 0u; c < color_channels; ++c) {
        cur_linear[off + c] = is_srgb
                                  ? lut[base_pixels[off + c]]
                                  : static_cast<float>(base_pixels[off + c]) * inv255;
      }
      if (channels == 2u || channels == 4u) {
        cur_linear[off + channels - 1u] =
            static_cast<float>(base_pixels[off + channels - 1u]) * inv255;
      }
    }
  };
  if (height >= 64u) {
    psycles::parallel_for(static_cast<std::size_t>(height), decode_base_row);
  } else {
    for (std::uint32_t y = 0u; y < height; ++y) {
      decode_base_row(y);
    }
  }

  std::vector<float> horiz_scratch;
  std::vector<float> next_linear;
  std::uint32_t cur_w = width;
  std::uint32_t cur_h = height;
  for (std::uint32_t m = 1u; m < mip_count; ++m) {
    const std::uint32_t dst_w = vt_mip_extent(width, m);
    const std::uint32_t dst_h = vt_mip_extent(height, m);
    downsample_image_channels(cur_linear.data(), cur_w, cur_h, dst_w, dst_h,
                              channels, horiz_scratch, next_linear);
    VtUnorm8MipLevel level;
    level.width = dst_w;
    level.height = dst_h;
    const std::size_t n_pixels = static_cast<std::size_t>(dst_w) * dst_h;
    level.rgba.resize(n_pixels * channels);
    auto encode_mip_row = [&](std::size_t y) {
      const std::size_t row_off = y * dst_w * channels;
      for (std::uint32_t x = 0u; x < dst_w; ++x) {
        const std::size_t off = row_off + static_cast<std::size_t>(x) * channels;
        for (std::uint32_t c = 0u; c < color_channels; ++c) {
          level.rgba[off + c] = is_srgb ? linear_to_srgb_u8(next_linear[off + c])
                                        : linear_to_unorm8(next_linear[off + c]);
        }
        if (channels == 2u || channels == 4u) {
          level.rgba[off + channels - 1u] =
              linear_to_unorm8(next_linear[off + channels - 1u]);
        }
      }
    };
    if (dst_h >= 64u) {
      psycles::parallel_for(static_cast<std::size_t>(dst_h), encode_mip_row);
    } else {
      for (std::uint32_t y = 0u; y < dst_h; ++y) {
        encode_mip_row(y);
      }
    }
    out.push_back(std::move(level));
    cur_linear.swap(next_linear);
    cur_w = dst_w;
    cur_h = dst_h;
  }
  return out;
}

std::vector<VtFloat4MipLevel> vt_build_float_channel_tail_mips(
    std::span<const float> base_pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t channels) {
  std::vector<VtFloat4MipLevel> out;
  if (channels != 1u && channels != 2u && channels != 4u) {
    return out;
  }
  const std::uint32_t mip_count = vt_compute_mip_count(width, height);
  if (mip_count <= 1u ||
      base_pixels.size() < static_cast<std::size_t>(width) * height * channels) {
    return out;
  }
  out.reserve(mip_count - 1u);

  std::vector<float> horiz_scratch;
  std::vector<float> next_pixels;
  const float *cur_ptr = base_pixels.data();
  std::uint32_t cur_w = width;
  std::uint32_t cur_h = height;
  for (std::uint32_t m = 1u; m < mip_count; ++m) {
    const std::uint32_t dst_w = vt_mip_extent(width, m);
    const std::uint32_t dst_h = vt_mip_extent(height, m);
    downsample_image_channels(cur_ptr, cur_w, cur_h, dst_w, dst_h, channels,
                              horiz_scratch, next_pixels);
    VtFloat4MipLevel level;
    level.width = dst_w;
    level.height = dst_h;
    level.rgba = std::move(next_pixels);
    cur_ptr = level.rgba.data();
    out.push_back(std::move(level));
    cur_w = dst_w;
    cur_h = dst_h;
  }
  return out;
}

std::vector<VtUnorm8MipLevel> vt_build_unorm8_tail_mips(
    std::span<const std::uint8_t> base_rgba,
    std::uint32_t width,
    std::uint32_t height,
    bool is_srgb) {
  return vt_build_unorm8_channel_tail_mips(base_rgba, width, height, 4u, is_srgb);
}

std::vector<VtFloat4MipLevel> vt_build_float4_tail_mips(
    std::span<const float> base_rgba,
    std::uint32_t width,
    std::uint32_t height) {
  return vt_build_float_channel_tail_mips(base_rgba, width, height, 4u);
}

std::string VirtualTextureCache::stats_json() const {
  const char *mode_str = "off";
  if (_options.prepack_mode == LuisaVirtualTexturePrepackMode::memory) {
    mode_str = "memory";
  } else if (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk) {
    mode_str = "disk";
  }
  const auto cq_str =
      luisa_virtual_texture_copy_queue_mode_name(_stats.active_copy_queue_mode);
  const auto comp_str =
      luisa_virtual_texture_compression_mode_name(_stats.active_compression_mode);
  const auto mip_str = luisa_texture_mip_mode_name(_stats.active_mip_mode);
  char buf[8192];
  const int written = std::snprintf(
      buf, sizeof(buf),
      "{\"total_images\":%u,\"virtual_images\":%u,\"images_decoded\":%llu,"
      "\"byte4_virtual_pages\":%u,\"byte4_cache_capacity\":%u,"
      "\"byte4_peak_resident\":%u,\"float4_virtual_pages\":%u,"
      "\"float4_cache_capacity\":%u,\"float4_peak_resident\":%u,"
      "\"physical_tile_size\":%u,\"tile_border\":%u,\"logical_tile_size\":%u,"
      "\"compression_mode\":\"%.*s\",\"pool_summary\":\"%s\","
      "\"byte4_tile_bytes\":%u,"
      "\"float4_tile_bytes\":%u,\"gpu_upload_bytes\":%llu,"
      "\"gpu_budget_bytes\":%llu,\"gpu_allocated_atlas_bytes\":%llu,"
      "\"gpu_allocated_metadata_bytes\":%llu,\"fault_batches\":%llu,"
      "\"aux_fault_batches\":%llu,\"tiles_loaded\":%llu,"
      "\"tiles_evicted\":%llu,\"secondary_misses_coalesced\":%llu,"
      "\"host_decode_evictions\":%llu,"
      "\"prepack_mode\":\"%s\",\"prepack_hits\":%llu,"
      "\"prepack_misses\":%llu,\"prepack_evictions\":%llu,"
      "\"prepack_disk_reads\":%llu,\"prepack_disk_writes\":%llu,"
      "\"prepack_resident_bytes\":%llu,\"prepack_disk_ms\":%.3f,"
      "\"collect_sync_ms\":%.3f,\"decode_ms\":%.3f,\"pack_ms\":%.3f,"
      "\"upload_ms\":%.3f,\"async_worker\":%s,\"async_batches\":%llu,"
      "\"async_poll_ready_hits\":%llu,\"async_wait_calls\":%llu,"
      "\"async_wait_ms\":%.3f,\"async_worker_busy_ms\":%.3f,"
      "\"async_aux_drains\":%llu,\"async_batch_slots\":%u,"
      "\"async_peak_in_flight\":%u,\"async_out_of_order_publishes\":%llu,"
      "\"async_deferred_admissions\":%llu,\"host_allocated_vt_bytes\":%llu,"
      "\"device_allocated_vt_bytes\":%llu,\"peak_host_decoded_bytes\":%llu,"
      "\"peak_prepack_resident_bytes\":%llu,"
      "\"copy_queue_mode\":\"%.*s\",\"copy_queue_batches\":%llu,"
      "\"copy_queue_h2d_bytes\":%llu,\"copy_queue_atlas_bytes\":%llu,"
      "\"copy_queue_submit_ms\":%.3f,"
      "\"preflight_enabled\":%s,\"preflight_overlapped\":%s,"
      "\"preflight_stride\":%u,\"preflight_bounces\":%u,"
      "\"preflight_tiles_discovered\":%u,\"preflight_tiles_loaded\":%u,"
      "\"preflight_upload_bytes\":%llu,\"preflight_compile_ms\":%.3f,"
      "\"preflight_trace_ms\":%.3f,\"preflight_service_ms\":%.3f,"
      "\"preflight_total_ms\":%.3f,\"pso_compile_ms\":%.3f,"
      "\"mip_mode\":\"%.*s\",\"total_mip_pages\":%llu,"
      "\"unique_pages_loaded\":%llu,\"tiles_reloaded\":%llu,"
      "\"bytes_loaded_total\":%llu,\"preflight_bytes_loaded\":%llu,"
      "\"fault_frames_total\":%llu,"
      "\"gpu_bc_backend\":\"%s\",\"gpu_bc_tiles_encoded\":%llu,"
      "\"gpu_bc_ms\":%.3f",
      _stats.total_images, _stats.virtual_images,
      static_cast<unsigned long long>(_stats.images_decoded),
      _stats.byte4_virtual_pages, _stats.byte4_cache_capacity,
      _stats.byte4_peak_resident, _stats.float4_virtual_pages,
      _stats.float4_cache_capacity, _stats.float4_peak_resident,
      _stats.physical_tile_size, _stats.tile_border, _stats.logical_tile_size,
      static_cast<int>(comp_str.size()), comp_str.data(),
      _stats.pool_summary.c_str(),
      _stats.byte4_tile_bytes, _stats.float4_tile_bytes,
      static_cast<unsigned long long>(_stats.gpu_upload_bytes),
      static_cast<unsigned long long>(_stats.gpu_budget_bytes),
      static_cast<unsigned long long>(_stats.gpu_allocated_atlas_bytes),
      static_cast<unsigned long long>(_stats.gpu_allocated_metadata_bytes),
      static_cast<unsigned long long>(_stats.fault_batches),
      static_cast<unsigned long long>(_stats.aux_fault_batches),
      static_cast<unsigned long long>(_stats.tiles_loaded),
      static_cast<unsigned long long>(_stats.tiles_evicted),
      static_cast<unsigned long long>(_stats.secondary_misses_coalesced),
      static_cast<unsigned long long>(_stats.host_decode_evictions),
      mode_str, static_cast<unsigned long long>(_stats.prepack_hits),
      static_cast<unsigned long long>(_stats.prepack_misses),
      static_cast<unsigned long long>(_stats.prepack_evictions),
      static_cast<unsigned long long>(_stats.prepack_disk_reads),
      static_cast<unsigned long long>(_stats.prepack_disk_writes),
      static_cast<unsigned long long>(_stats.prepack_resident_bytes),
      _stats.prepack_disk_ms, _stats.collect_sync_ms, _stats.decode_ms,
      _stats.pack_ms, _stats.upload_ms,
      _options.enable_async_worker ? "true" : "false",
      static_cast<unsigned long long>(_stats.async_batches),
      static_cast<unsigned long long>(_stats.async_poll_ready_hits),
      static_cast<unsigned long long>(_stats.async_wait_calls),
      _stats.async_wait_ms, _stats.async_worker_busy_ms,
      static_cast<unsigned long long>(_stats.async_aux_drains),
      _stats.async_batch_slots, _stats.async_peak_in_flight,
      static_cast<unsigned long long>(_stats.async_out_of_order_publishes),
      static_cast<unsigned long long>(_stats.async_deferred_admissions),
      static_cast<unsigned long long>(_stats.host_allocated_vt_bytes),
      static_cast<unsigned long long>(_stats.device_allocated_vt_bytes),
      static_cast<unsigned long long>(_stats.peak_host_decoded_bytes),
      static_cast<unsigned long long>(_stats.peak_prepack_resident_bytes),
      static_cast<int>(cq_str.size()), cq_str.data(),
      static_cast<unsigned long long>(_stats.copy_queue_batches),
      static_cast<unsigned long long>(_stats.copy_queue_h2d_bytes),
      static_cast<unsigned long long>(_stats.copy_queue_atlas_bytes),
      _stats.copy_queue_submit_ms,
      _stats.preflight_enabled ? "true" : "false",
      _stats.preflight_overlapped ? "true" : "false",
      _stats.preflight_stride, _stats.preflight_bounces,
      _stats.preflight_tiles_discovered, _stats.preflight_tiles_loaded,
      static_cast<unsigned long long>(_stats.preflight_upload_bytes),
      _stats.preflight_compile_ms, _stats.preflight_trace_ms,
      _stats.preflight_service_ms, _stats.preflight_total_ms,
      _stats.pso_compile_ms,
      static_cast<int>(mip_str.size()), mip_str.data(),
      static_cast<unsigned long long>(_stats.total_mip_pages),
      static_cast<unsigned long long>(_stats.unique_pages_loaded),
      static_cast<unsigned long long>(_stats.tiles_reloaded),
      static_cast<unsigned long long>(_stats.bytes_loaded_total),
      static_cast<unsigned long long>(_stats.preflight_bytes_loaded),
      static_cast<unsigned long long>(_stats.fault_frames_total),
      _gpu_bc ? _gpu_bc->backend_name() : _stats.gpu_bc_backend,
      static_cast<unsigned long long>(
          _gpu_bc ? _gpu_bc->tiles_encoded() : _stats.gpu_bc_tiles_encoded),
      _gpu_bc ? _gpu_bc->encode_ms() : _stats.gpu_bc_ms);
  std::string out;
  if (written > 0) {
    out.assign(buf, static_cast<std::size_t>(
                        std::min<int>(written, sizeof(buf) - 1)));
  } else {
    out = "{";
  }
  append_u64_array(out, "unique_pages_per_mip", _stats.unique_pages_per_mip);
  append_u64_array(out, "tiles_loaded_per_mip", _stats.tiles_loaded_per_mip);
  out.push_back('}');
  return out;
}

} // namespace psycles::luisa_backend::detail
