#pragma once

#include "virtual_texture_cache_formats.h"

#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute {
class TexCompressExt;
} // namespace luisa::compute

namespace psycles::luisa_backend::detail {

enum class VtGpuBcMode : std::uint8_t {
  auto_select = 0u,
  force_ext = 1u,
  force_dsl = 2u,
  off = 3u,
};

enum class VtGpuBcQuality : std::uint8_t {
  fast = 0u,
  balanced = 1u,
  high = 2u,
};

[[nodiscard]] constexpr const char *vt_gpu_bc_quality_name(
    VtGpuBcQuality quality) noexcept {
  switch (quality) {
    case VtGpuBcQuality::fast:
      return "fast";
    case VtGpuBcQuality::high:
      return "high";
    case VtGpuBcQuality::balanced:
    default:
      return "balanced";
  }
}

struct VtGpuBcTileJob {
  const std::uint8_t *unorm8_pixels{nullptr};
  const float *float_pixels{nullptr};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t tile_x{};
  std::uint32_t tile_y{};
  std::uint32_t extension{};
  std::uint32_t physical_tile_size{};
  std::uint32_t logical_tile_size{};
  std::uint32_t border{};
  VtTileFormat format{VtTileFormat::bc7};
  std::uint8_t *dst{nullptr};
  std::size_t dst_bytes{};
};

using VtGpuBcBlockShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::uint>,
    luisa::compute::Buffer<luisa::uint>,
    luisa::uint,
    luisa::uint,
    luisa::uint>;

using VtGpuBc6hBlockShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::uint>,
    luisa::uint,
    luisa::uint,
    luisa::uint>;

[[nodiscard]] VtGpuBcBlockShader compile_vt_gpu_bc7_shader(
    luisa::compute::Device &device);
[[nodiscard]] VtGpuBc6hBlockShader compile_vt_gpu_bc6h_shader(
    luisa::compute::Device &device);

// GPU-accelerated BC1 / BC3 / BC4 / BC5 / BC6H / BC7 tile compressor used on
// cold virtual texture streaming faults and disk prepack generation. Defaults
// to pure Luisa DSL across all backends for all 6 BC formats, with optional
// PSYCLES_VT_GPU_BC=ext override to select LuisaCompute's TexCompressExt for
// BC7 / BC6H where available. Quality is controlled via fast|balanced|high
// (PSYCLES_VT_GPU_BC_QUALITY / compression_quality).
class VtGpuBcCompressor final {
public:
  enum class Backend : std::uint8_t {
    none = 0u,
    tex_compress_ext = 1u,
    dsl = 2u,
  };

  explicit VtGpuBcCompressor(luisa::compute::Device &device) noexcept;
  VtGpuBcCompressor(
      luisa::compute::Device &device,
      VtGpuBcQuality quality_override) noexcept;
  VtGpuBcCompressor(
      luisa::compute::Device &device,
      VtGpuBcMode mode_override,
      VtGpuBcQuality quality_override = VtGpuBcQuality::balanced) noexcept;

  [[nodiscard]] bool available() const noexcept {
    return _backend != Backend::none;
  }
  [[nodiscard]] Backend backend() const noexcept { return _backend; }
  [[nodiscard]] const char *backend_name() const noexcept;
  [[nodiscard]] VtGpuBcMode mode() const noexcept { return _mode; }
  [[nodiscard]] VtGpuBcQuality quality() const noexcept { return _quality; }
  [[nodiscard]] const char *quality_name() const noexcept {
    return vt_gpu_bc_quality_name(_quality);
  }
  void set_quality(VtGpuBcQuality quality) noexcept { _quality = quality; }
  [[nodiscard]] std::uint32_t min_batch_tiles() const noexcept {
    return _min_batch_tiles;
  }
  [[nodiscard]] static bool supports_format(VtTileFormat fmt) noexcept {
    switch (fmt) {
      case VtTileFormat::bc1:
      case VtTileFormat::bc3:
      case VtTileFormat::bc4:
      case VtTileFormat::bc5:
      case VtTileFormat::bc6h:
      case VtTileFormat::bc7:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] bool compress_batch(
      std::span<const VtGpuBcTileJob> jobs) noexcept;
  void warmup(std::span<const VtTileFormat> formats) noexcept;
  void warmup(bool need_bc7, bool need_bc6h) noexcept;

  [[nodiscard]] std::uint64_t tiles_encoded() const noexcept;
  [[nodiscard]] double encode_ms() const noexcept;
  [[nodiscard]] double gpu_kernel_ms() const noexcept;
  [[nodiscard]] double last_gpu_kernel_ms() const noexcept;
  void reset_stats() noexcept;

private:
  struct ExtBucket {
    std::uint32_t tile_size{};
    std::uint32_t bucket_tiles{};
    bool is_bc6h{false};
    luisa::compute::Image<float> src_image;
    luisa::compute::Buffer<luisa::uint> dst_buffer;
  };

  struct DslBuffers {
    std::uint32_t tile_size{};
    std::uint32_t capacity_tiles{};
    luisa::compute::Buffer<luisa::uint> ldr_in_u32;
    luisa::compute::Buffer<luisa::float4> bc6h_in_f4;
    luisa::compute::Buffer<luisa::uint> out_blocks_u32;
  };

  void init_backend() noexcept;
  void ensure_stream_locked();
  void ensure_shader_locked(VtTileFormat fmt);
  void ensure_bc1_shader_locked();
  void ensure_bc3_shader_locked();
  void ensure_bc4_shader_locked();
  void ensure_bc5_shader_locked();
  void ensure_bc6h_shader_locked();
  void ensure_bc7_shader_locked();
  [[nodiscard]] ExtBucket *acquire_ext_bucket_locked(
      std::uint32_t tile_size,
      std::uint32_t tile_count,
      bool is_bc6h);
  void ensure_dsl_buffers_locked(
      std::uint32_t tile_size,
      std::uint32_t tile_count,
      bool need_bc6h);

  [[nodiscard]] bool compress_group_locked(
      std::span<const VtGpuBcTileJob *const> group_jobs,
      VtTileFormat fmt,
      std::uint32_t P);
  [[nodiscard]] bool compress_group_ext_locked(
      std::span<const VtGpuBcTileJob *const> group_jobs,
      VtTileFormat fmt,
      std::uint32_t P);
  [[nodiscard]] bool compress_group_dsl_locked(
      std::span<const VtGpuBcTileJob *const> group_jobs,
      VtTileFormat fmt,
      std::uint32_t P);

  luisa::compute::Device &_device;
  VtGpuBcMode _mode{VtGpuBcMode::auto_select};
  VtGpuBcQuality _quality{VtGpuBcQuality::balanced};
  Backend _backend{Backend::none};
  std::uint32_t _min_batch_tiles{1u};
  luisa::compute::TexCompressExt *_tex_compress_ext{nullptr};

  mutable std::mutex _mutex;
  luisa::compute::Stream _compute_stream;
  VtGpuBcBlockShader _bc1_shader;
  VtGpuBcBlockShader _bc3_shader;
  VtGpuBcBlockShader _bc4_shader;
  VtGpuBcBlockShader _bc5_shader;
  VtGpuBcBlockShader _bc7_shader;
  VtGpuBc6hBlockShader _bc6h_shader;
  bool _has_bc1_shader{false};
  bool _has_bc3_shader{false};
  bool _has_bc4_shader{false};
  bool _has_bc5_shader{false};
  bool _has_bc6h_shader{false};
  bool _has_bc7_shader{false};

  std::vector<ExtBucket> _ext_buckets;
  DslBuffers _dsl_buffers;
  std::vector<luisa::float4> _host_in_f4;
  std::vector<luisa::uint> _host_in_u32;
  std::vector<luisa::uint> _host_out_u32;

  std::uint64_t _tiles_encoded{};
  double _encode_ms{};
  double _gpu_kernel_ms{};
  double _last_gpu_kernel_ms{};
};

} // namespace psycles::luisa_backend::detail
