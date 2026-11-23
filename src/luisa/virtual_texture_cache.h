#pragma once

#include <psycles/compiler/cycles_svm_ir.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include "cycles_svm_scene_image.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/coro/schedulers/wavefront_extension.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace psycles::luisa_backend::detail {

struct LuisaSceneData;

inline constexpr std::uint32_t vt_invalid_page = ~std::uint32_t{0u};
inline constexpr std::uint32_t vt_border_size = 2u;
inline constexpr std::uint32_t vt_page_resident_bit = 0x80000000u;
inline constexpr std::uint32_t vt_page_slot_mask = 0x7fffffffu;
inline constexpr std::string_view vt_extension_schema =
    "org.psycles.texture_cache";
inline constexpr std::string_view vt_extension_binding = "page_id";

inline constexpr std::uint32_t vt_desc_interp_mask = 0x3u;
inline constexpr std::uint32_t vt_desc_ext_shift = 2u;
inline constexpr std::uint32_t vt_desc_ext_mask = 0x3u << vt_desc_ext_shift;
inline constexpr std::uint32_t vt_desc_flag_float4 = 1u << 4u;
inline constexpr std::uint32_t vt_desc_flag_virtual = 1u << 5u;
inline constexpr std::uint32_t vt_desc_flag_load_failed = 1u << 6u;

struct VtImageDescGpu {
  luisa::uint width{};
  luisa::uint height{};
  luisa::uint tiles_x{};
  luisa::uint tiles_y{};
  luisa::uint page_table_offset{};
  luisa::uint flags{};
  luisa::uint direct_texture_slot{};
  luisa::uint pad{};
};

static_assert(sizeof(VtImageDescGpu) == 32u);
static_assert(alignof(VtImageDescGpu) == alignof(luisa::uint));

struct VtSampleResult {
  luisa::float4 color{};
  luisa::uint miss_page{vt_invalid_page};
};

struct VtPoolConfigGpu {
  luisa::uint num_atlases{};
  luisa::uint atlas_tiles_x_log2{};
  luisa::uint atlas_tiles_x_mask{};
  luisa::uint slots_per_atlas_log2{};
  luisa::uint slots_per_atlas_mask{};
  luisa::uint bindless_base{};
  luisa::uint epoch_offset{};
  luisa::uint pad0{};
  float full_inv_w{};
  float full_inv_h{};
  float last_inv_h{};
  luisa::uint pad1{};
};

static_assert(sizeof(VtPoolConfigGpu) == 48u);
static_assert(alignof(VtPoolConfigGpu) == alignof(luisa::uint));

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::VtImageDescGpu,
             width, height, tiles_x, tiles_y, page_table_offset, flags,
             direct_texture_slot, pad){};
LUISA_STRUCT(psycles::luisa_backend::detail::VtSampleResult,
             color, miss_page){};
LUISA_STRUCT(psycles::luisa_backend::detail::VtPoolConfigGpu,
             num_atlases, atlas_tiles_x_log2, atlas_tiles_x_mask,
             slots_per_atlas_log2, slots_per_atlas_mask, bindless_base,
             epoch_offset, pad0, full_inv_w, full_inv_h, last_inv_h, pad1){};

namespace psycles::luisa_backend::detail {

struct VtProbedSceneImage {
  contract::ImageId id{};
  std::string name;
  std::vector<std::uint8_t> encoded_data;
  contract::ImageAlphaType alpha_type{contract::ImageAlphaType::straight};
  contract::ImageColorSpace color_space{contract::ImageColorSpace::data};
  bool load_failed{false};
  bool force_direct{false};
  io::ProbedImageInfo probed{};
};

struct VirtualTextureCacheStats {
  std::uint32_t total_images{};
  std::uint32_t virtual_images{};
  std::uint64_t images_decoded{};
  std::uint32_t byte4_virtual_pages{};
  std::uint32_t byte4_cache_capacity{};
  std::uint32_t byte4_peak_resident{};
  std::uint32_t float4_virtual_pages{};
  std::uint32_t float4_cache_capacity{};
  std::uint32_t float4_peak_resident{};
  std::uint32_t physical_tile_size{};
  std::uint32_t tile_border{vt_border_size};
  std::uint32_t logical_tile_size{};
  std::uint64_t gpu_budget_bytes{};
  std::uint64_t gpu_allocated_atlas_bytes{};
  std::uint64_t gpu_allocated_metadata_bytes{};
  std::uint64_t fault_batches{};
  std::uint64_t aux_fault_batches{};
  std::uint64_t tiles_loaded{};
  std::uint64_t tiles_evicted{};
  std::uint64_t secondary_misses_coalesced{};
  std::uint64_t host_decode_evictions{};
  double collect_sync_ms{};
  double decode_ms{};
  double pack_ms{};
  double upload_ms{};
};

[[nodiscard]] std::set<contract::ImageId> collect_shader_routine_image_ids(
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    std::span<const std::uint32_t> shader_indices,
    compiler::cycles_svm::ShaderType routine_type);

// RAII guard for pre-render standalone passes (displacement bake and
// background importance bake) that need direct access to a small subset of
// scene images without keeping the full texture set resident at render time.
class TemporaryDirectTextureScope {

private:
  std::shared_ptr<LuisaSceneData> _scene;
  std::vector<luisa::compute::Image<float>> _temp_images;
  std::vector<std::uint32_t> _rebound_slots;
  bool _active{false};

public:
  TemporaryDirectTextureScope() noexcept = default;
  TemporaryDirectTextureScope(
      std::shared_ptr<LuisaSceneData> scene,
      luisa::compute::Stream &stream,
      const std::set<contract::ImageId> &image_ids);
  TemporaryDirectTextureScope(const TemporaryDirectTextureScope &) = delete;
  TemporaryDirectTextureScope &
  operator=(const TemporaryDirectTextureScope &) = delete;
  TemporaryDirectTextureScope(TemporaryDirectTextureScope &&other) noexcept;
  TemporaryDirectTextureScope &
  operator=(TemporaryDirectTextureScope &&other) noexcept;
  ~TemporaryDirectTextureScope() noexcept;

  void release() noexcept;
};

class VirtualTextureCache final
    : public std::enable_shared_from_this<VirtualTextureCache> {

public:
  struct SourceImageEntry {
    contract::ImageId id{};
    std::string name;
    std::vector<std::uint8_t> encoded_data;
    contract::ImageAlphaType alpha_type{contract::ImageAlphaType::straight};
    contract::ImageColorSpace color_space{contract::ImageColorSpace::data};
    std::uint32_t width{};
    std::uint32_t height{};
    io::DecodedImageStorage storage{io::DecodedImageStorage::unorm8};
    bool load_failed{false};
    bool force_direct{false};
    bool decoded{false};
    std::uint64_t last_use_seq{};
    std::uint64_t decoded_bytes{};
    std::vector<std::uint8_t> unorm8_pixels;
    std::vector<float> float_pixels;
  };

  struct VirtualPageMeta {
    std::uint32_t source_index{};
    std::uint32_t tile_x{};
    std::uint32_t tile_y{};
    std::uint32_t extension{};
    bool is_float4{false};
  };

  struct PoolConfig {
    bool is_float4{false};
    bool eviction_possible{false};
    std::uint32_t page_begin{};
    std::uint32_t page_end{};
    std::uint32_t capacity{};
    std::uint32_t staging_capacity{};
    std::uint32_t epoch_offset{};
    std::uint32_t atlas_tiles_x{};
    std::uint32_t atlas_tiles_x_log2{};
    std::uint32_t atlas_tiles_x_mask{};
    std::uint32_t full_atlas_tiles_y{};
    std::uint32_t last_atlas_tiles_y{};
    std::uint32_t slots_per_atlas{};
    std::uint32_t slots_per_atlas_log2{};
    std::uint32_t slots_per_atlas_mask{};
    std::uint32_t bindless_base{};
    std::vector<luisa::compute::Image<float>> atlases;
    std::vector<std::uint32_t> atlas_widths;
    std::vector<std::uint32_t> atlas_heights;
    float full_inv_w{};
    float full_inv_h{};
    float last_inv_h{};
  };

  struct CollectShaderKey {
    bool soa{};
    bool single_field{};
    std::uint32_t field_index{};
    std::uint64_t layout_hash{};
    const luisa::compute::CoroFrameDesc *frame_desc{nullptr};
    luisa::uint boundary_index{};

    [[nodiscard]] bool operator==(const CollectShaderKey &other) const noexcept;
  };

  using CollectShader = luisa::compute::Shader1D<
      luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint>;

  using SampleCallable = luisa::compute::Callable<VtSampleResult(
      luisa::compute::Buffer<VtPoolConfigGpu>,
      luisa::compute::Buffer<VtImageDescGpu>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::BindlessArray,
      luisa::uint,
      luisa::uint,
      luisa::float2)>;

  struct CollectShaderEntry {
    CollectShaderKey key;
    CollectShader shader;
  };

  struct DecodedSourcePayload {
    bool ok{false};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t decoded_bytes{};
    std::vector<std::uint8_t> unorm8_pixels;
    std::vector<float> float_pixels;
  };

private:
  struct PoolSlotMeta {
    std::uint64_t load_wf{};
    std::uint64_t last_hit_wf{};
    std::uint64_t load_seq{};
    std::uint32_t hotness{};
    std::uint32_t initial_demand{};
    bool loaded_by_aux{false};
  };

  enum class MissSource {
    wavefront_demand,
    aux_demand,
    bitset_only,
  };

  luisa::compute::Device _device;
  LuisaVirtualTextureCacheOptions _options{};
  std::uint32_t _physical_tile_size{64u};
  std::uint32_t _tile_border{vt_border_size};
  std::uint32_t _logical_tile_size{60u};
  std::uint32_t _warp_size{32u};

  std::vector<SourceImageEntry> _sources;
  std::vector<std::vector<std::uint32_t>> _source_partners;
  std::unordered_map<std::uint64_t, std::uint32_t> _id_to_source;
  std::vector<VtImageDescGpu> _descs;
  std::vector<VirtualPageMeta> _pages;

  PoolConfig _byte4_pool{};
  PoolConfig _float4_pool{};

  luisa::compute::BindlessArray _atlas_heap;
  luisa::compute::Buffer<VtPoolConfigGpu> _pool_configs;
  luisa::compute::Buffer<VtImageDescGpu> _image_descs_buffer;
  luisa::compute::Buffer<luisa::uint> _page_table;
  luisa::compute::Buffer<luisa::uint> _miss_bitset;
  luisa::compute::Buffer<luisa::uint> _slot_epoch;
  luisa::compute::Buffer<luisa::uint> _current_epoch;
  luisa::compute::Buffer<luisa::uint> _demand_hist;
  luisa::compute::Buffer<luisa::uint> _aux_demand_hist;

  static constexpr std::size_t kUploadRingSize = 2u;

  luisa::compute::Event _upload_event;
  luisa::compute::PinnedMemoryExt *_pinned_ext{nullptr};
  bool _use_pinned_upload{false};
  bool _use_pinned_readback{false};

  std::array<luisa::compute::Buffer<luisa::uint>, kUploadRingSize>
      _staging_byte4;
  std::array<luisa::compute::Buffer<luisa::float4>, kUploadRingSize>
      _staging_float4;
  std::array<luisa::compute::Buffer<luisa::uint4>, kUploadRingSize> _upload_meta;
  std::array<luisa::compute::Buffer<luisa::uint2>, kUploadRingSize>
      _page_table_updates;
  std::array<luisa::compute::Buffer<luisa::uint2>, kUploadRingSize>
      _slot_epoch_updates;

  std::array<luisa::uint *, kUploadRingSize> _pinned_staging_byte4_ptr{
      nullptr, nullptr};
  std::array<luisa::float4 *, kUploadRingSize> _pinned_staging_float4_ptr{
      nullptr, nullptr};
  std::array<luisa::uint4 *, kUploadRingSize> _pinned_upload_meta_ptr{
      nullptr, nullptr};
  std::array<luisa::uint2 *, kUploadRingSize> _pinned_page_table_updates_ptr{
      nullptr, nullptr};
  std::array<luisa::uint2 *, kUploadRingSize> _pinned_slot_epoch_updates_ptr{
      nullptr, nullptr};

  std::array<std::uint64_t, kUploadRingSize> _chunk_slot_fence{0ull, 0ull};
  std::array<std::uint64_t, kUploadRingSize> _chunk_slot_stream{0ull, 0ull};
  std::size_t _chunk_ring_cursor{0u};
  std::array<std::uint64_t, kUploadRingSize> _meta_slot_fence{0ull, 0ull};
  std::array<std::uint64_t, kUploadRingSize> _meta_slot_stream{0ull, 0ull};
  std::size_t _meta_ring_cursor{0u};

  luisa::compute::Buffer<luisa::uint> _readback_demand_hist;
  luisa::compute::Buffer<luisa::uint> _readback_aux_demand_hist;
  luisa::compute::Buffer<luisa::uint> _readback_miss_bitset;
  luisa::compute::Buffer<luisa::uint> _readback_slot_epoch;
  luisa::uint *_pinned_demand_hist_ptr{nullptr};
  luisa::uint *_pinned_aux_demand_hist_ptr{nullptr};
  luisa::uint *_pinned_miss_bitset_ptr{nullptr};
  luisa::uint *_pinned_slot_epoch_ptr{nullptr};

  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>, luisa::uint>
      _clear_u32_shader;
  luisa::compute::Shader3D<luisa::compute::Image<float>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint4>,
                           luisa::uint, luisa::uint>
      _upload_byte4_shader;
  luisa::compute::Shader3D<luisa::compute::Image<float>,
                           luisa::compute::Buffer<luisa::float4>,
                           luisa::compute::Buffer<luisa::uint4>,
                           luisa::uint, luisa::uint>
      _upload_float4_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint2>,
                           luisa::uint>
      _apply_page_table_updates_shader;

  std::optional<SampleCallable> _sample_linear_repeat;
  std::optional<SampleCallable> _sample_linear_any_ext;
  std::optional<SampleCallable> _sample_general;

  std::deque<CollectShaderEntry> _collect_shader_cache;

  // Persistent host state shared across all fault handlers and aux servicing.
  luisa::vector<luisa::uint> _host_page_table;
  luisa::vector<luisa::uint> _host_byte4_slots;
  luisa::vector<luisa::uint> _host_float4_slots;
  luisa::vector<luisa::uint> _host_demand_hist;
  luisa::vector<luisa::uint> _host_aux_demand_hist;
  luisa::vector<luisa::uint> _host_miss_bitset;
  luisa::vector<luisa::uint> _host_slot_epoch;
  luisa::vector<luisa::uint> _host_current_epoch{0u, 0u};
  luisa::vector<luisa::uint> _host_staging_byte4;
  luisa::vector<luisa::float4> _host_staging_float4;
  luisa::vector<luisa::uint4> _host_upload_meta;
  luisa::vector<luisa::uint2> _host_page_table_updates;
  luisa::vector<luisa::uint2> _host_slot_epoch_updates;
  std::vector<PoolSlotMeta> _host_byte4_slot_meta;
  std::vector<PoolSlotMeta> _host_float4_slot_meta;
  std::array<std::uint64_t, 2u> _pool_wavefront_seq{0ull, 0ull};
  std::array<std::uint64_t, 2u> _pool_load_seq{0ull, 0ull};
  std::array<std::uint64_t, 2u> _pool_aux_streak_start_seq{
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max()};
  std::uint32_t _byte4_used_slots{0u};
  std::uint32_t _float4_used_slots{0u};
  std::uint64_t _host_decoded_bytes{0u};
  std::uint64_t _decode_use_seq{0u};
  bool _eager_fallback_uploaded{false};
  VirtualTextureCacheStats _stats{};

  std::mutex _bg_mutex;
  std::condition_variable _bg_cv;
  std::vector<std::thread> _bg_workers;
  std::deque<std::uint32_t> _bg_pending_queue;
  std::vector<std::uint8_t> _bg_source_queued;
  std::vector<std::optional<DecodedSourcePayload>> _bg_ready_payloads;
  bool _bg_stop{false};

  [[nodiscard]] const PoolConfig &pool_for(bool is_float4) const noexcept {
    return is_float4 ? _float4_pool : _byte4_pool;
  }

  void bg_decode_worker_loop() noexcept;
  void ensure_bg_decode_workers_started() noexcept;
  void stop_bg_decode_workers() noexcept;
  void enqueue_bg_decode(std::uint32_t source_index) noexcept;

  [[nodiscard]] std::size_t
  acquire_chunk_slot(std::uint64_t current_stream_handle) noexcept;
  [[nodiscard]] std::size_t
  acquire_meta_slot(std::uint64_t current_stream_handle) noexcept;
  void wait_all_upload_fences() noexcept;

  void service_misses_impl(
      luisa::compute::Stream &stream,
      MissSource source,
      luisa::compute::CommandList *wavefront_cmd = nullptr) noexcept;
  void decode_source_on_demand(std::uint32_t source_index) noexcept;
  void trim_host_decoded_cache(std::uint32_t protect_source_index) noexcept;
  void apply_decoded_source_payload(
      SourceImageEntry &src,
      DecodedSourcePayload &&payload) noexcept;
  void decode_sources_batch(
      std::span<const std::uint32_t> source_indices) noexcept;

  [[nodiscard]] static SampleCallable record_linear_repeat_sampler(
      std::uint32_t physical_tile_size,
      std::uint32_t tile_border,
      std::uint32_t logical_tile_size,
      luisa::string_view name) noexcept;
  [[nodiscard]] static SampleCallable record_linear_any_ext_sampler(
      std::uint32_t physical_tile_size,
      std::uint32_t tile_border,
      std::uint32_t logical_tile_size,
      luisa::string_view name) noexcept;
  [[nodiscard]] static SampleCallable record_general_sampler(
      std::uint32_t physical_tile_size,
      std::uint32_t tile_border,
      std::uint32_t logical_tile_size,
      luisa::string_view name) noexcept;

public:
  VirtualTextureCache(
      luisa::compute::Device device,
      luisa::compute::Stream &stream,
      const LuisaVirtualTextureCacheOptions &options,
      std::span<const VtProbedSceneImage> probed_images,
      std::span<const compiler::cycles_svm::ImageBinding> svm_image_handles,
      std::span<const CyclesSvmImageBindingGpu> svm_image_bindings,
      const compiler::cycles_svm::CompiledShaderTable *compilation = nullptr);
  ~VirtualTextureCache() noexcept;

  [[nodiscard]] static std::shared_ptr<VirtualTextureCache> create(
      luisa::compute::Device &device,
      luisa::compute::Stream &stream,
      const LuisaVirtualTextureCacheOptions &options,
      std::span<const VtProbedSceneImage> probed_images,
      std::span<const compiler::cycles_svm::ImageBinding> svm_image_handles,
      std::span<const CyclesSvmImageBindingGpu> svm_image_bindings,
      const compiler::cycles_svm::CompiledShaderTable *compilation = nullptr);

  [[nodiscard]] bool has_virtual_images() const noexcept {
    return !_pages.empty();
  }
  [[nodiscard]] bool eager_fallback_active() const noexcept {
    return _eager_fallback_uploaded;
  }
  [[nodiscard]] bool is_virtual_binding(std::int32_t binding_id) const noexcept;
  [[nodiscard]] std::size_t binding_count() const noexcept {
    return _descs.size();
  }
  [[nodiscard]] std::uint32_t physical_tile_size() const noexcept {
    return _physical_tile_size;
  }
  [[nodiscard]] std::uint32_t logical_tile_size() const noexcept {
    return _logical_tile_size;
  }
  [[nodiscard]] const VirtualTextureCacheStats &stats() const noexcept {
    return _stats;
  }
  [[nodiscard]] std::string stats_json() const;

  [[nodiscard]] luisa::compute::Bool
  is_resident(luisa::compute::Expr<luisa::uint> page_id) const noexcept;

  [[nodiscard]] luisa::compute::Float4 sample_2d_const(
      std::int32_t binding_id,
      luisa::compute::Expr<luisa::float2> uv,
      luisa::compute::UInt *miss_page_out,
      bool record_bitset = true) const noexcept;

  [[nodiscard]] luisa::compute::Float4 sample_2d_dynamic(
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float2> uv,
      const luisa::compute::BindlessArray &direct_heap,
      luisa::compute::UInt *miss_page_out,
      bool record_bitset = true) const noexcept;

  void record_aux_miss(luisa::compute::Expr<luisa::uint> page_id) const noexcept;

  [[nodiscard]] luisa::move_only_function<
      luisa::unique_ptr<luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>(
          luisa::compute::coro::WavefrontCoroExtensionPrepareContext &,
          const luisa::compute::coro::WavefrontCoroExtensionStage &)>
  make_wavefront_extension_handler() noexcept;

  void service_aux_misses(luisa::compute::Stream &stream) noexcept;
  void service_bitset_misses(luisa::compute::Stream &stream) noexcept;

  void dispatch_wavefront_collect(
      const luisa::compute::coro::WavefrontCoroExtensionDispatchContext &ctx,
      const CollectShader &collect_shader) noexcept;

  void ensure_eager_direct_textures_uploaded(LuisaSceneData &scene);

  void populate_temporary_direct_textures(
      LuisaSceneData &scene,
      luisa::compute::Stream &stream,
      const std::set<contract::ImageId> &image_ids,
      std::vector<luisa::compute::Image<float>> &temp_images,
      std::vector<std::uint32_t> &rebound_slots);
};

} // namespace psycles::luisa_backend::detail
