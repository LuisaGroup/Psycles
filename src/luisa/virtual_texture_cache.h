#pragma once

#include <psycles/compiler/cycles_svm_ir.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include "cycles_svm_scene_image.h"
#include "virtual_texture_cache_formats.h"
#include "virtual_texture_cache_gpu_bc.h"
#include "virtual_texture_cache_mipmap.h"
#include "virtual_texture_cache_prepack.h"
#include "virtual_texture_cache_profile.h"

#include <array>
#include <atomic>
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

#include <luisa/backends/ext/batch_copy_cmd.h>
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

struct VtMipLevelGpu {
  luisa::uint width{};
  luisa::uint height{};
  luisa::uint tiles_x{};
  luisa::uint tiles_y{};
  luisa::uint page_table_offset{};
  luisa::uint page_count{};
  luisa::uint pad0{};
  luisa::uint pad1{};
};

static_assert(sizeof(VtMipLevelGpu) == 32u);

struct VtImageDescGpu {
  luisa::uint width{};
  luisa::uint height{};
  luisa::uint tiles_x{};
  luisa::uint tiles_y{};
  luisa::uint page_table_offset{};
  luisa::uint flags{};
  luisa::uint direct_texture_slot{};
  luisa::uint pool_index{};
  luisa::uint mip_table_offset{};
  luisa::uint mip_count{};
  luisa::uint pad0{};
  luisa::uint pad1{};
};

static_assert(sizeof(VtImageDescGpu) == 48u);
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
  luisa::uint channel_swizzle{};
  float full_inv_w{};
  float full_inv_h{};
  float last_inv_h{};
  luisa::uint pad1{};
};

static_assert(sizeof(VtPoolConfigGpu) == 48u);
static_assert(alignof(VtPoolConfigGpu) == alignof(luisa::uint));

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::VtMipLevelGpu,
             width, height, tiles_x, tiles_y, page_table_offset, page_count,
             pad0, pad1){};
LUISA_STRUCT(psycles::luisa_backend::detail::VtImageDescGpu,
             width, height, tiles_x, tiles_y, page_table_offset, flags,
             direct_texture_slot, pool_index, mip_table_offset, mip_count,
             pad0, pad1){};
LUISA_STRUCT(psycles::luisa_backend::detail::VtSampleResult,
             color, miss_page){};
LUISA_STRUCT(psycles::luisa_backend::detail::VtPoolConfigGpu,
             num_atlases, atlas_tiles_x_log2, atlas_tiles_x_mask,
             slots_per_atlas_log2, slots_per_atlas_mask, bindless_base,
             epoch_offset, channel_swizzle, full_inv_w, full_inv_h, last_inv_h,
             pad1){};

namespace psycles::luisa_backend::detail {

struct VtProbedSceneImage {
  contract::ImageId id{};
  std::string name;
  std::vector<std::uint8_t> encoded_data;
  contract::ImageAlphaType alpha_type{contract::ImageAlphaType::straight};
  contract::ImageColorSpace color_space{contract::ImageColorSpace::data};
  bool load_failed{false};
  bool force_direct{false};
  std::uint32_t texture_limit{0u};
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
  std::uint64_t prepack_hits{};
  std::uint64_t prepack_misses{};
  std::uint64_t prepack_evictions{};
  std::uint64_t prepack_disk_reads{};
  std::uint64_t prepack_disk_writes{};
  std::uint64_t prepack_resident_bytes{};
  double prepack_disk_ms{};
  double collect_sync_ms{};
  double decode_ms{};
  double pack_ms{};
  double upload_ms{};
  std::uint64_t async_batches{};
  std::uint64_t async_poll_ready_hits{};
  std::uint64_t async_wait_calls{};
  double async_wait_ms{};
  double async_worker_busy_ms{};
  std::uint64_t async_aux_drains{};
  LuisaVirtualTextureCopyQueueMode active_copy_queue_mode{
      LuisaVirtualTextureCopyQueueMode::off};
  LuisaVirtualTextureCompressionMode active_compression_mode{
      LuisaVirtualTextureCompressionMode::off};
  std::uint32_t byte4_tile_bytes{};
  std::uint32_t float4_tile_bytes{};
  std::string pool_summary;
  std::uint64_t gpu_upload_bytes{};
  std::uint64_t copy_queue_batches{};
  std::uint64_t copy_queue_h2d_bytes{};
  std::uint64_t copy_queue_atlas_bytes{};
  double copy_queue_submit_ms{};
  std::uint32_t async_batch_slots{1u};
  std::uint32_t async_peak_in_flight{0u};
  std::uint64_t async_out_of_order_publishes{0ull};
  std::uint64_t async_deferred_admissions{0ull};
  std::uint64_t host_allocated_vt_bytes{};
  std::uint64_t device_allocated_vt_bytes{};
  std::uint64_t peak_host_decoded_bytes{};
  std::uint64_t peak_prepack_resident_bytes{};
  bool preflight_enabled{false};
  bool preflight_overlapped{false};
  std::uint32_t preflight_stride{0u};
  std::uint32_t preflight_bounces{0u};
  std::uint32_t preflight_tiles_discovered{0u};
  std::uint32_t preflight_tiles_loaded{0u};
  std::uint64_t preflight_upload_bytes{0ull};
  double preflight_compile_ms{0.0};
  double preflight_trace_ms{0.0};
  double preflight_service_ms{0.0};
  double preflight_total_ms{0.0};
  double pso_compile_ms{0.0};
  LuisaTextureMipMode active_mip_mode{LuisaTextureMipMode::mip0};
  std::uint64_t total_mip_pages{};
  std::uint64_t unique_pages_loaded{};
  std::uint64_t tiles_reloaded{};
  std::uint64_t bytes_loaded_total{};
  std::uint64_t preflight_bytes_loaded{};
  std::uint64_t fault_frames_total{};
  std::vector<std::uint64_t> unique_pages_per_mip;
  std::vector<std::uint64_t> tiles_loaded_per_mip;
  const char *gpu_bc_backend{"off"};
  std::uint64_t gpu_bc_tiles_encoded{};
  double gpu_bc_ms{};
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
    std::uint32_t texture_limit{0u};
    io::DecodedImageStorage storage{io::DecodedImageStorage::unorm8};
    io::ProbedImageInfo probed_info{};
    bool load_failed{false};
    bool force_direct{false};
    bool decoded{false};
    std::uint64_t last_use_seq{};
    std::uint64_t decoded_bytes{};
    std::vector<std::uint8_t> unorm8_pixels;
    std::vector<float> float_pixels;
    std::vector<VtUnorm8MipLevel> unorm8_mips;
    std::vector<VtFloat4MipLevel> float4_mips;
    VtSourceIdentityKey identity_key{};
  };

  struct VirtualPageMeta {
    std::uint32_t source_index{};
    std::uint32_t tile_x{};
    std::uint32_t tile_y{};
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    std::uint32_t extension{};
    std::uint32_t mip_level{};
    std::uint32_t block_id{};
    VtTileFormat format{VtTileFormat::rgba8};
    std::uint32_t pool_index{};
    bool is_high_precision{false};
  };

  struct PoolConfig {
    VtTileFormat format{VtTileFormat::rgba8};
    VtChannelSwizzle swizzle{VtChannelSwizzle::rgba};
    bool is_high_precision{false};
    bool eviction_possible{false};
    bool use_batch_copy_upload{false};
    luisa::compute::PixelStorage atlas_storage{
        luisa::compute::PixelStorage::BYTE4};
    std::uint32_t tile_bytes{0u};
    std::uint32_t tile_staging_u32s{0u};
    std::uint32_t page_begin{};
    std::uint32_t page_end{};
    std::uint32_t virtual_pages{};
    std::uint32_t capacity{};
    std::uint32_t staging_capacity{};
    std::size_t chunks_per_batch_slot{2u};
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

  using FilterReadyShader = luisa::compute::Shader1D<
      luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>, luisa::compute::Buffer<luisa::uint>,
      luisa::uint, luisa::uint>;

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

  using PreflightDemandCallable = luisa::compute::Callable<void(
      luisa::compute::Buffer<VtImageDescGpu>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::uint,
      luisa::float2,
      luisa::uint)>;

  using SampleMipCallable = luisa::compute::Callable<VtSampleResult(
      luisa::compute::Buffer<VtPoolConfigGpu>,
      luisa::compute::Buffer<VtImageDescGpu>,
      luisa::compute::Buffer<VtMipLevelGpu>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::compute::BindlessArray,
      luisa::uint,
      luisa::uint,
      luisa::float2,
      luisa::float2,
      luisa::float2)>;

  using PreflightMipDemandCallable = luisa::compute::Callable<void(
      luisa::compute::Buffer<VtImageDescGpu>,
      luisa::compute::Buffer<VtMipLevelGpu>,
      luisa::compute::Buffer<luisa::uint>,
      luisa::uint,
      luisa::float2,
      luisa::float2,
      luisa::float2,
      luisa::uint)>;

  struct CollectShaderEntry {
    CollectShaderKey key;
    CollectShader shader;
    FilterReadyShader filter_shader;
  };

  struct DecodedSourcePayload {
    bool ok{false};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t decoded_bytes{};
    std::vector<std::uint8_t> unorm8_pixels;
    std::vector<float> float_pixels;
    std::vector<VtUnorm8MipLevel> unorm8_mips;
    std::vector<VtFloat4MipLevel> float4_mips;
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

  struct PoolPinContext {
    std::span<const std::uint32_t> slot_inflight_owner{};
    std::span<const std::uint64_t> slot_pin_consumer_seq{};
    std::span<const std::uint64_t> slot_pin_aux_seq{};
    std::span<const std::uint8_t> slot_pin_by_aux{};
    std::span<const std::uint64_t> slot_publish_epoch{};
    std::uint64_t readback_publish_epoch{0ull};
    std::uint32_t readback_gpu_epoch{0u};
    bool has_readback_epoch{false};
    std::uint64_t retired_consumer_seq{0ull};
    std::uint64_t retired_aux_seq{0ull};
    bool is_oldest{true};
    bool force_oldest_progress{false};
  };

  enum class MissSource {
    wavefront_demand,
    aux_demand,
    bitset_only,
  };

  struct ServiceProfileContext {
    std::uint64_t collect_enter_ns{0ull};
    std::uint32_t queued_fault_frames{0u};
    luisa::compute::coro::WavefrontCoroSchedulerSnapshot snapshot{};
  };

  struct MissCandidate {
    std::uint32_t page{};
    std::uint32_t primary_demand{};
  };

  struct PoolAdmissionPlan {
    std::vector<MissCandidate> misses;
    std::vector<std::uint32_t> victim_slots;
    std::uint32_t new_slot_epoch{0u};
    std::uint64_t wf_seq{0ull};
    std::uint64_t load_seq{0ull};
    bool has_work{false};
  };

  struct ChunkStagingTarget {
    std::uint32_t pool_index{0u};
    std::size_t chunk_slot{0u};
    luisa::uint *pack_base{nullptr};
    luisa::uint4 *um_dst{nullptr};
  };

  struct PackedChunkMeta {
    std::uint32_t pool_index{0u};
    std::size_t chunk_slot{0u};
    std::uint32_t chunk_count{0u};
    std::vector<std::uint32_t> atlas_counts;
    std::vector<std::uint32_t> atlas_offsets;
    std::vector<luisa::compute::BufferToTextureCopyRegion> atlas_regions;
    luisa::vector<luisa::uint> fallback_staging;
    luisa::vector<luisa::uint4> fallback_um;
  };

  struct VictimSlot {
    std::uint32_t slot{};
    std::uint8_t tier{};
    std::uint32_t hotness{};
    std::uint32_t initial_demand{};
    std::uint64_t last_hit_wf{};
    std::uint64_t load_seq{};
  };

  struct RankedMiss {
    std::uint32_t page{};
    std::uint32_t primary_demand{};
    std::uint32_t source_index{};
    std::uint32_t bx{};
    std::uint32_t by{};
    std::uint64_t block_avg_demand{};
  };

  struct AssignedTile {
    std::uint32_t page{};
    std::uint32_t atlas_idx{};
    std::uint32_t dst_ox{};
    std::uint32_t dst_oy{};
    std::uint32_t staging_idx{};
  };

  struct PlanningScratch {
    std::vector<std::vector<MissCandidate>> pool_misses;
    std::vector<std::vector<MissCandidate>> pool_sec_misses;
    std::vector<PoolAdmissionPlan> plans;
    std::vector<std::uint32_t> src_hit_slots;
    std::vector<std::uint8_t> src_partner_hit;
    std::vector<std::uint8_t> aux_streak_protected;
    std::vector<std::uint32_t> streak_slots;
    std::vector<VictimSlot> unpinned_candidates;
    std::array<std::vector<VictimSlot>, 5u> tier_buckets;
    std::vector<std::uint32_t> new_hotness;
    std::vector<std::uint64_t> new_last_hit_wf;
    std::vector<std::uint32_t> hot_slots;
    std::vector<std::uint64_t> src_demand_sum;
    std::vector<std::uint32_t> src_miss_cnt;
    std::vector<std::uint64_t> src_avg_demand;
    std::vector<std::uint64_t> block_sum;
    std::vector<std::uint32_t> block_cnt;
    std::vector<std::uint32_t> touched_blocks;
    std::vector<RankedMiss> ranked;
    std::vector<std::uint8_t> src_admitted_now;
    std::vector<MissCandidate> sec_partners;
    std::vector<MissCandidate> sec_other;
    std::vector<AssignedTile> assigned;
    std::vector<std::uint32_t> chunk_page_ids;
    std::vector<VtPrepackedPageCache::ResolvedTileSlot> prepack_slots;
    std::vector<std::uint32_t> prepack_hit_indices;
    std::vector<std::uint8_t> slot_disk_ok;
    std::vector<std::size_t> disk_read_indices;
    std::vector<std::uint32_t> atlas_write_cursor;
  };

  enum class AsyncSlotLifecycle : std::uint8_t {
    Free = 0u,
    GpuReadbackInFlight = 1u,
    QueuedForWorker = 2u,
    DeferredNoVictim = 3u,
    WorkerActive = 4u,
    Ready = 5u,
    PublishedAwaitingAdvance = 6u,
  };

  [[nodiscard]] static constexpr bool
  is_unpublished_lifecycle(AsyncSlotLifecycle st) noexcept {
    return st == AsyncSlotLifecycle::GpuReadbackInFlight ||
           st == AsyncSlotLifecycle::QueuedForWorker ||
           st == AsyncSlotLifecycle::DeferredNoVictim ||
           st == AsyncSlotLifecycle::WorkerActive ||
           st == AsyncSlotLifecycle::Ready;
  }

  struct PreparedAsyncBatch {
    bool has_work{false};
    bool is_aux{false};
    bool any_miss_bit{false};
    bool any_epoch_advanced{false};
    bool any_epoch_wrapped{false};
    bool has_cold_decode{false};
    std::uint32_t slot_index{0u};
    std::uint64_t generation{0ull};
    std::uint64_t submit_seq{0ull};
    std::vector<std::uint64_t> dep_slot_gens;
    std::size_t meta_slot{0u};
    std::size_t pt_upd_count{0u};
    std::size_t ep_upd_count{0u};
    std::uint32_t b4_loaded{0u};
    std::uint32_t f4_loaded{0u};
    std::uint32_t tiles_loaded{0u};
    std::uint32_t tiles_evicted{0u};
    std::uint64_t fault_frames{0ull};
    std::uint32_t b4_peak_resident{0u};
    std::uint32_t f4_peak_resident{0u};
    std::vector<std::uint32_t> pool_loaded;
    std::vector<std::uint32_t> pool_resident;
    std::vector<PackedChunkMeta> chunks;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pinned_pool_slots;
    std::vector<std::uint32_t> pinned_evict_pages;
    std::vector<std::uint32_t> pinned_load_pages;
    luisa::vector<luisa::uint> current_epoch_upload;
    luisa::vector<luisa::uint> slot_epoch_full_upload;
    luisa::vector<luisa::uint2> fallback_pt_updates;
    luisa::vector<luisa::uint2> fallback_ep_updates;
    std::uint64_t callback_ns{0ull};
    double rb_flush_ms{0.0};
    double adm_ms{0.0};
    double decode_ms{0.0};
    double pack_ms{0.0};
    double slot_wait_ms{0.0};
    double copy_submit_ms{0.0};
    double worker_busy_ms{0.0};
    std::uint64_t h2d_fence{0ull};
    std::uint64_t h2d_bytes{0ull};
    std::vector<VtProfileWorkerEvent> worker_events;
  };

  struct AsyncSlotContext {
    AsyncSlotLifecycle state{AsyncSlotLifecycle::Free};
    luisa::compute::coro::WavefrontCoroAsyncBatchTicket ticket{};
    std::uint64_t submit_seq{0ull};
    std::uint64_t last_publish_fence{0ull};
    std::uint64_t deferred_at_retry_epoch{0ull};
    luisa::compute::Buffer<luisa::uint> demand_hist;
    luisa::compute::Buffer<luisa::uint> miss_bitset;
    luisa::compute::Buffer<luisa::uint> slot_epoch;
    luisa::uint *pinned_demand_hist_ptr{nullptr};
    luisa::uint *pinned_miss_bitset_ptr{nullptr};
    luisa::uint *pinned_slot_epoch_ptr{nullptr};
    luisa::vector<luisa::uint> fallback_demand_hist;
    luisa::vector<luisa::uint> fallback_miss_bitset;
    luisa::vector<luisa::uint> fallback_slot_epoch;
    std::uint64_t callback_ns{0ull};
    std::uint64_t stream_handle{0ull};
    std::uint64_t enter_ns{0ull};
    std::uint64_t submit_end_ns{0ull};
    std::uint64_t submit_seq_before_s1{0ull};
    std::uint64_t submit_seq_after_s1{0ull};
    std::uint32_t queued_fault_frames{0u};
    std::uint32_t in_flight_at_submit{0u};
    double collect_record_ms{0.0};
    double wait_ms{0.0};
    luisa::compute::coro::WavefrontCoroSchedulerSnapshot snapshot{};
    luisa::vector<luisa::uint> readback_gpu_epoch;
    std::uint64_t readback_publish_epoch{0ull};
    PreparedAsyncBatch ready_batch{};
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
  std::vector<VtPrepackedPageKey> _page_prepack_keys;
  std::unique_ptr<VtPrepackedPageCache> _prepack_cache;
  std::unique_ptr<VtPrepackedDiskStore> _prepack_disk;
  std::unique_ptr<VtGpuBcCompressor> _gpu_bc;

  struct PoolRuntime {
    static constexpr std::uint32_t kInvalidAsyncSlot =
        std::numeric_limits<std::uint32_t>::max();
    PoolConfig config{};
    std::vector<luisa::compute::Buffer<luisa::uint>> staging;
    std::vector<luisa::compute::Buffer<luisa::uint>> device_staging;
    std::vector<luisa::compute::Buffer<luisa::uint4>> upload_meta;
    std::vector<luisa::uint *> pinned_staging_ptr;
    std::vector<luisa::uint4 *> pinned_upload_meta_ptr;
    std::vector<std::uint64_t> slot_fence;
    std::vector<std::uint64_t> slot_stream;
    std::vector<std::uint64_t> slot_async_fence;
    std::size_t ring_cursor{0u};
    luisa::vector<luisa::uint> host_slots;
    luisa::vector<luisa::uint> host_staging;
    std::vector<PoolSlotMeta> host_slot_meta;
    std::vector<std::uint32_t> slot_inflight_owner;
    std::vector<std::uint64_t> slot_pin_consumer_seq;
    std::vector<std::uint64_t> slot_pin_aux_seq;
    std::vector<std::uint8_t> slot_pin_by_aux;
    std::vector<std::uint64_t> slot_publish_epoch;
    std::uint32_t last_readback_gpu_epoch{0u};
    std::uint32_t used_slots{0u};
    std::uint64_t wavefront_seq{0ull};
    std::uint64_t load_seq{0ull};
    std::uint64_t aux_streak_start_seq{
        std::numeric_limits<std::uint64_t>::max()};
  };
  std::vector<PoolRuntime> _pools;
  std::vector<std::uint32_t> _page_evict_owner_mask;
  std::vector<std::uint32_t> _page_load_inflight_slot;
  luisa::vector<luisa::uint> _published_gpu_epoch;

  luisa::compute::BindlessArray _atlas_heap;
  luisa::compute::Buffer<VtPoolConfigGpu> _pool_configs;
  luisa::compute::Buffer<VtImageDescGpu> _image_descs_buffer;
  luisa::compute::Buffer<VtMipLevelGpu> _mip_levels_buffer;
  std::vector<VtMipLevelGpu> _mip_levels;
  std::vector<std::uint8_t> _page_ever_loaded;
  luisa::compute::Buffer<luisa::uint> _page_table;
  luisa::compute::Buffer<luisa::uint> _miss_bitset;
  luisa::compute::Buffer<luisa::uint> _slot_epoch;
  luisa::compute::Buffer<luisa::uint> _current_epoch;
  luisa::compute::Buffer<luisa::uint> _demand_hist;
  luisa::compute::Buffer<luisa::uint> _aux_demand_hist;

  static constexpr std::size_t kSyncUploadRingSize = 2u;
  static constexpr std::size_t kMetaRingSize = 2u;

  std::uint32_t _async_batch_slots{1u};
  std::size_t _meta_ring_size{kMetaRingSize};

  luisa::compute::Event _upload_event;
  luisa::compute::PinnedMemoryExt *_pinned_ext{nullptr};
  bool _use_pinned_upload{false};
  bool _use_pinned_readback{false};
  LuisaVirtualTextureCopyQueueMode _copy_queue_mode{
      LuisaVirtualTextureCopyQueueMode::off};
  bool _filtered_advance_enabled{true};
  LuisaVirtualTextureCompressionMode _compression_mode{
      LuisaVirtualTextureCompressionMode::off};
  bool _use_copy_stream{false};
  bool _use_host_cached_staging{false};
  bool _use_batch_texture_copy{false};
  bool _supports_stream_signal_event{false};
  luisa::compute::Stream _copy_stream;
  luisa::compute::Event _copy_h2d_event;
  luisa::compute::Event _copy_acquire_event;
  luisa::compute::Event _copy_done_event;
  mutable std::mutex _copy_stream_mtx;

  std::vector<luisa::compute::Buffer<luisa::uint2>> _page_table_updates;
  std::vector<luisa::compute::Buffer<luisa::uint2>> _slot_epoch_updates;

  std::vector<luisa::uint2 *> _pinned_page_table_updates_ptr;
  std::vector<luisa::uint2 *> _pinned_slot_epoch_updates_ptr;

  std::vector<std::uint64_t> _meta_slot_fence;
  std::vector<std::uint64_t> _meta_slot_stream;
  std::size_t _meta_ring_cursor{0u};

  std::atomic<std::uint64_t> _async_publish_seq{0ull};
  std::atomic<std::uint64_t> _async_publish_completed{0ull};
  std::vector<std::uint64_t> _meta_slot_async_fence;

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
                           luisa::compute::Buffer<luisa::uint>,
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
  std::optional<PreflightDemandCallable> _preflight_demand_callable;
  std::optional<SampleMipCallable> _sample_mip_linear_repeat;
  std::optional<SampleMipCallable> _sample_mip_general;
  std::optional<PreflightMipDemandCallable> _preflight_mip_demand_callable;

  std::deque<CollectShaderEntry> _collect_shader_cache;

  // Persistent host state shared across all fault handlers and aux servicing.
  luisa::vector<luisa::uint> _host_page_table;
  luisa::vector<luisa::uint> _host_demand_hist;
  luisa::vector<luisa::uint> _host_aux_demand_hist;
  luisa::vector<luisa::uint> _host_miss_bitset;
  luisa::vector<luisa::uint> _host_slot_epoch;
  luisa::vector<luisa::uint> _host_current_epoch;
  luisa::vector<luisa::uint4> _host_upload_meta;
  luisa::vector<luisa::uint2> _host_page_table_updates;
  luisa::vector<luisa::uint2> _host_slot_epoch_updates;
  std::uint64_t _host_decoded_bytes{0u};
  std::uint64_t _decode_use_seq{0u};
  bool _eager_fallback_uploaded{false};
  VirtualTextureCacheStats _stats{};
  std::unique_ptr<VtProfileRecorder> _profiler;
  mutable bool _timeline_labels_registered{false};

  std::mutex _bg_mutex;
  std::condition_variable _bg_cv;
  std::vector<std::thread> _bg_workers;
  std::deque<std::uint32_t> _bg_pending_queue;
  std::vector<std::uint8_t> _bg_source_queued;
  std::vector<std::optional<DecodedSourcePayload>> _bg_ready_payloads;
  bool _bg_stop{false};

  std::vector<AsyncSlotContext> _async_slots;
  std::mutex _vt_worker_mtx;
  std::condition_variable _vt_worker_cv;
  std::condition_variable _vt_ready_cv;
  std::thread _vt_worker_thread;
  std::deque<std::size_t> _vt_worker_queue;
  bool _vt_worker_stop{false};
  std::atomic<std::uint32_t> _entry_drain_pressure{0u};
  std::atomic<std::uint32_t> _async_backpressure_cooldown{0u};
  std::uint64_t _latest_consumer_seq{0ull};
  std::uint64_t _retired_consumer_seq{0ull};
  std::uint64_t _latest_aux_seq{0ull};
  std::uint64_t _retired_aux_seq{0ull};
  std::uint64_t _last_snapshot_consumer_seq{0ull};
  std::uint64_t _last_snapshot_aux_seq{0ull};
  bool _has_snapshot_seq{false};
  std::uint64_t _publish_epoch{0ull};
  std::uint64_t _defer_retry_epoch{0ull};
  std::uint64_t _async_submit_seq{0ull};
  std::atomic<std::uint64_t> _host_decode_fallback_generation{0ull};
  bool _force_oldest_progress{false};
  std::atomic<double> _last_async_wait_ms{0.0};
  std::uint32_t _total_tile_blocks{0u};
  PlanningScratch _sync_planning_scratch;

  void bg_decode_worker_loop() noexcept;
  void ensure_bg_decode_workers_started() noexcept;
  void stop_bg_decode_workers() noexcept;
  void enqueue_bg_decode(std::uint32_t source_index) noexcept;

  void ensure_async_worker_started() noexcept;
  void stop_async_worker() noexcept;
  void vt_async_worker_loop() noexcept;
  void wait_async_fence(std::uint64_t fence) const noexcept;
  [[nodiscard]] bool batch_has_unsatisfied_dep(
      const PreparedAsyncBatch &batch) const noexcept;
  bool update_scheduler_progress_locked(
      const luisa::compute::coro::WavefrontCoroSchedulerSnapshot &snapshot,
      bool waiting_for_progress) noexcept;
  void drain_async_if_in_flight(luisa::compute::Stream &stream) noexcept;
  void publish_prepared_batch(
      std::size_t slot_idx,
      luisa::compute::Stream &stream,
      luisa::compute::CommandList &cmd,
      bool mark_awaiting_advance) noexcept;
  void emit_chunk_atlas_upload(
      luisa::compute::CommandList &target_cmd,
      const PackedChunkMeta &chunk,
      std::uint64_t *atlas_bytes_out = nullptr) noexcept;
  void append_staged_chunk_h2d(
      luisa::compute::CommandList &target_cmd,
      std::uint32_t pool_index,
      std::size_t chunk_slot,
      std::size_t chunk_count,
      std::uint64_t *h2d_bytes_out = nullptr) noexcept;
  [[nodiscard]] std::uint64_t submit_staged_batch_h2d(
      luisa::compute::CommandList &h2d_cmd) noexcept;
  [[nodiscard]] std::uint64_t submit_staged_chunk_h2d(
      std::uint32_t pool_index,
      std::size_t chunk_slot,
      std::size_t chunk_count,
      bool signal_event,
      std::uint64_t *h2d_bytes_out = nullptr) noexcept;
  [[nodiscard]] std::uint64_t signal_copy_h2d_event() noexcept;
  void submit_direct_copy_chunks(
      luisa::compute::Stream &stream,
      std::span<const PackedChunkMeta> chunks,
      std::uint64_t *atlas_bytes_out = nullptr) noexcept;

  void scan_miss_candidates(
      MissSource source,
      bool any_miss_bit,
      std::uint32_t bitset_words,
      const luisa::uint *demand_hist_read,
      const luisa::uint *aux_demand_hist_read,
      const luisa::uint *miss_bitset_read,
      std::vector<std::vector<MissCandidate>> &pool_misses,
      std::vector<std::vector<MissCandidate>> &pool_sec_misses,
      bool track_inflight_deps = false,
      std::uint32_t *dep_slot_mask_out = nullptr) const noexcept;

  void compute_active_source_hints(
      bool is_aux,
      const luisa::uint *slot_epoch_read,
      std::vector<std::uint32_t> &src_hit_slots,
      std::vector<std::uint8_t> &src_partner_hit,
      std::span<const PoolPinContext> pool_pin_ctxs = {}) const noexcept;

  void plan_pool_misses_host(
      std::uint32_t pool_idx,
      PoolRuntime &pool,
      std::vector<MissCandidate> &misses,
      std::vector<MissCandidate> &sec_misses,
      bool is_aux,
      MissSource source,
      std::uint32_t max_batch_tiles,
      const luisa::uint *slot_epoch_read,
      std::uint32_t total_slots,
      const std::vector<std::uint32_t> &src_hit_slots,
      std::vector<std::uint8_t> &src_partner_hit,
      bool &slot_epoch_host_seeded,
      bool &any_epoch_advanced,
      bool &any_epoch_wrapped,
      PoolAdmissionPlan &out_plan,
      PlanningScratch &scratch,
      const PoolPinContext *pin_ctx = nullptr,
      bool *deferred_no_victim_out = nullptr,
      bool *pool_backpressured_out = nullptr);

  [[nodiscard]] PackedChunkMeta prepare_chunk_staging(
      std::uint32_t pool_idx,
      PoolRuntime &pool,
      const PoolAdmissionPlan &plan,
      std::size_t chunk_begin,
      std::size_t chunk_end,
      std::size_t staging_cap,
      const ChunkStagingTarget &target,
      luisa::uint2 *pt_upd_dst,
      luisa::vector<luisa::uint2> &pt_upd_vec,
      std::size_t &pt_upd_count,
      luisa::uint2 *ep_upd_dst,
      luisa::vector<luisa::uint2> &ep_upd_vec,
      std::size_t &ep_upd_count,
      bool is_aux,
      bool &gpu_state_dirty,
      std::uint32_t &prof_b4_loaded,
      std::uint32_t &prof_f4_loaded,
      std::uint32_t &prof_pool_loaded,
      PlanningScratch &scratch);

  [[nodiscard]] std::size_t
  acquire_chunk_slot(std::uint32_t pool_idx,
                     std::uint64_t current_stream_handle) noexcept;
  [[nodiscard]] std::size_t
  acquire_meta_slot(std::uint64_t current_stream_handle) noexcept;
  void wait_all_upload_fences() noexcept;

  void service_misses_impl(
      luisa::compute::Stream &stream,
      MissSource source,
      luisa::compute::CommandList *wavefront_cmd = nullptr,
      const ServiceProfileContext *prof_ctx = nullptr) noexcept;
  void register_timeline_labels_if_active() const noexcept;
  [[nodiscard]] std::uint64_t query_vk_timeline_submit_seq() const noexcept;
  struct SourceMipView {
    const std::uint8_t *unorm8_pixels{nullptr};
    const float *float_pixels{nullptr};
    std::uint32_t width{};
    std::uint32_t height{};
  };

  [[nodiscard]] static SourceMipView source_mip_view(
      const SourceImageEntry &src,
      std::uint32_t mip_level) noexcept {
    if (!src.decoded || src.load_failed) {
      return {};
    }
    if (mip_level == 0u) {
      return {
          .unorm8_pixels =
              src.unorm8_pixels.empty() ? nullptr : src.unorm8_pixels.data(),
          .float_pixels =
              src.float_pixels.empty() ? nullptr : src.float_pixels.data(),
          .width = src.width,
          .height = src.height};
    }
    const std::size_t idx = static_cast<std::size_t>(mip_level) - 1u;
    if (!src.unorm8_mips.empty() && idx < src.unorm8_mips.size()) {
      const auto &m = src.unorm8_mips[idx];
      return {.unorm8_pixels = m.rgba.data(),
              .float_pixels = nullptr,
              .width = m.width,
              .height = m.height};
    }
    if (!src.float4_mips.empty() && idx < src.float4_mips.size()) {
      const auto &m = src.float4_mips[idx];
      return {.unorm8_pixels = nullptr,
              .float_pixels = m.rgba.data(),
              .width = m.width,
              .height = m.height};
    }
    return {};
  }

  void record_tile_loaded_stats(std::uint32_t page_id,
                                std::uint32_t tile_bytes) noexcept {
    if (page_id >= _pages.size()) {
      return;
    }
    ++_stats.tiles_loaded;
    _stats.bytes_loaded_total += tile_bytes;
    const std::uint32_t mip = _pages[page_id].mip_level;
    if (mip < _stats.tiles_loaded_per_mip.size()) {
      ++_stats.tiles_loaded_per_mip[mip];
    }
    if (page_id < _page_ever_loaded.size()) {
      if (_page_ever_loaded[page_id] == 0u) {
        _page_ever_loaded[page_id] = 1u;
        ++_stats.unique_pages_loaded;
        if (mip < _stats.unique_pages_per_mip.size()) {
          ++_stats.unique_pages_per_mip[mip];
        }
      } else {
        ++_stats.tiles_reloaded;
      }
    }
  }

  void decode_source_on_demand(std::uint32_t source_index) noexcept;
  void trim_host_decoded_cache(std::uint32_t protect_source_index) noexcept;
  void trim_host_decoded_cache(
      std::span<const std::uint32_t> protected_sources) noexcept;
  void apply_decoded_source_payload(
      SourceImageEntry &src,
      DecodedSourcePayload &&payload) noexcept;
  void decode_sources_batch(
      std::span<const std::uint32_t> source_indices) noexcept;
  void warm_prepack_on_init() noexcept;
  void sync_prepack_stats() noexcept;

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
  [[nodiscard]] static PreflightDemandCallable
  record_preflight_demand_callable(
      std::uint32_t logical_tile_size,
      luisa::string_view name) noexcept;
  [[nodiscard]] static SampleMipCallable record_mip_sampler(
      std::uint32_t physical_tile_size,
      std::uint32_t tile_border,
      std::uint32_t logical_tile_size,
      bool linear_repeat_only,
      luisa::string_view name) noexcept;
  [[nodiscard]] static PreflightMipDemandCallable
  record_preflight_mip_demand_callable(
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
  [[nodiscard]] LuisaTextureMipMode mip_mode() const noexcept {
    return _options.mip_mode;
  }
  [[nodiscard]] bool async_worker_enabled() const noexcept {
    return _options.enable_async_worker;
  }
  [[nodiscard]] std::uint32_t async_batch_slots() const noexcept {
    return _async_batch_slots;
  }
  [[nodiscard]] std::uint32_t
  current_async_in_flight_limit() const noexcept {
    return _async_backpressure_cooldown.load(std::memory_order_relaxed) > 0u
               ? 1u
               : _async_batch_slots;
  }
  [[nodiscard]] LuisaVirtualTextureCopyQueueMode
  active_copy_queue_mode() const noexcept {
    return _copy_queue_mode;
  }
  [[nodiscard]] LuisaVirtualTextureCompressionMode
  active_compression_mode() const noexcept {
    return _compression_mode;
  }
  [[nodiscard]] bool prefer_deferred_entry_drain() const noexcept {
    return _entry_drain_pressure.load(std::memory_order_relaxed) >= 4u;
  }
  [[nodiscard]] bool filtered_advance_enabled() const noexcept {
    return _filtered_advance_enabled;
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
  void dump_profile_if_enabled() const noexcept;

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

  [[nodiscard]] luisa::compute::Float4 sample_2d_const_mip(
      std::int32_t binding_id,
      luisa::compute::Expr<luisa::float2> uv,
      luisa::compute::Expr<luisa::float2> duv_dx,
      luisa::compute::Expr<luisa::float2> duv_dy,
      luisa::compute::UInt *miss_page_out,
      bool record_bitset = true) const noexcept;

  [[nodiscard]] luisa::compute::Float4 sample_2d_dynamic_mip(
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float2> uv,
      luisa::compute::Expr<luisa::float2> duv_dx,
      luisa::compute::Expr<luisa::float2> duv_dy,
      const luisa::compute::BindlessArray &direct_heap,
      luisa::compute::UInt *miss_page_out,
      bool record_bitset = true) const noexcept;

  void record_aux_miss(luisa::compute::Expr<luisa::uint> page_id) const noexcept;
  void record_preflight_demand_2d(
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float2> uv,
      luisa::compute::Expr<luisa::uint> weight) const noexcept;
  void record_preflight_demand_2d(
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float2> uv,
      luisa::compute::Expr<luisa::float2> duv_dx,
      luisa::compute::Expr<luisa::float2> duv_dy,
      luisa::compute::Expr<luisa::uint> weight) const noexcept;
  void clear_preflight_demand_buffers(
      luisa::compute::CommandList &cmd) noexcept;
  void enqueue_preflight_demand_readback(
      luisa::compute::CommandList &cmd) noexcept;
  void service_preflight_misses(
      luisa::compute::Stream &stream,
      bool readback_already_synced = false) noexcept;
  void prepare_timeline_labels_host() const noexcept {
    register_timeline_labels_if_active();
  }
  void record_preflight_event(std::uint64_t start_ns,
                              std::uint64_t end_ns,
                              std::string name,
                              std::string detail = {}) const noexcept;
  void record_pso_compile_event(std::uint64_t start_ns,
                                std::uint64_t end_ns,
                                std::string name,
                                std::string detail = {}) const noexcept;
  void set_preflight_timing_stats(bool enabled,
                                  bool overlapped,
                                  std::uint32_t stride,
                                  std::uint32_t bounces,
                                  double compile_ms,
                                  double trace_ms,
                                  double service_ms,
                                  double total_ms) noexcept;
  void set_pso_compile_ms(double ms) noexcept;
  [[nodiscard]] bool has_profiler() const noexcept {
    return _profiler != nullptr;
  }

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

  void encode_wavefront_filter_ready(
      const luisa::compute::coro::WavefrontCoroAdvanceFilterContext &ctx,
      const FilterReadyShader &filter_shader) noexcept;

  [[nodiscard]] bool submit_async_wavefront_collect(
      const luisa::compute::coro::WavefrontCoroExtensionDispatchContext &ctx,
      const CollectShader &collect_shader) noexcept;
  void notify_async_scheduler_progress(
      const luisa::compute::coro::WavefrontCoroSchedulerSnapshot &snapshot,
      bool waiting_for_progress) noexcept;
  [[nodiscard]] luisa::compute::coro::WavefrontCoroAsyncPollResult
  poll_async_wavefront() noexcept;
  [[nodiscard]] bool poll_async_wavefront_ready(
      luisa::compute::coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept;
  void wait_async_wavefront() noexcept;
  [[nodiscard]] bool wait_async_wavefront_any(
      luisa::compute::coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept;
  void publish_async_wavefront(
      const luisa::compute::coro::WavefrontCoroAsyncPublishContext &ctx) noexcept;

  void ensure_eager_direct_textures_uploaded(LuisaSceneData &scene);

  void populate_temporary_direct_textures(
      LuisaSceneData &scene,
      luisa::compute::Stream &stream,
      const std::set<contract::ImageId> &image_ids,
      std::vector<luisa::compute::Image<float>> &temp_images,
      std::vector<std::uint32_t> &rebound_slots);
};

} // namespace psycles::luisa_backend::detail
