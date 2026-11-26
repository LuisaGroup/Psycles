#include "virtual_texture_cache.h"

#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"
#include "virtual_texture_cache_bc.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;

[[nodiscard]] std::uint32_t next_power_of_two_u32(std::uint32_t v) noexcept {
  if (v <= 1u) {
    return 1u;
  }
  return std::bit_ceil(v);
}

void collect_block_image_ids(
    const compiler::cycles_svm::SvmBlockIr &block,
    std::span<const std::uint32_t> words,
    std::span<const compiler::cycles_svm::ImageBinding> images,
    std::set<contract::ImageId> &out_ids) {
  namespace abi = compiler::cycles_svm;
  for (const auto &op : block) {
    if (op.is_conditional) {
      collect_block_image_ids(op.children, words, images, out_ids);
      continue;
    }
    const auto opcode = op.instruction.opcode;
    if (opcode == abi::NODE_TEX_IMAGE ||
        opcode == abi::NODE_TEX_IMAGE_DERIVATIVE ||
        opcode == abi::NODE_TEX_IMAGE_BOX ||
        opcode == abi::NODE_TEX_IMAGE_BOX_DERIVATIVE ||
        opcode == abi::NODE_TEX_ENVIRONMENT ||
        opcode == abi::NODE_TEX_ENVIRONMENT_DERIVATIVE) {
      const auto offset = op.instruction.payload_offset;
      if (offset < words.size()) {
        const auto binding_id = words[offset];
        if (binding_id < images.size() && !images[binding_id].nishita) {
          out_ids.insert(contract::ImageId{images[binding_id].resource_id});
        }
      }
    }
  }
}

void init_pool_geometry(
    luisa::compute::Device &device,
    VirtualTextureCache::PoolConfig &pool,
    std::uint32_t tile_size,
    std::uint32_t capacity,
    std::uint32_t total_pages,
    std::uint32_t &next_bindless_slot) {
  pool.capacity = capacity;
  pool.eviction_possible = (capacity < total_pages);
  if (capacity == 0u) {
    return;
  }
  const std::uint32_t max_dim = pool.is_high_precision ? 4096u : 8192u;
  const std::uint32_t max_tiles_axis = std::max(max_dim / tile_size, 1u);
  const std::uint32_t max_tiles_per_atlas = max_tiles_axis * max_tiles_axis;

  const std::uint32_t num_atlases =
      (capacity + max_tiles_per_atlas - 1u) / max_tiles_per_atlas;
  if (num_atlases == 1u) {
    const auto approx_x = static_cast<std::uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(capacity))));
    auto tiles_x = std::min(next_power_of_two_u32(approx_x), max_tiles_axis);
    auto tiles_y =
        std::clamp((capacity + tiles_x - 1u) / tiles_x, 1u, max_tiles_axis);
    while (tiles_x * tiles_y < capacity && tiles_x < max_tiles_axis) {
      tiles_x *= 2u;
      tiles_y =
          std::clamp((capacity + tiles_x - 1u) / tiles_x, 1u, max_tiles_axis);
    }
    pool.atlas_tiles_x = tiles_x;
    pool.atlas_tiles_x_log2 = static_cast<std::uint32_t>(std::countr_zero(tiles_x));
    pool.atlas_tiles_x_mask = tiles_x - 1u;
    pool.full_atlas_tiles_y = tiles_y;
    pool.last_atlas_tiles_y = tiles_y;
    pool.slots_per_atlas = tiles_x * tiles_y;
    pool.slots_per_atlas_log2 = 31u;
    pool.slots_per_atlas_mask = 0xffffffffu;
  } else {
    pool.atlas_tiles_x = max_tiles_axis;
    pool.atlas_tiles_x_log2 =
        static_cast<std::uint32_t>(std::countr_zero(max_tiles_axis));
    pool.atlas_tiles_x_mask = max_tiles_axis - 1u;
    pool.full_atlas_tiles_y = max_tiles_axis;
    const auto rem_slots = capacity - (num_atlases - 1u) * max_tiles_per_atlas;
    pool.last_atlas_tiles_y = std::clamp(
        (rem_slots + max_tiles_axis - 1u) / max_tiles_axis, 1u, max_tiles_axis);
    pool.slots_per_atlas = max_tiles_per_atlas;
    pool.slots_per_atlas_log2 =
        static_cast<std::uint32_t>(std::countr_zero(max_tiles_per_atlas));
    pool.slots_per_atlas_mask = max_tiles_per_atlas - 1u;
  }

  pool.bindless_base = next_bindless_slot;
  next_bindless_slot += num_atlases * 2u;
  const auto storage = pool.atlas_storage;
  pool.atlases.reserve(num_atlases);
  pool.atlas_widths.reserve(num_atlases);
  pool.atlas_heights.reserve(num_atlases);
  for (std::uint32_t a = 0u; a < num_atlases; ++a) {
    const std::uint32_t ty = (a + 1u == num_atlases) ? pool.last_atlas_tiles_y
                                                     : pool.full_atlas_tiles_y;
    const std::uint32_t pw = pool.atlas_tiles_x * tile_size;
    const std::uint32_t ph = ty * tile_size;
    pool.atlases.emplace_back(device.create_image<float>(storage, pw, ph));
    pool.atlas_widths.push_back(pw);
    pool.atlas_heights.push_back(ph);
  }
  pool.full_inv_w = 1.0f / static_cast<float>(pool.atlas_tiles_x * tile_size);
  pool.full_inv_h =
      1.0f / static_cast<float>(pool.full_atlas_tiles_y * tile_size);
  pool.last_inv_h =
      1.0f / static_cast<float>(pool.last_atlas_tiles_y * tile_size);

  const std::uint32_t max_staging = pool.is_high_precision ? 128u : 512u;
  pool.staging_capacity = std::min(capacity, max_staging);
  const std::size_t max_chunks_for_cap =
      (capacity + std::max<std::uint32_t>(pool.staging_capacity, 1u) - 1u) /
      std::max<std::uint32_t>(pool.staging_capacity, 1u);
  pool.chunks_per_batch_slot = std::clamp<std::size_t>(max_chunks_for_cap, 1u, 4u);
}

[[nodiscard]] VtPoolConfigGpu make_pool_config_gpu(
    const VirtualTextureCache::PoolConfig &pool) noexcept {
  return VtPoolConfigGpu{
      .num_atlases = static_cast<luisa::uint>(pool.atlases.size()),
      .atlas_tiles_x_log2 = pool.atlas_tiles_x_log2,
      .atlas_tiles_x_mask = pool.atlas_tiles_x_mask,
      .slots_per_atlas_log2 = pool.slots_per_atlas_log2,
      .slots_per_atlas_mask = pool.slots_per_atlas_mask,
      .bindless_base = pool.bindless_base,
      .epoch_offset = pool.epoch_offset,
      .channel_swizzle = static_cast<luisa::uint>(pool.swizzle),
      .full_inv_w = pool.full_inv_w,
      .full_inv_h = pool.full_inv_h,
      .last_inv_h = pool.last_inv_h,
      .pad1 = 0u};
}

class VtWavefrontExtensionHandler final
    : public luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler {

public:
  using CollectShader = VirtualTextureCache::CollectShaderEntry;

private:
  std::shared_ptr<VirtualTextureCache> _cache;
  bool _use_fast_field_read{false};
  std::uint32_t _field_index{0u};
  coro::CoroFrameStorageLayout _layout{};
  bool _soa{true};
  const luisa::compute::CoroFrameDesc *_frame_desc{nullptr};
  const coro::CoroSlotAccess *_page_binding{nullptr};
  luisa::vector<size_t> _reconstruct_fields;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::uint, luisa::uint>
      _fallback_collect_shader;
  const luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                                 luisa::compute::Buffer<luisa::uint>,
                                 luisa::compute::Buffer<luisa::uint>,
                                 luisa::uint, luisa::uint>
      *_cached_shader{nullptr};
  const VirtualTextureCache::FilterReadyShader *_cached_filter_shader{nullptr};

public:
  VtWavefrontExtensionHandler(
      std::shared_ptr<VirtualTextureCache> cache,
      bool use_fast_field_read,
      std::uint32_t field_index,
      const coro::CoroFrameStorageLayout &layout,
      bool soa,
      const luisa::compute::CoroFrameDesc *frame_desc,
      const coro::CoroSlotAccess *page_binding,
      luisa::vector<size_t> reconstruct_fields,
      luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                               luisa::compute::Buffer<luisa::uint>,
                               luisa::compute::Buffer<luisa::uint>,
                               luisa::uint, luisa::uint>
          fallback_collect_shader,
      const luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                                     luisa::compute::Buffer<luisa::uint>,
                                     luisa::compute::Buffer<luisa::uint>,
                                     luisa::uint, luisa::uint> *cached_shader,
      const VirtualTextureCache::FilterReadyShader *cached_filter_shader)
      : _cache{std::move(cache)},
        _use_fast_field_read{use_fast_field_read},
        _field_index{field_index},
        _layout{layout},
        _soa{soa},
        _frame_desc{frame_desc},
        _page_binding{page_binding},
        _reconstruct_fields{std::move(reconstruct_fields)},
        _fallback_collect_shader{std::move(fallback_collect_shader)},
        _cached_shader{cached_shader},
        _cached_filter_shader{cached_filter_shader} {
    static_cast<void>(_use_fast_field_read);
    static_cast<void>(_field_index);
    static_cast<void>(_layout);
    static_cast<void>(_soa);
    static_cast<void>(_frame_desc);
    static_cast<void>(_page_binding);
    static_cast<void>(_reconstruct_fields);
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return vt_extension_schema;
  }

  [[nodiscard]] luisa::compute::coro::WavefrontCoroExtensionExecution
  execution() const noexcept override {
    return (_cache && _cache->async_worker_enabled())
               ? coro::WavefrontCoroExtensionExecution::blocking_async
               : coro::WavefrontCoroExtensionExecution::stage;
  }

  [[nodiscard]] luisa::string_view async_domain() const noexcept override {
    return vt_extension_schema;
  }

  [[nodiscard]] luisa::uint max_async_in_flight_batches() const noexcept override {
    return _cache ? _cache->async_batch_slots() : 1u;
  }

  [[nodiscard]] luisa::uint
  current_async_in_flight_limit() const noexcept override {
    return _cache ? _cache->current_async_in_flight_limit() : 1u;
  }

  void on_async_scheduler_progress(
      const coro::WavefrontCoroSchedulerSnapshot &snapshot,
      bool waiting_for_progress) noexcept override;

  [[nodiscard]] bool defer_while_other_queues_nonempty() const noexcept override {
    return true;
  }

  [[nodiscard]] bool prefer_deferred_entry_drain() const noexcept override {
    return _cache && _cache->prefer_deferred_entry_drain();
  }

  [[nodiscard]] bool supports_command_list() const noexcept override {
    return true;
  }

  [[nodiscard]] bool supports_filtered_advance() const noexcept override {
    return _cache && _cache->filtered_advance_enabled() &&
           _cache->has_virtual_images() && _cached_filter_shader != nullptr;
  }

  void encode_advance_filter(
      const coro::WavefrontCoroAdvanceFilterContext &ctx) noexcept override {
    if (_cache && _cached_filter_shader != nullptr) {
      _cache->encode_wavefront_filter_ready(ctx, *_cached_filter_shader);
    }
  }

  [[nodiscard]] const luisa::compute::Shader1D<
      luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint> &
  collect_shader() const noexcept {
    return _cached_shader != nullptr ? *_cached_shader
                                     : _fallback_collect_shader;
  }

  [[nodiscard]] bool submit_async(
      const coro::WavefrontCoroExtensionDispatchContext &ctx) noexcept override;
  [[nodiscard]] coro::WavefrontCoroAsyncPollResult
  poll_async() noexcept override;
  [[nodiscard]] bool poll_async_ready(
      coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override;
  void wait_async() noexcept override;
  [[nodiscard]] bool wait_async_any(
      coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override;
  void publish_async(
      const coro::WavefrontCoroAsyncPublishContext &ctx) noexcept override;

  void dispatch(const coro::WavefrontCoroExtensionDispatchContext &ctx) noexcept override;
};

} // namespace

std::set<contract::ImageId> collect_shader_routine_image_ids(
    const compiler::cycles_svm::CompiledShaderTable &compilation,
    std::span<const std::uint32_t> shader_indices,
    compiler::cycles_svm::ShaderType routine_type) {
  std::set<contract::ImageId> result;
  if (compilation.images.empty()) {
    return result;
  }
  const auto add_all = [&] {
    for (const auto &img : compilation.images) {
      if (!img.nishita) {
        result.insert(contract::ImageId{img.resource_id});
      }
    }
  };
  if (!compilation.table.ir.valid) {
    add_all();
    return result;
  }
  const auto &shaders = compilation.table.ir.shaders;
  const std::span<const std::uint32_t> words{compilation.table.words};
  for (const auto idx : shader_indices) {
    if (idx >= shaders.size()) {
      continue;
    }
    const auto &block = shaders[idx].routine_for(routine_type);
    if (!block.empty()) {
      collect_block_image_ids(block, words, compilation.images, result);
    }
  }
  return result;
}

TemporaryDirectTextureScope::TemporaryDirectTextureScope(
    std::shared_ptr<LuisaSceneData> scene,
    Stream &stream,
    const std::set<contract::ImageId> &image_ids)
    : _scene{std::move(scene)} {
  if (!_scene || !_scene->virtual_texture_cache || image_ids.empty()) {
    return;
  }
  _scene->virtual_texture_cache->populate_temporary_direct_textures(
      *_scene, stream, image_ids, _temp_images, _rebound_slots);
  _active = !_rebound_slots.empty();
}

TemporaryDirectTextureScope::TemporaryDirectTextureScope(
    TemporaryDirectTextureScope &&other) noexcept
    : _scene{std::move(other._scene)},
      _temp_images{std::move(other._temp_images)},
      _rebound_slots{std::move(other._rebound_slots)},
      _active{std::exchange(other._active, false)} {}

TemporaryDirectTextureScope &
TemporaryDirectTextureScope::operator=(TemporaryDirectTextureScope &&other) noexcept {
  if (this != &other) {
    release();
    _scene = std::move(other._scene);
    _temp_images = std::move(other._temp_images);
    _rebound_slots = std::move(other._rebound_slots);
    _active = std::exchange(other._active, false);
  }
  return *this;
}

TemporaryDirectTextureScope::~TemporaryDirectTextureScope() noexcept { release(); }

void TemporaryDirectTextureScope::release() noexcept {
  if (!_active || !_scene) {
    _temp_images.clear();
    _rebound_slots.clear();
    _active = false;
    return;
  }
  if (!_rebound_slots.empty() && !_scene->images.empty()) {
    const auto &dummy = _scene->images.front();
    for (const auto slot : _rebound_slots) {
      _scene->texture_heap.emplace_on_update(
          slot, dummy, Sampler::linear_point_repeat());
    }
    auto stream = _scene->device.create_stream();
    stream << _scene->texture_heap.update() << synchronize();
  }
  _rebound_slots.clear();
  _temp_images.clear();
  _active = false;
}

bool VirtualTextureCache::CollectShaderKey::operator==(
    const CollectShaderKey &other) const noexcept {
  return soa == other.soa && single_field == other.single_field &&
         field_index == other.field_index && layout_hash == other.layout_hash &&
         frame_desc == other.frame_desc &&
         boundary_index == other.boundary_index;
}

std::shared_ptr<VirtualTextureCache> VirtualTextureCache::create(
    Device &device,
    Stream &stream,
    const LuisaVirtualTextureCacheOptions &options,
    std::span<const VtProbedSceneImage> probed_images,
    std::span<const compiler::cycles_svm::ImageBinding> svm_image_handles,
    std::span<const CyclesSvmImageBindingGpu> svm_image_bindings,
    const compiler::cycles_svm::CompiledShaderTable *compilation) {
  return std::make_shared<VirtualTextureCache>(
      device, stream, options, probed_images, svm_image_handles,
      svm_image_bindings, compilation);
}

VirtualTextureCache::VirtualTextureCache(
    Device device,
    Stream &stream,
    const LuisaVirtualTextureCacheOptions &options,
    std::span<const VtProbedSceneImage> probed_images,
    std::span<const compiler::cycles_svm::ImageBinding> svm_image_handles,
    std::span<const CyclesSvmImageBindingGpu> svm_image_bindings,
    const compiler::cycles_svm::CompiledShaderTable *compilation)
    : _device{std::move(device)},
      _options{options},
      _physical_tile_size{valid_luisa_virtual_texture_tile_size(options.physical_tile_size)
                              ? options.physical_tile_size
                              : 64u},
      _tile_border{vt_border_size},
      _logical_tile_size{_physical_tile_size - 2u * vt_border_size},
      _warp_size{static_cast<std::uint32_t>(_device.compute_warp_size())},
      _upload_event{_device.create_event()},
      _pinned_ext{_device.extension<PinnedMemoryExt>()} {
  const bool supports_batch_copy =
      (_device.impl()->query("vk_batch_copy_cmd") == "1");
  _compression_mode = _options.compression_mode;
  if (_compression_mode != LuisaVirtualTextureCompressionMode::off &&
      !supports_batch_copy) {
    LUISA_WARNING(
        "VT compression={} requires vk_batch_copy_cmd; falling back to off.",
        luisa_virtual_texture_compression_mode_name(_compression_mode));
    _compression_mode = LuisaVirtualTextureCompressionMode::off;
  }
  if (const char *filt_raw = std::getenv("PSYCLES_VT_FILTERED_ADVANCE")) {
    _filtered_advance_enabled = (std::string_view{filt_raw} != "0");
  }
  _sources.reserve(probed_images.size());
  for (std::size_t i = 0u; i < probed_images.size(); ++i) {
    const auto &src = probed_images[i];
    _id_to_source.emplace(src.id.value, static_cast<std::uint32_t>(i));
    SourceImageEntry entry;
    entry.id = src.id;
    entry.name = src.name;
    entry.encoded_data = src.encoded_data;
    entry.alpha_type = src.alpha_type;
    entry.color_space = src.color_space;
    entry.width = src.probed.width;
    entry.height = src.probed.height;
    entry.texture_limit = src.texture_limit;
    entry.storage = src.probed.storage;
    entry.probed_info = src.probed;
    if (entry.probed_info.channels == 0u) {
      entry.probed_info.channels = 4u;
    }
    if (entry.probed_info.bit_depth == 0u) {
      entry.probed_info.is_floating_point =
          (entry.storage == io::DecodedImageStorage::float32);
      entry.probed_info.bit_depth =
          entry.probed_info.is_floating_point ? 32u : 8u;
    }
    entry.load_failed = src.load_failed;
    entry.force_direct = src.force_direct;
    entry.identity_key = vt_make_source_identity_key(
        entry.encoded_data, entry.width, entry.height, entry.storage,
        entry.alpha_type, entry.color_space);
    _sources.push_back(std::move(entry));
  }
  _bg_source_queued.assign(_sources.size(), 0u);
  _bg_ready_payloads.resize(_sources.size());

  _source_partners.resize(_sources.size());
  if (compilation != nullptr && compilation->table.ir.valid &&
      !compilation->images.empty()) {
    const std::span<const std::uint32_t> words{compilation->table.words};
    for (const auto &shader : compilation->table.ir.shaders) {
      const auto &block =
          shader.routine_for(compiler::cycles_svm::SHADER_TYPE_SURFACE);
      if (block.empty()) {
        continue;
      }
      std::set<contract::ImageId> shader_img_ids;
      collect_block_image_ids(block, words, compilation->images, shader_img_ids);
      if (shader_img_ids.size() < 2u) {
        continue;
      }
      std::vector<std::uint32_t> shader_sources;
      shader_sources.reserve(shader_img_ids.size());
      for (const auto img_id : shader_img_ids) {
        const auto it = _id_to_source.find(img_id.value);
        if (it == _id_to_source.end()) {
          continue;
        }
        const auto src_idx = it->second;
        const auto &src = _sources[src_idx];
        if (!src.load_failed && !src.force_direct) {
          shader_sources.push_back(src_idx);
        }
      }
      if (shader_sources.size() < 2u) {
        continue;
      }
      std::sort(shader_sources.begin(), shader_sources.end());
      shader_sources.erase(
          std::unique(shader_sources.begin(), shader_sources.end()),
          shader_sources.end());
      for (std::size_t i = 0u; i < shader_sources.size(); ++i) {
        const auto u = shader_sources[i];
        for (std::size_t j = 0u; j < shader_sources.size(); ++j) {
          if (i != j) {
            _source_partners[u].push_back(shader_sources[j]);
          }
        }
      }
    }
    for (auto &partners : _source_partners) {
      if (partners.size() > 1u) {
        std::sort(partners.begin(), partners.end());
        partners.erase(std::unique(partners.begin(), partners.end()),
                       partners.end());
      }
    }
  }

  const auto T = _logical_tile_size;
  struct PendingVirtualRange {
    std::uint32_t source_index{};
    std::uint32_t extension{};
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    VtTileFormat format{VtTileFormat::rgba8};
    bool is_high_precision{false};
  };
  // Deduplicate virtual page ranges by (image_id, extension).
  std::unordered_map<std::uint64_t, std::pair<VtTileFormat, std::size_t>>
      range_lookup;
  std::array<std::vector<PendingVirtualRange>, kVtTileFormatCount>
      ranges_by_format;
  std::vector<std::pair<VtTileFormat, std::size_t>> handle_range_ref(
      svm_image_handles.size(),
      {VtTileFormat::rgba8, std::numeric_limits<std::size_t>::max()});

  _descs.resize(svm_image_handles.size());
  for (std::size_t i = 0u; i < svm_image_handles.size(); ++i) {
    const auto &h = svm_image_handles[i];
    const auto &b = svm_image_bindings[i];
    auto &desc = _descs[i];
    desc.direct_texture_slot = b.texture_slot;
    const auto interp = b.sampler & cycles_svm_image_interpolation_mask;
    const auto ext =
        (b.sampler & cycles_svm_image_extension_mask) >>
        cycles_svm_image_extension_shift;
    const bool load_failed =
        h.nishita.has_value()
            ? false
            : ((b.sampler & cycles_svm_image_load_failed_flag) != 0u);
    desc.flags = (interp & 0x3u) | ((ext & 0x3u) << vt_desc_ext_shift) |
                 (load_failed ? vt_desc_flag_load_failed : 0u);
    if (h.nishita || load_failed) {
      continue;
    }
    const auto src_it = _id_to_source.find(h.resource_id);
    if (src_it == _id_to_source.end()) {
      desc.flags |= vt_desc_flag_load_failed;
      continue;
    }
    const auto &src = _sources[src_it->second];
    if (src.load_failed || src.width == 0u || src.height == 0u) {
      desc.flags |= vt_desc_flag_load_failed;
      continue;
    }
    desc.width = src.width;
    desc.height = src.height;
    if (src.storage == io::DecodedImageStorage::float32) {
      desc.flags |= vt_desc_flag_float4;
    }
    if (src.force_direct) {
      desc.mip_count =
          (_options.mip_mode == LuisaTextureMipMode::isotropic)
              ? vt_compute_mip_count(src.width, src.height)
              : 1u;
      continue;
    }
    desc.flags |= vt_desc_flag_virtual;
    const std::uint32_t tx = (src.width + T - 1u) / T;
    const std::uint32_t ty = (src.height + T - 1u) / T;
    desc.tiles_x = tx;
    desc.tiles_y = ty;

    const std::uint64_t dedup_key =
        (static_cast<std::uint64_t>(src_it->second) << 2u) |
        static_cast<std::uint64_t>(ext & 0x3u);
    if (const auto it = range_lookup.find(dedup_key); it != range_lookup.end()) {
      handle_range_ref[i] = it->second;
    } else {
      const VtTileFormat fmt = vt_select_tile_format(
          _compression_mode, src.probed_info, src.alpha_type, src.color_space,
          ext & 0x3u, supports_batch_copy);
      auto &vec = ranges_by_format[static_cast<std::size_t>(fmt)];
      const std::size_t idx = vec.size();
      vec.push_back(PendingVirtualRange{
          .source_index = src_it->second,
          .extension = ext & 0x3u,
          .tiles_x = tx,
          .tiles_y = ty,
          .format = fmt,
          .is_high_precision = vt_tile_format_is_high_precision(fmt)});
      const std::pair<VtTileFormat, std::size_t> ref{fmt, idx};
      range_lookup.emplace(dedup_key, ref);
      handle_range_ref[i] = ref;
    }
  }

  struct PendingRangeAlloc {
    std::uint32_t mip_table_offset{};
    std::uint32_t mip_count{1u};
    std::uint32_t base_page_offset{};
  };
  const bool use_mips = (_options.mip_mode == LuisaTextureMipMode::isotropic);
  std::array<std::vector<PendingRangeAlloc>, kVtTileFormatCount>
      allocs_by_format;
  std::array<std::uint32_t, kVtTileFormatCount> pool_index_by_format{};
  pool_index_by_format.fill(std::numeric_limits<std::uint32_t>::max());

  std::vector<std::uint32_t> src_max_tx(_sources.size(), 0u);
  std::vector<std::uint32_t> src_max_ty(_sources.size(), 0u);
  for (std::size_t fi = 0u; fi < kVtTileFormatCount; ++fi) {
    for (const auto &pr : ranges_by_format[fi]) {
      src_max_tx[pr.source_index] = std::max(src_max_tx[pr.source_index], pr.tiles_x);
      src_max_ty[pr.source_index] = std::max(src_max_ty[pr.source_index], pr.tiles_y);
    }
  }
  std::vector<std::uint32_t> src_blocks_x(_sources.size(), 0u);
  std::vector<std::uint32_t> src_block_offset(_sources.size(), 0u);
  _total_tile_blocks = 0u;
  for (std::size_t s = 0u; s < _sources.size(); ++s) {
    src_block_offset[s] = _total_tile_blocks;
    src_blocks_x[s] = (src_max_tx[s] + 7u) >> 3u;
    const std::uint32_t by = (src_max_ty[s] + 7u) >> 3u;
    _total_tile_blocks += src_blocks_x[s] * by;
  }

  std::uint32_t max_mip_count = 1u;
  std::uint32_t page_cursor = 0u;
  for (std::size_t fi = 0u; fi < kVtTileFormatCount; ++fi) {
    const auto &ranges = ranges_by_format[fi];
    if (ranges.empty()) {
      continue;
    }
    const auto fmt = static_cast<VtTileFormat>(fi);
    const std::uint32_t pool_idx = static_cast<std::uint32_t>(_pools.size());
    pool_index_by_format[fi] = pool_idx;
    _pools.emplace_back();
    auto &pool = _pools.back().config;
    pool.format = fmt;
    pool.swizzle = vt_tile_format_swizzle(fmt);
    pool.is_high_precision = vt_tile_format_is_high_precision(fmt);
    pool.atlas_storage = vt_tile_format_storage(fmt);
    pool.tile_bytes = vt_tile_format_bytes(_physical_tile_size, fmt);
    pool.tile_staging_u32s =
        (pool.tile_bytes +
         static_cast<std::uint32_t>(sizeof(luisa::uint)) - 1u) /
        static_cast<std::uint32_t>(sizeof(luisa::uint));
    pool.page_begin = page_cursor;

    auto &allocs = allocs_by_format[fi];
    allocs.resize(ranges.size());
    const std::uint8_t disk_comp = vt_disk_compression_tag(fmt);
    for (std::size_t r = 0u; r < ranges.size(); ++r) {
      const auto &pr = ranges[r];
      const auto &src = _sources[pr.source_index];
      const std::uint32_t mip_count =
          use_mips ? vt_compute_mip_count(src.width, src.height) : 1u;
      max_mip_count = std::max(max_mip_count, mip_count);
      allocs[r] = PendingRangeAlloc{
          .mip_table_offset = static_cast<std::uint32_t>(_mip_levels.size()),
          .mip_count = mip_count,
          .base_page_offset = page_cursor};
      for (std::uint32_t m = 0u; m < mip_count; ++m) {
        const std::uint32_t mw = vt_mip_extent(src.width, m);
        const std::uint32_t mh = vt_mip_extent(src.height, m);
        const std::uint32_t mtx = std::max(1u, (mw + T - 1u) / T);
        const std::uint32_t mty = std::max(1u, (mh + T - 1u) / T);
        const std::uint32_t mip_page_off = page_cursor;
        _mip_levels.push_back(VtMipLevelGpu{
            .width = mw,
            .height = mh,
            .tiles_x = mtx,
            .tiles_y = mty,
            .page_table_offset = mip_page_off,
            .page_count = mtx * mty});
        const VtPrepackedImageKey img_key{
            .source = src.identity_key,
            .extension = pr.extension,
            .physical_tile_size = _physical_tile_size,
            .tile_border = _tile_border,
            .compression = disk_comp,
            .mip_level = m};
        for (std::uint32_t ty = 0u; ty < mty; ++ty) {
          const std::uint32_t block_row =
              src_block_offset[pr.source_index] +
              (ty >> 3u) * src_blocks_x[pr.source_index];
          for (std::uint32_t tx = 0u; tx < mtx; ++tx) {
            _pages.push_back(VirtualPageMeta{
                .source_index = pr.source_index,
                .tile_x = tx,
                .tile_y = ty,
                .tiles_x = mtx,
                .tiles_y = mty,
                .extension = pr.extension,
                .mip_level = m,
                .block_id = block_row + (tx >> 3u),
                .format = fmt,
                .pool_index = pool_idx,
                .is_high_precision = pool.is_high_precision});
            _page_prepack_keys.push_back(VtPrepackedPageKey{
                .image = img_key, .tile_x = tx, .tile_y = ty});
            ++page_cursor;
          }
        }
      }
    }
    pool.page_end = page_cursor;
    pool.virtual_pages = pool.page_end - pool.page_begin;
  }

  for (std::size_t i = 0u; i < svm_image_handles.size(); ++i) {
    if ((_descs[i].flags & vt_desc_flag_virtual) == 0u) {
      _descs[i].mip_count = 1u;
      continue;
    }
    const auto [fmt, idx] = handle_range_ref[i];
    const auto fi = static_cast<std::size_t>(fmt);
    const auto &alloc = allocs_by_format[fi][idx];
    _descs[i].page_table_offset = alloc.base_page_offset;
    _descs[i].mip_table_offset = alloc.mip_table_offset;
    _descs[i].mip_count = alloc.mip_count;
    _descs[i].pool_index = pool_index_by_format[fi];
  }
  _page_ever_loaded.assign(_pages.size(), 0u);
  _stats.active_mip_mode = _options.mip_mode;
  _stats.total_mip_pages = _pages.size();
  _stats.unique_pages_per_mip.assign(max_mip_count, 0ull);
  _stats.tiles_loaded_per_mip.assign(max_mip_count, 0ull);

  _stats.total_images = static_cast<std::uint32_t>(_sources.size());
  for (const auto &s : _sources) {
    if (!s.load_failed && !s.force_direct) {
      ++_stats.virtual_images;
    }
  }
  _stats.physical_tile_size = _physical_tile_size;
  _stats.tile_border = _tile_border;
  _stats.logical_tile_size = _logical_tile_size;
  _stats.gpu_budget_bytes = options.gpu_cache_budget_bytes;
  _stats.active_compression_mode = _compression_mode;

  // Partition GPU budget proportionally across all active format pools.
  // Isotropic dual-mip sampling requires at least 2 slots in a pool to hold mip0/mip1 simultaneously.
  const std::uint64_t min_pool_slots =
      (_options.mip_mode != LuisaTextureMipMode::mip0) ? 2ull : 1ull;
  std::vector<std::uint32_t> pool_caps(_pools.size(), 0u);
  if (_pools.size() == 1u) {
    const auto &pc = _pools[0u].config;
    pool_caps[0u] = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        pc.virtual_pages,
        std::max<std::uint64_t>(
            min_pool_slots, options.gpu_cache_budget_bytes / pc.tile_bytes)));
  } else if (!_pools.empty()) {
    const std::size_t n_pools = _pools.size();
    std::vector<std::uint64_t> demand_bytes(n_pools, 0ull);
    std::uint64_t total_demand_bytes = 0ull;
    for (std::size_t p = 0u; p < n_pools; ++p) {
      demand_bytes[p] =
          static_cast<std::uint64_t>(_pools[p].config.virtual_pages) *
          _pools[p].config.tile_bytes;
      total_demand_bytes += demand_bytes[p];
    }
    if (options.gpu_cache_budget_bytes >= total_demand_bytes) {
      for (std::size_t p = 0u; p < n_pools; ++p) {
        pool_caps[p] = _pools[p].config.virtual_pages;
      }
    } else {
      std::vector<std::uint64_t> min_tiles(n_pools, 1ull);
      std::uint64_t total_floor_bytes = 0ull;
      for (std::size_t p = 0u; p < n_pools; ++p) {
        const std::uint64_t tb = _pools[p].config.tile_bytes;
        const std::uint64_t vp = _pools[p].config.virtual_pages;
        const std::uint64_t f = std::min<std::uint64_t>(
            vp, std::max<std::uint64_t>(
                    1ull, std::min<std::uint64_t>(
                              16ull, (options.gpu_cache_budget_bytes / n_pools) /
                                         std::max<std::uint64_t>(1ull, 4ull * tb))));
        min_tiles[p] = f;
        total_floor_bytes += f * tb;
      }
      if (total_floor_bytes > options.gpu_cache_budget_bytes) {
        total_floor_bytes = 0ull;
        for (std::size_t p = 0u; p < n_pools; ++p) {
          min_tiles[p] = 1ull;
          total_floor_bytes += _pools[p].config.tile_bytes;
        }
      }
      const std::uint64_t remaining_budget =
          options.gpu_cache_budget_bytes > total_floor_bytes
              ? options.gpu_cache_budget_bytes - total_floor_bytes
              : 0ull;
      std::vector<std::uint64_t> rem_demand(n_pools, 0ull);
      std::uint64_t total_rem_demand = 0ull;
      for (std::size_t p = 0u; p < n_pools; ++p) {
        const std::uint64_t floor_b = min_tiles[p] * _pools[p].config.tile_bytes;
        rem_demand[p] =
            demand_bytes[p] > floor_b ? demand_bytes[p] - floor_b : 0ull;
        total_rem_demand += rem_demand[p];
      }
      for (std::size_t p = 0u; p < n_pools; ++p) {
        const std::uint64_t tb = _pools[p].config.tile_bytes;
        const std::uint64_t vp = _pools[p].config.virtual_pages;
        std::uint64_t pool_budget = min_tiles[p] * tb;
        if (total_rem_demand > 0ull && remaining_budget > 0ull) {
          const double ratio = static_cast<double>(rem_demand[p]) /
                               static_cast<double>(total_rem_demand);
          pool_budget += static_cast<std::uint64_t>(
              static_cast<double>(remaining_budget) * ratio);
        }
        pool_budget = std::min(pool_budget, demand_bytes[p]);
        const std::uint64_t min_slots = std::min<std::uint64_t>(min_pool_slots, vp);
        pool_caps[p] = static_cast<std::uint32_t>(
            std::clamp<std::uint64_t>(pool_budget / tb, min_slots, vp));
      }
    }
  }

  std::uint32_t next_bindless_slot = 0u;
  std::uint32_t epoch_cursor = 0u;
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    auto &pc = _pools[p].config;
    init_pool_geometry(_device, pc, _physical_tile_size, pool_caps[p],
                       pc.virtual_pages, next_bindless_slot);
    pc.epoch_offset = epoch_cursor;
    epoch_cursor += pool_caps[p];
  }

  std::uint64_t b4_weighted_tb = 0ull;
  std::uint64_t b4_weight_sum = 0ull;
  std::uint64_t f4_weighted_tb = 0ull;
  std::uint64_t f4_weight_sum = 0ull;
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    const auto &pc = _pools[p].config;
    if (p > 0u) {
      _stats.pool_summary += ",";
    }
    _stats.pool_summary += std::string(vt_tile_format_name(pc.format)) + ":" +
                           std::to_string(pc.capacity) + "/" +
                           std::to_string(pc.virtual_pages);
    if (pc.is_high_precision) {
      _stats.float4_virtual_pages += pc.virtual_pages;
      _stats.float4_cache_capacity += pc.capacity;
      f4_weighted_tb +=
          static_cast<std::uint64_t>(pc.capacity) * pc.tile_bytes;
      f4_weight_sum += pc.capacity;
    } else {
      _stats.byte4_virtual_pages += pc.virtual_pages;
      _stats.byte4_cache_capacity += pc.capacity;
      b4_weighted_tb +=
          static_cast<std::uint64_t>(pc.capacity) * pc.tile_bytes;
      b4_weight_sum += pc.capacity;
    }
  }
  _stats.byte4_tile_bytes =
      b4_weight_sum > 0ull
          ? static_cast<std::uint32_t>((b4_weighted_tb + b4_weight_sum / 2ull) /
                                       b4_weight_sum)
          : vt_tile_format_bytes(_physical_tile_size, VtTileFormat::rgba8);
  _stats.float4_tile_bytes =
      f4_weight_sum > 0ull
          ? static_cast<std::uint32_t>((f4_weighted_tb + f4_weight_sum / 2ull) /
                                       f4_weight_sum)
          : vt_tile_format_bytes(_physical_tile_size, VtTileFormat::float4);

  const auto heap_slots = std::max<std::uint32_t>(next_bindless_slot, 1u);
  _atlas_heap = _device.create_bindless_array(heap_slots);
  for (const auto &pr : _pools) {
    const auto &pc = pr.config;
    for (std::size_t a = 0u; a < pc.atlases.size(); ++a) {
      const auto slot_base =
          pc.bindless_base + static_cast<std::uint32_t>(a) * 2u;
      _atlas_heap.emplace_on_update(slot_base, pc.atlases[a],
                                    Sampler::linear_point_edge());
      _atlas_heap.emplace_on_update(slot_base + 1u, pc.atlases[a],
                                    Sampler::point_edge());
      _stats.gpu_allocated_atlas_bytes += luisa::compute::pixel_storage_size(
          pc.atlas_storage,
          luisa::make_uint3(pc.atlas_widths[a], pc.atlas_heights[a], 1u));
    }
  }

  const auto desc_count = std::max<std::size_t>(_descs.size(), 1u);
  std::vector<VtImageDescGpu> desc_upload(desc_count);
  if (!_descs.empty()) {
    std::copy(_descs.begin(), _descs.end(), desc_upload.begin());
  }
  const std::size_t cfg_count = std::max<std::size_t>(_pools.size(), 1u);
  _pool_configs = _device.create_buffer<VtPoolConfigGpu>(cfg_count);
  std::vector<VtPoolConfigGpu> pool_cfg_upload(cfg_count);
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    pool_cfg_upload[p] = make_pool_config_gpu(_pools[p].config);
  }
  _image_descs_buffer = _device.create_buffer<VtImageDescGpu>(desc_count);
  if (_mip_levels.empty()) {
    _mip_levels.push_back(VtMipLevelGpu{
        .width = 1u, .height = 1u, .tiles_x = 1u, .tiles_y = 1u,
        .page_table_offset = 0u, .page_count = 1u});
  }
  const std::size_t mip_level_count = _mip_levels.size();
  _mip_levels_buffer = _device.create_buffer<VtMipLevelGpu>(mip_level_count);

  const auto total_pages = std::max<std::size_t>(_pages.size(), 1u);
  const auto bitset_words = (total_pages + 31u) / 32u;
  const auto total_slots = std::max<std::uint32_t>(epoch_cursor, 1u);
  _page_table = _device.create_buffer<luisa::uint>(total_pages);
  _miss_bitset = _device.create_buffer<luisa::uint>(bitset_words);
  _slot_epoch = _device.create_buffer<luisa::uint>(total_slots);
  _current_epoch = _device.create_buffer<luisa::uint>(cfg_count);
  _demand_hist = _device.create_buffer<luisa::uint>(total_pages);
  _aux_demand_hist = _device.create_buffer<luisa::uint>(total_pages);

  _copy_queue_mode = _options.copy_queue_mode;
  _supports_stream_signal_event = supports_batch_copy;
  switch (_copy_queue_mode) {
    case LuisaVirtualTextureCopyQueueMode::off:
      break;
    case LuisaVirtualTextureCopyQueueMode::direct:
      if (!supports_batch_copy) {
        LUISA_WARNING(
            "VT copy_queue_mode=direct requires vk_batch_copy_cmd; falling back to off.");
        _copy_queue_mode = LuisaVirtualTextureCopyQueueMode::off;
      }
      break;
    case LuisaVirtualTextureCopyQueueMode::staged:
      if (_pinned_ext == nullptr) {
        LUISA_WARNING(
            "VT copy_queue_mode=staged requires PinnedMemoryExt; falling back to off.");
        _copy_queue_mode = LuisaVirtualTextureCopyQueueMode::off;
      } else if (!supports_batch_copy) {
        LUISA_WARNING(
            "VT copy_queue_mode=staged requires vk_batch_copy_cmd; falling back to staged_shader.");
        _copy_queue_mode = LuisaVirtualTextureCopyQueueMode::staged_shader;
      }
      break;
    case LuisaVirtualTextureCopyQueueMode::staged_shader:
      if (_pinned_ext == nullptr) {
        LUISA_WARNING(
            "VT copy_queue_mode=staged_shader requires PinnedMemoryExt; falling back to off.");
        _copy_queue_mode = LuisaVirtualTextureCopyQueueMode::off;
      }
      break;
  }
  _use_copy_stream =
      (_copy_queue_mode != LuisaVirtualTextureCopyQueueMode::off);
  _use_host_cached_staging =
      (_copy_queue_mode == LuisaVirtualTextureCopyQueueMode::staged ||
       _copy_queue_mode == LuisaVirtualTextureCopyQueueMode::staged_shader);
  _use_batch_texture_copy =
      (_copy_queue_mode == LuisaVirtualTextureCopyQueueMode::direct ||
       _copy_queue_mode == LuisaVirtualTextureCopyQueueMode::staged);
  for (auto &pr : _pools) {
    pr.config.use_batch_copy_upload =
        _use_batch_texture_copy ||
        (pr.config.format != VtTileFormat::rgba8 &&
         pr.config.format != VtTileFormat::float4);
  }
  if (_use_copy_stream) {
    _copy_stream = _device.create_stream(StreamTag::COPY);
    _copy_h2d_event = _device.create_event();
    _copy_acquire_event = _device.create_event();
    _copy_done_event = _device.create_event();
  }

  _async_batch_slots =
      _options.enable_async_worker
          ? std::clamp<std::uint32_t>(_options.async_batch_slots, 1u, 8u)
          : 1u;
  _stats.async_batch_slots = _async_batch_slots;
  _meta_ring_size = _options.enable_async_worker
                        ? static_cast<std::size_t>(_async_batch_slots) * 2u
                        : kMetaRingSize;
  for (auto &pr : _pools) {
    const std::size_t pool_upload_slots =
        _options.enable_async_worker
            ? static_cast<std::size_t>(_async_batch_slots) *
                  pr.config.chunks_per_batch_slot
            : kSyncUploadRingSize;
    pr.staging.resize(pool_upload_slots);
    if (_use_host_cached_staging) {
      pr.device_staging.resize(pool_upload_slots);
    }
    pr.upload_meta.resize(pool_upload_slots);
    pr.pinned_staging_ptr.assign(pool_upload_slots, nullptr);
    pr.pinned_upload_meta_ptr.assign(pool_upload_slots, nullptr);
    pr.slot_fence.assign(pool_upload_slots, 0ull);
    pr.slot_stream.assign(pool_upload_slots, 0ull);
    pr.slot_async_fence.assign(pool_upload_slots, 0ull);
  }
  _page_table_updates.resize(_meta_ring_size);
  _slot_epoch_updates.resize(_meta_ring_size);
  _pinned_page_table_updates_ptr.assign(_meta_ring_size, nullptr);
  _pinned_slot_epoch_updates_ptr.assign(_meta_ring_size, nullptr);
  _meta_slot_fence.assign(_meta_ring_size, 0ull);
  _meta_slot_stream.assign(_meta_ring_size, 0ull);
  _meta_slot_async_fence.assign(_meta_ring_size, 0ull);

  std::size_t max_meta_cap = 1u;
  for (const auto &pr : _pools) {
    const std::size_t mt_cap =
        std::max<std::size_t>(pr.config.staging_capacity, 1u);
    max_meta_cap = std::max(max_meta_cap, mt_cap);
  }
  const std::size_t pt_update_cap =
      std::max<std::size_t>(2u * total_slots, 1u);

  if (_pinned_ext != nullptr) {
    bool use_wb_cached = false;
    if (const char *env = std::getenv("PSYCLES_VT_STAGING_HOST_CACHED")) {
      use_wb_cached = (std::strcmp(env, "1") == 0);
    }
    const PinnedMemoryOption staging_opt =
        _use_host_cached_staging
            ? (use_wb_cached
                   ? PinnedMemoryOption{.write_combined = false,
                                        .host_cached_upload = true}
                   : PinnedMemoryOption{.write_combined = true,
                                        .prefer_host = true})
            : PinnedMemoryOption{.write_combined = true};
    bool upload_ok = true;
    for (auto &pr : _pools) {
      const std::size_t st_len = std::max<std::size_t>(
          static_cast<std::size_t>(pr.config.staging_capacity) *
              pr.config.tile_staging_u32s,
          1u);
      const std::size_t mt_cap =
          std::max<std::size_t>(pr.config.staging_capacity, 1u);
      for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
        auto st = _pinned_ext->allocate_pinned_memory<luisa::uint>(
            st_len, staging_opt);
        auto um = _pinned_ext->allocate_pinned_memory<luisa::uint4>(
            mt_cap, PinnedMemoryOption{.write_combined = true});
        auto *p_st = static_cast<luisa::uint *>(st.native_handle());
        auto *p_um = static_cast<luisa::uint4 *>(um.native_handle());
        if (!st || !um || p_st == nullptr || p_um == nullptr) {
          upload_ok = false;
          break;
        }
        pr.staging[i] = std::move(st);
        pr.upload_meta[i] = std::move(um);
        pr.pinned_staging_ptr[i] = p_st;
        pr.pinned_upload_meta_ptr[i] = p_um;
      }
      if (!upload_ok) {
        break;
      }
    }
    if (upload_ok) {
      for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
        auto ptu = _pinned_ext->allocate_pinned_memory<luisa::uint2>(
            pt_update_cap, PinnedMemoryOption{.write_combined = true});
        auto seu = _pinned_ext->allocate_pinned_memory<luisa::uint2>(
            total_slots, PinnedMemoryOption{.write_combined = true});
        auto *p_ptu = static_cast<luisa::uint2 *>(ptu.native_handle());
        auto *p_seu = static_cast<luisa::uint2 *>(seu.native_handle());
        if (!ptu || !seu || p_ptu == nullptr || p_seu == nullptr) {
          upload_ok = false;
          break;
        }
        _page_table_updates[i] = std::move(ptu);
        _slot_epoch_updates[i] = std::move(seu);
        _pinned_page_table_updates_ptr[i] = p_ptu;
        _pinned_slot_epoch_updates_ptr[i] = p_seu;
      }
    }
    if (upload_ok) {
      _use_pinned_upload = true;
    } else {
      for (auto &pr : _pools) {
        for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
          pr.staging[i] = {};
          pr.upload_meta[i] = {};
          pr.pinned_staging_ptr[i] = nullptr;
          pr.pinned_upload_meta_ptr[i] = nullptr;
        }
      }
      for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
        _page_table_updates[i] = {};
        _slot_epoch_updates[i] = {};
        _pinned_page_table_updates_ptr[i] = nullptr;
        _pinned_slot_epoch_updates_ptr[i] = nullptr;
      }
    }

    auto rb_dh = _pinned_ext->allocate_pinned_memory<luisa::uint>(
        total_pages, PinnedMemoryOption{.write_combined = false});
    auto rb_adh = _pinned_ext->allocate_pinned_memory<luisa::uint>(
        total_pages, PinnedMemoryOption{.write_combined = false});
    auto rb_mb = _pinned_ext->allocate_pinned_memory<luisa::uint>(
        bitset_words, PinnedMemoryOption{.write_combined = false});
    auto rb_se = _pinned_ext->allocate_pinned_memory<luisa::uint>(
        total_slots, PinnedMemoryOption{.write_combined = false});
    auto *p_dh = static_cast<luisa::uint *>(rb_dh.native_handle());
    auto *p_adh = static_cast<luisa::uint *>(rb_adh.native_handle());
    auto *p_mb = static_cast<luisa::uint *>(rb_mb.native_handle());
    auto *p_se = static_cast<luisa::uint *>(rb_se.native_handle());
    if (rb_dh && rb_adh && rb_mb && rb_se && p_dh != nullptr &&
        p_adh != nullptr && p_mb != nullptr && p_se != nullptr) {
      _readback_demand_hist = std::move(rb_dh);
      _readback_aux_demand_hist = std::move(rb_adh);
      _readback_miss_bitset = std::move(rb_mb);
      _readback_slot_epoch = std::move(rb_se);
      _pinned_demand_hist_ptr = p_dh;
      _pinned_aux_demand_hist_ptr = p_adh;
      _pinned_miss_bitset_ptr = p_mb;
      _pinned_slot_epoch_ptr = p_se;
      _use_pinned_readback = true;
    }
  }

  if (!_use_pinned_upload) {
    if (_use_host_cached_staging) {
      _use_host_cached_staging = false;
      for (auto &pr : _pools) {
        pr.device_staging.clear();
      }
      _copy_queue_mode = _use_batch_texture_copy
                             ? LuisaVirtualTextureCopyQueueMode::direct
                             : LuisaVirtualTextureCopyQueueMode::off;
      _use_copy_stream =
          (_copy_queue_mode != LuisaVirtualTextureCopyQueueMode::off);
    }
    for (auto &pr : _pools) {
      const std::size_t st_len = std::max<std::size_t>(
          static_cast<std::size_t>(pr.config.staging_capacity) *
              pr.config.tile_staging_u32s,
          1u);
      const std::size_t mt_cap =
          std::max<std::size_t>(pr.config.staging_capacity, 1u);
      for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
        pr.staging[i] = _device.create_buffer<luisa::uint>(st_len);
        pr.upload_meta[i] = _device.create_buffer<luisa::uint4>(mt_cap);
      }
    }
    for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
      _page_table_updates[i] =
          _device.create_buffer<luisa::uint2>(pt_update_cap);
      _slot_epoch_updates[i] = _device.create_buffer<luisa::uint2>(total_slots);
    }
  }
  if (_use_host_cached_staging) {
    for (auto &pr : _pools) {
      const std::size_t st_len = std::max<std::size_t>(
          static_cast<std::size_t>(pr.config.staging_capacity) *
              pr.config.tile_staging_u32s,
          1u);
      for (std::size_t i = 0u; i < pr.device_staging.size(); ++i) {
        pr.device_staging[i] = _device.create_buffer<luisa::uint>(st_len);
      }
    }
  }
  _stats.active_copy_queue_mode = _copy_queue_mode;

  if (_options.enable_async_worker) {
    _async_slots.resize(_async_batch_slots);
    for (auto &slot : _async_slots) {
      bool pinned_ok = false;
      if (_use_pinned_readback && _pinned_ext != nullptr) {
        auto s_dh = _pinned_ext->allocate_pinned_memory<luisa::uint>(
            total_pages, PinnedMemoryOption{.write_combined = false});
        auto s_mb = _pinned_ext->allocate_pinned_memory<luisa::uint>(
            bitset_words, PinnedMemoryOption{.write_combined = false});
        auto s_se = _pinned_ext->allocate_pinned_memory<luisa::uint>(
            total_slots, PinnedMemoryOption{.write_combined = false});
        auto *pdh = static_cast<luisa::uint *>(s_dh.native_handle());
        auto *pmb = static_cast<luisa::uint *>(s_mb.native_handle());
        auto *pse = static_cast<luisa::uint *>(s_se.native_handle());
        if (s_dh && s_mb && s_se && pdh != nullptr && pmb != nullptr &&
            pse != nullptr) {
          slot.demand_hist = std::move(s_dh);
          slot.miss_bitset = std::move(s_mb);
          slot.slot_epoch = std::move(s_se);
          slot.pinned_demand_hist_ptr = pdh;
          slot.pinned_miss_bitset_ptr = pmb;
          slot.pinned_slot_epoch_ptr = pse;
          pinned_ok = true;
        }
      }
      if (!pinned_ok) {
        slot.fallback_demand_hist.assign(total_pages, 0u);
        slot.fallback_miss_bitset.assign(bitset_words, 0u);
        slot.fallback_slot_epoch.assign(total_slots, 0u);
      }
    }
  }

  _page_evict_owner_mask.assign(total_pages, 0u);
  _page_load_inflight_slot.assign(total_pages, PoolRuntime::kInvalidAsyncSlot);
  _host_page_table.assign(total_pages, 0u);
  _host_demand_hist.assign(total_pages, 0u);
  _host_aux_demand_hist.assign(total_pages, 0u);
  _host_miss_bitset.assign(bitset_words, 0u);
  _host_slot_epoch.assign(total_slots, 0u);
  _host_current_epoch.assign(cfg_count, 0u);
  std::uint64_t host_sync_staging_bytes = 0ull;
  std::uint64_t upload_staging_bytes_total = 0ull;
  std::uint64_t upload_meta_bytes_total = 0ull;
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    auto &pr = _pools[p];
    pr.host_slots.assign(pr.config.capacity, vt_invalid_page);
    pr.host_slot_meta.assign(pr.config.capacity, PoolSlotMeta{});
    pr.slot_inflight_owner.assign(pr.config.capacity, PoolRuntime::kInvalidAsyncSlot);
    pr.slot_pin_consumer_seq.assign(pr.config.capacity, 0ull);
    pr.slot_pin_aux_seq.assign(pr.config.capacity, 0ull);
    pr.slot_pin_by_aux.assign(pr.config.capacity, 0u);
    pr.slot_publish_epoch.assign(pr.config.capacity, 0ull);
    pr.last_readback_gpu_epoch = 0u;
    _host_current_epoch[p] = pr.config.eviction_possible ? 2u : 0u;
    const std::size_t st_len = std::max<std::size_t>(
        static_cast<std::size_t>(pr.config.staging_capacity) *
            pr.config.tile_staging_u32s,
        1u);
    const std::size_t mt_cap =
        std::max<std::size_t>(pr.config.staging_capacity, 1u);
    upload_staging_bytes_total +=
        static_cast<std::uint64_t>(pr.staging.size()) * st_len * sizeof(luisa::uint);
    upload_meta_bytes_total +=
        static_cast<std::uint64_t>(pr.upload_meta.size()) * mt_cap * sizeof(luisa::uint4);
    if (!_use_pinned_upload) {
      pr.host_staging.assign(st_len, 0u);
      host_sync_staging_bytes += static_cast<std::uint64_t>(st_len) * sizeof(luisa::uint);
    }
  }
  _published_gpu_epoch = _host_current_epoch;
  if (!_use_pinned_upload) {
    _host_upload_meta.resize(max_meta_cap);
    _host_page_table_updates.reserve(pt_update_cap);
    _host_slot_epoch_updates.reserve(total_slots);
  }
  const std::uint64_t meta_upd_bytes =
      static_cast<std::uint64_t>(_meta_ring_size) *
      (pt_update_cap + total_slots) * sizeof(luisa::uint2);
  const std::uint64_t resident_dev_meta_bytes =
      cfg_count * sizeof(VtPoolConfigGpu) +
      desc_count * sizeof(VtImageDescGpu) +
      mip_level_count * sizeof(VtMipLevelGpu) +
      (total_pages * 3u + bitset_words + total_slots + cfg_count) *
          sizeof(luisa::uint);
  const std::uint64_t sync_rb_bytes =
      (total_pages * 2u + bitset_words + total_slots) * sizeof(luisa::uint);
  std::uint64_t async_rb_pinned_bytes = 0ull;
  std::uint64_t async_rb_host_bytes = 0ull;
  for (const auto &slot : _async_slots) {
    const std::uint64_t b =
        (total_pages + bitset_words + total_slots) * sizeof(luisa::uint);
    if (slot.pinned_demand_hist_ptr != nullptr) {
      async_rb_pinned_bytes += b;
    } else {
      async_rb_host_bytes += b;
    }
  }
  _stats.gpu_allocated_metadata_bytes =
      resident_dev_meta_bytes + upload_staging_bytes_total +
      upload_meta_bytes_total + meta_upd_bytes +
      (_use_pinned_readback ? sync_rb_bytes : 0ull) +
      (_use_host_cached_staging ? upload_staging_bytes_total : 0ull) +
      async_rb_pinned_bytes;
  _stats.device_allocated_vt_bytes =
      _stats.gpu_allocated_atlas_bytes + resident_dev_meta_bytes +
      (_use_pinned_upload
           ? 0ull
           : (upload_staging_bytes_total + upload_meta_bytes_total + meta_upd_bytes)) +
      (_use_host_cached_staging ? upload_staging_bytes_total : 0ull);
  _stats.host_allocated_vt_bytes =
      (_use_pinned_upload
           ? (upload_staging_bytes_total + upload_meta_bytes_total + meta_upd_bytes)
           : (host_sync_staging_bytes + max_meta_cap * sizeof(luisa::uint4) +
              (pt_update_cap + total_slots) * sizeof(luisa::uint2))) +
      (_use_pinned_readback ? sync_rb_bytes : 0ull) +
      async_rb_pinned_bytes + async_rb_host_bytes +
      static_cast<std::uint64_t>(total_pages * 3u + bitset_words + total_slots) *
          sizeof(luisa::uint) +
      static_cast<std::uint64_t>(_mip_levels.size()) * sizeof(VtMipLevelGpu) +
      static_cast<std::uint64_t>(_page_ever_loaded.size()) +
      static_cast<std::uint64_t>(_pages.size()) *
          (sizeof(VirtualPageMeta) + sizeof(VtPrepackedPageKey) +
           sizeof(std::uint32_t) * 2u) +
      static_cast<std::uint64_t>(total_slots) *
          (sizeof(luisa::uint) + sizeof(PoolSlotMeta) + sizeof(std::uint32_t) +
           sizeof(std::uint64_t) * 2u + sizeof(std::uint8_t));

  stream << _atlas_heap.update()
         << _pool_configs.copy_from(luisa::span{pool_cfg_upload})
         << _image_descs_buffer.copy_from(luisa::span{desc_upload})
         << _mip_levels_buffer.copy_from(luisa::span{_mip_levels})
         << _page_table.copy_from(luisa::span{_host_page_table})
         << _miss_bitset.copy_from(luisa::span{_host_miss_bitset})
         << _slot_epoch.copy_from(luisa::span{_host_slot_epoch})
         << _current_epoch.copy_from(luisa::span{_host_current_epoch})
         << _demand_hist.copy_from(luisa::span{_host_demand_hist})
         << _aux_demand_hist.copy_from(luisa::span{_host_aux_demand_hist});

  _clear_u32_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> buf, UInt count) noexcept {
        const auto x = dispatch_x();
        $if(x < count) { buf.write(x, 0u); };
      });

  bool need_b4_shader = false;
  bool need_f4_shader = false;
  for (const auto &pr : _pools) {
    if (!pr.config.use_batch_copy_upload) {
      if (pr.config.format == VtTileFormat::rgba8) {
        need_b4_shader = true;
      } else if (pr.config.format == VtTileFormat::float4) {
        need_f4_shader = true;
      }
    }
  }

  if (need_b4_shader) {
    _upload_byte4_shader = _device.compile<3>(
        [](ImageVar<float> atlas, BufferVar<luisa::uint> staging,
           BufferVar<luisa::uint4> meta, UInt meta_offset,
           UInt tile_size) noexcept {
          const auto coord = dispatch_id();
          const auto m = meta.read(meta_offset + coord.z);
          const auto texel_idx =
              (m.x * tile_size + coord.y) * tile_size + coord.x;
          const auto packed = staging.read(texel_idx);
          constexpr float inv255 = 1.0f / 255.0f;
          const auto rgba = make_float4(
              cast<float>(packed & 0xffu) * inv255,
              cast<float>((packed >> 8u) & 0xffu) * inv255,
              cast<float>((packed >> 16u) & 0xffu) * inv255,
              cast<float>((packed >> 24u) & 0xffu) * inv255);
          atlas.write(make_uint2(m.y + coord.x, m.z + coord.y), rgba);
        });
  }

  if (need_f4_shader) {
    _upload_float4_shader = _device.compile<3>(
        [](ImageVar<float> atlas, BufferVar<luisa::uint> staging,
           BufferVar<luisa::uint4> meta, UInt meta_offset,
           UInt tile_size) noexcept {
          const auto coord = dispatch_id();
          const auto m = meta.read(meta_offset + coord.z);
          const auto texel_idx =
              (m.x * tile_size + coord.y) * tile_size + coord.x;
          const auto word_base = texel_idx * 4u;
          const auto rgba = make_float4(
              staging.read(word_base + 0u).as<float>(),
              staging.read(word_base + 1u).as<float>(),
              staging.read(word_base + 2u).as<float>(),
              staging.read(word_base + 3u).as<float>());
          atlas.write(make_uint2(m.y + coord.x, m.z + coord.y), rgba);
        });
  }

  _apply_page_table_updates_shader = _device.compile<1>(
      [](BufferVar<luisa::uint> page_table,
         BufferVar<luisa::uint2> updates, UInt count) noexcept {
        const auto x = dispatch_x();
        $if(x < count) {
          const auto u = updates.read(x);
          page_table.write(u.x, u.y);
        };
      });

  if (!_pages.empty()) {
    _sample_linear_repeat.emplace(record_linear_repeat_sampler(
        _physical_tile_size, _tile_border, _logical_tile_size,
        "vt_sample_linear_repeat"));
    _sample_linear_any_ext.emplace(record_linear_any_ext_sampler(
        _physical_tile_size, _tile_border, _logical_tile_size,
        "vt_sample_linear_any_ext"));
    _sample_general.emplace(record_general_sampler(
        _physical_tile_size, _tile_border, _logical_tile_size,
        "vt_sample_general"));
    _preflight_demand_callable.emplace(record_preflight_demand_callable(
        _logical_tile_size, "vt_preflight_demand"));
    if (_options.mip_mode == LuisaTextureMipMode::isotropic) {
      _sample_mip_linear_repeat.emplace(record_mip_sampler(
          _physical_tile_size, _tile_border, _logical_tile_size, true,
          "vt_sample_mip_linear_repeat"));
      _sample_mip_general.emplace(record_mip_sampler(
          _physical_tile_size, _tile_border, _logical_tile_size, false,
          "vt_sample_mip_general"));
      _preflight_mip_demand_callable.emplace(
          record_preflight_mip_demand_callable(
              _logical_tile_size, "vt_preflight_mip_demand"));
    }
  }

  if (!_pages.empty()) {
    const auto gpu_bc_quality = [&]() noexcept -> VtGpuBcQuality {
      switch (_options.compression_quality) {
        case LuisaVirtualTextureCompressionQuality::fast:
          return VtGpuBcQuality::fast;
        case LuisaVirtualTextureCompressionQuality::high:
          return VtGpuBcQuality::high;
        case LuisaVirtualTextureCompressionQuality::balanced:
        default:
          return VtGpuBcQuality::balanced;
      }
    }();
    _gpu_bc = std::make_unique<VtGpuBcCompressor>(_device, gpu_bc_quality);
    _stats.gpu_bc_backend = _gpu_bc->backend_name();
    std::vector<VtTileFormat> pool_formats;
    pool_formats.reserve(_pools.size());
    for (const auto &pr : _pools) {
      if (VtGpuBcCompressor::supports_format(pr.config.format)) {
        pool_formats.push_back(pr.config.format);
      }
    }
    _gpu_bc->warmup(pool_formats);
  }

  if (_options.prepack_mode != LuisaVirtualTexturePrepackMode::off) {
    _prepack_cache = std::make_unique<VtPrepackedPageCache>(
        _options.host_prepacked_cache_budget_bytes);
    if (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk &&
        !_options.prepack_disk_dir.empty()) {
      _prepack_disk = std::make_unique<VtPrepackedDiskStore>(
          _options.prepack_disk_dir);
    }
    if (_options.prepack_warm_on_init && !_pages.empty()) {
      warm_prepack_on_init();
    }
  }

  _profiler = VtProfileRecorder::create_from_env();
  register_timeline_labels_if_active();
  stream << synchronize();
}

void VirtualTextureCache::sync_prepack_stats() noexcept {
  if (_prepack_cache) {
    const auto s = _prepack_cache->stats();
    _stats.prepack_hits = s.hits;
    _stats.prepack_misses = s.misses;
    _stats.prepack_evictions = s.evictions;
    _stats.prepack_resident_bytes = s.resident_bytes;
    _stats.peak_prepack_resident_bytes =
        std::max(_stats.peak_prepack_resident_bytes, s.resident_bytes);
  }
}

void VirtualTextureCache::warm_prepack_on_init() noexcept {
  if (!_prepack_cache || _pages.empty()) {
    return;
  }
  const std::uint32_t P = _physical_tile_size;
  const std::uint64_t saved_decode_evictions = _stats.host_decode_evictions;

  if (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk &&
      _prepack_disk && _prepack_disk->enabled()) {
    std::vector<std::size_t> img_first_pages;
    VtPrepackedImageKey prev_img{};
    bool has_prev = false;
    std::vector<std::uint32_t> sources_to_decode;
    for (std::size_t pid = 0u; pid < _pages.size(); ++pid) {
      const auto &img_key = _page_prepack_keys[pid].image;
      if (has_prev && img_key == prev_img) {
        continue;
      }
      has_prev = true;
      prev_img = img_key;
      img_first_pages.push_back(pid);
      const auto &pm = _pages[pid];
      if (!_prepack_disk->has_valid_file(img_key, pm.tiles_x, pm.tiles_y)) {
        const auto &src = _sources[pm.source_index];
        if (!src.decoded && !src.load_failed &&
            std::find(sources_to_decode.begin(), sources_to_decode.end(),
                      pm.source_index) == sources_to_decode.end()) {
          sources_to_decode.push_back(pm.source_index);
        }
      }
    }
    if (!sources_to_decode.empty()) {
      decode_sources_batch(sources_to_decode);
    }
    for (const auto pid : img_first_pages) {
      const auto &pm = _pages[pid];
      const auto &img_key = _page_prepack_keys[pid].image;
      if (!_prepack_disk->has_valid_file(img_key, pm.tiles_x, pm.tiles_y)) {
        const auto &src = _sources[pm.source_index];
        if (src.decoded && !src.load_failed) {
          const auto mv = source_mip_view(src, pm.mip_level);
          if (_prepack_disk->ensure_image_file(
                  img_key, mv.unorm8_pixels, mv.float_pixels,
                  pm.tiles_x, pm.tiles_y, _gpu_bc.get())) {
            ++_stats.prepack_disk_writes;
          }
        }
      }
    }
  } else {
    std::vector<std::uint32_t> all_virtual_sources;
    for (std::uint32_t si = 0u; si < _sources.size(); ++si) {
      const auto &src = _sources[si];
      if (!src.load_failed && !src.force_direct && !src.decoded) {
        all_virtual_sources.push_back(si);
      }
    }
    if (!all_virtual_sources.empty()) {
      decode_sources_batch(all_virtual_sources);
    }
  }

  // Pack warm tiles up to host_prepacked_cache_budget_bytes across virtual pages.
  const std::uint64_t budget = _options.host_prepacked_cache_budget_bytes;
  if (budget > 0ull) {
    std::uint64_t packed_bytes = 0ull;
    constexpr std::size_t kWarmBatch = 256u;
    std::size_t max_tb = 1u;
    for (const auto &pr : _pools) {
      max_tb = std::max<std::size_t>(max_tb, pr.config.tile_bytes);
    }
    std::vector<std::uint8_t> scratch;
    for (std::size_t base = 0u; base < _pages.size() && packed_bytes < budget;
         base += kWarmBatch) {
      const std::size_t count = std::min(kWarmBatch, _pages.size() - base);
      scratch.resize(count * max_tb);
      psycles::parallel_for(count, [&](std::size_t off) {
        const std::size_t pid = base + off;
        const auto &pm = _pages[pid];
        const auto &src = _sources[pm.source_index];
        std::uint8_t *dst = scratch.data() + off * max_tb;
        const std::size_t tb = _pools[pm.pool_index].config.tile_bytes;
        if (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk &&
            _prepack_disk &&
            _prepack_disk->has_valid_file(_page_prepack_keys[pid].image,
                                          pm.tiles_x, pm.tiles_y)) {
          if (_prepack_disk->read_single_page(
                  _page_prepack_keys[pid], pm.tiles_x, pm.tiles_y,
                  std::span<std::uint8_t>{dst, tb})) {
            return;
          }
        }
        const auto mv = source_mip_view(src, pm.mip_level);
        vt_pack_tile_formatted(
            mv.unorm8_pixels, mv.float_pixels, mv.width, mv.height,
            pm.tile_x, pm.tile_y, pm.extension, P, _logical_tile_size,
            _tile_border, pm.format, std::span<std::uint8_t>{dst, tb});
      });
      for (std::size_t off = 0u; off < count; ++off) {
        const std::size_t pid = base + off;
        const std::size_t tb =
            _pools[_pages[pid].pool_index].config.tile_bytes;
        if (packed_bytes + tb > budget) {
          break;
        }
        const std::uint8_t *dst = scratch.data() + off * max_tb;
        if (_prepack_cache->insert_warm_tile(
                _page_prepack_keys[pid],
                std::span<const std::uint8_t>{dst, tb}, false)) {
          packed_bytes += tb;
        } else {
          packed_bytes = budget;
          break;
        }
      }
    }
  }
  // Warm-init cost belongs to initialization rather than render integration time.
  _stats.decode_ms = 0.0;
  _stats.prepack_disk_ms = 0.0;
  if (_gpu_bc) {
    _gpu_bc->reset_stats();
  }
  _stats.host_decode_evictions = saved_decode_evictions;
  sync_prepack_stats();
}

bool VirtualTextureCache::is_virtual_binding(
    std::int32_t binding_id) const noexcept {
  if (binding_id < 0 ||
      static_cast<std::size_t>(binding_id) >= _descs.size()) {
    return false;
  }
  return (_descs[static_cast<std::size_t>(binding_id)].flags &
          vt_desc_flag_virtual) != 0u;
}

// VirtualTextureCache::stats_json() is implemented in virtual_texture_cache_mipmap.cpp.

void VirtualTextureCache::dump_profile_if_enabled() const noexcept {
  if (_profiler) {
    _profiler->write_json();
  }
}

std::uint64_t VirtualTextureCache::query_vk_timeline_submit_seq() const noexcept {
  if (!_device) {
    return 0ull;
  }
  const auto res = _device.impl()->query("vk_timeline_submit_seq");
  if (res.empty()) {
    return 0ull;
  }
  return static_cast<std::uint64_t>(std::strtoull(res.c_str(), nullptr, 10));
}

void VirtualTextureCache::register_timeline_labels_if_active() const noexcept {
  if (_timeline_labels_registered || !_device) {
    return;
  }
  if (_device.impl()->query("vk_timeline_active") != "1") {
    return;
  }
  _timeline_labels_registered = true;
  auto lbl = [&](const char *kind, std::uint64_t handle, const char *name) noexcept {
    if (handle == 0ull || handle == luisa::compute::invalid_resource_handle) {
      return;
    }
    char qbuf[192];
    std::snprintf(qbuf, sizeof(qbuf), "vk_timeline_label:%s:%llu:%s", kind,
                  static_cast<unsigned long long>(handle), name);
    static_cast<void>(_device.impl()->query(qbuf));
  };
  if (_clear_u32_shader) { lbl("shader", _clear_u32_shader.handle(), "vt_clear_u32"); }
  if (_upload_byte4_shader) { lbl("shader", _upload_byte4_shader.handle(), "vt_upload_byte4"); }
  if (_upload_float4_shader) { lbl("shader", _upload_float4_shader.handle(), "vt_upload_float4"); }
  if (_apply_page_table_updates_shader) {
    lbl("shader", _apply_page_table_updates_shader.handle(), "vt_apply_pt_or_epoch");
  }
  lbl("buffer", _demand_hist.handle(), "vt_demand_hist");
  lbl("buffer", _aux_demand_hist.handle(), "vt_aux_demand_hist");
  lbl("buffer", _miss_bitset.handle(), "vt_miss_bitset");
  lbl("buffer", _slot_epoch.handle(), "vt_slot_epoch");
  lbl("buffer", _current_epoch.handle(), "vt_current_epoch");
  lbl("buffer", _page_table.handle(), "vt_page_table");
  lbl("buffer", _pool_configs.handle(), "vt_pool_configs");
  lbl("buffer", _image_descs_buffer.handle(), "vt_image_descs");
  if (_use_pinned_readback) {
    lbl("buffer", _readback_demand_hist.handle(), "vt_readback_demand_hist");
    lbl("buffer", _readback_aux_demand_hist.handle(), "vt_readback_aux_demand_hist");
    lbl("buffer", _readback_miss_bitset.handle(), "vt_readback_miss_bitset");
    lbl("buffer", _readback_slot_epoch.handle(), "vt_readback_slot_epoch");
  }
  for (const auto &pr : _pools) {
    for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
      lbl("buffer", pr.staging[i].handle(), "vt_staging");
      lbl("buffer", pr.upload_meta[i].handle(), "vt_upload_meta");
      if (_use_host_cached_staging && i < pr.device_staging.size()) {
        lbl("buffer", pr.device_staging[i].handle(), "vt_dev_staging");
      }
    }
    for (const auto &atlas : pr.config.atlases) {
      lbl("texture", atlas.handle(),
          pr.config.is_high_precision ? "vt_atlas_hp" : "vt_atlas_lp");
    }
  }
  for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
    lbl("buffer", _page_table_updates[i].handle(), "vt_page_table_updates");
    lbl("buffer", _slot_epoch_updates[i].handle(), "vt_slot_epoch_updates");
  }
  for (const auto &slot : _async_slots) {
    if (slot.pinned_demand_hist_ptr != nullptr) {
      lbl("buffer", slot.demand_hist.handle(), "vt_async_rb_demand");
      lbl("buffer", slot.miss_bitset.handle(), "vt_async_rb_bitset");
      lbl("buffer", slot.slot_epoch.handle(), "vt_async_rb_epoch");
    }
  }
}

namespace {

bool VtWavefrontExtensionHandler::submit_async(
    const coro::WavefrontCoroExtensionDispatchContext &ctx) noexcept {
  return _cache && _cache->submit_async_wavefront_collect(ctx, collect_shader());
}

void VtWavefrontExtensionHandler::on_async_scheduler_progress(
    const coro::WavefrontCoroSchedulerSnapshot &snapshot,
    bool waiting_for_progress) noexcept {
  if (_cache) {
    _cache->notify_async_scheduler_progress(snapshot, waiting_for_progress);
  }
}

coro::WavefrontCoroAsyncPollResult
VtWavefrontExtensionHandler::poll_async() noexcept {
  return _cache ? _cache->poll_async_wavefront()
                : coro::WavefrontCoroAsyncPollResult::idle;
}

bool VtWavefrontExtensionHandler::poll_async_ready(
    coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
  return _cache && _cache->poll_async_wavefront_ready(out_ticket);
}

void VtWavefrontExtensionHandler::wait_async() noexcept {
  if (_cache) {
    _cache->wait_async_wavefront();
  }
}

bool VtWavefrontExtensionHandler::wait_async_any(
    coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
  return _cache && _cache->wait_async_wavefront_any(out_ticket);
}

void VtWavefrontExtensionHandler::publish_async(
    const coro::WavefrontCoroAsyncPublishContext &ctx) noexcept {
  if (_cache) {
    _cache->publish_async_wavefront(ctx);
  }
}

void VtWavefrontExtensionHandler::dispatch(
    const coro::WavefrontCoroExtensionDispatchContext &ctx) noexcept {
  if (_cache) {
    _cache->dispatch_wavefront_collect(ctx, collect_shader());
  }
}

} // namespace

void VirtualTextureCache::encode_wavefront_filter_ready(
    const coro::WavefrontCoroAdvanceFilterContext &ctx,
    const FilterReadyShader &filter_shader) noexcept {
  if (!has_virtual_images() || ctx.frame_count == 0u) {
    return;
  }
  ctx.command_list
      << filter_shader(ctx.frame_buffer, ctx.frame_indices, _page_table,
                       ctx.ready_flags, ctx.frame_capacity, ctx.frame_count)
             .dispatch(ctx.frame_count);
}

void VirtualTextureCache::dispatch_wavefront_collect(
    const coro::WavefrontCoroExtensionDispatchContext &ctx,
    const CollectShader &collect_shader) noexcept {
  if (!has_virtual_images() || ctx.frame_count == 0u) {
    return;
  }
  const std::uint64_t enter_ns =
      _profiler ? VtProfileRecorder::steady_ns() : 0ull;
  CommandList fallback_collect_cmd;
  CommandList *active_cmd =
      ctx.command_list != nullptr ? ctx.command_list : &fallback_collect_cmd;
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  *active_cmd
      << _clear_u32_shader(_demand_hist, total_pages).dispatch(total_pages)
      << collect_shader(ctx.frame_buffer, ctx.frame_indices, _demand_hist,
                        ctx.frame_capacity, ctx.frame_count)
             .dispatch(ctx.frame_count);
  if (ctx.command_list == nullptr) {
    ctx.stream << fallback_collect_cmd.commit();
  }
  if (_profiler) {
    const ServiceProfileContext prof_ctx{
        .collect_enter_ns = enter_ns,
        .queued_fault_frames = ctx.frame_count,
        .snapshot = ctx.scheduler_snapshot};
    service_misses_impl(ctx.stream, MissSource::wavefront_demand,
                        ctx.command_list, &prof_ctx);
  } else {
    service_misses_impl(ctx.stream, MissSource::wavefront_demand,
                        ctx.command_list, nullptr);
  }
}

luisa::move_only_function<
    luisa::unique_ptr<coro::WavefrontCoroSchedulerExtensionHandler>(
        coro::WavefrontCoroExtensionPrepareContext &,
        const coro::WavefrontCoroExtensionStage &)>
VirtualTextureCache::make_wavefront_extension_handler() noexcept {
  auto self = shared_from_this();
  return [self = std::move(self)](
             coro::WavefrontCoroExtensionPrepareContext &ctx,
             const coro::WavefrontCoroExtensionStage &stage)
             -> luisa::unique_ptr<coro::WavefrontCoroSchedulerExtensionHandler> {
    if (!stage.extension ||
        stage.extension->schema() != vt_extension_schema) {
      return nullptr;
    }
    const auto &page_binding = stage.binding(vt_extension_binding);
    const auto pieces = page_binding.pieces();
    const bool single_field =
        (pieces.size() == 1u) && (pieces.front().bit_offset == 0u) &&
        (pieces.front().logical_type == Type::of<luisa::uint>()) &&
        (pieces.front().physical_type == Type::of<luisa::uint>());
    const std::uint32_t field_index =
        single_field ? static_cast<std::uint32_t>(pieces.front().field_index)
                     : 0u;
    const auto fields = page_binding.reconstruct_slots();
    const auto &layout = ctx.frame_layout;

    std::uint64_t hash = 14695981039346656037ull;
    const auto mix_u64 = [&](std::uint64_t v) noexcept {
      hash ^= v;
      hash *= 1099511628211ull;
    };
    mix_u64(layout.has_runtime_capacity() ? 1ull : 0ull);
    if (single_field) {
      mix_u64(layout.field_offsets[field_index]);
      mix_u64(layout.field_strides[field_index]);
      mix_u64(field_index < layout.field_capacity_strides.size()
                  ? layout.field_capacity_strides[field_index]
                  : 0ull);
    } else {
      for (const auto f : fields) {
        mix_u64(f);
        mix_u64(f < layout.field_offsets.size() ? layout.field_offsets[f] : 0ull);
        mix_u64(f < layout.field_strides.size() ? layout.field_strides[f] : 0ull);
        mix_u64(f < layout.field_capacity_strides.size()
                    ? layout.field_capacity_strides[f]
                    : 0ull);
      }
    }

    const CollectShaderKey key{
        .soa = ctx.global_memory_soa,
        .single_field = single_field,
        .field_index = single_field ? field_index : 0u,
        .layout_hash = hash,
        .frame_desc = single_field ? nullptr : &ctx.frame_desc,
        .boundary_index =
            (single_field || stage.boundary == nullptr)
                ? 0u
                : static_cast<luisa::uint>(stage.boundary->index)};

    const CollectShader *cached_ptr = nullptr;
    const FilterReadyShader *cached_filter_ptr = nullptr;
    for (const auto &entry : self->_collect_shader_cache) {
      if (entry.key == key) {
        cached_ptr = &entry.shader;
        cached_filter_ptr = &entry.filter_shader;
        break;
      }
    }

    if (cached_ptr == nullptr) {
      const auto total_pages = static_cast<luisa::uint>(self->_pages.size());
      const coro::CoroSlotAccess *binding_ptr = &page_binding;
      const luisa::compute::CoroFrameDesc *desc_ptr = &ctx.frame_desc;
      luisa::vector<size_t> fields_copy{fields.begin(), fields.end()};
      auto emit_read_page_id =
          [single_field, field_index, binding_ptr, desc_ptr,
           fields_copy = std::move(fields_copy), layout,
           frame_soa = ctx.global_memory_soa](
              const ByteBufferVar &storage, const UInt &idx,
              const UInt &capacity) noexcept -> UInt {
        if (single_field) {
          return layout.has_runtime_capacity()
                     ? coro::coro_frame_read_field<luisa::uint>(
                           storage, idx, capacity, layout, frame_soa,
                           field_index)
                     : coro::coro_frame_read_field<luisa::uint>(
                           storage, idx, layout, frame_soa, field_index);
        }
        auto frame = luisa::compute::CoroFrame::create(desc_ptr);
        if (layout.has_runtime_capacity()) {
          coro::coro_frame_load_into(frame, storage, idx, capacity, layout,
                                     frame_soa, fields_copy, false, false);
        } else {
          coro::coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                                     fields_copy, false, false);
        }
        return binding_ptr->read<luisa::uint>(frame);
      };
      Kernel1D collect = [total_pages, emit_read_page_id](
                             ByteBufferVar storage, BufferUInt indices,
                             BufferUInt demand_hist, UInt capacity,
                             UInt n) noexcept {
        const auto x = dispatch_x();
        $if(x >= n) { $return(); };
        const auto idx = indices.read(x);
        const auto page_id = emit_read_page_id(storage, idx, capacity);
        $if(page_id < total_pages) {
          demand_hist.atomic(page_id).fetch_add(1u);
        };
      };
      Kernel1D filter_ready = [total_pages, emit_read_page_id](
                                  ByteBufferVar storage, BufferUInt indices,
                                  BufferUInt page_table, BufferUInt ready_flags,
                                  UInt capacity, UInt n) noexcept {
        const auto x = dispatch_x();
        $if(x >= n) { $return(); };
        const auto idx = indices.read(x);
        const auto page_id = emit_read_page_id(storage, idx, capacity);
        UInt ready = def(1u);
        $if(page_id < total_pages) {
          ready = select(
              0u, 1u, (page_table.read(page_id) & vt_page_resident_bit) != 0u);
        };
        ready_flags.write(x, ready);
      };
      auto &emplaced = self->_collect_shader_cache.emplace_back(CollectShaderEntry{
          .key = key,
          .shader = ctx.device.compile(collect, ctx.shader_option),
          .filter_shader =
              ctx.device.compile(filter_ready, ctx.shader_option)});
      cached_ptr = &emplaced.shader;
      cached_filter_ptr = &emplaced.filter_shader;
      if (ctx.device.impl()->query("vk_timeline_active") == "1") {
        char qbuf[192];
        std::snprintf(qbuf, sizeof(qbuf), "vk_timeline_label:shader:%llu:vt_collect",
                      static_cast<unsigned long long>(emplaced.shader.handle()));
        static_cast<void>(ctx.device.impl()->query(qbuf));
        std::snprintf(
            qbuf, sizeof(qbuf), "vk_timeline_label:shader:%llu:vt_filter_ready",
            static_cast<unsigned long long>(emplaced.filter_shader.handle()));
        static_cast<void>(ctx.device.impl()->query(qbuf));
      }
    }

    luisa::vector<size_t> reconstruct_copy{fields.begin(), fields.end()};
    return luisa::make_unique<VtWavefrontExtensionHandler>(
        self, single_field, field_index, layout, ctx.global_memory_soa,
        &ctx.frame_desc, &page_binding, std::move(reconstruct_copy),
        Shader1D<ByteBuffer, Buffer<luisa::uint>, Buffer<luisa::uint>,
                 luisa::uint, luisa::uint>{},
        cached_ptr, cached_filter_ptr);
  };
}

VirtualTextureCache::~VirtualTextureCache() noexcept {
  stop_async_worker();
  stop_bg_decode_workers();
  if (_use_copy_stream && _copy_stream) {
    _copy_stream.synchronize();
  }
  wait_all_upload_fences();
  dump_profile_if_enabled();
}

} // namespace psycles::luisa_backend::detail
