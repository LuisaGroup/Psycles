#include "virtual_texture_cache.h"

#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
        opcode == abi::NODE_TEX_ENVIRONMENT) {
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
  const std::uint32_t max_dim = pool.is_float4 ? 4096u : 8192u;
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
  const auto storage = pool.is_float4 ? luisa::compute::PixelStorage::FLOAT4
                                      : luisa::compute::PixelStorage::BYTE4;
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

  const std::uint32_t max_staging = pool.is_float4 ? 128u : 512u;
  pool.staging_capacity = std::min(capacity, max_staging);
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
      .pad0 = 0u,
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
                                     luisa::uint, luisa::uint> *cached_shader)
      : _cache{std::move(cache)},
        _use_fast_field_read{use_fast_field_read},
        _field_index{field_index},
        _layout{layout},
        _soa{soa},
        _frame_desc{frame_desc},
        _page_binding{page_binding},
        _reconstruct_fields{std::move(reconstruct_fields)},
        _fallback_collect_shader{std::move(fallback_collect_shader)},
        _cached_shader{cached_shader} {
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

  [[nodiscard]] bool defer_while_other_queues_nonempty() const noexcept override {
    return true;
  }

  [[nodiscard]] bool supports_command_list() const noexcept override {
    return true;
  }

  [[nodiscard]] const luisa::compute::Shader1D<
      luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
      luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint> &
  collect_shader() const noexcept {
    return _cached_shader != nullptr ? *_cached_shader
                                     : _fallback_collect_shader;
  }

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
    entry.storage = src.probed.storage;
    entry.load_failed = src.load_failed;
    entry.force_direct = src.force_direct;
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
  struct PageRange {
    std::uint32_t offset{};
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    bool is_float4{false};
  };
  struct PendingVirtualRange {
    std::uint32_t source_index{};
    std::uint32_t extension{};
    std::uint32_t tiles_x{};
    std::uint32_t tiles_y{};
    bool is_float4{false};
  };
  // Deduplicate virtual page ranges by (image_id, extension).
  std::unordered_map<std::uint64_t, std::size_t> range_lookup;
  std::vector<PendingVirtualRange> byte4_ranges;
  std::vector<PendingVirtualRange> float4_ranges;
  std::vector<std::pair<bool, std::size_t>> handle_range_ref(
      svm_image_handles.size(), {false, std::numeric_limits<std::size_t>::max()});

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
    const bool is_float4 =
        src.storage == io::DecodedImageStorage::float32;
    if (is_float4) {
      desc.flags |= vt_desc_flag_float4;
    }
    if (src.force_direct) {
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
      handle_range_ref[i] = {is_float4, it->second};
    } else {
      auto &vec = is_float4 ? float4_ranges : byte4_ranges;
      const std::size_t idx = vec.size();
      vec.push_back(PendingVirtualRange{
          .source_index = src_it->second,
          .extension = ext & 0x3u,
          .tiles_x = tx,
          .tiles_y = ty,
          .is_float4 = is_float4});
      range_lookup.emplace(dedup_key, idx);
      handle_range_ref[i] = {is_float4, idx};
    }
  }

  std::vector<std::uint32_t> byte4_offsets(byte4_ranges.size(), 0u);
  std::uint32_t page_cursor = 0u;
  _byte4_pool.is_float4 = false;
  _byte4_pool.page_begin = page_cursor;
  for (std::size_t r = 0u; r < byte4_ranges.size(); ++r) {
    const auto &pr = byte4_ranges[r];
    byte4_offsets[r] = page_cursor;
    for (std::uint32_t ty = 0u; ty < pr.tiles_y; ++ty) {
      for (std::uint32_t tx = 0u; tx < pr.tiles_x; ++tx) {
        _pages.push_back(VirtualPageMeta{
            .source_index = pr.source_index,
            .tile_x = tx,
            .tile_y = ty,
            .extension = pr.extension,
            .is_float4 = false});
        ++page_cursor;
      }
    }
  }
  _byte4_pool.page_end = page_cursor;

  std::vector<std::uint32_t> float4_offsets(float4_ranges.size(), 0u);
  _float4_pool.is_float4 = true;
  _float4_pool.page_begin = page_cursor;
  for (std::size_t r = 0u; r < float4_ranges.size(); ++r) {
    const auto &pr = float4_ranges[r];
    float4_offsets[r] = page_cursor;
    for (std::uint32_t ty = 0u; ty < pr.tiles_y; ++ty) {
      for (std::uint32_t tx = 0u; tx < pr.tiles_x; ++tx) {
        _pages.push_back(VirtualPageMeta{
            .source_index = pr.source_index,
            .tile_x = tx,
            .tile_y = ty,
            .extension = pr.extension,
            .is_float4 = true});
        ++page_cursor;
      }
    }
  }
  _float4_pool.page_end = page_cursor;

  for (std::size_t i = 0u; i < svm_image_handles.size(); ++i) {
    if ((_descs[i].flags & vt_desc_flag_virtual) == 0u) {
      continue;
    }
    const auto [is_float4, idx] = handle_range_ref[i];
    _descs[i].page_table_offset =
        is_float4 ? float4_offsets[idx] : byte4_offsets[idx];
  }

  _stats.total_images = static_cast<std::uint32_t>(_sources.size());
  for (const auto &s : _sources) {
    if (!s.load_failed && !s.force_direct) {
      ++_stats.virtual_images;
    }
  }
  _stats.byte4_virtual_pages = _byte4_pool.page_end - _byte4_pool.page_begin;
  _stats.float4_virtual_pages = _float4_pool.page_end - _float4_pool.page_begin;
  _stats.physical_tile_size = _physical_tile_size;
  _stats.tile_border = _tile_border;
  _stats.logical_tile_size = _logical_tile_size;
  _stats.gpu_budget_bytes = options.gpu_cache_budget_bytes;

  // Partition GPU budget across active BYTE4 and FLOAT4 physical pools.
  const std::uint64_t b4_tile_bytes =
      static_cast<std::uint64_t>(_physical_tile_size) *
      static_cast<std::uint64_t>(_physical_tile_size) * 4ull;
  const std::uint64_t f4_tile_bytes =
      static_cast<std::uint64_t>(_physical_tile_size) *
      static_cast<std::uint64_t>(_physical_tile_size) * 16ull;

  const std::uint64_t b4_demand_bytes =
      static_cast<std::uint64_t>(_stats.byte4_virtual_pages) * b4_tile_bytes;
  const std::uint64_t f4_demand_bytes =
      static_cast<std::uint64_t>(_stats.float4_virtual_pages) * f4_tile_bytes;
  const std::uint64_t total_demand_bytes = b4_demand_bytes + f4_demand_bytes;

  std::uint32_t b4_cap = 0u;
  std::uint32_t f4_cap = 0u;
  if (_stats.byte4_virtual_pages > 0u && _stats.float4_virtual_pages == 0u) {
    b4_cap = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        _stats.byte4_virtual_pages,
        std::max<std::uint64_t>(1ull, options.gpu_cache_budget_bytes / b4_tile_bytes)));
  } else if (_stats.float4_virtual_pages > 0u && _stats.byte4_virtual_pages == 0u) {
    f4_cap = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        _stats.float4_virtual_pages,
        std::max<std::uint64_t>(1ull, options.gpu_cache_budget_bytes / f4_tile_bytes)));
  } else if (total_demand_bytes > 0ull) {
    if (options.gpu_cache_budget_bytes >= total_demand_bytes) {
      b4_cap = _stats.byte4_virtual_pages;
      f4_cap = _stats.float4_virtual_pages;
    } else {
      const double f4_ratio = static_cast<double>(f4_demand_bytes) /
                              static_cast<double>(total_demand_bytes);
      const std::uint64_t f4_floor_tiles = std::min<std::uint64_t>(
          _stats.float4_virtual_pages,
          std::max<std::uint64_t>(
              4ull,
              std::min<std::uint64_t>(
                  64ull, options.gpu_cache_budget_bytes /
                             (8ull * f4_tile_bytes))));
      const std::uint64_t f4_budget_min = f4_floor_tiles * f4_tile_bytes;
      const std::uint64_t b4_budget_min = std::min<std::uint64_t>(
          b4_demand_bytes, std::max<std::uint64_t>(4ull * b4_tile_bytes, 0ull));
      std::uint64_t f4_budget = static_cast<std::uint64_t>(
          static_cast<double>(options.gpu_cache_budget_bytes) * f4_ratio);
      if (f4_budget < f4_budget_min &&
          options.gpu_cache_budget_bytes >= f4_budget_min + b4_budget_min) {
        f4_budget = f4_budget_min;
      }
      f4_budget = std::min(f4_budget, f4_demand_bytes);
      std::uint64_t b4_budget =
          options.gpu_cache_budget_bytes > f4_budget
              ? options.gpu_cache_budget_bytes - f4_budget
              : b4_tile_bytes;
      b4_budget = std::min(b4_budget, b4_demand_bytes);
      f4_cap = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
          f4_budget / f4_tile_bytes, 1ull, _stats.float4_virtual_pages));
      b4_cap = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
          b4_budget / b4_tile_bytes, 1ull, _stats.byte4_virtual_pages));
    }
  }

  std::uint32_t next_bindless_slot = 0u;
  init_pool_geometry(_device, _byte4_pool, _physical_tile_size, b4_cap,
                     _stats.byte4_virtual_pages, next_bindless_slot);
  _byte4_pool.epoch_offset = 0u;
  init_pool_geometry(_device, _float4_pool, _physical_tile_size, f4_cap,
                     _stats.float4_virtual_pages, next_bindless_slot);
  _float4_pool.epoch_offset = b4_cap;

  _stats.byte4_cache_capacity = b4_cap;
  _stats.float4_cache_capacity = f4_cap;

  const auto heap_slots = std::max<std::uint32_t>(next_bindless_slot, 1u);
  _atlas_heap = _device.create_bindless_array(heap_slots);
  for (std::size_t a = 0u; a < _byte4_pool.atlases.size(); ++a) {
    const auto slot_base =
        _byte4_pool.bindless_base + static_cast<std::uint32_t>(a) * 2u;
    _atlas_heap.emplace_on_update(slot_base, _byte4_pool.atlases[a],
                                Sampler::linear_point_edge());
    _atlas_heap.emplace_on_update(slot_base + 1u, _byte4_pool.atlases[a],
                                Sampler::point_edge());
    _stats.gpu_allocated_atlas_bytes +=
        static_cast<std::uint64_t>(_byte4_pool.atlas_widths[a]) *
        static_cast<std::uint64_t>(_byte4_pool.atlas_heights[a]) * 4ull;
  }
  for (std::size_t a = 0u; a < _float4_pool.atlases.size(); ++a) {
    const auto slot_base =
        _float4_pool.bindless_base + static_cast<std::uint32_t>(a) * 2u;
    _atlas_heap.emplace_on_update(slot_base, _float4_pool.atlases[a],
                                Sampler::linear_point_edge());
    _atlas_heap.emplace_on_update(slot_base + 1u, _float4_pool.atlases[a],
                                Sampler::point_edge());
    _stats.gpu_allocated_atlas_bytes +=
        static_cast<std::uint64_t>(_float4_pool.atlas_widths[a]) *
        static_cast<std::uint64_t>(_float4_pool.atlas_heights[a]) * 16ull;
  }

  const auto desc_count = std::max<std::size_t>(_descs.size(), 1u);
  std::vector<VtImageDescGpu> desc_upload(desc_count);
  if (!_descs.empty()) {
    std::copy(_descs.begin(), _descs.end(), desc_upload.begin());
  }
  _pool_configs = _device.create_buffer<VtPoolConfigGpu>(2u);
  const std::array<VtPoolConfigGpu, 2u> pool_cfg_upload{
      make_pool_config_gpu(_byte4_pool),
      make_pool_config_gpu(_float4_pool)};
  _image_descs_buffer = _device.create_buffer<VtImageDescGpu>(desc_count);

  const auto total_pages = std::max<std::size_t>(_pages.size(), 1u);
  const auto bitset_words = (total_pages + 31u) / 32u;
  const auto total_slots = std::max<std::uint32_t>(b4_cap + f4_cap, 1u);
  _page_table = _device.create_buffer<luisa::uint>(total_pages);
  _miss_bitset = _device.create_buffer<luisa::uint>(bitset_words);
  _slot_epoch = _device.create_buffer<luisa::uint>(total_slots);
  _current_epoch = _device.create_buffer<luisa::uint>(2u);
  _demand_hist = _device.create_buffer<luisa::uint>(total_pages);
  _aux_demand_hist = _device.create_buffer<luisa::uint>(total_pages);

  const std::uint32_t tile_texels = _physical_tile_size * _physical_tile_size;
  const std::size_t staging_b4_len = std::max<std::size_t>(
      static_cast<std::size_t>(_byte4_pool.staging_capacity) * tile_texels, 1u);
  const std::size_t staging_f4_len = std::max<std::size_t>(
      static_cast<std::size_t>(_float4_pool.staging_capacity) * tile_texels, 1u);
  const std::size_t max_chunk_tiles = std::max<std::size_t>(
      std::max(_byte4_pool.staging_capacity, _float4_pool.staging_capacity), 1u);
  const std::size_t pt_update_cap =
      std::max<std::size_t>(2u * total_slots, 1u);

  if (_pinned_ext != nullptr) {
    bool upload_ok = true;
    for (std::size_t i = 0u; i < kUploadRingSize; ++i) {
      auto sb4 = _pinned_ext->allocate_pinned_memory<luisa::uint>(
          staging_b4_len, PinnedMemoryOption{.write_combined = true});
      auto sf4 = _pinned_ext->allocate_pinned_memory<luisa::float4>(
          staging_f4_len, PinnedMemoryOption{.write_combined = true});
      auto um = _pinned_ext->allocate_pinned_memory<luisa::uint4>(
          max_chunk_tiles, PinnedMemoryOption{.write_combined = true});
      auto ptu = _pinned_ext->allocate_pinned_memory<luisa::uint2>(
          pt_update_cap, PinnedMemoryOption{.write_combined = true});
      auto seu = _pinned_ext->allocate_pinned_memory<luisa::uint2>(
          total_slots, PinnedMemoryOption{.write_combined = true});
      auto *p_sb4 = static_cast<luisa::uint *>(sb4.native_handle());
      auto *p_sf4 = static_cast<luisa::float4 *>(sf4.native_handle());
      auto *p_um = static_cast<luisa::uint4 *>(um.native_handle());
      auto *p_ptu = static_cast<luisa::uint2 *>(ptu.native_handle());
      auto *p_seu = static_cast<luisa::uint2 *>(seu.native_handle());
      if (!sb4 || !sf4 || !um || !ptu || !seu || p_sb4 == nullptr ||
          p_sf4 == nullptr || p_um == nullptr || p_ptu == nullptr ||
          p_seu == nullptr) {
        upload_ok = false;
        break;
      }
      _staging_byte4[i] = std::move(sb4);
      _staging_float4[i] = std::move(sf4);
      _upload_meta[i] = std::move(um);
      _page_table_updates[i] = std::move(ptu);
      _slot_epoch_updates[i] = std::move(seu);
      _pinned_staging_byte4_ptr[i] = p_sb4;
      _pinned_staging_float4_ptr[i] = p_sf4;
      _pinned_upload_meta_ptr[i] = p_um;
      _pinned_page_table_updates_ptr[i] = p_ptu;
      _pinned_slot_epoch_updates_ptr[i] = p_seu;
    }
    if (upload_ok) {
      _use_pinned_upload = true;
    } else {
      for (std::size_t i = 0u; i < kUploadRingSize; ++i) {
        _staging_byte4[i] = {};
        _staging_float4[i] = {};
        _upload_meta[i] = {};
        _page_table_updates[i] = {};
        _slot_epoch_updates[i] = {};
        _pinned_staging_byte4_ptr[i] = nullptr;
        _pinned_staging_float4_ptr[i] = nullptr;
        _pinned_upload_meta_ptr[i] = nullptr;
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
    for (std::size_t i = 0u; i < kUploadRingSize; ++i) {
      _staging_byte4[i] = _device.create_buffer<luisa::uint>(staging_b4_len);
      _staging_float4[i] = _device.create_buffer<luisa::float4>(staging_f4_len);
      _upload_meta[i] = _device.create_buffer<luisa::uint4>(max_chunk_tiles);
      _page_table_updates[i] =
          _device.create_buffer<luisa::uint2>(pt_update_cap);
      _slot_epoch_updates[i] = _device.create_buffer<luisa::uint2>(total_slots);
    }
  }

  _stats.gpu_allocated_metadata_bytes =
      2u * sizeof(VtPoolConfigGpu) +
      desc_count * sizeof(VtImageDescGpu) +
      (total_pages * 3u + bitset_words + total_slots + 2u) *
          sizeof(luisa::uint) +
      kUploadRingSize *
          (staging_b4_len * sizeof(luisa::uint) +
           staging_f4_len * sizeof(luisa::float4) +
           max_chunk_tiles * sizeof(luisa::uint4) +
           (pt_update_cap + total_slots) * sizeof(luisa::uint2));
  if (_use_pinned_readback) {
    _stats.gpu_allocated_metadata_bytes +=
        (total_pages * 2u + bitset_words + total_slots) * sizeof(luisa::uint);
  }

  _host_page_table.assign(total_pages, 0u);
  _host_byte4_slots.assign(b4_cap, vt_invalid_page);
  _host_float4_slots.assign(f4_cap, vt_invalid_page);
  _host_byte4_slot_meta.assign(b4_cap, PoolSlotMeta{});
  _host_float4_slot_meta.assign(f4_cap, PoolSlotMeta{});
  _host_demand_hist.assign(total_pages, 0u);
  _host_aux_demand_hist.assign(total_pages, 0u);
  _host_miss_bitset.assign(bitset_words, 0u);
  _host_slot_epoch.assign(total_slots, 0u);
  _host_current_epoch.assign(2u, 0u);
  _host_current_epoch[0u] = _byte4_pool.eviction_possible ? 2u : 0u;
  _host_current_epoch[1u] = _float4_pool.eviction_possible ? 2u : 0u;
  if (!_use_pinned_upload) {
    _host_staging_byte4.resize(staging_b4_len, 0u);
    _host_staging_float4.resize(staging_f4_len, luisa::make_float4(0.0f));
    _host_upload_meta.resize(max_chunk_tiles);
    _host_page_table_updates.reserve(pt_update_cap);
    _host_slot_epoch_updates.reserve(total_slots);
  }

  stream << _atlas_heap.update()
         << _pool_configs.copy_from(luisa::span{pool_cfg_upload})
         << _image_descs_buffer.copy_from(luisa::span{desc_upload})
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

  _upload_float4_shader = _device.compile<3>(
      [](ImageVar<float> atlas, BufferVar<luisa::float4> staging,
         BufferVar<luisa::uint4> meta, UInt meta_offset,
         UInt tile_size) noexcept {
        const auto coord = dispatch_id();
        const auto m = meta.read(meta_offset + coord.z);
        const auto texel_idx =
            (m.x * tile_size + coord.y) * tile_size + coord.x;
        atlas.write(make_uint2(m.y + coord.x, m.z + coord.y),
                    staging.read(texel_idx));
      });

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
  }

  stream << synchronize();
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

std::string VirtualTextureCache::stats_json() const {
  char buf[1024];
  std::snprintf(
      buf, sizeof(buf),
      "{\"total_images\":%u,\"virtual_images\":%u,\"images_decoded\":%llu,"
      "\"byte4_virtual_pages\":%u,\"byte4_cache_capacity\":%u,"
      "\"byte4_peak_resident\":%u,\"float4_virtual_pages\":%u,"
      "\"float4_cache_capacity\":%u,\"float4_peak_resident\":%u,"
      "\"physical_tile_size\":%u,\"tile_border\":%u,\"logical_tile_size\":%u,"
      "\"gpu_budget_bytes\":%llu,\"gpu_allocated_atlas_bytes\":%llu,"
      "\"gpu_allocated_metadata_bytes\":%llu,\"fault_batches\":%llu,"
      "\"aux_fault_batches\":%llu,\"tiles_loaded\":%llu,"
      "\"tiles_evicted\":%llu,\"secondary_misses_coalesced\":%llu,"
      "\"host_decode_evictions\":%llu,\"collect_sync_ms\":%.3f,"
      "\"decode_ms\":%.3f,\"pack_ms\":%.3f,\"upload_ms\":%.3f}",
      _stats.total_images, _stats.virtual_images,
      static_cast<unsigned long long>(_stats.images_decoded),
      _stats.byte4_virtual_pages, _stats.byte4_cache_capacity,
      _stats.byte4_peak_resident, _stats.float4_virtual_pages,
      _stats.float4_cache_capacity, _stats.float4_peak_resident,
      _stats.physical_tile_size, _stats.tile_border, _stats.logical_tile_size,
      static_cast<unsigned long long>(_stats.gpu_budget_bytes),
      static_cast<unsigned long long>(_stats.gpu_allocated_atlas_bytes),
      static_cast<unsigned long long>(_stats.gpu_allocated_metadata_bytes),
      static_cast<unsigned long long>(_stats.fault_batches),
      static_cast<unsigned long long>(_stats.aux_fault_batches),
      static_cast<unsigned long long>(_stats.tiles_loaded),
      static_cast<unsigned long long>(_stats.tiles_evicted),
      static_cast<unsigned long long>(_stats.secondary_misses_coalesced),
      static_cast<unsigned long long>(_stats.host_decode_evictions),
      _stats.collect_sync_ms, _stats.decode_ms, _stats.pack_ms,
      _stats.upload_ms);
  return std::string(buf);
}

namespace {

void VtWavefrontExtensionHandler::dispatch(
    const coro::WavefrontCoroExtensionDispatchContext &ctx) noexcept {
  if (_cache) {
    _cache->dispatch_wavefront_collect(ctx, collect_shader());
  }
}

} // namespace

void VirtualTextureCache::dispatch_wavefront_collect(
    const coro::WavefrontCoroExtensionDispatchContext &ctx,
    const CollectShader &collect_shader) noexcept {
  if (!has_virtual_images() || ctx.frame_count == 0u) {
    return;
  }
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
  service_misses_impl(ctx.stream, MissSource::wavefront_demand,
                      ctx.command_list);
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

    const Shader1D<ByteBuffer, Buffer<luisa::uint>, Buffer<luisa::uint>,
                   luisa::uint, luisa::uint> *cached_ptr = nullptr;
    for (const auto &entry : self->_collect_shader_cache) {
      if (entry.key == key) {
        cached_ptr = &entry.shader;
        break;
      }
    }

    if (cached_ptr == nullptr) {
      const auto total_pages = static_cast<luisa::uint>(self->_pages.size());
      const coro::CoroSlotAccess *binding_ptr = &page_binding;
      const luisa::compute::CoroFrameDesc *desc_ptr = &ctx.frame_desc;
      luisa::vector<size_t> fields_copy{fields.begin(), fields.end()};
      Kernel1D collect = [total_pages, single_field, field_index, binding_ptr,
                          desc_ptr, fields_copy = std::move(fields_copy),
                          layout, frame_soa = ctx.global_memory_soa](
                             ByteBufferVar storage, BufferUInt indices,
                             BufferUInt demand_hist, UInt capacity,
                             UInt n) noexcept {
        const auto x = dispatch_x();
        $if(x >= n) { $return(); };
        const auto idx = indices.read(x);
        UInt page_id = def(vt_invalid_page);
        if (single_field) {
          if (layout.has_runtime_capacity()) {
            page_id = coro::coro_frame_read_field<luisa::uint>(
                storage, idx, capacity, layout, frame_soa, field_index);
          } else {
            page_id = coro::coro_frame_read_field<luisa::uint>(
                storage, idx, layout, frame_soa, field_index);
          }
        } else {
          auto frame = luisa::compute::CoroFrame::create(desc_ptr);
          if (layout.has_runtime_capacity()) {
            coro::coro_frame_load_into(frame, storage, idx, capacity, layout,
                                       frame_soa, fields_copy, false, false);
          } else {
            coro::coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                                       fields_copy, false, false);
          }
          page_id = binding_ptr->read<luisa::uint>(frame);
        }
        $if(page_id < total_pages) {
          demand_hist.atomic(page_id).fetch_add(1u);
        };
      };
      auto &emplaced = self->_collect_shader_cache.emplace_back(CollectShaderEntry{
          .key = key,
          .shader = ctx.device.compile(collect, ctx.shader_option)});
      cached_ptr = &emplaced.shader;
    }

    luisa::vector<size_t> reconstruct_copy{fields.begin(), fields.end()};
    return luisa::make_unique<VtWavefrontExtensionHandler>(
        self, single_field, field_index, layout, ctx.global_memory_soa,
        &ctx.frame_desc, &page_binding, std::move(reconstruct_copy),
        Shader1D<ByteBuffer, Buffer<luisa::uint>, Buffer<luisa::uint>,
                 luisa::uint, luisa::uint>{},
        cached_ptr);
  };
}

VirtualTextureCache::~VirtualTextureCache() noexcept {
  stop_bg_decode_workers();
  wait_all_upload_fences();
}

} // namespace psycles::luisa_backend::detail
