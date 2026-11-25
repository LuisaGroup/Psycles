#include "virtual_texture_cache.h"

#include "virtual_texture_cache_bc.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;
using DecodedSourcePayload = VirtualTextureCache::DecodedSourcePayload;

namespace {

[[nodiscard]] double elapsed_ms_since(
    std::chrono::steady_clock::time_point t0) noexcept {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

} // namespace

VirtualTextureCache::PackedChunkMeta
VirtualTextureCache::prepare_chunk_staging(
    std::uint32_t pool_idx,
    PoolRuntime &pool_rt,
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
    PlanningScratch &scratch) {
  const auto &pool = pool_rt.config;
  auto &host_slots = pool_rt.host_slots;
  auto &slot_meta = pool_rt.host_slot_meta;
  const std::size_t chunk_count = chunk_end - chunk_begin;
  const std::size_t total_load = plan.misses.size();
  const std::size_t num_atlases = pool.atlases.size();
  const std::size_t tile_bytes = pool.tile_bytes;
  const bool use_prepack =
      (_options.prepack_mode != LuisaVirtualTexturePrepackMode::off) &&
      (_prepack_cache != nullptr) &&
      (_page_prepack_keys.size() == _pages.size());
  const bool use_disk_prepack =
      use_prepack &&
      (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk) &&
      (_prepack_disk != nullptr) && _prepack_disk->enabled();

  std::uint8_t *const pack_base_u8 =
      reinterpret_cast<std::uint8_t *>(target.pack_base);

  auto &assigned = scratch.assigned;
  assigned.clear();
  assigned.reserve(chunk_count);

  struct PrepackUnpinGuard {
    VtPrepackedPageCache *cache{nullptr};
    const std::vector<VtPrepackedPageCache::ResolvedTileSlot> *slots{nullptr};
    double *pack_ms_out{nullptr};

    void release() noexcept {
      if (cache != nullptr && slots != nullptr && !slots->empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        cache->unpin_batch(*slots);
        if (pack_ms_out != nullptr) {
          *pack_ms_out += elapsed_ms_since(t0);
        }
      }
      cache = nullptr;
    }
    ~PrepackUnpinGuard() noexcept { release(); }
  };
  PrepackUnpinGuard prepack_unpin_guard;

  auto &chunk_page_ids = scratch.chunk_page_ids;
  chunk_page_ids.clear();
  auto &prepack_slots = scratch.prepack_slots;
  prepack_slots.clear();
  auto &prepack_hit_indices = scratch.prepack_hit_indices;
  prepack_hit_indices.clear();
  auto &slot_disk_ok = scratch.slot_disk_ok;
  slot_disk_ok.clear();
  auto &disk_read_indices = scratch.disk_read_indices;
  disk_read_indices.clear();
  bool any_live_pack = !use_prepack;

  if (use_prepack) {
    chunk_page_ids.resize(chunk_count);
    for (std::size_t k = 0u; k < chunk_count; ++k) {
      chunk_page_ids[k] = plan.misses[chunk_begin + k].page;
    }
    const auto res_t0 = std::chrono::steady_clock::now();
    _prepack_cache->resolve_batch_indexed(_page_prepack_keys, chunk_page_ids,
                                          tile_bytes, prepack_slots);
    _stats.pack_ms += elapsed_ms_since(res_t0);
    prepack_unpin_guard.cache = _prepack_cache.get();
    prepack_unpin_guard.slots = &prepack_slots;
    prepack_unpin_guard.pack_ms_out = &_stats.pack_ms;

    for (std::size_t k = 0u; k < chunk_count; ++k) {
      const auto &s = prepack_slots[k];
      if (s.is_hit && s.hit_ptr != nullptr) {
        prepack_hit_indices.push_back(static_cast<std::uint32_t>(k));
      }
    }

    slot_disk_ok.resize(chunk_count);
    std::fill(slot_disk_ok.begin(), slot_disk_ok.end(), std::uint8_t{0u});
    if (use_disk_prepack) {
      struct DiskMemoEntry {
        VtPrepackedImageKey key;
        bool ok{false};
      };
      std::vector<DiskMemoEntry> disk_memo;
      const auto disk_chk_t0 = std::chrono::steady_clock::now();
      for (std::size_t k = 0u; k < chunk_count; ++k) {
        const auto &s = prepack_slots[k];
        if (s.is_hit || s.duplicate_of >= 0) {
          continue;
        }
        const auto &pm = _pages[chunk_page_ids[k]];
        const auto &src = _sources[pm.source_index];
        if (src.decoded || src.load_failed) {
          continue;
        }
        const auto &img_key = _page_prepack_keys[chunk_page_ids[k]].image;
        bool found = false;
        bool ok = false;
        for (const auto &m : disk_memo) {
          if (m.key == img_key) {
            found = true;
            ok = m.ok;
            break;
          }
        }
        if (!found) {
          ok = _prepack_disk->has_valid_file(img_key, pm.tiles_x, pm.tiles_y);
          disk_memo.push_back(DiskMemoEntry{.key = img_key, .ok = ok});
        }
        if (ok) {
          slot_disk_ok[k] = 1u;
          disk_read_indices.push_back(k);
        }
      }
      _stats.prepack_disk_ms += elapsed_ms_since(disk_chk_t0);
    }
    for (std::size_t k = 0u; k < chunk_count; ++k) {
      const auto &s = prepack_slots[k];
      if (!s.is_hit && (s.duplicate_of < 0) && (slot_disk_ok[k] == 0u)) {
        any_live_pack = true;
        break;
      }
    }
  }

  auto slot_needs_live_pack = [&](std::size_t k) noexcept {
    if (!use_prepack) {
      return true;
    }
    const auto &s = prepack_slots[k];
    return !s.is_hit && (s.duplicate_of < 0) && (slot_disk_ok[k] == 0u);
  };

  auto push_pt = [&](luisa::uint page, luisa::uint val) noexcept {
    const auto u = luisa::make_uint2(page, val);
    if (pt_upd_dst != nullptr) {
      pt_upd_dst[pt_upd_count++] = u;
    } else {
      pt_upd_vec.push_back(u);
      ++pt_upd_count;
    }
  };
  auto push_ep = [&](luisa::uint idx, luisa::uint val) noexcept {
    const auto u = luisa::make_uint2(idx, val);
    if (ep_upd_dst != nullptr) {
      ep_upd_dst[ep_upd_count++] = u;
    } else {
      ep_upd_vec.push_back(u);
      ++ep_upd_count;
    }
  };

  auto copy_prepack_hits_chunk = [&]() {
    if (!use_prepack || prepack_hit_indices.empty()) {
      return;
    }
    const auto hit_t0 = std::chrono::steady_clock::now();
    const std::size_t n = prepack_hit_indices.size();
    if (n < 16u) {
      for (std::size_t i = 0u; i < n; ++i) {
        const std::uint32_t k = prepack_hit_indices[i];
        std::memcpy(
            pack_base_u8 + static_cast<std::size_t>(k) * tile_bytes,
            prepack_slots[k].hit_ptr, tile_bytes);
      }
    } else {
      const std::size_t stripes = std::min<std::size_t>(8u, (n + 15u) / 16u);
      psycles::parallel_for(
          stripes,
          [&](std::size_t stripe) noexcept {
            const std::size_t begin_i = (stripe * n) / stripes;
            const std::size_t end_i = ((stripe + 1u) * n) / stripes;
            for (std::size_t i = begin_i; i < end_i; ++i) {
              const std::uint32_t k = prepack_hit_indices[i];
              std::memcpy(
                  pack_base_u8 + static_cast<std::size_t>(k) * tile_bytes,
                  prepack_slots[k].hit_ptr, tile_bytes);
            }
          },
          stripes);
    }
    _stats.pack_ms += elapsed_ms_since(hit_t0);
  };

  auto run_disk_reads = [&]() {
    if (!use_disk_prepack || disk_read_indices.empty()) {
      return;
    }
    const auto dr_t0 = std::chrono::steady_clock::now();
    auto read_one = [&](std::size_t idx_pos) noexcept {
      const std::size_t k = disk_read_indices[idx_pos];
      const auto pid = chunk_page_ids[k];
      const auto &pm = _pages[pid];
      const auto &key = _page_prepack_keys[pid];
      auto &s = prepack_slots[k];
      std::uint8_t *staging_ptr = pack_base_u8 + k * tile_bytes;
      bool ok = false;
      if (s.fill_ptr != nullptr) {
        ok = _prepack_disk->read_single_page(
            key, pm.tiles_x, pm.tiles_y,
            std::span<std::uint8_t>{s.fill_ptr, tile_bytes});
        if (ok) {
          std::memcpy(staging_ptr, s.fill_ptr, tile_bytes);
        } else {
          vt_fill_zero_tile(
              _physical_tile_size, pool.format,
              std::span<std::uint8_t>{s.fill_ptr, tile_bytes});
          std::memcpy(staging_ptr, s.fill_ptr, tile_bytes);
        }
      } else {
        ok = _prepack_disk->read_single_page(
            key, pm.tiles_x, pm.tiles_y,
            std::span<std::uint8_t>{staging_ptr, tile_bytes});
        if (!ok) {
          vt_fill_zero_tile(
              _physical_tile_size, pool.format,
              std::span<std::uint8_t>{staging_ptr, tile_bytes});
        }
      }
    };
    const std::size_t count = disk_read_indices.size();
    if (count >= 4u) {
      psycles::parallel_for(count, read_one);
    } else {
      for (std::size_t i = 0u; i < count; ++i) {
        read_one(i);
      }
    }
    _stats.prepack_disk_reads += count;
    _stats.prepack_disk_ms += elapsed_ms_since(dr_t0);
  };

  auto ensure_disk_files_for_range = [&](std::size_t begin_k,
                                         std::size_t end_k) {
    if (!use_disk_prepack) {
      return;
    }
    std::vector<VtPrepackedImageKey> checked_keys;
    const auto disk_wr_t0 = std::chrono::steady_clock::now();
    for (std::size_t k = begin_k; k < end_k; ++k) {
      if (!slot_needs_live_pack(k)) {
        continue;
      }
      const auto pid = chunk_page_ids[k];
      const auto &pm = _pages[pid];
      const auto &src = _sources[pm.source_index];
      if (!src.decoded || src.load_failed) {
        continue;
      }
      const auto &img_key = _page_prepack_keys[pid].image;
      if (std::find(checked_keys.begin(), checked_keys.end(), img_key) !=
          checked_keys.end()) {
        continue;
      }
      checked_keys.push_back(img_key);
      if (!_prepack_disk->has_valid_file(img_key, pm.tiles_x, pm.tiles_y)) {
        const auto mip_view = source_mip_view(src, pm.mip_level);
        if (_prepack_disk->ensure_image_file(
                img_key, mip_view.unorm8_pixels, mip_view.float_pixels,
                pm.tiles_x, pm.tiles_y, _gpu_bc.get())) {
          ++_stats.prepack_disk_writes;
        }
      }
    }
    if (_gpu_bc) {
      _stats.gpu_bc_tiles_encoded = _gpu_bc->tiles_encoded();
      _stats.gpu_bc_ms = _gpu_bc->encode_ms();
    }
    // Reuse freshly created .vtpages bytes so cold uploads match warm disk reads
    // bit-for-bit without re-encoding those tiles on host or GPU.
    for (std::size_t k = begin_k; k < end_k; ++k) {
      if (!slot_needs_live_pack(k)) {
        continue;
      }
      const auto pid = chunk_page_ids[k];
      const auto &pm = _pages[pid];
      std::uint8_t *staging_dst = pack_base_u8 + k * tile_bytes;
      const bool has_fill =
          use_prepack && (prepack_slots[k].fill_ptr != nullptr);
      const std::span<std::uint8_t> dst_span{
          has_fill ? prepack_slots[k].fill_ptr : staging_dst, tile_bytes};
      if (_prepack_disk->read_single_page(
              _page_prepack_keys[pid], pm.tiles_x, pm.tiles_y, dst_span)) {
        if (has_fill) {
          std::memcpy(staging_dst, dst_span.data(), tile_bytes);
        }
        slot_disk_ok[k] = 1u;
      }
    }
    _stats.prepack_disk_ms += elapsed_ms_since(disk_wr_t0);
  };

  auto assign_tile = [&](std::size_t k) {
    const auto &miss = plan.misses[chunk_begin + k];
    const std::uint32_t page_id = miss.page;
    const std::uint32_t victim = plan.victim_slots[chunk_begin + k];
    if (pool.eviction_possible) {
      push_ep(pool.epoch_offset + victim, plan.new_slot_epoch);
      slot_meta[victim] = PoolSlotMeta{
          .load_wf = plan.wf_seq,
          .last_hit_wf = 0ull,
          .load_seq = plan.load_seq,
          .hotness = 0u,
          .initial_demand = miss.primary_demand,
          .loaded_by_aux = is_aux};
    }
    const std::uint32_t old_page = host_slots[victim];
    if (old_page != vt_invalid_page) {
      _host_page_table[old_page] = 0u;
      push_pt(old_page, 0u);
      ++_stats.tiles_evicted;
    }
    host_slots[victim] = page_id;
    _host_page_table[page_id] = vt_page_resident_bit | victim;
    push_pt(page_id, vt_page_resident_bit | victim);
    record_tile_loaded_stats(page_id, pool.tile_bytes);
    ++prof_pool_loaded;
    if (pool.is_high_precision) {
      ++prof_f4_loaded;
    } else {
      ++prof_b4_loaded;
    }
    gpu_state_dirty = true;

    std::uint32_t atlas_idx = 0u;
    std::uint32_t slot_in_atlas = victim;
    if (num_atlases > 1u) {
      atlas_idx = victim >> pool.slots_per_atlas_log2;
      slot_in_atlas = victim & pool.slots_per_atlas_mask;
    }
    const std::uint32_t stx = slot_in_atlas & pool.atlas_tiles_x_mask;
    const std::uint32_t sty = slot_in_atlas >> pool.atlas_tiles_x_log2;
    assigned.push_back(AssignedTile{
        .page = page_id,
        .atlas_idx = atlas_idx,
        .dst_ox = stx * _physical_tile_size,
        .dst_oy = sty * _physical_tile_size,
        .staging_idx = static_cast<std::uint32_t>(k)});
  };

  auto pack_range = [&](std::size_t begin_k, std::size_t end_k) {
    const auto pack_t0 = std::chrono::steady_clock::now();
    const bool try_gpu_bc =
        _gpu_bc && _gpu_bc->available() &&
        VtGpuBcCompressor::supports_format(pool.format);
    bool packed_by_gpu = false;
    if (try_gpu_bc && (end_k - begin_k) >= _gpu_bc->min_batch_tiles()) {
      std::vector<VtGpuBcTileJob> gpu_jobs;
      std::vector<std::size_t> gpu_ks;
      gpu_jobs.reserve(end_k - begin_k);
      gpu_ks.reserve(end_k - begin_k);
      for (std::size_t k = begin_k; k < end_k; ++k) {
        if (use_prepack) {
          if (slot_disk_ok[k] != 0u) {
            continue;
          }
          const auto &s = prepack_slots[k];
          if (s.duplicate_of >= 0 || (s.is_hit && s.hit_ptr != nullptr)) {
            continue;
          }
        }
        const auto &pm = _pages[plan.misses[chunk_begin + k].page];
        const auto &src = _sources[pm.source_index];
        const auto mip_view = source_mip_view(src, pm.mip_level);
        const bool has_fill =
            use_prepack && (prepack_slots[k].fill_ptr != nullptr);
        std::uint8_t *pack_dst =
            has_fill ? prepack_slots[k].fill_ptr : (pack_base_u8 + k * tile_bytes);
        gpu_jobs.push_back(VtGpuBcTileJob{
            .unorm8_pixels = mip_view.unorm8_pixels,
            .float_pixels = mip_view.float_pixels,
            .width = mip_view.width,
            .height = mip_view.height,
            .tile_x = pm.tile_x,
            .tile_y = pm.tile_y,
            .extension = pm.extension,
            .physical_tile_size = _physical_tile_size,
            .logical_tile_size = _logical_tile_size,
            .border = _tile_border,
            .format = pool.format,
            .dst = pack_dst,
            .dst_bytes = tile_bytes});
        gpu_ks.push_back(k);
      }
      if (gpu_jobs.size() >= _gpu_bc->min_batch_tiles() &&
          _gpu_bc->compress_batch(gpu_jobs)) {
        for (std::size_t i = 0u; i < gpu_ks.size(); ++i) {
          const std::size_t k = gpu_ks[i];
          if (use_prepack && prepack_slots[k].fill_ptr != nullptr) {
            std::memcpy(pack_base_u8 + k * tile_bytes,
                        prepack_slots[k].fill_ptr, tile_bytes);
          }
        }
        _stats.gpu_bc_tiles_encoded = _gpu_bc->tiles_encoded();
        _stats.gpu_bc_ms = _gpu_bc->encode_ms();
        packed_by_gpu = true;
      }
    }
    if (!packed_by_gpu) {
      auto pack_one = [&](std::size_t offset) noexcept {
        const std::size_t k = begin_k + offset;
        if (use_prepack) {
          if (slot_disk_ok[k] != 0u) {
            return;
          }
          const auto &s = prepack_slots[k];
          if (s.duplicate_of >= 0 || (s.is_hit && s.hit_ptr != nullptr)) {
            return;
          }
        }
        const auto &pm = _pages[plan.misses[chunk_begin + k].page];
        const auto &src = _sources[pm.source_index];
        const auto mip_view = source_mip_view(src, pm.mip_level);
        std::uint8_t *staging_dst = pack_base_u8 + k * tile_bytes;
        const bool has_fill =
            use_prepack && (prepack_slots[k].fill_ptr != nullptr);
        const luisa::span<std::uint8_t> pack_dst =
            has_fill ? std::span<std::uint8_t>{prepack_slots[k].fill_ptr,
                                               tile_bytes}
                     : std::span<std::uint8_t>{staging_dst, tile_bytes};
        vt_pack_tile_formatted(
            mip_view.unorm8_pixels, mip_view.float_pixels,
            mip_view.width, mip_view.height, pm.tile_x, pm.tile_y, pm.extension,
            _physical_tile_size, _logical_tile_size, _tile_border, pool.format,
            pack_dst);
        if (has_fill) {
          std::memcpy(staging_dst, pack_dst.data(), tile_bytes);
        }
      };
      const std::size_t count = end_k - begin_k;
      const bool is_compressed =
          (pool.format == VtTileFormat::bc1 || pool.format == VtTileFormat::bc3 ||
           pool.format == VtTileFormat::bc4 || pool.format == VtTileFormat::bc5 ||
           pool.format == VtTileFormat::bc6h || pool.format == VtTileFormat::bc7);
      if (count >= (is_compressed ? 2u : 16u)) {
        psycles::parallel_for(count, pack_one);
      } else {
        for (std::size_t offset = 0u; offset < count; ++offset) {
          pack_one(offset);
        }
      }
    }
    _stats.pack_ms += elapsed_ms_since(pack_t0);
  };

  auto decode_unique_parallel = [&](std::span<const std::uint32_t> src_indices) {
    if (src_indices.size() == 1u) {
      decode_source_on_demand(src_indices.front());
      return;
    }
    std::vector<std::uint32_t> decode_now;
    std::vector<std::uint32_t> wait_bg;
    std::vector<std::pair<std::uint32_t, DecodedSourcePayload>> ready_now;
    decode_now.reserve(src_indices.size());
    if (!_bg_source_queued.empty()) {
      std::lock_guard lock(_bg_mutex);
      for (const std::uint32_t idx : src_indices) {
        if (idx >= _sources.size() || _sources[idx].decoded ||
            _sources[idx].load_failed) {
          continue;
        }
        const auto st =
            idx < _bg_source_queued.size() ? _bg_source_queued[idx] : 0u;
        if (st == 3u && _bg_ready_payloads[idx].has_value()) {
          ready_now.emplace_back(idx, std::move(*_bg_ready_payloads[idx]));
          _bg_ready_payloads[idx].reset();
          _bg_source_queued[idx] = 0u;
        } else if (st == 2u) {
          wait_bg.push_back(idx);
        } else {
          if (st == 1u) {
            const auto it = std::find(_bg_pending_queue.begin(),
                                      _bg_pending_queue.end(), idx);
            if (it != _bg_pending_queue.end()) {
              _bg_pending_queue.erase(it);
            }
            _bg_source_queued[idx] = 0u;
          }
          decode_now.push_back(idx);
        }
      }
    } else {
      decode_now.assign(src_indices.begin(), src_indices.end());
    }
    for (auto &item : ready_now) {
      apply_decoded_source_payload(_sources[item.first],
                                   std::move(item.second));
    }
    if (!decode_now.empty()) {
      decode_sources_batch(decode_now);
    }
    for (const std::uint32_t idx : wait_bg) {
      decode_source_on_demand(idx);
    }
  };

  if (_options.host_decoded_cache_budget_bytes == 0ull) {
    if (any_live_pack) {
      std::vector<std::uint32_t> unique_sources;
      std::uint32_t prev_src = std::numeric_limits<std::uint32_t>::max();
      for (std::size_t k = 0u; k < chunk_count; ++k) {
        if (!slot_needs_live_pack(k)) {
          continue;
        }
        const auto src_idx =
            _pages[plan.misses[chunk_begin + k].page].source_index;
        if (src_idx != prev_src) {
          prev_src = src_idx;
          const auto &src = _sources[src_idx];
          if (!src.decoded && !src.load_failed &&
              std::find(unique_sources.begin(), unique_sources.end(),
                        src_idx) == unique_sources.end()) {
            unique_sources.push_back(src_idx);
          }
        }
      }
      decode_unique_parallel(unique_sources);
    }
    if (chunk_end < total_load) {
      const std::size_t lookahead_end =
          std::min(total_load, chunk_end + staging_cap);
      std::uint32_t next_prev_src = std::numeric_limits<std::uint32_t>::max();
      std::size_t queued_cnt = 0u;
      for (std::size_t idx = chunk_end; idx < lookahead_end && queued_cnt < 4u;
           ++idx) {
        const auto pid = plan.misses[idx].page;
        const auto &pm = _pages[pid];
        const auto src_idx = pm.source_index;
        if (src_idx != next_prev_src) {
          next_prev_src = src_idx;
          const auto &src = _sources[src_idx];
          const bool disk_ok =
              use_disk_prepack &&
              _prepack_disk->has_valid_file(_page_prepack_keys[pid].image,
                                            pm.tiles_x, pm.tiles_y);
          if (!src.decoded && !src.load_failed && !disk_ok) {
            enqueue_bg_decode(src_idx);
            ++queued_cnt;
          }
        }
      }
    }
    run_disk_reads();
    copy_prepack_hits_chunk();
    if (any_live_pack) {
      ensure_disk_files_for_range(0u, chunk_count);
    }
    for (std::size_t k = 0u; k < chunk_count; ++k) {
      assign_tile(k);
    }
    if (any_live_pack) {
      pack_range(0u, chunk_count);
    }
  } else {
    {
      const auto first_src =
          _pages[plan.misses[chunk_begin].page].source_index;
      std::uint32_t prev_src = first_src;
      std::size_t queued_cnt = 0u;
      for (std::size_t k = 1u; k < chunk_count && queued_cnt < 1u; ++k) {
        const auto pid = plan.misses[chunk_begin + k].page;
        const auto &pm = _pages[pid];
        const auto src_idx = pm.source_index;
        if (src_idx != prev_src) {
          prev_src = src_idx;
          const auto &src = _sources[src_idx];
          const bool disk_ok =
              use_disk_prepack &&
              _prepack_disk->has_valid_file(_page_prepack_keys[pid].image,
                                            pm.tiles_x, pm.tiles_y);
          if (!src.decoded && !src.load_failed && !disk_ok) {
            enqueue_bg_decode(src_idx);
            ++queued_cnt;
          }
        }
      }
    }
    run_disk_reads();
    copy_prepack_hits_chunk();
    for (std::size_t run_begin = 0u; run_begin < chunk_count;) {
      const auto src_idx =
          _pages[plan.misses[chunk_begin + run_begin].page].source_index;
      std::size_t run_end = run_begin + 1u;
      while (run_end < chunk_count &&
             _pages[plan.misses[chunk_begin + run_end].page].source_index ==
                 src_idx) {
        ++run_end;
      }
      bool run_needs_decode = false;
      for (std::size_t k = run_begin; k < run_end; ++k) {
        if (slot_needs_live_pack(k)) {
          run_needs_decode = true;
          break;
        }
      }
      if (run_needs_decode) {
        decode_source_on_demand(src_idx);
        ensure_disk_files_for_range(run_begin, run_end);
      }
      for (std::size_t k = run_begin; k < run_end; ++k) {
        assign_tile(k);
      }
      if (run_needs_decode) {
        pack_range(run_begin, run_end);
        const bool chunk_continues_src =
            (chunk_end < total_load) &&
            (run_end == chunk_count) &&
            (_pages[plan.misses[chunk_end].page].source_index == src_idx);
        if (!chunk_continues_src) {
          trim_host_decoded_cache(src_idx);
        }
      }
      run_begin = run_end;
    }
  }

  if (use_prepack) {
    const auto dup_t0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0u; k < chunk_count; ++k) {
      const auto &s = prepack_slots[k];
      if (s.duplicate_of < 0) {
        continue;
      }
      const std::size_t owner_k = static_cast<std::size_t>(s.duplicate_of);
      const std::uint8_t *src_bytes = prepack_slots[owner_k].fill_ptr;
      const std::uint8_t *fallback = pack_base_u8 + owner_k * tile_bytes;
      std::memcpy(pack_base_u8 + k * tile_bytes,
                  src_bytes != nullptr ? src_bytes : fallback, tile_bytes);
    }
    _stats.pack_ms += elapsed_ms_since(dup_t0);
    prepack_unpin_guard.release();
  }

  PackedChunkMeta out_chunk;
  out_chunk.pool_index = pool_idx;
  out_chunk.chunk_slot = target.chunk_slot;
  out_chunk.chunk_count = static_cast<std::uint32_t>(chunk_count);
  out_chunk.atlas_counts.assign(num_atlases, 0u);
  for (const auto &a : assigned) {
    ++out_chunk.atlas_counts[a.atlas_idx];
  }
  out_chunk.atlas_offsets.assign(num_atlases, 0u);
  for (std::size_t a = 1u; a < num_atlases; ++a) {
    out_chunk.atlas_offsets[a] =
        out_chunk.atlas_offsets[a - 1u] + out_chunk.atlas_counts[a - 1u];
  }
  auto &atlas_write_cursor = scratch.atlas_write_cursor;
  atlas_write_cursor.resize(num_atlases);
  std::copy(out_chunk.atlas_offsets.begin(), out_chunk.atlas_offsets.end(),
            atlas_write_cursor.begin());
  if (pool.use_batch_copy_upload) {
    out_chunk.atlas_regions.resize(chunk_count);
    const std::uint64_t tile_bytes_u64 = static_cast<std::uint64_t>(tile_bytes);
    for (const auto &a : assigned) {
      const auto pos = atlas_write_cursor[a.atlas_idx]++;
      out_chunk.atlas_regions[pos] = BufferToTextureCopyRegion{
          .buffer_offset =
              static_cast<std::uint64_t>(a.staging_idx) * tile_bytes_u64,
          .buffer_row_length = 0u,
          .buffer_image_height = 0u,
          .mip_level = 0u,
          .texture_offset = luisa::make_uint3(a.dst_ox, a.dst_oy, 0u),
          .texture_extent = luisa::make_uint3(_physical_tile_size,
                                              _physical_tile_size, 1u)};
    }
  } else {
    for (const auto &a : assigned) {
      const auto pos = atlas_write_cursor[a.atlas_idx]++;
      target.um_dst[pos] =
          luisa::make_uint4(a.staging_idx, a.dst_ox, a.dst_oy, a.atlas_idx);
    }
  }
  return out_chunk;
}

void VirtualTextureCache::emit_chunk_atlas_upload(
    CommandList &target_cmd,
    const PackedChunkMeta &chunk,
    std::uint64_t *atlas_bytes_out) noexcept {
  const auto &pr = _pools[chunk.pool_index];
  const auto &pool = pr.config;
  const std::size_t num_atlases = pool.atlases.size();
  const auto &src_buf = _use_host_cached_staging
                            ? pr.device_staging[chunk.chunk_slot]
                            : pr.staging[chunk.chunk_slot];
  const std::uint64_t tile_bytes = pool.tile_bytes;
  const std::uint64_t src_handle = src_buf.handle();
  const PixelStorage storage = pool.atlas_storage;

  for (std::size_t a = 0u; a < num_atlases; ++a) {
    const std::uint32_t count = chunk.atlas_counts[a];
    if (count == 0u) {
      continue;
    }
    const std::uint32_t off = chunk.atlas_offsets[a];
    if (pool.use_batch_copy_upload) {
      luisa::vector<BufferToTextureCopyRegion> regions;
      regions.reserve(count);
      std::uint64_t min_off = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t max_end = 0ull;
      for (std::uint32_t i = 0u; i < count; ++i) {
        const auto &r = chunk.atlas_regions[off + i];
        min_off = std::min(min_off, r.buffer_offset);
        max_end = std::max(max_end, r.buffer_offset + tile_bytes);
        regions.push_back(r);
      }
      target_cmd << luisa::make_unique<BatchBufferToTextureCopyCommand>(
          src_handle, min_off, max_end - min_off, pool.atlases[a].handle(),
          storage, std::move(regions));
    } else if (pool.format == VtTileFormat::float4) {
      target_cmd << _upload_float4_shader(
                        pool.atlases[a], src_buf,
                        pr.upload_meta[chunk.chunk_slot], off,
                        _physical_tile_size)
                        .dispatch(_physical_tile_size, _physical_tile_size,
                                  count);
    } else {
      target_cmd << _upload_byte4_shader(
                        pool.atlases[a], src_buf,
                        pr.upload_meta[chunk.chunk_slot], off,
                        _physical_tile_size)
                        .dispatch(_physical_tile_size, _physical_tile_size,
                                  count);
    }
    if (atlas_bytes_out != nullptr) {
      *atlas_bytes_out += static_cast<std::uint64_t>(count) * tile_bytes;
    }
  }
}

void VirtualTextureCache::append_staged_chunk_h2d(
    CommandList &target_cmd,
    std::uint32_t pool_index,
    std::size_t chunk_slot,
    std::size_t chunk_count,
    std::uint64_t *h2d_bytes_out) noexcept {
  if (!_use_host_cached_staging || !_use_copy_stream || chunk_count == 0u) {
    return;
  }
  const auto &pr = _pools[pool_index];
  const std::size_t elem_count =
      chunk_count * static_cast<std::size_t>(pr.config.tile_staging_u32s);
  target_cmd << pr.staging[chunk_slot]
                    .view(0u, elem_count)
                    .copy_to(pr.device_staging[chunk_slot].view(0u, elem_count));
  if (h2d_bytes_out != nullptr) {
    *h2d_bytes_out +=
        static_cast<std::uint64_t>(elem_count) * sizeof(luisa::uint);
  }
}

std::uint64_t VirtualTextureCache::submit_staged_batch_h2d(
    CommandList &h2d_cmd) noexcept {
  if (!_use_host_cached_staging || !_use_copy_stream || h2d_cmd.empty()) {
    return 0ull;
  }
  std::lock_guard lock{_copy_stream_mtx};
  auto sig = _copy_h2d_event.signal();
  const auto fence = sig.fence;
  if (_supports_stream_signal_event) {
    h2d_cmd << luisa::make_unique<StreamSignalEventCommand>(sig.handle, fence);
    _copy_stream << h2d_cmd.commit();
  } else {
    _copy_stream << h2d_cmd.commit() << std::move(sig);
  }
  return fence;
}

std::uint64_t VirtualTextureCache::submit_staged_chunk_h2d(
    std::uint32_t pool_index,
    std::size_t chunk_slot,
    std::size_t chunk_count,
    bool signal_event,
    std::uint64_t *h2d_bytes_out) noexcept {
  if (!_use_host_cached_staging || !_use_copy_stream || chunk_count == 0u) {
    return 0ull;
  }
  CommandList h2d_cmd;
  append_staged_chunk_h2d(h2d_cmd, pool_index, chunk_slot, chunk_count,
                          h2d_bytes_out);
  if (signal_event) {
    return submit_staged_batch_h2d(h2d_cmd);
  }
  std::lock_guard lock{_copy_stream_mtx};
  _copy_stream << h2d_cmd.commit();
  return 0ull;
}

std::uint64_t VirtualTextureCache::signal_copy_h2d_event() noexcept {
  if (!_use_copy_stream) {
    return 0ull;
  }
  std::lock_guard lock{_copy_stream_mtx};
  auto sig = _copy_h2d_event.signal();
  const auto fence = sig.fence;
  _copy_stream << std::move(sig);
  return fence;
}

void VirtualTextureCache::submit_direct_copy_chunks(
    Stream &stream,
    std::span<const PackedChunkMeta> chunks,
    std::uint64_t *atlas_bytes_out) noexcept {
  if (!_use_copy_stream || chunks.empty()) {
    return;
  }
  CommandList copy_cmd;
  for (const auto &c : chunks) {
    emit_chunk_atlas_upload(copy_cmd, c, atlas_bytes_out);
  }
  if (copy_cmd.empty()) {
    return;
  }
  auto acq = _copy_acquire_event.signal();
  const auto acq_fence = acq.fence;
  stream << std::move(acq);
  std::uint64_t done_fence = 0ull;
  {
    std::lock_guard lock{_copy_stream_mtx};
    auto done = _copy_done_event.signal();
    done_fence = done.fence;
    if (_supports_stream_signal_event) {
      copy_cmd << luisa::make_unique<StreamSignalEventCommand>(done.handle,
                                                               done_fence);
      _copy_stream << _copy_acquire_event.wait(acq_fence) << copy_cmd.commit();
    } else {
      _copy_stream << _copy_acquire_event.wait(acq_fence) << copy_cmd.commit()
                   << std::move(done);
    }
  }
  stream << _copy_done_event.wait(done_fence);
}


} // namespace psycles::luisa_backend::detail
