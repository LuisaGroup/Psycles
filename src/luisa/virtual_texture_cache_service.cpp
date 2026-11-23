#include "virtual_texture_cache.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
using DecodedSourcePayload = VirtualTextureCache::DecodedSourcePayload;

[[nodiscard]] std::int32_t wrap_index(std::int32_t coord, std::int32_t extent,
                                      std::uint32_t ext_mode) noexcept {
  switch (ext_mode) {
    case 0u: { // repeat
      std::int32_t m = coord % extent;
      if (m < 0) {
        m += extent;
      }
      return m;
    }
    case 1u: { // clip -> negative sentinel for out-of-range
      if (coord < 0 || coord >= extent) {
        return -1;
      }
      return coord;
    }
    case 2u: { // extend
      return std::clamp<std::int32_t>(coord, 0, extent - 1);
    }
    case 3u: { // mirror
      const std::int32_t period = extent * 2;
      std::int32_t m = coord % period;
      if (m < 0) {
        m += period;
      }
      if (m >= extent) {
        m = period - 1 - m;
      }
      return std::clamp<std::int32_t>(m, 0, extent - 1);
    }
    default:
      return std::clamp<std::int32_t>(coord, 0, extent - 1);
  }
}

void pack_byte4_tile(
    const VirtualTextureCache::SourceImageEntry &src,
    const VirtualTextureCache::VirtualPageMeta &page,
    std::uint32_t tile_size,
    std::uint32_t logical_tile,
    std::uint32_t border,
    std::span<luisa::uint> dst_tile) noexcept {
  const auto w = static_cast<std::int32_t>(src.width);
  const auto h = static_cast<std::int32_t>(src.height);
  const auto base_x =
      static_cast<std::int32_t>(page.tile_x * logical_tile) -
      static_cast<std::int32_t>(border);
  const auto base_y =
      static_cast<std::int32_t>(page.tile_y * logical_tile) -
      static_cast<std::int32_t>(border);
  const auto P = static_cast<std::int32_t>(tile_size);

  const bool interior_x = (base_x >= 0) && (base_x + P <= w);
  const bool interior_y = (base_y >= 0) && (base_y + P <= h);
  const std::uint8_t *pixels = src.unorm8_pixels.data();
  const std::size_t row_bytes = static_cast<std::size_t>(P) * 4u;

  if (interior_x && interior_y) {
    for (std::int32_t dy = 0; dy < P; ++dy) {
      const auto sy = static_cast<std::size_t>(base_y + dy);
      const auto sx = static_cast<std::size_t>(base_x);
      const std::uint8_t *src_row = pixels + (sy * src.width + sx) * 4u;
      std::memcpy(dst_tile.data() + static_cast<std::size_t>(dy) * P, src_row,
                  row_bytes);
    }
    return;
  }

  std::array<std::int32_t, 128u> sx_table{};
  for (std::int32_t dx = 0; dx < P; ++dx) {
    sx_table[static_cast<std::size_t>(dx)] =
        wrap_index(base_x + dx, w, page.extension);
  }
  for (std::int32_t dy = 0; dy < P; ++dy) {
    const auto sy = wrap_index(base_y + dy, h, page.extension);
    luisa::uint *dst_row =
        dst_tile.data() + static_cast<std::size_t>(dy) * P;
    if (sy < 0) {
      std::memset(dst_row, 0, row_bytes);
      continue;
    }
    const std::uint8_t *src_row =
        pixels + static_cast<std::size_t>(sy) * src.width * 4u;
    if (interior_x) {
      std::memcpy(dst_row, src_row + static_cast<std::size_t>(base_x) * 4u,
                  row_bytes);
    } else {
      for (std::int32_t dx = 0; dx < P; ++dx) {
        const auto sx = sx_table[static_cast<std::size_t>(dx)];
        if (sx < 0) {
          dst_row[static_cast<std::size_t>(dx)] = 0u;
        } else {
          std::uint32_t packed = 0u;
          std::memcpy(&packed, src_row + static_cast<std::size_t>(sx) * 4u, 4u);
          dst_row[static_cast<std::size_t>(dx)] = packed;
        }
      }
    }
  }
}

void pack_float4_tile(
    const VirtualTextureCache::SourceImageEntry &src,
    const VirtualTextureCache::VirtualPageMeta &page,
    std::uint32_t tile_size,
    std::uint32_t logical_tile,
    std::uint32_t border,
    std::span<luisa::float4> dst_tile) noexcept {
  const auto w = static_cast<std::int32_t>(src.width);
  const auto h = static_cast<std::int32_t>(src.height);
  const auto base_x =
      static_cast<std::int32_t>(page.tile_x * logical_tile) -
      static_cast<std::int32_t>(border);
  const auto base_y =
      static_cast<std::int32_t>(page.tile_y * logical_tile) -
      static_cast<std::int32_t>(border);
  const auto P = static_cast<std::int32_t>(tile_size);

  const bool interior_x = (base_x >= 0) && (base_x + P <= w);
  const bool interior_y = (base_y >= 0) && (base_y + P <= h);
  const float *pixels = src.float_pixels.data();
  const std::size_t row_bytes =
      static_cast<std::size_t>(P) * sizeof(luisa::float4);

  if (interior_x && interior_y) {
    for (std::int32_t dy = 0; dy < P; ++dy) {
      const auto sy = static_cast<std::size_t>(base_y + dy);
      const auto sx = static_cast<std::size_t>(base_x);
      const float *src_row = pixels + (sy * src.width + sx) * 4u;
      std::memcpy(dst_tile.data() + static_cast<std::size_t>(dy) * P, src_row,
                  row_bytes);
    }
    return;
  }

  std::array<std::int32_t, 128u> sx_table{};
  for (std::int32_t dx = 0; dx < P; ++dx) {
    sx_table[static_cast<std::size_t>(dx)] =
        wrap_index(base_x + dx, w, page.extension);
  }
  for (std::int32_t dy = 0; dy < P; ++dy) {
    const auto sy = wrap_index(base_y + dy, h, page.extension);
    luisa::float4 *dst_row =
        dst_tile.data() + static_cast<std::size_t>(dy) * P;
    if (sy < 0) {
      std::memset(dst_row, 0, row_bytes);
      continue;
    }
    const float *src_row =
        pixels + static_cast<std::size_t>(sy) * src.width * 4u;
    if (interior_x) {
      std::memcpy(dst_row, src_row + static_cast<std::size_t>(base_x) * 4u,
                  row_bytes);
    } else {
      for (std::int32_t dx = 0; dx < P; ++dx) {
        const auto sx = sx_table[static_cast<std::size_t>(dx)];
        if (sx < 0) {
          dst_row[static_cast<std::size_t>(dx)] = luisa::make_float4(0.0f);
        } else {
          const float *p = src_row + static_cast<std::size_t>(sx) * 4u;
          dst_row[static_cast<std::size_t>(dx)] =
              luisa::make_float4(p[0], p[1], p[2], p[3]);
        }
      }
    }
  }
}

} // namespace

void VirtualTextureCache::wait_all_upload_fences() noexcept {
  for (std::size_t i = 0u; i < kUploadRingSize; ++i) {
    if (_chunk_slot_fence[i] != 0ull) {
      _upload_event.synchronize(_chunk_slot_fence[i]);
      _chunk_slot_fence[i] = 0ull;
    }
    _chunk_slot_stream[i] = 0ull;
    if (_meta_slot_fence[i] != 0ull) {
      _upload_event.synchronize(_meta_slot_fence[i]);
      _meta_slot_fence[i] = 0ull;
    }
    _meta_slot_stream[i] = 0ull;
  }
}

std::size_t VirtualTextureCache::acquire_chunk_slot(
    std::uint64_t current_stream_handle) noexcept {
  std::size_t slot = _chunk_ring_cursor;
  for (std::size_t k = 0u; k < kUploadRingSize; ++k) {
    const std::size_t cand = (_chunk_ring_cursor + k) % kUploadRingSize;
    if (_chunk_slot_fence[cand] == 0ull &&
        (_chunk_slot_stream[cand] == 0ull ||
         _chunk_slot_stream[cand] == current_stream_handle)) {
      slot = cand;
      break;
    }
  }
  _chunk_ring_cursor = (slot + 1u) % kUploadRingSize;
  if (_chunk_slot_fence[slot] != 0ull) {
    _upload_event.synchronize(_chunk_slot_fence[slot]);
    _chunk_slot_fence[slot] = 0ull;
  }
  _chunk_slot_stream[slot] = 0ull;
  return slot;
}

std::size_t VirtualTextureCache::acquire_meta_slot(
    std::uint64_t current_stream_handle) noexcept {
  std::size_t slot = _meta_ring_cursor;
  for (std::size_t k = 0u; k < kUploadRingSize; ++k) {
    const std::size_t cand = (_meta_ring_cursor + k) % kUploadRingSize;
    if (_meta_slot_fence[cand] == 0ull &&
        (_meta_slot_stream[cand] == 0ull ||
         _meta_slot_stream[cand] == current_stream_handle)) {
      slot = cand;
      break;
    }
  }
  _meta_ring_cursor = (slot + 1u) % kUploadRingSize;
  if (_meta_slot_fence[slot] != 0ull) {
    _upload_event.synchronize(_meta_slot_fence[slot]);
    _meta_slot_fence[slot] = 0ull;
  }
  _meta_slot_stream[slot] = 0ull;
  return slot;
}

void VirtualTextureCache::service_aux_misses(Stream &stream) noexcept {
  if (!has_virtual_images()) {
    return;
  }
  service_misses_impl(stream, MissSource::aux_demand);
}

void VirtualTextureCache::service_bitset_misses(Stream &stream) noexcept {
  if (!has_virtual_images()) {
    return;
  }
  service_misses_impl(stream, MissSource::bitset_only);
}

void VirtualTextureCache::service_misses_impl(
    Stream &stream,
    MissSource source,
    CommandList *wavefront_cmd) noexcept {
  if (_pages.empty()) {
    if (wavefront_cmd != nullptr && !wavefront_cmd->empty()) {
      stream << wavefront_cmd->commit() << synchronize();
    }
    return;
  }
  const auto collect_t0 = std::chrono::steady_clock::now();
  const bool need_epoch_readback =
      _byte4_pool.eviction_possible || _float4_pool.eviction_possible;
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  const auto bitset_words = static_cast<luisa::uint>(_host_miss_bitset.size());
  const auto total_slots = static_cast<luisa::uint>(_host_slot_epoch.size());
  const bool is_aux = (source == MissSource::aux_demand);
  const bool is_wavefront = (source == MissSource::wavefront_demand);
  if (!is_aux) {
    _pool_aux_streak_start_seq[0u] = std::numeric_limits<std::uint64_t>::max();
    _pool_aux_streak_start_seq[1u] = std::numeric_limits<std::uint64_t>::max();
  }

  CommandList local_cmd;
  CommandList &cmd = (wavefront_cmd != nullptr) ? *wavefront_cmd : local_cmd;
  switch (source) {
    case MissSource::wavefront_demand:
      if (_use_pinned_readback) {
        cmd << _demand_hist.view().copy_to(_readback_demand_hist.view())
            << _miss_bitset.view().copy_to(_readback_miss_bitset.view());
      } else {
        cmd << _demand_hist.copy_to(luisa::span{_host_demand_hist})
            << _miss_bitset.copy_to(luisa::span{_host_miss_bitset});
      }
      break;
    case MissSource::aux_demand:
      if (_use_pinned_readback) {
        cmd << _aux_demand_hist.view().copy_to(
            _readback_aux_demand_hist.view());
      } else {
        cmd << _aux_demand_hist.copy_to(luisa::span{_host_aux_demand_hist});
      }
      break;
    case MissSource::bitset_only:
      if (_use_pinned_readback) {
        cmd << _miss_bitset.view().copy_to(_readback_miss_bitset.view());
      } else {
        cmd << _miss_bitset.copy_to(luisa::span{_host_miss_bitset});
      }
      break;
  }
  if (need_epoch_readback) {
    if (_use_pinned_readback) {
      cmd << _slot_epoch.view().copy_to(_readback_slot_epoch.view());
    } else {
      cmd << _slot_epoch.copy_to(luisa::span{_host_slot_epoch});
    }
  }
  stream << cmd.commit() << synchronize();
  const auto cur_stream_h = static_cast<std::uint64_t>(stream.handle());
  for (std::size_t i = 0u; i < kUploadRingSize; ++i) {
    if (_chunk_slot_stream[i] == cur_stream_h) {
      _chunk_slot_stream[i] = 0ull;
    }
    if (_meta_slot_stream[i] == cur_stream_h) {
      _meta_slot_stream[i] = 0ull;
    }
  }
  if (_use_pinned_readback) {
    switch (source) {
      case MissSource::wavefront_demand:
        _pinned_ext->flush_range(
            _readback_demand_hist.handle(), 0ull,
            static_cast<std::uint64_t>(total_pages) * sizeof(luisa::uint));
        _pinned_ext->flush_range(
            _readback_miss_bitset.handle(), 0ull,
            static_cast<std::uint64_t>(bitset_words) * sizeof(luisa::uint));
        break;
      case MissSource::aux_demand:
        _pinned_ext->flush_range(
            _readback_aux_demand_hist.handle(), 0ull,
            static_cast<std::uint64_t>(total_pages) * sizeof(luisa::uint));
        break;
      case MissSource::bitset_only:
        _pinned_ext->flush_range(
            _readback_miss_bitset.handle(), 0ull,
            static_cast<std::uint64_t>(bitset_words) * sizeof(luisa::uint));
        break;
    }
    if (need_epoch_readback) {
      _pinned_ext->flush_range(
          _readback_slot_epoch.handle(), 0ull,
          static_cast<std::uint64_t>(total_slots) * sizeof(luisa::uint));
    }
  }
  const auto collect_t1 = std::chrono::steady_clock::now();
  _stats.collect_sync_ms +=
      std::chrono::duration<double, std::milli>(collect_t1 - collect_t0)
          .count();

  const luisa::uint *demand_hist_read =
      _use_pinned_readback ? _pinned_demand_hist_ptr : _host_demand_hist.data();
  const luisa::uint *aux_demand_hist_read =
      _use_pinned_readback ? _pinned_aux_demand_hist_ptr
                           : _host_aux_demand_hist.data();
  const luisa::uint *miss_bitset_read =
      _use_pinned_readback ? _pinned_miss_bitset_ptr : _host_miss_bitset.data();
  const luisa::uint *slot_epoch_read =
      _use_pinned_readback ? _pinned_slot_epoch_ptr : _host_slot_epoch.data();

  bool any_miss_bit = false;
  if (is_wavefront || source == MissSource::bitset_only) {
    for (std::uint32_t wi = 0u; wi < bitset_words; ++wi) {
      if (miss_bitset_read[wi] != 0u) {
        any_miss_bit = true;
        break;
      }
    }
  }

  struct MissCandidate {
    std::uint32_t page{};
    std::uint32_t primary_demand{};
  };
  std::vector<MissCandidate> b4_misses;
  std::vector<MissCandidate> f4_misses;
  std::vector<MissCandidate> b4_sec_misses;
  std::vector<MissCandidate> f4_sec_misses;
  if (source == MissSource::bitset_only) {
    if (any_miss_bit) {
      for (std::uint32_t p = 0u; p < _pages.size(); ++p) {
        const bool bit_set =
            ((miss_bitset_read[p >> 5u] >> (p & 31u)) & 1u) != 0u;
        if (!bit_set || (_host_page_table[p] & vt_page_resident_bit) != 0u) {
          continue;
        }
        auto &dst = _pages[p].is_float4 ? f4_misses : b4_misses;
        dst.push_back(MissCandidate{.page = p, .primary_demand = 0u});
      }
    }
  } else if (is_aux) {
    for (std::uint32_t p = 0u; p < _pages.size(); ++p) {
      const auto primary = aux_demand_hist_read[p];
      if (primary == 0u || (_host_page_table[p] & vt_page_resident_bit) != 0u) {
        continue;
      }
      auto &dst = _pages[p].is_float4 ? f4_misses : b4_misses;
      dst.push_back(MissCandidate{.page = p, .primary_demand = primary});
    }
  } else {
    for (std::uint32_t p = 0u; p < _pages.size(); ++p) {
      if ((_host_page_table[p] & vt_page_resident_bit) != 0u) {
        continue;
      }
      const auto primary = demand_hist_read[p];
      const bool bit_set =
          any_miss_bit &&
          (((miss_bitset_read[p >> 5u] >> (p & 31u)) & 1u) != 0u);
      if (primary != 0u) {
        auto &dst = _pages[p].is_float4 ? f4_misses : b4_misses;
        dst.push_back(MissCandidate{.page = p, .primary_demand = primary});
      } else if (bit_set) {
        auto &dst = _pages[p].is_float4 ? f4_sec_misses : b4_sec_misses;
        dst.push_back(MissCandidate{.page = p, .primary_demand = 0u});
      }
    }
  }

  if (b4_misses.empty() && f4_misses.empty() &&
      b4_sec_misses.empty() && f4_sec_misses.empty()) {
    if (is_aux) {
      CommandList local_clear_cmd;
      CommandList &clear_cmd =
          (wavefront_cmd != nullptr) ? *wavefront_cmd : local_clear_cmd;
      clear_cmd << _clear_u32_shader(_aux_demand_hist, total_pages)
                       .dispatch(total_pages);
      if (wavefront_cmd == nullptr) {
        stream << local_clear_cmd.commit();
      }
    } else if (any_miss_bit) {
      if (!_use_pinned_readback) {
        std::fill(_host_miss_bitset.begin(), _host_miss_bitset.end(), 0u);
      }
      CommandList local_clear_cmd;
      CommandList &clear_cmd =
          (wavefront_cmd != nullptr) ? *wavefront_cmd : local_clear_cmd;
      clear_cmd << _clear_u32_shader(_miss_bitset, bitset_words)
                       .dispatch(bitset_words);
      if (wavefront_cmd == nullptr) {
        stream << local_clear_cmd.commit();
      }
    }
    return;
  }

  std::vector<std::uint32_t> src_hit_slots(_sources.size(), 0u);
  std::vector<std::uint8_t> src_partner_hit(_sources.size(), 0u);
  auto mark_pool_active_sources =
      [&](std::uint32_t pool_idx, const PoolConfig &pool,
          const luisa::vector<luisa::uint> &host_slots,
          const std::vector<PoolSlotMeta> &slot_meta) {
        const std::uint32_t prev_ep = _host_current_epoch[pool_idx];
        const std::uint64_t next_wf =
            _pool_wavefront_seq[pool_idx] + (is_aux ? 0ull : 1ull);
        for (std::uint32_t s = 0u; s < pool.capacity; ++s) {
          const std::uint32_t p = host_slots[s];
          if (p == vt_invalid_page) {
            continue;
          }
          const bool active =
              !pool.eviction_possible ||
              (slot_epoch_read[pool.epoch_offset + s] == prev_ep) ||
              (slot_epoch_read[pool.epoch_offset + s] + 1u == prev_ep &&
               next_wf <= slot_meta[s].load_wf + 1ull);
          if (active) {
            const std::uint32_t src_idx = _pages[p].source_index;
            if (src_idx < src_hit_slots.size()) {
              ++src_hit_slots[src_idx];
            }
          }
        }
      };
  mark_pool_active_sources(0u, _byte4_pool, _host_byte4_slots,
                           _host_byte4_slot_meta);
  mark_pool_active_sources(1u, _float4_pool, _host_float4_slots,
                           _host_float4_slot_meta);
  auto propagate_partners = [&](std::uint32_t src_idx) {
    if (src_idx >= _source_partners.size()) {
      return;
    }
    for (const auto partner : _source_partners[src_idx]) {
      if (partner < src_partner_hit.size()) {
        src_partner_hit[partner] = 1u;
      }
    }
  };
  for (std::uint32_t s = 0u; s < src_hit_slots.size(); ++s) {
    if (src_hit_slots[s] > 0u) {
      propagate_partners(s);
    }
  }

  const std::size_t meta_slot = acquire_meta_slot(cur_stream_h);
  luisa::uint2 *pt_upd_dst =
      _use_pinned_upload ? _pinned_page_table_updates_ptr[meta_slot] : nullptr;
  luisa::uint2 *ep_upd_dst =
      _use_pinned_upload ? _pinned_slot_epoch_updates_ptr[meta_slot] : nullptr;
  std::size_t pt_upd_count = 0u;
  std::size_t ep_upd_count = 0u;
  if (!_use_pinned_upload) {
    _host_page_table_updates.clear();
    _host_slot_epoch_updates.clear();
  }
  auto push_pt_update = [&](luisa::uint page, luisa::uint val) noexcept {
    const auto u = luisa::make_uint2(page, val);
    if (pt_upd_dst != nullptr) {
      pt_upd_dst[pt_upd_count++] = u;
    } else {
      _host_page_table_updates.push_back(u);
      ++pt_upd_count;
    }
  };
  auto push_ep_update = [&](luisa::uint idx, luisa::uint val) noexcept {
    const auto u = luisa::make_uint2(idx, val);
    if (ep_upd_dst != nullptr) {
      ep_upd_dst[ep_upd_count++] = u;
    } else {
      _host_slot_epoch_updates.push_back(u);
      ++ep_upd_count;
    }
  };

  const std::uint32_t tile_texels = _physical_tile_size * _physical_tile_size;
  bool gpu_state_dirty = false;
  bool any_epoch_advanced = false;
  bool any_epoch_wrapped = false;
  bool slot_epoch_host_seeded = !_use_pinned_readback;

  auto decode_unique_parallel = [&](std::span<const std::uint32_t> source_indices) {
    if (source_indices.size() == 1u) {
      decode_source_on_demand(source_indices.front());
      return;
    }
    std::vector<std::uint32_t> decode_now;
    std::vector<std::uint32_t> wait_bg;
    std::vector<std::pair<std::uint32_t, DecodedSourcePayload>> ready_now;
    decode_now.reserve(source_indices.size());
    if (!_bg_source_queued.empty()) {
      std::lock_guard lock(_bg_mutex);
      for (const std::uint32_t idx : source_indices) {
        if (idx >= _sources.size() || _sources[idx].decoded ||
            _sources[idx].load_failed) {
          continue;
        }
        const auto st = idx < _bg_source_queued.size()
                            ? _bg_source_queued[idx]
                            : 0u;
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
      decode_now.assign(source_indices.begin(), source_indices.end());
    }
    for (auto &item : ready_now) {
      apply_decoded_source_payload(_sources[item.first], std::move(item.second));
    }
    if (!decode_now.empty()) {
      decode_sources_batch(decode_now);
    }
    for (const std::uint32_t idx : wait_bg) {
      decode_source_on_demand(idx);
    }
  };

  auto process_pool = [&](std::uint32_t pool_idx,
                          const PoolConfig &pool,
                          luisa::vector<luisa::uint> &host_slots,
                          std::vector<PoolSlotMeta> &slot_meta,
                          std::uint32_t &used_slots,
                          std::vector<MissCandidate> &misses,
                          std::vector<MissCandidate> &sec_misses,
                          bool is_final_pool) {
    if ((misses.empty() && sec_misses.empty()) || pool.capacity == 0u ||
        pool.atlases.empty()) {
      return;
    }

    const std::uint32_t cap = pool.capacity;
    std::vector<std::uint32_t> victim_slots;
    std::uint32_t new_slot_epoch = 0u;
    std::uint64_t wf_seq = _pool_wavefront_seq[pool_idx];
    std::uint64_t load_seq = _pool_load_seq[pool_idx];

    if (!pool.eviction_possible) {
      if (!sec_misses.empty()) {
        _stats.secondary_misses_coalesced += sec_misses.size();
        misses.insert(misses.end(), sec_misses.begin(), sec_misses.end());
      }
      victim_slots.resize(misses.size());
      for (std::size_t k = 0u; k < misses.size(); ++k) {
        victim_slots[k] = used_slots++;
      }
      for (const auto &m : misses) {
        propagate_partners(_pages[m.page].source_index);
      }
    } else {
      const std::uint32_t prev_gpu_epoch = _host_current_epoch[pool_idx];
      const std::uint64_t next_wf = wf_seq + (is_aux ? 0ull : 1ull);
      const std::size_t free_slots =
          used_slots < cap ? static_cast<std::size_t>(cap - used_slots) : 0u;

      std::vector<std::uint8_t> aux_streak_protected(cap, 0u);
      if (is_aux) {
        const std::uint64_t streak_start = _pool_aux_streak_start_seq[pool_idx];
        const std::uint32_t protect_cap = cap / 2u;
        if (streak_start != std::numeric_limits<std::uint64_t>::max() &&
            protect_cap > 0u) {
          std::vector<std::uint32_t> streak_slots;
          streak_slots.reserve(cap);
          for (std::uint32_t s = 0u; s < cap; ++s) {
            if (host_slots[s] != vt_invalid_page &&
                slot_meta[s].loaded_by_aux &&
                slot_meta[s].load_seq >= streak_start) {
              streak_slots.push_back(s);
            }
          }
          if (streak_slots.size() > protect_cap) {
            std::nth_element(
                streak_slots.begin(), streak_slots.begin() + protect_cap,
                streak_slots.end(), [&](std::uint32_t a, std::uint32_t b) {
                  if (slot_meta[a].load_seq != slot_meta[b].load_seq) {
                    return slot_meta[a].load_seq > slot_meta[b].load_seq;
                  }
                  return a > b;
                });
            streak_slots.resize(protect_cap);
          }
          for (const std::uint32_t s : streak_slots) {
            aux_streak_protected[s] = 1u;
          }
        }
      }

      struct VictimSlot {
        std::uint32_t slot{};
        std::uint8_t tier{};
        std::uint32_t hotness{};
        std::uint32_t initial_demand{};
        std::uint64_t last_hit_wf{};
        std::uint64_t load_seq{};
      };
      std::vector<VictimSlot> candidates(cap);
      std::vector<std::uint32_t> new_hotness(cap, 0u);
      std::vector<std::uint64_t> new_last_hit_wf(cap, 0ull);
      std::uint32_t safe_count = 0u;
      for (std::uint32_t s = 0u; s < cap; ++s) {
        VictimSlot v{};
        v.slot = s;
        if (host_slots[s] == vt_invalid_page) {
          v.tier = 0u;
          ++safe_count;
        } else {
          const std::uint32_t ep = slot_epoch_read[pool.epoch_offset + s];
          const auto &meta = slot_meta[s];
          std::uint32_t updated_hot = meta.hotness;
          std::uint64_t updated_hit_wf = meta.last_hit_wf;
          if (is_aux) {
            if (aux_streak_protected[s] != 0u) {
              v.tier = 4u;
            } else if (ep == prev_gpu_epoch ||
                       (ep + 1u == prev_gpu_epoch &&
                        next_wf <= meta.load_wf + 2ull)) {
              v.tier = 3u;
            } else if (next_wf <=
                       std::max(meta.last_hit_wf, meta.load_wf) + 3ull) {
              v.tier = 2u;
            } else {
              v.tier = 1u;
              ++safe_count;
            }
          } else {
            if (ep == prev_gpu_epoch) {
              updated_hit_wf = next_wf;
              updated_hot = std::min(meta.hotness + 1u, 32u);
              v.tier = 3u;
            } else if (ep + 1u == prev_gpu_epoch &&
                       next_wf <= meta.load_wf + 2ull) {
              v.tier = 3u;
            } else if (next_wf <=
                       std::max(meta.last_hit_wf, meta.load_wf) + 3ull) {
              if (next_wf >= meta.last_hit_wf + 2ull) {
                updated_hot = (meta.hotness * 3u) / 4u;
              }
              v.tier = 2u;
            } else {
              updated_hot = meta.hotness >> 1u;
              v.tier = 1u;
              ++safe_count;
            }
          }
          new_hotness[s] = updated_hot;
          new_last_hit_wf[s] = updated_hit_wf;
          v.hotness = updated_hot;
          v.initial_demand = meta.initial_demand;
          v.last_hit_wf = updated_hit_wf;
          v.load_seq = meta.load_seq;
        }
        candidates[s] = v;
      }

      if (!is_aux) {
        const std::uint32_t hot_pin_cap = cap / 4u;
        if (hot_pin_cap > 0u) {
          std::vector<std::uint32_t> hot_slots;
          hot_slots.reserve(cap);
          for (std::uint32_t s = 0u; s < cap; ++s) {
            if (host_slots[s] != vt_invalid_page &&
                candidates[s].tier >= 2u &&
                candidates[s].hotness >= 3u &&
                candidates[s].last_hit_wf > 0ull &&
                next_wf <= candidates[s].last_hit_wf + 2ull) {
              hot_slots.push_back(s);
            }
          }
          if (hot_slots.size() > hot_pin_cap) {
            std::nth_element(
                hot_slots.begin(), hot_slots.begin() + hot_pin_cap,
                hot_slots.end(), [&](std::uint32_t a, std::uint32_t b) {
                  return std::tie(candidates[a].hotness,
                                  candidates[a].last_hit_wf,
                                  candidates[a].initial_demand,
                                  candidates[a].load_seq, a) >
                         std::tie(candidates[b].hotness,
                                  candidates[b].last_hit_wf,
                                  candidates[b].initial_demand,
                                  candidates[b].load_seq, b);
                });
            hot_slots.resize(hot_pin_cap);
          }
          for (const std::uint32_t s : hot_slots) {
            candidates[s].tier = 4u;
          }
        }
      }

      const std::uint32_t min_progress =
          pool.is_float4
              ? std::max<std::uint32_t>(1u, cap / 4u)
              : std::max<std::uint32_t>(1u, cap / 6u);
      const std::uint32_t batch_budget =
          std::min(cap, std::max(safe_count, min_progress));

      if (misses.empty()) {
        std::size_t partner_sec_count = 0u;
        for (const auto &m : sec_misses) {
          const auto src_idx = _pages[m.page].source_index;
          if (src_idx < src_partner_hit.size() &&
              src_partner_hit[src_idx] != 0u) {
            ++partner_sec_count;
          }
        }
        if (partner_sec_count == 0u && safe_count == 0u) {
          return;
        }
      }

      load_seq = ++_pool_load_seq[pool_idx];
      wf_seq = next_wf;
      if (!is_aux) {
        _pool_wavefront_seq[pool_idx] = wf_seq;
        const bool wrapped = (prev_gpu_epoch >= 0xfffffffcu);
        const std::uint32_t next_gpu_epoch =
            wrapped ? 2u : (prev_gpu_epoch + 2u);
        _host_current_epoch[pool_idx] = next_gpu_epoch;
        new_slot_epoch = next_gpu_epoch - 1u;
        any_epoch_advanced = true;
        if (wrapped) {
          if (!slot_epoch_host_seeded) {
            std::copy_n(_pinned_slot_epoch_ptr, total_slots,
                        _host_slot_epoch.data());
            slot_epoch_host_seeded = true;
          }
          any_epoch_wrapped = true;
          std::fill_n(_host_slot_epoch.data() + pool.epoch_offset, cap, 0u);
        }
      } else {
        if (_pool_aux_streak_start_seq[pool_idx] ==
            std::numeric_limits<std::uint64_t>::max()) {
          _pool_aux_streak_start_seq[pool_idx] = load_seq;
        }
        new_slot_epoch =
            (prev_gpu_epoch > 0u) ? (prev_gpu_epoch - 1u) : 0u;
      }
      for (std::uint32_t s = 0u; s < cap; ++s) {
        if (host_slots[s] != vt_invalid_page) {
          slot_meta[s].hotness = new_hotness[s];
          slot_meta[s].last_hit_wf = new_last_hit_wf[s];
        }
      }

      if (misses.size() > batch_budget) {
        if (source == MissSource::bitset_only) {
          const auto salt =
              static_cast<std::uint32_t>(load_seq * 2246822519ull);
          std::nth_element(
              misses.begin(), misses.begin() + batch_budget, misses.end(),
              [salt](const MissCandidate &a, const MissCandidate &b) {
                const auto ha = (a.page * 2654435761u) ^ salt;
                const auto hb = (b.page * 2654435761u) ^ salt;
                if (ha != hb) {
                  return ha < hb;
                }
                return a.page < b.page;
              });
          misses.resize(batch_budget);
        } else {
          std::vector<std::uint64_t> src_demand_sum(_sources.size(), 0ull);
          std::vector<std::uint32_t> src_miss_cnt(_sources.size(), 0u);
          for (const auto &m : misses) {
            const auto src_idx = _pages[m.page].source_index;
            if (src_idx < src_demand_sum.size()) {
              src_demand_sum[src_idx] += m.primary_demand;
              ++src_miss_cnt[src_idx];
            }
          }
          std::vector<std::uint64_t> src_avg_demand(_sources.size(), 0ull);
          for (std::size_t s = 0u; s < _sources.size(); ++s) {
            if (src_miss_cnt[s] != 0u) {
              src_avg_demand[s] = src_demand_sum[s] / src_miss_cnt[s];
            }
          }
          struct RankedMiss {
            std::uint32_t page{};
            std::uint32_t primary_demand{};
            std::uint32_t source_index{};
            std::uint32_t bx{};
            std::uint32_t by{};
            std::uint64_t block_avg_demand{};
          };
          std::vector<RankedMiss> ranked;
          ranked.reserve(misses.size());
          for (const auto &m : misses) {
            const auto &pm = _pages[m.page];
            ranked.push_back(RankedMiss{
                .page = m.page,
                .primary_demand = m.primary_demand,
                .source_index = pm.source_index,
                .bx = pm.tile_x >> 3u,
                .by = pm.tile_y >> 3u,
                .block_avg_demand = 0ull});
          }
          std::sort(ranked.begin(), ranked.end(),
                    [](const RankedMiss &a, const RankedMiss &b) {
                      return std::tie(a.source_index, a.by, a.bx, a.page) <
                             std::tie(b.source_index, b.by, b.bx, b.page);
                    });
          for (std::size_t run_begin = 0u; run_begin < ranked.size();) {
            std::uint64_t bsum = 0ull;
            std::size_t run_end = run_begin;
            const auto rs = ranked[run_begin].source_index;
            const auto rby = ranked[run_begin].by;
            const auto rbx = ranked[run_begin].bx;
            while (run_end < ranked.size() &&
                   ranked[run_end].source_index == rs &&
                   ranked[run_end].by == rby &&
                   ranked[run_end].bx == rbx) {
              bsum += ranked[run_end].primary_demand;
              ++run_end;
            }
            const std::uint64_t bavg =
                bsum / static_cast<std::uint64_t>(run_end - run_begin);
            for (std::size_t idx = run_begin; idx < run_end; ++idx) {
              ranked[idx].block_avg_demand = bavg;
            }
            run_begin = run_end;
          }
          auto miss_prio_src = [&](std::uint32_t src_idx) noexcept -> std::uint8_t {
            if (src_idx < src_partner_hit.size() &&
                src_partner_hit[src_idx] != 0u) {
              return 3u;
            }
            if (src_idx < src_hit_slots.size() && src_hit_slots[src_idx] > 0u) {
              return 2u;
            }
            return 1u;
          };
          auto ranked_better = [&](const RankedMiss &a,
                                   const RankedMiss &b) noexcept {
            const auto pa = miss_prio_src(a.source_index);
            const auto pb = miss_prio_src(b.source_index);
            if (pa != pb) {
              return pa > pb;
            }
            const auto da = a.source_index < src_avg_demand.size()
                                ? src_avg_demand[a.source_index]
                                : 0ull;
            const auto db = b.source_index < src_avg_demand.size()
                                ? src_avg_demand[b.source_index]
                                : 0ull;
            if (da != db) {
              return da > db;
            }
            if (a.source_index != b.source_index) {
              return a.source_index < b.source_index;
            }
            if (a.block_avg_demand != b.block_avg_demand) {
              return a.block_avg_demand > b.block_avg_demand;
            }
            if (a.by != b.by) {
              return a.by < b.by;
            }
            if (a.bx != b.bx) {
              return a.bx < b.bx;
            }
            if (a.primary_demand != b.primary_demand) {
              return a.primary_demand > b.primary_demand;
            }
            return a.page < b.page;
          };
          std::nth_element(ranked.begin(), ranked.begin() + batch_budget,
                           ranked.end(), ranked_better);
          misses.resize(batch_budget);
          for (std::size_t i = 0u; i < batch_budget; ++i) {
            misses[i] = MissCandidate{
                .page = ranked[i].page,
                .primary_demand = ranked[i].primary_demand};
          }
          if (_options.host_decoded_cache_budget_bytes == 0ull) {
            std::array<std::uint32_t, 2u> top_src{
                std::numeric_limits<std::uint32_t>::max(),
                std::numeric_limits<std::uint32_t>::max()};
            std::array<std::uint64_t, 2u> top_score{0ull, 0ull};
            for (std::size_t i = batch_budget; i < ranked.size(); ++i) {
              const auto src_idx = ranked[i].source_index;
              if (src_idx >= _sources.size() || _sources[src_idx].decoded ||
                  _sources[src_idx].load_failed) {
                continue;
              }
              const std::uint64_t score =
                  (static_cast<std::uint64_t>(miss_prio_src(src_idx)) << 48u) |
                  std::min<std::uint64_t>(ranked[i].block_avg_demand,
                                          (1ull << 48u) - 1ull);
              if (src_idx == top_src[0u]) {
                top_score[0u] = std::max(top_score[0u], score);
              } else if (src_idx == top_src[1u]) {
                top_score[1u] = std::max(top_score[1u], score);
              } else if (score > top_score[0u]) {
                top_src[1u] = top_src[0u];
                top_score[1u] = top_score[0u];
                top_src[0u] = src_idx;
                top_score[0u] = score;
              } else if (score > top_score[1u]) {
                top_src[1u] = src_idx;
                top_score[1u] = score;
              }
            }
            for (const auto src_idx : top_src) {
              if (src_idx < _sources.size()) {
                enqueue_bg_decode(src_idx);
              }
            }
          }
        }
      }

      std::vector<std::uint8_t> src_admitted_now(_sources.size(), 0u);
      for (const auto &m : misses) {
        const auto src_idx = _pages[m.page].source_index;
        if (src_idx < src_admitted_now.size()) {
          src_admitted_now[src_idx] = 1u;
          propagate_partners(src_idx);
        }
      }

      if (misses.size() < batch_budget && !sec_misses.empty()) {
        std::vector<MissCandidate> sec_partners;
        std::vector<MissCandidate> sec_other;
        for (const auto &m : sec_misses) {
          const auto src_idx = _pages[m.page].source_index;
          const bool is_partner =
              (src_idx < src_partner_hit.size() &&
               src_partner_hit[src_idx] != 0u) ||
              (src_idx < src_admitted_now.size() &&
               src_admitted_now[src_idx] != 0u);
          if (is_partner) {
            sec_partners.push_back(m);
          } else {
            sec_other.push_back(m);
          }
        }
        auto sec_less = [&](const MissCandidate &a, const MissCandidate &b) {
          const auto &pa = _pages[a.page];
          const auto &pb = _pages[b.page];
          return std::tuple{pa.source_index, pa.tile_y >> 3u, pa.tile_x >> 3u,
                            a.page} <
                 std::tuple{pb.source_index, pb.tile_y >> 3u, pb.tile_x >> 3u,
                            b.page};
        };
        const std::size_t budget_rem = batch_budget - misses.size();
        const std::size_t take_p = std::min(sec_partners.size(), budget_rem);
        if (take_p > 0u) {
          if (take_p < sec_partners.size()) {
            std::nth_element(sec_partners.begin(),
                             sec_partners.begin() + take_p, sec_partners.end(),
                             sec_less);
          }
          misses.insert(misses.end(), sec_partners.begin(),
                        sec_partners.begin() + take_p);
          _stats.secondary_misses_coalesced += take_p;
        }
        const std::size_t safe_cap =
            std::min<std::size_t>(batch_budget, safe_count);
        const std::size_t other_rem =
            misses.size() < safe_cap ? (safe_cap - misses.size()) : 0u;
        const std::size_t take_o = std::min(sec_other.size(), other_rem);
        if (take_o > 0u) {
          if (take_o < sec_other.size()) {
            std::nth_element(sec_other.begin(), sec_other.begin() + take_o,
                             sec_other.end(), sec_less);
          }
          misses.insert(misses.end(), sec_other.begin(),
                        sec_other.begin() + take_o);
          _stats.secondary_misses_coalesced += take_o;
        }
        for (std::size_t idx = misses.size() - (take_p + take_o);
             idx < misses.size(); ++idx) {
          propagate_partners(_pages[misses[idx].page].source_index);
        }
      }

      if (misses.empty()) {
        return;
      }

      const std::size_t max_load = misses.size();
      if (max_load <= free_slots) {
        victim_slots.resize(max_load);
        for (std::size_t k = 0u; k < max_load; ++k) {
          victim_slots[k] = used_slots++;
        }
      } else {
        auto victim_less = [](const VictimSlot &a, const VictimSlot &b) {
          if (a.tier != b.tier) {
            return a.tier < b.tier;
          }
          switch (a.tier) {
            case 0u:
              return a.slot < b.slot;
            case 1u:
              return std::tie(a.last_hit_wf, a.hotness, a.initial_demand,
                              a.slot) <
                     std::tie(b.last_hit_wf, b.hotness, b.initial_demand,
                              b.slot);
            case 2u:
              return std::tie(a.hotness, a.last_hit_wf, a.load_seq,
                              a.initial_demand, a.slot) <
                     std::tie(b.hotness, b.last_hit_wf, b.load_seq,
                              b.initial_demand, b.slot);
            case 3u:
              return std::tie(a.load_seq, a.hotness, a.initial_demand, a.slot) <
                     std::tie(b.load_seq, b.hotness, b.initial_demand, b.slot);
            default:
              return std::tie(a.load_seq, a.slot) <
                     std::tie(b.load_seq, b.slot);
          }
        };
        if (max_load < candidates.size()) {
          std::nth_element(candidates.begin(), candidates.begin() + max_load,
                           candidates.end(), victim_less);
        }
        victim_slots.resize(max_load);
        for (std::size_t k = 0u; k < max_load; ++k) {
          victim_slots[k] = candidates[k].slot;
        }
        used_slots = cap;
      }
    }

    std::sort(misses.begin(), misses.end(),
              [&](const MissCandidate &a, const MissCandidate &b) {
                const auto &pa = _pages[a.page];
                const auto &pb = _pages[b.page];
                return std::tie(pa.source_index, pa.tile_y, pa.tile_x) <
                       std::tie(pb.source_index, pb.tile_y, pb.tile_x);
              });

    const std::size_t total_load = misses.size();
    const std::size_t staging_cap = std::max<std::size_t>(pool.staging_capacity, 1u);
    const std::size_t num_atlases = pool.atlases.size();

    for (std::size_t chunk_begin = 0u; chunk_begin < total_load;
         chunk_begin += staging_cap) {
      const std::size_t chunk_end =
          std::min(chunk_begin + staging_cap, total_load);
      const std::size_t chunk_count = chunk_end - chunk_begin;
      const std::size_t chunk_slot = acquire_chunk_slot(cur_stream_h);

      luisa::uint *b4_pack_base =
          _use_pinned_upload ? _pinned_staging_byte4_ptr[chunk_slot]
                             : _host_staging_byte4.data();
      luisa::float4 *f4_pack_base =
          _use_pinned_upload ? _pinned_staging_float4_ptr[chunk_slot]
                             : _host_staging_float4.data();

      struct AssignedTile {
        std::uint32_t page{};
        std::uint32_t atlas_idx{};
        std::uint32_t dst_ox{};
        std::uint32_t dst_oy{};
        std::uint32_t staging_idx{};
      };
      std::vector<AssignedTile> assigned;
      assigned.reserve(chunk_count);

      auto assign_tile = [&](std::size_t k) {
        const auto &miss = misses[chunk_begin + k];
        const std::uint32_t page_id = miss.page;
        const std::uint32_t victim = victim_slots[chunk_begin + k];
        if (pool.eviction_possible) {
          push_ep_update(pool.epoch_offset + victim, new_slot_epoch);
          slot_meta[victim] = PoolSlotMeta{
              .load_wf = wf_seq,
              .last_hit_wf = 0ull,
              .load_seq = load_seq,
              .hotness = 0u,
              .initial_demand = miss.primary_demand,
              .loaded_by_aux = is_aux};
        }
        const std::uint32_t old_page = host_slots[victim];
        if (old_page != vt_invalid_page) {
          _host_page_table[old_page] = 0u;
          push_pt_update(old_page, 0u);
          ++_stats.tiles_evicted;
        }
        host_slots[victim] = page_id;
        _host_page_table[page_id] = vt_page_resident_bit | victim;
        push_pt_update(page_id, vt_page_resident_bit | victim);
        ++_stats.tiles_loaded;
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
        auto pack_one = [&](std::size_t offset) noexcept {
          const std::size_t k = begin_k + offset;
          const auto &pm = _pages[misses[chunk_begin + k].page];
          const auto &src = _sources[pm.source_index];
          if (pool.is_float4) {
            const luisa::span<luisa::float4> dst{
                f4_pack_base + k * tile_texels, tile_texels};
            if (!src.decoded || src.load_failed || src.float_pixels.empty()) {
              std::fill(dst.begin(), dst.end(), luisa::make_float4(0.0f));
            } else {
              pack_float4_tile(src, pm, _physical_tile_size, _logical_tile_size,
                               _tile_border, dst);
            }
          } else {
            const luisa::span<luisa::uint> dst{
                b4_pack_base + k * tile_texels, tile_texels};
            if (!src.decoded || src.load_failed || src.unorm8_pixels.empty()) {
              std::fill(dst.begin(), dst.end(), 0u);
            } else {
              pack_byte4_tile(src, pm, _physical_tile_size, _logical_tile_size,
                              _tile_border, dst);
            }
          }
        };
        const std::size_t count = end_k - begin_k;
        if (count >= 16u) {
          psycles::parallel_for(count, pack_one);
        } else {
          for (std::size_t offset = 0u; offset < count; ++offset) {
            pack_one(offset);
          }
        }
        const auto pack_t1 = std::chrono::steady_clock::now();
        _stats.pack_ms +=
            std::chrono::duration<double, std::milli>(pack_t1 - pack_t0)
                .count();
      };

      if (_options.host_decoded_cache_budget_bytes == 0ull) {
        std::vector<std::uint32_t> unique_sources;
        std::uint32_t prev_src = std::numeric_limits<std::uint32_t>::max();
        for (std::size_t k = 0u; k < chunk_count; ++k) {
          const auto src_idx = _pages[misses[chunk_begin + k].page].source_index;
          if (src_idx != prev_src) {
            prev_src = src_idx;
            const auto &src = _sources[src_idx];
            if (!src.decoded && !src.load_failed &&
                std::find(unique_sources.begin(), unique_sources.end(), src_idx) ==
                    unique_sources.end()) {
              unique_sources.push_back(src_idx);
            }
          }
        }
        decode_unique_parallel(unique_sources);
        if (chunk_end < total_load) {
          const std::size_t lookahead_end =
              std::min(total_load, chunk_end + staging_cap);
          std::uint32_t next_prev_src = std::numeric_limits<std::uint32_t>::max();
          std::size_t queued_cnt = 0u;
          for (std::size_t idx = chunk_end; idx < lookahead_end && queued_cnt < 4u;
               ++idx) {
            const auto src_idx = _pages[misses[idx].page].source_index;
            if (src_idx != next_prev_src) {
              next_prev_src = src_idx;
              const auto &src = _sources[src_idx];
              if (!src.decoded && !src.load_failed) {
                enqueue_bg_decode(src_idx);
                ++queued_cnt;
              }
            }
          }
        }
        for (std::size_t k = 0u; k < chunk_count; ++k) {
          assign_tile(k);
        }
        pack_range(0u, chunk_count);
      } else {
        {
          const auto first_src = _pages[misses[chunk_begin].page].source_index;
          std::uint32_t prev_src = first_src;
          std::size_t queued_cnt = 0u;
          for (std::size_t k = 1u; k < chunk_count && queued_cnt < 1u; ++k) {
            const auto src_idx = _pages[misses[chunk_begin + k].page].source_index;
            if (src_idx != prev_src) {
              prev_src = src_idx;
              const auto &src = _sources[src_idx];
              if (!src.decoded && !src.load_failed) {
                enqueue_bg_decode(src_idx);
                ++queued_cnt;
              }
            }
          }
        }
        for (std::size_t run_begin = 0u; run_begin < chunk_count;) {
          const auto src_idx =
              _pages[misses[chunk_begin + run_begin].page].source_index;
          std::size_t run_end = run_begin + 1u;
          while (run_end < chunk_count &&
                 _pages[misses[chunk_begin + run_end].page].source_index ==
                     src_idx) {
            ++run_end;
          }
          decode_source_on_demand(src_idx);
          for (std::size_t k = run_begin; k < run_end; ++k) {
            assign_tile(k);
          }
          pack_range(run_begin, run_end);
          const bool chunk_continues_src =
              (chunk_end < total_load) &&
              (run_end == chunk_count) &&
              (_pages[misses[chunk_end].page].source_index == src_idx);
          if (!chunk_continues_src) {
            trim_host_decoded_cache(src_idx);
          }
          run_begin = run_end;
        }
      }

      std::vector<std::uint32_t> atlas_counts(num_atlases, 0u);
      for (const auto &a : assigned) {
        ++atlas_counts[a.atlas_idx];
      }
      std::vector<std::uint32_t> atlas_offsets(num_atlases, 0u);
      for (std::size_t a = 1u; a < num_atlases; ++a) {
        atlas_offsets[a] = atlas_offsets[a - 1u] + atlas_counts[a - 1u];
      }
      std::vector<std::uint32_t> atlas_write_cursor = atlas_offsets;
      luisa::uint4 *um_dst =
          _use_pinned_upload ? _pinned_upload_meta_ptr[chunk_slot]
                             : _host_upload_meta.data();
      for (const auto &a : assigned) {
        const auto pos = atlas_write_cursor[a.atlas_idx]++;
        um_dst[pos] =
            luisa::make_uint4(a.staging_idx, a.dst_ox, a.dst_oy, a.atlas_idx);
      }

      const auto upload_t0 = std::chrono::steady_clock::now();
      const auto chunk_u32 = static_cast<luisa::uint>(chunk_count);
      const auto elem_count =
          static_cast<luisa::uint>(chunk_count * tile_texels);
      if (_use_pinned_upload) {
        _pinned_ext->flush_range(
            _upload_meta[chunk_slot].handle(), 0ull,
            static_cast<std::uint64_t>(chunk_count) * sizeof(luisa::uint4));
        if (pool.is_float4) {
          _pinned_ext->flush_range(
              _staging_float4[chunk_slot].handle(), 0ull,
              static_cast<std::uint64_t>(elem_count) * sizeof(luisa::float4));
        } else {
          _pinned_ext->flush_range(
              _staging_byte4[chunk_slot].handle(), 0ull,
              static_cast<std::uint64_t>(elem_count) * sizeof(luisa::uint));
        }
      }
      auto emit_upload_into = [&](CommandList &target) {
        if (!_use_pinned_upload) {
          target << _upload_meta[chunk_slot]
                        .view(0u, chunk_u32)
                        .copy_from(luisa::span{_host_upload_meta}
                                       .subspan(0u, chunk_count));
        }
        if (pool.is_float4) {
          if (!_use_pinned_upload) {
            target << _staging_float4[chunk_slot]
                          .view(0u, elem_count)
                          .copy_from(luisa::span{_host_staging_float4}
                                         .subspan(0u, elem_count));
          }
          for (std::size_t a = 0u; a < num_atlases; ++a) {
            if (atlas_counts[a] == 0u) {
              continue;
            }
            target << _upload_float4_shader(
                          pool.atlases[a], _staging_float4[chunk_slot],
                          _upload_meta[chunk_slot], atlas_offsets[a],
                          _physical_tile_size)
                          .dispatch(_physical_tile_size, _physical_tile_size,
                                    atlas_counts[a]);
          }
        } else {
          if (!_use_pinned_upload) {
            target << _staging_byte4[chunk_slot]
                          .view(0u, elem_count)
                          .copy_from(luisa::span{_host_staging_byte4}
                                         .subspan(0u, elem_count));
          }
          for (std::size_t a = 0u; a < num_atlases; ++a) {
            if (atlas_counts[a] == 0u) {
              continue;
            }
            target << _upload_byte4_shader(
                          pool.atlases[a], _staging_byte4[chunk_slot],
                          _upload_meta[chunk_slot], atlas_offsets[a],
                          _physical_tile_size)
                          .dispatch(_physical_tile_size, _physical_tile_size,
                                    atlas_counts[a]);
          }
        }
      };

      const bool fuse_last_chunk =
          (wavefront_cmd != nullptr) && is_final_pool &&
          (chunk_end == total_load);
      if (fuse_last_chunk) {
        emit_upload_into(*wavefront_cmd);
        _chunk_slot_stream[chunk_slot] = cur_stream_h;
      } else if (_use_pinned_upload) {
        CommandList upload_cmd;
        emit_upload_into(upload_cmd);
        auto sig = _upload_event.signal();
        const auto fence = sig.fence;
        stream << upload_cmd.commit() << std::move(sig);
        _chunk_slot_fence[chunk_slot] = fence;
      } else {
        CommandList upload_cmd;
        emit_upload_into(upload_cmd);
        stream << upload_cmd.commit() << synchronize();
      }
      const auto upload_t1 = std::chrono::steady_clock::now();
      _stats.upload_ms +=
          std::chrono::duration<double, std::milli>(upload_t1 - upload_t0)
              .count();
    }
  };

  const bool have_f4_work = !f4_misses.empty() || !f4_sec_misses.empty();
  process_pool(0u, _byte4_pool, _host_byte4_slots, _host_byte4_slot_meta,
               _byte4_used_slots, b4_misses, b4_sec_misses, !have_f4_work);
  process_pool(1u, _float4_pool, _host_float4_slots, _host_float4_slot_meta,
               _float4_used_slots, f4_misses, f4_sec_misses, true);

  const auto finalize_t0 = std::chrono::steady_clock::now();
  CommandList local_finalize_cmd;
  CommandList &finalize_cmd =
      (wavefront_cmd != nullptr) ? *wavefront_cmd : local_finalize_cmd;
  if (is_aux) {
    finalize_cmd << _clear_u32_shader(_aux_demand_hist, total_pages)
                        .dispatch(total_pages);
  } else if (any_miss_bit) {
    if (!_use_pinned_readback) {
      std::fill(_host_miss_bitset.begin(), _host_miss_bitset.end(), 0u);
    }
    finalize_cmd << _clear_u32_shader(_miss_bitset, bitset_words)
                        .dispatch(bitset_words);
  }
  bool used_meta_buffers = false;
  if (gpu_state_dirty && pt_upd_count > 0u) {
    ++_stats.fault_batches;
    if (is_aux) {
      ++_stats.aux_fault_batches;
    }
    std::uint32_t b4_resident = 0u;
    for (const auto s : _host_byte4_slots) {
      if (s != vt_invalid_page) {
        ++b4_resident;
      }
    }
    std::uint32_t f4_resident = 0u;
    for (const auto s : _host_float4_slots) {
      if (s != vt_invalid_page) {
        ++f4_resident;
      }
    }
    _stats.byte4_peak_resident =
        std::max(_stats.byte4_peak_resident, b4_resident);
    _stats.float4_peak_resident =
        std::max(_stats.float4_peak_resident, f4_resident);

    const auto upd_count = static_cast<luisa::uint>(pt_upd_count);
    if (_use_pinned_upload) {
      _pinned_ext->flush_range(
          _page_table_updates[meta_slot].handle(), 0ull,
          static_cast<std::uint64_t>(upd_count) * sizeof(luisa::uint2));
    } else {
      finalize_cmd
          << _page_table_updates[meta_slot]
                 .view(0u, upd_count)
                 .copy_from(luisa::span{_host_page_table_updates});
    }
    finalize_cmd
        << _apply_page_table_updates_shader(
               _page_table, _page_table_updates[meta_slot], upd_count)
               .dispatch(upd_count);
    used_meta_buffers = true;
    if (need_epoch_readback) {
      if (any_epoch_advanced) {
        finalize_cmd << _current_epoch.copy_from(
            luisa::span{_host_current_epoch});
      }
      if (any_epoch_wrapped) {
        finalize_cmd << _slot_epoch.copy_from(luisa::span{_host_slot_epoch});
      }
      if (ep_upd_count > 0u) {
        const auto ep_upd_count_u32 = static_cast<luisa::uint>(ep_upd_count);
        if (_use_pinned_upload) {
          _pinned_ext->flush_range(
              _slot_epoch_updates[meta_slot].handle(), 0ull,
              static_cast<std::uint64_t>(ep_upd_count_u32) *
                  sizeof(luisa::uint2));
        } else {
          finalize_cmd
              << _slot_epoch_updates[meta_slot]
                     .view(0u, ep_upd_count_u32)
                     .copy_from(luisa::span{_host_slot_epoch_updates});
        }
        finalize_cmd
            << _apply_page_table_updates_shader(
                   _slot_epoch, _slot_epoch_updates[meta_slot],
                   ep_upd_count_u32)
                   .dispatch(ep_upd_count_u32);
      }
    }
  }
  if (wavefront_cmd != nullptr) {
    if (used_meta_buffers) {
      _meta_slot_stream[meta_slot] = cur_stream_h;
    }
  } else if (!local_finalize_cmd.empty()) {
    if (_use_pinned_upload) {
      auto sig = _upload_event.signal();
      const auto fence = sig.fence;
      stream << local_finalize_cmd.commit() << std::move(sig);
      _meta_slot_fence[meta_slot] = fence;
    } else {
      stream << local_finalize_cmd.commit() << synchronize();
    }
  }
  const auto finalize_t1 = std::chrono::steady_clock::now();
  _stats.upload_ms +=
      std::chrono::duration<double, std::milli>(finalize_t1 - finalize_t0)
          .count();
}

} // namespace psycles::luisa_backend::detail
