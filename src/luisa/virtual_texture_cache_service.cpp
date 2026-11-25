#include "virtual_texture_cache.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <bit>
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

using namespace luisa::compute;

void VirtualTextureCache::wait_async_fence(std::uint64_t fence) const noexcept {
  if (fence == 0ull) {
    return;
  }
  std::uint64_t done = _async_publish_completed.load(std::memory_order_acquire);
  while (done < fence) {
    _async_publish_completed.wait(done, std::memory_order_acquire);
    done = _async_publish_completed.load(std::memory_order_acquire);
  }
}

void VirtualTextureCache::wait_all_upload_fences() noexcept {
  for (auto &pr : _pools) {
    for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
      if (i < pr.slot_fence.size() && pr.slot_fence[i] != 0ull) {
        _upload_event.synchronize(pr.slot_fence[i]);
        pr.slot_fence[i] = 0ull;
      }
      if (i < pr.slot_stream.size()) {
        pr.slot_stream[i] = 0ull;
      }
      if (i < pr.slot_async_fence.size() && pr.slot_async_fence[i] != 0ull) {
        wait_async_fence(pr.slot_async_fence[i]);
        pr.slot_async_fence[i] = 0ull;
      }
    }
  }
  for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
    if (i < _meta_slot_fence.size() && _meta_slot_fence[i] != 0ull) {
      _upload_event.synchronize(_meta_slot_fence[i]);
      _meta_slot_fence[i] = 0ull;
    }
    if (i < _meta_slot_stream.size()) {
      _meta_slot_stream[i] = 0ull;
    }
    if (i < _meta_slot_async_fence.size() && _meta_slot_async_fence[i] != 0ull) {
      wait_async_fence(_meta_slot_async_fence[i]);
      _meta_slot_async_fence[i] = 0ull;
    }
  }
  wait_async_fence(_async_publish_seq.load(std::memory_order_acquire));
  if (_use_copy_stream) {
    if (_copy_h2d_event && _copy_h2d_event.last_fence() > 0ull) {
      _copy_h2d_event.synchronize();
    }
    if (_copy_acquire_event && _copy_acquire_event.last_fence() > 0ull) {
      _copy_acquire_event.synchronize();
    }
    if (_copy_done_event && _copy_done_event.last_fence() > 0ull) {
      _copy_done_event.synchronize();
    }
    if (_copy_stream) {
      _copy_stream.synchronize();
    }
  }
}

std::size_t VirtualTextureCache::acquire_chunk_slot(
    std::uint32_t pool_idx, std::uint64_t current_stream_handle) noexcept {
  auto &pr = _pools[pool_idx];
  auto &slot_fence = pr.slot_fence;
  auto &slot_stream = pr.slot_stream;
  auto &slot_async_fence = pr.slot_async_fence;
  auto &cursor = pr.ring_cursor;
  const std::size_t ring_sz = pr.staging.size();
  if (ring_sz == 0u) {
    return 0u;
  }

  std::size_t slot = cursor % ring_sz;
  const std::uint64_t done_async =
      _async_publish_completed.load(std::memory_order_acquire);
  for (std::size_t k = 0u; k < ring_sz; ++k) {
    const std::size_t cand = (cursor + k) % ring_sz;
    if (slot_fence[cand] != 0ull &&
        _upload_event.is_completed(slot_fence[cand])) {
      slot_fence[cand] = 0ull;
    }
    if (slot_fence[cand] == 0ull && slot_async_fence[cand] <= done_async &&
        (slot_stream[cand] == 0ull ||
         slot_stream[cand] == current_stream_handle)) {
      slot = cand;
      break;
    }
  }
  cursor = (slot + 1u) % ring_sz;
  if (slot_fence[slot] != 0ull) {
    _upload_event.synchronize(slot_fence[slot]);
    slot_fence[slot] = 0ull;
  }
  if (slot_async_fence[slot] != 0ull) {
    wait_async_fence(slot_async_fence[slot]);
    slot_async_fence[slot] = 0ull;
  }
  slot_stream[slot] = 0ull;
  return slot;
}

std::size_t VirtualTextureCache::acquire_meta_slot(
    std::uint64_t current_stream_handle) noexcept {
  std::size_t slot = _meta_ring_cursor;
  const std::uint64_t done_async =
      _async_publish_completed.load(std::memory_order_acquire);
  for (std::size_t k = 0u; k < _meta_ring_size; ++k) {
    const std::size_t cand = (_meta_ring_cursor + k) % _meta_ring_size;
    if (_meta_slot_fence[cand] != 0ull &&
        _upload_event.is_completed(_meta_slot_fence[cand])) {
      _meta_slot_fence[cand] = 0ull;
    }
    if (_meta_slot_fence[cand] == 0ull &&
        _meta_slot_async_fence[cand] <= done_async &&
        (_meta_slot_stream[cand] == 0ull ||
         _meta_slot_stream[cand] == current_stream_handle)) {
      slot = cand;
      break;
    }
  }
  _meta_ring_cursor = (slot + 1u) % _meta_ring_size;
  if (_meta_slot_fence[slot] != 0ull) {
    _upload_event.synchronize(_meta_slot_fence[slot]);
    _meta_slot_fence[slot] = 0ull;
  }
  if (_meta_slot_async_fence[slot] != 0ull) {
    wait_async_fence(_meta_slot_async_fence[slot]);
    _meta_slot_async_fence[slot] = 0ull;
  }
  _meta_slot_stream[slot] = 0ull;
  return slot;
}

void VirtualTextureCache::scan_miss_candidates(
    MissSource source,
    bool any_miss_bit,
    std::uint32_t bitset_words,
    const luisa::uint *demand_hist_read,
    const luisa::uint *aux_demand_hist_read,
    const luisa::uint *miss_bitset_read,
    std::vector<std::vector<MissCandidate>> &pool_misses,
    std::vector<std::vector<MissCandidate>> &pool_sec_misses,
    bool track_inflight_deps,
    std::uint32_t *dep_slot_mask_out) const noexcept {
  pool_misses.resize(_pools.size());
  for (auto &v : pool_misses) {
    v.clear();
  }
  pool_sec_misses.resize(_pools.size());
  for (auto &v : pool_sec_misses) {
    v.clear();
  }
  const std::uint32_t num_pages = static_cast<std::uint32_t>(_pages.size());
  std::uint32_t dep_mask = 0u;
  if (source == MissSource::bitset_only) {
    if (any_miss_bit) {
      const std::uint32_t max_words =
          std::min<std::uint32_t>(bitset_words, (num_pages + 31u) >> 5u);
      for (std::uint32_t wi = 0u; wi < max_words; ++wi) {
        std::uint32_t w = miss_bitset_read[wi];
        while (w != 0u) {
          const std::uint32_t bit = static_cast<std::uint32_t>(std::countr_zero(w));
          w &= (w - 1u);
          const std::uint32_t p = (wi << 5u) | bit;
          if (p >= num_pages ||
              (_host_page_table[p] & vt_page_resident_bit) != 0u) {
            continue;
          }
          pool_misses[_pages[p].pool_index].push_back(
              MissCandidate{.page = p, .primary_demand = 0u});
        }
      }
    }
  } else if (source == MissSource::aux_demand) {
    auto handle_aux_page = [&](std::uint32_t pk) noexcept {
      const auto primary = aux_demand_hist_read[pk];
      if (primary == 0u ||
          (_host_page_table[pk] & vt_page_resident_bit) != 0u) {
        return;
      }
      pool_misses[_pages[pk].pool_index].push_back(
          MissCandidate{.page = pk, .primary_demand = primary});
    };
    std::uint32_t p = 0u;
    for (; p + 4u <= num_pages; p += 4u) {
      if ((aux_demand_hist_read[p] | aux_demand_hist_read[p + 1u] |
           aux_demand_hist_read[p + 2u] | aux_demand_hist_read[p + 3u]) == 0u) {
        continue;
      }
      for (std::uint32_t k = 0u; k < 4u; ++k) {
        handle_aux_page(p + k);
      }
    }
    for (; p < num_pages; ++p) {
      handle_aux_page(p);
    }
  } else {
    auto handle_wf_page = [&](std::uint32_t pk, std::uint32_t primary,
                              bool bit_set) noexcept {
      if (primary == 0u && !bit_set) {
        return;
      }
      if (track_inflight_deps) {
        if (pk < _page_load_inflight_slot.size() &&
            _page_load_inflight_slot[pk] != PoolRuntime::kInvalidAsyncSlot) {
          if (_page_load_inflight_slot[pk] < 32u) {
            dep_mask |= (1u << _page_load_inflight_slot[pk]);
          }
          return;
        }
        if (pk < _page_evict_owner_mask.size() &&
            _page_evict_owner_mask[pk] != 0u) {
          dep_mask |= _page_evict_owner_mask[pk];
          return;
        }
      }
      if ((_host_page_table[pk] & vt_page_resident_bit) != 0u) {
        return;
      }
      if (primary != 0u) {
        pool_misses[_pages[pk].pool_index].push_back(
            MissCandidate{.page = pk, .primary_demand = primary});
      } else if (bit_set) {
        pool_sec_misses[_pages[pk].pool_index].push_back(
            MissCandidate{.page = pk, .primary_demand = 0u});
      }
    };
    const std::uint32_t num_words = (num_pages + 31u) >> 5u;
    for (std::uint32_t wi = 0u; wi < num_words; ++wi) {
      const std::uint32_t bits =
          (any_miss_bit && wi < bitset_words) ? miss_bitset_read[wi] : 0u;
      const std::uint32_t word_begin = wi << 5u;
      const std::uint32_t word_end = std::min(num_pages, word_begin + 32u);
      std::uint32_t p = word_begin;
      for (; p + 4u <= word_end; p += 4u) {
        const std::uint32_t quad_bits = (bits >> (p & 31u)) & 0xFu;
        if (quad_bits == 0u &&
            (demand_hist_read[p] | demand_hist_read[p + 1u] |
             demand_hist_read[p + 2u] | demand_hist_read[p + 3u]) == 0u) {
          continue;
        }
        for (std::uint32_t k = 0u; k < 4u; ++k) {
          const std::uint32_t pk = p + k;
          handle_wf_page(pk, demand_hist_read[pk],
                         ((quad_bits >> k) & 1u) != 0u);
        }
      }
      for (; p < word_end; ++p) {
        handle_wf_page(p, demand_hist_read[p],
                       ((bits >> (p & 31u)) & 1u) != 0u);
      }
    }
  }
  if (dep_slot_mask_out != nullptr) {
    *dep_slot_mask_out = dep_mask;
  }
}

void VirtualTextureCache::compute_active_source_hints(
    bool is_aux,
    const luisa::uint *slot_epoch_read,
    std::vector<std::uint32_t> &src_hit_slots,
    std::vector<std::uint8_t> &src_partner_hit,
    std::span<const PoolPinContext> pool_pin_ctxs) const noexcept {
  src_hit_slots.resize(_sources.size());
  std::fill(src_hit_slots.begin(), src_hit_slots.end(), 0u);
  src_partner_hit.resize(_sources.size());
  std::fill(src_partner_hit.begin(), src_partner_hit.end(), std::uint8_t{0u});
  for (std::size_t pool_idx = 0u; pool_idx < _pools.size(); ++pool_idx) {
    const auto &pr = _pools[pool_idx];
    const auto &pool = pr.config;
    const PoolPinContext *pin_ctx =
        (pool_idx < pool_pin_ctxs.size()) ? &pool_pin_ctxs[pool_idx] : nullptr;
    const std::uint32_t readback_ep =
        (pin_ctx != nullptr && pin_ctx->has_readback_epoch)
            ? pin_ctx->readback_gpu_epoch
            : _host_current_epoch[pool_idx];
    const bool readback_not_newer =
        (readback_ep != 0u && pr.last_readback_gpu_epoch >= readback_ep);
    const bool same_or_stale_rb =
        (!is_aux && pin_ctx != nullptr && pin_ctx->has_readback_epoch &&
         readback_not_newer);
    const std::uint32_t hit_window_lo =
        readback_not_newer
            ? (readback_ep >= 2u ? readback_ep - 2u : 0u)
            : pr.last_readback_gpu_epoch;
    const std::uint64_t next_wf =
        pr.wavefront_seq + ((is_aux || same_or_stale_rb) ? 0ull : 1ull);
    for (std::uint32_t s = 0u; s < pool.capacity; ++s) {
      const std::uint32_t p = pr.host_slots[s];
      if (p == vt_invalid_page) {
        continue;
      }
      const bool slot_outdated =
          pin_ctx != nullptr &&
          ((s < pin_ctx->slot_inflight_owner.size() &&
            pin_ctx->slot_inflight_owner[s] != PoolRuntime::kInvalidAsyncSlot) ||
           (pin_ctx->has_readback_epoch &&
            s < pin_ctx->slot_publish_epoch.size() &&
            pin_ctx->slot_publish_epoch[s] > pin_ctx->readback_publish_epoch));
      bool active = !pool.eviction_possible || slot_outdated;
      if (!active) {
        const std::uint32_t ep = slot_epoch_read[pool.epoch_offset + s];
        const bool hit_in_window =
            (ep != 0u) && ((ep & 1u) == 0u) && (ep > hit_window_lo) &&
            (ep <= readback_ep);
        const bool fresh_in_window =
            ((ep & 1u) == 1u) && (ep + 1u > hit_window_lo) &&
            (ep + 1u <= readback_ep) &&
            (next_wf <= pr.host_slot_meta[s].load_wf + 1ull);
        active = hit_in_window || fresh_in_window;
      }
      if (active) {
        const std::uint32_t src_idx = _pages[p].source_index;
        if (src_idx < src_hit_slots.size()) {
          ++src_hit_slots[src_idx];
        }
      }
    }
  }
  for (std::uint32_t s = 0u; s < src_hit_slots.size(); ++s) {
    if (src_hit_slots[s] > 0u && s < _source_partners.size()) {
      for (const auto partner : _source_partners[s]) {
        if (partner < src_partner_hit.size()) {
          src_partner_hit[partner] = 1u;
        }
      }
    }
  }
}

void VirtualTextureCache::plan_pool_misses_host(
    std::uint32_t pool_idx,
    PoolRuntime &pool_rt,
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
    const PoolPinContext *pin_ctx,
    bool *deferred_no_victim_out,
    bool *pool_backpressured_out) {
  out_plan.misses.clear();
  out_plan.victim_slots.clear();
  out_plan.new_slot_epoch = 0u;
  out_plan.wf_seq = 0ull;
  out_plan.load_seq = 0ull;
  out_plan.has_work = false;
  if (deferred_no_victim_out != nullptr) {
    *deferred_no_victim_out = false;
  }
  if (pool_backpressured_out != nullptr) {
    *pool_backpressured_out = false;
  }
  const auto &pool = pool_rt.config;
  const auto &host_slots = pool_rt.host_slots;
  auto &slot_meta = pool_rt.host_slot_meta;
  auto &used_slots = pool_rt.used_slots;
  if ((misses.empty() && sec_misses.empty()) || pool.capacity == 0u ||
      pool.atlases.empty()) {
    return;
  }
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

  const std::uint32_t cap = pool.capacity;
  out_plan.wf_seq = pool_rt.wavefront_seq;
  out_plan.load_seq = pool_rt.load_seq;

  if (!pool.eviction_possible) {
    if (max_batch_tiles == std::numeric_limits<std::uint32_t>::max()) {
      if (!sec_misses.empty()) {
        _stats.secondary_misses_coalesced += sec_misses.size();
        misses.insert(misses.end(), sec_misses.begin(), sec_misses.end());
      }
    } else {
      if (misses.size() > max_batch_tiles) {
        std::nth_element(
            misses.begin(), misses.begin() + max_batch_tiles, misses.end(),
            [](const MissCandidate &a, const MissCandidate &b) {
              if (a.primary_demand != b.primary_demand) {
                return a.primary_demand > b.primary_demand;
              }
              return a.page < b.page;
            });
        misses.resize(max_batch_tiles);
      }
      if (misses.size() < max_batch_tiles && !sec_misses.empty()) {
        const std::size_t take = std::min<std::size_t>(
            sec_misses.size(), max_batch_tiles - misses.size());
        misses.insert(misses.end(), sec_misses.begin(),
                      sec_misses.begin() + take);
        _stats.secondary_misses_coalesced += take;
      }
    }
    if (misses.empty()) {
      return;
    }
    out_plan.victim_slots.resize(misses.size());
    for (std::size_t k = 0u; k < misses.size(); ++k) {
      out_plan.victim_slots[k] = used_slots++;
    }
    for (const auto &m : misses) {
      propagate_partners(_pages[m.page].source_index);
    }
  } else {
    const std::uint32_t prev_host_epoch = _host_current_epoch[pool_idx];
    const std::uint32_t readback_gpu_epoch =
        (pin_ctx != nullptr && pin_ctx->has_readback_epoch)
            ? pin_ctx->readback_gpu_epoch
            : prev_host_epoch;
    const bool readback_not_newer =
        (readback_gpu_epoch != 0u &&
         pool_rt.last_readback_gpu_epoch >= readback_gpu_epoch);
    const bool same_or_stale_readback =
        (!is_aux && pin_ctx != nullptr && pin_ctx->has_readback_epoch &&
         readback_not_newer);
    const std::uint32_t hit_window_lo =
        readback_not_newer
            ? (readback_gpu_epoch >= 2u ? readback_gpu_epoch - 2u : 0u)
            : pool_rt.last_readback_gpu_epoch;
    const std::uint64_t next_wf =
        out_plan.wf_seq + ((is_aux || same_or_stale_readback) ? 0ull : 1ull);
    const std::size_t free_slots =
        used_slots < cap ? static_cast<std::size_t>(cap - used_slots) : 0u;
    const bool cache_was_full = (used_slots >= cap);

    auto &aux_streak_protected = scratch.aux_streak_protected;
    if (is_aux) {
      aux_streak_protected.resize(cap);
      std::fill(aux_streak_protected.begin(), aux_streak_protected.end(),
                std::uint8_t{0u});
      const std::uint64_t streak_start = pool_rt.aux_streak_start_seq;
      const std::uint32_t protect_cap = cap / 2u;
      if (streak_start != std::numeric_limits<std::uint64_t>::max() &&
          protect_cap > 0u) {
        auto &streak_slots = scratch.streak_slots;
        streak_slots.clear();
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

    auto &unpinned_candidates = scratch.unpinned_candidates;
    unpinned_candidates.clear();
    unpinned_candidates.reserve(cap);
    auto &new_hotness = scratch.new_hotness;
    new_hotness.resize(cap);
    auto &new_last_hit_wf = scratch.new_last_hit_wf;
    new_last_hit_wf.resize(cap);
    std::uint32_t unpinned_safe_count = 0u;
    std::uint32_t unpinned_total_count = 0u;
    const bool bypass_soft_pin =
        pin_ctx != nullptr && pin_ctx->is_oldest && pin_ctx->force_oldest_progress;
    for (std::uint32_t s = 0u; s < cap; ++s) {
      const bool hard_pinned =
          pin_ctx != nullptr && s < pin_ctx->slot_inflight_owner.size() &&
          pin_ctx->slot_inflight_owner[s] != PoolRuntime::kInvalidAsyncSlot;
      const bool soft_pinned =
          pin_ctx != nullptr && !bypass_soft_pin &&
          host_slots[s] != vt_invalid_page &&
          ((s < pin_ctx->slot_pin_consumer_seq.size() &&
            pin_ctx->slot_pin_consumer_seq[s] > pin_ctx->retired_consumer_seq) ||
           (s < pin_ctx->slot_pin_by_aux.size() &&
            pin_ctx->slot_pin_by_aux[s] != 0u &&
            s < pin_ctx->slot_pin_aux_seq.size() &&
            pin_ctx->slot_pin_aux_seq[s] > pin_ctx->retired_aux_seq));
      const bool pinned = hard_pinned || soft_pinned;
      const bool slot_epoch_outdated =
          hard_pinned ||
          (pin_ctx != nullptr && pin_ctx->has_readback_epoch &&
           s < pin_ctx->slot_publish_epoch.size() &&
           pin_ctx->slot_publish_epoch[s] > pin_ctx->readback_publish_epoch);
      VictimSlot v{};
      v.slot = s;
      if (host_slots[s] == vt_invalid_page) {
        v.tier = 0u;
        if (!pinned) {
          ++unpinned_safe_count;
          ++unpinned_total_count;
          unpinned_candidates.push_back(v);
        }
      } else {
        const auto &meta = slot_meta[s];
        std::uint32_t updated_hot = meta.hotness;
        std::uint64_t updated_hit_wf = meta.last_hit_wf;
        if (slot_epoch_outdated) {
          v.tier = (is_aux && aux_streak_protected[s] != 0u) ? 4u : 3u;
        } else {
          const std::uint32_t ep = slot_epoch_read[pool.epoch_offset + s];
          const bool hit_in_window =
              (ep != 0u) && ((ep & 1u) == 0u) && (ep > hit_window_lo) &&
              (ep <= readback_gpu_epoch);
          const bool fresh_in_window =
              ((ep & 1u) == 1u) && (ep + 1u > hit_window_lo) &&
              (ep + 1u <= readback_gpu_epoch) &&
              (next_wf <= meta.load_wf + 2ull);
          if (is_aux) {
            if (aux_streak_protected[s] != 0u) {
              v.tier = 4u;
            } else if (hit_in_window || fresh_in_window) {
              v.tier = 3u;
            } else if (next_wf <=
                       std::max(meta.last_hit_wf, meta.load_wf) + 3ull) {
              v.tier = 2u;
            } else {
              v.tier = 1u;
              if (!pinned) {
                ++unpinned_safe_count;
              }
            }
          } else {
            if (hit_in_window) {
              updated_hit_wf = next_wf;
              if (!same_or_stale_readback || meta.last_hit_wf < next_wf) {
                updated_hot = std::min(meta.hotness + 1u, 32u);
              }
              v.tier = 3u;
            } else if (fresh_in_window) {
              v.tier = 3u;
            } else if (next_wf <=
                       std::max(meta.last_hit_wf, meta.load_wf) + 3ull) {
              if (!same_or_stale_readback && next_wf >= meta.last_hit_wf + 2ull) {
                updated_hot = (meta.hotness * 3u) / 4u;
              }
              v.tier = 2u;
            } else {
              if (!same_or_stale_readback) {
                updated_hot = meta.hotness >> 1u;
              }
              v.tier = 1u;
              if (!pinned) {
                ++unpinned_safe_count;
              }
            }
          }
        }
        new_hotness[s] = updated_hot;
        new_last_hit_wf[s] = updated_hit_wf;
        v.hotness = updated_hot;
        v.initial_demand = meta.initial_demand;
        v.last_hit_wf = updated_hit_wf;
        v.load_seq = meta.load_seq;
        if (!pinned) {
          ++unpinned_total_count;
          unpinned_candidates.push_back(v);
        }
      }
    }

    if (!is_aux) {
      const std::uint32_t hot_pin_cap = cap / 4u;
      if (hot_pin_cap > 0u) {
        auto &hot_slots = scratch.hot_slots;
        hot_slots.clear();
        hot_slots.reserve(unpinned_candidates.size());
        for (std::size_t idx = 0u; idx < unpinned_candidates.size(); ++idx) {
          const auto s = unpinned_candidates[idx].slot;
          if (host_slots[s] != vt_invalid_page &&
              unpinned_candidates[idx].tier >= 2u &&
              unpinned_candidates[idx].hotness >= 3u &&
              unpinned_candidates[idx].last_hit_wf > 0ull &&
              next_wf <= unpinned_candidates[idx].last_hit_wf + 2ull) {
            hot_slots.push_back(static_cast<std::uint32_t>(idx));
          }
        }
        if (hot_slots.size() > hot_pin_cap) {
          std::nth_element(
              hot_slots.begin(), hot_slots.begin() + hot_pin_cap,
              hot_slots.end(), [&](std::uint32_t ia, std::uint32_t ib) {
                const auto &ca = unpinned_candidates[ia];
                const auto &cb = unpinned_candidates[ib];
                return std::tie(ca.hotness, ca.last_hit_wf, ca.initial_demand,
                                ca.load_seq, ca.slot) >
                       std::tie(cb.hotness, cb.last_hit_wf, cb.initial_demand,
                                cb.load_seq, cb.slot);
              });
          hot_slots.resize(hot_pin_cap);
        }
        for (const std::uint32_t idx : hot_slots) {
          unpinned_candidates[idx].tier = 4u;
        }
      }
    }

    const std::uint32_t min_progress =
        pool.is_high_precision
            ? std::max<std::uint32_t>(1u, cap / 4u)
            : std::max<std::uint32_t>(1u, cap / 6u);
    const bool is_speculative_non_oldest =
        (pin_ctx != nullptr && !pin_ctx->is_oldest);
    const std::uint32_t oldest_reserve =
        is_speculative_non_oldest
            ? std::min<std::uint32_t>(cap, min_progress)
            : 0u;
    const std::uint32_t usable_unpinned =
        is_speculative_non_oldest
            ? (unpinned_safe_count > oldest_reserve
                   ? (unpinned_safe_count - oldest_reserve)
                   : 0u)
            : unpinned_total_count;
    if (pin_ctx != nullptr && !misses.empty() && usable_unpinned == 0u) {
      if (deferred_no_victim_out != nullptr) {
        *deferred_no_victim_out = true;
      }
      if (pool_backpressured_out != nullptr) {
        *pool_backpressured_out = true;
      }
      return;
    }
    const std::uint32_t raw_budget =
        is_speculative_non_oldest
            ? usable_unpinned
            : std::min(usable_unpinned,
                       std::max(unpinned_safe_count, min_progress));
    const std::uint32_t batch_budget =
        std::min(raw_budget, std::max<std::uint32_t>(1u, max_batch_tiles));

    if (misses.empty()) {
      std::size_t partner_sec_count = 0u;
      for (const auto &m : sec_misses) {
        const auto src_idx = _pages[m.page].source_index;
        if (src_idx < src_partner_hit.size() &&
            src_partner_hit[src_idx] != 0u) {
          ++partner_sec_count;
        }
      }
      const std::uint32_t partner_avail =
          pin_ctx != nullptr ? usable_unpinned : unpinned_safe_count;
      if ((partner_sec_count == 0u && unpinned_safe_count == 0u) ||
          usable_unpinned == 0u || batch_budget == 0u || partner_avail == 0u) {
        return;
      }
    }

    out_plan.load_seq = ++pool_rt.load_seq;
    out_plan.wf_seq = next_wf;
    if (!is_aux) {
      pool_rt.wavefront_seq = out_plan.wf_seq;
      const bool wrapped = (prev_host_epoch >= 0xfffffffcu);
      const std::uint32_t next_gpu_epoch =
          wrapped ? 2u : (prev_host_epoch + 2u);
      _host_current_epoch[pool_idx] = next_gpu_epoch;
      out_plan.new_slot_epoch = next_gpu_epoch - 1u;
      pool_rt.last_readback_gpu_epoch =
          wrapped ? 0u
                  : std::max(pool_rt.last_readback_gpu_epoch, readback_gpu_epoch);
      any_epoch_advanced = true;
      if (wrapped) {
        if (!slot_epoch_host_seeded) {
          std::copy_n(slot_epoch_read, total_slots, _host_slot_epoch.data());
          slot_epoch_host_seeded = true;
        }
        any_epoch_wrapped = true;
        std::fill_n(_host_slot_epoch.data() + pool.epoch_offset, cap, 0u);
      }
    } else {
      if (pool_rt.aux_streak_start_seq ==
          std::numeric_limits<std::uint64_t>::max()) {
        pool_rt.aux_streak_start_seq = out_plan.load_seq;
      }
      out_plan.new_slot_epoch =
          (prev_host_epoch > 0u) ? (prev_host_epoch - 1u) : 0u;
    }
    for (std::uint32_t s = 0u; s < cap; ++s) {
      if (host_slots[s] != vt_invalid_page) {
        slot_meta[s].hotness = new_hotness[s];
        slot_meta[s].last_hit_wf = new_last_hit_wf[s];
      }
    }

    const std::size_t raw_primary_misses = misses.size();
    if (pool_backpressured_out != nullptr && !is_aux && cache_was_full &&
        unpinned_safe_count < min_progress &&
        raw_primary_misses > min_progress) {
      *pool_backpressured_out = true;
    }
    if (misses.size() > batch_budget) {
      if (source == MissSource::bitset_only) {
        const auto salt =
            static_cast<std::uint32_t>(out_plan.load_seq * 2246822519ull);
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
        auto &src_demand_sum = scratch.src_demand_sum;
        src_demand_sum.resize(_sources.size());
        std::fill(src_demand_sum.begin(), src_demand_sum.end(), 0ull);
        auto &src_miss_cnt = scratch.src_miss_cnt;
        src_miss_cnt.resize(_sources.size());
        std::fill(src_miss_cnt.begin(), src_miss_cnt.end(), 0u);
        for (const auto &m : misses) {
          const auto src_idx = _pages[m.page].source_index;
          if (src_idx < src_demand_sum.size()) {
            src_demand_sum[src_idx] += m.primary_demand;
            ++src_miss_cnt[src_idx];
          }
        }
        auto &src_avg_demand = scratch.src_avg_demand;
        src_avg_demand.resize(_sources.size());
        std::fill(src_avg_demand.begin(), src_avg_demand.end(), 0ull);
        for (std::size_t s = 0u; s < _sources.size(); ++s) {
          if (src_miss_cnt[s] != 0u) {
            src_avg_demand[s] = src_demand_sum[s] / src_miss_cnt[s];
          }
        }
        if (scratch.block_sum.size() < _total_tile_blocks) {
          scratch.block_sum.assign(_total_tile_blocks, 0ull);
          scratch.block_cnt.assign(_total_tile_blocks, 0u);
        }
        auto &touched_blocks = scratch.touched_blocks;
        touched_blocks.clear();
        for (const auto &m : misses) {
          const std::uint32_t bid = _pages[m.page].block_id;
          if (scratch.block_cnt[bid] == 0u) {
            touched_blocks.push_back(bid);
          }
          scratch.block_sum[bid] += m.primary_demand;
          ++scratch.block_cnt[bid];
        }
        auto &ranked = scratch.ranked;
        ranked.clear();
        ranked.reserve(misses.size());
        for (const auto &m : misses) {
          const auto &pm = _pages[m.page];
          const std::uint32_t bid = pm.block_id;
          ranked.push_back(RankedMiss{
              .page = m.page,
              .primary_demand = m.primary_demand,
              .source_index = pm.source_index,
              .bx = pm.tile_x >> 3u,
              .by = pm.tile_y >> 3u,
              .block_avg_demand = scratch.block_sum[bid] / scratch.block_cnt[bid]});
        }
        for (const std::uint32_t bid : touched_blocks) {
          scratch.block_sum[bid] = 0ull;
          scratch.block_cnt[bid] = 0u;
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

    auto &src_admitted_now = scratch.src_admitted_now;
    src_admitted_now.resize(_sources.size());
    std::fill(src_admitted_now.begin(), src_admitted_now.end(), std::uint8_t{0u});
    for (const auto &m : misses) {
      const auto src_idx = _pages[m.page].source_index;
      if (src_idx < src_admitted_now.size()) {
        src_admitted_now[src_idx] = 1u;
        propagate_partners(src_idx);
      }
    }

    if (misses.size() < batch_budget && !sec_misses.empty()) {
      auto &sec_partners = scratch.sec_partners;
      sec_partners.clear();
      auto &sec_other = scratch.sec_other;
      sec_other.clear();
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
          std::min<std::size_t>(batch_budget, unpinned_safe_count);
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
    bool free_slots_unpinned = (max_load <= free_slots);
    if (free_slots_unpinned && pin_ctx != nullptr) {
      for (std::size_t k = 0u; k < max_load; ++k) {
        const std::uint32_t s = used_slots + static_cast<std::uint32_t>(k);
        if (s < pin_ctx->slot_inflight_owner.size() &&
            pin_ctx->slot_inflight_owner[s] != PoolRuntime::kInvalidAsyncSlot) {
          free_slots_unpinned = false;
          break;
        }
      }
    }
    if (free_slots_unpinned) {
      out_plan.victim_slots.resize(max_load);
      for (std::size_t k = 0u; k < max_load; ++k) {
        out_plan.victim_slots[k] = used_slots++;
      }
    } else {
      auto &buckets = scratch.tier_buckets;
      for (auto &b : buckets) {
        b.clear();
      }
      for (auto &v : unpinned_candidates) {
        if (_options.mip_mode == LuisaTextureMipMode::isotropic && !is_aux &&
            v.tier == 3u) {
          const std::uint32_t p = host_slots[v.slot];
          if (p != vt_invalid_page) {
            const std::uint32_t src_idx = _pages[p].source_index;
            if (src_idx < src_admitted_now.size() &&
                src_admitted_now[src_idx] != 0u) {
              v.tier = 4u;
            }
          }
        }
        buckets[std::min<std::size_t>(v.tier, 4u)].push_back(v);
      }
      const std::size_t chosen =
          std::min(max_load, unpinned_candidates.size());
      out_plan.victim_slots.resize(chosen);
      std::size_t out_pos = 0u;
      auto tier_victim_less = [](std::size_t tier, const VictimSlot &a,
                                 const VictimSlot &b) noexcept {
        switch (tier) {
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
      for (std::size_t t = 0u; t < buckets.size() && out_pos < chosen; ++t) {
        auto &b = buckets[t];
        const std::size_t rem = chosen - out_pos;
        if (b.size() > rem) {
          std::nth_element(b.begin(), b.begin() + rem, b.end(),
                           [&](const VictimSlot &va, const VictimSlot &vb) {
                             return tier_victim_less(t, va, vb);
                           });
        }
        const std::size_t take = std::min(b.size(), rem);
        for (std::size_t k = 0u; k < take; ++k) {
          out_plan.victim_slots[out_pos++] = b[k].slot;
        }
      }
      if (chosen < max_load) {
        misses.resize(chosen);
      }
      used_slots = cap;
    }
    if (!is_aux && !pool.is_high_precision && cap > 0u && cache_was_full &&
        raw_primary_misses > static_cast<std::size_t>(cap) * 2u) {
      const auto cur = _entry_drain_pressure.load(std::memory_order_relaxed);
      if (cur < 4u) {
        _entry_drain_pressure.store(cur + 1u, std::memory_order_relaxed);
      }
    }
  }

  std::sort(misses.begin(), misses.end(),
            [&](const MissCandidate &a, const MissCandidate &b) {
              const auto &pa = _pages[a.page];
              const auto &pb = _pages[b.page];
              return std::tie(pa.source_index, pa.tile_y, pa.tile_x) <
                     std::tie(pb.source_index, pb.tile_y, pb.tile_x);
            });
  out_plan.misses.swap(misses);
  out_plan.has_work = !out_plan.misses.empty();
}

void VirtualTextureCache::service_aux_misses(Stream &stream) noexcept {
  if (!has_virtual_images()) {
    return;
  }
  drain_async_if_in_flight(stream);
  service_misses_impl(stream, MissSource::aux_demand);
}

void VirtualTextureCache::service_bitset_misses(Stream &stream) noexcept {
  if (!has_virtual_images()) {
    return;
  }
  drain_async_if_in_flight(stream);
  service_misses_impl(stream, MissSource::bitset_only);
}

void VirtualTextureCache::service_misses_impl(
    Stream &stream,
    MissSource source,
    CommandList *wavefront_cmd,
    const ServiceProfileContext *prof_ctx) noexcept {
  if (_pages.empty()) {
    if (wavefront_cmd != nullptr && !wavefront_cmd->empty()) {
      stream << wavefront_cmd->commit() << synchronize();
    }
    return;
  }
  const bool prof_on = (_profiler != nullptr);
  const std::uint64_t prof_enter_ns =
      prof_on
          ? ((prof_ctx != nullptr && prof_ctx->collect_enter_ns != 0ull)
                 ? prof_ctx->collect_enter_ns
                 : VtProfileRecorder::steady_ns())
          : 0ull;
  const auto collect_t0 = std::chrono::steady_clock::now();
  bool need_epoch_readback = false;
  for (const auto &pr : _pools) {
    if (pr.config.eviction_possible) {
      need_epoch_readback = true;
      break;
    }
  }
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  const auto bitset_words = static_cast<luisa::uint>(_host_miss_bitset.size());
  const auto total_slots = static_cast<luisa::uint>(_host_slot_epoch.size());
  const bool is_aux = (source == MissSource::aux_demand);
  const bool is_wavefront = (source == MissSource::wavefront_demand);
  if (!is_aux) {
    for (auto &pr : _pools) {
      pr.aux_streak_start_seq = std::numeric_limits<std::uint64_t>::max();
    }
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
  const auto prof_rec_end_t =
      prof_on ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
  const std::uint64_t prof_seq_before_s1 =
      prof_on ? query_vk_timeline_submit_seq() : 0ull;
  stream << cmd.commit() << synchronize();
  const auto prof_sync_end_t =
      prof_on ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
  const std::uint64_t prof_seq_after_s1 =
      prof_on ? query_vk_timeline_submit_seq() : 0ull;
  const auto cur_stream_h = static_cast<std::uint64_t>(stream.handle());
  for (auto &pr : _pools) {
    for (std::size_t i = 0u; i < pr.staging.size(); ++i) {
      if (pr.slot_stream[i] == cur_stream_h) {
        pr.slot_stream[i] = 0ull;
      }
      if (pr.slot_fence[i] != 0ull &&
          _upload_event.is_completed(pr.slot_fence[i])) {
        pr.slot_fence[i] = 0ull;
      }
    }
  }
  for (std::size_t i = 0u; i < _meta_ring_size; ++i) {
    if (_meta_slot_stream[i] == cur_stream_h) {
      _meta_slot_stream[i] = 0ull;
    }
    if (_meta_slot_fence[i] != 0ull &&
        _upload_event.is_completed(_meta_slot_fence[i])) {
      _meta_slot_fence[i] = 0ull;
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

  auto &pool_misses = _sync_planning_scratch.pool_misses;
  auto &pool_sec_misses = _sync_planning_scratch.pool_sec_misses;
  scan_miss_candidates(source, any_miss_bit, bitset_words, demand_hist_read,
                       aux_demand_hist_read, miss_bitset_read, pool_misses,
                       pool_sec_misses);

  std::uint64_t batch_fault_frames = 0ull;
  bool any_candidates = false;
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    for (const auto &m : pool_misses[p]) {
      batch_fault_frames += m.primary_demand;
    }
    if (!pool_misses[p].empty() || !pool_sec_misses[p].empty()) {
      any_candidates = true;
    }
  }
  if (!any_candidates) {
    if (prof_on) {
      _profiler->record_empty_check();
    }
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

  auto &src_hit_slots = _sync_planning_scratch.src_hit_slots;
  auto &src_partner_hit = _sync_planning_scratch.src_partner_hit;
  compute_active_source_hints(is_aux, slot_epoch_read, src_hit_slots,
                              src_partner_hit);

  const auto prof_meta_acq_t0 =
      prof_on ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point{};
  const std::size_t meta_slot = acquire_meta_slot(cur_stream_h);
  const double prof_meta_acq_ms =
      prof_on ? std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prof_meta_acq_t0)
                    .count()
              : 0.0;
  const double prof_base_decode_ms = _stats.decode_ms;
  const double prof_base_pack_ms = _stats.pack_ms;
  const double prof_base_upload_ms = _stats.upload_ms;
  const std::uint64_t prof_base_images_decoded = _stats.images_decoded;
  const std::uint64_t prof_base_tiles_loaded = _stats.tiles_loaded;
  const std::uint64_t prof_base_tiles_evicted = _stats.tiles_evicted;
  std::uint32_t prof_b4_loaded = 0u;
  std::uint32_t prof_f4_loaded = 0u;
  std::uint32_t prof_chunks_submitted = 0u;
  bool prof_fused_final = false;
  bool prof_isolated_pub = false;
  double prof_chunk_acq_wait_ms = 0.0;
  std::vector<std::uint64_t> prof_chunk_submit_seqs;
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

  bool gpu_state_dirty = false;
  bool any_epoch_advanced = false;
  bool any_epoch_wrapped = false;
  bool slot_epoch_host_seeded = !_use_pinned_readback;

  auto &plans = _sync_planning_scratch.plans;
  plans.resize(_pools.size());
  std::size_t last_active_pool = _pools.size();
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    plan_pool_misses_host(
        static_cast<std::uint32_t>(p), _pools[p], pool_misses[p],
        pool_sec_misses[p], is_aux, source,
        std::numeric_limits<std::uint32_t>::max(), slot_epoch_read, total_slots,
        src_hit_slots, src_partner_hit, slot_epoch_host_seeded,
        any_epoch_advanced, any_epoch_wrapped, plans[p],
        _sync_planning_scratch);
    if (plans[p].has_work) {
      last_active_pool = p;
    }
  }

  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    const auto &plan = plans[p];
    if (!plan.has_work) {
      continue;
    }
    auto &pool_rt = _pools[p];
    const auto &pool = pool_rt.config;
    const bool is_final_pool = (p == last_active_pool);
    std::uint32_t prof_pool_loaded = 0u;

    const std::size_t total_load = plan.misses.size();
    const std::size_t staging_cap = std::max<std::size_t>(pool.staging_capacity, 1u);

    for (std::size_t chunk_begin = 0u; chunk_begin < total_load;
         chunk_begin += staging_cap) {
      const std::size_t chunk_end =
          std::min(chunk_begin + staging_cap, total_load);
      const std::size_t chunk_count = chunk_end - chunk_begin;
      const auto prof_acq0 =
          prof_on ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
      const std::size_t chunk_slot =
          acquire_chunk_slot(static_cast<std::uint32_t>(p), cur_stream_h);
      if (prof_on) {
        prof_chunk_acq_wait_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prof_acq0)
                .count();
      }

      ChunkStagingTarget target{.pool_index = static_cast<std::uint32_t>(p),
                                .chunk_slot = chunk_slot};
      if (_use_pinned_upload) {
        target.pack_base = pool_rt.pinned_staging_ptr[chunk_slot];
        target.um_dst = pool.use_batch_copy_upload
                            ? nullptr
                            : pool_rt.pinned_upload_meta_ptr[chunk_slot];
      } else {
        target.pack_base = pool_rt.host_staging.data();
        target.um_dst =
            pool.use_batch_copy_upload ? nullptr : _host_upload_meta.data();
      }

      const PackedChunkMeta packed = prepare_chunk_staging(
          static_cast<std::uint32_t>(p), pool_rt, plan, chunk_begin, chunk_end,
          staging_cap, target, pt_upd_dst, _host_page_table_updates,
          pt_upd_count, ep_upd_dst, _host_slot_epoch_updates, ep_upd_count,
          is_aux, gpu_state_dirty, prof_b4_loaded, prof_f4_loaded,
          prof_pool_loaded, _sync_planning_scratch);
      if (is_aux) {
        const std::uint64_t aux_pin_seq = _latest_aux_seq + 1ull;
        for (std::size_t k = chunk_begin; k < chunk_end; ++k) {
          const std::uint32_t v = plan.victim_slots[k];
          if (v < pool_rt.slot_pin_by_aux.size()) {
            pool_rt.slot_pin_by_aux[v] = 1u;
            pool_rt.slot_pin_aux_seq[v] = aux_pin_seq;
          }
        }
      }

      const auto upload_t0 = std::chrono::steady_clock::now();
      const auto chunk_u32 = static_cast<luisa::uint>(chunk_count);
      const auto elem_count = static_cast<luisa::uint>(
          chunk_count * pool.tile_staging_u32s);
      if (_use_pinned_upload) {
        if (!pool.use_batch_copy_upload) {
          _pinned_ext->flush_range(
              pool_rt.upload_meta[chunk_slot].handle(), 0ull,
              static_cast<std::uint64_t>(chunk_count) * sizeof(luisa::uint4));
        }
        _pinned_ext->flush_range(
            pool_rt.staging[chunk_slot].handle(), 0ull,
            static_cast<std::uint64_t>(elem_count) * sizeof(luisa::uint));
      }

      std::uint64_t h2d_fence = 0ull;
      if (_use_host_cached_staging && _use_copy_stream && chunk_count > 0u) {
        const auto h2d_t0 = std::chrono::steady_clock::now();
        std::uint64_t h2d_bytes = 0ull;
        h2d_fence = submit_staged_chunk_h2d(
            static_cast<std::uint32_t>(p), chunk_slot, chunk_count, true,
            &h2d_bytes);
        const double h2d_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - h2d_t0)
                .count();
        ++_stats.copy_queue_batches;
        _stats.copy_queue_h2d_bytes += h2d_bytes;
        _stats.copy_queue_submit_ms += h2d_ms;
      }

      auto emit_fallback_uploads = [&](CommandList &target_cmd) {
        if (_use_pinned_upload) {
          return;
        }
        if (!pool.use_batch_copy_upload) {
          target_cmd << pool_rt.upload_meta[chunk_slot]
                            .view(0u, chunk_u32)
                            .copy_from(luisa::span{_host_upload_meta}
                                           .subspan(0u, chunk_count));
        }
        target_cmd << pool_rt.staging[chunk_slot]
                          .view(0u, elem_count)
                          .copy_from(luisa::span{pool_rt.host_staging}
                                         .subspan(0u, elem_count));
      };

      const bool isolate_pub = prof_on && _profiler->isolate_pub();
      const bool would_fuse_final =
          (wavefront_cmd != nullptr) && is_final_pool &&
          (chunk_end == total_load);
      const bool can_fuse_direct =
          (_copy_queue_mode != LuisaVirtualTextureCopyQueueMode::direct);
      const bool fuse_last_chunk =
          can_fuse_direct && !isolate_pub && would_fuse_final;
      std::uint64_t atlas_bytes = 0ull;

      if (_copy_queue_mode == LuisaVirtualTextureCopyQueueMode::direct) {
        if (isolate_pub && would_fuse_final) {
          prof_isolated_pub = true;
        }
        if (wavefront_cmd != nullptr && !wavefront_cmd->empty()) {
          stream << wavefront_cmd->commit();
        }
        if (!_use_pinned_upload) {
          CommandList pre_cmd;
          emit_fallback_uploads(pre_cmd);
          stream << pre_cmd.commit();
        }
        const auto cq_t0 = std::chrono::steady_clock::now();
        std::array<PackedChunkMeta, 1u> one_chunk{packed};
        submit_direct_copy_chunks(stream, one_chunk, &atlas_bytes);
        ++_stats.copy_queue_batches;
        _stats.copy_queue_atlas_bytes += atlas_bytes;
        _stats.copy_queue_submit_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cq_t0)
                .count();
        if (_use_pinned_upload) {
          auto sig = _upload_event.signal();
          const auto fence = sig.fence;
          stream << std::move(sig);
          pool_rt.slot_fence[chunk_slot] = fence;
        } else {
          stream << synchronize();
        }
        ++prof_chunks_submitted;
        if (prof_on) {
          prof_chunk_submit_seqs.push_back(query_vk_timeline_submit_seq());
        }
      } else if (fuse_last_chunk) {
        prof_fused_final = true;
        if (h2d_fence != 0ull) {
          if (!wavefront_cmd->empty()) {
            stream << wavefront_cmd->commit();
          }
          stream << _copy_h2d_event.wait(h2d_fence);
        }
        emit_fallback_uploads(*wavefront_cmd);
        emit_chunk_atlas_upload(*wavefront_cmd, packed, &atlas_bytes);
        if (_use_host_cached_staging) {
          _stats.copy_queue_atlas_bytes += atlas_bytes;
        }
        pool_rt.slot_stream[chunk_slot] = cur_stream_h;
      } else {
        if (isolate_pub && would_fuse_final) {
          prof_isolated_pub = true;
        }
        if (h2d_fence != 0ull) {
          stream << _copy_h2d_event.wait(h2d_fence);
        }
        CommandList upload_cmd;
        emit_fallback_uploads(upload_cmd);
        emit_chunk_atlas_upload(upload_cmd, packed, &atlas_bytes);
        if (_use_host_cached_staging) {
          _stats.copy_queue_atlas_bytes += atlas_bytes;
        }
        if (_use_pinned_upload) {
          auto sig = _upload_event.signal();
          const auto fence = sig.fence;
          stream << upload_cmd.commit() << std::move(sig);
          pool_rt.slot_fence[chunk_slot] = fence;
        } else {
          stream << upload_cmd.commit() << synchronize();
        }
        ++prof_chunks_submitted;
        if (prof_on) {
          prof_chunk_submit_seqs.push_back(query_vk_timeline_submit_seq());
        }
      }
      _stats.gpu_upload_bytes += atlas_bytes;
      const auto upload_t1 = std::chrono::steady_clock::now();
      _stats.upload_ms +=
          std::chrono::duration<double, std::milli>(upload_t1 - upload_t0)
              .count();
    }
  }

  const double prof_chunk_upload_end_stats_ms = _stats.upload_ms;
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
    _stats.fault_frames_total += batch_fault_frames;
    if (is_aux) {
      ++_stats.aux_fault_batches;
    }
    std::uint32_t b4_resident = 0u;
    std::uint32_t f4_resident = 0u;
    for (const auto &pr : _pools) {
      std::uint32_t cnt = 0u;
      for (const auto s : pr.host_slots) {
        if (s != vt_invalid_page) {
          ++cnt;
        }
      }
      if (pr.config.is_high_precision) {
        f4_resident += cnt;
      } else {
        b4_resident += cnt;
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
      if (any_epoch_advanced || any_epoch_wrapped) {
        _published_gpu_epoch = _host_current_epoch;
      }
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
    {
      std::lock_guard lock{_vt_worker_mtx};
      const std::uint64_t next_pub_epoch = ++_publish_epoch;
      for (std::size_t p = 0u; p < _pools.size(); ++p) {
        if (!plans[p].has_work) {
          continue;
        }
        auto &pr = _pools[p];
        for (const auto v : plans[p].victim_slots) {
          if (v < pr.slot_publish_epoch.size()) {
            pr.slot_publish_epoch[v] = next_pub_epoch;
          }
        }
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
  sync_prepack_stats();
  if (prof_on) {
    if (gpu_state_dirty && pt_upd_count > 0u) {
      const std::uint64_t exit_ns = VtProfileRecorder::steady_ns();
      const double rec_ms =
          static_cast<double>(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      prof_rec_end_t.time_since_epoch())
                      .count()) -
              prof_enter_ns) /
          1.0e6;
      const double rb_sync_ms =
          std::chrono::duration<double, std::milli>(
              prof_sync_end_t - prof_rec_end_t)
              .count();
      const double rb_flush_ms =
          std::chrono::duration<double, std::milli>(
              collect_t1 - prof_sync_end_t)
              .count();
      const double d_decode = _stats.decode_ms - prof_base_decode_ms;
      const double d_pack = _stats.pack_ms - prof_base_pack_ms;
      const double chunk_upload_loop_ms =
          prof_chunk_upload_end_stats_ms - prof_base_upload_ms;
      const double total_pool_ms =
          std::chrono::duration<double, std::milli>(finalize_t0 - collect_t1)
              .count();
      const double adm_ms = std::max(
          0.0, total_pool_ms - prof_meta_acq_ms - prof_chunk_acq_wait_ms -
                   d_decode - d_pack - chunk_upload_loop_ms);
      const double pub_ms = prof_meta_acq_ms +
                            (_stats.upload_ms - prof_chunk_upload_end_stats_ms);
      const char *src_name =
          source == MissSource::wavefront_demand
              ? "wavefront_demand"
              : (source == MissSource::aux_demand ? "aux_demand"
                                                  : "bitset_only");
      const auto snap =
          (prof_ctx != nullptr && prof_ctx->snapshot.valid)
              ? prof_ctx->snapshot
              : luisa::compute::coro::detail::g_current_wavefront_scheduler_snapshot;
      _profiler->record_batch(VtProfileBatchRecord{
          .source = src_name,
          .stream_handle = static_cast<std::uint64_t>(stream.handle()),
          .queued_fault_frames =
              prof_ctx != nullptr ? prof_ctx->queued_fault_frames : 0u,
          .tiles_loaded = static_cast<std::uint32_t>(
              _stats.tiles_loaded - prof_base_tiles_loaded),
          .tiles_evicted = static_cast<std::uint32_t>(
              _stats.tiles_evicted - prof_base_tiles_evicted),
          .b4_loaded = prof_b4_loaded,
          .f4_loaded = prof_f4_loaded,
          .chunks_submitted = prof_chunks_submitted,
          .pt_updates = static_cast<std::uint32_t>(pt_upd_count),
          .ep_updates = static_cast<std::uint32_t>(ep_upd_count),
          .has_cold_decode = _stats.images_decoded > prof_base_images_decoded,
          .fused_final_chunk = prof_fused_final,
          .isolated_pub = prof_isolated_pub,
          .cpu_enter_ns = prof_enter_ns,
          .cpu_exit_ns = exit_ns,
          .cpu_collect_record_ms = std::max(0.0, rec_ms),
          .cpu_readback_sync_ms = rb_sync_ms,
          .cpu_readback_flush_ms = rb_flush_ms,
          .cpu_admission_rank_ms = adm_ms,
          .cpu_decode_ms = d_decode,
          .cpu_pack_ms = d_pack,
          .cpu_upload_submit_ms = prof_chunk_acq_wait_ms + chunk_upload_loop_ms,
          .cpu_page_table_pub_ms = pub_ms,
          .cpu_fault_to_submit_ready_ms =
              static_cast<double>(exit_ns - prof_enter_ns) / 1.0e6,
          .submit_seq_before_s1 = prof_seq_before_s1,
          .submit_seq_after_s1 = prof_seq_after_s1,
          .chunk_submit_seqs = std::move(prof_chunk_submit_seqs),
          .submit_seq_after_service = query_vk_timeline_submit_seq(),
          .snapshot = snap});
    } else {
      _profiler->record_empty_check();
    }
  }
}

} // namespace psycles::luisa_backend::detail
