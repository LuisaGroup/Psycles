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


void VirtualTextureCache::ensure_async_worker_started() noexcept {
  std::lock_guard lock{_vt_worker_mtx};
  if (_vt_worker_thread.joinable()) {
    return;
  }
  _vt_worker_stop = false;
  _vt_worker_thread = std::thread([this]() noexcept {
    vt_async_worker_loop();
  });
}

void VirtualTextureCache::stop_async_worker() noexcept {
  {
    std::unique_lock lock{_vt_worker_mtx};
    auto any_slot_busy = [&]() noexcept {
      if (!_vt_worker_queue.empty()) {
        return true;
      }
      for (const auto &slot : _async_slots) {
        if (slot.state != AsyncSlotLifecycle::Free &&
            slot.state != AsyncSlotLifecycle::PublishedAwaitingAdvance) {
          return true;
        }
      }
      return false;
    };
    _vt_ready_cv.wait(lock, [&]() noexcept { return !any_slot_busy(); });
    _vt_worker_stop = true;
  }
  _vt_worker_cv.notify_all();
  if (_vt_worker_thread.joinable()) {
    _vt_worker_thread.join();
  }
}

void VirtualTextureCache::vt_async_worker_loop() noexcept {
  const auto total_pages = static_cast<std::uint32_t>(_pages.size());
  const auto bitset_words = static_cast<std::uint32_t>(_host_miss_bitset.size());
  const auto total_slots = static_cast<std::uint32_t>(_host_slot_epoch.size());
  bool need_epoch_readback = false;
  for (const auto &pr : _pools) {
    if (pr.config.eviction_possible) {
      need_epoch_readback = true;
      break;
    }
  }
  const bool prof_on = (_profiler != nullptr);
  const std::size_t num_async_slots = std::max<std::size_t>(_async_slots.size(), 1u);
  std::vector<std::vector<std::size_t>> local_pool_cursors(
      num_async_slots, std::vector<std::size_t>(_pools.size(), 0u));
  std::vector<std::size_t> local_meta_cursors(num_async_slots, 0u);
  PlanningScratch worker_scratch;
  struct PoolPinSnapshot {
    std::vector<std::uint32_t> slot_inflight_owner;
    std::vector<std::uint64_t> slot_pin_consumer_seq;
    std::vector<std::uint64_t> slot_pin_aux_seq;
    std::vector<std::uint8_t> slot_pin_by_aux;
    std::vector<std::uint64_t> slot_publish_epoch;
  };
  std::vector<PoolPinSnapshot> pin_snapshots(_pools.size());
  std::vector<PoolPinContext> pin_ctxs(_pools.size());
  luisa::vector<luisa::uint> readback_gpu_epoch;

  auto acquire_owned_slot = [&](std::vector<std::uint64_t> &async_fences,
                                std::vector<std::uint64_t> &sync_fences,
                                std::size_t &cursor,
                                std::size_t slice_base,
                                std::size_t slice_len,
                                std::uint64_t &used_mask,
                                double &wait_ms_out) noexcept {
    LUISA_ASSERT(slice_len > 0u && slice_base + slice_len <= async_fences.size(),
                 "Invalid VT async slot slice [{}, {}) for fence array size {}.",
                 slice_base, slice_base + slice_len, async_fences.size());
    const std::uint64_t done =
        _async_publish_completed.load(std::memory_order_acquire);
    for (std::size_t k = 0u; k < slice_len; ++k) {
      const std::size_t cand = slice_base + k;
      if (sync_fences[cand] != 0ull &&
          _upload_event.is_completed(sync_fences[cand])) {
        sync_fences[cand] = 0ull;
      }
    }
    std::size_t chosen = async_fences.size();
    for (std::size_t k = 0u; k < slice_len; ++k) {
      const std::size_t local = (cursor + k) % slice_len;
      const std::size_t cand = slice_base + local;
      if (local < 64u && (used_mask & (1ull << local)) != 0ull) {
        continue;
      }
      if (async_fences[cand] <= done && sync_fences[cand] == 0ull) {
        chosen = cand;
        cursor = (local + 1u) % slice_len;
        if (local < 64u) {
          used_mask |= (1ull << local);
        }
        break;
      }
    }
    if (chosen == async_fences.size()) {
      for (std::size_t k = 0u; k < slice_len; ++k) {
        const std::size_t local = (cursor + k) % slice_len;
        const std::size_t cand = slice_base + local;
        if (local < 64u && (used_mask & (1ull << local)) != 0ull) {
          continue;
        }
        chosen = cand;
        cursor = (local + 1u) % slice_len;
        if (local < 64u) {
          used_mask |= (1ull << local);
        }
        break;
      }
    }
    LUISA_ASSERT(chosen < async_fences.size(),
                 "VT async worker slice exhausted within a single batch.");
    if (sync_fences[chosen] != 0ull) {
      const auto t0 = std::chrono::steady_clock::now();
      _upload_event.synchronize(sync_fences[chosen]);
      sync_fences[chosen] = 0ull;
      wait_ms_out += elapsed_ms_since(t0);
    }
    if (async_fences[chosen] > done) {
      const auto t0 = std::chrono::steady_clock::now();
      wait_async_fence(async_fences[chosen]);
      wait_ms_out += elapsed_ms_since(t0);
    }
    return chosen;
  };

  while (true) {
    std::size_t slot_idx = 0u;
    bool is_oldest = true;
    std::uint64_t retry_epoch_at_start = 0ull;
    std::uint64_t slot_gen = 0ull;
    std::uint64_t slot_sub_seq = 0ull;
    std::uint64_t cb_ns = 0ull;
    {
      std::unique_lock lock{_vt_worker_mtx};
      auto find_next_runnable = [&]() noexcept -> std::size_t {
        while (!_vt_worker_queue.empty()) {
          _vt_worker_queue.pop_front();
        }
        std::size_t best_idx = _async_slots.size();
        std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t s = 0u; s < _async_slots.size(); ++s) {
          const auto &sl = _async_slots[s];
          const bool runnable =
              (sl.state == AsyncSlotLifecycle::QueuedForWorker) ||
              (sl.state == AsyncSlotLifecycle::DeferredNoVictim &&
               _defer_retry_epoch > sl.deferred_at_retry_epoch);
          if (runnable && sl.submit_seq < best_seq) {
            best_seq = sl.submit_seq;
            best_idx = s;
          }
        }
        return best_idx;
      };
      auto has_gpu_inflight = [&]() noexcept {
        for (const auto &sl : _async_slots) {
          if (sl.state == AsyncSlotLifecycle::GpuReadbackInFlight) {
            return true;
          }
        }
        return false;
      };
      _vt_worker_cv.wait(lock, [&]() noexcept {
        if (find_next_runnable() < _async_slots.size()) {
          return true;
        }
        return _vt_worker_stop && !has_gpu_inflight();
      });
      slot_idx = find_next_runnable();
      if (_vt_worker_stop && slot_idx >= _async_slots.size()) {
        return;
      }
      LUISA_ASSERT(slot_idx < _async_slots.size(),
                   "VT async worker woke without runnable slot.");
      std::uint64_t min_unpublished_seq = std::numeric_limits<std::uint64_t>::max();
      for (const auto &sl : _async_slots) {
        if (sl.state != AsyncSlotLifecycle::Free &&
            sl.state != AsyncSlotLifecycle::PublishedAwaitingAdvance) {
          min_unpublished_seq = std::min(min_unpublished_seq, sl.submit_seq);
        }
      }
      auto &rb_slot = _async_slots[slot_idx];
      is_oldest = (rb_slot.submit_seq == min_unpublished_seq);
      rb_slot.state = AsyncSlotLifecycle::WorkerActive;
      retry_epoch_at_start = _defer_retry_epoch;
      slot_gen = rb_slot.ticket.generation;
      slot_sub_seq = rb_slot.submit_seq;
      cb_ns = rb_slot.callback_ns;
    }

    auto &rb_slot = _async_slots[slot_idx];
    const auto worker_t0 = std::chrono::steady_clock::now();
    const std::uint64_t worker_start_ns =
        prof_on ? VtProfileRecorder::steady_ns() : 0ull;
    const bool use_slot_pinned = (rb_slot.pinned_demand_hist_ptr != nullptr);

    const auto rb_flush_t0 = std::chrono::steady_clock::now();
    if (use_slot_pinned && _pinned_ext != nullptr) {
      _pinned_ext->flush_range(
          rb_slot.demand_hist.handle(), 0ull,
          static_cast<std::uint64_t>(total_pages) * sizeof(luisa::uint));
      _pinned_ext->flush_range(
          rb_slot.miss_bitset.handle(), 0ull,
          static_cast<std::uint64_t>(bitset_words) * sizeof(luisa::uint));
      if (need_epoch_readback) {
        _pinned_ext->flush_range(
            rb_slot.slot_epoch.handle(), 0ull,
            static_cast<std::uint64_t>(total_slots) * sizeof(luisa::uint));
      }
    }

    const luisa::uint *demand_hist_read =
        use_slot_pinned ? rb_slot.pinned_demand_hist_ptr
                        : rb_slot.fallback_demand_hist.data();
    const luisa::uint *miss_bitset_read =
        use_slot_pinned ? rb_slot.pinned_miss_bitset_ptr
                        : rb_slot.fallback_miss_bitset.data();
    const luisa::uint *slot_epoch_read =
        use_slot_pinned ? rb_slot.pinned_slot_epoch_ptr
                        : rb_slot.fallback_slot_epoch.data();

    bool any_miss_bit = false;
    for (std::uint32_t wi = 0u; wi < bitset_words; ++wi) {
      if (miss_bitset_read[wi] != 0u) {
        any_miss_bit = true;
        break;
      }
    }

    auto &pool_misses = worker_scratch.pool_misses;
    auto &pool_sec_misses = worker_scratch.pool_sec_misses;
    std::uint32_t dep_slot_mask = 0u;
    std::vector<std::uint64_t> dep_slot_gens(_async_slots.size(), 0ull);
    std::uint64_t retired_consumer_seq = 0ull;
    std::uint64_t retired_aux_seq = 0ull;
    bool force_oldest_progress = false;
    pin_snapshots.resize(_pools.size());
    readback_gpu_epoch.clear();
    std::uint64_t readback_publish_epoch = 0ull;
    {
      std::lock_guard lock{_vt_worker_mtx};
      readback_gpu_epoch = rb_slot.readback_gpu_epoch;
      readback_publish_epoch = rb_slot.readback_publish_epoch;
      scan_miss_candidates(MissSource::wavefront_demand, any_miss_bit,
                           bitset_words, demand_hist_read, nullptr,
                           miss_bitset_read, pool_misses, pool_sec_misses,
                           true, &dep_slot_mask);
      for (std::size_t other = 0u; other < _async_slots.size(); ++other) {
        if (other != slot_idx && other < 32u &&
            _async_slots[other].state == AsyncSlotLifecycle::Ready &&
            _async_slots[other].ready_batch.any_epoch_wrapped) {
          dep_slot_mask |= (1u << static_cast<std::uint32_t>(other));
        }
      }
      for (std::size_t other = 0u; other < _async_slots.size(); ++other) {
        if (other != slot_idx && other < 32u &&
            (dep_slot_mask & (1u << static_cast<std::uint32_t>(other))) != 0u &&
            is_unpublished_lifecycle(_async_slots[other].state)) {
          dep_slot_gens[other] = _async_slots[other].ticket.generation;
        }
      }
      retired_consumer_seq = _retired_consumer_seq;
      retired_aux_seq = _retired_aux_seq;
      force_oldest_progress = _force_oldest_progress;
      for (std::size_t p = 0u; p < _pools.size(); ++p) {
        pin_snapshots[p].slot_inflight_owner = _pools[p].slot_inflight_owner;
        pin_snapshots[p].slot_pin_consumer_seq = _pools[p].slot_pin_consumer_seq;
        pin_snapshots[p].slot_pin_aux_seq = _pools[p].slot_pin_aux_seq;
        pin_snapshots[p].slot_pin_by_aux = _pools[p].slot_pin_by_aux;
        pin_snapshots[p].slot_publish_epoch = _pools[p].slot_publish_epoch;
      }
    }

    PreparedAsyncBatch batch;
    batch.callback_ns = cb_ns;
    batch.any_miss_bit = any_miss_bit;
    batch.slot_index = static_cast<std::uint32_t>(slot_idx);
    batch.generation = slot_gen;
    batch.submit_seq = slot_sub_seq;
    batch.dep_slot_gens = std::move(dep_slot_gens);
    batch.rb_flush_ms = elapsed_ms_since(rb_flush_t0);
    const std::uint64_t rb_end_ns =
        prof_on ? VtProfileRecorder::steady_ns() : 0ull;
    if (prof_on && batch.rb_flush_ms > 0.0002) {
      batch.worker_events.push_back(VtProfileWorkerEvent{
          .start_ns = worker_start_ns,
          .end_ns = rb_end_ns,
          .cat = "vt_rb",
          .name = "Worker Readback Flush+Scan"});
    }

    bool any_candidates = false;
    for (std::size_t p = 0u; p < _pools.size(); ++p) {
      for (const auto &m : pool_misses[p]) {
        batch.fault_frames += m.primary_demand;
      }
      if (!pool_misses[p].empty() || !pool_sec_misses[p].empty()) {
        any_candidates = true;
      }
    }

    bool any_pool_deferred = false;
    if (any_candidates) {
      const auto adm_t0 = std::chrono::steady_clock::now();
      const std::uint64_t adm_start_ns =
          prof_on ? VtProfileRecorder::steady_ns() : 0ull;

      for (auto &pr : _pools) {
        pr.aux_streak_start_seq = std::numeric_limits<std::uint64_t>::max();
      }

      pin_ctxs.resize(_pools.size());
      for (std::size_t p = 0u; p < _pools.size(); ++p) {
        const bool has_rb_ep = (p < readback_gpu_epoch.size());
        pin_ctxs[p] = PoolPinContext{
            .slot_inflight_owner = pin_snapshots[p].slot_inflight_owner,
            .slot_pin_consumer_seq = pin_snapshots[p].slot_pin_consumer_seq,
            .slot_pin_aux_seq = pin_snapshots[p].slot_pin_aux_seq,
            .slot_pin_by_aux = pin_snapshots[p].slot_pin_by_aux,
            .slot_publish_epoch = pin_snapshots[p].slot_publish_epoch,
            .readback_publish_epoch = readback_publish_epoch,
            .readback_gpu_epoch = has_rb_ep ? readback_gpu_epoch[p] : 0u,
            .has_readback_epoch = has_rb_ep,
            .retired_consumer_seq = retired_consumer_seq,
            .retired_aux_seq = retired_aux_seq,
            .is_oldest = is_oldest,
            .force_oldest_progress =
                (num_async_slots <= 1u) || force_oldest_progress};
      }

      auto &src_hit_slots = worker_scratch.src_hit_slots;
      auto &src_partner_hit = worker_scratch.src_partner_hit;
      compute_active_source_hints(false, slot_epoch_read, src_hit_slots,
                                  src_partner_hit, pin_ctxs);

      bool slot_epoch_host_seeded = !use_slot_pinned;
      auto &plans = worker_scratch.plans;
      plans.resize(_pools.size());
      bool any_plan_work = false;
      std::uint32_t pools_with_misses = 0u;
      std::uint32_t pools_backpressured = 0u;
      for (std::size_t p = 0u; p < _pools.size(); ++p) {
        const std::uint32_t max_pool_tiles =
            std::max<std::uint32_t>(1u, _pools[p].config.staging_capacity) *
            static_cast<std::uint32_t>(
                std::max<std::size_t>(_pools[p].config.chunks_per_batch_slot, 1u));
        const bool had_primary_misses = !pool_misses[p].empty();
        bool pool_deferred = false;
        bool pool_bp = false;
        plan_pool_misses_host(
            static_cast<std::uint32_t>(p), _pools[p], pool_misses[p],
            pool_sec_misses[p], false, MissSource::wavefront_demand,
            max_pool_tiles, slot_epoch_read, total_slots, src_hit_slots,
            src_partner_hit, slot_epoch_host_seeded, batch.any_epoch_advanced,
            batch.any_epoch_wrapped, plans[p], worker_scratch, &pin_ctxs[p],
            &pool_deferred, &pool_bp);
        any_pool_deferred |= pool_deferred;
        if (had_primary_misses) {
          ++pools_with_misses;
          if (pool_bp) {
            ++pools_backpressured;
          }
        }
        if (plans[p].has_work) {
          any_plan_work = true;
        }
      }

      constexpr std::uint32_t kBackpressureHoldBatches = 16u;
      const bool domain_backpressured =
          (!any_plan_work && any_pool_deferred) ||
          (pools_with_misses > 0u && pools_backpressured == pools_with_misses);
      if (domain_backpressured) {
        _async_backpressure_cooldown.store(kBackpressureHoldBatches,
                                           std::memory_order_relaxed);
      } else if (any_plan_work) {
        auto cur = _async_backpressure_cooldown.load(std::memory_order_relaxed);
        if (cur > 0u) {
          _async_backpressure_cooldown.store(cur - 1u, std::memory_order_relaxed);
        }
      }

      batch.adm_ms = elapsed_ms_since(adm_t0);
      const std::uint64_t adm_end_ns =
          prof_on ? VtProfileRecorder::steady_ns() : 0ull;
      if (prof_on && batch.adm_ms > 0.0002) {
        batch.worker_events.push_back(VtProfileWorkerEvent{
            .start_ns = adm_start_ns,
            .end_ns = adm_end_ns,
            .cat = "vt_adm",
            .name = "Worker Admission+Rank"});
      }

      if (!any_plan_work && any_pool_deferred) {
        std::lock_guard lock{_vt_worker_mtx};
        rb_slot.state = AsyncSlotLifecycle::DeferredNoVictim;
        rb_slot.deferred_at_retry_epoch = retry_epoch_at_start;
        ++_stats.async_deferred_admissions;
        _stats.async_worker_busy_ms += elapsed_ms_since(worker_t0);
        continue;
      }

      if (any_plan_work) {
        for (std::size_t p = 0u; p < _pools.size(); ++p) {
          if (!plans[p].has_work) {
            continue;
          }
          const auto &pr = _pools[p];
          for (const auto v : plans[p].victim_slots) {
            batch.pinned_pool_slots.emplace_back(
                static_cast<std::uint32_t>(p), v);
            if (v < pr.host_slots.size()) {
              const std::uint32_t old_p = pr.host_slots[v];
              if (old_p != vt_invalid_page) {
                batch.pinned_evict_pages.push_back(old_p);
              }
            }
          }
          for (const auto &m : plans[p].misses) {
            batch.pinned_load_pages.push_back(m.page);
          }
        }

        const std::size_t meta_slice_len =
            std::min<std::size_t>(2u, _meta_ring_size / num_async_slots);
        const std::size_t meta_slice_base = slot_idx * meta_slice_len;
        std::uint64_t meta_used_mask = 0ull;
        batch.meta_slot = acquire_owned_slot(
            _meta_slot_async_fence, _meta_slot_fence,
            local_meta_cursors[slot_idx], meta_slice_base,
            std::max<std::size_t>(meta_slice_len, 1u), meta_used_mask,
            batch.slot_wait_ms);
        luisa::uint2 *pt_upd_dst =
            _use_pinned_upload ? _pinned_page_table_updates_ptr[batch.meta_slot]
                               : nullptr;
        luisa::uint2 *ep_upd_dst =
            _use_pinned_upload ? _pinned_slot_epoch_updates_ptr[batch.meta_slot]
                               : nullptr;

        const double base_decode_ms = _stats.decode_ms;
        const double base_pack_ms = _stats.pack_ms;
        const std::uint64_t base_images_decoded = _stats.images_decoded;
        const std::uint64_t base_tiles_loaded = _stats.tiles_loaded;
        const std::uint64_t base_tiles_evicted = _stats.tiles_evicted;
        bool gpu_state_dirty = false;

        const auto pack_stage_t0 = std::chrono::steady_clock::now();
        const std::uint64_t pack_start_ns =
            prof_on ? VtProfileRecorder::steady_ns() : 0ull;
        std::vector<VtProfileWorkerEvent> h2d_worker_events;
        CommandList batch_h2d_cmd;
        batch.pool_loaded.assign(_pools.size(), 0u);
        batch.pool_resident.assign(_pools.size(), 0u);

        auto run_pool_chunks = [&](std::uint32_t pool_idx,
                                   const PoolAdmissionPlan &plan) {
          if (!plan.has_work) {
            return;
          }
          auto &pool_rt = _pools[pool_idx];
          const auto &pool = pool_rt.config;
          const std::size_t slice_len =
              std::max<std::size_t>(pool.chunks_per_batch_slot, 1u);
          const std::size_t slice_base = slot_idx * slice_len;
          std::uint64_t pool_used_mask = 0ull;
          const std::size_t staging_cap =
              std::max<std::size_t>(pool.staging_capacity, 1u);
          const std::size_t total_load = plan.misses.size();
          for (std::size_t chunk_begin = 0u; chunk_begin < total_load;
               chunk_begin += staging_cap) {
            const std::size_t chunk_end =
                std::min(chunk_begin + staging_cap, total_load);
            const std::size_t chunk_count = chunk_end - chunk_begin;
            const std::size_t chunk_slot = acquire_owned_slot(
                pool_rt.slot_async_fence, pool_rt.slot_fence,
                local_pool_cursors[slot_idx][pool_idx], slice_base, slice_len,
                pool_used_mask, batch.slot_wait_ms);

            luisa::vector<luisa::uint> fb_staging;
            luisa::vector<luisa::uint4> fb_um;
            ChunkStagingTarget target{.pool_index = pool_idx,
                                      .chunk_slot = chunk_slot};
            if (_use_pinned_upload) {
              target.pack_base = pool_rt.pinned_staging_ptr[chunk_slot];
              target.um_dst = pool.use_batch_copy_upload
                                  ? nullptr
                                  : pool_rt.pinned_upload_meta_ptr[chunk_slot];
            } else {
              fb_staging.resize(chunk_count * pool.tile_staging_u32s);
              target.pack_base = fb_staging.data();
              if (!pool.use_batch_copy_upload) {
                fb_um.resize(chunk_count);
                target.um_dst = fb_um.data();
              }
            }

            PackedChunkMeta chunk = prepare_chunk_staging(
                pool_idx, pool_rt, plan, chunk_begin, chunk_end, staging_cap,
                target, pt_upd_dst, batch.fallback_pt_updates,
                batch.pt_upd_count, ep_upd_dst, batch.fallback_ep_updates,
                batch.ep_upd_count, false, gpu_state_dirty, batch.b4_loaded,
                batch.f4_loaded, batch.pool_loaded[pool_idx], worker_scratch);
            if (_use_pinned_upload && _pinned_ext != nullptr) {
              if (!pool.use_batch_copy_upload) {
                _pinned_ext->flush_range(
                    pool_rt.upload_meta[chunk_slot].handle(), 0ull,
                    static_cast<std::uint64_t>(chunk_count) *
                        sizeof(luisa::uint4));
              }
              _pinned_ext->flush_range(
                  pool_rt.staging[chunk_slot].handle(), 0ull,
                  static_cast<std::uint64_t>(chunk_count) *
                      pool.tile_staging_u32s * sizeof(luisa::uint));
              if (_use_host_cached_staging && _use_copy_stream &&
                  chunk.chunk_count > 0u) {
                append_staged_chunk_h2d(batch_h2d_cmd, pool_idx, chunk_slot,
                                        chunk.chunk_count, &batch.h2d_bytes);
              }
            } else {
              chunk.fallback_staging = std::move(fb_staging);
              chunk.fallback_um = std::move(fb_um);
            }
            batch.chunks.push_back(std::move(chunk));
          }
        };

        for (std::size_t p = 0u; p < _pools.size(); ++p) {
          run_pool_chunks(static_cast<std::uint32_t>(p), plans[p]);
        }
        if (_use_host_cached_staging && _use_copy_stream &&
            !batch_h2d_cmd.empty()) {
          const auto h2d_t0 = std::chrono::steady_clock::now();
          const std::uint64_t h2d_start_ns =
              prof_on ? VtProfileRecorder::steady_ns() : 0ull;
          batch.h2d_fence = submit_staged_batch_h2d(batch_h2d_cmd);
          const double h2d_ms = elapsed_ms_since(h2d_t0);
          batch.copy_submit_ms += h2d_ms;
          if (prof_on && h2d_ms > 0.0002) {
            h2d_worker_events.push_back(VtProfileWorkerEvent{
                .start_ns = h2d_start_ns,
                .end_ns = VtProfileRecorder::steady_ns(),
                .cat = "vt_up",
                .name = "Worker H2D CopyQ"});
          }
        }

        batch.has_work = gpu_state_dirty && batch.pt_upd_count > 0u;
        batch.decode_ms = _stats.decode_ms - base_decode_ms;
        batch.pack_ms = _stats.pack_ms - base_pack_ms;
        batch.has_cold_decode = _stats.images_decoded > base_images_decoded;
        batch.tiles_loaded =
            static_cast<std::uint32_t>(_stats.tiles_loaded - base_tiles_loaded);
        batch.tiles_evicted = static_cast<std::uint32_t>(
            _stats.tiles_evicted - base_tiles_evicted);

        const std::uint64_t pack_end_ns =
            prof_on ? VtProfileRecorder::steady_ns() : 0ull;
        const double pack_stage_ms = elapsed_ms_since(pack_stage_t0);
        if (prof_on && pack_stage_ms > 0.0002) {
          batch.worker_events.push_back(VtProfileWorkerEvent{
              .start_ns = pack_start_ns,
              .end_ns = pack_end_ns,
              .cat = batch.has_cold_decode ? "vt_dec" : "vt_pack",
              .name = batch.has_cold_decode ? "Worker Decode+Pack"
                                            : "Worker Pack+Stage"});
          for (auto &hev : h2d_worker_events) {
            batch.worker_events.push_back(std::move(hev));
          }
        }

        if (batch.has_work) {
          for (std::size_t p = 0u; p < _pools.size(); ++p) {
            std::uint32_t cnt = 0u;
            for (const auto s : _pools[p].host_slots) {
              if (s != vt_invalid_page) {
                ++cnt;
              }
            }
            batch.pool_resident[p] = cnt;
            if (_pools[p].config.is_high_precision) {
              batch.f4_peak_resident += cnt;
            } else {
              batch.b4_peak_resident += cnt;
            }
          }
          if (_use_pinned_upload && _pinned_ext != nullptr) {
            _pinned_ext->flush_range(
                _page_table_updates[batch.meta_slot].handle(), 0ull,
                static_cast<std::uint64_t>(batch.pt_upd_count) *
                    sizeof(luisa::uint2));
            if (need_epoch_readback && batch.ep_upd_count > 0u) {
              _pinned_ext->flush_range(
                  _slot_epoch_updates[batch.meta_slot].handle(), 0ull,
                  static_cast<std::uint64_t>(batch.ep_upd_count) *
                      sizeof(luisa::uint2));
            }
          }
          if (need_epoch_readback) {
            if (batch.any_epoch_advanced) {
              batch.current_epoch_upload = _host_current_epoch;
            }
            if (batch.any_epoch_wrapped) {
              batch.slot_epoch_full_upload = _host_slot_epoch;
            }
          }
        }
      }
      sync_prepack_stats();
    }

    batch.worker_busy_ms = elapsed_ms_since(worker_t0);
    {
      std::lock_guard lock{_vt_worker_mtx};
      _stats.async_worker_busy_ms += batch.worker_busy_ms;
      if (any_pool_deferred) {
        ++_stats.async_deferred_admissions;
      }
      if (batch.has_work) {
        for (const auto &[p, v] : batch.pinned_pool_slots) {
          if (p < _pools.size() && v < _pools[p].slot_inflight_owner.size()) {
            _pools[p].slot_inflight_owner[v] =
                static_cast<std::uint32_t>(slot_idx);
          }
        }
        if (slot_idx < 32u) {
          const std::uint32_t slot_bit =
              (1u << static_cast<std::uint32_t>(slot_idx));
          for (const auto old_p : batch.pinned_evict_pages) {
            if (old_p < _page_evict_owner_mask.size()) {
              _page_evict_owner_mask[old_p] |= slot_bit;
            }
          }
        }
        for (const auto new_p : batch.pinned_load_pages) {
          if (new_p < _page_load_inflight_slot.size()) {
            _page_load_inflight_slot[new_p] =
                static_cast<std::uint32_t>(slot_idx);
          }
        }
      } else {
        batch.pinned_pool_slots.clear();
        batch.pinned_evict_pages.clear();
        batch.pinned_load_pages.clear();
      }
      rb_slot.ready_batch = std::move(batch);
      rb_slot.state = AsyncSlotLifecycle::Ready;
    }
    _vt_ready_cv.notify_all();
  }
}

bool VirtualTextureCache::submit_async_wavefront_collect(
    const coro::WavefrontCoroExtensionDispatchContext &ctx,
    const CollectShader &collect_shader) noexcept {
  if (!has_virtual_images() || ctx.frame_count == 0u ||
      !_options.enable_async_worker) {
    return false;
  }
  const auto enter_t0 = std::chrono::steady_clock::now();
  const std::uint64_t enter_ns =
      _profiler ? VtProfileRecorder::steady_ns() : 0ull;
  ensure_async_worker_started();

  const std::size_t slot_idx =
      ctx.async_ticket.valid()
          ? static_cast<std::size_t>(ctx.async_ticket.slot_index)
          : 0u;
  LUISA_ASSERT(slot_idx < _async_slots.size(),
               "VT async ticket slot_index {} out of range ({}).",
               slot_idx, _async_slots.size());
  const coro::WavefrontCoroAsyncBatchTicket ticket =
      ctx.async_ticket.valid()
          ? ctx.async_ticket
          : coro::WavefrontCoroAsyncBatchTicket{.slot_index = 0u,
                                                .generation = 1ull};
  bool wake_worker_on_progress = false;
  {
    std::unique_lock lock{_vt_worker_mtx};
    _vt_ready_cv.wait(lock, [&]() noexcept {
      return _async_slots[slot_idx].state == AsyncSlotLifecycle::Free;
    });
    auto &sl = _async_slots[slot_idx];
    sl.state = AsyncSlotLifecycle::GpuReadbackInFlight;
    sl.ticket = ticket;
    sl.submit_seq = ++_async_submit_seq;
    sl.callback_ns = 0ull;
    sl.deferred_at_retry_epoch = 0ull;
    sl.stream_handle = static_cast<std::uint64_t>(ctx.stream.handle());
    sl.enter_ns = enter_ns;
    sl.submit_end_ns = 0ull;
    sl.submit_seq_before_s1 = 0ull;
    sl.submit_seq_after_s1 = 0ull;
    sl.queued_fault_frames = ctx.frame_count;
    sl.collect_record_ms = 0.0;
    sl.wait_ms = 0.0;
    sl.snapshot = ctx.scheduler_snapshot;
    sl.readback_gpu_epoch = _published_gpu_epoch;
    sl.readback_publish_epoch = _publish_epoch;
    sl.ready_batch = {};
    if (ctx.scheduler_snapshot.valid) {
      wake_worker_on_progress =
          update_scheduler_progress_locked(ctx.scheduler_snapshot, false);
    }
    std::uint32_t active_slots = 0u;
    for (const auto &other : _async_slots) {
      if (other.state != AsyncSlotLifecycle::Free) {
        ++active_slots;
      }
    }
    sl.in_flight_at_submit = active_slots;
    _stats.async_peak_in_flight =
        std::max(_stats.async_peak_in_flight, active_slots);
  }
  if (wake_worker_on_progress) {
    _vt_worker_cv.notify_one();
  }
  _last_async_wait_ms.store(0.0, std::memory_order_relaxed);

  CommandList local_cmd;
  CommandList *active_cmd =
      ctx.command_list != nullptr ? ctx.command_list : &local_cmd;
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  const auto bitset_words = static_cast<luisa::uint>(_host_miss_bitset.size());
  bool need_epoch_readback = false;
  for (const auto &pr : _pools) {
    if (pr.config.eviction_possible) {
      need_epoch_readback = true;
      break;
    }
  }
  auto &rb_slot = _async_slots[slot_idx];

  *active_cmd
      << _clear_u32_shader(_demand_hist, total_pages).dispatch(total_pages)
      << collect_shader(ctx.frame_buffer, ctx.frame_indices, _demand_hist,
                        ctx.frame_capacity, ctx.frame_count)
             .dispatch(ctx.frame_count);
  if (rb_slot.pinned_demand_hist_ptr != nullptr) {
    *active_cmd << _demand_hist.view().copy_to(rb_slot.demand_hist.view())
                << _miss_bitset.view().copy_to(rb_slot.miss_bitset.view());
    if (need_epoch_readback) {
      *active_cmd << _slot_epoch.view().copy_to(rb_slot.slot_epoch.view());
    }
  } else {
    *active_cmd
        << _demand_hist.copy_to(luisa::span{rb_slot.fallback_demand_hist})
        << _miss_bitset.copy_to(luisa::span{rb_slot.fallback_miss_bitset});
    if (need_epoch_readback) {
      *active_cmd
          << _slot_epoch.copy_to(luisa::span{rb_slot.fallback_slot_epoch});
    }
  }
  *active_cmd << _clear_u32_shader(_miss_bitset, bitset_words)
                     .dispatch(bitset_words);

  const std::uint64_t gen = ticket.generation;
  active_cmd->add_callback([this, slot_idx, gen]() noexcept {
    const std::uint64_t cb_ns =
        _profiler ? VtProfileRecorder::steady_ns() : 0ull;
    {
      std::lock_guard lock{_vt_worker_mtx};
      auto &sl = _async_slots[slot_idx];
      if (sl.ticket.generation == gen &&
          sl.state == AsyncSlotLifecycle::GpuReadbackInFlight) {
        sl.callback_ns = cb_ns;
        sl.state = AsyncSlotLifecycle::QueuedForWorker;
        _vt_worker_queue.push_back(slot_idx);
      }
    }
    _vt_worker_cv.notify_one();
  });

  rb_slot.submit_seq_before_s1 =
      _profiler ? query_vk_timeline_submit_seq() : 0ull;
  rb_slot.collect_record_ms = elapsed_ms_since(enter_t0);
  ctx.stream << active_cmd->commit();
  rb_slot.submit_end_ns = _profiler ? VtProfileRecorder::steady_ns() : 0ull;
  rb_slot.submit_seq_after_s1 =
      _profiler ? query_vk_timeline_submit_seq() : 0ull;
  return true;
}

bool VirtualTextureCache::update_scheduler_progress_locked(
    const coro::WavefrontCoroSchedulerSnapshot &snapshot,
    bool waiting_for_progress) noexcept {
  bool changed = false;
  if (!_has_snapshot_seq) {
    _has_snapshot_seq = true;
    _last_snapshot_consumer_seq = snapshot.async_consumer_dispatch_seq;
    _last_snapshot_aux_seq = snapshot.auxiliary_dispatch_seq;
    if (snapshot.async_consumer_dispatch_seq > _latest_consumer_seq) {
      _latest_consumer_seq = snapshot.async_consumer_dispatch_seq;
      if (_latest_consumer_seq > _retired_consumer_seq) {
        _retired_consumer_seq = _latest_consumer_seq;
        changed = true;
      }
    }
    if (snapshot.auxiliary_dispatch_seq > _latest_aux_seq) {
      _latest_aux_seq = snapshot.auxiliary_dispatch_seq;
      if (_latest_aux_seq > _retired_aux_seq) {
        _retired_aux_seq = _latest_aux_seq;
        changed = true;
      }
    }
  } else {
    if (snapshot.async_consumer_dispatch_seq != _last_snapshot_consumer_seq) {
      const std::uint64_t delta_c =
          (snapshot.async_consumer_dispatch_seq > _last_snapshot_consumer_seq)
              ? (snapshot.async_consumer_dispatch_seq -
                 _last_snapshot_consumer_seq)
              : (snapshot.async_consumer_dispatch_seq + 1ull);
      _last_snapshot_consumer_seq = snapshot.async_consumer_dispatch_seq;
      _latest_consumer_seq += delta_c;
      _retired_consumer_seq =
          std::max(_retired_consumer_seq, _latest_consumer_seq);
      changed = true;
    }
    if (snapshot.auxiliary_dispatch_seq != _last_snapshot_aux_seq) {
      const std::uint64_t delta_a =
          (snapshot.auxiliary_dispatch_seq > _last_snapshot_aux_seq)
              ? (snapshot.auxiliary_dispatch_seq - _last_snapshot_aux_seq)
              : (snapshot.auxiliary_dispatch_seq + 1ull);
      _last_snapshot_aux_seq = snapshot.auxiliary_dispatch_seq;
      _latest_aux_seq += delta_a;
      _retired_aux_seq = std::max(_retired_aux_seq, _latest_aux_seq);
      changed = true;
    }
  }
  if (snapshot.auxiliary_ready_frames == 0ull &&
      _retired_aux_seq <= _latest_aux_seq) {
    _retired_aux_seq = _latest_aux_seq + 1ull;
    changed = true;
  }
  if (waiting_for_progress && !_force_oldest_progress) {
    _force_oldest_progress = true;
    changed = true;
  }
  bool any_deferred = false;
  for (const auto &sl : _async_slots) {
    if (sl.state == AsyncSlotLifecycle::DeferredNoVictim) {
      any_deferred = true;
      break;
    }
  }
  if (changed || (waiting_for_progress && any_deferred)) {
    ++_defer_retry_epoch;
    return any_deferred;
  }
  return false;
}

void VirtualTextureCache::notify_async_scheduler_progress(
    const coro::WavefrontCoroSchedulerSnapshot &snapshot,
    bool waiting_for_progress) noexcept {
  bool wake_worker = false;
  {
    std::lock_guard lock{_vt_worker_mtx};
    wake_worker =
        update_scheduler_progress_locked(snapshot, waiting_for_progress);
  }
  if (wake_worker) {
    _vt_worker_cv.notify_one();
  }
}

bool VirtualTextureCache::batch_has_unsatisfied_dep(
    const PreparedAsyncBatch &batch) const noexcept {
  const std::size_t dep_count =
      std::min(batch.dep_slot_gens.size(), _async_slots.size());
  for (std::size_t other = 0u; other < dep_count; ++other) {
    const std::uint64_t dep_gen = batch.dep_slot_gens[other];
    if (dep_gen != 0ull &&
        is_unpublished_lifecycle(_async_slots[other].state) &&
        _async_slots[other].ticket.generation == dep_gen) {
      return true;
    }
  }
  return false;
}

coro::WavefrontCoroAsyncPollResult
VirtualTextureCache::poll_async_wavefront() noexcept {
  coro::WavefrontCoroAsyncBatchTicket ticket{};
  if (poll_async_wavefront_ready(ticket)) {
    return coro::WavefrontCoroAsyncPollResult::ready;
  }
  std::lock_guard lock{_vt_worker_mtx};
  for (const auto &sl : _async_slots) {
    if (sl.state != AsyncSlotLifecycle::Free) {
      return coro::WavefrontCoroAsyncPollResult::pending;
    }
  }
  return coro::WavefrontCoroAsyncPollResult::idle;
}

bool VirtualTextureCache::poll_async_wavefront_ready(
    coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
  std::lock_guard lock{_vt_worker_mtx};
  const std::size_t n = _async_slots.size();
  std::size_t awaiting_slot = n;
  std::uint64_t awaiting_seq = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t s = 0u; s < n; ++s) {
    const auto &sl = _async_slots[s];
    if (sl.state == AsyncSlotLifecycle::PublishedAwaitingAdvance &&
        sl.submit_seq < awaiting_seq) {
      awaiting_seq = sl.submit_seq;
      awaiting_slot = s;
    }
  }
  if (awaiting_slot < n) {
    out_ticket = _async_slots[awaiting_slot].ticket;
    return true;
  }
  std::size_t best_slot = n;
  std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t s = 0u; s < n; ++s) {
    const auto &sl = _async_slots[s];
    if (sl.state != AsyncSlotLifecycle::Ready) {
      continue;
    }
    if (batch_has_unsatisfied_dep(sl.ready_batch)) {
      continue;
    }
    if (_use_host_cached_staging && _use_copy_stream &&
        sl.ready_batch.h2d_fence != 0ull &&
        !_copy_h2d_event.is_completed(sl.ready_batch.h2d_fence)) {
      continue;
    }
    if (sl.submit_seq < best_seq) {
      best_seq = sl.submit_seq;
      best_slot = s;
    }
  }
  if (best_slot < n) {
    out_ticket = _async_slots[best_slot].ticket;
    return true;
  }
  return false;
}

void VirtualTextureCache::wait_async_wavefront() noexcept {
  coro::WavefrontCoroAsyncBatchTicket ticket{};
  static_cast<void>(wait_async_wavefront_any(ticket));
}

bool VirtualTextureCache::wait_async_wavefront_any(
    coro::WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
  std::unique_lock lock{_vt_worker_mtx};
  const std::size_t n = _async_slots.size();
  auto find_wait_ready = [&](bool include_awaiting) noexcept -> std::size_t {
    if (include_awaiting) {
      std::size_t awaiting_slot = n;
      std::uint64_t awaiting_seq = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t s = 0u; s < n; ++s) {
        const auto &sl = _async_slots[s];
        if (sl.state == AsyncSlotLifecycle::PublishedAwaitingAdvance &&
            sl.submit_seq < awaiting_seq) {
          awaiting_seq = sl.submit_seq;
          awaiting_slot = s;
        }
      }
      if (awaiting_slot < n) {
        return awaiting_slot;
      }
    }
    std::size_t best_slot = n;
    std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t s = 0u; s < n; ++s) {
      const auto &sl = _async_slots[s];
      if (sl.state != AsyncSlotLifecycle::Ready) {
        continue;
      }
      if (batch_has_unsatisfied_dep(sl.ready_batch)) {
        continue;
      }
      if (sl.submit_seq < best_seq) {
        best_seq = sl.submit_seq;
        best_slot = s;
      }
    }
    return best_slot;
  };
  auto any_active = [&]() noexcept {
    for (const auto &sl : _async_slots) {
      if (sl.state != AsyncSlotLifecycle::Free) {
        return true;
      }
    }
    return false;
  };

  if (const std::size_t immediate = find_wait_ready(true); immediate < n) {
    out_ticket = _async_slots[immediate].ticket;
    return true;
  }
  if (!any_active()) {
    return false;
  }

  _force_oldest_progress = true;
  ++_defer_retry_epoch;
  lock.unlock();
  _vt_worker_cv.notify_one();
  lock.lock();

  const std::size_t again = find_wait_ready(true);
  if (again < n) {
    out_ticket = _async_slots[again].ticket;
    return true;
  }
  if (!any_active()) {
    return false;
  }

  const auto wait_t0 = std::chrono::steady_clock::now();
  std::size_t ready_slot = n;
  while (!_vt_ready_cv.wait_for(lock, std::chrono::seconds(5), [&]() noexcept {
    ready_slot = find_wait_ready(true);
    return ready_slot < n || !any_active();
  })) {
    std::string dump;
    for (std::size_t s = 0u; s < n; ++s) {
      dump += fmt::format(
          "[s={}:st={},seq={},gen={},unsat={},h2d={}] ",
          s, static_cast<unsigned>(_async_slots[s].state),
          _async_slots[s].submit_seq,
          _async_slots[s].ticket.generation,
          batch_has_unsatisfied_dep(_async_slots[s].ready_batch) ? 1u : 0u,
          _async_slots[s].ready_batch.h2d_fence);
    }
    LUISA_WARNING(
        "wait_async_wavefront_any stalled >5s: defer_epoch={} pub_epoch={} "
        "force_oldest={} latest_c={} retired_c={} latest_a={} retired_a={} slots={}",
        _defer_retry_epoch, _publish_epoch, _force_oldest_progress,
        _latest_consumer_seq, _retired_consumer_seq,
        _latest_aux_seq, _retired_aux_seq, dump);
  }
  const double w_ms = elapsed_ms_since(wait_t0);
  _stats.async_wait_ms += w_ms;
  _last_async_wait_ms.fetch_add(w_ms, std::memory_order_relaxed);
  if (ready_slot < n) {
    ++_stats.async_wait_calls;
    _async_slots[ready_slot].wait_ms += w_ms;
    out_ticket = _async_slots[ready_slot].ticket;
    return true;
  }
  return false;
}

void VirtualTextureCache::publish_prepared_batch(
    std::size_t slot_idx,
    Stream &stream,
    CommandList &cmd,
    bool mark_awaiting_advance) noexcept {
  auto kept_batch = std::make_shared<PreparedAsyncBatch>();
  std::uint64_t slot_stream_handle = 0ull;
  std::uint64_t slot_enter_ns = 0ull;
  std::uint64_t slot_submit_end_ns = 0ull;
  std::uint64_t slot_seq_before_s1 = 0ull;
  std::uint64_t slot_seq_after_s1 = 0ull;
  std::uint32_t slot_fault_frames = 0u;
  std::uint32_t slot_in_flight_at_submit = 0u;
  double slot_collect_record_ms = 0.0;
  double slot_wait_ms = 0.0;
  coro::WavefrontCoroSchedulerSnapshot slot_snapshot{};
  {
    std::lock_guard lock{_vt_worker_mtx};
    auto &sl = _async_slots[slot_idx];
    for (std::size_t k = 0u; k < _async_slots.size(); ++k) {
      if (k == slot_idx) {
        continue;
      }
      const auto &other = _async_slots[k];
      if (other.state != AsyncSlotLifecycle::Free &&
          other.state != AsyncSlotLifecycle::PublishedAwaitingAdvance &&
          other.submit_seq < sl.submit_seq) {
        ++_stats.async_out_of_order_publishes;
        break;
      }
    }
    *kept_batch = std::move(sl.ready_batch);
    slot_stream_handle = sl.stream_handle;
    slot_enter_ns = sl.enter_ns;
    slot_submit_end_ns = sl.submit_end_ns;
    slot_seq_before_s1 = sl.submit_seq_before_s1;
    slot_seq_after_s1 = sl.submit_seq_after_s1;
    slot_fault_frames = sl.queued_fault_frames;
    slot_in_flight_at_submit = sl.in_flight_at_submit;
    slot_collect_record_ms = sl.collect_record_ms;
    slot_wait_ms = sl.wait_ms;
    sl.wait_ms = 0.0;
    slot_snapshot = sl.snapshot;
  }

  auto &batch = *kept_batch;
  const bool prof_on = (_profiler != nullptr);
  const auto pub_t0 = std::chrono::steady_clock::now();
  const std::uint64_t pub_enter_ns =
      prof_on ? VtProfileRecorder::steady_ns() : 0ull;
  bool need_epoch_readback = false;
  for (const auto &pr : _pools) {
    if (pr.config.eviction_possible) {
      need_epoch_readback = true;
      break;
    }
  }

  std::uint64_t pub_fence = 0ull;
  if (batch.has_work) {
    ++_stats.fault_batches;
    _stats.fault_frames_total += batch.fault_frames;
    ++_stats.async_batches;
    _stats.byte4_peak_resident =
        std::max(_stats.byte4_peak_resident, batch.b4_peak_resident);
    _stats.float4_peak_resident =
        std::max(_stats.float4_peak_resident, batch.f4_peak_resident);

    std::vector<std::uint64_t> seen_masks(_pools.size(), 0ull);
    for (const auto &c : batch.chunks) {
      const std::uint64_t slot_bit =
          (c.chunk_slot < 64u) ? (1ull << c.chunk_slot) : 0ull;
      auto &seen_mask = seen_masks[c.pool_index];
      LUISA_ASSERT(c.chunk_slot < _pools[c.pool_index].staging.size() &&
                       (seen_mask & slot_bit) == 0ull,
                   "Duplicate VT async chunk_slot {} within a single batch (pool={}).",
                   c.chunk_slot, c.pool_index);
      seen_mask |= slot_bit;
      if (!_use_pinned_upload) {
        auto &pr = _pools[c.pool_index];
        const auto &chunk_pool = pr.config;
        const auto elem_count = static_cast<luisa::uint>(
            c.chunk_count * chunk_pool.tile_staging_u32s);
        if (!chunk_pool.use_batch_copy_upload) {
          cmd << pr.upload_meta[c.chunk_slot]
                     .view(0u, c.chunk_count)
                     .copy_from(luisa::span{c.fallback_um});
        }
        cmd << pr.staging[c.chunk_slot]
                   .view(0u, elem_count)
                   .copy_from(luisa::span{c.fallback_staging});
      }
    }

    std::uint64_t atlas_bytes = 0ull;
    if (_copy_queue_mode == LuisaVirtualTextureCopyQueueMode::direct) {
      if (!cmd.empty()) {
        stream << cmd.commit();
      }
      const auto cq_t0 = std::chrono::steady_clock::now();
      submit_direct_copy_chunks(stream, batch.chunks, &atlas_bytes);
      ++_stats.copy_queue_batches;
      _stats.copy_queue_atlas_bytes += atlas_bytes;
      _stats.copy_queue_submit_ms += elapsed_ms_since(cq_t0);
    } else {
      if (_use_host_cached_staging && _use_copy_stream &&
          batch.h2d_fence != 0ull) {
        if (!cmd.empty()) {
          stream << cmd.commit();
        }
        stream << _copy_h2d_event.wait(batch.h2d_fence);
      }
      for (const auto &c : batch.chunks) {
        emit_chunk_atlas_upload(cmd, c, &atlas_bytes);
      }
      if (_use_host_cached_staging) {
        ++_stats.copy_queue_batches;
        _stats.copy_queue_h2d_bytes += batch.h2d_bytes;
        _stats.copy_queue_atlas_bytes += atlas_bytes;
        _stats.copy_queue_submit_ms += batch.copy_submit_ms;
      }
    }
    _stats.gpu_upload_bytes += atlas_bytes;

    const auto upd_count = static_cast<luisa::uint>(batch.pt_upd_count);
    if (!_use_pinned_upload) {
      cmd << _page_table_updates[batch.meta_slot]
                 .view(0u, upd_count)
                 .copy_from(luisa::span{batch.fallback_pt_updates});
    }
    cmd << _apply_page_table_updates_shader(
               _page_table, _page_table_updates[batch.meta_slot], upd_count)
               .dispatch(upd_count);

    if (need_epoch_readback) {
      if (batch.any_epoch_wrapped) {
        _published_gpu_epoch = batch.current_epoch_upload;
        cmd << _current_epoch.copy_from(luisa::span{_published_gpu_epoch});
        cmd << _slot_epoch.copy_from(luisa::span{batch.slot_epoch_full_upload});
      } else if (batch.any_epoch_advanced) {
        for (std::size_t p = 0u; p < _pools.size(); ++p) {
          if (p < batch.current_epoch_upload.size() &&
              p < _published_gpu_epoch.size()) {
            _published_gpu_epoch[p] =
                std::max(_published_gpu_epoch[p], batch.current_epoch_upload[p]);
          }
        }
        cmd << _current_epoch.copy_from(luisa::span{_published_gpu_epoch});
      }
      if (batch.ep_upd_count > 0u) {
        const auto ep_upd_u32 = static_cast<luisa::uint>(batch.ep_upd_count);
        if (!_use_pinned_upload) {
          cmd << _slot_epoch_updates[batch.meta_slot]
                     .view(0u, ep_upd_u32)
                     .copy_from(luisa::span{batch.fallback_ep_updates});
        }
        cmd << _apply_page_table_updates_shader(
                   _slot_epoch, _slot_epoch_updates[batch.meta_slot],
                   ep_upd_u32)
                   .dispatch(ep_upd_u32);
      }
    }

    pub_fence =
        _async_publish_seq.fetch_add(1ull, std::memory_order_relaxed) + 1ull;
    cmd.add_callback([this, pub_fence, kept_batch]() noexcept {
      _async_publish_completed.store(pub_fence, std::memory_order_release);
      _async_publish_completed.notify_all();
    });
    stream << cmd.commit();

    for (const auto &c : batch.chunks) {
      _pools[c.pool_index].slot_async_fence[c.chunk_slot] = pub_fence;
      _pools[c.pool_index].slot_stream[c.chunk_slot] = 0ull;
    }
    _meta_slot_async_fence[batch.meta_slot] = pub_fence;
    _meta_slot_stream[batch.meta_slot] = 0ull;

    const double pub_ms = elapsed_ms_since(pub_t0);
    _stats.upload_ms += pub_ms;

    if (prof_on) {
      const std::uint64_t exit_ns = VtProfileRecorder::steady_ns();
      const double rb_sync_ms =
          (batch.callback_ns > slot_submit_end_ns)
              ? static_cast<double>(batch.callback_ns - slot_submit_end_ns) /
                    1.0e6
              : 0.0;
      const std::uint64_t bid = _profiler->record_batch(VtProfileBatchRecord{
          .source = "wavefront_demand",
          .stream_handle = slot_stream_handle,
          .queued_fault_frames = slot_fault_frames,
          .tiles_loaded = batch.tiles_loaded,
          .tiles_evicted = batch.tiles_evicted,
          .b4_loaded = batch.b4_loaded,
          .f4_loaded = batch.f4_loaded,
          .chunks_submitted = static_cast<std::uint32_t>(batch.chunks.size()),
          .pt_updates = static_cast<std::uint32_t>(batch.pt_upd_count),
          .ep_updates = static_cast<std::uint32_t>(batch.ep_upd_count),
          .has_cold_decode = batch.has_cold_decode,
          .fused_final_chunk = false,
          .isolated_pub = false,
          .async_worker = true,
          .async_slot_index = batch.slot_index,
          .async_generation = batch.generation,
          .in_flight_at_submit = slot_in_flight_at_submit,
          .cpu_enter_ns = slot_enter_ns,
          .publish_enter_ns = pub_enter_ns,
          .cpu_exit_ns = exit_ns,
          .cpu_collect_record_ms = slot_collect_record_ms,
          .cpu_readback_sync_ms = rb_sync_ms,
          .cpu_readback_flush_ms = batch.rb_flush_ms,
          .cpu_admission_rank_ms = batch.adm_ms,
          .cpu_decode_ms = batch.decode_ms,
          .cpu_pack_ms = batch.pack_ms,
          .cpu_upload_submit_ms = batch.slot_wait_ms + batch.copy_submit_ms,
          .cpu_page_table_pub_ms = pub_ms,
          .cpu_wait_async_ms = slot_wait_ms,
          .cpu_fault_to_submit_ready_ms =
              static_cast<double>(exit_ns - slot_enter_ns) / 1.0e6,
          .submit_seq_before_s1 = slot_seq_before_s1,
          .submit_seq_after_s1 = slot_seq_after_s1,
          .chunk_submit_seqs = {},
          .submit_seq_after_service = query_vk_timeline_submit_seq(),
          .snapshot = slot_snapshot});
      _profiler->record_worker_events(bid, std::move(batch.worker_events));
    }
  } else if (prof_on) {
    _profiler->record_empty_check();
  }
  _stats.collect_sync_ms += slot_collect_record_ms;

  {
    std::lock_guard lock{_vt_worker_mtx};
    const std::uint64_t next_pub_epoch = ++_publish_epoch;
    const std::uint64_t next_consumer_pin = _latest_consumer_seq + 1ull;
    for (const auto &[p, v] : batch.pinned_pool_slots) {
      if (p >= _pools.size()) {
        continue;
      }
      auto &pr = _pools[p];
      if (v < pr.slot_inflight_owner.size() &&
          pr.slot_inflight_owner[v] == static_cast<std::uint32_t>(slot_idx)) {
        pr.slot_inflight_owner[v] = PoolRuntime::kInvalidAsyncSlot;
      }
      if (v < pr.slot_pin_consumer_seq.size()) {
        pr.slot_pin_consumer_seq[v] =
            std::max(pr.slot_pin_consumer_seq[v], next_consumer_pin);
      }
      if (v < pr.slot_pin_by_aux.size()) {
        pr.slot_pin_by_aux[v] = 0u;
      }
      if (v < pr.slot_publish_epoch.size()) {
        pr.slot_publish_epoch[v] = next_pub_epoch;
      }
    }
    if (slot_idx < 32u) {
      const std::uint32_t clear_mask =
          ~(1u << static_cast<std::uint32_t>(slot_idx));
      for (const auto old_p : batch.pinned_evict_pages) {
        if (old_p < _page_evict_owner_mask.size()) {
          _page_evict_owner_mask[old_p] &= clear_mask;
        }
      }
    }
    for (const auto new_p : batch.pinned_load_pages) {
      if (new_p < _page_load_inflight_slot.size() &&
          _page_load_inflight_slot[new_p] ==
              static_cast<std::uint32_t>(slot_idx)) {
        _page_load_inflight_slot[new_p] = PoolRuntime::kInvalidAsyncSlot;
      }
    }
    auto &sl = _async_slots[slot_idx];
    if (pub_fence != 0ull) {
      sl.last_publish_fence = pub_fence;
    }
    if (mark_awaiting_advance) {
      sl.state = AsyncSlotLifecycle::PublishedAwaitingAdvance;
    } else {
      sl.state = AsyncSlotLifecycle::Free;
      sl.ticket = {};
    }
    ++_defer_retry_epoch;
    _force_oldest_progress = false;
  }
  _vt_worker_cv.notify_one();
  _vt_ready_cv.notify_all();
}

void VirtualTextureCache::publish_async_wavefront(
    const coro::WavefrontCoroAsyncPublishContext &ctx) noexcept {
  if (_async_slots.empty()) {
    return;
  }
  std::size_t slot_idx = _async_slots.size();
  AsyncSlotLifecycle st = AsyncSlotLifecycle::Free;
  double wait_ms = 0.0;
  bool wake_worker_on_progress = false;
  {
    std::unique_lock lock{_vt_worker_mtx};
    if (ctx.scheduler_snapshot.valid) {
      wake_worker_on_progress =
          update_scheduler_progress_locked(ctx.scheduler_snapshot, false);
    }
    if (ctx.async_ticket.valid()) {
      const std::size_t cand = static_cast<std::size_t>(ctx.async_ticket.slot_index);
      if (cand < _async_slots.size() &&
          _async_slots[cand].ticket == ctx.async_ticket) {
        slot_idx = cand;
      }
    } else {
      for (std::size_t s = 0u; s < _async_slots.size(); ++s) {
        if (_async_slots[s].state != AsyncSlotLifecycle::Free) {
          slot_idx = s;
          break;
        }
      }
    }
    if (slot_idx >= _async_slots.size()) {
      lock.unlock();
      if (wake_worker_on_progress) {
        _vt_worker_cv.notify_one();
      }
      return;
    }
    auto &sl = _async_slots[slot_idx];
    if (sl.state == AsyncSlotLifecycle::PublishedAwaitingAdvance) {
      sl.state = AsyncSlotLifecycle::Free;
      sl.ticket = {};
      lock.unlock();
      if (wake_worker_on_progress) {
        _vt_worker_cv.notify_one();
      }
      _vt_ready_cv.notify_all();
      return;
    }
    if (sl.state == AsyncSlotLifecycle::Free) {
      lock.unlock();
      if (wake_worker_on_progress) {
        _vt_worker_cv.notify_one();
      }
      return;
    }
    auto slot_ready_to_publish = [&]() noexcept {
      return (sl.state == AsyncSlotLifecycle::Ready &&
              !batch_has_unsatisfied_dep(sl.ready_batch)) ||
             sl.state == AsyncSlotLifecycle::Free ||
             sl.state == AsyncSlotLifecycle::PublishedAwaitingAdvance;
    };
    if (!slot_ready_to_publish()) {
      _force_oldest_progress = true;
      ++_defer_retry_epoch;
      wake_worker_on_progress = false;
      lock.unlock();
      _vt_worker_cv.notify_one();
      lock.lock();
      if (!slot_ready_to_publish()) {
        const auto wait_t0 = std::chrono::steady_clock::now();
        _vt_ready_cv.wait(lock, slot_ready_to_publish);
        const double w_ms = elapsed_ms_since(wait_t0);
        _stats.async_wait_ms += w_ms;
        _last_async_wait_ms.fetch_add(w_ms, std::memory_order_relaxed);
        if (sl.state == AsyncSlotLifecycle::Ready &&
            !batch_has_unsatisfied_dep(sl.ready_batch)) {
          ++_stats.async_wait_calls;
          sl.wait_ms += w_ms;
        }
      }
    }
    st = sl.state;
    wait_ms = sl.wait_ms;
    if (st == AsyncSlotLifecycle::PublishedAwaitingAdvance) {
      sl.state = AsyncSlotLifecycle::Free;
      sl.ticket = {};
      lock.unlock();
      if (wake_worker_on_progress) {
        _vt_worker_cv.notify_one();
      }
      _vt_ready_cv.notify_all();
      return;
    }
  }
  if (st != AsyncSlotLifecycle::Ready) {
    if (wake_worker_on_progress) {
      _vt_worker_cv.notify_one();
    }
    return;
  }
  if (wake_worker_on_progress) {
    _vt_worker_cv.notify_one();
  }
  if (wait_ms == 0.0) {
    ++_stats.async_poll_ready_hits;
  }
  publish_prepared_batch(slot_idx, ctx.stream, ctx.command_list, false);
}

void VirtualTextureCache::drain_async_if_in_flight(Stream &stream) noexcept {
  if (_async_slots.empty()) {
    return;
  }
  const std::size_t n = _async_slots.size();
  while (true) {
    std::size_t publish_slot = n;
    {
      std::unique_lock lock{_vt_worker_mtx};
      auto find_ready_unpublished = [&]() noexcept -> std::size_t {
        std::size_t best_slot = n;
        std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t s = 0u; s < n; ++s) {
          const auto &sl = _async_slots[s];
          if (sl.state != AsyncSlotLifecycle::Ready) {
            continue;
          }
          if (batch_has_unsatisfied_dep(sl.ready_batch)) {
            continue;
          }
          if (sl.submit_seq < best_seq) {
            best_seq = sl.submit_seq;
            best_slot = s;
          }
        }
        return best_slot;
      };
      auto any_unpublished = [&]() noexcept {
        for (const auto &sl : _async_slots) {
          if (is_unpublished_lifecycle(sl.state)) {
            return true;
          }
        }
        return false;
      };
      if (!any_unpublished()) {
        return;
      }
      publish_slot = find_ready_unpublished();
      if (publish_slot >= n) {
        _force_oldest_progress = true;
        ++_defer_retry_epoch;
        lock.unlock();
        _vt_worker_cv.notify_one();
        lock.lock();
        publish_slot = find_ready_unpublished();
        if (publish_slot >= n && any_unpublished()) {
          const auto wait_t0 = std::chrono::steady_clock::now();
          while (!_vt_ready_cv.wait_for(lock, std::chrono::seconds(5), [&]() noexcept {
            publish_slot = find_ready_unpublished();
            return publish_slot < n || !any_unpublished();
          })) {
            std::string dump;
            for (std::size_t s = 0u; s < n; ++s) {
              dump += fmt::format(
                  "[s={}:st={},seq={},gen={},unsat={},h2d={}] ",
                  s, static_cast<unsigned>(_async_slots[s].state),
                  _async_slots[s].submit_seq,
                  _async_slots[s].ticket.generation,
                  batch_has_unsatisfied_dep(_async_slots[s].ready_batch) ? 1u : 0u,
                  _async_slots[s].ready_batch.h2d_fence);
            }
            LUISA_WARNING(
                "drain_async_if_in_flight stalled >5s: defer_epoch={} pub_epoch={} "
                "force_oldest={} slots={}",
                _defer_retry_epoch, _publish_epoch, _force_oldest_progress, dump);
          }
          const double w_ms = elapsed_ms_since(wait_t0);
          _stats.async_wait_ms += w_ms;
          _last_async_wait_ms.fetch_add(w_ms, std::memory_order_relaxed);
          if (publish_slot < n) {
            ++_stats.async_wait_calls;
            _async_slots[publish_slot].wait_ms += w_ms;
          }
        }
      }
    }
    if (publish_slot >= n) {
      return;
    }
    CommandList local_cmd;
    publish_prepared_batch(publish_slot, stream, local_cmd, true);
    ++_stats.async_aux_drains;
  }
}

} // namespace psycles::luisa_backend::detail
