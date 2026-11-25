#include "cycles_svm_scene_image.h"
#include "cycles_texture_sampling.h"
#include "path_tracer_image_decode.h"
#include "path_tracer_internal.h"
#include "virtual_texture_cache.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/path_tracer.h>

#if defined(PSYCLES_WITH_OPENIMAGEIO)
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>
#endif

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/wavefront_auxiliary.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using namespace luisa::compute;
using namespace luisa::compute::coro;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

[[nodiscard]] std::vector<std::uint8_t>
encode_png_rgba8(std::uint32_t width, std::uint32_t height,
                 const std::vector<std::uint8_t> &rgba8) {
  require(rgba8.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA8 buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::UINT8};
  if (auto out = OIIO::ImageOutput::create("vt_ms_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_ms_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_ms_tmp_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".png");
  {
    auto out = OIIO::ImageOutput::create(temp_path.string());
    require(bool(out), "OIIO create temp PNG failed");
    require(out->open(temp_path.string(), spec), "OIIO open temp PNG failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write temp PNG failed");
    require(out->close(), "OIIO close temp PNG failed");
  }
  std::ifstream ifs{temp_path, std::ios::binary};
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>()};
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  require(!bytes.empty(), "empty PNG file readback");
  return bytes;
#else
  std::ostringstream hdr;
  hdr << "P6\n" << width << ' ' << height << "\n255\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  for (std::uint32_t i = 0u; i < width * height; ++i) {
    bytes.push_back(rgba8[i * 4u + 0u]);
    bytes.push_back(rgba8[i * 4u + 1u]);
    bytes.push_back(rgba8[i * 4u + 2u]);
  }
  return bytes;
#endif
}

[[nodiscard]] std::vector<std::uint8_t>
encode_exr_rgba32f(std::uint32_t width, std::uint32_t height,
                   const std::vector<float> &rgba32f) {
  require(rgba32f.size() == static_cast<std::size_t>(width) * height * 4u,
          "RGBA32F buffer size mismatch");
#if defined(PSYCLES_WITH_OPENIMAGEIO)
  std::vector<unsigned char> encoded_u8;
  OIIO::Filesystem::IOVecOutput mem_io{encoded_u8};
  OIIO::ImageSpec spec{static_cast<int>(width), static_cast<int>(height), 4,
                       OIIO::TypeDesc::FLOAT};
  if (auto out = OIIO::ImageOutput::create("vt_ms_test.exr", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_ms_test.exr", spec), "OIIO open EXR mem failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write EXR mem failed");
    require(out->close(), "OIIO close EXR mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_ms_tmp_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".exr");
  {
    auto out = OIIO::ImageOutput::create(temp_path.string());
    require(bool(out), "OIIO create temp EXR failed");
    require(out->open(temp_path.string(), spec), "OIIO open temp EXR failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write temp EXR failed");
    require(out->close(), "OIIO close temp EXR failed");
  }
  std::ifstream ifs{temp_path, std::ios::binary};
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>()};
  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  require(!bytes.empty(), "empty EXR file readback");
  return bytes;
#else
  std::ostringstream hdr;
  hdr << "PF\n" << width << ' ' << height << "\n-1.0\n";
  const auto header = hdr.str();
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  for (std::uint32_t row = height; row > 0u; --row) {
    const std::size_t y = row - 1u;
    for (std::uint32_t x = 0u; x < width; ++x) {
      const float *px = &rgba32f[(y * width + x) * 4u];
      const std::uint8_t *raw = reinterpret_cast<const std::uint8_t *>(px);
      bytes.insert(bytes.end(), raw, raw + 3u * sizeof(float));
    }
  }
  return bytes;
#endif
}

// =========================================================================
// Test 1: Generic WavefrontCoroScheduler multi-slot OOO + stale ticket + reuse
// =========================================================================

struct MultiSlotHandlerStats {
  std::atomic<std::uint64_t> submit_calls{0ull};
  std::atomic<std::uint64_t> publish_calls{0ull};
  std::atomic<std::uint64_t> out_of_order_publishes{0ull};
  std::atomic<std::uint64_t> stale_ticket_injections{0ull};
  std::atomic<std::uint64_t> wait_calls{0ull};
  std::atomic<std::uint32_t> peak_in_flight{0u};
  std::atomic<std::uint64_t> max_slot_reuses{0ull};
};

class MultiSlotCoordinator {
public:
  static constexpr std::size_t kMaxSlots = 3u;

  struct SlotState {
    bool active{false};
    bool ready{false};
    bool has_work{false};
    std::uint64_t generation{0ull};
    std::uint64_t submit_seq{0ull};
    std::uint64_t reuse_count{0ull};
    std::uint64_t last_generation{0ull};
    std::uint32_t delay_ms{0u};
    std::uint32_t count{0u};
    luisa::compute::Buffer<luisa::uint> staging_id;
    luisa::compute::Buffer<luisa::uint> staging_val;
    luisa::compute::Buffer<luisa::uint> upload_by_id;
    std::vector<luisa::uint> host_id;
    std::vector<luisa::uint> host_val;
    std::vector<luisa::uint> host_upload_by_id;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;
    bool stop{false};
  };

private:
  std::uint32_t _slot_count{1u};
  std::uint32_t _frame_capacity{0u};
  std::uint32_t _total_threads{0u};
  bool _async_mode{false};
  bool _allow_compaction{false};
  bool _inject_stale_once{false};
  bool _wait_for_overlap{false};
  std::shared_ptr<MultiSlotHandlerStats> _stats;
  std::array<SlotState, kMaxSlots> _slots{};
  std::mutex _coord_mtx;
  std::condition_variable _slot_active_cv;
  std::condition_variable _ready_cv;
  std::uint64_t _submit_seq{0ull};
  bool _stale_injected{false};
  bool _initialized{false};

  void slot_worker_loop(std::size_t s) noexcept {
    auto &sl = _slots[s];
    while (true) {
      std::uint32_t cnt = 0u;
      std::uint32_t delay = 0u;
      std::uint64_t sub_seq = 0ull;
      {
        std::unique_lock lock{sl.mtx};
        sl.cv.wait(lock, [&]() noexcept { return sl.stop || sl.has_work; });
        if (sl.stop && !sl.has_work) {
          return;
        }
        cnt = sl.count;
        delay = sl.delay_ms;
        sub_seq = sl.submit_seq;
        sl.has_work = false;
      }
      if (_wait_for_overlap && _slot_count >= 2u && sub_seq == 1ull) {
        std::unique_lock cl{_coord_mtx};
        _slot_active_cv.wait_for(
            cl, std::chrono::milliseconds(200), [&]() noexcept {
              std::uint32_t n_act = 0u;
              for (std::size_t k = 0u; k < _slot_count; ++k) {
                if (_slots[k].active) {
                  ++n_act;
                }
              }
              return n_act >= 2u || sl.stop;
            });
      } else if (delay > 0u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      }
      for (std::uint32_t i = 0u; i < cnt; ++i) {
        const auto fid = sl.host_id[i];
        const auto old_v = sl.host_val[i];
        if (fid < sl.host_upload_by_id.size()) {
          sl.host_upload_by_id[fid] = (old_v * 13u + 0x12345u) ^ (fid * 7u);
        }
      }
      {
        std::lock_guard lock{_coord_mtx};
        sl.ready = true;
      }
      _ready_cv.notify_all();
    }
  }

public:
  MultiSlotCoordinator(std::uint32_t slot_count,
                       std::uint32_t total_threads,
                       bool async_mode,
                       bool allow_compaction,
                       bool inject_stale_once,
                       std::shared_ptr<MultiSlotHandlerStats> stats,
                       bool wait_for_overlap = false)
      : _slot_count{std::clamp(slot_count, 1u, static_cast<std::uint32_t>(kMaxSlots))},
        _total_threads{total_threads},
        _async_mode{async_mode},
        _allow_compaction{allow_compaction},
        _inject_stale_once{inject_stale_once},
        _wait_for_overlap{wait_for_overlap},
        _stats{std::move(stats)} {}

  ~MultiSlotCoordinator() noexcept {
    {
      std::lock_guard lock{_coord_mtx};
      for (std::size_t s = 0u; s < _slot_count; ++s) {
        std::lock_guard sl_lock{_slots[s].mtx};
        _slots[s].stop = true;
      }
    }
    _slot_active_cv.notify_all();
    for (std::size_t s = 0u; s < _slot_count; ++s) {
      _slots[s].cv.notify_all();
      if (_slots[s].worker.joinable()) {
        _slots[s].worker.join();
      }
    }
  }

  void ensure_initialized(Device &device, std::uint32_t frame_capacity) noexcept {
    std::lock_guard lock{_coord_mtx};
    if (_initialized) {
      require(frame_capacity == _frame_capacity,
              "MultiSlotCoordinator frame_capacity mismatch");
      return;
    }
    _initialized = true;
    _frame_capacity = frame_capacity;
    for (std::size_t s = 0u; s < _slot_count; ++s) {
      _slots[s].staging_id =
          device.create_buffer<luisa::uint>(frame_capacity);
      _slots[s].staging_val =
          device.create_buffer<luisa::uint>(frame_capacity);
      _slots[s].upload_by_id =
          device.create_buffer<luisa::uint>(_total_threads);
      _slots[s].host_id.assign(frame_capacity, 0u);
      _slots[s].host_val.assign(frame_capacity, 0u);
      _slots[s].host_upload_by_id.assign(_total_threads, 0u);
      if (_async_mode) {
        _slots[s].worker =
            std::thread([this, s]() noexcept { slot_worker_loop(s); });
      }
    }
  }

  [[nodiscard]] bool async_mode() const noexcept { return _async_mode; }
  [[nodiscard]] std::uint32_t slot_count() const noexcept { return _slot_count; }
  [[nodiscard]] bool allow_compaction() const noexcept {
    return _allow_compaction;
  }
  [[nodiscard]] const std::shared_ptr<MultiSlotHandlerStats> &
  stats() const noexcept {
    return _stats;
  }

  SlotState &acquire_submit_slot(const WavefrontCoroAsyncBatchTicket &ticket,
                                 std::uint32_t frame_count) {
    const std::size_t slot_idx =
        ticket.valid() ? static_cast<std::size_t>(ticket.slot_index) : 0u;
    require(slot_idx < _slot_count, "submit_async slot_index out of range");
    const std::uint64_t gen = ticket.valid() ? ticket.generation : 1ull;

    std::lock_guard lock{_coord_mtx};
    auto &sl = _slots[slot_idx];
    require(!sl.active, "submit_async assigned busy slot");
    if (sl.reuse_count > 0ull) {
      require(gen > sl.last_generation,
              "Slot reused without strictly monotonic generation");
    }
    sl.active = true;
    sl.ready = false;
    sl.generation = gen;
    sl.last_generation = gen;
    ++sl.reuse_count;
    sl.submit_seq = ++_submit_seq;
    static constexpr std::array<std::uint32_t, 3u> kDelaysMs{25u, 6u, 1u};
    sl.delay_ms =
        (_slot_count >= 2u) ? kDelaysMs[(sl.submit_seq - 1ull) % 3u] : 2u;
    sl.count = frame_count;

    std::uint32_t in_flight = 0u;
    for (std::size_t i = 0u; i < _slot_count; ++i) {
      if (_slots[i].active) {
        ++in_flight;
      }
    }
    auto cur_peak = _stats->peak_in_flight.load();
    while (in_flight > cur_peak &&
           !_stats->peak_in_flight.compare_exchange_weak(cur_peak, in_flight)) {
    }
    auto cur_reuse = _stats->max_slot_reuses.load();
    while (sl.reuse_count > cur_reuse &&
           !_stats->max_slot_reuses.compare_exchange_weak(cur_reuse,
                                                          sl.reuse_count)) {
    }
    if (in_flight >= 2u) {
      _slot_active_cv.notify_all();
    }
    return sl;
  }

  void notify_submitted(std::size_t slot_idx) noexcept {
    {
      std::lock_guard lock{_slots[slot_idx].mtx};
      _slots[slot_idx].has_work = true;
    }
    _slots[slot_idx].cv.notify_one();
  }

  [[nodiscard]] bool
  poll_ready(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
    std::lock_guard lock{_coord_mtx};
    std::size_t best_slot = _slot_count;
    std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t s = 0u; s < _slot_count; ++s) {
      if (_slots[s].active && _slots[s].ready &&
          _slots[s].submit_seq < best_seq) {
        best_seq = _slots[s].submit_seq;
        best_slot = s;
      }
    }
    if (best_slot >= _slot_count) {
      return false;
    }
    if (_inject_stale_once && !_stale_injected) {
      _stale_injected = true;
      ++_stats->stale_ticket_injections;
      out_ticket = WavefrontCoroAsyncBatchTicket{
          .slot_index = static_cast<luisa::uint>(best_slot),
          .generation = _slots[best_slot].generation + 777ull};
      return true;
    }
    out_ticket = WavefrontCoroAsyncBatchTicket{
        .slot_index = static_cast<luisa::uint>(best_slot),
        .generation = _slots[best_slot].generation};
    return true;
  }

  [[nodiscard]] bool
  wait_any(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
    std::unique_lock lock{_coord_mtx};
    auto find_ready = [&]() noexcept -> std::size_t {
      std::size_t best_slot = _slot_count;
      std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t s = 0u; s < _slot_count; ++s) {
        if (_slots[s].active && _slots[s].ready &&
            _slots[s].submit_seq < best_seq) {
          best_seq = _slots[s].submit_seq;
          best_slot = s;
        }
      }
      return best_slot;
    };
    auto any_active = [&]() noexcept {
      for (std::size_t s = 0u; s < _slot_count; ++s) {
        if (_slots[s].active) {
          return true;
        }
      }
      return false;
    };
    std::size_t slot = find_ready();
    if (slot >= _slot_count) {
      if (!any_active()) {
        return false;
      }
      ++_stats->wait_calls;
      _ready_cv.wait(lock, [&]() noexcept {
        slot = find_ready();
        return slot < _slot_count || !any_active();
      });
    }
    if (slot >= _slot_count) {
      return false;
    }
    out_ticket = WavefrontCoroAsyncBatchTicket{
        .slot_index = static_cast<luisa::uint>(slot),
        .generation = _slots[slot].generation};
    return true;
  }

  SlotState &prepare_publish(const WavefrontCoroAsyncBatchTicket &ticket,
                             std::shared_ptr<std::vector<luisa::uint>> &kept_upload) {
    const std::size_t slot_idx =
        ticket.valid() ? static_cast<std::size_t>(ticket.slot_index) : 0u;
    require(slot_idx < _slot_count, "publish_async slot_index out of range");
    kept_upload = std::make_shared<std::vector<luisa::uint>>();
    std::unique_lock lock{_coord_mtx};
    auto &sl = _slots[slot_idx];
    require(sl.active && (!ticket.valid() || sl.generation == ticket.generation),
            "publish_async ticket mismatch");
    if (!sl.ready) {
      _ready_cv.wait(lock, [&]() noexcept { return sl.ready; });
    }
    for (std::size_t k = 0u; k < _slot_count; ++k) {
      if (k != slot_idx && _slots[k].active &&
          _slots[k].submit_seq < sl.submit_seq) {
        ++_stats->out_of_order_publishes;
        break;
      }
    }
    *kept_upload = sl.host_upload_by_id;
    return sl;
  }

  void finish_publish(std::size_t slot_idx) noexcept {
    {
      std::lock_guard lock{_coord_mtx};
      _slots[slot_idx].active = false;
      _slots[slot_idx].ready = false;
    }
    _ready_cv.notify_all();
  }
};

class TestMultiSlotOooHandler final
    : public WavefrontCoroSchedulerExtensionHandler {
private:
  std::shared_ptr<MultiSlotCoordinator> _coord;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::uint, luisa::uint>
      _sync_rmw;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint>
      _extract;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint>
      _inject;

public:
  TestMultiSlotOooHandler(WavefrontCoroExtensionPrepareContext &ctx,
                          const WavefrontCoroExtensionStage &stage,
                          std::shared_ptr<MultiSlotCoordinator> coord)
      : _coord{std::move(coord)} {
    _coord->ensure_initialized(ctx.device, ctx.frame_capacity);

    const auto *desc = &ctx.frame_desc;
    const auto *id_binding = &stage.binding("id");
    const auto *val_binding = &stage.binding("val");
    const auto fields = stage.dataflow->reconstruct_slots;
    const auto writes = stage.dataflow->required_def.slots;
    const auto layout = ctx.frame_layout;
    const bool frame_soa = ctx.global_memory_soa;
    const bool rt_cap = layout.has_runtime_capacity();
    luisa::vector<size_t> fields_vec{fields.begin(), fields.end()};
    luisa::vector<size_t> writes_vec{writes.begin(), writes.end()};

    Kernel1D sync_k = [desc, id_binding, val_binding, fields_vec, writes_vec,
                       layout, frame_soa,
                       rt_cap](ByteBufferVar storage, BufferUInt indices,
                               UInt capacity, UInt n) noexcept {
      const auto x = dispatch_x();
      $if(x >= n) { $return(); };
      const auto idx = indices.read(x);
      auto frame = CoroFrame::create(desc);
      if (rt_cap) {
        coro_frame_load_into(frame, storage, idx, capacity, layout, frame_soa,
                             fields_vec, false, false);
      } else {
        coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                             fields_vec, false, false);
      }
      const auto fid = id_binding->read<luisa::uint>(frame);
      const auto v = val_binding->read<luisa::uint>(frame);
      val_binding->write<luisa::uint>(
          frame, (v * 13u + 0x12345u) ^ (fid * 7u));
      if (rt_cap) {
        coro_frame_store(storage, idx, capacity, frame, layout, frame_soa,
                         writes_vec, false, false);
      } else {
        coro_frame_store(storage, idx, frame, layout, frame_soa, writes_vec,
                         false, false);
      }
    };
    _sync_rmw = ctx.device.compile(sync_k, ctx.shader_option);

    Kernel1D extract_k = [desc, id_binding, val_binding, fields_vec, layout,
                          frame_soa, rt_cap](ByteBufferVar storage,
                                             BufferUInt indices,
                                             BufferUInt id_out,
                                             BufferUInt val_out,
                                             UInt capacity,
                                             UInt n) noexcept {
      const auto x = dispatch_x();
      $if(x >= n) { $return(); };
      const auto idx = indices.read(x);
      auto frame = CoroFrame::create(desc);
      if (rt_cap) {
        coro_frame_load_into(frame, storage, idx, capacity, layout, frame_soa,
                             fields_vec, false, false);
      } else {
        coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                             fields_vec, false, false);
      }
      id_out.write(x, id_binding->read<luisa::uint>(frame));
      val_out.write(x, val_binding->read<luisa::uint>(frame));
    };
    _extract = ctx.device.compile(extract_k, ctx.shader_option);

    Kernel1D inject_k = [desc, id_binding, val_binding, fields_vec, writes_vec,
                         layout, frame_soa,
                         rt_cap](ByteBufferVar storage, BufferUInt indices,
                                 BufferUInt upload_by_id, UInt capacity,
                                 UInt n) noexcept {
      const auto x = dispatch_x();
      $if(x >= n) { $return(); };
      const auto idx = indices.read(x);
      auto frame = CoroFrame::create(desc);
      if (rt_cap) {
        coro_frame_load_into(frame, storage, idx, capacity, layout, frame_soa,
                             fields_vec, false, false);
      } else {
        coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                             fields_vec, false, false);
      }
      const auto fid = id_binding->read<luisa::uint>(frame);
      val_binding->write<luisa::uint>(frame, upload_by_id.read(fid));
      if (rt_cap) {
        coro_frame_store(storage, idx, capacity, frame, layout, frame_soa,
                         writes_vec, false, false);
      } else {
        coro_frame_store(storage, idx, frame, layout, frame_soa, writes_vec,
                         false, false);
      }
    };
    _inject = ctx.device.compile(inject_k, ctx.shader_option);
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "test.ms_rw";
  }

  [[nodiscard]] WavefrontCoroExtensionExecution
  execution() const noexcept override {
    return _coord->async_mode() ? WavefrontCoroExtensionExecution::blocking_async
                                : WavefrontCoroExtensionExecution::stage;
  }

  [[nodiscard]] luisa::string_view async_domain() const noexcept override {
    return "test.ms_rw_domain";
  }

  [[nodiscard]] luisa::uint
  max_async_in_flight_batches() const noexcept override {
    return _coord->slot_count();
  }

  [[nodiscard]] bool supports_compaction_while_async() const noexcept override {
    return _coord->allow_compaction();
  }

  [[nodiscard]] bool supports_command_list() const noexcept override {
    return true;
  }

  void dispatch(const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    ++_coord->stats()->submit_calls;
    ++_coord->stats()->publish_calls;
    CommandList local_cmd;
    CommandList *cmd = ctx.command_list ? ctx.command_list : &local_cmd;
    *cmd << _sync_rmw(ctx.frame_buffer, ctx.frame_indices,
                      ctx.frame_capacity, ctx.frame_count)
                .dispatch(ctx.frame_count);
    if (ctx.command_list == nullptr) {
      ctx.stream << local_cmd.commit();
    }
  }

  [[nodiscard]] bool submit_async(
      const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    if (ctx.frame_count == 0u || !_coord->async_mode()) {
      return false;
    }
    auto &sl = _coord->acquire_submit_slot(ctx.async_ticket, ctx.frame_count);
    ++_coord->stats()->submit_calls;
    const std::size_t slot_idx =
        ctx.async_ticket.valid()
            ? static_cast<std::size_t>(ctx.async_ticket.slot_index)
            : 0u;

    CommandList local_cmd;
    CommandList *cmd = ctx.command_list ? ctx.command_list : &local_cmd;
    *cmd << _extract(ctx.frame_buffer, ctx.frame_indices, sl.staging_id,
                     sl.staging_val, ctx.frame_capacity, ctx.frame_count)
                .dispatch(ctx.frame_count)
         << sl.staging_id.view(0u, ctx.frame_count)
                .copy_to(luisa::span{sl.host_id}.subspan(0u, ctx.frame_count))
         << sl.staging_val.view(0u, ctx.frame_count)
                .copy_to(luisa::span{sl.host_val}.subspan(0u, ctx.frame_count));
    cmd->add_callback([coord = _coord, slot_idx]() noexcept {
      coord->notify_submitted(slot_idx);
    });
    ctx.stream << cmd->commit();
    return true;
  }

  [[nodiscard]] bool
  poll_async_ready(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override {
    return _coord->poll_ready(out_ticket);
  }

  [[nodiscard]] bool
  wait_async_any(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override {
    return _coord->wait_any(out_ticket);
  }

  void publish_async(const WavefrontCoroAsyncPublishContext &ctx) noexcept override {
    const std::size_t slot_idx =
        ctx.async_ticket.valid()
            ? static_cast<std::size_t>(ctx.async_ticket.slot_index)
            : 0u;
    std::shared_ptr<std::vector<luisa::uint>> kept_upload;
    auto &sl = _coord->prepare_publish(ctx.async_ticket, kept_upload);

    ctx.command_list
        << sl.upload_by_id.copy_from(luisa::span{*kept_upload})
        << _inject(ctx.frame_buffer, ctx.frame_indices, sl.upload_by_id,
                   ctx.frame_capacity, ctx.frame_count)
               .dispatch(ctx.frame_count);
    ctx.command_list.add_callback([kept_upload]() noexcept {});
    ++_coord->stats()->publish_calls;
    _coord->finish_publish(slot_idx);
  }
};

void test_coro_scheduler_multislot_ooo_and_compaction(Device &device,
                                                      Stream &stream) {
  std::cout << "[MultiSlot Test 1] WavefrontCoroScheduler 1/2/3 slots OOO + stale ticket + compaction...\n";
  constexpr std::uint32_t kThreads = 4096u;
  constexpr std::uint32_t kPoolCap = 512u;

  auto run_branching_coro = [&](bool async_mode,
                                std::uint32_t slots,
                                bool compact,
                                bool allow_compaction_while_async,
                                bool inject_stale) {
    Coroutine<void(Buffer<luisa::uint>)> coro{[async_mode](BufferUInt output) {
      const auto tid = dispatch_x();
      const UInt lane = tid % 3u;
      UInt id = tid;
      UInt v = def(tid * 17u + 3u);
      $if(lane == 0u) {
        $suspend("branch_a", coro_stage("test.ms_rw")
                                 .read_write("id", id)
                                 .read_write("val", v)
                                 .blocking_async(async_mode)
                                 .async_domain("test.ms_rw_domain"));
        output.write(tid, v ^ 0x5a5a5a5au);
      }
      $elif(lane == 1u) {
        $suspend("branch_b", coro_stage("test.ms_rw")
                                 .read_write("id", id)
                                 .read_write("val", v)
                                 .blocking_async(async_mode)
                                 .async_domain("test.ms_rw_domain"));
        output.write(tid, v ^ 0x13572468u);
      }
      $else {
        $suspend("branch_c", coro_stage("test.ms_rw")
                                 .read_write("id", id)
                                 .read_write("val", v)
                                 .blocking_async(async_mode)
                                 .async_domain("test.ms_rw_domain"));
        $suspend("branch_c_post");
        output.write(tid, (v * 3u + 7u) ^ 0x24681357u);
      };
    }};

    WavefrontCoroSchedulerConfig cfg;
    cfg.thread_count = kPoolCap;
    cfg.execution_block_size = 64u;
    cfg.global_memory_soa = true;
    cfg.frame_buffer_compaction = compact;
    cfg.largest_continuation_first = true;
    cfg.incremental_continuation_counts = true;
    cfg.fused_continuation_counts = true;
    cfg.max_async_in_flight_batches = slots;

    auto stats = std::make_shared<MultiSlotHandlerStats>();
    auto coord = std::make_shared<MultiSlotCoordinator>(
        slots, kThreads, async_mode, allow_compaction_while_async, inject_stale,
        stats);
    WavefrontCoroScheduler<Buffer<luisa::uint>> scheduler{device, coro, cfg};
    scheduler.register_extension_handler(
        stream, [coord](WavefrontCoroExtensionPrepareContext &ctx,
                        const WavefrontCoroExtensionStage &stage)
                    -> luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
          if (!stage.extension ||
              stage.extension->schema() != "test.ms_rw") {
            return nullptr;
          }
          return luisa::make_unique<TestMultiSlotOooHandler>(ctx, stage, coord);
        });

    auto out_buf = device.create_buffer<luisa::uint>(kThreads);
    std::vector<luisa::uint> host_out(kThreads, 0u);
    stream << out_buf.copy_from(luisa::span{host_out}) << commit();
    stream << scheduler(out_buf).dispatch(kThreads);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();
    return std::pair{host_out, stats};
  };

  const auto [sync_ref, sync_stats] =
      run_branching_coro(false, 1u, true, false, false);
  for (std::uint32_t tid = 0u; tid < kThreads; ++tid) {
    const luisa::uint lane = tid % 3u;
    const luisa::uint v0 = tid * 17u + 3u;
    const luisa::uint v1 = (v0 * 13u + 0x12345u) ^ (tid * 7u);
    const luisa::uint expected =
        (lane == 0u) ? (v1 ^ 0x5a5a5a5au)
                     : ((lane == 1u) ? (v1 ^ 0x13572468u)
                                     : ((v1 * 3u + 7u) ^ 0x24681357u));
    require(sync_ref[tid] == expected,
            "MultiSlot Test 1 sync reference mismatch at tid=" +
                std::to_string(tid));
  }

  for (const bool compact : {true, false}) {
    for (const std::uint32_t slots : {1u, 2u, 3u}) {
      const bool inject_stale = (slots >= 2u);
      const auto [out, st] =
          run_branching_coro(true, slots, compact, false, inject_stale);
      require(out == sync_ref,
              "MultiSlot Test 1 branching parity failed for slots=" +
                  std::to_string(slots));
      require(st->submit_calls.load() == st->publish_calls.load(),
              "MultiSlot Test 1 submit != publish count");
      require(st->peak_in_flight.load() == slots,
              "MultiSlot Test 1 expected peak_in_flight == " +
                  std::to_string(slots));
      require(st->max_slot_reuses.load() >= 2ull,
              "MultiSlot Test 1 expected slot reuse >= 2 generations");
      if (slots >= 2u) {
        require(st->out_of_order_publishes.load() > 0ull,
                "MultiSlot Test 1 expected out_of_order_publishes > 0 for slots >= 2");
        require(st->stale_ticket_injections.load() > 0ull,
                "MultiSlot Test 1 expected stale_ticket_injections > 0");
      }
      std::cout << "  compact=" << (compact ? 1 : 0) << " slots=" << slots
                << " peak=" << st->peak_in_flight.load()
                << " reuse=" << st->max_slot_reuses.load()
                << " ooo=" << st->out_of_order_publishes.load()
                << " stale=" << st->stale_ticket_injections.load()
                << " (bit-exact OK)\n";
    }
  }

  // Test compaction while async batches are parked (supports_compaction_while_async=true).
  {
    Coroutine<void(Buffer<luisa::uint>)> compact_coro{[](BufferUInt output) {
      const auto tid = dispatch_x();
      UInt id = tid;
      UInt v = def(tid * 17u + 3u);
      $if((tid & 7u) == 0u) {
        $suspend("parked_rw", coro_stage("test.ms_rw")
                                  .read_write("id", id)
                                  .read_write("val", v)
                                  .blocking_async(true)
                                  .async_domain("test.ms_rw_domain"));
        output.write(tid, v ^ 0x55aa55aau);
      }
      $else {
        UInt acc = def(tid * 29u + 7u);
        $suspend("work_0");
        acc = acc * 3u + 1u;
        $suspend("work_1");
        output.write(tid, acc ^ 0x33cc33ccu);
      };
    }};

    WavefrontCoroSchedulerConfig cfg;
    cfg.thread_count = kPoolCap;
    cfg.refill_threshold = 448u;
    cfg.execution_block_size = 64u;
    cfg.global_memory_soa = true;
    cfg.frame_buffer_compaction = true;
    cfg.largest_continuation_first = true;
    cfg.incremental_continuation_counts = true;
    cfg.fused_continuation_counts = true;
    cfg.max_async_in_flight_batches = 3u;
    cfg.report_stats = true;

    auto stats = std::make_shared<MultiSlotHandlerStats>();
    auto coord = std::make_shared<MultiSlotCoordinator>(
        3u, kThreads, true, true, true, stats, true);
    WavefrontCoroScheduler<Buffer<luisa::uint>> scheduler{device, compact_coro, cfg};
    scheduler.register_extension_handler(
        stream, [coord](WavefrontCoroExtensionPrepareContext &ctx,
                        const WavefrontCoroExtensionStage &stage)
                    -> luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
          if (!stage.extension ||
              stage.extension->schema() != "test.ms_rw") {
            return nullptr;
          }
          return luisa::make_unique<TestMultiSlotOooHandler>(ctx, stage, coord);
        });

    auto out_buf = device.create_buffer<luisa::uint>(kThreads);
    std::vector<luisa::uint> host_out(kThreads, 0u);
    stream << out_buf.copy_from(luisa::span{host_out}) << commit();
    stream << scheduler(out_buf).dispatch(kThreads);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();
    for (std::uint32_t tid = 0u; tid < kThreads; ++tid) {
      luisa::uint expected = 0u;
      if ((tid & 7u) == 0u) {
        const luisa::uint v0 = tid * 17u + 3u;
        const luisa::uint v1 = (v0 * 13u + 0x12345u) ^ (tid * 7u);
        expected = v1 ^ 0x55aa55aau;
      } else {
        const luisa::uint acc = (tid * 29u + 7u) * 3u + 1u;
        expected = acc ^ 0x33cc33ccu;
      }
      require(host_out[tid] == expected,
              "Compaction-while-parked parity failed at tid=" +
                  std::to_string(tid));
    }
    require(scheduler.last_dispatch_stats().compact_scan_count > 0ull,
            "Expected compaction scans while parked frames were relocated");
    require(stats->peak_in_flight.load() >= 2u,
            "Expected multi-slot concurrency during parked compaction");
    std::cout << "  compact_while_parked=1 peak=" << stats->peak_in_flight.load()
              << " ooo=" << stats->out_of_order_publishes.load()
              << " compact_scan=" << scheduler.last_dispatch_stats().compact_scan_count
              << " (bit-exact OK)\n";
  }
}

// =========================================================================
// Test 2: Real VirtualTextureCache multi-slot parity + stress (comp/pp/cq/256KiB)
// =========================================================================

class TestVtAuxWork final
    : public WavefrontCoroAuxiliaryWork<luisa::compute::Buffer<luisa::float4>,
                                        luisa::compute::Buffer<luisa::float4>,
                                        luisa::compute::Buffer<luisa::uint>> {
private:
  std::uint32_t _capacity;
  std::shared_ptr<VirtualTextureCache> _vt;
  luisa::compute::BindlessArray _direct_heap;
  luisa::compute::Buffer<luisa::uint2> _item_ids;
  luisa::compute::Buffer<luisa::float2> _item_uv;
  luisa::compute::Buffer<luisa::uint> _active;
  luisa::compute::Buffer<luisa::uint> _free_slots;
  luisa::compute::Buffer<luisa::uint> _free_count;
  luisa::compute::Buffer<luisa::uint> _counts;
  std::array<luisa::uint, 2u> _host_counts{0u, 0u};
  std::array<WavefrontCoroAuxiliaryProducer, 1u> _producers{
      {{.continuation = "after_vt_c", .max_emitted_per_invocation = 1u}}};
  luisa::compute::Shader1D<luisa::uint> _init_shader;
  luisa::compute::Shader1D<luisa::uint> _clear_flag_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::float4>, luisa::uint>
      _consume_shader;

public:
  TestVtAuxWork(Device &device, std::uint32_t capacity,
                std::shared_ptr<VirtualTextureCache> vt,
                const BindlessArray &direct_heap)
      : _capacity{capacity},
        _vt{std::move(vt)},
        _direct_heap{direct_heap},
        _item_ids{device.create_buffer<luisa::uint2>(capacity)},
        _item_uv{device.create_buffer<luisa::float2>(capacity)},
        _active{device.create_buffer<luisa::uint>(capacity)},
        _free_slots{device.create_buffer<luisa::uint>(capacity)},
        _free_count{device.create_buffer<luisa::uint>(1u)},
        _counts{device.create_buffer<luisa::uint>(2u)} {
    Kernel1D init_k = [this](UInt cap) noexcept {
      const auto x = dispatch_x();
      $if(x < cap) {
        _active->write(x, 0u);
        _free_slots->write(x, x);
      };
    };
    _init_shader = device.compile(init_k);

    Kernel1D clr_k = [this](UInt idx) noexcept { _counts->write(idx, 0u); };
    _clear_flag_shader = device.compile(clr_k);

    Kernel1D consume_k = [this](BufferFloat4 aux_out, UInt cap) noexcept {
      const auto slot = dispatch_x();
      $if(slot >= cap) { $return(); };
      $if(_active->read(slot) != 0u) {
        const auto ids = _item_ids->read(slot);
        const auto uv = _item_uv->read(slot);
        UInt miss = vt_invalid_page;
        const auto col =
            _vt->sample_2d_dynamic(Int(ids.y), uv, _direct_heap, &miss, false);
        $if(miss != vt_invalid_page) {
          _vt->record_aux_miss(miss);
          _counts->atomic(1u).fetch_add(1u);
        }
        $else {
          aux_out.write(ids.x, col);
          _active->write(slot, 0u);
          _counts->atomic(0u).fetch_sub(1u);
          const auto fi = _free_count->atomic(0u).fetch_add(1u);
          $if(fi < cap) { _free_slots->write(fi, slot); };
        };
      };
    };
    _consume_shader = device.compile(consume_k);
  }

  void emit(Expr<luisa::uint> tid, Expr<luisa::float2> uv,
            Expr<luisa::uint> binding_id) const noexcept {
    const auto fc = _free_count->atomic(0u).fetch_sub(1u);
    $if((fc != 0u) & (fc <= _capacity)) {
      const auto slot = _free_slots->read(fc - 1u);
      _item_ids->write(slot, make_uint2(tid, binding_id));
      _item_uv->write(slot, uv);
      _active->write(slot, 1u);
      _counts->atomic(0u).fetch_add(1u);
    };
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "test_vt_ms_aux";
  }
  [[nodiscard]] luisa::uint capacity() const noexcept override {
    return _capacity;
  }
  [[nodiscard]] luisa::span<const WavefrontCoroAuxiliaryProducer>
  producers() const noexcept override {
    return _producers;
  }
  void reset(Stream &stream) noexcept override {
    _host_counts = {0u, 0u};
    const std::array<luisa::uint, 2u> zeros{0u, 0u};
    stream << _init_shader(_capacity).dispatch(_capacity)
           << _free_count.copy_from(luisa::span{&_capacity, 1u})
           << _counts.copy_from(luisa::span{zeros});
  }
  void enqueue_count_readback(Stream &stream) noexcept override {
    stream << _counts.copy_to(luisa::span{_host_counts});
  }
  [[nodiscard]] luisa::uint host_count() const noexcept override {
    return _host_counts[0u];
  }
  [[nodiscard]] luisa::uint host_available_slots() const noexcept override {
    return _capacity - _host_counts[0u];
  }
  [[nodiscard]] bool stage_blocks_async_domain(
      luisa::uint /*stage*/, luisa::string_view domain) const noexcept override {
    return _vt != nullptr && _host_counts[1u] != 0u &&
           domain == vt_extension_schema;
  }
  void encode_dispatch_stage(luisa::uint /*stage*/, Stream &stream,
                             CommandList &commands,
                             BufferView<luisa::float4>,
                             BufferView<luisa::float4> aux_out,
                             BufferView<luisa::uint>) noexcept override {
    if (_vt && _host_counts[1u] != 0u) {
      if (!commands.empty()) {
        stream << commands.commit();
      }
      _vt->service_aux_misses(stream);
      _host_counts[1u] = 0u;
      commands << _clear_flag_shader(1u).dispatch(1u);
    }
    commands << _consume_shader(aux_out, _capacity).dispatch(_capacity);
  }
  void dispatch(Stream &stream, BufferView<luisa::float4> main_out,
                BufferView<luisa::float4> aux_out,
                BufferView<luisa::uint> indep_out) noexcept override {
    CommandList cmd;
    encode_dispatch_stage(0u, stream, cmd, main_out, aux_out, indep_out);
    stream << cmd.commit();
  }
};

void test_vt_cache_multislot_parity_and_stress(Device &device, Stream &stream) {
  std::cout << "[MultiSlot Test 2] VirtualTextureCache 1/2/3 slots parity + comp + prepack + copyQ + 256KiB stress...\n";

  constexpr std::uint32_t b4_w = 313u;
  constexpr std::uint32_t b4_h = 197u;
  constexpr std::uint32_t f4_w = 251u;
  constexpr std::uint32_t f4_h = 193u;

  std::vector<std::uint8_t> b4_px(static_cast<std::size_t>(b4_w) * b4_h * 4u);
  for (std::uint32_t y = 0u; y < b4_h; ++y) {
    for (std::uint32_t x = 0u; x < b4_w; ++x) {
      const std::size_t i = (static_cast<std::size_t>(y) * b4_w + x) * 4u;
      b4_px[i + 0u] = static_cast<std::uint8_t>((x * 19u + y * 11u + 5u) & 0xffu);
      b4_px[i + 1u] = static_cast<std::uint8_t>((x * 7u + y * 37u + 41u) & 0xffu);
      b4_px[i + 2u] = static_cast<std::uint8_t>((x * 43u + y * 13u + 89u) & 0xffu);
      b4_px[i + 3u] = static_cast<std::uint8_t>(255u);
    }
  }
  std::vector<float> f4_px(static_cast<std::size_t>(f4_w) * f4_h * 4u);
  for (std::uint32_t y = 0u; y < f4_h; ++y) {
    for (std::uint32_t x = 0u; x < f4_w; ++x) {
      const float u = static_cast<float>(x) / static_cast<float>(f4_w - 1u);
      const float v = static_cast<float>(y) / static_cast<float>(f4_h - 1u);
      const std::size_t i = (static_cast<std::size_t>(y) * f4_w + x) * 4u;
      f4_px[i + 0u] = -0.15f + 4.5f * u + 0.25f * v;
      f4_px[i + 1u] = 3.25f * (1.0f - v) + 0.1f * std::sin(u * 11.0f);
      f4_px[i + 2u] = 0.2f + 2.5f * u * v;
      f4_px[i + 3u] = 0.5f + 0.5f * std::cos(v * 7.0f);
    }
  }

  const auto b4_png = encode_png_rgba8(b4_w, b4_h, b4_px);
  const auto f4_exr = encode_exr_rgba32f(f4_w, f4_h, f4_px);
  const auto b4_probe = probe_scene_image_info(b4_png, "ms_b4.png");
  const auto f4_probe = probe_scene_image_info(f4_exr, "ms_f4.exr");
  require(b4_probe.has_value() && f4_probe.has_value(),
          "MultiSlot Test 2 failed to probe synthetic images");

  const std::vector<VtProbedSceneImage> probed_images{
      VtProbedSceneImage{.id = ImageId{1u},
                         .name = "ms_b4.png",
                         .encoded_data = b4_png,
                         .alpha_type = ImageAlphaType::ignore,
                         .color_space = ImageColorSpace::srgb,
                         .probed = *b4_probe},
      VtProbedSceneImage{.id = ImageId{2u},
                         .name = "ms_f4.exr",
                         .encoded_data = f4_exr,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *f4_probe}};

  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  constexpr std::array exts{ImageExtension::repeat, ImageExtension::clip,
                            ImageExtension::extend, ImageExtension::mirror};
  constexpr std::array interps{ImageInterpolation::closest,
                               ImageInterpolation::linear,
                               ImageInterpolation::cubic};

  std::vector<ImageBinding> handles;
  std::vector<CyclesSvmImageBindingGpu> gpu_bindings;
  for (const std::uint32_t res_id : {1u, 2u}) {
    for (const auto ext : exts) {
      for (const auto interp : interps) {
        const auto idx = static_cast<std::uint32_t>(handles.size());
        handles.push_back(ImageBinding{
            .resource_id = res_id, .interpolation = interp, .extension = ext});
        gpu_bindings.push_back(
            make_cycles_svm_image_binding(idx, interp, ext));
      }
    }
  }
  const std::uint32_t num_bindings = static_cast<std::uint32_t>(handles.size());

  BindlessArray dummy_direct_heap = device.create_bindless_array(num_bindings);
  auto dummy_b4 = device.create_image<float>(PixelStorage::BYTE4, 4u, 4u);
  auto dummy_f4 = device.create_image<float>(PixelStorage::FLOAT4, 4u, 4u);
  std::vector<std::uint8_t> d_b4(64u, 128u);
  std::vector<float> d_f4(64u, 0.5f);
  for (std::uint32_t i = 0u; i < num_bindings; ++i) {
    dummy_direct_heap.emplace_on_update(
        i, i < 12u ? dummy_b4 : dummy_f4,
        cycles_svm_texture_sampler(handles[i].interpolation,
                                   handles[i].extension));
  }
  stream << dummy_b4.copy_from(d_b4.data())
         << dummy_f4.copy_from(d_f4.data())
         << dummy_direct_heap.update() << synchronize();

  constexpr std::uint32_t kThreads = 2048u;
  constexpr std::uint32_t kPoolCap = 1024u;

  struct VtRunResult {
    VirtualTextureCacheStats stats{};
    std::vector<luisa::float4> main_out;
    std::vector<luisa::float4> aux_out;
    std::vector<luisa::uint> indep_out;
  };

  auto run_vt_case = [&](std::uint64_t budget_bytes,
                         LuisaVirtualTextureCompressionMode comp_mode,
                         LuisaVirtualTexturePrepackMode prepack_mode,
                         LuisaVirtualTextureCopyQueueMode cq_mode,
                         std::uint32_t slots) -> VtRunResult {
    LuisaVirtualTextureCacheOptions opts{
        .enabled = true,
        .gpu_cache_budget_bytes = budget_bytes,
        .physical_tile_size = 64u,
        .prepack_mode = prepack_mode,
        .host_prepacked_cache_budget_bytes = 64ull * 1024ull * 1024ull,
        .prepack_warm_on_init =
            (prepack_mode != LuisaVirtualTexturePrepackMode::off),
        .enable_async_worker = true,
        .async_batch_slots = slots,
        .copy_queue_mode = cq_mode,
        .compression_mode = comp_mode};

    VtRunResult result;
    result.main_out.resize(kThreads);
    result.aux_out.resize(kThreads);
    result.indep_out.resize(kThreads);
    {
      auto vt = VirtualTextureCache::create(device, stream, opts, probed_images,
                                            handles, gpu_bindings);

      auto vt_wait_site = [&](const char *site_name,
                              Expr<luisa::uint> page_id) noexcept {
        UInt missed_page = page_id;
        $loop {
          $suspend(site_name,
                   coro_stage(luisa::string{vt_extension_schema})
                       .read(luisa::string{vt_extension_binding}, missed_page)
                       .blocking_async()
                       .async_domain(luisa::string{vt_extension_schema}));
          $if(vt->is_resident(missed_page)) { $break; };
        };
      };

      auto sample_bucket = [&](const char *site_name,
                               Expr<luisa::uint> b, Expr<luisa::float2> uv) {
        Float4 col = make_float4(0.0f);
        $loop {
          UInt miss = vt_invalid_page;
          col = vt->sample_2d_dynamic(Int(b), uv, dummy_direct_heap, &miss);
          $if(miss == vt_invalid_page) { $break; };
          vt_wait_site(site_name, miss);
        };
        return col;
      };

      auto aux_work = std::make_shared<TestVtAuxWork>(
          device, kThreads, vt, dummy_direct_heap);

      Coroutine<void(Buffer<luisa::float4>, Buffer<luisa::float4>,
                     Buffer<luisa::uint>)>
          coro{[&](BufferFloat4 main_out, BufferFloat4 aux_out,
                   BufferUInt indep_out) {
            (void)aux_out;
            const auto tid = dispatch_x();
            const UInt bucket = tid % 3u;
            const UInt b0 = (tid * 7u + (tid / 32u)) % num_bindings;
            const UInt gx = (tid * 13u + 3u) % 256u;
            const UInt gy = (tid * 19u + 7u) % 256u;
            const Float2 uv0 = make_float2(
                -0.25f + 1.50f * (cast<float>(gx) / 255.0f),
                -0.20f + 1.45f * (cast<float>(gy) / 255.0f));
            $if(bucket == 0u) {
              const auto c = sample_bucket("vt_wait_a", b0, uv0);
              main_out.write(tid, c);
              indep_out.write(tid, tid ^ 0xa5a5a5a5u);
            }
            $elif(bucket == 1u) {
              const auto c = sample_bucket("vt_wait_b", b0, uv0);
              main_out.write(tid, c);
              indep_out.write(tid, tid ^ 0x5a5a5a5au);
            }
            $else {
              UInt acc = def(tid * 11u + 5u);
              $suspend("indep_0");
              acc = acc * 3u + 1u;
              $suspend("indep_1");
              indep_out.write(tid, acc ^ (tid * 7u));
              const auto c = sample_bucket("vt_wait_c", b0, uv0);
              main_out.write(tid, c);
              $suspend("after_vt_c");
              $if((tid & 7u) == 0u) {
                const UInt aux_b = (b0 + 13u) % num_bindings;
                aux_work->emit(tid, uv0, aux_b);
              };
            };
          }};

      WavefrontCoroSchedulerConfig cfg;
      cfg.thread_count = kPoolCap;
      cfg.execution_block_size = 64u;
      cfg.global_memory_soa = true;
      cfg.frame_buffer_compaction = true;
      cfg.largest_continuation_first = true;
      cfg.incremental_continuation_counts = true;
      cfg.fused_continuation_counts = true;
      cfg.max_async_in_flight_batches = slots;

      WavefrontCoroScheduler<Buffer<luisa::float4>, Buffer<luisa::float4>,
                             Buffer<luisa::uint>>
          scheduler{device, coro, cfg};
      scheduler.register_extension_handler(
          stream, vt->make_wavefront_extension_handler());
      scheduler.register_auxiliary_work(aux_work);

      auto main_buf = device.create_buffer<luisa::float4>(kThreads);
      auto aux_buf = device.create_buffer<luisa::float4>(kThreads);
      auto indep_buf = device.create_buffer<luisa::uint>(kThreads);
      std::vector<luisa::float4> zero_f4(kThreads, luisa::make_float4(0.0f));
      std::vector<luisa::uint> zero_u32(kThreads, 0u);
      stream << main_buf.copy_from(luisa::span{zero_f4})
             << aux_buf.copy_from(luisa::span{zero_f4})
             << indep_buf.copy_from(luisa::span{zero_u32}) << commit();
      stream << scheduler(main_buf, aux_buf, indep_buf).dispatch(kThreads);
      stream << main_buf.copy_to(luisa::span{result.main_out})
             << aux_buf.copy_to(luisa::span{result.aux_out})
             << indep_buf.copy_to(luisa::span{result.indep_out})
             << synchronize();

      result.stats = vt->stats();
    }
    return result;
  };

  auto check_f4_parity = [&](const std::string &tag,
                             const std::string &buf_name,
                             const std::vector<luisa::float4> &a,
                             const std::vector<luisa::float4> &b,
                             std::uint32_t binding_offset) {
    std::size_t closest_diff = 0u;
    std::size_t filtered_fail = 0u;
    double max_abs = 0.0;
    for (std::size_t i = 0u; i < a.size(); ++i) {
      const auto tid = static_cast<std::uint32_t>(i);
      const auto bind_id =
          ((tid * 7u + (tid / 32u)) % num_bindings + binding_offset) %
          num_bindings;
      const bool is_closest = (bind_id % 3u) == 0u;
      if (is_closest) {
        if (std::memcmp(&a[i], &b[i], sizeof(luisa::float4)) != 0) {
          ++closest_diff;
        }
      } else {
        for (int c = 0; c < 4; ++c) {
          const double av = static_cast<double>((&a[i].x)[c]);
          const double bv = static_cast<double>((&b[i].x)[c]);
          const double diff = std::abs(av - bv);
          max_abs = std::max(max_abs, diff);
          const double tol =
              1e-4 * std::max(1.0, std::max(std::abs(av), std::abs(bv)));
          if (diff > tol) {
            ++filtered_fail;
            break;
          }
        }
      }
    }
    require(closest_diff == 0u,
            tag + " " + buf_name + " closest 0-ULP memcmp failed vs slots=1");
    require(filtered_fail == 0u,
            tag + " " + buf_name + " filtered parity > 1e-4 vs slots=1");
    return max_abs;
  };

  const bool has_batch_copy =
      (device.impl()->query("vk_batch_copy_cmd") == "1");
  struct ConfigCase {
    const char *label;
    std::uint64_t budget;
    LuisaVirtualTextureCompressionMode comp;
    LuisaVirtualTexturePrepackMode pp;
    LuisaVirtualTextureCopyQueueMode cq;
  };
  std::vector<ConfigCase> cases{
      {.label = "1MiB_comp=off_pp=off_cq=off",
       .budget = 1024ull * 1024ull,
       .comp = LuisaVirtualTextureCompressionMode::off,
       .pp = LuisaVirtualTexturePrepackMode::off,
       .cq = LuisaVirtualTextureCopyQueueMode::off},
      {.label = "256KiB_comp=off_pp=off_cq=off",
       .budget = 256ull * 1024ull,
       .comp = LuisaVirtualTextureCompressionMode::off,
       .pp = LuisaVirtualTexturePrepackMode::off,
       .cq = LuisaVirtualTextureCopyQueueMode::off},
  };
  if (has_batch_copy) {
    cases.push_back(ConfigCase{
        .label = "1MiB_comp=auto_hq_pp=mem_cq=staged",
        .budget = 1024ull * 1024ull,
        .comp = LuisaVirtualTextureCompressionMode::auto_hq,
        .pp = LuisaVirtualTexturePrepackMode::memory,
        .cq = LuisaVirtualTextureCopyQueueMode::staged});
    cases.push_back(ConfigCase{
        .label = "256KiB_comp=auto_hq_pp=off_cq=off",
        .budget = 256ull * 1024ull,
        .comp = LuisaVirtualTextureCompressionMode::auto_hq,
        .pp = LuisaVirtualTexturePrepackMode::off,
        .cq = LuisaVirtualTextureCopyQueueMode::off});
  }

  for (const auto &c : cases) {
    const auto base = run_vt_case(c.budget, c.comp, c.pp, c.cq, 1u);
    require(base.stats.async_batches > 0ull,
            std::string{c.label} + " slots=1 expected async_batches > 0");
    require(base.stats.tiles_evicted > 0ull,
            std::string{c.label} + " slots=1 expected tiles_evicted > 0");
    for (const std::uint32_t slots : {2u, 3u}) {
      const auto res = run_vt_case(c.budget, c.comp, c.pp, c.cq, slots);
      const double main_max = check_f4_parity(
          c.label, "main_out", base.main_out, res.main_out, 0u);
      const double aux_max = check_f4_parity(
          c.label, "aux_out", base.aux_out, res.aux_out, 13u);
      require(base.indep_out == res.indep_out,
              std::string{c.label} + " indep_out mismatch vs slots=1");
      require(res.stats.async_batches > 0ull,
              std::string{c.label} + " expected async_batches > 0");
      require(res.stats.async_batch_slots == slots,
              std::string{c.label} + " stats async_batch_slots mismatch");
      require(res.stats.async_peak_in_flight >= 2u,
              std::string{c.label} + " expected async_peak_in_flight >= 2");
      const std::uint64_t max_allowed_loads = std::max<std::uint64_t>(
          base.stats.tiles_loaded * 2u, base.stats.tiles_loaded + 128u);
      require(res.stats.tiles_loaded <= max_allowed_loads,
              std::string{c.label} + " slots=" + std::to_string(slots) +
                  " tiles_loaded inflation too high: " +
                  std::to_string(res.stats.tiles_loaded) + " vs base " +
                  std::to_string(base.stats.tiles_loaded));
      const double load_ratio =
          base.stats.tiles_loaded > 0ull
              ? static_cast<double>(res.stats.tiles_loaded) /
                    static_cast<double>(base.stats.tiles_loaded)
              : 1.0;
      std::cout << "  [" << c.label << " slots=" << slots << "]"
                << " batches=" << res.stats.async_batches
                << " loaded=" << res.stats.tiles_loaded
                << " base_loaded=" << base.stats.tiles_loaded
                << " ratio=" << load_ratio
                << " evicted=" << res.stats.tiles_evicted
                << " peak=" << res.stats.async_peak_in_flight
                << " ooo=" << res.stats.async_out_of_order_publishes
                << " deferred=" << res.stats.async_deferred_admissions
                << " aux_drains=" << res.stats.async_aux_drains
                << " main_max=" << main_max << " aux_max=" << aux_max
                << " (0-ULP OK)\n";
    }
  }
}

class FilteredAdvanceCoordinator {
public:
  struct Slot {
    bool active{false};
    bool ready{false};
    std::uint64_t generation{0ull};
    std::uint64_t submit_seq{0ull};
  };

private:
  bool _async_mode{false};
  std::uint32_t _slot_count{1u};
  std::uint32_t _total_threads{0u};
  bool _always_unready{false};
  std::mutex _mtx;
  std::condition_variable _cv;
  std::vector<Slot> _slots;
  std::uint64_t _next_submit_seq{0ull};
  std::uint32_t _round{0u};
  luisa::compute::Buffer<luisa::uint> _ready_table;
  bool _initialized{false};

public:
  FilteredAdvanceCoordinator(bool async_mode,
                             std::uint32_t slot_count,
                             std::uint32_t total_threads,
                             bool always_unready)
      : _async_mode{async_mode},
        _slot_count{std::max<std::uint32_t>(slot_count, 1u)},
        _total_threads{total_threads},
        _always_unready{always_unready},
        _slots(_slot_count) {}

  void ensure_initialized(Device &device, Stream &stream) {
    std::lock_guard lock{_mtx};
    if (_initialized) {
      return;
    }
    _ready_table = device.create_buffer<luisa::uint>(_total_threads);
    std::vector<luisa::uint> zeros(_total_threads, 0u);
    stream << _ready_table.copy_from(luisa::span{zeros}) << synchronize();
    _initialized = true;
  }

  [[nodiscard]] bool async_mode() const noexcept { return _async_mode; }
  [[nodiscard]] std::uint32_t slot_count() const noexcept { return _slot_count; }
  [[nodiscard]] std::uint32_t total_threads() const noexcept {
    return _total_threads;
  }
  [[nodiscard]] bool always_unready() const noexcept { return _always_unready; }
  [[nodiscard]] const luisa::compute::Buffer<luisa::uint> &
  ready_table() const noexcept {
    return _ready_table;
  }

  std::uint32_t next_round() noexcept {
    std::lock_guard lock{_mtx};
    return _round++;
  }

  void begin_submit(const WavefrontCoroAsyncBatchTicket &ticket) {
    const std::size_t idx =
        ticket.valid() ? static_cast<std::size_t>(ticket.slot_index) : 0u;
    require(idx < _slot_count, "FilteredAdvance submit slot out of range");
    std::lock_guard lock{_mtx};
    _slots[idx] = Slot{
        .active = true,
        .ready = false,
        .generation = ticket.generation,
        .submit_seq = ++_next_submit_seq};
  }

  void mark_submitted(std::size_t slot_idx) noexcept {
    {
      std::lock_guard lock{_mtx};
      if (slot_idx < _slots.size() && _slots[slot_idx].active) {
        _slots[slot_idx].ready = true;
      }
    }
    _cv.notify_all();
  }

  [[nodiscard]] bool
  poll_ready(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
    std::lock_guard lock{_mtx};
    std::size_t best = _slot_count;
    std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t s = 0u; s < _slot_count; ++s) {
      if (_slots[s].active && _slots[s].ready &&
          _slots[s].submit_seq < best_seq) {
        best_seq = _slots[s].submit_seq;
        best = s;
      }
    }
    if (best >= _slot_count) {
      return false;
    }
    out_ticket = WavefrontCoroAsyncBatchTicket{
        .slot_index = static_cast<luisa::uint>(best),
        .generation = _slots[best].generation};
    return true;
  }

  [[nodiscard]] bool
  wait_any(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept {
    std::unique_lock lock{_mtx};
    auto find_ready = [&]() noexcept -> std::size_t {
      std::size_t best = _slot_count;
      std::uint64_t best_seq = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t s = 0u; s < _slot_count; ++s) {
        if (_slots[s].active && _slots[s].ready &&
            _slots[s].submit_seq < best_seq) {
          best_seq = _slots[s].submit_seq;
          best = s;
        }
      }
      return best;
    };
    std::size_t slot = find_ready();
    if (slot >= _slot_count) {
      const bool any_act = std::any_of(
          _slots.begin(), _slots.end(), [](const Slot &s) { return s.active; });
      if (!any_act) {
        return false;
      }
      _cv.wait(lock, [&]() noexcept {
        slot = find_ready();
        const bool active_rem = std::any_of(
            _slots.begin(), _slots.end(), [](const Slot &s) { return s.active; });
        return slot < _slot_count || !active_rem;
      });
    }
    if (slot >= _slot_count) {
      return false;
    }
    out_ticket = WavefrontCoroAsyncBatchTicket{
        .slot_index = static_cast<luisa::uint>(slot),
        .generation = _slots[slot].generation};
    return true;
  }

  void finish_publish(const WavefrontCoroAsyncBatchTicket &ticket) {
    const std::size_t idx =
        ticket.valid() ? static_cast<std::size_t>(ticket.slot_index) : 0u;
    require(idx < _slot_count, "FilteredAdvance publish slot out of range");
    {
      std::lock_guard lock{_mtx};
      require(_slots[idx].active &&
                  (!ticket.valid() || _slots[idx].generation == ticket.generation),
              "FilteredAdvance publish ticket mismatch");
      _slots[idx].active = false;
      _slots[idx].ready = false;
    }
    _cv.notify_all();
  }
};

class TestFilteredAdvanceHandler final
    : public WavefrontCoroSchedulerExtensionHandler {
private:
  std::shared_ptr<FilteredAdvanceCoordinator> _coord;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>,
                           luisa::uint, luisa::uint>
      _mark_round;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::uint, luisa::uint, luisa::uint>
      _eval_filter;

public:
  TestFilteredAdvanceHandler(WavefrontCoroExtensionPrepareContext &ctx,
                             const WavefrontCoroExtensionStage &stage,
                             std::shared_ptr<FilteredAdvanceCoordinator> coord)
      : _coord{std::move(coord)} {
    _coord->ensure_initialized(ctx.device, ctx.stream);
    const auto *desc = &ctx.frame_desc;
    const auto *id_binding = &stage.binding("id");
    const auto fields = stage.dataflow->reconstruct_slots;
    const auto layout = ctx.frame_layout;
    const bool frame_soa = ctx.global_memory_soa;
    const bool rt_cap = layout.has_runtime_capacity();
    luisa::vector<size_t> fields_vec{fields.begin(), fields.end()};

    Kernel1D mark_k = [](BufferUInt ready_table, UInt round_mod4, UInt n) noexcept {
      const auto x = dispatch_x();
      $if(x >= n) { $return(); };
      $if((x & 3u) == round_mod4) {
        ready_table.write(x, 1u);
      };
    };
    _mark_round = ctx.device.compile(mark_k, ctx.shader_option);

    Kernel1D filter_k = [desc, id_binding, fields_vec, layout, frame_soa, rt_cap](
                            ByteBufferVar storage, BufferUInt indices,
                            BufferUInt ready_table, BufferUInt ready_flags,
                            UInt force_zero, UInt capacity, UInt n) noexcept {
      const auto x = dispatch_x();
      $if(x >= n) { $return(); };
      $if(force_zero != 0u) {
        ready_flags.write(x, 0u);
        $return();
      };
      const auto idx = indices.read(x);
      auto frame = CoroFrame::create(desc);
      if (rt_cap) {
        coro_frame_load_into(frame, storage, idx, capacity, layout, frame_soa,
                             fields_vec, false, false);
      } else {
        coro_frame_load_into(frame, storage, idx, layout, frame_soa,
                             fields_vec, false, false);
      }
      const auto fid = id_binding->read<luisa::uint>(frame);
      ready_flags.write(x, select(0u, 1u, ready_table.read(fid) != 0u));
    };
    _eval_filter = ctx.device.compile(filter_k, ctx.shader_option);
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "test.filt_ro";
  }

  [[nodiscard]] WavefrontCoroExtensionExecution
  execution() const noexcept override {
    return _coord->async_mode() ? WavefrontCoroExtensionExecution::blocking_async
                                : WavefrontCoroExtensionExecution::stage;
  }

  [[nodiscard]] luisa::string_view async_domain() const noexcept override {
    return "test.filt_domain";
  }

  [[nodiscard]] luisa::uint
  max_async_in_flight_batches() const noexcept override {
    return _coord->slot_count();
  }

  [[nodiscard]] bool supports_command_list() const noexcept override {
    return false;
  }

  [[nodiscard]] bool supports_filtered_advance() const noexcept override {
    return true;
  }

  void encode_advance_filter(
      const WavefrontCoroAdvanceFilterContext &ctx) noexcept override {
    if (ctx.frame_count == 0u) {
      return;
    }
    const luisa::uint force_zero = _coord->always_unready() ? 1u : 0u;
    ctx.command_list
        << _eval_filter(ctx.frame_buffer, ctx.frame_indices,
                        _coord->ready_table(), ctx.ready_flags, force_zero,
                        ctx.frame_capacity, ctx.frame_count)
               .dispatch(ctx.frame_count);
  }

  void dispatch(const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    const std::uint32_t r = _coord->next_round();
    if (!_coord->always_unready()) {
      CommandList cmd;
      cmd << _mark_round(_coord->ready_table(), r & 3u, _coord->total_threads())
                 .dispatch(_coord->total_threads());
      ctx.stream << cmd.commit();
    }
  }

  [[nodiscard]] bool submit_async(
      const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    if (ctx.frame_count == 0u || !_coord->async_mode()) {
      return false;
    }
    _coord->begin_submit(ctx.async_ticket);
    const std::size_t slot_idx =
        ctx.async_ticket.valid()
            ? static_cast<std::size_t>(ctx.async_ticket.slot_index)
            : 0u;
    CommandList cmd;
    cmd.add_callback([coord = _coord, slot_idx]() noexcept {
      coord->mark_submitted(slot_idx);
    });
    ctx.stream << cmd.commit();
    return true;
  }

  [[nodiscard]] bool
  poll_async_ready(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override {
    return _coord->poll_ready(out_ticket);
  }

  [[nodiscard]] bool
  wait_async_any(WavefrontCoroAsyncBatchTicket &out_ticket) noexcept override {
    return _coord->wait_any(out_ticket);
  }

  void publish_async(const WavefrontCoroAsyncPublishContext &ctx) noexcept override {
    const std::uint32_t r = _coord->next_round();
    if (!_coord->always_unready()) {
      ctx.command_list
          << _mark_round(_coord->ready_table(), r & 3u, _coord->total_threads())
                 .dispatch(_coord->total_threads());
    }
    _coord->finish_publish(ctx.async_ticket);
  }
};

void test_coro_scheduler_filtered_advance(Device &device, Stream &stream) {
  std::cout << "[MultiSlot Test 3] WavefrontCoroScheduler filtered advance (sync/async/liveness)...\n";

  auto run_case = [&](bool async_mode,
                      std::uint32_t slots,
                      std::uint32_t total_threads,
                      std::uint32_t pool_cap,
                      bool always_unready) {
    Coroutine<void(Buffer<luisa::uint>, Buffer<luisa::uint>)> coro{[async_mode](
        BufferUInt output, BufferUInt ready_table) {
      const auto tid = dispatch_x();
      UInt id = tid;
      UInt fallback_hit = def(0u);
      $loop {
        $suspend("filter_wait", coro_stage("test.filt_ro")
                                  .read("id", id)
                                  .blocking_async(async_mode)
                                  .async_domain("test.filt_domain"));
        $if(ready_table.read(id) != 0u) {
          fallback_hit = 0u;
          $break;
        };
        fallback_hit = 1u;
        $break;
      };
      output.write(tid, select(tid * 13u + 5u, tid ^ 0xCAFEBABEu,
                               fallback_hit != 0u));
    }};

    WavefrontCoroSchedulerConfig cfg;
    cfg.thread_count = pool_cap;
    cfg.execution_block_size = 64u;
    cfg.global_memory_soa = true;
    cfg.frame_buffer_compaction = true;
    cfg.largest_continuation_first = true;
    cfg.incremental_continuation_counts = true;
    cfg.fused_continuation_counts = true;
    cfg.max_async_in_flight_batches = slots;
    cfg.report_stats = true;

    auto coord = std::make_shared<FilteredAdvanceCoordinator>(
        async_mode, slots, total_threads, always_unready);
    WavefrontCoroScheduler<Buffer<luisa::uint>, Buffer<luisa::uint>> scheduler{
        device, coro, cfg};
    scheduler.register_extension_handler(
        stream, [coord](WavefrontCoroExtensionPrepareContext &ctx,
                        const WavefrontCoroExtensionStage &stage)
                    -> luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
          if (!stage.extension ||
              stage.extension->schema() != "test.filt_ro") {
            return nullptr;
          }
          return luisa::make_unique<TestFilteredAdvanceHandler>(
              ctx, stage, coord);
        });

    auto out_buf = device.create_buffer<luisa::uint>(total_threads);
    std::vector<luisa::uint> host_out(total_threads, 0u);
    stream << out_buf.copy_from(luisa::span{host_out}) << commit();
    stream << scheduler(out_buf, coord->ready_table()).dispatch(total_threads);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    const auto &stats = scheduler.last_dispatch_stats();
    std::uint64_t post_cont_exec = 0ull;
    for (std::size_t c = 1u; c < stats.continuations.size(); ++c) {
      post_cont_exec += stats.continuations[c].executed_count;
    }
    require(post_cont_exec == total_threads,
            "Filtered advance leaked or missed continuation executions: " +
                std::to_string(post_cont_exec) + " vs " +
                std::to_string(total_threads));
    require(!stats.extensions.empty(), "Missing extension stats");
    const auto ext_exec = stats.extensions[0u].executed_count;
    const auto ext_disp = stats.extensions[0u].dispatch_count;
    require(ext_exec > total_threads,
            "Expected filtered advance to re-queue unready frames inside extension (ext_exec=" +
                std::to_string(ext_exec) + ")");

    for (std::uint32_t tid = 0u; tid < total_threads; ++tid) {
      const luisa::uint expected =
          always_unready ? (tid ^ 0xCAFEBABEu) : (tid * 13u + 5u);
      require(host_out[tid] == expected,
              "Filtered advance output mismatch at tid=" + std::to_string(tid));
    }
    if (always_unready) {
      require(ext_disp >= 3u,
              "Expected zero-advance fallback after >=2 zero-advance rounds");
    }
    std::cout << "  [" << (async_mode ? "async" : "sync")
              << " slots=" << slots << " unready=" << (always_unready ? 1 : 0)
              << "] cont_exec=" << post_cont_exec
              << " ext_exec=" << ext_exec << " ext_disp=" << ext_disp
              << " (OK)\n";
  };

  run_case(false, 1u, 2048u, 512u, false);
  run_case(true, 1u, 2048u, 512u, false);
  run_case(true, 2u, 2048u, 512u, false);
  run_case(true, 1u, 256u, 256u, true);
  run_case(false, 1u, 257u, 257u, false);
  run_case(true, 2u, 257u, 257u, false);
}

} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running Psycles Stage 3D multi-slot tests on backend: "
              << backend << '\n';

    test_coro_scheduler_multislot_ooo_and_compaction(device, stream);
    test_vt_cache_multislot_parity_and_stress(device, stream);
    test_coro_scheduler_filtered_advance(device, stream);

    std::cout << "ALL STAGE 3D MULTI-SLOT TESTS PASSED on " << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
