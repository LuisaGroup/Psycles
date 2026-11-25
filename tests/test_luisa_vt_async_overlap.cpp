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
  if (auto out = OIIO::ImageOutput::create("vt_async_test.png", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_async_test.png", spec), "OIIO open PNG mem failed");
    require(out->write_image(OIIO::TypeDesc::UINT8, rgba8.data()),
            "OIIO write PNG mem failed");
    require(out->close(), "OIIO close PNG mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_async_tmp_" +
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
  bytes.reserve(bytes.size() + static_cast<std::size_t>(width) * height * 3u);
  for (std::size_t i = 0u; i < rgba8.size(); i += 4u) {
    bytes.push_back(rgba8[i]);
    bytes.push_back(rgba8[i + 1u]);
    bytes.push_back(rgba8[i + 2u]);
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
  if (auto out = OIIO::ImageOutput::create("vt_async_test.exr", &mem_io);
      out && out->supports("ioproxy")) {
    require(out->open("vt_async_test.exr", spec), "OIIO open EXR mem failed");
    require(out->write_image(OIIO::TypeDesc::FLOAT, rgba32f.data()),
            "OIIO write EXR mem failed");
    require(out->close(), "OIIO close EXR mem failed");
    return std::vector<std::uint8_t>(encoded_u8.begin(), encoded_u8.end());
  }
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      ("psycles_vt_async_tmp_" +
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
// Test 1: Generic CoroGraph blocking_async + read_write + overlap parity
// =========================================================================

struct AsyncRwStats {
  std::atomic<std::uint64_t> submit_calls{0ull};
  std::atomic<std::uint64_t> poll_pending_hits{0ull};
  std::atomic<std::uint64_t> poll_ready_hits{0ull};
  std::atomic<std::uint64_t> wait_calls{0ull};
  std::atomic<std::uint64_t> publish_calls{0ull};
};

class TestAsyncRwHandler final : public WavefrontCoroSchedulerExtensionHandler {
private:
  enum class BatchState : std::uint8_t { Idle = 0u, InFlight = 1u, Ready = 2u };

  bool _async_mode{false};
  std::shared_ptr<AsyncRwStats> _stats;
  luisa::compute::Buffer<luisa::uint> _staging_by_frame;
  luisa::compute::Buffer<luisa::uint> _submitted_indices;
  luisa::compute::Buffer<luisa::uint> _upload_by_frame;
  std::vector<luisa::uint> _host_staging;
  std::vector<luisa::uint> _host_submitted;

  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint>
      _sync_rmw;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::compute::Buffer<luisa::uint>,
                           luisa::uint, luisa::uint>
      _extract;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer, luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint, luisa::uint>
      _inject;

  std::mutex _mtx;
  std::condition_variable _worker_cv;
  std::condition_variable _ready_cv;
  std::thread _worker;
  bool _stop{false};
  bool _has_work{false};
  std::uint32_t _pending_count{0u};
  std::atomic<BatchState> _state{BatchState::Idle};
  std::chrono::microseconds _worker_sleep{std::chrono::milliseconds(5)};

  void worker_loop() noexcept {
    while (true) {
      std::uint32_t count = 0u;
      {
        std::unique_lock lock{_mtx};
        _worker_cv.wait(lock, [&]() noexcept { return _stop || _has_work; });
        if (_stop && !_has_work) {
          return;
        }
        count = _pending_count;
        _has_work = false;
      }
      // Hold worker in-flight briefly so independent GPU continuations run & poll pending.
      if (_worker_sleep.count() > 0) {
        std::this_thread::sleep_for(_worker_sleep);
      }
      for (std::uint32_t i = 0u; i < count; ++i) {
        const auto fidx = _host_submitted[i];
        _host_staging[fidx] = _host_staging[fidx] * 13u + 0x12345u;
      }
      {
        std::lock_guard lock{_mtx};
        _state.store(BatchState::Ready, std::memory_order_release);
      }
      _ready_cv.notify_all();
    }
  }

public:
  TestAsyncRwHandler(WavefrontCoroExtensionPrepareContext &ctx,
                     const WavefrontCoroExtensionStage &stage,
                     bool async_mode,
                     std::shared_ptr<AsyncRwStats> stats,
                     std::chrono::microseconds worker_sleep = std::chrono::milliseconds(5))
      : _async_mode{async_mode},
        _stats{std::move(stats)},
        _worker_sleep{worker_sleep},
        _staging_by_frame{ctx.device.create_buffer<luisa::uint>(ctx.frame_capacity)},
        _submitted_indices{ctx.device.create_buffer<luisa::uint>(ctx.frame_capacity)},
        _upload_by_frame{ctx.device.create_buffer<luisa::uint>(ctx.frame_capacity)},
        _host_staging(ctx.frame_capacity, 0u),
        _host_submitted(ctx.frame_capacity, 0u) {
    const auto *desc = &ctx.frame_desc;
    const auto *val_binding = &stage.binding("val");
    const auto fields = stage.dataflow->reconstruct_slots;
    const auto writes = stage.dataflow->required_def.slots;
    const auto layout = ctx.frame_layout;
    const bool frame_soa = ctx.global_memory_soa;
    const bool rt_cap = layout.has_runtime_capacity();

    luisa::vector<size_t> fields_vec{fields.begin(), fields.end()};
    luisa::vector<size_t> writes_vec{writes.begin(), writes.end()};

    Kernel1D sync_k = [desc, val_binding, fields_vec, writes_vec, layout, frame_soa,
                       rt_cap](ByteBufferVar storage, BufferUInt indices,
                               BufferUInt unused_staging, UInt capacity,
                               UInt n) noexcept {
      (void)unused_staging;
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
      const auto v = val_binding->read<luisa::uint>(frame);
      val_binding->write<luisa::uint>(frame, v * 13u + 0x12345u);
      if (rt_cap) {
        coro_frame_store(storage, idx, capacity, frame, layout, frame_soa,
                         writes_vec, false, false);
      } else {
        coro_frame_store(storage, idx, frame, layout, frame_soa,
                         writes_vec, false, false);
      }
    };
    _sync_rmw = ctx.device.compile(sync_k, ctx.shader_option);

    Kernel1D extract_k = [desc, val_binding, fields_vec, layout, frame_soa,
                          rt_cap](ByteBufferVar storage, BufferUInt indices,
                                  BufferUInt staging_out, BufferUInt submitted_out,
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
      staging_out.write(idx, val_binding->read<luisa::uint>(frame));
      submitted_out.write(x, idx);
    };
    _extract = ctx.device.compile(extract_k, ctx.shader_option);

    Kernel1D inject_k = [desc, val_binding, fields_vec, writes_vec, layout, frame_soa,
                         rt_cap](ByteBufferVar storage, BufferUInt indices,
                                 BufferUInt upload_in, UInt capacity,
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
      val_binding->write<luisa::uint>(frame, upload_in.read(idx));
      if (rt_cap) {
        coro_frame_store(storage, idx, capacity, frame, layout, frame_soa,
                         writes_vec, false, false);
      } else {
        coro_frame_store(storage, idx, frame, layout, frame_soa,
                         writes_vec, false, false);
      }
    };
    _inject = ctx.device.compile(inject_k, ctx.shader_option);

    if (_async_mode) {
      _worker = std::thread([this]() noexcept { worker_loop(); });
    }
  }

  ~TestAsyncRwHandler() noexcept override {
    if (_async_mode) {
      {
        std::unique_lock lock{_mtx};
        _ready_cv.wait(lock, [&]() noexcept {
          return _state.load(std::memory_order_acquire) != BatchState::InFlight;
        });
        _stop = true;
      }
      _worker_cv.notify_all();
      if (_worker.joinable()) {
        _worker.join();
      }
    }
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "test.async_rw";
  }

  [[nodiscard]] WavefrontCoroExtensionExecution execution() const noexcept override {
    return _async_mode ? WavefrontCoroExtensionExecution::blocking_async
                       : WavefrontCoroExtensionExecution::stage;
  }

  [[nodiscard]] bool defer_while_other_queues_nonempty() const noexcept override {
    return true;
  }

  [[nodiscard]] bool supports_command_list() const noexcept override {
    return true;
  }

  [[nodiscard]] luisa::string_view async_domain() const noexcept override {
    return "test.async_rw_domain";
  }

  void dispatch(const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    ++_stats->submit_calls;
    ++_stats->publish_calls;
    CommandList local_cmd;
    CommandList *cmd = ctx.command_list ? ctx.command_list : &local_cmd;
    *cmd << _sync_rmw(ctx.frame_buffer, ctx.frame_indices, _staging_by_frame,
                      ctx.frame_capacity, ctx.frame_count)
                .dispatch(ctx.frame_count);
    if (ctx.command_list == nullptr) {
      ctx.stream << local_cmd.commit();
    }
  }

  [[nodiscard]] bool
  submit_async(const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
    if (ctx.frame_count == 0u) {
      return false;
    }
    ++_stats->submit_calls;
    _state.store(BatchState::InFlight, std::memory_order_release);
    CommandList local_cmd;
    CommandList *cmd = ctx.command_list ? ctx.command_list : &local_cmd;
    *cmd << _extract(ctx.frame_buffer, ctx.frame_indices, _staging_by_frame,
                     _submitted_indices, ctx.frame_capacity, ctx.frame_count)
                .dispatch(ctx.frame_count)
         << _staging_by_frame.copy_to(luisa::span{_host_staging})
         << _submitted_indices.view(0u, ctx.frame_count)
                .copy_to(luisa::span{_host_submitted}.subspan(0u, ctx.frame_count));
    cmd->add_callback([this, cnt = ctx.frame_count]() noexcept {
      {
        std::lock_guard lock{_mtx};
        _pending_count = cnt;
        _has_work = true;
      }
      _worker_cv.notify_one();
    });
    ctx.stream << cmd->commit();
    return true;
  }

  [[nodiscard]] WavefrontCoroAsyncPollResult poll_async() noexcept override {
    const auto st = _state.load(std::memory_order_acquire);
    if (st == BatchState::Idle) {
      return WavefrontCoroAsyncPollResult::idle;
    }
    if (st == BatchState::InFlight) {
      ++_stats->poll_pending_hits;
      return WavefrontCoroAsyncPollResult::pending;
    }
    ++_stats->poll_ready_hits;
    return WavefrontCoroAsyncPollResult::ready;
  }

  void wait_async() noexcept override {
    if (_state.load(std::memory_order_acquire) != BatchState::InFlight) {
      return;
    }
    ++_stats->wait_calls;
    std::unique_lock lock{_mtx};
    _ready_cv.wait(lock, [&]() noexcept {
      return _state.load(std::memory_order_acquire) != BatchState::InFlight;
    });
  }

  void publish_async(const WavefrontCoroAsyncPublishContext &ctx) noexcept override {
    if (_state.load(std::memory_order_acquire) == BatchState::InFlight) {
      wait_async();
    }
    ++_stats->publish_calls;
    auto kept_upload = std::make_shared<std::vector<luisa::uint>>(_host_staging);
    ctx.command_list
        << _upload_by_frame.copy_from(luisa::span{*kept_upload})
        << _inject(ctx.frame_buffer, ctx.frame_indices, _upload_by_frame,
                   ctx.frame_capacity, ctx.frame_count)
               .dispatch(ctx.frame_count);
    ctx.command_list.add_callback([kept_upload]() noexcept {});
    _state.store(BatchState::Idle, std::memory_order_release);
  }
};

void test_coro_graph_blocking_async_overlap_and_rw_parity(Device &device,
                                                          Stream &stream) {
  std::cout << "[Async Test 1] CoroGraph blocking_async + read_write + overlap parity...\n";
  constexpr std::uint32_t kThreads = 4096u;
  constexpr std::uint32_t kPoolCap = 1024u;

  auto run_mode = [&](bool async_mode, AsyncRwStats &out_stats) {
    Coroutine<void(Buffer<luisa::uint>)> coro{[async_mode](BufferUInt output) {
      const auto tid = dispatch_x();
      $if((tid & 1u) == 0u) {
        UInt v = def(tid * 17u + 3u);
        $suspend("branch_a", coro_stage("test.async_rw")
                                 .read_write("val", v)
                                 .blocking_async(async_mode)
                                 .async_domain("test.async_rw_domain"));
        output.write(tid, v ^ 0x5a5a5a5au);
      }
      $else {
        UInt acc = def(tid * 31u + 11u);
        $suspend("branch_b_0");
        acc = acc * 3u + 7u;
        $suspend("branch_b_1");
        acc = acc ^ (tid * 5u);
        $suspend("branch_b_2");
        output.write(tid, acc + 19u);
      };
    }};

    bool found_stage = false;
    for (const auto &boundary : coro.graph().boundaries()) {
      for (const auto &stage : boundary.stages) {
        if (boundary.extensions[stage.extension_index]->schema() ==
            "test.async_rw") {
          found_stage = true;
          require(stage.is_blocking_async() == async_mode,
                  "CoroGraph Stage::is_blocking_async mismatch");
          require(stage.async_domain == "test.async_rw_domain",
                  "CoroGraph Stage::async_domain mismatch");
        }
      }
    }
    require(found_stage, "CoroGraph missing test.async_rw stage");

    WavefrontCoroSchedulerConfig cfg;
    cfg.thread_count = kPoolCap;
    cfg.execution_block_size = 64u;
    cfg.global_memory_soa = true;
    cfg.frame_buffer_compaction = true;
    cfg.largest_continuation_first = true;
    cfg.incremental_continuation_counts = true;
    cfg.fused_continuation_counts = true;

    auto stats_ptr = std::make_shared<AsyncRwStats>();
    WavefrontCoroScheduler<Buffer<luisa::uint>> scheduler{device, coro, cfg};
    scheduler.register_extension_handler(
        stream, [async_mode, stats_ptr](
                    WavefrontCoroExtensionPrepareContext &ctx,
                    const WavefrontCoroExtensionStage &stage)
                    -> luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
          if (!stage.extension ||
              stage.extension->schema() != "test.async_rw") {
            return nullptr;
          }
          return luisa::make_unique<TestAsyncRwHandler>(ctx, stage, async_mode,
                                                      stats_ptr);
        });

    auto out_buf = device.create_buffer<luisa::uint>(kThreads);
    std::vector<luisa::uint> host_out(kThreads, 0u);
    stream << out_buf.copy_from(luisa::span{host_out}) << commit();
    stream << scheduler(out_buf).dispatch(kThreads);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    out_stats.submit_calls.store(stats_ptr->submit_calls.load());
    out_stats.poll_pending_hits.store(stats_ptr->poll_pending_hits.load());
    out_stats.poll_ready_hits.store(stats_ptr->poll_ready_hits.load());
    out_stats.wait_calls.store(stats_ptr->wait_calls.load());
    out_stats.publish_calls.store(stats_ptr->publish_calls.load());
    return host_out;
  };

  AsyncRwStats sync_stats;
  const auto sync_out = run_mode(false, sync_stats);
  AsyncRwStats async_stats;
  const auto async_out = run_mode(true, async_stats);

  for (std::uint32_t tid = 0u; tid < kThreads; ++tid) {
    luisa::uint expected = 0u;
    if ((tid & 1u) == 0u) {
      expected = ((tid * 17u + 3u) * 13u + 0x12345u) ^ 0x5a5a5a5au;
    } else {
      const luisa::uint acc0 = tid * 31u + 11u;
      const luisa::uint acc1 = acc0 * 3u + 7u;
      const luisa::uint acc2 = acc1 ^ (tid * 5u);
      expected = acc2 + 19u;
    }
    require(sync_out[tid] == expected,
            "Sync reference value mismatch at tid=" + std::to_string(tid));
  }
  require(sync_out.size() == async_out.size() &&
              std::memcmp(sync_out.data(), async_out.data(),
                          sync_out.size() * sizeof(luisa::uint)) == 0,
          "Test 1 bit-exact parity failed between sync and blocking_async");
  require(async_stats.submit_calls.load() > 0ull,
          "Test 1 expected submit_calls > 0");
  require(async_stats.publish_calls.load() == async_stats.submit_calls.load(),
          "Test 1 expected publish_calls == submit_calls");
  require(async_stats.poll_pending_hits.load() > 0ull,
          "Test 1 expected poll_pending_hits > 0 proving overlap with independent branches");

  std::cout << "  submit=" << async_stats.submit_calls.load()
            << " poll_pending=" << async_stats.poll_pending_hits.load()
            << " poll_ready=" << async_stats.poll_ready_hits.load()
            << " wait=" << async_stats.wait_calls.load()
            << " publish=" << async_stats.publish_calls.load()
            << " (bit-exact OK)\n";
}

// =========================================================================
// Test 2: Real VirtualTextureCache async worker parity + eviction + aux drain
// =========================================================================

class TestVtAuxWork final
    : public WavefrontCoroAuxiliaryWork<luisa::compute::Buffer<luisa::float4>,
                                        luisa::compute::Buffer<luisa::float4>,
                                        luisa::compute::Buffer<luisa::uint>> {
private:
  std::uint32_t _capacity;
  std::shared_ptr<VirtualTextureCache> _vt;
  luisa::compute::BindlessArray _direct_heap;
  luisa::compute::Buffer<luisa::uint2> _item_ids; // (tid, binding)
  luisa::compute::Buffer<luisa::float2> _item_uv;
  luisa::compute::Buffer<luisa::uint> _active;
  luisa::compute::Buffer<luisa::uint> _free_slots;
  luisa::compute::Buffer<luisa::uint> _free_count;
  luisa::compute::Buffer<luisa::uint> _counts; // [0]=active_count, [1]=miss_flag
  std::array<luisa::uint, 2u> _host_counts{0u, 0u};
  std::array<WavefrontCoroAuxiliaryProducer, 1u> _producers{
      {{.continuation = "after_vt_even", .max_emitted_per_invocation = 1u}}};
  luisa::compute::Shader1D<luisa::uint> _init_shader;
  luisa::compute::Shader1D<luisa::uint> _clear_flag_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::float4>, luisa::uint>
      _consume_shader;

public:
  std::uint64_t aux_miss_drains_seen{0ull};

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

    Kernel1D clr_k = [this](UInt idx) noexcept {
      _counts->write(idx, 0u);
    };
    _clear_flag_shader = device.compile(clr_k);

    Kernel1D consume_k = [this](BufferFloat4 aux_out, UInt cap) noexcept {
      const auto slot = dispatch_x();
      $if(slot >= cap) { $return(); };
      $if(_active->read(slot) != 0u) {
        const auto ids = _item_ids->read(slot);
        const auto uv = _item_uv->read(slot);
        UInt miss = vt_invalid_page;
        const auto col = _vt->sample_2d_dynamic(
            Int(ids.y), uv, _direct_heap, &miss, false);
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
    return "test_vt_aux";
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
    aux_miss_drains_seen = 0ull;
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
      ++aux_miss_drains_seen;
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

void test_vt_cache_async_worker_parity_and_aux_drain(Device &device,
                                                     Stream &stream) {
  std::cout << "[Async Test 2] VirtualTextureCache async worker NPOT BYTE4+FLOAT4 + eviction + aux parity...\n";

  // NPOT dimensions spanning multiple 60x60 logical tiles:
  // BYTE4 313x197 -> 6x4 = 24 tiles/ext * 4 exts = 96 virtual pages
  // FLOAT4 251x193 -> 5x4 = 20 tiles/ext * 4 exts = 80 virtual pages
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
      b4_px[i + 3u] = static_cast<std::uint8_t>((x * 3u + y * 29u + 160u) & 0xffu);
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
  const auto b4_probe = probe_scene_image_info(b4_png, "async_b4.png");
  const auto f4_probe = probe_scene_image_info(f4_exr, "async_f4.exr");
  require(b4_probe.has_value() && f4_probe.has_value(),
          "Test 2 failed to probe synthetic NPOT images");

  const std::vector<VtProbedSceneImage> probed_images{
      VtProbedSceneImage{.id = ImageId{1u},
                         .name = "async_b4.png",
                         .encoded_data = b4_png,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *b4_probe},
      VtProbedSceneImage{.id = ImageId{2u},
                         .name = "async_f4.exr",
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
  require(num_bindings == 24u, "Test 2 expected 24 wrap x filter bindings");

  BindlessArray dummy_direct_heap = device.create_bindless_array(num_bindings);
  // Populate bindless array with decoded dummy textures to keep Vulkan descriptors valid.
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

  auto run_vt_case = [&](LuisaVirtualTexturePrepackMode prepack_mode,
                         bool async_worker,
                         LuisaVirtualTextureCopyQueueMode cq_mode =
                             LuisaVirtualTextureCopyQueueMode::off,
                         std::uint32_t dispatch_repeats = 1u)
      -> VtRunResult {
    LuisaVirtualTextureCacheOptions opts{
        .enabled = true,
        .gpu_cache_budget_bytes = 1024ull * 1024ull, // 1 MiB forces eviction on ~6.5 MiB demand
        .physical_tile_size = 64u,
        .prepack_mode = prepack_mode,
        .host_prepacked_cache_budget_bytes = 64ull * 1024ull * 1024ull,
        .prepack_warm_on_init =
            (prepack_mode != LuisaVirtualTexturePrepackMode::off),
        .enable_async_worker = async_worker,
        .async_batch_slots = 1u,
        .copy_queue_mode = cq_mode};

    const std::size_t total_out =
        static_cast<std::size_t>(kThreads) * dispatch_repeats;
    VtRunResult result;
    result.main_out.resize(total_out);
    result.aux_out.resize(total_out);
    result.indep_out.resize(total_out);
    {
      auto vt = VirtualTextureCache::create(device, stream, opts, probed_images,
                                            handles, gpu_bindings);

      auto vt_wait_resident = [&](Expr<luisa::uint> page_id) noexcept {
        UInt missed_page = page_id;
        $loop {
          $suspend("vt_wait",
                   coro_stage(luisa::string{vt_extension_schema})
                       .read(luisa::string{vt_extension_binding}, missed_page)
                       .blocking_async()
                       .async_domain(luisa::string{vt_extension_schema}));
          $if(vt->is_resident(missed_page)) { $break; };
        };
      };

      auto sample_with_vt = [&](Expr<luisa::uint> b, Expr<luisa::float2> uv) {
        Float4 col = make_float4(0.0f);
        $loop {
          UInt miss = vt_invalid_page;
          col = vt->sample_2d_dynamic(Int(b), uv, dummy_direct_heap, &miss);
          $if(miss == vt_invalid_page) { $break; };
          vt_wait_resident(miss);
        };
        return col;
      };

      auto aux_work = std::make_shared<TestVtAuxWork>(
          device, kThreads, vt, dummy_direct_heap);
      auto pass_salt_buf = device.create_buffer<luisa::uint>(1u);

      Coroutine<void(Buffer<luisa::float4>, Buffer<luisa::float4>,
                     Buffer<luisa::uint>)>
          coro{[&](BufferFloat4 main_out, BufferFloat4 aux_out,
                   BufferUInt indep_out) {
            (void)aux_out;
            const auto tid = dispatch_x();
            const UInt rep_salt = pass_salt_buf->read(0u);
            const UInt b0 = (tid * 7u + (tid / 32u)) % num_bindings;
            const UInt gx = (tid * 13u + 3u) % 256u;
            const UInt gy = (tid * 19u + 7u) % 256u;
            // Cover interior, edges, negative/over-1 wraps, and tile seams.
            const Float2 uv0 = make_float2(
                -0.25f + 1.50f * (cast<float>(gx) / 255.0f),
                -0.20f + 1.45f * (cast<float>(gy) / 255.0f));
            $if((tid & 1u) == 0u) {
              const auto c0 = sample_with_vt(b0, uv0);
              Float4 c_const = make_float4(0.0f);
              $loop {
                UInt miss_c = vt_invalid_page;
                c_const = vt->sample_2d_const(0, uv0, &miss_c);
                $if(miss_c == vt_invalid_page) { $break; };
                vt_wait_resident(miss_c);
              };
              main_out.write(tid, c0 + c_const);
              indep_out.write(tid, tid ^ (rep_salt * 0x13579bdfu) ^ 0xa5a5a5a5u);
              $suspend("after_vt_even");
              $if((tid & 7u) == 0u) {
                const UInt aux_b = (b0 + 13u) % num_bindings;
                aux_work->emit(tid, uv0, aux_b);
              };
            }
            $else {
              UInt acc = def(tid * 11u + 5u + rep_salt * 17u);
              $suspend("indep_0");
              acc = acc * 3u + 1u;
              $suspend("indep_1");
              acc = acc ^ (tid * 7u);
              $suspend("indep_2");
              indep_out.write(tid, acc);
              const auto c1 = sample_with_vt(b0, uv0);
              main_out.write(tid, c1);
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
      for (std::uint32_t rep = 0u; rep < dispatch_repeats; ++rep) {
        const std::size_t off = static_cast<std::size_t>(rep) * kThreads;
        stream << pass_salt_buf.copy_from(&rep)
               << main_buf.copy_from(luisa::span{zero_f4})
               << aux_buf.copy_from(luisa::span{zero_f4})
               << indep_buf.copy_from(luisa::span{zero_u32}) << commit();
        stream << scheduler(main_buf, aux_buf, indep_buf).dispatch(kThreads);
        stream << main_buf.copy_to(luisa::span{result.main_out.data() + off, kThreads})
               << aux_buf.copy_to(luisa::span{result.aux_out.data() + off, kThreads})
               << indep_buf.copy_to(luisa::span{result.indep_out.data() + off, kThreads})
               << synchronize();
      }

      result.stats = vt->stats();
    } // Verifies clean VirtualTextureCache destruction / stop_async_worker() join.
    return result;
  };

  auto check_f4_parity = [&](const std::string &tag,
                             const std::string &buf_name,
                             const std::vector<luisa::float4> &a,
                             const std::vector<luisa::float4> &b,
                             std::uint32_t binding_offset) {
    std::size_t closest_diff = 0u;
    std::size_t filtered_fail = 0u;
    double filtered_max_abs = 0.0;
    for (std::size_t i = 0u; i < a.size(); ++i) {
      const auto tid = static_cast<std::uint32_t>(i % kThreads);
      const auto bind_id =
          ((tid * 7u + (tid / 32u)) % num_bindings + binding_offset) %
          num_bindings;
      const bool is_closest = (bind_id % 3u) == 0u;
      if (is_closest) {
        if (std::memcmp(&a[i], &b[i], sizeof(luisa::float4)) != 0) {
          if (closest_diff < 8u) {
            std::cout << "    CLOSEST DIFF " << buf_name << " tid=" << tid
                      << " bind=" << bind_id << " sync=(" << a[i].x << ","
                      << a[i].y << "," << a[i].z << "," << a[i].w
                      << ") async=(" << b[i].x << "," << b[i].y << ","
                      << b[i].z << "," << b[i].w << ")\n";
          }
          ++closest_diff;
        }
      } else {
        bool row_bad = false;
        for (int c = 0; c < 4; ++c) {
          const double av = static_cast<double>((&a[i].x)[c]);
          const double bv = static_cast<double>((&b[i].x)[c]);
          const double diff = std::abs(av - bv);
          filtered_max_abs = std::max(filtered_max_abs, diff);
          const double tol =
              1e-4 * std::max(1.0, std::max(std::abs(av), std::abs(bv)));
          if (diff > tol) {
            row_bad = true;
          }
        }
        if (row_bad) {
          if (filtered_fail < 8u) {
            std::cout << "    FILTERED FAIL " << buf_name << " tid=" << tid
                      << " bind=" << bind_id << " sync=(" << a[i].x << ","
                      << a[i].y << "," << a[i].z << "," << a[i].w
                      << ") async=(" << b[i].x << "," << b[i].y << ","
                      << b[i].z << "," << b[i].w << ")\n";
          }
          ++filtered_fail;
        }
      }
    }
    require(closest_diff == 0u,
            tag + " " + buf_name + " closest 0-ULP memcmp failed vs sync");
    require(filtered_fail == 0u,
            tag + " " + buf_name + " filtered parity exceeded 1e-4 vs sync");
    return filtered_max_abs;
  };

  for (const auto prepack_mode : {LuisaVirtualTexturePrepackMode::off,
                                   LuisaVirtualTexturePrepackMode::memory}) {
    const std::string tag =
        prepack_mode == LuisaVirtualTexturePrepackMode::off ? "prepack=off"
                                                            : "prepack=mem_warm";
    const auto sync_res = run_vt_case(prepack_mode, false);
    const auto async_res = run_vt_case(prepack_mode, true);

    for (std::size_t i = 0u; i < kThreads; ++i) {
      for (int c = 0; c < 4; ++c) {
        require(std::isfinite((&async_res.main_out[i].x)[c]),
                tag + " non-finite main_out");
        require(std::isfinite((&async_res.aux_out[i].x)[c]),
                tag + " non-finite aux_out");
      }
    }
    std::cout << "  " << tag << " stats:\n"
              << "    sync : fault_batches=" << sync_res.stats.fault_batches
              << " aux_batches=" << sync_res.stats.aux_fault_batches
              << " loaded=" << sync_res.stats.tiles_loaded
              << " evicted=" << sync_res.stats.tiles_evicted << '\n'
              << "    async: async_batches=" << async_res.stats.async_batches
              << " fault_batches=" << async_res.stats.fault_batches
              << " aux_batches=" << async_res.stats.aux_fault_batches
              << " loaded=" << async_res.stats.tiles_loaded
              << " evicted=" << async_res.stats.tiles_evicted
              << " poll_ready=" << async_res.stats.async_poll_ready_hits
              << " wait_calls=" << async_res.stats.async_wait_calls
              << " aux_drains=" << async_res.stats.async_aux_drains << '\n';
    const double main_max_diff = check_f4_parity(
        tag, "main_out", sync_res.main_out, async_res.main_out, 0u);
    const double aux_max_diff = check_f4_parity(
        tag, "aux_out", sync_res.aux_out, async_res.aux_out, 13u);
    require(sync_res.indep_out.size() == async_res.indep_out.size() &&
                std::memcmp(sync_res.indep_out.data(), async_res.indep_out.data(),
                            sync_res.indep_out.size() * sizeof(luisa::uint)) == 0,
            tag + " indep_out memcmp parity failed vs sync");

    require(async_res.stats.async_batches > 0ull,
            tag + " expected async_batches > 0");
    require(async_res.stats.tiles_loaded > 0ull,
            tag + " expected tiles_loaded > 0");
    require(async_res.stats.tiles_evicted > 0ull,
            tag + " expected tiles_evicted > 0 under 1 MiB budget");
    std::cout << "    parity OK (closest=0-ULP, main_filtered_max_abs="
              << main_max_diff << ", aux_filtered_max_abs=" << aux_max_diff
              << ")\n";

    if (device.impl()->query("vk_batch_copy_cmd") == "1") {
      auto verify_cq = [&](LuisaVirtualTextureCopyQueueMode cq_mode,
                           bool async_w) {
        const std::string cq_name{
            luisa_virtual_texture_copy_queue_mode_name(cq_mode)};
        const std::string sub_tag =
            tag + " cq=" + cq_name + (async_w ? " async" : " sync");
        const auto cq_res = run_vt_case(prepack_mode, async_w, cq_mode);
        for (std::size_t i = 0u; i < kThreads; ++i) {
          for (int c = 0; c < 4; ++c) {
            require(std::isfinite((&cq_res.main_out[i].x)[c]),
                    sub_tag + " non-finite main_out");
            require(std::isfinite((&cq_res.aux_out[i].x)[c]),
                    sub_tag + " non-finite aux_out");
          }
        }
        require(cq_res.stats.active_copy_queue_mode == cq_mode,
                sub_tag + " active_copy_queue_mode mismatch");
        require(cq_res.stats.copy_queue_batches > 0ull,
                sub_tag + " expected copy_queue_batches > 0");
        if (cq_mode == LuisaVirtualTextureCopyQueueMode::staged ||
            cq_mode == LuisaVirtualTextureCopyQueueMode::staged_shader) {
          require(cq_res.stats.copy_queue_h2d_bytes > 0ull,
                  sub_tag + " expected copy_queue_h2d_bytes > 0");
        }
        if (cq_mode == LuisaVirtualTextureCopyQueueMode::direct ||
            cq_mode == LuisaVirtualTextureCopyQueueMode::staged) {
          require(cq_res.stats.copy_queue_atlas_bytes > 0ull,
                  sub_tag + " expected copy_queue_atlas_bytes > 0");
        }
        const double cq_main_diff = check_f4_parity(
            sub_tag, "cq_main_out", sync_res.main_out, cq_res.main_out, 0u);
        const double cq_aux_diff = check_f4_parity(
            sub_tag, "cq_aux_out", sync_res.aux_out, cq_res.aux_out, 13u);
        require(sync_res.indep_out.size() == cq_res.indep_out.size() &&
                    std::memcmp(sync_res.indep_out.data(),
                                cq_res.indep_out.data(),
                                sync_res.indep_out.size() *
                                    sizeof(luisa::uint)) == 0,
                sub_tag + " indep_out parity failed");
        std::cout << "    [" << sub_tag
                  << "] batches=" << cq_res.stats.copy_queue_batches
                  << " h2d_B=" << cq_res.stats.copy_queue_h2d_bytes
                  << " atlas_B=" << cq_res.stats.copy_queue_atlas_bytes
                  << " main_max_abs=" << cq_main_diff
                  << " aux_max_abs=" << cq_aux_diff << " (0-ULP OK)\n";
      };
      for (const auto cq_mode :
           {LuisaVirtualTextureCopyQueueMode::direct,
            LuisaVirtualTextureCopyQueueMode::staged,
            LuisaVirtualTextureCopyQueueMode::staged_shader}) {
        verify_cq(cq_mode, false);
        verify_cq(cq_mode, true);
      }
    }
  }

  {
    const std::string tag = "multi_dispatch(3x)";
    const auto sync3 = run_vt_case(
        LuisaVirtualTexturePrepackMode::memory, false,
        LuisaVirtualTextureCopyQueueMode::off, 3u);
    const auto async3 = run_vt_case(
        LuisaVirtualTexturePrepackMode::memory, true,
        LuisaVirtualTextureCopyQueueMode::off, 3u);
    const double m3 = check_f4_parity(
        tag, "main_out", sync3.main_out, async3.main_out, 0u);
    const double a3 = check_f4_parity(
        tag, "aux_out", sync3.aux_out, async3.aux_out, 13u);
    require(sync3.indep_out.size() == async3.indep_out.size() &&
                std::memcmp(sync3.indep_out.data(), async3.indep_out.data(),
                            sync3.indep_out.size() * sizeof(luisa::uint)) == 0,
            tag + " indep_out memcmp parity failed vs sync");
    require(async3.stats.async_batches >= 6ull,
            tag + " expected async_batches >= 6 across 3 dispatches");
    require(async3.stats.async_deferred_admissions == 0ull,
            tag + " expected async_deferred_admissions == 0 across 3 dispatches");
    std::cout << "  " << tag << ": async_batches=" << async3.stats.async_batches
              << " loaded=" << async3.stats.tiles_loaded
              << " evicted=" << async3.stats.tiles_evicted
              << " deferred_admissions=" << async3.stats.async_deferred_admissions
              << " main_max_abs=" << m3 << " aux_max_abs=" << a3
              << " (bit-exact OK)\n";
  }
}

void test_vt_cache_async_multichunk_aux_interleave_parity(Device &device,
                                                          Stream &stream) {
  std::cout << "[Async Test 3] VirtualTextureCache multi-chunk (>512 tiles) + aux interleave parity...\n";

  // 2347x1807 NPOT RGBA8 -> 40x31 = 1240 logical 60x60 tiles per image.
  // Two distinct images = 2480 tiles, forcing multi-chunk (>512) batches
  // and eviction interleaved with multi-chunk sync aux servicing.
  constexpr std::uint32_t img_w = 2347u;
  constexpr std::uint32_t img_h = 1807u;
  constexpr std::uint32_t tiles_x = 40u;
  constexpr std::uint32_t tiles_y = 31u;
  constexpr std::uint32_t tiles_per_img = tiles_x * tiles_y; // 1240

  auto make_unique_tile_image = [&](std::uint32_t salt) {
    std::vector<std::uint8_t> px(static_cast<std::size_t>(img_w) * img_h * 4u);
    for (std::uint32_t y = 0u; y < img_h; ++y) {
      const std::uint32_t ty = y / 60u;
      const std::uint32_t ly = y % 60u;
      for (std::uint32_t x = 0u; x < img_w; ++x) {
        const std::uint32_t tx = x / 60u;
        const std::uint32_t lx = x % 60u;
        const std::uint32_t tid = ty * tiles_x + tx;
        const std::size_t i = (static_cast<std::size_t>(y) * img_w + x) * 4u;
        px[i + 0u] = static_cast<std::uint8_t>((tid * 37u + salt * 19u + lx * 3u + 11u) & 0xffu);
        px[i + 1u] = static_cast<std::uint8_t>(((tid >> 3u) * 53u + salt * 41u + ly * 5u + 29u) & 0xffu);
        px[i + 2u] = static_cast<std::uint8_t>((tx * 17u + ty * 43u + salt * 7u + lx + ly + 71u) & 0xffu);
        px[i + 3u] = static_cast<std::uint8_t>(128u + ((tid + salt + lx * 7u + ly * 11u) & 0x7fu));
      }
    }
    return encode_png_rgba8(img_w, img_h, px);
  };

  const auto png_a = make_unique_tile_image(1u);
  const auto png_b = make_unique_tile_image(2u);
  const auto probe_a = probe_scene_image_info(png_a, "multichunk_a.png");
  const auto probe_b = probe_scene_image_info(png_b, "multichunk_b.png");
  require(probe_a.has_value() && probe_b.has_value(),
          "Test 3 failed to probe synthetic NPOT images");

  const std::vector<VtProbedSceneImage> probed_images{
      VtProbedSceneImage{.id = ImageId{1u},
                         .name = "multichunk_a.png",
                         .encoded_data = png_a,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *probe_a},
      VtProbedSceneImage{.id = ImageId{2u},
                         .name = "multichunk_b.png",
                         .encoded_data = png_b,
                         .alpha_type = ImageAlphaType::straight,
                         .color_space = ImageColorSpace::data,
                         .probed = *probe_b}};

  using compiler::cycles_svm::ImageBinding;
  using compiler::cycles_svm::ImageExtension;
  using compiler::cycles_svm::ImageInterpolation;
  // 0: img A closest, 1: img A linear, 2: img B closest, 3: img B linear
  const std::vector<ImageBinding> handles{
      {.resource_id = 1u, .interpolation = ImageInterpolation::closest, .extension = ImageExtension::repeat},
      {.resource_id = 1u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat},
      {.resource_id = 2u, .interpolation = ImageInterpolation::closest, .extension = ImageExtension::repeat},
      {.resource_id = 2u, .interpolation = ImageInterpolation::linear, .extension = ImageExtension::repeat}};
  std::vector<CyclesSvmImageBindingGpu> gpu_bindings;
  for (std::uint32_t i = 0u; i < 4u; ++i) {
    gpu_bindings.push_back(make_cycles_svm_image_binding(
        i, handles[i].interpolation, handles[i].extension));
  }

  BindlessArray dummy_direct_heap = device.create_bindless_array(4u);
  auto dummy_b4 = device.create_image<float>(PixelStorage::BYTE4, 4u, 4u);
  std::vector<std::uint8_t> d_b4(64u, 128u);
  for (std::uint32_t i = 0u; i < 4u; ++i) {
    dummy_direct_heap.emplace_on_update(
        i, dummy_b4,
        cycles_svm_texture_sampler(handles[i].interpolation, handles[i].extension));
  }
  stream << dummy_b4.copy_from(d_b4.data())
         << dummy_direct_heap.update() << synchronize();

  constexpr std::uint32_t kThreads = 8192u;
  constexpr std::uint32_t kPoolCap = 4096u;

  struct RunRes {
    VirtualTextureCacheStats stats{};
    std::vector<luisa::float4> main_out;
    std::vector<luisa::float4> aux_out;
  };

  auto run_case = [&](std::uint64_t budget_mb,
                      LuisaVirtualTexturePrepackMode prepack_mode,
                      bool async_worker,
                      LuisaVirtualTextureCopyQueueMode cq_mode =
                          LuisaVirtualTextureCopyQueueMode::off) -> RunRes {
    LuisaVirtualTextureCacheOptions opts{
        .enabled = true,
        .gpu_cache_budget_bytes = budget_mb * 1024ull * 1024ull,
        .physical_tile_size = 64u,
        .prepack_mode = prepack_mode,
        .host_prepacked_cache_budget_bytes = 64ull * 1024ull * 1024ull,
        .prepack_warm_on_init =
            (prepack_mode != LuisaVirtualTexturePrepackMode::off),
        .enable_async_worker = async_worker,
        .copy_queue_mode = cq_mode};

    RunRes res;
    res.main_out.resize(kThreads);
    res.aux_out.resize(kThreads);
    {
      auto vt = VirtualTextureCache::create(device, stream, opts, probed_images,
                                            handles, gpu_bindings);

      auto vt_wait_resident = [&](Expr<luisa::uint> page_id) noexcept {
        UInt missed_page = page_id;
        $loop {
          $suspend("vt_wait",
                   coro_stage(luisa::string{vt_extension_schema})
                       .read(luisa::string{vt_extension_binding}, missed_page)
                       .blocking_async()
                       .async_domain(luisa::string{vt_extension_schema}));
          $if(vt->is_resident(missed_page)) { $break; };
        };
      };

      auto sample_with_vt = [&](Expr<luisa::uint> b, Expr<luisa::float2> uv) {
        Float4 col = make_float4(0.0f);
        $loop {
          UInt miss = vt_invalid_page;
          col = vt->sample_2d_dynamic(Int(b), uv, dummy_direct_heap, &miss);
          $if(miss == vt_invalid_page) { $break; };
          vt_wait_resident(miss);
        };
        return col;
      };

      auto tile_uv = [&](Expr<luisa::uint> tile_idx, Expr<luisa::uint> seed) noexcept {
        const UInt t = tile_idx % tiles_per_img;
        const UInt tx = t % tiles_x;
        const UInt ty = t / tiles_x;
        const UInt lx = (seed * 13u + 7u) % 58u + 1u;
        const UInt ly = (seed * 29u + 11u) % 58u + 1u;
        const Float u =
            (cast<float>(tx * 60u + lx) + 0.5f) / static_cast<float>(img_w);
        const Float v =
            (cast<float>(ty * 60u + ly) + 0.5f) / static_cast<float>(img_h);
        return make_float2(u, v);
      };

      auto aux_work = std::make_shared<TestVtAuxWork>(
          device, kThreads, vt, dummy_direct_heap);

      Coroutine<void(Buffer<luisa::float4>, Buffer<luisa::float4>,
                     Buffer<luisa::uint>)>
          coro{[&](BufferFloat4 main_out, BufferFloat4 aux_out,
                   BufferUInt indep_out) {
            (void)aux_out;
            const auto tid = dispatch_x();
            const UInt local = tid % kPoolCap;
            const UInt tile_a = (tid * 17u + (tid / 64u)) % tiles_per_img;
            const UInt tile_b = (tid * 31u + 5u) % tiles_per_img;
            const Float2 uv_a = tile_uv(tile_a, tid);
            const Float2 uv_b = tile_uv(tile_b, tid ^ 0x5au);
            const UInt bind_a = tid & 1u;
            const UInt bind_b = 2u + (tid & 1u);

            // Majority enters after_vt_even first -> aux_work emits 2560 items on Image A while
            // continuations split into 1280-thread buckets (<2560), so sync aux_work wins and
            // services >1000 Image A tiles across 3 chunks (dirtying _b4_slot_fence[0..2]).
            // Next, threads converge onto Image B in multi-chunk async waves with eviction + refill aux interleave.
            $if(local < 2560u) {
              $suspend("after_vt_even");
              aux_work->emit(tid, uv_a, bind_a);
              $if((tid & 1u) == 0u) {
                $suspend("after_aux_even");
              }
              $else {
                $suspend("after_aux_odd");
              };
              const auto cb = sample_with_vt(bind_b, uv_b);
              main_out.write(tid, cb);
            }
            $else {
              UInt acc = def(tid * 13u + 3u);
              $suspend("indep_0");
              acc = acc * 3u + 7u;
              $suspend("indep_1");
              indep_out.write(tid, acc);
              const auto cb = sample_with_vt(bind_b, uv_b);
              main_out.write(tid, cb);
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
      stream << main_buf.copy_to(luisa::span{res.main_out})
             << aux_buf.copy_to(luisa::span{res.aux_out}) << synchronize();
      res.stats = vt->stats();
    }
    return res;
  };

  for (const std::uint64_t budget_mb : {24ull, 16ull}) {
    for (const auto prepack_mode : {LuisaVirtualTexturePrepackMode::off,
                                     LuisaVirtualTexturePrepackMode::memory}) {
      const std::string tag =
          "mb=" + std::to_string(budget_mb) + " " +
          (prepack_mode == LuisaVirtualTexturePrepackMode::off
               ? "prepack=off"
               : "prepack=mem_warm");
      const auto sync_res = run_case(budget_mb, prepack_mode, false);
      const auto async_res = run_case(budget_mb, prepack_mode, true);

      auto verify_buf = [&](const std::string &name,
                            const std::vector<luisa::float4> &a,
                            const std::vector<luisa::float4> &b) {
        std::size_t closest_diff = 0u;
        std::size_t linear_fail = 0u;
        double max_abs = 0.0;
        for (std::size_t i = 0u; i < a.size(); ++i) {
          const bool is_closest = (i & 1u) == 0u;
          for (int c = 0; c < 4; ++c) {
            require(std::isfinite((&b[i].x)[c]), tag + " " + name + " non-finite");
          }
          if (is_closest) {
            if (std::memcmp(&a[i], &b[i], sizeof(luisa::float4)) != 0) {
              if (closest_diff < 8u) {
                std::cout << "    MULTICHUNK CLOSEST DIFF " << name << " tid=" << i
                          << " sync=(" << a[i].x << "," << a[i].y << ","
                          << a[i].z << "," << a[i].w << ") async=(" << b[i].x
                          << "," << b[i].y << "," << b[i].z << "," << b[i].w
                          << ")\n";
              }
              ++closest_diff;
            }
          } else {
            for (int c = 0; c < 4; ++c) {
              const double diff = std::abs(static_cast<double>((&a[i].x)[c]) -
                                           static_cast<double>((&b[i].x)[c]));
              max_abs = std::max(max_abs, diff);
              if (diff > 1e-4) {
                ++linear_fail;
                break;
              }
            }
          }
        }
        require(closest_diff == 0u,
                tag + " " + name + " closest 0-ULP memcmp failed vs sync");
        require(linear_fail == 0u,
                tag + " " + name + " linear parity >1e-4 vs sync");
        return max_abs;
      };

      const double main_max = verify_buf("main_out", sync_res.main_out, async_res.main_out);
      const double aux_max = verify_buf("aux_out", sync_res.aux_out, async_res.aux_out);
      require(async_res.stats.async_batches > 0ull, tag + " expected async_batches > 0");
      require(async_res.stats.aux_fault_batches > 0ull, tag + " expected aux_fault_batches > 0");
      require(async_res.stats.tiles_loaded > 512ull, tag + " expected tiles_loaded > 512");
      std::cout << "  " << tag << ": batches=" << async_res.stats.fault_batches
                << " async=" << async_res.stats.async_batches
                << " aux=" << async_res.stats.aux_fault_batches
                << " loaded=" << async_res.stats.tiles_loaded
                << " evicted=" << async_res.stats.tiles_evicted
                << " main_max_abs=" << main_max << " aux_max_abs=" << aux_max
                << " (0-ULP OK)\n";

      if (budget_mb == 16ull &&
          device.impl()->query("vk_batch_copy_cmd") == "1") {
        for (const auto cq_mode :
             {LuisaVirtualTextureCopyQueueMode::direct,
              LuisaVirtualTextureCopyQueueMode::staged,
              LuisaVirtualTextureCopyQueueMode::staged_shader}) {
          for (const bool aw : {false, true}) {
            const auto cq_res =
                run_case(budget_mb, prepack_mode, aw, cq_mode);
            const double cq_main = verify_buf(
                "cq_main_out", sync_res.main_out, cq_res.main_out);
            const double cq_aux =
                verify_buf("cq_aux_out", sync_res.aux_out, cq_res.aux_out);
            require(cq_res.stats.copy_queue_batches > 0ull,
                    tag + " multichunk expected copy_queue_batches > 0");
            require(cq_res.stats.tiles_loaded > 512ull,
                    tag + " multichunk expected tiles_loaded > 512");
            std::cout << "    [" << tag << " cq="
                      << luisa_virtual_texture_copy_queue_mode_name(cq_mode)
                      << (aw ? " async" : " sync") << "] cq_batches="
                      << cq_res.stats.copy_queue_batches
                      << " loaded=" << cq_res.stats.tiles_loaded
                      << " main_max_abs=" << cq_main
                      << " aux_max_abs=" << cq_aux << " (0-ULP OK)\n";
          }
        }
      }
    }
  }
}

void test_coro_scheduler_online_timing_and_time_budget(Device &device,
                                                       Stream &stream,
                                                       std::string_view backend) {
  std::cout << "[Async Test 4] Online kernel timing + time-budget overlap scheduling...\n";
  constexpr std::uint32_t kPoolCap = 16384u;
  constexpr std::uint32_t kWaves = 10u;
  constexpr std::uint32_t kThreads = kPoolCap * kWaves;
  constexpr std::uint32_t kHeavyIters = 32768u;
  constexpr std::chrono::microseconds kAsyncSleep{2500};

  struct RunResult {
    std::vector<luisa::uint> output;
    std::uint64_t poll_pending_hits{0ull};
    std::uint64_t wait_calls{0ull};
    WavefrontCoroDispatchStats sched_stats;
  };

  auto run_mode = [&](bool async_mode, bool enable_online_timing,
                      bool enable_time_budget) -> RunResult {
    Coroutine<void(Buffer<luisa::uint>)> coro{[async_mode, kHeavyIters](BufferUInt output) {
      const auto tid = dispatch_x();
      const auto bucket = tid % 100u;
      $if(bucket < 65u) {
        UInt v = def(tid * 17u + 3u);
        $suspend("branch_async", coro_stage("test.async_rw")
                                     .read_write("val", v)
                                     .blocking_async(async_mode)
                                     .async_domain("test.async_rw_domain"));
        output.write(tid, v ^ 0x5a5a5a5au);
      }
      $elif(bucket < 71u) {
        UInt acc = def(tid * 29u + 7u);
        $suspend("branch_heavy");
        $for(k, kHeavyIters) {
          acc = (acc ^ (acc >> 13u)) * 1664525u + 1013904223u;
          acc = (acc ^ (acc >> 15u)) * 22695477u + 1u;
        };
        $suspend("branch_heavy_done");
        output.write(tid, acc + 19u);
      }
      $elif(bucket == 71u) {
        UInt acc = def(tid * 13u + 5u);
        $suspend("branch_tiny");
        $suspend("branch_tiny_done");
        output.write(tid, acc ^ 0x2468ace0u);
      }
      $else {
        UInt acc = def(tid * 31u + 11u);
        acc = acc * 3u + 7u;
        output.write(tid, acc ^ 0x13572468u);
      };
    }};

    WavefrontCoroSchedulerConfig cfg;
    cfg.thread_count = kPoolCap;
    cfg.execution_block_size = 64u;
    cfg.global_memory_soa = true;
    cfg.frame_buffer_compaction = true;
    cfg.report_stats = true;
    cfg.largest_continuation_first = true;
    cfg.incremental_continuation_counts = true;
    cfg.fused_continuation_counts = true;
    cfg.enable_online_timing = enable_online_timing;
    cfg.enable_time_budget_scheduling = enable_time_budget;

    auto ext_stats = std::make_shared<AsyncRwStats>();
    WavefrontCoroScheduler<Buffer<luisa::uint>> scheduler{device, coro, cfg};
    scheduler.register_extension_handler(
        stream, [async_mode, ext_stats, kAsyncSleep](
                    WavefrontCoroExtensionPrepareContext &ctx,
                    const WavefrontCoroExtensionStage &stage)
                    -> luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> {
          if (!stage.extension ||
              stage.extension->schema() != "test.async_rw") {
            return nullptr;
          }
          return luisa::make_unique<TestAsyncRwHandler>(
              ctx, stage, async_mode, ext_stats, kAsyncSleep);
        });

    auto out_buf = device.create_buffer<luisa::uint>(kThreads);
    std::vector<luisa::uint> host_out(kThreads, 0u);
    stream << out_buf.copy_from(luisa::span{host_out}) << commit();
    stream << scheduler(out_buf).dispatch(kThreads);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    RunResult res;
    res.output = std::move(host_out);
    res.poll_pending_hits = ext_stats->poll_pending_hits.load();
    res.wait_calls = ext_stats->wait_calls.load();
    res.sched_stats = scheduler.last_dispatch_stats();
    return res;
  };

  const auto sync_res = run_mode(false, false, false);
  const auto static_res = run_mode(true, true, false);
  const auto budget_res = run_mode(true, true, true);

  for (std::uint32_t tid = 0u; tid < kThreads; ++tid) {
    const std::uint32_t bucket = tid % 100u;
    luisa::uint expected = 0u;
    if (bucket < 65u) {
      expected = ((tid * 17u + 3u) * 13u + 0x12345u) ^ 0x5a5a5a5au;
    } else if (bucket < 71u) {
      luisa::uint acc = tid * 29u + 7u;
      for (std::uint32_t k = 0u; k < kHeavyIters; ++k) {
        acc = (acc ^ (acc >> 13u)) * 1664525u + 1013904223u;
        acc = (acc ^ (acc >> 15u)) * 22695477u + 1u;
      }
      expected = acc + 19u;
    } else if (bucket == 71u) {
      expected = (tid * 13u + 5u) ^ 0x2468ace0u;
    } else {
      expected = (tid * 31u + 11u) * 3u + 7u;
      expected ^= 0x13572468u;
    }
    require(sync_res.output[tid] == expected,
            "Test 4 sync reference value mismatch at tid=" + std::to_string(tid));
  }
  require(sync_res.output.size() == static_res.output.size() &&
              std::memcmp(sync_res.output.data(), static_res.output.data(),
                          sync_res.output.size() * sizeof(luisa::uint)) == 0,
          "Test 4 bit-exact parity failed between sync and static-12%");
  require(sync_res.output.size() == budget_res.output.size() &&
              std::memcmp(sync_res.output.data(), budget_res.output.data(),
                          sync_res.output.size() * sizeof(luisa::uint)) == 0,
          "Test 4 bit-exact parity failed between sync and time-budget");

  require(static_res.sched_stats.online_timing_active,
          "Test 4 static-12% expected online_timing_active == true");
  require(budget_res.sched_stats.online_timing_active,
          "Test 4 time-budget expected online_timing_active == true");
  if (backend == "vk") {
    require(budget_res.sched_stats.online_timing_hw_timestamps,
            "Test 4 expected online_timing_hw_timestamps == true on vk");
  }
  require(budget_res.sched_stats.online_sync_bubble_ms > 0.0,
          "Test 4 expected online_sync_bubble_ms > 0");
  require(budget_res.sched_stats.time_budget_overlap_dispatches > 0ull,
          "Test 4 expected time_budget_overlap_dispatches > 0");
  require(budget_res.sched_stats.time_budget_overlap_dispatches >
              static_res.sched_stats.time_budget_overlap_dispatches,
          "Test 4 expected time-budget overlap_dispatches > static-12%");

  std::cout << "  static12: overlap_dispatches="
            << static_res.sched_stats.time_budget_overlap_dispatches
            << " overlap_waits="
            << static_res.sched_stats.time_budget_overlap_waits
            << " poll_pending=" << static_res.poll_pending_hits
            << " wait=" << static_res.wait_calls << '\n';
  std::cout << "  time_budget: hw_ts="
            << (budget_res.sched_stats.online_timing_hw_timestamps ? 1 : 0)
            << " t_sync_ms=" << budget_res.sched_stats.online_sync_bubble_ms
            << " overlap_dispatches="
            << budget_res.sched_stats.time_budget_overlap_dispatches
            << " overlap_waits="
            << budget_res.sched_stats.time_budget_overlap_waits
            << " poll_pending=" << budget_res.poll_pending_hits
            << " wait=" << budget_res.wait_calls
            << " (bit-exact OK)\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::cout << "Running Psycles VT async overlap tests on backend: "
              << backend << '\n';

    test_coro_graph_blocking_async_overlap_and_rw_parity(device, stream);
    test_vt_cache_async_worker_parity_and_aux_drain(device, stream);
    test_vt_cache_async_multichunk_aux_interleave_parity(device, stream);
    test_coro_scheduler_online_timing_and_time_budget(device, stream, backend);

    std::cout << "ALL STAGE 3A & 3B ASYNC / COPY QUEUE TESTS PASSED on "
              << backend << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
