#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/coro/schedulers/wavefront_extension.h>

namespace psycles::luisa_backend::detail {

struct VtProfileBatchRecord {
  std::uint64_t batch_id{0ull};
  const char *source{"unknown"};
  std::uint64_t stream_handle{0ull};
  std::uint32_t queued_fault_frames{0u};
  std::uint32_t tiles_loaded{0u};
  std::uint32_t tiles_evicted{0u};
  std::uint32_t b4_loaded{0u};
  std::uint32_t f4_loaded{0u};
  std::uint32_t chunks_submitted{0u};
  std::uint32_t pt_updates{0u};
  std::uint32_t ep_updates{0u};
  bool has_cold_decode{false};
  bool fused_final_chunk{false};
  bool isolated_pub{false};
  bool async_worker{false};
  std::uint32_t async_slot_index{0u};
  std::uint64_t async_generation{0ull};
  std::uint32_t in_flight_at_submit{0u};
  std::uint64_t cpu_enter_ns{0ull};
  std::uint64_t publish_enter_ns{0ull};
  std::uint64_t cpu_exit_ns{0ull};
  double cpu_collect_record_ms{0.0};
  double cpu_readback_sync_ms{0.0};
  double cpu_readback_flush_ms{0.0};
  double cpu_admission_rank_ms{0.0};
  double cpu_decode_ms{0.0};
  double cpu_pack_ms{0.0};
  double cpu_upload_submit_ms{0.0};
  double cpu_page_table_pub_ms{0.0};
  double cpu_wait_async_ms{0.0};
  double cpu_fault_to_submit_ready_ms{0.0};
  std::uint64_t submit_seq_before_s1{0ull};
  std::uint64_t submit_seq_after_s1{0ull};
  std::vector<std::uint64_t> chunk_submit_seqs;
  std::uint64_t submit_seq_after_service{0ull};
  luisa::compute::coro::WavefrontCoroSchedulerSnapshot snapshot{};
};

struct VtProfileWorkerEvent {
  std::uint64_t batch_id{0ull};
  std::uint64_t start_ns{0ull};
  std::uint64_t end_ns{0ull};
  const char *cat{"worker"};
  std::string name;
};

struct VtProfilePhaseEvent {
  std::uint64_t start_ns{0ull};
  std::uint64_t end_ns{0ull};
  const char *cat{"preflight"};
  std::string name;
  std::string detail;
};

class VtProfileRecorder {
private:
  std::string _path;
  bool _isolate_pub{false};
  mutable std::mutex _mtx;
  std::uint64_t _next_batch_id{1ull};
  std::uint64_t _empty_fault_checks{0ull};
  std::vector<VtProfileBatchRecord> _batches;
  std::vector<VtProfileWorkerEvent> _worker_events;
  std::vector<VtProfilePhaseEvent> _preflight_events;
  std::vector<VtProfilePhaseEvent> _pso_compile_events;

  static void write_json_escaped(std::ofstream &ofs, std::string_view s) {
    ofs << '"';
    for (const char ch : s) {
      switch (ch) {
        case '"':
          ofs << "\\\"";
          break;
        case '\\':
          ofs << "\\\\";
          break;
        case '\n':
          ofs << "\\n";
          break;
        case '\r':
          ofs << "\\r";
          break;
        case '\t':
          ofs << "\\t";
          break;
        default:
          ofs << ch;
          break;
      }
    }
    ofs << '"';
  }

public:
  VtProfileRecorder(std::string path, bool isolate_pub) noexcept
      : _path{std::move(path)}, _isolate_pub{isolate_pub} {}

  [[nodiscard]] static std::unique_ptr<VtProfileRecorder>
  create_from_env() noexcept {
    const char *raw = std::getenv("PSYCLES_VT_PROFILE_JSON");
    if (raw == nullptr || raw[0] == '\0') {
      return nullptr;
    }
    const char *iso_raw = std::getenv("PSYCLES_VT_PROFILE_ISOLATE_PUB");
    const bool isolate_pub =
        iso_raw != nullptr && std::string_view{iso_raw} == "1";
    return std::make_unique<VtProfileRecorder>(std::string{raw}, isolate_pub);
  }

  [[nodiscard]] static std::uint64_t steady_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  [[nodiscard]] bool isolate_pub() const noexcept { return _isolate_pub; }

  void record_empty_check() noexcept {
    std::lock_guard lock{_mtx};
    ++_empty_fault_checks;
  }

  std::uint64_t record_batch(VtProfileBatchRecord &&rec) noexcept {
    std::lock_guard lock{_mtx};
    rec.batch_id = _next_batch_id++;
    const auto id = rec.batch_id;
    _batches.push_back(std::move(rec));
    return id;
  }

  void record_worker_events(
      std::uint64_t batch_id,
      std::vector<VtProfileWorkerEvent> &&events) noexcept {
    if (events.empty()) {
      return;
    }
    std::lock_guard lock{_mtx};
    for (auto &ev : events) {
      ev.batch_id = batch_id;
      _worker_events.push_back(std::move(ev));
    }
  }

  void record_preflight_event(std::uint64_t start_ns, std::uint64_t end_ns,
                              std::string name,
                              std::string detail = {}) noexcept {
    std::lock_guard lock{_mtx};
    _preflight_events.push_back(VtProfilePhaseEvent{
        .start_ns = start_ns,
        .end_ns = end_ns,
        .cat = "preflight",
        .name = std::move(name),
        .detail = std::move(detail)});
  }

  void record_pso_compile_event(std::uint64_t start_ns, std::uint64_t end_ns,
                                std::string name,
                                std::string detail = {}) noexcept {
    std::lock_guard lock{_mtx};
    _pso_compile_events.push_back(VtProfilePhaseEvent{
        .start_ns = start_ns,
        .end_ns = end_ns,
        .cat = "pso",
        .name = std::move(name),
        .detail = std::move(detail)});
  }

  void write_json() const noexcept {
    std::lock_guard lock{_mtx};
    if (_path.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::path p{_path};
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::ofstream ofs{_path, std::ios::binary | std::ios::trunc};
    if (!ofs.is_open()) {
      return;
    }
    ofs << "{\n"
        << "  \"isolate_pub\": " << (_isolate_pub ? "true" : "false") << ",\n"
        << "  \"empty_fault_checks\": " << _empty_fault_checks << ",\n"
        << "  \"batch_count\": " << _batches.size() << ",\n"
        << "  \"batches\": [\n";
    for (std::size_t i = 0u; i < _batches.size(); ++i) {
      const auto &b = _batches[i];
      ofs << "    {\n"
          << "      \"batch_id\": " << b.batch_id << ",\n"
          << "      \"source\": \"" << b.source << "\",\n"
          << "      \"stream_handle\": " << b.stream_handle << ",\n"
          << "      \"queued_fault_frames\": " << b.queued_fault_frames << ",\n"
          << "      \"tiles_loaded\": " << b.tiles_loaded << ",\n"
          << "      \"tiles_evicted\": " << b.tiles_evicted << ",\n"
          << "      \"b4_loaded\": " << b.b4_loaded << ",\n"
          << "      \"f4_loaded\": " << b.f4_loaded << ",\n"
          << "      \"chunks_submitted\": " << b.chunks_submitted << ",\n"
          << "      \"pt_updates\": " << b.pt_updates << ",\n"
          << "      \"ep_updates\": " << b.ep_updates << ",\n"
          << "      \"has_cold_decode\": " << (b.has_cold_decode ? "true" : "false") << ",\n"
          << "      \"fused_final_chunk\": " << (b.fused_final_chunk ? "true" : "false") << ",\n"
          << "      \"isolated_pub\": " << (b.isolated_pub ? "true" : "false") << ",\n"
          << "      \"async_worker\": " << (b.async_worker ? "true" : "false") << ",\n"
          << "      \"async_slot_index\": " << b.async_slot_index << ",\n"
          << "      \"async_generation\": " << b.async_generation << ",\n"
          << "      \"in_flight_at_submit\": " << b.in_flight_at_submit << ",\n"
          << "      \"cpu_enter_ns\": " << b.cpu_enter_ns << ",\n"
          << "      \"publish_enter_ns\": " << b.publish_enter_ns << ",\n"
          << "      \"cpu_exit_ns\": " << b.cpu_exit_ns << ",\n"
          << "      \"cpu_collect_record_ms\": " << b.cpu_collect_record_ms << ",\n"
          << "      \"cpu_readback_sync_ms\": " << b.cpu_readback_sync_ms << ",\n"
          << "      \"cpu_readback_flush_ms\": " << b.cpu_readback_flush_ms << ",\n"
          << "      \"cpu_admission_rank_ms\": " << b.cpu_admission_rank_ms << ",\n"
          << "      \"cpu_decode_ms\": " << b.cpu_decode_ms << ",\n"
          << "      \"cpu_pack_ms\": " << b.cpu_pack_ms << ",\n"
          << "      \"cpu_upload_submit_ms\": " << b.cpu_upload_submit_ms << ",\n"
          << "      \"cpu_page_table_pub_ms\": " << b.cpu_page_table_pub_ms << ",\n"
          << "      \"cpu_wait_async_ms\": " << b.cpu_wait_async_ms << ",\n"
          << "      \"cpu_fault_to_submit_ready_ms\": " << b.cpu_fault_to_submit_ready_ms << ",\n"
          << "      \"submit_seq_before_s1\": " << b.submit_seq_before_s1 << ",\n"
          << "      \"submit_seq_after_s1\": " << b.submit_seq_after_s1 << ",\n"
          << "      \"chunk_submit_seqs\": [";
      for (std::size_t ci = 0u; ci < b.chunk_submit_seqs.size(); ++ci) {
        if (ci != 0u) {
          ofs << ", ";
        }
        ofs << b.chunk_submit_seqs[ci];
      }
      ofs << "],\n"
          << "      \"submit_seq_after_service\": " << b.submit_seq_after_service << ",\n"
          << "      \"scheduler_snapshot\": {\n"
          << "        \"valid\": " << (b.snapshot.valid ? "true" : "false") << ",\n"
          << "        \"iteration\": " << b.snapshot.iteration << ",\n"
          << "        \"active_frames\": " << b.snapshot.active_frames << ",\n"
          << "        \"deferred_frames\": " << b.snapshot.deferred_frames << ",\n"
          << "        \"non_deferred_frames\": " << b.snapshot.non_deferred_frames << ",\n"
          << "        \"non_deferred_largest\": " << b.snapshot.non_deferred_largest << ",\n"
          << "        \"deferred_blocker_frames\": " << b.snapshot.deferred_blocker_frames << ",\n"
          << "        \"auxiliary_ready_frames\": " << b.snapshot.auxiliary_ready_frames << ",\n"
          << "        \"unspawned_threads\": " << b.snapshot.unspawned_threads << ",\n"
          << "        \"empty_frame_slots\": " << b.snapshot.empty_frame_slots << ",\n"
          << "        \"active_frame_capacity\": " << b.snapshot.active_frame_capacity << ",\n"
          << "        \"refill_threshold\": " << b.snapshot.refill_threshold << ",\n"
          << "        \"deferred_backpressure\": " << (b.snapshot.deferred_backpressure ? "true" : "false") << "\n"
          << "      }\n"
          << "    }" << (i + 1u < _batches.size() ? "," : "") << "\n";
    }
    ofs << "  ],\n"
        << "  \"worker_events\": [\n";
    for (std::size_t wi = 0u; wi < _worker_events.size(); ++wi) {
      const auto &ev = _worker_events[wi];
      ofs << "    {\"batch_id\": " << ev.batch_id
          << ", \"start_ns\": " << ev.start_ns
          << ", \"end_ns\": " << ev.end_ns
          << ", \"cat\": ";
      write_json_escaped(ofs, ev.cat != nullptr ? ev.cat : "worker");
      ofs << ", \"name\": ";
      write_json_escaped(ofs, ev.name);
      ofs << "}" << (wi + 1u < _worker_events.size() ? "," : "") << "\n";
    }
    auto write_phase_array = [&](std::string_view key,
                                 const std::vector<VtProfilePhaseEvent> &evs,
                                 bool trailing_comma) {
      ofs << "  ],\n"
          << "  \"" << key << "\": [\n";
      for (std::size_t pi = 0u; pi < evs.size(); ++pi) {
        const auto &ev = evs[pi];
        ofs << "    {\"start_ns\": " << ev.start_ns
            << ", \"end_ns\": " << ev.end_ns
            << ", \"cat\": ";
        write_json_escaped(ofs, ev.cat != nullptr ? ev.cat : "preflight");
        ofs << ", \"name\": ";
        write_json_escaped(ofs, ev.name);
        ofs << ", \"detail\": ";
        write_json_escaped(ofs, ev.detail);
        ofs << "}" << (pi + 1u < evs.size() ? "," : "") << "\n";
      }
      ofs << (trailing_comma ? "  ]\n" : "");
    };
    write_phase_array("preflight_events", _preflight_events, false);
    write_phase_array("pso_compile_events", _pso_compile_events, false);
    ofs << "  ]\n}\n";
    ofs.flush();
  }
};

} // namespace psycles::luisa_backend::detail
