#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/path_tracer.h> through a Psycles Luisa target."
#endif

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <psycles/contract/render.h>
#include <psycles/luisa/path_trace_schema.h>

#include <luisa/runtime/device.h>

namespace psycles::luisa_backend {

inline constexpr auto luisa_wavefront_auto_tail_threshold =
    std::numeric_limits<std::uint32_t>::max();

enum class LuisaPathScheduler : std::uint8_t {
    megakernel,
    megakernel_per_sample,
    wavefront,
    wavefront_graph,
    wavefront_staged,
    persistent,
};

[[nodiscard]] constexpr std::string_view
luisa_path_scheduler_name(LuisaPathScheduler scheduler) noexcept {
    switch (scheduler) {
        case LuisaPathScheduler::megakernel:
            return "megakernel";
        case LuisaPathScheduler::megakernel_per_sample:
            return "megakernel-per-sample";
        case LuisaPathScheduler::wavefront:
            return "wavefront";
        case LuisaPathScheduler::wavefront_graph:
            return "wavefront-graph";
        case LuisaPathScheduler::wavefront_staged:
            return "wavefront-staged";
        case LuisaPathScheduler::persistent:
            return "persistent";
    }
    return {};
}

[[nodiscard]] constexpr std::optional<LuisaPathScheduler>
parse_luisa_path_scheduler(std::string_view name) noexcept {
    if (name == "megakernel") {
        return LuisaPathScheduler::megakernel;
    }
    if (name == "megakernel-per-sample") {
        return LuisaPathScheduler::megakernel_per_sample;
    }
    if (name == "wavefront") {
        return LuisaPathScheduler::wavefront;
    }
    if (name == "wavefront-graph") {
        return LuisaPathScheduler::wavefront_graph;
    }
    if (name == "wavefront-staged") {
        return LuisaPathScheduler::wavefront_staged;
    }
    if (name == "persistent") {
        return LuisaPathScheduler::persistent;
    }
    return std::nullopt;
}

[[nodiscard]] constexpr bool
valid_luisa_execution_block_size(std::uint32_t size) noexcept {
    // Mirrors the Luisa DSL workgroup contract, so invalid scheduler options
    // are rejected before shader AST construction reaches set_block_size().
    return size >= 32u && size <= 1024u && size % 32u == 0u;
}

[[nodiscard]] constexpr bool
valid_luisa_wavefront_execution_block_size(std::uint32_t size) noexcept {
    return valid_luisa_execution_block_size(size);
}

[[nodiscard]] constexpr bool
valid_luisa_persistent_scheduler_shape(std::uint32_t worker_count,
    std::uint32_t block_size,
    std::uint32_t fetch_size) noexcept {
  return worker_count != 0u && valid_luisa_execution_block_size(block_size) &&
           fetch_size != 0u &&
           static_cast<std::uint64_t>(block_size) * fetch_size <=
               std::numeric_limits<std::uint32_t>::max();
}

// The serial megakernel gives one invocation exclusive ownership of a pixel
// and loops over its sample batch. Every other mode launches the Cartesian
// product of pixels and samples and therefore requires race-free film writes.
[[nodiscard]] constexpr bool luisa_path_scheduler_uses_per_sample_dispatch(
    LuisaPathScheduler scheduler) noexcept {
    return scheduler != LuisaPathScheduler::megakernel;
}

// Scheduler policy and coroutine partitioning are orthogonal. Every
// coroutine scheduler consumes the same Cycles-stage CoroGraph so measured
// differences describe scheduling alone; the two megakernels contain no
// suspension points and remain the unsplit baselines.
[[nodiscard]] constexpr bool
luisa_path_scheduler_uses_cycles_stage_partition(
    LuisaPathScheduler scheduler) noexcept {
    switch (scheduler) {
        case LuisaPathScheduler::wavefront:
        case LuisaPathScheduler::wavefront_graph:
        case LuisaPathScheduler::wavefront_staged:
        case LuisaPathScheduler::persistent:
            return true;
        case LuisaPathScheduler::megakernel:
        case LuisaPathScheduler::megakernel_per_sample:
            return false;
    }
    return false;
}

struct LuisaPathTrace {
    using Slot = std::array<float, 4u>;

    std::uint32_t pixel_x{};
    // Cycles film convention: zero is the lower row of the full image.
    std::uint32_t pixel_y{};
    std::uint32_t sample{};
    std::array<Slot, path_trace_schema::slot_count> slots{};
};

class LuisaPathTraceSink {

public:
    virtual ~LuisaPathTraceSink() noexcept = default;
    virtual void write(const LuisaPathTrace &trace) = 0;
};

struct LuisaPathTraceRequest {
    std::uint32_t pixel_x{};
    // Cycles film convention: zero is the lower row of the full image.
    std::uint32_t pixel_y{};
    std::uint32_t sample{};
    std::shared_ptr<LuisaPathTraceSink> sink;
};

// Hit-weighted diagnostic over the post-population ShaderData-equivalent
// closure count. Cycles and Psycles both cap the retained surface closure
// sequence at 64, so the map is injective: bin i records exactly count == i.
// This is intentionally not a material census; one event is recorded for
// every surface-shading invocation that reaches closure population.
inline constexpr std::uint32_t luisa_surface_closure_count_histogram_max = 64u;
inline constexpr std::size_t luisa_surface_closure_count_histogram_bin_count =
    static_cast<std::size_t>(luisa_surface_closure_count_histogram_max) + 1u;

struct LuisaSurfaceClosureCountHistogram {
    std::array<std::uint64_t, luisa_surface_closure_count_histogram_bin_count> counts{};
    // Device counters are four-way sharded IEEE-754 floats. `exact` is false
    // if any shard reached the largest consecutive integer or contained an
    // invalid/non-integral value; callers must then reject quantitative use.
    bool exact{};
};

class LuisaSurfaceClosureCountHistogramSink {

public:
    virtual ~LuisaSurfaceClosureCountHistogramSink() noexcept = default;
    virtual void write(const LuisaSurfaceClosureCountHistogram &histogram) = 0;
};

struct LuisaSurfaceClosureCountHistogramRequest {
    std::shared_ptr<LuisaSurfaceClosureCountHistogramSink> sink;
};

// Hit-weighted execution census for the compact populate-once surface SVM.
// A device invocation records only its topology tag. The host then projects
// that exact count through the topology's immutable preparation program, so
// the reported value-handler executions require one atomic per surface hit,
// not one atomic per bytecode instruction.
struct LuisaSurfaceValueHandlerExecutionCount {
  std::uint32_t variant_index{};
  std::uint32_t handler_key{};
  std::uint32_t operation{};
  std::uint32_t result_bank{};
  std::uint32_t svm_immediate{};
  std::uint64_t executions{};

  auto operator<=>(
      const LuisaSurfaceValueHandlerExecutionCount &) const noexcept = default;
};

struct LuisaSurfaceClosureLeafVisitCount {
  std::uint32_t static_variant{};
  std::uint32_t operation{};
  std::uint64_t visits{};

  auto operator<=>(const LuisaSurfaceClosureLeafVisitCount &) const noexcept =
      default;
};

struct LuisaSurfaceProgramExecutionHistogram {
  std::vector<std::uint64_t> topology_surface_populations;
  std::vector<LuisaSurfaceValueHandlerExecutionCount> value_handlers;
  std::uint64_t value_instruction_executions{};
  std::uint64_t surface_normal_transition_executions{};
  // Dense SurfaceClosureInstructionKind order: leaf, mix_both, mix_left,
  // mix_right. These are first-traversal instruction visits. A zero-weight
  // leaf is visited but does not invoke its decoder, so leaf visits are not
  // mislabeled as physical-closure allocations.
  std::array<std::uint64_t, 4u> closure_instruction_kind_visits{};
  std::vector<LuisaSurfaceClosureLeafVisitCount> closure_leaf_variants;
  std::uint64_t closure_instruction_visits{};
  // False means a float counter shard exceeded its consecutive-integer
  // domain, host projection overflowed, or the scene did not use the compact
  // populate-once execution contract required by the projection.
  bool exact{};

  auto operator<=>(
      const LuisaSurfaceProgramExecutionHistogram &) const noexcept = default;
};

class LuisaSurfaceProgramExecutionHistogramSink {

public:
  virtual ~LuisaSurfaceProgramExecutionHistogramSink() noexcept = default;
  virtual void
  write(const LuisaSurfaceProgramExecutionHistogram &histogram) = 0;
};

struct LuisaSurfaceProgramExecutionHistogramRequest {
  std::shared_ptr<LuisaSurfaceProgramExecutionHistogramSink> sink;
};

struct LuisaPathTracerOptions {
    bool next_event_estimation{true};
    // Cycles' HIP kernels are compiled with fast math and explicitly select
    // device fast transcendental functions. Keep that as the production
    // policy for every Psycles path-kernel scheduler; false is the strict
    // diagnostic baseline. This is a host/JIT choice, never a device branch.
    bool enable_fast_math{true};
    // All three modes record the same path program. The scheduler is a host-
    // side execution policy and is never inferred from the backend name, so a
    // GPU can use the megakernel baseline and fallback can exercise both
    // coroutine schedulers as well.
  LuisaPathScheduler scheduler{LuisaPathScheduler::megakernel};
    // Maximum number of live global frames in the wavefront scheduler. More
    // logical pixel/sample instances are admitted by backpressured refills.
    // Cycles' production wavefront benchmark admits at most 2^20 path states;
    // this is also a practical 848 MiB frame budget at the current 848-byte
    // Psycles frame. Callers can still request the GPU Coroutines paper's
    // 2^24 configuration explicitly when device memory permits it.
    std::uint32_t wavefront_frame_capacity{1u << 20u};
    // Host/JIT choice for Luisa's thread-local wavefront generate/resume
    // kernels. Unlike frame capacity, this is part of shader structure.
    std::uint32_t wavefront_execution_block_size{32u};
    // Runtime grid-stride lane ceiling for graph-wavefront consumers. Zero
    // launches one lane per active frame; nonzero values do not specialize
    // the shader and are intended for scheduler-policy sweeps.
    std::uint32_t wavefront_graph_worker_count{0u};
    // Exact one-continuation-at-a-time graph scheduling. This currently uses
    // one counter snapshot per action; delayed Markov prediction is a separate
    // policy so approximation can never enter correctness decisions.
    bool wavefront_graph_selective_scheduling{false};
    std::uint32_t wavefront_graph_refill_threshold{0u};
    // Graph-wavefront counter snapshots are accumulated on device and copied
    // to the host in contiguous batches. Batch size and pipeline depth are
    // runtime scheduling policy and do not specialize the path shaders.
    std::uint32_t wavefront_counter_readback_batch_size{4u};
    std::uint32_t wavefront_counter_readback_pipeline_depth{2u};
    // Once all logical samples have been generated, finish a residual set no
    // larger than this in one CoroGraph-derived state-machine kernel. The
    // automatic sentinel resolves from active dispatch capacity on the host;
    // zero disables and avoids compiling the optional tail kernel.
    std::uint32_t wavefront_tail_megakernel_threshold{
        luisa_wavefront_auto_tail_threshold};
    // Host/JIT policy for ordering the staged shade_surface queue by the
    // structure-deduplicated SurfaceDispatch tag. False preserves the same
    // continuation cuts without recording a key resolver, frame export, or
    // sorting kernel, providing a matched scheduler-only baseline.
    bool staged_surface_sorting{true};
  // Experimental host/JIT policy that splits reduced direct-light visibility
  // work from shade_surface. Both modes use the same PathKernelPipeline and
  // evaluator. It remains opt-in because the current HIP side-queue launch is
  // slower than inline shadow evaluation on the measured production scene.
    bool staged_direct_light_queue{false};
    // Persistent workers are independent of the logical image size. The
    // scheduler rounds this count up to a complete block.
    // The paper's persistent configuration is 2^15 workers, 128 threads per
    // block, and 16 block-sized batches fetched per acquisition.
    std::uint32_t persistent_worker_count{1u << 15u};
    // Kept backend-independent: fallback executes the same block scheduler
    // instead of being silently normalized to a scalar special case. The
    // single-wave default is an application tuning choice; callers retain
    // control because the best occupancy depends on the coroutine frame and
    // continuation resource footprint.
    std::uint32_t persistent_block_size{32u};
    // Runtime task-allocation granularity, measured in block-sized batches.
    // It does not enter shader AST/cache identity.
    std::uint32_t persistent_fetch_size{1u};
    bool persistent_shared_memory_soa{true};
    bool persistent_global_memory_extension{true};
    // Each batch is submitted and synchronized independently. Per-sample
    // schedulers launch this as dispatch.z; the serial megakernel uses the
    // same runtime value as its loop bound. It is never part of shader AST or
    // cache identity. Setting it to one is the deterministic diagnostic path:
    // each pixel has only one atomic contributor between stream barriers, so
    // exact hashes remain meaningful. Zero is invalid.
    std::uint32_t max_samples_per_dispatch{64u};
    // Upper bound on pixel/sample work submitted in one kernel dispatch.
    // Backends without a practical watchdog use the unbounded default;
    // Metal and Vulkan apply a conservative device-safe cap.
    std::uint32_t max_pixel_samples_per_dispatch{
        std::numeric_limits<std::uint32_t>::max()};
    // Diagnostic-only, observational trace. The kernel writes this fixed
    // schema only for the requested full-film pixel and absolute sample.
    std::optional<LuisaPathTraceRequest> path_trace;
    // Diagnostic-only, hit-weighted count distribution. Presence is a
    // host/JIT specialization: production shaders contain neither the atomic
    // write nor a device-side enable branch. The sink receives cumulative
    // counts after each completed render_samples() call.
    std::optional<LuisaSurfaceClosureCountHistogramRequest> surface_closure_count_histogram;
    // Diagnostic-only exact topology census plus host projection through the
    // immutable compact preparation program. Presence specializes the host/JIT
    // build; production shaders contain neither its atomic nor an enable flag.
    std::optional<LuisaSurfaceProgramExecutionHistogramRequest>
        surface_program_execution_histogram;
};

class LuisaPathTracerBackend final : public contract::RendererBackend {

private:
    luisa::compute::Device _device;
    LuisaPathTracerOptions _options;

public:
  explicit LuisaPathTracerBackend(luisa::compute::Device device,
        LuisaPathTracerOptions options = {}) noexcept;

  [[nodiscard]] contract::SceneCompilation
  compile_scene(const contract::SceneSnapshot &snapshot) override;

  [[nodiscard]] std::unique_ptr<contract::RenderSession>
  create_session(const contract::CompiledScene &scene,
        const contract::RenderSettings &settings) override;
};

}// namespace psycles::luisa_backend
