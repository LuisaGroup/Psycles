#include "path_kernel_executor.h"

#include "path_kernel_builder.h"
#include "path_kernel_caustics_queue.h"
#include "path_kernel_direct_light_queue.h"
#include "path_kernel_surface_buckets.h"
#include "path_kernel_surface_queue.h"
#include "path_kernel_transitions.h"
#include "cycles_surface_sort.h"
#include "sample_dispatch_partition.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_cache_profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#if defined(_M_X64) || defined(__x86_64__)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/schedulers/graph_wavefront.h>
#include <luisa/coro/schedulers/persistent_threads.h>
#include <luisa/coro/schedulers/wavefront.h>

namespace psycles::luisa_backend::detail {
namespace {

static_assert(
    luisa_wavefront_auto_tail_threshold ==
    luisa::compute::coro::graph_wavefront_auto_tail_threshold);

template <typename Signature> struct RenderSchedulerTypes;

template <typename... Args> struct RenderSchedulerTypes<void(Args...)> {
    using Base = luisa::compute::coro::CoroScheduler<Args...>;
  using Wavefront = luisa::compute::coro::WavefrontCoroScheduler<Args...>;
    using GraphWavefront =
        luisa::compute::coro::GraphWavefrontCoroScheduler<Args...>;
    using Persistent =
        luisa::compute::coro::PersistentThreadsCoroScheduler<Args...>;
};

using RenderSchedulers = RenderSchedulerTypes<RenderKernelSignature>;

void log_coroutine_frame_if_requested(
    const RenderCoroutine &coroutine) noexcept {
  if (std::getenv("PSYCLES_DUMP_COROUTINE_FRAME") != nullptr) {
    LUISA_INFO("Psycles coroutine frame layout:\n{}",
               coroutine.frame().dump());
  }
}

[[nodiscard]] PathKernelConfig make_local_shadow_path(
    const PathKernelConfig &path, bool enable_vt = false) {
    auto scheduled_path = path;
    // The traversal returns a complete bounded batch. Coro lowering owns its
    // cross-stage lifetime; no second invocation-indexed hit buffer is needed.
    auto shadow = make_shadow_trace_callables(
        scheduled_path.scene, scheduled_path.light_transport.safe_normalize, {},
        enable_vt);
    scheduled_path.intersect_shadow = std::move(shadow.intersect);
    scheduled_path.shade_shadow_surface = std::move(shadow.shade_surface);
    scheduled_path.trace_shadow = std::move(shadow.trace);
    return scheduled_path;
}

void validate_vt_extension_abi(const RenderCoroutine &coroutine) noexcept {
    const auto *tex_node =
        coroutine.graph().node_by_name(path_transition::texture_cache);
    if (tex_node == nullptr) {
        return;
    }
    bool matched = false;
    for (const auto &boundary : coroutine.graph().boundaries()) {
        if (boundary.to_index != tex_node->index) {
            continue;
        }
        for (const auto &ext : boundary.extensions) {
            if (!ext || ext->schema() != vt_extension_schema) {
                continue;
            }
            for (const auto &desc : ext->bindings()) {
                if (desc.name == vt_extension_binding &&
                    desc.index < boundary.bindings.size()) {
                    const auto &binding = boundary.bindings[desc.index];
                    LUISA_ASSERT(
                        binding.type() ==
                                luisa::compute::Type::of<luisa::uint>() &&
                            binding.readable(),
                        "Invalid VT cache extension '{}' binding '{}'.",
                        vt_extension_schema, vt_extension_binding);
                    matched = true;
                }
            }
        }
    }
    LUISA_ASSERT(matched,
                 "VT texture_cache continuation did not bind '{}' '{}'.",
                 vt_extension_schema, vt_extension_binding);
}

template<typename Program>
[[nodiscard]] auto
bind_path_program(Program &program,
    const PathKernelDispatch &dispatch) noexcept {
  return program(dispatch.combined, dispatch.normal, dispatch.albedo,
                 dispatch.light_passes, dispatch.sample_count,
                 dispatch.volume_guiding_raw, dispatch.volume_guiding_denoised,
                 dispatch.path_trace, dispatch.sample_first, dispatch.samples,
                 dispatch.sobol_table, dispatch.filter_table,
               dispatch.parameters);
}

void validate_surface_queue_hint_abi(
    const RenderCoroutine &coroutine) noexcept {
    const auto *surface_node =
      coroutine.graph().node_by_name(path_transition::shade_surface);
  LUISA_ASSERT(surface_node != nullptr && surface_node->index != 0u &&
            surface_node->index < coroutine.subroutine_count(),
        "Staged path pipeline did not materialize '{}' as a "
        "non-entry continuation.",
        path_transition::shade_surface);
    const auto *hint_field =
      coroutine.frame().field(path_transition::scheduler_hint);
  LUISA_ASSERT(hint_field != nullptr &&
            hint_field->type == luisa::compute::Type::of<luisa::uint>(),
        "Staged path pipeline must explicitly export a uint '{}' frame "
        "field at the surface queue boundary.",
        path_transition::scheduler_hint);
}

template<typename SchedulerConfig>
[[nodiscard]] bool configure_surface_queue_hint(
    const PathKernelConfig &path, const RenderCoroutine &coroutine,
    SchedulerConfig &scheduler_config) noexcept {
    const auto surface_count = surface_queue_key_range(*path.scene);
    const auto enabled = path.staged_surface_sorting &&
                         surface_count != 0u &&
                         (!path.scene->geometries.empty() ||
                          !path.scene->curve_geometries.empty());
    if (!enabled) { return false; }
    validate_surface_queue_hint_abi(coroutine);
    scheduler_config.hint_range = surface_count;
    scheduler_config.hint_fields = {path_transition::shade_surface};
    scheduler_config.hint_partition_size = cycles_surface_sort_partition_size(
        scheduler_config.thread_count, surface_count);
    return true;
}

void validate_cycles_stage_partition_abi(
    const RenderCoroutine &coroutine,
    const PathKernelSceneStagePlan &plan,
    const PathSurfaceBucketPlan &surface_buckets,
    bool has_volume,
    bool has_subsurface,
    bool has_split_shadow_path) noexcept {
    const auto require = [&](luisa::string_view name,
                             bool reachable) noexcept {
        const auto *node = coroutine.graph().node_by_name(name);
        LUISA_ASSERT(
            !reachable ||
                (node != nullptr && node->index != 0u &&
                 node->index < coroutine.subroutine_count()),
            "Cycles-wavefront path pipeline did not materialize reachable "
            "stage '{}' as a non-entry continuation.",
            name);
        LUISA_ASSERT(
            reachable || node == nullptr,
            "Cycles-wavefront path pipeline materialized unreachable stage "
            "{}'. Host-stage reachability and the coroutine graph disagree.",
            name);
    };
    const bool has_surfaces = !plan.traversal.primitives.empty();
    require(path_transition::intersect_closest, true);
    require(path_transition::shade_volume, has_volume);
    require(path_transition::shade_light_forward,
            plan.analytic_light_endpoints);
    require(path_transition::shade_background, true);
    for (unsigned b = 0u; b < path_transition::max_surface_buckets; ++b) {
        require(path_transition::shade_surface_bucket_name(b),
                has_surfaces && b < surface_buckets.buckets.size());
    }
    require(path_transition::shade_surface_post,
            has_surfaces && surface_buckets.post_closure_substage);
    require(path_transition::shade_light_nee,
            has_split_shadow_path);
    require(path_transition::intersect_shadow,
            has_split_shadow_path);
    require(path_transition::shade_shadow,
            has_split_shadow_path);
    require(path_transition::intersect_subsurface,
            has_subsurface && has_surfaces &&
                (!surface_buckets.native_specialized ||
                 surface_buckets.any_may_have_bssrdf()));
    LUISA_ASSERT(
        coroutine.graph().node_by_name(path_transition::path_bounce) == nullptr &&
            coroutine.graph().node_by_name(
                path_transition::surface_shading) == nullptr,
        "Cycles-stage graph must not retain legacy compact transitions.");
}

[[nodiscard]] RenderCoroutine build_cycles_stage_coroutine(
    const PathKernelConfig &path,
    LuisaPathScheduler scheduler) {
    LUISA_ASSERT(
        luisa_path_scheduler_uses_cycles_stage_partition(scheduler),
        "Scheduler '{}' does not own a Cycles-stage coroutine partition.",
        luisa_path_scheduler_name(scheduler));
    const auto stage_plan = make_path_kernel_scene_stage_plan(
        path.next_event_estimation,
        path.scene->environment_in_light_distribution,
        path.scene->emissive_triangle_count, path.scene->light_count,
        path.scene->geometries.size(), path.scene->curve_geometries.size(),
        path.scene->intersectable_light_count);
    const auto surface_buckets = make_path_surface_bucket_plan(path);
    auto coroutine = build_path_coroutine(
        path, PathCoroutineCutPolicy::cycles_wavefront);
    validate_cycles_stage_partition_abi(
        coroutine, stage_plan, surface_buckets, path.volume_state != nullptr,
        path.has_subsurface,
        !path.path_trace_enabled &&
            path.direct_light_task_sink == nullptr &&
            stage_plan.direct_lighting.transport_stage_count() != 0u);
    return coroutine;
}

}// namespace

class PathKernelExecutorImpl {

public:
    virtual ~PathKernelExecutorImpl() noexcept = default;
  [[nodiscard]] virtual LuisaPathScheduler scheduler() const noexcept = 0;
  virtual void dispatch(luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept = 0;
};

namespace {

[[nodiscard]] bool megakernel_stage_profile_enabled() noexcept {
  const char *raw = std::getenv("PSYCLES_MEGAKERNEL_STAGE_PROFILE");
  return raw != nullptr && std::string_view{raw} == "1";
}

class SerialMegakernelExecutor final : public PathKernelExecutorImpl {

private:
    RenderSerialCompiledShader _shader;
    std::optional<luisa::compute::Buffer<luisa::ulong>> _profile_buffer;
    bool _simd_sample_parallel{false};

public:
  SerialMegakernelExecutor(
      RenderSerialCompiledShader shader,
      std::optional<luisa::compute::Buffer<luisa::ulong>> profile_buffer =
          std::nullopt,
      bool simd_sample_parallel = false) noexcept
      : _shader{std::move(shader)},
        _profile_buffer{std::move(profile_buffer)},
        _simd_sample_parallel{simd_sample_parallel} {}

  [[nodiscard]] LuisaPathScheduler scheduler() const noexcept override {
        return LuisaPathScheduler::megakernel;
    }

  void dispatch(luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept override {
    const auto dispatch_w =
        _simd_sample_parallel ? dispatch.width * 8u : dispatch.width;
    if (!_profile_buffer.has_value()) {
      stream
          << bind_path_program(_shader, dispatch)
                 .dispatch(dispatch_w, dispatch.height);
      return;
    }
    constexpr std::size_t total_words =
        MegakernelStageProfileState::stripe_count *
        MegakernelStageProfileState::stripe_stride;
    std::vector<luisa::ulong> zeros(total_words, 0ull);
    std::vector<luisa::ulong> host_words(total_words, 0ull);
    stream << _profile_buffer->copy_from(zeros.data())
           << luisa::compute::synchronize();
    const auto wall_t0 = std::chrono::steady_clock::now();
#if defined(_M_X64) || defined(__x86_64__)
    const std::uint64_t tsc_t0 = __rdtsc();
#else
    const std::uint64_t tsc_t0 = 0ull;
#endif
    stream << bind_path_program(_shader, dispatch)
                  .dispatch(dispatch_w, dispatch.height)
           << luisa::compute::synchronize();
#if defined(_M_X64) || defined(__x86_64__)
    const std::uint64_t tsc_t1 = __rdtsc();
#else
    const std::uint64_t tsc_t1 = 0ull;
#endif
    const auto wall_t1 = std::chrono::steady_clock::now();
    stream << _profile_buffer->copy_to(host_words.data())
           << luisa::compute::synchronize();

    std::array<std::uint64_t, MegakernelStageProfileState::stage_count>
        stage_cycles{};
    std::array<std::uint64_t, MegakernelStageProfileState::counter_count>
        stage_counts{};
    for (std::uint32_t s = 0u; s < MegakernelStageProfileState::stripe_count;
         ++s) {
      const std::size_t base =
          static_cast<std::size_t>(s) *
          MegakernelStageProfileState::stripe_stride;
      for (std::uint32_t i = 0u; i < MegakernelStageProfileState::stage_count;
           ++i) {
        stage_cycles[i] += host_words[base + i];
      }
      for (std::uint32_t i = 0u; i < MegakernelStageProfileState::counter_count;
           ++i) {
        stage_counts[i] +=
            host_words[base + MegakernelStageProfileState::stage_count + i];
      }
    }
    const double wall_sec =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    const std::uint64_t wall_tsc = tsc_t1 - tsc_t0;
    const double tsc_hz = wall_sec > 0.0 ? static_cast<double>(wall_tsc) / wall_sec : 0.0;
    std::uint64_t sum_cycles = 0ull;
    for (const auto c : stage_cycles) { sum_cycles += c; }
    std::printf(
        "PSYCLES_MEGAKERNEL_STAGE_PROFILE_JSON: "
        "{\"width\":%u,\"height\":%u,\"samples\":%u,"
        "\"dispatch_wall_seconds\":%.9f,\"host_wall_tsc\":%llu,\"tsc_hz\":%.3f,"
        "\"sum_thread_cycles\":%llu,"
        "\"stages_cycles\":{"
        "\"setup_and_film\":%llu,"
        "\"closest_trace\":%llu,"
        "\"volume_and_background\":%llu,"
        "\"surface_geometry\":%llu,"
        "\"surface_svm_shading\":%llu,"
        "\"nee_light_and_bsdf_eval\":%llu,"
        "\"bsdf_scatter\":%llu,"
        "\"shadow_trace\":%llu,"
        "\"subsurface_trace\":%llu,"
        "\"shadow_light_nee\":%llu},"
        "\"counts\":{"
        "\"closest_traces\":%llu,"
        "\"surface_shades\":%llu,"
        "\"nee_evals\":%llu,"
        "\"shadow_traces\":%llu,"
        "\"subsurface_traces\":%llu}}\n",
        dispatch.width, dispatch.height, dispatch.samples, wall_sec,
        static_cast<unsigned long long>(wall_tsc), tsc_hz,
        static_cast<unsigned long long>(sum_cycles),
        static_cast<unsigned long long>(stage_cycles[0]),
        static_cast<unsigned long long>(stage_cycles[1]),
        static_cast<unsigned long long>(stage_cycles[2]),
        static_cast<unsigned long long>(stage_cycles[3]),
        static_cast<unsigned long long>(stage_cycles[4]),
        static_cast<unsigned long long>(stage_cycles[5]),
        static_cast<unsigned long long>(stage_cycles[6]),
        static_cast<unsigned long long>(stage_cycles[7]),
        static_cast<unsigned long long>(stage_cycles[8]),
        static_cast<unsigned long long>(stage_cycles[9]),
        static_cast<unsigned long long>(stage_counts[0]),
        static_cast<unsigned long long>(stage_counts[1]),
        static_cast<unsigned long long>(stage_counts[2]),
        static_cast<unsigned long long>(stage_counts[3]),
        static_cast<unsigned long long>(stage_counts[4]));
    std::fflush(stdout);
  }
};

class SampleMegakernelExecutor final : public PathKernelExecutorImpl {

private:
    RenderSampleCompiledShader _shader;

public:
  explicit SampleMegakernelExecutor(RenderSampleCompiledShader shader) noexcept
        : _shader{std::move(shader)} {}

  [[nodiscard]] LuisaPathScheduler scheduler() const noexcept override {
        return LuisaPathScheduler::megakernel_per_sample;
    }

  void dispatch(luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept override {
        stream << bind_path_program(_shader, dispatch)
                  .dispatch(dispatch.width, dispatch.height, dispatch.samples);
    }
};

class CoroutineExecutor final : public PathKernelExecutorImpl {

private:
    LuisaPathScheduler _scheduler_kind;
    std::unique_ptr<RenderSchedulers::Base> _scheduler;

public:
  CoroutineExecutor(LuisaPathScheduler scheduler_kind,
                    std::unique_ptr<RenderSchedulers::Base> scheduler) noexcept
      : _scheduler_kind{scheduler_kind},
        _scheduler{std::move(scheduler)} {}

  [[nodiscard]] LuisaPathScheduler scheduler() const noexcept override {
        return _scheduler_kind;
    }

  void dispatch(luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept override {
        stream << bind_path_program(*_scheduler, dispatch)
                      .dispatch(dispatch.width, dispatch.height, dispatch.samples);
    }
};

}// namespace

PathKernelExecutor::PathKernelExecutor() noexcept = default;

PathKernelExecutor::PathKernelExecutor(
    std::unique_ptr<PathKernelExecutorImpl> impl) noexcept
    : _impl{std::move(impl)} {}

PathKernelExecutor::~PathKernelExecutor() noexcept = default;
PathKernelExecutor::PathKernelExecutor(PathKernelExecutor &&) noexcept =
    default;
PathKernelExecutor &
PathKernelExecutor::operator=(PathKernelExecutor &&) noexcept = default;

PathKernelExecutor::operator bool() const noexcept { return _impl != nullptr; }

LuisaPathScheduler PathKernelExecutor::scheduler() const noexcept {
  return _impl == nullptr ? LuisaPathScheduler::megakernel : _impl->scheduler();
}

void PathKernelExecutor::dispatch(
    luisa::compute::Stream &stream,
    const PathKernelDispatch &dispatch) const noexcept {
  LUISA_ASSERT(_impl != nullptr, "Invalid path-kernel executor.");
    LUISA_ASSERT(dispatch.pixel_count != 0u,
                 "A path-kernel dispatch must contain pixels.");
    LUISA_ASSERT(dispatch.width != 0u && dispatch.height != 0u,
                 "A path-kernel dispatch must have a nonempty 2D shape.");
    LUISA_ASSERT(dispatch.samples != 0u,
                 "A path-kernel dispatch must contain samples.");
  LUISA_ASSERT(static_cast<std::uint64_t>(dispatch.width) *
                static_cast<std::uint64_t>(dispatch.height) ==
            dispatch.pixel_count,
        "Path-kernel shape {} x {} does not cover {} pixels.",
               dispatch.width, dispatch.height, dispatch.pixel_count);
    LUISA_ASSERT(
      PixelSampleDispatchPlan::make(dispatch.pixel_count, dispatch.samples)
            .has_value(),
        "Path-kernel pixel/sample product {} x {} exceeds the "
        "32-bit logical dispatch domain.",
      dispatch.pixel_count, dispatch.samples);
    _impl->dispatch(stream, dispatch);
}

PathKernelExecutor
build_path_kernel_executor(luisa::compute::Device &device,
    const PathKernelConfig &path,
    const PathKernelExecutorConfig &config) {
    const bool vt_supported_scheduler =
        config.scheduler == LuisaPathScheduler::wavefront ||
        config.scheduler == LuisaPathScheduler::wavefront_staged;
    const bool vt_active =
        vt_supported_scheduler && path.scene &&
        path.scene->virtual_texture_cache &&
        path.scene->virtual_texture_cache->has_virtual_images();
    if (!vt_supported_scheduler && path.scene &&
        path.scene->virtual_texture_cache &&
        path.scene->virtual_texture_cache->has_virtual_images()) {
        path.scene->virtual_texture_cache->ensure_eager_direct_textures_uploaded(
            *path.scene);
    }
    switch (config.scheduler) {
        case LuisaPathScheduler::megakernel: {
            std::optional<luisa::compute::Buffer<luisa::ulong>> profile_buffer;
            auto profiled_path = path;
            if (megakernel_stage_profile_enabled()) {
              profile_buffer = device.create_buffer<luisa::ulong>(
                  MegakernelStageProfileState::stripe_count *
                  MegakernelStageProfileState::stripe_stride);
              profiled_path.megakernel_stage_profile_buffer =
                  &(*profile_buffer);
            }
            const bool sample_parallel =
                simd_sample_parallel_enabled(profiled_path);
            auto kernel = build_path_serial_kernel(profiled_path);
            auto shader = luisa::compute::coro::detail::
                coro_scheduler_label_shader(
                    device.compile(kernel, config.shader_option),
                    "path_megakernel");
            return PathKernelExecutor{
                std::make_unique<SerialMegakernelExecutor>(
                    std::move(shader), std::move(profile_buffer),
                    sample_parallel)};
        }
        case LuisaPathScheduler::megakernel_per_sample: {
            auto kernel = build_path_sample_kernel(path);
            auto shader = luisa::compute::coro::detail::
                coro_scheduler_label_shader(
                    device.compile(kernel, config.shader_option),
                    "path_megakernel_per_sample");
            return PathKernelExecutor{
        std::make_unique<SampleMegakernelExecutor>(std::move(shader))};
        }
        case LuisaPathScheduler::wavefront: {
    LUISA_ASSERT(config.wavefront_frame_capacity != 0u,
                "Wavefront frame capacity must be positive.");
    auto shadow_path = make_local_shadow_path(path, vt_active);
    shadow_path.enable_virtual_texture_cache = vt_active;
    const auto pso_t0 = VtProfileRecorder::steady_ns();
    auto coroutine =
        build_cycles_stage_coroutine(shadow_path, config.scheduler);
    if (vt_active) {
        validate_vt_extension_abi(coroutine);
    }
    log_coroutine_frame_if_requested(coroutine);
    LUISA_INFO("Psycles wavefront path coroutine: subroutines={} "
                "frame_fields={} frame_bytes={} capacity={}.",
                coroutine.subroutine_count(),
                coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
                config.wavefront_frame_capacity);
    luisa::compute::coro::WavefrontCoroSchedulerConfig scheduler_config;
    scheduler_config.thread_count = config.wavefront_frame_capacity;
            scheduler_config.execution_block_size =
                config.wavefront_execution_block_size;
    scheduler_config.shader_compile_threads =
        std::clamp<std::uint32_t>(std::thread::hardware_concurrency(), 1u, 16u);
    if (const auto *override_threads =
            std::getenv("PSYCLES_SHADER_COMPILE_THREADS")) {
        const auto parsed = static_cast<std::uint32_t>(
            std::strtoul(override_threads, nullptr, 10));
        if (parsed != 0u) {
            scheduler_config.shader_compile_threads = parsed;
        }
    }
    scheduler_config.shader_option = config.shader_option;
    if (vt_active && shadow_path.scene->virtual_texture_cache->async_worker_enabled()) {
        scheduler_config.max_async_in_flight_batches =
            shadow_path.scene->virtual_texture_cache->async_batch_slots();
    }
    if (config.on_after_coroutine_build) {
        config.on_after_coroutine_build();
    }
    auto scheduler = std::make_unique<RenderSchedulers::Wavefront>(
                    device, coroutine, scheduler_config);
    const auto pso_t1 = VtProfileRecorder::steady_ns();
    if (vt_active) {
        shadow_path.scene->virtual_texture_cache->record_pso_compile_event(
            pso_t0, pso_t1, "Wavefront Coro PSO Compile");
    }
    if (config.on_before_register_extensions) {
        config.on_before_register_extensions();
    }
    if (vt_active) {
        const auto ext_t0 = VtProfileRecorder::steady_ns();
        auto preparation_stream = device.create_stream();
        scheduler->register_extension_handler(
            preparation_stream,
            shadow_path.scene->virtual_texture_cache
                ->make_wavefront_extension_handler());
        preparation_stream << luisa::compute::synchronize();
        const auto ext_t1 = VtProfileRecorder::steady_ns();
        shadow_path.scene->virtual_texture_cache->record_pso_compile_event(
            ext_t0, ext_t1, "VT Extension Handler Compile");
    }
    return PathKernelExecutor{std::make_unique<CoroutineExecutor>(
        config.scheduler, std::move(scheduler))};
        }
        case LuisaPathScheduler::wavefront_graph: {
            LUISA_ASSERT(config.wavefront_frame_capacity != 0u,
                         "Graph-wavefront frame capacity must be positive.");
            auto shadow_path = make_local_shadow_path(path);
            auto coroutine = build_cycles_stage_coroutine(
                shadow_path, config.scheduler);
            log_coroutine_frame_if_requested(coroutine);
            luisa::compute::coro::GraphWavefrontCoroSchedulerConfig
                scheduler_config;
            scheduler_config.thread_count = config.wavefront_frame_capacity;
            scheduler_config.execution_block_size =
                config.wavefront_execution_block_size;
            scheduler_config.worker_count =
                config.wavefront_graph_worker_count;
            scheduler_config.selective_scheduling =
                config.wavefront_graph_selective_scheduling;
            scheduler_config.refill_threshold =
                config.wavefront_graph_refill_threshold;
            if (config.wavefront_graph_selective_scheduling) {
                scheduler_config.refill_continuations = {
                    path_transition::intersect_closest};
            }
            scheduler_config.counter_readback_batch_size =
                config.wavefront_graph_selective_scheduling ?
                    1u : config.wavefront_counter_readback_batch_size;
            scheduler_config.counter_readback_pipeline_depth =
                config.wavefront_graph_selective_scheduling ?
                    1u : config.wavefront_counter_readback_pipeline_depth;
            scheduler_config.tail_megakernel_threshold =
                config.wavefront_tail_megakernel_threshold;
            const auto has_surface_queue_hint =
                config.wavefront_graph_selective_scheduling &&
                configure_surface_queue_hint(
                    shadow_path, coroutine, scheduler_config);
            scheduler_config.shader_option = config.shader_option;
            auto scheduler =
                std::make_unique<RenderSchedulers::GraphWavefront>(
                    device, coroutine, scheduler_config);
            LUISA_INFO(
                "Psycles graph-wavefront path coroutine: subroutines={} "
                "frame_fields={} frame_bytes={} capacity={} block={} "
                "workers={} selective={} refill_threshold={} "
                "readback_batch={} readback_depth={} "
                "tail_threshold={} surface_hint_sort={} surface_keys={}.",
                coroutine.subroutine_count(),
                coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
                config.wavefront_frame_capacity,
                config.wavefront_execution_block_size,
                config.wavefront_graph_worker_count,
                config.wavefront_graph_selective_scheduling,
                config.wavefront_graph_refill_threshold,
                config.wavefront_counter_readback_batch_size,
                config.wavefront_counter_readback_pipeline_depth,
                config.wavefront_tail_megakernel_threshold,
                has_surface_queue_hint,
                scheduler_config.hint_range);
            return PathKernelExecutor{std::make_unique<CoroutineExecutor>(
                config.scheduler, std::move(scheduler))};
        }
        case LuisaPathScheduler::wavefront_staged: {
    LUISA_ASSERT(config.wavefront_frame_capacity != 0u,
                "Staged wavefront frame capacity must be positive.");
    auto staged_path = make_local_shadow_path(path, vt_active);
    staged_path.staged_surface_sort_annotation = true;
    staged_path.enable_virtual_texture_cache = vt_active;
    DirectLightTaskQueueBinding direct_light_queue;
    CausticsTaskQueueBinding caustics_queue;
    const auto stage_plan = make_path_kernel_scene_stage_plan(
        path.next_event_estimation,
        path.scene->environment_in_light_distribution,
        path.scene->emissive_triangle_count, path.scene->light_count,
        path.scene->geometries.size(), path.scene->curve_geometries.size(),
        path.scene->intersectable_light_count);
    const auto can_publish_direct_light = can_stage_direct_light_queue(
        config.staged_direct_light_queue, path.path_trace_enabled, stage_plan);
    const auto pso_t0 = VtProfileRecorder::steady_ns();
    const auto backend_name = device.backend_name();
    auto effective_execution_block_size = config.wavefront_execution_block_size;
    if (const auto *override_ebs =
            std::getenv("PSYCLES_WAVEFRONT_EXECUTION_BLOCK_SIZE")) {
      const auto parsed =
          static_cast<std::uint32_t>(std::strtoul(override_ebs, nullptr, 10));
      if (valid_luisa_execution_block_size(parsed)) {
        effective_execution_block_size = parsed;
      }
    }
    if (can_publish_direct_light) {
      direct_light_queue = make_direct_light_task_queue(
          device, staged_path, config.wavefront_frame_capacity,
          effective_execution_block_size, config.shader_option);
      staged_path.direct_light_task_sink = direct_light_queue.sink;
    }
    if (staged_path.caustics_active() && !path.path_trace_enabled) {
      caustics_queue = make_caustics_task_queue(
          device, staged_path, config.wavefront_frame_capacity,
          effective_execution_block_size, config.shader_option);
      staged_path.caustics_task_sink = caustics_queue.sink;
    }
            auto coroutine =
                build_cycles_stage_coroutine(staged_path, config.scheduler);
            if (vt_active) {
              validate_vt_extension_abi(coroutine);
            }
            log_coroutine_frame_if_requested(coroutine);
    LUISA_INFO("Psycles staged wavefront path coroutine: "
                "subroutines={} frame_fields={} frame_bytes={} capacity={}.",
                coroutine.subroutine_count(),
                coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
                config.wavefront_frame_capacity);
    bool has_surface_block_override = false;
    auto shade_surface_block_size =
        (backend_name == "vk") ? 32u : 64u;
    if (const auto *override_block =
            std::getenv("PSYCLES_SHADE_SURFACE_BLOCK_SIZE")) {
      const auto parsed =
          static_cast<std::uint32_t>(std::strtoul(override_block, nullptr, 10));
      if (valid_luisa_execution_block_size(parsed)) {
        shade_surface_block_size = parsed;
        has_surface_block_override = true;
      }
    }
    bool has_bucket0_block_override = false;
    auto bucket0_block_size = shade_surface_block_size;
    if (const auto *override_b0 =
            std::getenv("PSYCLES_SHADE_SURFACE_BUCKET0_BLOCK_SIZE")) {
      const auto parsed =
          static_cast<std::uint32_t>(std::strtoul(override_b0, nullptr, 10));
      if (valid_luisa_execution_block_size(parsed)) {
        bucket0_block_size = parsed;
        has_bucket0_block_override = true;
      }
    }
    const auto surface_buckets = make_path_surface_bucket_plan(staged_path);
    luisa::compute::coro::WavefrontCoroSchedulerConfig scheduler_config;
    scheduler_config.thread_count = config.wavefront_frame_capacity;
    scheduler_config.execution_block_size = effective_execution_block_size;
    for (std::size_t b = 0u; b < surface_buckets.buckets.size(); ++b) {
      std::uint32_t block_size = shade_surface_block_size;
      if (b == 0u && has_bucket0_block_override) {
        block_size = bucket0_block_size;
      } else if (!has_surface_block_override && backend_name == "vk") {
        const auto &spec = surface_buckets.buckets[b];
        block_size =
            (spec.max_surface_closures <= 4u && !spec.may_have_bssrdf) ? 64u
                                                                       : 32u;
      }
      scheduler_config.continuation_block_sizes.push_back(
          {.continuation = surface_buckets.buckets[b].continuation_name,
           .execution_block_size = block_size});
    }
    if (surface_buckets.post_closure_substage) {
      scheduler_config.continuation_block_sizes.push_back(
          {.continuation = path_transition::shade_surface_post,
           .execution_block_size = effective_execution_block_size});
    }
            scheduler_config.largest_continuation_first = true;
            scheduler_config.incremental_continuation_counts = true;
            scheduler_config.fused_continuation_counts = true;
            scheduler_config.frame_buffer_compaction = true;
            if (const auto *override_compact =
                    std::getenv("PSYCLES_WAVEFRONT_FRAME_COMPACTION")) {
                scheduler_config.frame_buffer_compaction =
                    std::string_view{override_compact} != "0";
            }
            scheduler_config.refill_continuations = {
                path_transition::intersect_closest};
            const bool collect_intersect = [&]() noexcept {
              const char *raw = std::getenv("PSYCLES_WAVEFRONT_COLLECT_INTERSECT");
              if (raw != nullptr && *raw != '\0') {
                return std::string_view{raw} != "0";
              }
              return surface_buckets.buckets.size() >= 3u;
            }();
            if (collect_intersect) {
              scheduler_config.collect_continuations = {
                  path_transition::intersect_closest};
            }
            float refill_ratio = 0.5f;
            if (const auto *override_ratio =
                    std::getenv("PSYCLES_WAVEFRONT_REFILL_RATIO")) {
                refill_ratio = std::clamp(
                    std::strtof(override_ratio, nullptr), 0.0f, 1.0f);
            }
            scheduler_config.refill_threshold =
                std::clamp<std::uint32_t>(
                    static_cast<std::uint32_t>(
                        std::lround(static_cast<double>(refill_ratio) *
                                    static_cast<double>(
                                        scheduler_config.thread_count))),
                    1u, scheduler_config.thread_count);
            if (const auto *override_refill =
                    std::getenv("PSYCLES_WAVEFRONT_REFILL_THRESHOLD")) {
                scheduler_config.refill_threshold = static_cast<std::uint32_t>(
                    std::strtoul(override_refill, nullptr, 10));
            }
    scheduler_config.shader_compile_threads =
        std::clamp<std::uint32_t>(std::thread::hardware_concurrency(), 1u, 16u);
    if (const auto *override_threads =
            std::getenv("PSYCLES_SHADER_COMPILE_THREADS")) {
        const auto parsed = static_cast<std::uint32_t>(
            std::strtoul(override_threads, nullptr, 10));
        if (parsed != 0u) {
            scheduler_config.shader_compile_threads = parsed;
        }
    }
    scheduler_config.shader_option = config.shader_option;
    if (vt_active && staged_path.scene->virtual_texture_cache->async_worker_enabled()) {
        scheduler_config.max_async_in_flight_batches =
            staged_path.scene->virtual_texture_cache->async_batch_slots();
    }
    if (config.on_after_coroutine_build) {
        config.on_after_coroutine_build();
    }
    auto scheduler = std::make_unique<RenderSchedulers::Wavefront>(
                    device, coroutine, scheduler_config);
    const auto pso_t1 = VtProfileRecorder::steady_ns();
    if (vt_active) {
        staged_path.scene->virtual_texture_cache->record_pso_compile_event(
            pso_t0, pso_t1, "Wavefront Staged Coro PSO Compile");
    }
    if (config.on_before_register_extensions) {
        config.on_before_register_extensions();
    }
    const auto ext_t0 = VtProfileRecorder::steady_ns();
    auto preparation_stream = device.create_stream();
    scheduler->register_extension_handler(preparation_stream, make_surface_sort_handler);
    if (vt_active) {
      scheduler->register_extension_handler(
          preparation_stream,
          staged_path.scene->virtual_texture_cache
              ->make_wavefront_extension_handler());
    }
    preparation_stream << luisa::compute::synchronize();
    const auto ext_t1 = VtProfileRecorder::steady_ns();
    if (vt_active) {
        staged_path.scene->virtual_texture_cache->record_pso_compile_event(
            ext_t0, ext_t1, "Extension Handlers Compile");
    }
    if (direct_light_queue.work) {
      scheduler->register_auxiliary_work(direct_light_queue.work);
    }
    if (caustics_queue.work) {
      scheduler->register_auxiliary_work(caustics_queue.work);
    }
    LUISA_INFO("Psycles staged surface queue: keys={} requested={} "
               "typed_annotation={} direct_light_queue={}",
               surface_queue_key_range(*path.scene),
               path.staged_surface_sorting,
               staged_path.staged_surface_sort_annotation,
               direct_light_queue.work != nullptr);
    return PathKernelExecutor{std::make_unique<CoroutineExecutor>(
        config.scheduler, std::move(scheduler))};
        }
        case LuisaPathScheduler::persistent: {
    LUISA_ASSERT(config.persistent_worker_count != 0u,
                "Persistent worker count must be positive.");
    LUISA_ASSERT(config.persistent_block_size != 0u,
                "Persistent block size must be positive.");
    LUISA_ASSERT(config.persistent_fetch_size != 0u,
                "Persistent fetch size must be positive.");
    auto shadow_path = make_local_shadow_path(path);
    auto coroutine =
        build_cycles_stage_coroutine(shadow_path, config.scheduler);
    log_coroutine_frame_if_requested(coroutine);
    luisa::compute::coro::PersistentThreadsCoroSchedulerConfig scheduler_config;
    scheduler_config.thread_count = config.persistent_worker_count;
    scheduler_config.block_size = config.persistent_block_size;
    scheduler_config.fetch_size = config.persistent_fetch_size;
    scheduler_config.shared_memory_soa = config.persistent_shared_memory_soa;
            scheduler_config.global_memory_ext =
                config.persistent_global_memory_extension;
    scheduler_config.shader_option = config.shader_option;
    auto scheduler = std::make_unique<RenderSchedulers::Persistent>(
                    device, coroutine, scheduler_config);
            LUISA_INFO(
                "Psycles persistent path coroutine: subroutines={} "
                "frame_fields={} frame_bytes={} workers={} "
                "requested_block={} selected_block={} static_shared_bytes={} "
                "fetch={} shared_soa={} global_extension={} "
                "global_frames={}.",
        coroutine.subroutine_count(), coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
        scheduler->config().thread_count, config.persistent_block_size,
                scheduler->config().block_size,
                scheduler->static_shared_memory_size_bytes(),
        scheduler->config().fetch_size, scheduler->config().shared_memory_soa,
                scheduler->config().global_memory_ext,
                scheduler->config().global_memory_frames);
    return PathKernelExecutor{std::make_unique<CoroutineExecutor>(
        config.scheduler, std::move(scheduler))};
        }
    }
  LUISA_ERROR_WITH_LOCATION("Invalid Psycles path scheduler value {}.",
        static_cast<std::uint32_t>(config.scheduler));
}

}// namespace psycles::luisa_backend::detail
