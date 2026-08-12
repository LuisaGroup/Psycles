#include "path_kernel_executor.h"

#include "path_kernel_builder.h"
#include "path_kernel_direct_light_queue.h"
#include "path_kernel_transitions.h"
#include "sample_dispatch_partition.h"

#include <limits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/schedulers/persistent_threads.h>
#include <luisa/coro/schedulers/wavefront.h>

namespace psycles::luisa_backend::detail {
namespace {

template <typename Signature> struct RenderSchedulerTypes;

template <typename... Args> struct RenderSchedulerTypes<void(Args...)> {
  using Base = luisa::compute::coro::CoroScheduler<Args...>;
  using Wavefront = luisa::compute::coro::WavefrontCoroScheduler<Args...>;
  using Persistent =
      luisa::compute::coro::PersistentThreadsCoroScheduler<Args...>;
};

using RenderSchedulers = RenderSchedulerTypes<RenderKernelSignature>;

template <typename Program>
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

} // namespace

class PathKernelExecutorImpl {

public:
  virtual ~PathKernelExecutorImpl() noexcept = default;
  [[nodiscard]] virtual LuisaPathScheduler scheduler() const noexcept = 0;
  virtual void dispatch(luisa::compute::Stream &stream,
                        const PathKernelDispatch &dispatch) noexcept = 0;
};

namespace {

class SerialMegakernelExecutor final : public PathKernelExecutorImpl {

private:
  RenderSerialCompiledShader _shader;

public:
  explicit SerialMegakernelExecutor(RenderSerialCompiledShader shader) noexcept
      : _shader{std::move(shader)} {}

  [[nodiscard]] LuisaPathScheduler scheduler() const noexcept override {
    return LuisaPathScheduler::megakernel;
  }

  void dispatch(luisa::compute::Stream &stream,
                const PathKernelDispatch &dispatch) noexcept override {
    stream
        << bind_path_program(_shader, dispatch).dispatch(dispatch.pixel_count);
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
      : _scheduler_kind{scheduler_kind}, _scheduler{std::move(scheduler)} {}

  [[nodiscard]] LuisaPathScheduler scheduler() const noexcept override {
    return _scheduler_kind;
  }

  void dispatch(luisa::compute::Stream &stream,
                const PathKernelDispatch &dispatch) noexcept override {
    auto command =
        bind_path_program(*_scheduler, dispatch)
            .dispatch(dispatch.width, dispatch.height, dispatch.samples);
    command(stream);
  }
};

} // namespace

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
  switch (config.scheduler) {
  case LuisaPathScheduler::megakernel: {
    auto kernel = build_path_serial_kernel(path);
    auto shader = device.compile(kernel, config.shader_option);
    return PathKernelExecutor{
        std::make_unique<SerialMegakernelExecutor>(std::move(shader))};
  }
  case LuisaPathScheduler::megakernel_per_sample: {
    auto kernel = build_path_sample_kernel(path);
    auto shader = device.compile(kernel, config.shader_option);
    return PathKernelExecutor{
        std::make_unique<SampleMegakernelExecutor>(std::move(shader))};
  }
  case LuisaPathScheduler::wavefront: {
    LUISA_ASSERT(config.wavefront_frame_capacity != 0u,
                 "Wavefront frame capacity must be positive.");
    auto coroutine =
        build_path_coroutine(path, PathCoroutineCutPolicy::compact);
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
    scheduler_config.shader_option = config.shader_option;
    auto scheduler = std::make_unique<RenderSchedulers::Wavefront>(
        device, coroutine, scheduler_config);
    return PathKernelExecutor{std::make_unique<CoroutineExecutor>(
        config.scheduler, std::move(scheduler))};
  }
  case LuisaPathScheduler::wavefront_staged: {
    LUISA_ASSERT(config.wavefront_frame_capacity != 0u,
                 "Staged wavefront frame capacity must be positive.");
    auto staged_path = path;
    DirectLightTaskQueueBinding direct_light_queue;
    const auto stage_plan = make_path_kernel_scene_stage_plan(
        path.next_event_estimation,
        path.scene->environment_in_light_distribution,
        path.scene->emissive_triangle_count, path.scene->light_count,
        path.scene->geometries.size(), path.scene->curve_geometries.size(),
        path.scene->cycles_completion_source_dense_count,
        path.scene->cycles_completion_source_sparse_count);
    const auto can_publish_direct_light = can_stage_direct_light_queue(
        config.staged_direct_light_queue, path.path_trace_enabled, stage_plan);
    if (can_publish_direct_light) {
      direct_light_queue = make_direct_light_task_queue(
          device, path, config.wavefront_frame_capacity, config.shader_option);
      staged_path.direct_light_task_sink = direct_light_queue.sink;
    }
    auto coroutine = build_path_coroutine(
        staged_path, PathCoroutineCutPolicy::staged_wavefront);
    const auto surface_count = path.scene->surfaces.size();
    const auto has_surface_queue_hint = path.staged_surface_sorting &&
                                        surface_count != 0u &&
                                        (!path.scene->geometries.empty() ||
                                         !path.scene->curve_geometries.empty());
    if (has_surface_queue_hint) {
      validate_surface_queue_hint_abi(coroutine);
    }
    LUISA_INFO("Psycles staged wavefront path coroutine: "
               "subroutines={} frame_fields={} frame_bytes={} capacity={}.",
               coroutine.subroutine_count(),
               coroutine.frame().frame_field_count(),
               coroutine.frame().frame_type()->size(),
               config.wavefront_frame_capacity);
    luisa::compute::coro::WavefrontCoroSchedulerConfig scheduler_config;
    scheduler_config.thread_count = config.wavefront_frame_capacity;
    scheduler_config.execution_block_size =
        config.wavefront_execution_block_size;
    scheduler_config.largest_continuation_first = true;
    scheduler_config.incremental_continuation_counts = true;
    scheduler_config.refill_continuations = {
        path_transition::intersect_closest};
    if (has_surface_queue_hint) {
      LUISA_ASSERT(surface_count <= std::numeric_limits<std::uint32_t>::max(),
                   "Surface queue key range {} exceeds the uint32 "
                   "scheduler ABI.",
                   surface_count);
      scheduler_config.hint_range = static_cast<std::uint32_t>(surface_count);
      scheduler_config.hint_fields = {path_transition::shade_surface};
    }
    scheduler_config.shader_option = config.shader_option;
    auto scheduler = std::make_unique<RenderSchedulers::Wavefront>(
        device, coroutine, scheduler_config);
    if (direct_light_queue.work) {
      scheduler->register_auxiliary_work(direct_light_queue.work);
    }
    LUISA_INFO("Psycles staged surface queue: keys={} requested={} "
               "hint_sort={} direct_light_queue={}",
               surface_count, path.staged_surface_sorting,
               !scheduler->config().hint_fields.empty(),
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
    auto coroutine =
        build_path_coroutine(path, PathCoroutineCutPolicy::compact);
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

} // namespace psycles::luisa_backend::detail
