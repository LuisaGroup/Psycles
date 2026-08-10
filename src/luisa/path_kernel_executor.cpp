#include "path_kernel_executor.h"

#include "path_kernel_builder.h"

#include <utility>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/coro/schedulers/persistent_threads.h>
#include <luisa/coro/schedulers/wavefront.h>

namespace psycles::luisa_backend::detail {
namespace {

template<typename Signature>
struct RenderSchedulerTypes;

template<typename... Args>
struct RenderSchedulerTypes<void(Args...)> {
    using Base = luisa::compute::coro::CoroScheduler<Args...>;
    using Wavefront =
        luisa::compute::coro::WavefrontCoroScheduler<Args...>;
    using Persistent =
        luisa::compute::coro::PersistentThreadsCoroScheduler<Args...>;
};

using RenderSchedulers =
    RenderSchedulerTypes<RenderKernelSignature>;

template<typename Program>
[[nodiscard]] auto bind_path_program(
    Program &program,
    const PathKernelDispatch &dispatch) noexcept {
    return program(
               dispatch.combined,
               dispatch.normal,
               dispatch.albedo,
               dispatch.light_passes,
               dispatch.sample_count,
               dispatch.volume_guiding_raw,
               dispatch.volume_guiding_denoised,
               dispatch.path_trace,
               dispatch.sample_first,
               dispatch.samples,
               dispatch.sobol_table,
               dispatch.filter_table,
               dispatch.parameters)
        .dispatch(dispatch.pixel_count);
}

}// namespace

class PathKernelExecutorImpl {

public:
    virtual ~PathKernelExecutorImpl() noexcept = default;
    [[nodiscard]] virtual LuisaPathScheduler
    scheduler() const noexcept = 0;
    virtual void dispatch(
        luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept = 0;
};

namespace {

class MegakernelExecutor final : public PathKernelExecutorImpl {

private:
    RenderCompiledShader _shader;

public:
    explicit MegakernelExecutor(
        RenderCompiledShader shader) noexcept
        : _shader{std::move(shader)} {}

    [[nodiscard]] LuisaPathScheduler
    scheduler() const noexcept override {
        return LuisaPathScheduler::megakernel;
    }

    void dispatch(
        luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept override {
        stream << bind_path_program(_shader, dispatch);
    }
};

class CoroutineExecutor final : public PathKernelExecutorImpl {

private:
    LuisaPathScheduler _scheduler_kind;
    std::unique_ptr<RenderSchedulers::Base> _scheduler;

public:
    CoroutineExecutor(
        LuisaPathScheduler scheduler_kind,
        std::unique_ptr<RenderSchedulers::Base> scheduler) noexcept
        : _scheduler_kind{scheduler_kind},
          _scheduler{std::move(scheduler)} {}

    [[nodiscard]] LuisaPathScheduler
    scheduler() const noexcept override {
        return _scheduler_kind;
    }

    void dispatch(
        luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) noexcept override {
        auto command =
            bind_path_program(*_scheduler, dispatch);
        command(stream);
    }
};

}// namespace

PathKernelExecutor::PathKernelExecutor() noexcept = default;

PathKernelExecutor::PathKernelExecutor(
    std::unique_ptr<PathKernelExecutorImpl> impl) noexcept
    : _impl{std::move(impl)} {}

PathKernelExecutor::~PathKernelExecutor() noexcept = default;
PathKernelExecutor::PathKernelExecutor(
    PathKernelExecutor &&) noexcept = default;
PathKernelExecutor &PathKernelExecutor::operator=(
    PathKernelExecutor &&) noexcept = default;

PathKernelExecutor::operator bool() const noexcept {
    return _impl != nullptr;
}

LuisaPathScheduler
PathKernelExecutor::scheduler() const noexcept {
    return _impl == nullptr
               ? LuisaPathScheduler::megakernel
               : _impl->scheduler();
}

void PathKernelExecutor::dispatch(
    luisa::compute::Stream &stream,
    const PathKernelDispatch &dispatch) const noexcept {
    LUISA_ASSERT(_impl != nullptr,
                 "Invalid path-kernel executor.");
    LUISA_ASSERT(dispatch.pixel_count != 0u,
                 "A path-kernel dispatch must contain pixels.");
    _impl->dispatch(stream, dispatch);
}

PathKernelExecutor build_path_kernel_executor(
    luisa::compute::Device &device,
    const PathKernelConfig &path,
    const PathKernelExecutorConfig &config) {
    switch (config.scheduler) {
        case LuisaPathScheduler::megakernel: {
            auto kernel = build_path_kernel(path);
            auto shader = device.compile(
                kernel, config.shader_option);
            return PathKernelExecutor{
                std::make_unique<MegakernelExecutor>(
                    std::move(shader))};
        }
        case LuisaPathScheduler::wavefront: {
            LUISA_ASSERT(
                config.wavefront_frame_capacity != 0u,
                "Wavefront frame capacity must be positive.");
            auto coroutine = build_path_coroutine(path);
            LUISA_INFO(
                "Psycles wavefront path coroutine: subroutines={} "
                "frame_fields={} frame_bytes={} capacity={}.",
                coroutine.subroutine_count(),
                coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
                config.wavefront_frame_capacity);
            luisa::compute::coro::WavefrontCoroSchedulerConfig
                scheduler_config;
            scheduler_config.thread_count =
                config.wavefront_frame_capacity;
            scheduler_config.shader_option =
                config.shader_option;
            auto scheduler =
                std::make_unique<RenderSchedulers::Wavefront>(
                    device, coroutine, scheduler_config);
            return PathKernelExecutor{
                std::make_unique<CoroutineExecutor>(
                    config.scheduler,
                    std::move(scheduler))};
        }
        case LuisaPathScheduler::persistent: {
            LUISA_ASSERT(
                config.persistent_worker_count != 0u,
                "Persistent worker count must be positive.");
            LUISA_ASSERT(
                config.persistent_block_size != 0u,
                "Persistent block size must be positive.");
            LUISA_ASSERT(
                config.persistent_fetch_size != 0u,
                "Persistent fetch size must be positive.");
            auto coroutine = build_path_coroutine(path);
            LUISA_INFO(
                "Psycles persistent path coroutine: subroutines={} "
                "frame_fields={} frame_bytes={} workers={} block={} "
                "fetch={} shared_soa={} global_extension={}.",
                coroutine.subroutine_count(),
                coroutine.frame().frame_field_count(),
                coroutine.frame().frame_type()->size(),
                config.persistent_worker_count,
                config.persistent_block_size,
                config.persistent_fetch_size,
                config.persistent_shared_memory_soa,
                config.persistent_global_memory_extension);
            luisa::compute::coro::PersistentThreadsCoroSchedulerConfig
                scheduler_config;
            scheduler_config.thread_count =
                config.persistent_worker_count;
            scheduler_config.block_size =
                config.persistent_block_size;
            scheduler_config.fetch_size =
                config.persistent_fetch_size;
            scheduler_config.shared_memory_soa =
                config.persistent_shared_memory_soa;
            scheduler_config.global_memory_ext =
                config.persistent_global_memory_extension;
            scheduler_config.shader_option =
                config.shader_option;
            auto scheduler =
                std::make_unique<RenderSchedulers::Persistent>(
                    device, coroutine, scheduler_config);
            return PathKernelExecutor{
                std::make_unique<CoroutineExecutor>(
                    config.scheduler,
                    std::move(scheduler))};
        }
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid Psycles path scheduler value {}.",
        static_cast<std::uint32_t>(config.scheduler));
}

}// namespace psycles::luisa_backend::detail
