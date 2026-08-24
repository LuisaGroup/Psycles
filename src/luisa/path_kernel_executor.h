#pragma once

#include <psycles/luisa/path_tracer.h>

#include "path_tracer_types.h"

#include <cstdint>
#include <memory>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace psycles::luisa_backend::detail {

struct PathKernelConfig;
class PathKernelExecutorImpl;

struct PathKernelExecutorConfig {
  LuisaPathScheduler scheduler{LuisaPathScheduler::megakernel};
    std::uint32_t wavefront_frame_capacity{1u << 20u};
    std::uint32_t wavefront_execution_block_size{32u};
    std::uint32_t wavefront_graph_worker_count{0u};
    bool wavefront_graph_selective_scheduling{false};
    std::uint32_t wavefront_graph_refill_threshold{0u};
    std::uint32_t wavefront_counter_readback_batch_size{4u};
    std::uint32_t wavefront_counter_readback_pipeline_depth{2u};
    std::uint32_t wavefront_tail_megakernel_threshold{
        luisa_wavefront_auto_tail_threshold};
    bool staged_direct_light_queue{false};
    std::uint32_t persistent_worker_count{1u << 15u};
    std::uint32_t persistent_block_size{32u};
    std::uint32_t persistent_fetch_size{1u};
    bool persistent_shared_memory_soa{true};
    bool persistent_global_memory_extension{true};
    luisa::compute::ShaderOption shader_option{};
};

// Complete binding packet for one logical path-program dispatch. Keeping this
// packet scheduler-neutral makes the session batching policy independent of
// whether the program is submitted as one shader or through Luisa's coroutine
// scheduler.
struct PathKernelDispatch {
    luisa::compute::BufferView<luisa::float4> combined;
    luisa::compute::BufferView<luisa::float4> normal;
    luisa::compute::BufferView<luisa::float4> albedo;
    luisa::compute::BufferView<luisa::float4> light_passes;
    luisa::compute::BufferView<luisa::uint> sample_count;
    luisa::compute::BufferView<luisa::float4> volume_guiding_raw;
    luisa::compute::BufferView<luisa::uint> volume_guiding_denoised;
    luisa::compute::BufferView<luisa::float4> path_trace;
    std::uint32_t sample_first{};
    std::uint32_t samples{};
    luisa::compute::BufferView<luisa::float4> sobol_table;
    luisa::compute::BufferView<float> filter_table;
    RenderKernelParameters parameters{};
    // Logical launch shape for a contiguous row band. Per-sample execution
    // uses (width, height, samples), with samples mapped exclusively through
    // dispatch.z; the serial megakernel dispatches pixel_count invocations.
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t pixel_count{};
};

class PathKernelExecutor final {

private:
    std::unique_ptr<PathKernelExecutorImpl> _impl;

private:
    explicit PathKernelExecutor(
        std::unique_ptr<PathKernelExecutorImpl> impl) noexcept;

  friend PathKernelExecutor
  build_path_kernel_executor(luisa::compute::Device &device,
        const PathKernelConfig &path,
        const PathKernelExecutorConfig &config);

public:
    PathKernelExecutor() noexcept;
    ~PathKernelExecutor() noexcept;
    PathKernelExecutor(PathKernelExecutor &&) noexcept;
    PathKernelExecutor &operator=(PathKernelExecutor &&) noexcept;
    PathKernelExecutor(const PathKernelExecutor &) = delete;
    PathKernelExecutor &operator=(const PathKernelExecutor &) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] LuisaPathScheduler scheduler() const noexcept;

  void dispatch(luisa::compute::Stream &stream,
        const PathKernelDispatch &dispatch) const noexcept;
};

[[nodiscard]] PathKernelExecutor
build_path_kernel_executor(luisa::compute::Device &device,
    const PathKernelConfig &path,
    const PathKernelExecutorConfig &config);

}// namespace psycles::luisa_backend::detail
