#pragma once

#include "path_kernel_builder.h"
#include "path_kernel_direct_light_task.h"

#include <memory>

#include <luisa/coro/schedulers/wavefront_auxiliary.h>
#include <luisa/dsl/soa.h>

namespace psycles::luisa_backend::detail {

template <typename Signature> struct RenderAuxiliaryWorkTypes;

template <typename... Args> struct RenderAuxiliaryWorkTypes<void(Args...)> {
  using Work = luisa::compute::coro::WavefrontCoroAuxiliaryWork<Args...>;
};

using RenderAuxiliaryWork = RenderAuxiliaryWorkTypes<RenderKernelSignature>::Work;

// Both views own the same queue object. The sink is captured while recording
// the canonical path pipeline; the scheduler view controls only host-side
// admission, draining, and dispatch order.
struct DirectLightTaskQueueBinding {
  std::shared_ptr<const DirectLightTaskSink> sink;
  std::shared_ptr<RenderAuxiliaryWork> work;
};

// Interpret the queue's complete backing allocation with a device-provided
// capacity. SOA member bases are functions of this expression, not of the host
// allocation size, so changing the scheduler frame-pool capacity cannot alter
// the producer/consumer shader structure. The caller proves
// runtime_capacity <= the allocation capacity.
[[nodiscard]] inline auto make_runtime_direct_light_task_storage(
    const luisa::compute::SOA<DirectLightTaskCall> &storage,
    luisa::compute::Expr<std::uint32_t> runtime_capacity) noexcept {
  return luisa::compute::Expr<luisa::compute::SOA<DirectLightTaskCall>>{
      luisa::compute::Expr<luisa::compute::Buffer<std::uint32_t>>{storage.buffer()},
      luisa::compute::UInt{0u}, runtime_capacity, luisa::compute::UInt{0u}};
}

[[nodiscard]] DirectLightTaskQueueBinding
make_direct_light_task_queue(luisa::compute::Device &device,
                             DirectLightTaskEvaluator evaluator, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept;

[[nodiscard]] DirectLightTaskQueueBinding
make_direct_light_task_queue(luisa::compute::Device &device,
                             const PathKernelConfig &config, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept;

} // namespace psycles::luisa_backend::detail
