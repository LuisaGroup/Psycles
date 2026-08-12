#pragma once

#include "path_kernel_builder.h"
#include "path_kernel_direct_light_task.h"

#include <memory>

#include <luisa/coro/schedulers/wavefront_auxiliary.h>

namespace psycles::luisa_backend::detail {

template <typename Signature> struct RenderAuxiliaryWorkTypes;

template <typename... Args> struct RenderAuxiliaryWorkTypes<void(Args...)> {
  using Work = luisa::compute::coro::WavefrontCoroAuxiliaryWork<Args...>;
};

using RenderAuxiliaryWork =
    RenderAuxiliaryWorkTypes<RenderKernelSignature>::Work;

// Both views own the same queue object. The sink is captured while recording
// the canonical path pipeline; the scheduler view controls only host-side
// admission, draining, and dispatch order.
struct DirectLightTaskQueueBinding {
  std::shared_ptr<const DirectLightTaskSink> sink;
  std::shared_ptr<RenderAuxiliaryWork> work;
};

[[nodiscard]] DirectLightTaskQueueBinding make_direct_light_task_queue(
    luisa::compute::Device &device, const PathKernelConfig &config,
    std::uint32_t capacity,
    const luisa::compute::ShaderOption &shader_option) noexcept;

} // namespace psycles::luisa_backend::detail
