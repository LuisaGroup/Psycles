#pragma once

#include "path_kernel_caustics_manifold.h"
#include "path_kernel_direct_light_queue.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct PathKernelConfig;

struct CausticsTaskQueueBinding {
  std::shared_ptr<const CausticsTaskSink> sink;
  std::shared_ptr<RenderAuxiliaryWork> work;
};

[[nodiscard]] CausticsTaskQueueBinding
make_caustics_task_queue(
    luisa::compute::Device &device,
    const PathKernelConfig &config,
    std::uint32_t capacity,
    std::uint32_t execution_block_size,
    const luisa::compute::ShaderOption &shader_option) noexcept;

} // namespace psycles::luisa_backend::detail
