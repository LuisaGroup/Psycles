#pragma once

#include "path_kernel_builder.h"

#include <memory>

#include <luisa/coro/schedulers/wavefront_extension.h>

namespace psycles::luisa_backend::detail {

// Resolves the Cycles shader identity (or the topology tag for the legacy
// diagnostic evaluator) at the traversal-to-surface boundary. This is
// scheduler metadata only; surface geometry, graph evaluation, closures, and
// sampling remain in their existing
// PathKernelPipeline stages.
class SurfaceQueueKeyStage {

public:
  virtual ~SurfaceQueueKeyStage() noexcept = default;

  [[nodiscard]] virtual UInt
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<SurfaceQueueKeyStage>
make_surface_queue_key_stage(ScenePrimitiveStagePlan plan);

[[nodiscard]] std::uint32_t surface_queue_key_range(const LuisaSceneData &scene) noexcept;

[[nodiscard]] luisa::compute::CoroSuspendExtensionPtr
make_surface_sort_annotation(UInt shader, std::uint32_t shader_count,
                             bool native_cycles) noexcept;

// Cycles locality policy belongs to Psycles. The SDK sees only the generic
// typed binding and the exact pre-resume queue handoff.
[[nodiscard]] luisa::unique_ptr<
    luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>
make_surface_sort_handler(
    luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
    const luisa::compute::coro::WavefrontCoroExtensionStage &stage) noexcept;

} // namespace psycles::luisa_backend::detail
