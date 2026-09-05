#pragma once

#include "path_kernel_builder.h"

#include <memory>

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

[[nodiscard]] std::uint32_t
surface_queue_key_range(const LuisaSceneData &scene) noexcept;

}// namespace psycles::luisa_backend::detail
