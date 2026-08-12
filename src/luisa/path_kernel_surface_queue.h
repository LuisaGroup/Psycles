#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

// Resolves the structure-deduplicated material-program tag at the narrow
// traversal-to-surface boundary. This is scheduler metadata only; surface
// geometry, graph evaluation, closures, and sampling remain in their existing
// PathKernelPipeline stages.
class SurfaceQueueKeyStage {

  public:
    virtual ~SurfaceQueueKeyStage() noexcept = default;

    [[nodiscard]] virtual UInt
    emit(const PathBounceContext &bounce) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<SurfaceQueueKeyStage>
make_surface_queue_key_stage(ScenePrimitiveStagePlan plan);

}// namespace psycles::luisa_backend::detail
