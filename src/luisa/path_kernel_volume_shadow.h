#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

// Host-stage shadow-volume transport. Surface transparency remains the
// responsibility of TraceShadowCallable; this component independently walks
// the same ordered boundary intervals, updates a shadow-visible stack, and
// selects Cycles' analytic or residual-ratio estimator per interval.
class VolumeShadowComponent {

  public:
    virtual ~VolumeShadowComponent() noexcept =
        default;

    [[nodiscard]] virtual Float3
    emit(const PathSampleContext &sample,
         const VolumeStack &path_stack,
         Var<luisa::compute::Ray> shadow_ray,
         UInt light_instance,
         UInt light_primitive) const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<VolumeShadowComponent>
make_volume_shadow_component(
    const PathKernelConfig &config);

}// namespace psycles::luisa_backend::detail
