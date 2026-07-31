#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

// Host-stage shadow transport for the homogeneous subset. Surface
// transparency remains the responsibility of TraceShadowCallable; this
// component independently integrates extinction over the same ordered
// boundary intervals. Multiplication is commutative, so the two traversals
// preserve the estimator while keeping volume-stack state local to the
// calling fused path kernel.
class HomogeneousVolumeShadowComponent {

  public:
    virtual ~HomogeneousVolumeShadowComponent() noexcept =
        default;

    [[nodiscard]] virtual Float3
    emit(const PathSampleContext &sample,
         const VolumeStack &path_stack,
         Var<luisa::compute::Ray> shadow_ray,
         UInt light_instance,
         UInt light_primitive) const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<HomogeneousVolumeShadowComponent>
make_homogeneous_volume_shadow_component(
    const PathKernelConfig &config);

}// namespace psycles::luisa_backend::detail
