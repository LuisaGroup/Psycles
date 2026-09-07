#pragma once

#include <psycles/luisa/volume_phase_set.h>
#include <psycles/luisa/volume_stack.h>

namespace psycles::luisa_backend {

// Host-stage boundary between ShaderData/SVM and volume transport. A missing
// phase destination does not disable closure allocation: sigma_s still comes
// from the successfully allocated ShaderData closures.
class VolumeShaderEvaluator {
public:
    virtual ~VolumeShaderEvaluator() noexcept = default;
    [[nodiscard]] virtual VolumeCoefficients evaluate(
        const VolumeStack &stack, Float3 position,
        VolumePhaseSet *phases = nullptr) const noexcept = 0;
};

} // namespace psycles::luisa_backend
