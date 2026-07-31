#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/heterogeneous_volume_collision.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <memory>

#include <psycles/luisa/stacked_volume.h>

namespace psycles::luisa_backend {

// Host-stage source of original closure coefficients at a candidate point.
// Production implementations evaluate the active VolumeStack at `distance`;
// no coefficient grid or material pre-bake is permitted at this boundary.
class HeterogeneousVolumeCollisionProvider {

  public:
    virtual ~HeterogeneousVolumeCollisionProvider()
        noexcept = default;

    [[nodiscard]] virtual VolumeCoefficients
    evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases) const noexcept = 0;
};

// Production collision provider for original stacked Volume graphs. The base
// state carries path semantics while position is reconstructed from the ray
// and candidate distance on every evaluation.
[[nodiscard]]
std::unique_ptr<HeterogeneousVolumeCollisionProvider>
make_stacked_heterogeneous_volume_collision_provider(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const VolumeStack &stack,
    const ShaderServices &services,
    const VolumeShadingState &base_state,
    Float3 ray_origin,
    Float3 ray_direction);

}// namespace psycles::luisa_backend
