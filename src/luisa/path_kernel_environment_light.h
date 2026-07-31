#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

struct EnvironmentLightSample {
    Float3 direction;
    Float3 radiance;
    Float pdf;
    Bool valid;
};

// Shared host-stage component for Cycles' background-light directional
// measure and raw world-closure evaluation. The reference position is part of
// the contract because portal sampling is position-dependent, even though the
// current map/sun subset does not use it.
class EnvironmentLightComponent {

  public:
    virtual ~EnvironmentLightComponent() noexcept =
        default;

    [[nodiscard]] virtual EnvironmentLightSample
    from_position(
        PathSampleContext &sample,
        Float3 reference,
        Float2 random,
        Float selection_pdf)
        const noexcept = 0;

    [[nodiscard]] virtual Float
    from_direction(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float3 direction,
        Float selection_pdf)
        const noexcept = 0;
};

[[nodiscard]]
std::shared_ptr<
    const EnvironmentLightComponent>
make_environment_light_component();

}// namespace psycles::luisa_backend::detail
