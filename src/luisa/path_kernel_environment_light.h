#pragma once

#include "path_kernel_builder.h"

#include <memory>

namespace psycles::luisa_backend::detail {

// Directional proposal only. World closure evaluation is intentionally kept
// out of this value so invalid proposals cannot evaluate shader AST early.
struct EnvironmentLightProposal {
    Float3 direction;
    Float pdf;
    Bool valid;
};

// Shared host-stage component for Cycles' background-light directional
// measure. Raw world-closure evaluation is a separate operation. The reference
// position is part of the proposal contract because portal sampling is
// position-dependent, even though the current map/sun subset does not use it.
class EnvironmentLightComponent {

  public:
    virtual ~EnvironmentLightComponent() noexcept =
        default;

    [[nodiscard]] virtual EnvironmentLightProposal
    from_position(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float2 random,
        Float selection_pdf)
        const noexcept = 0;

    [[nodiscard]] virtual Float3
    evaluate_emission(
        PathSampleContext &sample,
        Float3 direction,
        const cycles_path_state::
            ShaderEvaluationState
                &shader_state)
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
