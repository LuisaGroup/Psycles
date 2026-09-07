#pragma once

#include "path_tracer_cycles_svm_background.h"

namespace psycles::luisa_backend::detail {

// The shader can observe camera projection and final render extent while
// baking. Each render session therefore owns its CDF, not the shared scene.
struct BackgroundSamplingDistribution {
    Buffer<luisa::float2> conditional;
    Buffer<luisa::float2> marginal;
};

[[nodiscard]] Float3 constant_environment_emission(
    const LuisaSceneData &scene, Float3 background) noexcept;
[[nodiscard]] Float3 evaluate_environment_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    Float3 origin, Float3 direction, Float differential, Float time,
    const cycles_svm::PathState &state, UInt lcg_state,
    CyclesSvmBackgroundEvaluation evaluation) noexcept;
[[nodiscard]] Float3 evaluate_background_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters, Float u, Float v) noexcept;

void configure_background_sampling(
    LuisaSceneData &scene,
    const SceneSnapshot &snapshot,
    bool include_environment) noexcept;

[[nodiscard]] std::shared_ptr<const BackgroundSamplingDistribution>
build_background_sampling_distribution(
    const std::shared_ptr<LuisaSceneData> &scene,
    Stream &stream, const RenderKernelParameters &parameters);

}// namespace psycles::luisa_backend::detail
