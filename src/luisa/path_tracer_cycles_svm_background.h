#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

enum class CyclesSvmBackgroundEvaluation { forward, light, importance_bake };

[[nodiscard]] std::uint32_t cycles_svm_background_feature_mask(
    CyclesSvmBackgroundEvaluation evaluation) noexcept;

// shader_setup_from_background, plus the entry-specific compact differential
// clamp. No SurfacePoint, material replay or object attribute preparation.
[[nodiscard]] cycles_svm::ShaderData setup_cycles_svm_background_shader_data(
    const LuisaSceneData &scene, Float3 origin, Float3 direction,
    Float differential, Float time, UInt visibility,
    CyclesSvmBackgroundEvaluation evaluation) noexcept;

struct CyclesSvmBackgroundBakeRay {
  Float3 direction;
  Float differential;
};
[[nodiscard]] CyclesSvmBackgroundBakeRay cycles_svm_background_bake_ray(
    Float u, Float v, std::uint32_t width, std::uint32_t height) noexcept;

[[nodiscard]] Float3 evaluate_cycles_svm_background_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    Float3 origin, Float3 direction, Float differential, Float time,
    const cycles_svm::PathState &state, UInt lcg_state,
    CyclesSvmBackgroundEvaluation evaluation) noexcept;

} // namespace psycles::luisa_backend::detail
