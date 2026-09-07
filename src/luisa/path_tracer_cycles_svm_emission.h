#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

// Cycles surface_shader_constant_emission: the native KernelShader record is
// the authority, including a black constant emitter. No graph is evaluated.
[[nodiscard]] Bool cycles_svm_constant_emission(
    const LuisaSceneData &scene, UInt shader, Float3 &emission) noexcept;

// Forward and volume-light consumers already have the sampled lamp point.
// Preserve shader_setup_from_sample's input P/Ng/u/v rather than converting
// it to the retired SurfacePoint/parameter-block evaluator.
[[nodiscard]] Float3 evaluate_cycles_svm_lamp_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    UInt shader, UInt object, UInt primitive,
    Float3 position, Float3 normal, Float3 incoming, Float2 uv,
    Float distance, Float time, const cycles_svm::PathState &state) noexcept;

} // namespace psycles::luisa_backend::detail
