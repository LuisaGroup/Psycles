#pragma once

#include "path_tracer_geometry.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

struct CyclesSvmShadowShaderData {
  cycles_svm::ShaderData shader_data;
  cycles_svm::TransformState transforms;
};

// Cycles shader_setup_from_ray, with intersection barycentrics and native
// object/triangle identity. ShaderData and SVM scratch never cross a cut.
[[nodiscard]] CyclesSvmShadowShaderData setup_cycles_svm_shadow_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene,
    const cycles_svm::KernelGlobals &kernel_globals,
    const Var<luisa::compute::Ray> &ray,
    const Var<ShadowIntersectionCall> &intersection, Expr<float> ray_dP,
    Expr<float> ray_dD, const Var<ShadowShaderContextCall> &context,
    const Var<RenderKernelParameters> &parameters) noexcept;

[[nodiscard]] EvaluateShadowSurfaceCallable
make_cycles_svm_shadow_surface_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

} // namespace psycles::luisa_backend::detail
