#pragma once

#include "path_tracer_geometry.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

struct CyclesSvmShadowShaderData {
  cycles_svm::ShaderData shader_data;
  cycles_svm::TransformState transforms;
};

// Common shader_setup_from_ray for main and shadow surface hits. Identity,
// precomputed object transforms and packed attributes have one source.
[[nodiscard]] CyclesSvmShadowShaderData setup_cycles_svm_ray_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene,
    const cycles_svm::KernelGlobals &kernel_globals,
    const Var<luisa::compute::Ray> &ray, Expr<unsigned> instance,
    Expr<unsigned> primitive, Expr<luisa::float2> barycentric,
    Expr<float> distance, Expr<float> ray_dP, Expr<float> ray_dD,
    Expr<float> ray_time, Expr<unsigned> lcg_state,
    const Var<RenderKernelParameters> &parameters) noexcept;

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
