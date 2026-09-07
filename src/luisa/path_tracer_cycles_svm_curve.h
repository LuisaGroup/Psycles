#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

// Recover the procedural hit's Cycles curve/segment/u/v identity, then record
// curve_shader_setup. Geometry is read from KernelCurve/curve_keys, not from
// the former eager SurfacePoint attribute projection.
void cycles_svm_curve_shader_setup(
    const std::shared_ptr<LuisaSceneData> &scene,
    const cycles_svm::KernelGlobals &kernel_globals,
    const cycles_svm::TransformState &transforms,
    const Var<InstanceGpu> &instance, Expr<unsigned> segment_id,
    const Var<luisa::compute::Ray> &ray,
    cycles_svm::ShaderData &shader_data) noexcept;

} // namespace psycles::luisa_backend::detail
