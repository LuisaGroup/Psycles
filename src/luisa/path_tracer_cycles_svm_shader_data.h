#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

// The common Cycles triangle_shader_setup / shader_setup_from_ray tail.
// Callers provide the actual intersection (surface/shadow) or sampled-light
// barycentrics. These host helpers record DSL, not device call boundaries.
void cycles_svm_triangle_shader_setup(
    const cycles_svm::KernelGlobals &kernel_globals,
    const cycles_svm::TransformState &transforms,
    const cycles_svm::TriangleVertices &vertices,
    cycles_svm::ShaderData &shader_data) noexcept;

void cycles_svm_shader_setup_backfacing(
    cycles_svm::ShaderData &shader_data) noexcept;

void cycles_svm_shader_setup_dudv(cycles_svm::ShaderData &shader_data) noexcept;

} // namespace psycles::luisa_backend::detail
