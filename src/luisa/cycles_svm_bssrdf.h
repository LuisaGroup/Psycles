#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

/* Exact Cycles 5.2.1 Bssrdf allocation and setup transition. Input radius is
 * already scaled and clamped by the SVM node handler. */
void bssrdf_setup(ShaderData &shader_data, const PathState &path_state,
                  luisa::compute::Expr<std::uint32_t> type,
                  luisa::compute::Expr<luisa::float3> weight,
                  luisa::compute::Expr<luisa::float3> radius,
                  luisa::compute::Expr<luisa::float3> albedo,
                  luisa::compute::Expr<luisa::float3> normal,
                  luisa::compute::Expr<float> alpha,
                  luisa::compute::Expr<float> ior,
                  luisa::compute::Expr<float> anisotropy) noexcept;

/* Cycles' OpenPBR thin-wall subsurface transition. It emits two ordinary
 * surface closures and never allocates a Bssrdf record. */
void thin_subsurface_setup(ShaderData &shader_data,
                           luisa::compute::Expr<luisa::float3> normal,
                           luisa::compute::Expr<luisa::float3> weight,
                           luisa::compute::Expr<float> anisotropy,
                           luisa::compute::Expr<float> roughness,
                           luisa::compute::Expr<luisa::float3> color) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
