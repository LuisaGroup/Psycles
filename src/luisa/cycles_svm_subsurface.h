/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

// integrate_surface's exact SSS-exit material-evaluation predicate. The
// caller separately handles the compile-time __SUBSURFACE__ feature guard.
[[nodiscard]] luisa::compute::Bool surface_shader_material_eval_required(
    luisa::compute::Expr<bool> subsurface_exit,
    luisa::compute::Expr<std::uint32_t> shader_flags) noexcept;

// Cycles integrator/subsurface.h. Reset closure metadata, not ShaderData.N
// or storage bytes: the exit closure normal may differ from sd.N, and that
// distinction is consumed by the ordinary BSDF bump-shadowing correction.
void subsurface_shader_data_setup(ShaderData &shader_data) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
