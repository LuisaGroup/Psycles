#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] luisa::compute::Float3
bsdf_principled_hair_sigma_from_reflectance(
    luisa::compute::Expr<luisa::float3> color,
    luisa::compute::Expr<float> azimuthal_roughness) noexcept;

[[nodiscard]] luisa::compute::Float3
bsdf_principled_hair_sigma_from_concentration(
    luisa::compute::Expr<float> eumelanin,
    luisa::compute::Expr<float> pheomelanin) noexcept;

/* Exact shared CLOSURE_BSDF_HAIR_CHIANG_ID / CLOSURE_BSDF_HAIR_HUANG_ID
 * transition from Cycles 5.2.1 svm_node_closure_bsdf(). The cursor initially
 * names the first word of SVMNodePrincipledHairBsdfData and leaves it
 * immediately after all 26 words. */
void node_principled_hair(
    const KernelGlobals &kernel_globals, Cursor &cursor, Stack &stack,
    luisa::compute::Expr<std::uint32_t> closure_type,
    luisa::compute::Expr<luisa::float3> closure_weight,
    luisa::compute::Expr<float> mix_weight, ShaderData &shader_data,
    const PathState &path_state) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
