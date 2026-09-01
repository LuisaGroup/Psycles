#pragma once

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

/* Cycles 5.2.1 svm_node_closure_bsdf's Principled transition. The host bool is
 * the same JIT specialization boundary as Cycles' node_feature_mask template:
 * true evaluates the surface BSDF path, false evaluates emission only. */
void node_principled_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                          Stack &stack, luisa::compute::Expr<float> mix_weight,
                          bool evaluate_bsdf, ShaderData &shader_data,
                          const PathState &path_state,
                          luisa::compute::Bool &supported) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
