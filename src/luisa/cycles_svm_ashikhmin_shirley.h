/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] BsdfEvaluation bsdf_ashikhmin_shirley_eval(
    const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_ashikhmin_shirley_sample(
    const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
