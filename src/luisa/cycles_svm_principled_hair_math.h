/* SPDX-FileCopyrightText: 2018-2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <luisa/dsl/expr.h>
#include <luisa/dsl/var.h>

namespace psycles::luisa_backend::cycles_svm::detail::principled_hair_math {

/* Shared Cycles 5.2.1 Principled Hair longitudinal distribution. Chiang and
 * Huang use the same finite Bessel approximation and variance branches. */
[[nodiscard]] luisa::compute::Float
bessel_I0(luisa::compute::Expr<float> argument) noexcept;

[[nodiscard]] luisa::compute::Float
log_bessel_I0(luisa::compute::Expr<float> argument) noexcept;

[[nodiscard]] luisa::compute::Float
longitudinal_scattering(luisa::compute::Expr<float> sine_incoming,
                        luisa::compute::Expr<float> cosine_incoming,
                        luisa::compute::Expr<float> sine_outgoing,
                        luisa::compute::Expr<float> cosine_outgoing,
                        luisa::compute::Expr<float> variance) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail::principled_hair_math
