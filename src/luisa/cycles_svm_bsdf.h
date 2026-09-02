/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

/* Direct Luisa projections of the output arguments of Cycles 5.2.1
 * bsdf_eval() and bsdf_sample(). Values remain unweighted here: the surface
 * one-sample-model fold applies ShaderClosure::weight exactly once. */
struct BsdfEvaluation {
  luisa::compute::Float3 value;
  luisa::compute::Float pdf;
};

struct BsdfSample {
  luisa::compute::Float3 value;
  luisa::compute::Float3 wo;
  luisa::compute::Float pdf;
  luisa::compute::Float2 sampled_roughness;
  luisa::compute::Float eta;
  luisa::compute::UInt label;
};

} // namespace psycles::luisa_backend::cycles_svm::detail
