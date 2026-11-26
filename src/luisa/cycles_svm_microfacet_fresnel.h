/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

struct DielectricFresnel {
  luisa::compute::Float reflectance;
  luisa::compute::Float cosine_transmitted;
};

struct MicrofacetFresnelEvaluation {
  luisa::compute::Float3 reflectance;
  luisa::compute::Float3 transmittance;
  luisa::compute::Float cosine_transmitted;
};

struct MicrofacetFresnelSpectralEvaluation {
  luisa::compute::Float4 reflectance;
  luisa::compute::Float4 transmittance;
  luisa::compute::Float4 cosine_transmitted;
};

[[nodiscard]] luisa::compute::Float
f0_from_ior(luisa::compute::Expr<float> ior) noexcept;

[[nodiscard]] DielectricFresnel
fresnel_dielectric(luisa::compute::Expr<float> cosine_incoming,
                   luisa::compute::Expr<float> ior) noexcept;

[[nodiscard]] luisa::compute::Float3
fresnel_conductor(luisa::compute::Expr<float> cosine_incoming,
                  luisa::compute::Expr<luisa::float3> ior,
                  luisa::compute::Expr<luisa::float3> extinction) noexcept;

[[nodiscard]] luisa::compute::Float3
fresnel_f82(luisa::compute::Expr<float> cosine_incoming,
            luisa::compute::Expr<luisa::float3> f0,
            luisa::compute::Expr<luisa::float3> b) noexcept;

/* Exact Cycles 5.2.1 MicrofacetFresnel dispatch for each retained typed
 * payload. The C++ overload is the tagged-union projection; only
 * MicrofacetClosure retains the runtime NONE/DIELECTRIC/GENERALIZED_SCHLICK
 * discriminator used by SVM. */
[[nodiscard]] MicrofacetFresnelEvaluation
microfacet_fresnel(const KernelGlobals &kernel_globals,
                   const MicrofacetClosure &closure,
                   luisa::compute::Expr<float> cosine_incoming,
                   bool may_transmit = true) noexcept;

[[nodiscard]] MicrofacetFresnelEvaluation
microfacet_fresnel(const KernelGlobals &kernel_globals,
                   const MicrofacetConductorClosure &closure,
                   luisa::compute::Expr<float> cosine_incoming,
                   bool may_transmit = false) noexcept;

[[nodiscard]] MicrofacetFresnelEvaluation
microfacet_fresnel(const KernelGlobals &kernel_globals,
                   const MicrofacetF82TintClosure &closure,
                   luisa::compute::Expr<float> cosine_incoming,
                   bool may_transmit = false) noexcept;

[[nodiscard]] MicrofacetFresnelSpectralEvaluation
microfacet_fresnel_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<float> cosine_incoming,
    luisa::compute::Expr<luisa::float4> eta_lambda,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<bool> is_dispersive,
    bool may_transmit = true) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
