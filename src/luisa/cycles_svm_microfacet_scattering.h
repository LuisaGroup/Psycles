/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_microfacet_ggx_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_ggx_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_microfacet_beckmann_eval(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfSample bsdf_microfacet_beckmann_sample(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfEvaluation bsdf_thin_glass_transmission_eval(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_thin_glass_transmission_sample(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random) noexcept;

[[nodiscard]] BsdfSpectralEvaluation bsdf_microfacet_ggx_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

[[nodiscard]] BsdfSpectralSample bsdf_microfacet_ggx_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

[[nodiscard]] BsdfSpectralEvaluation bsdf_microfacet_beckmann_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

[[nodiscard]] BsdfSpectralSample bsdf_microfacet_beckmann_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

[[nodiscard]] BsdfSpectralEvaluation bsdf_thin_glass_transmission_eval_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> wo,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

[[nodiscard]] BsdfSpectralSample bsdf_thin_glass_transmission_sample_spectral(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float3> random,
    const cycles_spectrum::SpectralContext &spectral,
    luisa::compute::Expr<float> dispersion_abbe) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
