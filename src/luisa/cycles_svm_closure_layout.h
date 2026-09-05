#pragma once

#include <cstdint>

// Generated from the device-side sizeof/offsetof values emitted by
// tools/cycles_svm_closure_pool_oracle.hip against Cycles 5.2.1 on HIP.
// Keep the byte offsets, including pointer-field positions, independent of
// host float3 padding. Pointer values are projected as indices into this pool.
#define PSYCLES_CYCLES_CLOSURE_LAYOUT_FIELDS(X)                                \
  X(ShaderClosure_sizeof, "ShaderClosure.sizeof", 80u)                         \
  X(ShaderClosure_weight, "ShaderClosure.weight", 0u)                          \
  X(ShaderClosure_type, "ShaderClosure.type", 12u)                             \
  X(ShaderClosure_sample_weight, "ShaderClosure.sample_weight", 16u)           \
  X(ShaderClosure_N, "ShaderClosure.N", 20u)                                   \
  X(OrenNayarBsdf_sizeof, "OrenNayarBsdf.sizeof", 56u)                         \
  X(OrenNayarBsdf_param, "OrenNayarBsdf.param", 32u)                           \
  X(OrenNayarParam_roughness, "OrenNayarParam.roughness", 0u)                  \
  X(OrenNayarParam_a, "OrenNayarParam.a", 4u)                                  \
  X(OrenNayarParam_b, "OrenNayarParam.b", 8u)                                  \
  X(OrenNayarParam_multiscatter_term, "OrenNayarParam.multiscatter_term", 12u) \
  X(SheenBsdf_sizeof, "SheenBsdf.sizeof", 68u)                                 \
  X(SheenBsdf_roughness, "SheenBsdf.roughness", 32u)                           \
  X(SheenBsdf_transformA, "SheenBsdf.transformA", 36u)                         \
  X(SheenBsdf_transformB, "SheenBsdf.transformB", 40u)                         \
  X(SheenBsdf_T, "SheenBsdf.T", 44u)                                           \
  X(SheenBsdf_B, "SheenBsdf.B", 56u)                                           \
  X(VelvetBsdf_sizeof, "VelvetBsdf.sizeof", 40u)                               \
  X(VelvetBsdf_sigma, "VelvetBsdf.sigma", 32u)                                 \
  X(VelvetBsdf_invsigma2, "VelvetBsdf.invsigma2", 36u)                         \
  X(ToonBsdf_sizeof, "ToonBsdf.sizeof", 40u)                                   \
  X(ToonBsdf_size, "ToonBsdf.size", 32u)                                       \
  X(ToonBsdf_smooth, "ToonBsdf.smooth", 36u)                                   \
  X(RayPortalClosure_sizeof, "RayPortalClosure.sizeof", 56u)                   \
  X(RayPortalClosure_P, "RayPortalClosure.P", 32u)                             \
  X(RayPortalClosure_D, "RayPortalClosure.D", 44u)                             \
  X(HairBsdf_sizeof, "HairBsdf.sizeof", 56u)                                   \
  X(HairBsdf_T, "HairBsdf.T", 32u)                                             \
  X(HairBsdf_roughness1, "HairBsdf.roughness1", 44u)                           \
  X(HairBsdf_roughness2, "HairBsdf.roughness2", 48u)                           \
  X(HairBsdf_offset, "HairBsdf.offset", 52u)                                   \
  X(ChiangHairBSDF_sizeof, "ChiangHairBSDF.sizeof", 68u)                       \
  X(ChiangHairBSDF_sigma, "ChiangHairBSDF.sigma", 32u)                         \
  X(ChiangHairBSDF_v, "ChiangHairBSDF.v", 44u)                                 \
  X(ChiangHairBSDF_s, "ChiangHairBSDF.s", 48u)                                 \
  X(ChiangHairBSDF_alpha, "ChiangHairBSDF.alpha", 52u)                         \
  X(ChiangHairBSDF_eta, "ChiangHairBSDF.eta", 56u)                             \
  X(ChiangHairBSDF_m0_roughness, "ChiangHairBSDF.m0_roughness", 60u)           \
  X(ChiangHairBSDF_h, "ChiangHairBSDF.h", 64u)                                 \
  X(HuangHairBSDF_sizeof, "HuangHairBSDF.sizeof", 72u)                         \
  X(HuangHairBSDF_sigma, "HuangHairBSDF.sigma", 32u)                           \
  X(HuangHairBSDF_roughness, "HuangHairBSDF.roughness", 44u)                   \
  X(HuangHairBSDF_tilt, "HuangHairBSDF.tilt", 48u)                             \
  X(HuangHairBSDF_eta, "HuangHairBSDF.eta", 52u)                               \
  X(HuangHairBSDF_aspect_ratio, "HuangHairBSDF.aspect_ratio", 56u)             \
  X(HuangHairBSDF_h, "HuangHairBSDF.h", 60u)                                   \
  X(HuangHairBSDF_extra, "HuangHairBSDF.extra", 64u)                           \
  X(HuangHairExtra_sizeof, "HuangHairExtra.sizeof", 68u)                       \
  X(HuangHairExtra_R, "HuangHairExtra.R", 0u)                                  \
  X(HuangHairExtra_TT, "HuangHairExtra.TT", 4u)                                \
  X(HuangHairExtra_TRT, "HuangHairExtra.TRT", 8u)                              \
  X(HuangHairExtra_Y, "HuangHairExtra.Y", 12u)                                 \
  X(HuangHairExtra_Z, "HuangHairExtra.Z", 24u)                                 \
  X(HuangHairExtra_wi, "HuangHairExtra.wi", 36u)                               \
  X(HuangHairExtra_radius, "HuangHairExtra.radius", 48u)                       \
  X(HuangHairExtra_e2, "HuangHairExtra.e2", 52u)                               \
  X(HuangHairExtra_pixel_coverage, "HuangHairExtra.pixel_coverage", 56u)       \
  X(HuangHairExtra_h, "HuangHairExtra.h", 60u)                                 \
  X(Bssrdf_sizeof, "Bssrdf.sizeof", 68u)                                       \
  X(Bssrdf_radius, "Bssrdf.radius", 32u)                                       \
  X(Bssrdf_albedo, "Bssrdf.albedo", 44u)                                       \
  X(Bssrdf_anisotropy, "Bssrdf.anisotropy", 56u)                               \
  X(Bssrdf_ior, "Bssrdf.ior", 60u)                                             \
  X(Bssrdf_alpha, "Bssrdf.alpha", 64u)                                         \
  X(MicrofacetBsdf_sizeof, "MicrofacetBsdf.sizeof", 80u)                       \
  X(MicrofacetBsdf_alpha_x, "MicrofacetBsdf.alpha_x", 32u)                     \
  X(MicrofacetBsdf_alpha_y, "MicrofacetBsdf.alpha_y", 36u)                     \
  X(MicrofacetBsdf_ior, "MicrofacetBsdf.ior", 40u)                             \
  X(MicrofacetBsdf_energy_scale, "MicrofacetBsdf.energy_scale", 44u)           \
  X(MicrofacetBsdf_fresnel_type, "MicrofacetBsdf.fresnel_type", 48u)           \
  X(MicrofacetBsdf_fresnel, "MicrofacetBsdf.fresnel", 56u)                     \
  X(MicrofacetBsdf_T, "MicrofacetBsdf.T", 64u)                                 \
  X(FresnelThinFilm_sizeof, "FresnelThinFilm.sizeof", 8u)                      \
  X(FresnelThinFilm_thickness, "FresnelThinFilm.thickness", 0u)                \
  X(FresnelThinFilm_ior, "FresnelThinFilm.ior", 4u)                            \
  X(FresnelGeneralizedSchlick_sizeof, "FresnelGeneralizedSchlick.sizeof", 60u) \
  X(FresnelGeneralizedSchlick_thin_film,                                       \
    "FresnelGeneralizedSchlick.thin_film", 0u)                                 \
  X(FresnelGeneralizedSchlick_reflection_tint,                                 \
    "FresnelGeneralizedSchlick.reflection_tint", 8u)                           \
  X(FresnelGeneralizedSchlick_transmission_tint,                               \
    "FresnelGeneralizedSchlick.transmission_tint", 20u)                        \
  X(FresnelGeneralizedSchlick_f0, "FresnelGeneralizedSchlick.f0", 32u)         \
  X(FresnelGeneralizedSchlick_f90, "FresnelGeneralizedSchlick.f90", 44u)       \
  X(FresnelGeneralizedSchlick_exponent, "FresnelGeneralizedSchlick.exponent",  \
    56u)                                                                       \
  X(FresnelConductor_sizeof, "FresnelConductor.sizeof", 32u)                   \
  X(FresnelConductor_thin_film, "FresnelConductor.thin_film", 0u)              \
  X(FresnelConductor_ior, "FresnelConductor.ior", 8u)                          \
  X(complex_Spectrum_re, "complex<Spectrum>.re", 0u)                           \
  X(complex_Spectrum_im, "complex<Spectrum>.im", 12u)                          \
  X(FresnelF82Tint_sizeof, "FresnelF82Tint.sizeof", 32u)                       \
  X(FresnelF82Tint_thin_film, "FresnelF82Tint.thin_film", 0u)                  \
  X(FresnelF82Tint_f0, "FresnelF82Tint.f0", 8u)                                \
  X(FresnelF82Tint_b, "FresnelF82Tint.b", 20u)

namespace psycles::luisa_backend::cycles_svm::detail::closure_layout {
#define PSYCLES_CLOSURE_LAYOUT_CONSTANT(symbol, source, value)                 \
  inline constexpr std::uint32_t symbol = value;
PSYCLES_CYCLES_CLOSURE_LAYOUT_FIELDS(PSYCLES_CLOSURE_LAYOUT_CONSTANT)
#undef PSYCLES_CLOSURE_LAYOUT_CONSTANT
} // namespace psycles::luisa_backend::cycles_svm::detail::closure_layout
