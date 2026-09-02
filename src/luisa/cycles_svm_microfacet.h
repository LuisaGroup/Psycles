#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] ClosurePool::Allocation
bsdf_allocate(ShaderData &shader_data,
              luisa::compute::Expr<luisa::float3> input_weight) noexcept;

[[nodiscard]] luisa::compute::Float3 maybe_ensure_valid_specular_reflection(
    const ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> normal) noexcept;

[[nodiscard]] luisa::compute::Float3 rotate_around_axis(
    luisa::compute::Expr<luisa::float3> point,
    luisa::compute::Expr<luisa::float3> axis,
    luisa::compute::Expr<float> angle) noexcept;

/* Exact Cycles 5.2.1 bsdf_microfacet_estimate_albedo() projections for the
 * discriminated ClosurePool payload variants. The returned factor excludes
 * ShaderClosure::weight. */
[[nodiscard]] luisa::compute::Float3 bsdf_microfacet_estimate_albedo(
    const KernelGlobals &kernel_globals, const MicrofacetClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<bool> reflection,
    luisa::compute::Expr<bool> transmission) noexcept;

[[nodiscard]] luisa::compute::Float3 bsdf_microfacet_estimate_albedo(
    const KernelGlobals &kernel_globals,
    const MicrofacetConductorClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<bool> reflection,
    luisa::compute::Expr<bool> transmission) noexcept;

[[nodiscard]] luisa::compute::Float3 bsdf_microfacet_estimate_albedo(
    const KernelGlobals &kernel_globals,
    const MicrofacetF82TintClosure &closure,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<bool> reflection,
    luisa::compute::Expr<bool> transmission) noexcept;

/* Exact Cycles Principled dielectric layer transition. The returned albedo
 * includes the post-Multi-GGX closure weight and is therefore ready for
 * closure_layering_weight(). A failed allocation returns zero, matching the
 * source branch that leaves lower-layer weight unchanged. */
[[nodiscard]] luisa::compute::Float3 principled_specular_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> tangent,
    luisa::compute::Expr<float> alpha_x,
    luisa::compute::Expr<float> alpha_y, luisa::compute::Expr<float> eta,
    luisa::compute::Expr<float> f0,
    luisa::compute::Expr<luisa::float3> specular_tint,
    luisa::compute::Expr<float> thin_film_thickness,
    luisa::compute::Expr<float> thin_film_ior,
    luisa::compute::Expr<bool> preserve_energy) noexcept;

/* Exact Cycles Principled Coat dielectric-GGX transition. This includes
 * bsdf_alloc_maybe_emission, dielectric albedo estimation, and Cycles'
 * unconditional GGX multiple-scattering energy preservation. */
[[nodiscard]] luisa::compute::Float3 principled_coat_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const PathState &path_state,
    luisa::compute::Expr<luisa::float3> input_weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<float> roughness,
    luisa::compute::Expr<float> ior) noexcept;

/* Exact Cycles Principled Metallic F82-GGX transition. Distribution controls
 * only the Multi-GGX energy-preservation branch; Cycles stores the resulting
 * closure with the ordinary GGX runtime tag. */
void principled_metallic_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> tangent,
    luisa::compute::Expr<float> alpha_x,
    luisa::compute::Expr<float> alpha_y,
    luisa::compute::Expr<luisa::float3> base_color,
    luisa::compute::Expr<luisa::float3> f82_tint,
    luisa::compute::Expr<float> thin_film_thickness,
    luisa::compute::Expr<float> thin_film_ior,
    luisa::compute::Expr<bool> preserve_energy) noexcept;

/* Exact Cycles thick-walled Principled Transmission generalized-Schlick GGX
 * glass transition. Reflection and refraction caustics independently mask
 * their Fresnel tints while sharing one allocated glass closure. */
void principled_transmission_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<float> roughness,
    luisa::compute::Expr<float> ior,
    luisa::compute::Expr<bool> reflective_caustics,
    luisa::compute::Expr<bool> refractive_caustics,
    luisa::compute::Expr<luisa::float3> specular_tint,
    luisa::compute::Expr<luisa::float3> transmission_tint,
    luisa::compute::Expr<float> thin_film_thickness,
    luisa::compute::Expr<float> thin_film_ior,
    luisa::compute::Expr<bool> preserve_energy) noexcept;

/* Exact Cycles Principled thin-walled Transmission transition. Cycles first
 * sums the two-interface Fresnel series, then allocates independent GGX
 * reflection and type-22 mirrored-transmission closures. Near-singular
 * non-camera transmission is routed through bsdf_transparent_setup instead. */
void principled_thin_wall_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const PathState &path_state,
    luisa::compute::Expr<luisa::float3> weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<float> alpha,
    luisa::compute::Expr<float> ior,
    luisa::compute::Expr<bool> reflective_caustics,
    luisa::compute::Expr<bool> refractive_caustics,
    luisa::compute::Expr<luisa::float3> specular_tint,
    luisa::compute::Expr<luisa::float3> transmission_tint,
    luisa::compute::Expr<float> thin_film_thickness,
    luisa::compute::Expr<float> thin_film_ior) noexcept;

void glass_setup(const KernelGlobals &kernel_globals, ShaderData &shader_data,
                 const PathState &path_state,
                 luisa::compute::Expr<std::uint32_t> input_type,
                 luisa::compute::Expr<float> mix_weight,
                 luisa::compute::Expr<luisa::float3> normal,
                 luisa::compute::Expr<luisa::float3> color,
                 luisa::compute::Expr<float> roughness,
                 luisa::compute::Expr<float> ior,
                 luisa::compute::Expr<float> thin_film_thickness,
                 luisa::compute::Expr<float> thin_film_ior) noexcept;

void glossy_setup(const KernelGlobals &kernel_globals, ShaderData &shader_data,
                  const PathState &path_state,
                  luisa::compute::Expr<std::uint32_t> input_type,
                  luisa::compute::Expr<float> mix_weight,
                  luisa::compute::Expr<luisa::float3> closure_weight,
                  luisa::compute::Expr<luisa::float3> normal,
                  luisa::compute::Expr<luisa::float3> color,
                  luisa::compute::Expr<float> roughness,
                  luisa::compute::Expr<float> anisotropy,
                  luisa::compute::Expr<float> rotation,
                  luisa::compute::Expr<luisa::float3> tangent,
                  luisa::compute::Expr<bool> tangent_valid) noexcept;

void refraction_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const PathState &path_state,
    luisa::compute::Expr<std::uint32_t> input_type,
    luisa::compute::Expr<float> mix_weight,
    luisa::compute::Expr<luisa::float3> closure_weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<float> roughness,
    luisa::compute::Expr<float> ior) noexcept;

void metallic_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const PathState &path_state, luisa::compute::Expr<std::uint32_t> input_type,
    luisa::compute::Expr<std::uint32_t> distribution,
    luisa::compute::Expr<float> mix_weight,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> base_ior,
    luisa::compute::Expr<luisa::float3> edge_tint_k,
    luisa::compute::Expr<float> roughness,
    luisa::compute::Expr<float> anisotropy,
    luisa::compute::Expr<float> rotation,
    luisa::compute::Expr<float> thin_film_thickness,
    luisa::compute::Expr<float> thin_film_ior,
    luisa::compute::Expr<luisa::float3> tangent,
    luisa::compute::Expr<bool> tangent_valid) noexcept;

}// namespace psycles::luisa_backend::cycles_svm::detail
