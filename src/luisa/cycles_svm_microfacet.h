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
