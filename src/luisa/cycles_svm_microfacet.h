#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] ClosurePool::Allocation
bsdf_allocate(ShaderData &shader_data,
              luisa::compute::Expr<luisa::float3> input_weight) noexcept;

[[nodiscard]] luisa::compute::Float3 maybe_ensure_valid_specular_reflection(
    const ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> normal) noexcept;

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

}// namespace psycles::luisa_backend::cycles_svm::detail
