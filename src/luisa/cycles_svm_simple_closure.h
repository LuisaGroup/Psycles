#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] OrenNayarParam
oren_nayar_param(luisa::compute::Expr<luisa::float3> color,
                 luisa::compute::Expr<float> normal_view,
                 luisa::compute::Expr<float> roughness) noexcept;

void diffuse_setup(
    ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> weight) noexcept;

void oren_nayar_setup(
    ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> weight,
    luisa::compute::Expr<float> roughness,
    luisa::compute::Expr<luisa::float3> color) noexcept;

void translucent_setup(
    ShaderData &shader_data,
    luisa::compute::Expr<luisa::float3> normal,
    luisa::compute::Expr<luisa::float3> weight) noexcept;

void transparent_setup(
    ShaderData &shader_data, const PathState &path_state,
    luisa::compute::Expr<luisa::float3> weight) noexcept;

}// namespace psycles::luisa_backend::cycles_svm::detail
