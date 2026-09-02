#pragma once

#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::cycles_svm::detail {

[[nodiscard]] OrenNayarParam
oren_nayar_param(luisa::compute::Expr<luisa::float3> color,
                 luisa::compute::Expr<float> normal_view,
                 luisa::compute::Expr<float> roughness) noexcept;

void diffuse_setup(ShaderData &shader_data,
                   luisa::compute::Expr<luisa::float3> normal,
                   luisa::compute::Expr<luisa::float3> weight) noexcept;

void oren_nayar_setup(ShaderData &shader_data,
                      luisa::compute::Expr<luisa::float3> normal,
                      luisa::compute::Expr<luisa::float3> weight,
                      luisa::compute::Expr<float> roughness,
                      luisa::compute::Expr<luisa::float3> color) noexcept;

void translucent_setup(ShaderData &shader_data,
                       luisa::compute::Expr<luisa::float3> normal,
                       luisa::compute::Expr<luisa::float3> weight) noexcept;

void transparent_setup(ShaderData &shader_data, const PathState &path_state,
                       luisa::compute::Expr<luisa::float3> weight) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_diffuse_eval(const ShaderClosureCommon &closure,
                  luisa::compute::Expr<luisa::float3> wi,
                  luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_diffuse_sample(const ShaderClosureCommon &closure,
                    luisa::compute::Expr<luisa::float3> Ng,
                    luisa::compute::Expr<luisa::float3> wi,
                    luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_translucent_eval(const ShaderClosureCommon &closure,
                      luisa::compute::Expr<luisa::float3> wi,
                      luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_translucent_sample(const ShaderClosureCommon &closure,
                        luisa::compute::Expr<luisa::float3> Ng,
                        luisa::compute::Expr<luisa::float3> wi,
                        luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_oren_nayar_eval(const OrenNayarClosure &closure,
                     luisa::compute::Expr<luisa::float3> wi,
                     luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_oren_nayar_sample(const OrenNayarClosure &closure,
                       luisa::compute::Expr<luisa::float3> Ng,
                       luisa::compute::Expr<luisa::float3> wi,
                       luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_rough_translucent_eval(const OrenNayarClosure &closure,
                            luisa::compute::Expr<luisa::float3> wi,
                            luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample bsdf_rough_translucent_sample(
    const OrenNayarClosure &closure, luisa::compute::Expr<luisa::float3> Ng,
    luisa::compute::Expr<luisa::float3> wi,
    luisa::compute::Expr<luisa::float2> random) noexcept;

[[nodiscard]] BsdfEvaluation
bsdf_transparent_eval(const ShaderClosureCommon &closure,
                      luisa::compute::Expr<luisa::float3> wi,
                      luisa::compute::Expr<luisa::float3> wo) noexcept;

[[nodiscard]] BsdfSample
bsdf_transparent_sample(const ShaderClosureCommon &closure,
                        luisa::compute::Expr<luisa::float3> Ng,
                        luisa::compute::Expr<luisa::float3> wi) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
