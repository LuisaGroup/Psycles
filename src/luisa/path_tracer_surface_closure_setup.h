#pragma once

#include <psycles/luisa/surface.h>

#include <luisa/dsl/struct.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>

namespace psycles::luisa_backend::detail {

struct PrincipledMetallicSetupCall {
    luisa::float3 weight{};
    luisa::float3 albedo{};
    luisa::float3 normal{};
    luisa::float3 color{};
    luisa::float3 specular_tint{};
    luisa::float3 evaluation_scale{};
    luisa::float3 lower_weight{};
    float allocation_weight{};
    float sample_weight{};
};

struct PrincipledMetallicSetupInputCall {
    luisa::float3 lower_weight{};
    luisa::float3 color{};
    luisa::float3 normal{};
    luisa::float3 incoming{};
    luisa::float3 surface_shading_normal{};
    luisa::float3 surface_geometric_normal{};
    luisa::float3 specular_tint{};
    float roughness{};
    float metallic{};
    bool use_bump_map_correction{};
};

struct PrincipledDiffuseSetupCall {
    luisa::float3 weight{};
    float allocation_weight{};
    float sample_weight{};
};

struct PrincipledDiffuseSetupInputCall {
    luisa::float3 lower_weight{};
    luisa::float3 color{};
    float subsurface_weight{};
};

struct PrincipledDielectricSetupCall {
    luisa::float3 weight{};
    float allocation_weight{};
    float sample_weight{};
    luisa::float3 albedo{};
    luisa::float3 normal{};
    luisa::float3 color{};
    float ior{};
    luisa::float3 evaluation_scale{};
    luisa::float3 lower_weight{};
};

struct PrincipledDielectricSetupInputCall {
    luisa::float3 lower_weight{};
    luisa::float3 normal{};
    luisa::float3 incoming{};
    luisa::float3 surface_shading_normal{};
    luisa::float3 surface_geometric_normal{};
    float roughness{};
    float ior{};
    float specular_ior_level{};
    luisa::float3 specular_tint{};
    bool use_bump_map_correction{};
};

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledMetallicSetupCall,
    weight,
    albedo,
    normal,
    color,
    specular_tint,
    evaluation_scale,
    lower_weight,
    allocation_weight,
    sample_weight) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledMetallicSetupInputCall,
    lower_weight,
    color,
    normal,
    incoming,
    surface_shading_normal,
    surface_geometric_normal,
    specular_tint,
    roughness,
    metallic,
    use_bump_map_correction) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledDiffuseSetupCall,
    weight,
    allocation_weight,
    sample_weight) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledDiffuseSetupInputCall,
    lower_weight,
    color,
    subsurface_weight) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledDielectricSetupCall,
    weight,
    allocation_weight,
    sample_weight,
    albedo,
    normal,
    color,
    ior,
    evaluation_scale,
    lower_weight) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::
        PrincipledDielectricSetupInputCall,
    lower_weight,
    normal,
    incoming,
    surface_shading_normal,
    surface_geometric_normal,
    roughness,
    ior,
    specular_ior_level,
    specular_tint,
    use_bump_map_correction) {};

namespace psycles::luisa_backend::detail {

using PrincipledMetallicSetupCallable =
    luisa::compute::Callable<PrincipledMetallicSetupCall(
        luisa::compute::Buffer<float>,
        PrincipledMetallicSetupInputCall,
        bool)>;
using PrincipledDiffuseSetupCallable =
    luisa::compute::Callable<PrincipledDiffuseSetupCall(
        PrincipledDiffuseSetupInputCall)>;
using PrincipledDielectricSetupCallable =
    luisa::compute::Callable<PrincipledDielectricSetupCall(
        luisa::compute::Buffer<float>,
        PrincipledDielectricSetupInputCall,
        bool)>;

struct SurfaceClosureSetupCallables {
    PrincipledMetallicSetupCallable
        principled_metallic;
    PrincipledMetallicSetupCallable
        principled_metallic_preserve_energy;
    PrincipledDiffuseSetupCallable
        principled_diffuse;
    PrincipledDielectricSetupCallable
        principled_dielectric;
    PrincipledDielectricSetupCallable
        principled_dielectric_preserve_energy;
};

[[nodiscard]] SurfaceClosureSetupCallables
make_surface_closure_setup_callables() noexcept;

class CallableSurfaceClosureSetupProvider final
    : public SurfaceClosureSetupProvider {

private:
    luisa::compute::Expr<
        luisa::compute::Buffer<float>> _cycles_bsdf_tables;
    const SurfaceClosureSetupCallables &_callables;

public:
    CallableSurfaceClosureSetupProvider(
        luisa::compute::Expr<
            luisa::compute::Buffer<float>> cycles_bsdf_tables,
        const SurfaceClosureSetupCallables &callables) noexcept;

    [[nodiscard]] PrincipledMetallicSetupResult
    principled_metallic(
        const PrincipledMetallicSetupInput &input,
        luisa::compute::Expr<bool> reflective_caustics) const noexcept override;

    [[nodiscard]] PrincipledDiffuseSetupResult
    principled_diffuse(
        const PrincipledDiffuseSetupInput &input) const noexcept override;

    [[nodiscard]] PrincipledDielectricSetupResult
    principled_dielectric(
        const PrincipledDielectricSetupInput &input,
        luisa::compute::Expr<bool> reflective_caustics) const noexcept override;
};

}// namespace psycles::luisa_backend::detail
