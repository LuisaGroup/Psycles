#pragma once

#include <psycles/luisa/surface.h>

#include <luisa/dsl/struct.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>

namespace psycles::luisa_backend::detail {

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

using PrincipledDielectricSetupCallable =
    luisa::compute::Callable<PrincipledDielectricSetupCall(
        luisa::compute::Buffer<float>,
        PrincipledDielectricSetupInputCall,
        bool)>;

struct SurfaceClosureSetupCallables {
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
    const luisa::compute::BufferFloat &_cycles_bsdf_tables;
    const SurfaceClosureSetupCallables &_callables;

public:
    CallableSurfaceClosureSetupProvider(
        const luisa::compute::BufferFloat &cycles_bsdf_tables,
        const SurfaceClosureSetupCallables &callables) noexcept;

    [[nodiscard]] PrincipledDielectricSetupResult
    principled_dielectric(
        const PrincipledDielectricSetupInput &input,
        luisa::compute::Expr<bool> reflective_caustics) const noexcept override;
};

}// namespace psycles::luisa_backend::detail
