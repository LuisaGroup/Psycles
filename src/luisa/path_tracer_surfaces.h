#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using SurfacePreparationCallable = Callable<SurfacePreparationCall(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    SurfacePreparationQueryCall)>;

using SurfaceEvaluateLightCallable = Callable<SurfaceEvaluationCall(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    luisa::float3,
    luisa::uint,
    luisa::uint,
    float,
    bool,
    bool,
    luisa::uint)>;
using SurfaceEmissionCallable = Callable<luisa::float3(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    luisa::float3,
    bool)>;
using SurfaceConstantEmissionCallable =
    Callable<luisa::float3(
        Buffer<float>,
        Buffer<luisa::float3>,
        luisa::uint,
        luisa::uint)>;
using SurfaceSampleCallable = Callable<SurfaceSampleCall(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    float,
    luisa::float2,
    luisa::uint,
    luisa::uint,
    float,
    bool,
    bool)>;
using SurfaceClosureTraceCallable =
    Callable<SurfaceClosureTraceCall(
        Buffer<float>,
        Buffer<luisa::float3>,
        Buffer<float>,
        BindlessArray,
        BindlessArray,
        luisa::uint,
        SurfacePointCall,
        luisa::uint,
        bool,
        bool)>;
using SurfaceSampleTraceCallable =
    Callable<SurfaceSampleTraceCall(
        Buffer<float>,
        Buffer<luisa::float3>,
        Buffer<float>,
        BindlessArray,
        BindlessArray,
        luisa::uint,
        SurfacePointCall,
        float,
        luisa::float2,
        luisa::uint,
        luisa::uint,
        float,
        bool,
        bool)>;
using SurfaceBssrdfNormalCallable = Callable<luisa::float3(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    bool,
    bool)>;
using SurfaceShadingNormalCallable = Callable<luisa::float3(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall)>;

// Host/JIT object whose device-local storage is populated exactly once for a
// path hit. Every method is a consumer of the retained original closures; no
// method is allowed to dispatch or replay the material graph.
class PopulatedSurfaceShader {

  public:
    virtual ~PopulatedSurfaceShader() noexcept = default;

    [[nodiscard]] virtual SurfacePreparation preparation(
        const SurfacePreparationQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceEvaluation evaluate_light(
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> requested_index,
        const SurfaceQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceSample sample(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceSampleTrace sample_trace(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;
};

class SurfacePopulationComponent {

  public:
    virtual ~SurfacePopulationComponent() noexcept = default;

    [[nodiscard]] virtual std::shared_ptr<PopulatedSurfaceShader> populate(
        Expr<std::uint32_t> surface_tag,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query) const noexcept = 0;
};

struct SurfaceCallables {
    // Present only for the explicit populate-once A/B route until its storage
    // layout and HIP performance have been validated on complete scenes.
    std::shared_ptr<const SurfacePopulationComponent> population;
    SurfacePreparationCallable preparation;
    SurfaceEvaluateLightCallable evaluate_light;
    SurfaceConstantEmissionCallable constant_emission;
    SurfaceEmissionCallable emission;
    SurfaceSampleCallable sample;
    SurfaceClosureTraceCallable closure_trace;
    SurfaceSampleTraceCallable sample_trace;
    SurfaceBssrdfNormalCallable bssrdf_normal;
    SurfaceShadingNormalCallable shading_normal;
};

[[nodiscard]] SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

[[nodiscard]] SurfacePreparationCallable
make_compact_surface_preparation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

}// namespace psycles::luisa_backend::detail
