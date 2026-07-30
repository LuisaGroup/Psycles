#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using SurfaceEvaluateCallable = Callable<SurfaceEvaluationCall(
    Buffer<luisa::float4>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    luisa::float3,
    luisa::uint,
    luisa::uint,
    float)>;
using SurfaceEmissionCallable = Callable<luisa::float3(
    Buffer<luisa::float4>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    luisa::float3)>;
using SurfaceSampleCallable = Callable<SurfaceSampleCall(
    Buffer<luisa::float4>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    float,
    luisa::float2,
    luisa::uint,
    luisa::uint,
    float)>;
using SurfaceClosureTraceCallable =
    Callable<SurfaceClosureTraceCall(
        Buffer<luisa::float4>,
        Buffer<float>,
        BindlessArray,
        BindlessArray,
        luisa::uint,
        SurfacePointCall,
        luisa::uint)>;
using SurfaceSampleTraceCallable =
    Callable<SurfaceSampleTraceCall(
        Buffer<luisa::float4>,
        Buffer<float>,
        BindlessArray,
        BindlessArray,
        luisa::uint,
        SurfacePointCall,
        float,
        luisa::float2,
        luisa::uint,
        luisa::uint,
        float)>;
using SurfaceAovCallable = Callable<SurfaceAovCall(
    Buffer<luisa::float4>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall)>;
using SurfaceShadingNormalCallable = Callable<luisa::float3(
    Buffer<luisa::float4>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall)>;

struct SurfaceCallables {
    SurfaceEvaluateCallable evaluate;
    SurfaceEmissionCallable emission;
    SurfaceSampleCallable sample;
    SurfaceClosureTraceCallable closure_trace;
    SurfaceSampleTraceCallable sample_trace;
    SurfaceAovCallable aov;
    SurfaceShadingNormalCallable shading_normal;
};

[[nodiscard]] SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

}// namespace psycles::luisa_backend::detail
