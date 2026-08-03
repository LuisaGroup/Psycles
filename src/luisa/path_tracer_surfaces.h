#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

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
using SurfaceRuntimeFlagsCallable = Callable<luisa::uint(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall,
    float,
    bool,
    bool)>;
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
using SurfaceAovCallable = Callable<SurfaceAovCall(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall)>;
using SurfaceShadingNormalCallable = Callable<luisa::float3(
    Buffer<float>,
    Buffer<luisa::float3>,
    Buffer<float>,
    BindlessArray,
    BindlessArray,
    luisa::uint,
    SurfacePointCall)>;

struct SurfaceCallables {
    SurfaceEvaluateLightCallable evaluate_light;
    SurfaceRuntimeFlagsCallable runtime_flags;
    SurfaceConstantEmissionCallable constant_emission;
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
