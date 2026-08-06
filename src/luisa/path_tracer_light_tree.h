#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using LightTreeSampleCallable = Callable<LightDistributionGpu(
    float, luisa::float3, luisa::float3, float, bool)>;
using LightTreePdfCallable = Callable<float(
    luisa::uint, luisa::float3, luisa::float3, float, bool)>;
using LightTreeForwardPdfCallable = Callable<float(
    luisa::uint,
    luisa::float3,
    luisa::float3,
    float,
    luisa::uint,
    luisa::uint)>;
using LightTreeTriangleEmitterCallable =
    Callable<luisa::uint(luisa::uint, luisa::uint)>;

struct LightTreeCallables {
    LightTreeSampleCallable surface_sample;
    LightTreeSampleCallable volume_sample;
    LightTreePdfCallable surface_pdf;
    LightTreePdfCallable volume_pdf;
    LightTreeForwardPdfCallable forward_pdf;
    LightTreeTriangleEmitterCallable triangle_emitter;
};

// The surface and volume callables are built separately, so Luisa specializes
// the two importance equations while tracing the kernel AST. No per-query
// volume-mode branch remains in generated code.
[[nodiscard]] LightTreeCallables make_light_tree_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

}// namespace psycles::luisa_backend::detail
