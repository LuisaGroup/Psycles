#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error \
    "Include <psycles/luisa/cycles_voronoi.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend::cycles_voronoi {

// Direct projection of Cycles' VoronoiOutput. Color and distance share one
// vector only at this ABI boundary; the evaluator computes one semantic
// result and the SVM handler stores each independently live stack output.
struct EvaluationCall {
    luisa::float4 color_distance{};
    luisa::float4 position{};
    float radius{};
};

}// namespace psycles::luisa_backend::cycles_voronoi

LUISA_STRUCT(
    psycles::luisa_backend::cycles_voronoi::EvaluationCall,
    color_distance,
    position,
    radius) {};

namespace psycles::luisa_backend::cycles_voronoi {

// Every field is a Blender node property known while Luisa traces the shader
// AST. A prepared configuration therefore emits one specialized Callable;
// feature, dimension, metric, normalization, and output do not become weakly
// typed per-shading-point switches.
struct Configuration {
    std::uint32_t dimensions{3u};
    compiler::VoronoiFeature feature{compiler::VoronoiFeature::f1};
    compiler::VoronoiDistanceMetric metric{
        compiler::VoronoiDistanceMetric::euclidean};
    compiler::VoronoiOutput output{compiler::VoronoiOutput::distance};
    bool normalize{false};

    bool operator==(const Configuration &) const noexcept = default;
};

[[nodiscard]] bool
is_operation(compiler::ValueOperation operation) noexcept;

[[nodiscard]] Configuration
decode_configuration(const compiler::ValueInstruction &instruction) noexcept;

void prepare(const Configuration &configuration) noexcept;

[[nodiscard]] Float4 evaluate(const Configuration &configuration, Float3 vector,
                              Float w, Float scale, Float detail,
                              Float roughness, Float lacunarity,
                              Float smoothness, Float exponent,
                              Float randomness) noexcept;

// Runtime payload form used by the isomorphic Cycles SVM interpreter. The
// four coordinate domains are host/JIT-specialized callables, while feature,
// metric, Normalize, and all socket values remain the original SVM fields.
[[nodiscard]] luisa::compute::Var<EvaluationCall> evaluate_runtime(
    bool voronoi_extra_enabled,
    luisa::compute::Expr<std::uint32_t> dimensions,
    luisa::compute::Expr<std::uint32_t> feature,
    luisa::compute::Expr<std::uint32_t> metric,
    luisa::compute::Expr<bool> normalize,
    Float3 vector, Float w, Float scale, Float detail, Float roughness,
    Float lacunarity, Float smoothness, Float exponent,
    Float randomness) noexcept;

}// namespace psycles::luisa_backend::cycles_voronoi
