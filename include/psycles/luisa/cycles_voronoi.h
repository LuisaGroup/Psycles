#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error \
    "Include <psycles/luisa/cycles_voronoi.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <luisa/dsl/struct.h>
#include <luisa/dsl/syntax.h>

namespace psycles::compiler {

enum class VoronoiFeature : std::uint8_t {
  f1,
  f2,
  smooth_f1,
  distance_to_edge,
  n_sphere_radius
};

enum class VoronoiDistanceMetric : std::uint8_t {
  euclidean,
  manhattan,
  chebychev,
  minkowski
};

enum class VoronoiOutput : std::uint8_t {
  distance,
  color,
  position,
  w,
  radius
};

} // namespace psycles::compiler

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

void prepare(const Configuration &configuration) noexcept;

[[nodiscard]] luisa::compute::Float4 evaluate(
    const Configuration &configuration,
    luisa::compute::Float3 vector,
    luisa::compute::Float w,
    luisa::compute::Float scale,
    luisa::compute::Float detail,
    luisa::compute::Float roughness,
    luisa::compute::Float lacunarity,
    luisa::compute::Float smoothness,
    luisa::compute::Float exponent,
    luisa::compute::Float randomness) noexcept;

struct RuntimeFilterOptions {
    std::uint32_t dimension_mask{~0u};
    std::uint32_t feature_mask{~0u};
    std::uint32_t metric_mask{~0u};
    bool may_have_detail{true};
    bool may_normalize{true};
    bool may_need_color{true};
    bool may_need_position{true};
    bool may_need_radius{true};

    [[nodiscard]] constexpr bool is_unfiltered() const noexcept {
        return dimension_mask == ~0u && feature_mask == ~0u &&
               metric_mask == ~0u && may_have_detail && may_normalize &&
               may_need_color && may_need_position && may_need_radius;
    }
};

// Runtime payload form used by the isomorphic Cycles SVM interpreter. The
// four coordinate domains are host/JIT-specialized callables, while feature,
// metric, Normalize, and all socket values remain the original SVM fields.
[[nodiscard]] luisa::compute::Var<EvaluationCall> evaluate_runtime(
    bool voronoi_extra_enabled,
    luisa::compute::Expr<std::uint32_t> dimensions,
    luisa::compute::Expr<std::uint32_t> feature,
    luisa::compute::Expr<std::uint32_t> metric,
    luisa::compute::Expr<bool> normalize,
    luisa::compute::Float3 vector, luisa::compute::Float w,
    luisa::compute::Float scale, luisa::compute::Float detail,
    luisa::compute::Float roughness, luisa::compute::Float lacunarity,
    luisa::compute::Float smoothness, luisa::compute::Float exponent,
    luisa::compute::Float randomness) noexcept;

[[nodiscard]] luisa::compute::Var<EvaluationCall> evaluate_runtime_filtered(
    bool voronoi_extra_enabled,
    RuntimeFilterOptions filter_options,
    luisa::compute::Expr<std::uint32_t> dimensions,
    luisa::compute::Expr<std::uint32_t> feature,
    luisa::compute::Expr<std::uint32_t> metric,
    luisa::compute::Expr<bool> normalize,
    luisa::compute::Float3 vector, luisa::compute::Float w,
    luisa::compute::Float scale, luisa::compute::Float detail,
    luisa::compute::Float roughness, luisa::compute::Float lacunarity,
    luisa::compute::Float smoothness, luisa::compute::Float exponent,
    luisa::compute::Float randomness) noexcept;

}// namespace psycles::luisa_backend::cycles_voronoi
