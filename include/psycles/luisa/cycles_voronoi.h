#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error \
    "Include <psycles/luisa/cycles_voronoi.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

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

}// namespace psycles::luisa_backend::cycles_voronoi
