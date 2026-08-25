#pragma once

#include <psycles/contract/shader_graph.h>

#include <cstdint>
#include <string>
#include <vector>

namespace psycles::luisa_backend::detail {
struct SurfaceValueRuntime;
}

namespace psycles::test_support {

struct CompactSurfaceProgramEvidence {
    bool domains_match{};
    bool bump_partition_exact{};
    std::uint32_t bump_variant{~std::uint32_t{0u}};
};

[[nodiscard]] contract::ShaderGraph make_minimal_principled_graph();

// The shifted form places a live Add node before the Color Ramp, giving the
// runtime table a different ParameterId without changing the ramp evaluator.
[[nodiscard]] contract::ShaderGraph make_sampled_color_ramp_graph(
    std::string table, bool shifted_parameter);

// A live graph containing both Cycles Clamp modes. Compact execution must use
// one shared typed handler whose record domain contains both immediates.
[[nodiscard]] contract::ShaderGraph make_typed_clamp_graph();

[[nodiscard]] bool has_typed_clamp_record_domain(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

// One graph per configuration keeps each authored result directly observable
// while the scene runtime must quotient all configurations to one scalar and
// one vector handler.
[[nodiscard]] std::vector<contract::ShaderGraph>
make_typed_map_range_graphs();

[[nodiscard]] bool has_typed_map_range_record_domains(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

// Proves the regression really exercises one shared SVM-mode handler carrying
// at least two distinct late-bound table ParameterIds.
[[nodiscard]] bool has_color_ramp_record_product(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

// Inspect the host bytecode image rather than pixels. The controlled fixture
// has two nested Bump records: configuration 1 is a root-only outer Bump and
// configuration 0 is reachable from both roots and offset-height programs.
// Cycles-style typed records share one execution handler for both records, so
// exact call-graph evidence must come from the instruction stream.
[[nodiscard]] CompactSurfaceProgramEvidence inspect_compact_surface_program(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

} // namespace psycles::test_support
