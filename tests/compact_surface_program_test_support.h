#pragma once

#include <psycles/contract/shader_graph.h>

#include <cstdint>

namespace psycles::luisa_backend::detail {
struct SurfaceValueRuntime;
}

namespace psycles::test_support {

struct CompactSurfaceProgramEvidence {
    bool domains_match{};
    bool bump_partition_exact{};
    std::uint32_t bump_variant{~std::uint32_t{0u}};
};

// A live graph containing two Clamp records with one primary opcode key but
// different exact semantics. This exercises the interpreter's ambiguous-fiber
// refinement rather than merely inspecting the compiler partition.
[[nodiscard]] contract::ShaderGraph make_ambiguous_clamp_graph();

[[nodiscard]] bool has_ambiguous_clamp_handler_fiber(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

// Inspect the host bytecode image rather than pixels. The controlled fixture
// has two nested Bump records: configuration 1 is a root-only outer Bump and
// configuration 0 is reachable from both roots and offset-height programs.
// Cycles-style typed records share one execution handler for both records, so
// exact call-graph evidence must come from the instruction stream.
[[nodiscard]] CompactSurfaceProgramEvidence inspect_compact_surface_program(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

} // namespace psycles::test_support
