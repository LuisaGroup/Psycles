#pragma once

#include <psycles/contract/shader_graph.h>
#include <psycles/luisa/surface.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace psycles::luisa_backend::detail {
struct LuisaSceneData;
struct SurfaceClosureTraceCall;
struct SurfaceEvaluationCall;
struct SurfacePreparationCall;
struct SurfaceSampleTraceCall;
struct SurfaceValueRuntime;
} // namespace psycles::luisa_backend::detail

namespace psycles::test_support {

struct CompactSurfaceProgramEvidence {
    bool domains_match{};
    bool normal_transactions_exact{};
    bool bump_stream_exact{};
    bool unified_scene_exact{};
    bool unified_variant_bijection{};
    bool unified_closure_domains_exact{};
    std::uint32_t bump_variant{~std::uint32_t{0u}};
};

[[nodiscard]] contract::ShaderGraph make_minimal_principled_graph();

// Forces the direct Convert and Math families through observable Principled
// inputs; in particular Scalar to Boolean uses the Cycles float-to-int
// truncation boundary.
[[nodiscard]] contract::ShaderGraph make_direct_math_convert_graph();

// Exercises both output subtypes of the direct Cycles Vector Math family with
// runtime geometry inputs, preventing constant folding from bypassing the
// compact typed-stack handler.
[[nodiscard]] contract::ShaderGraph make_direct_vector_math_graph();

// One closed texture trunk exercising Mapping, Image Color/Alpha, sampled RGB
// Ramp Color/Alpha, and Mix Color. The Boolean selects Cycles' distinct BOX
// execution shape without changing the graph dataflow.
[[nodiscard]] contract::ShaderGraph
make_direct_texture_trunk_graph(bool box_projection);

// Forces the Cycles ShaderData projection families through a coordinate-driven
// Bump height. The expanded stream must contain both ordinary Geometry/
// Texture Coordinate records and their explicit derivative counterparts,
// including plain and transformed Object coordinates.
[[nodiscard]] contract::ShaderGraph make_direct_state_bump_graph();

// Validates the direct-family semantic domain and the injective executable
// record -> PackedTransform metadata ownership relation. Empty means success.
[[nodiscard]] std::string validate_direct_state_surface_runtime(
    const luisa_backend::detail::SurfaceValueRuntime &runtime);

// Covers every Cycles Noise type and dimension, with both Normalize values
// sharing one factor shape and one color shape. Each graph keeps its coordinate
// dynamic so the direct typed-stack evaluator cannot be constant-folded away.
[[nodiscard]] std::vector<contract::ShaderGraph> make_direct_noise_graphs();

// Proves that factor/color each form one exact evaluator class and that their
// finite instruction-immediate images preserve every authored Noise semantic.
// Empty means success.
[[nodiscard]] std::string validate_direct_noise_surface_runtime(
    const luisa_backend::detail::SurfaceValueRuntime &runtime);

// Covers the complete Cycles ShaderData/path context projection, including
// linked and unlinked Fresnel/Layer Weight records in one immediate domain.
[[nodiscard]] std::vector<contract::ShaderGraph> make_direct_context_graphs();

// Proves the exact operation/family/bank/arity/immediate relation. Empty means
// success.
[[nodiscard]] std::string validate_direct_context_surface_runtime(
    const luisa_backend::detail::SurfaceValueRuntime &runtime);

// Four graphs span the ClampFactor x uniformity product for Cycles Mix,
// sampled/control RGB Curve, and all RGB/HSV/HSL Separate/Combine modes while
// keeping every value dependent on runtime geometry.
[[nodiscard]] std::vector<contract::ShaderGraph>
make_direct_color_algebra_graphs();

// Proves operation/family/bank/arity/immediate domains for the complete pure
// color/algebra direct-SVM boundary. Empty means success.
[[nodiscard]] std::string validate_direct_color_algebra_surface_runtime(
    const luisa_backend::detail::SurfaceValueRuntime &runtime);

// A depth-three Mix tree. `restore_after` appends an Add emission sibling;
// otherwise the Mix consumes the root tail. The transparent leaf also forces
// physical replay, so the paired device tests cover both scalar tail frames
// and full parent-restoring frames with the same closure topology.
[[nodiscard]] contract::ShaderGraph
make_nested_mix_replay_graph(bool restore_after);

// Differential fixture for the two-phase automatic-normal transaction. Every
// undisplaced geometry member is distinct from its displaced counterpart.
[[nodiscard]] luisa_backend::SurfacePoint
make_surface_value_transaction_test_point() noexcept;

// Returns an empty string when the unified per-record dispatcher preserves a
// narrow point-reference ABI, returns no aggregate transaction state, and owns
// one semantic handler boundary per active exact evaluator.
[[nodiscard]] std::string validate_compact_surface_value_program_abi(
    const std::shared_ptr<luisa_backend::detail::LuisaSceneData> &scene);

// The compact interpreter's lane stack has a fresh logical lifetime for every
// surface invocation. Its root definition must remain one non-material
// lifetime witness rather than executable zero initialization that reaches
// the HIP kernel.
[[nodiscard]] std::string validate_surface_value_fresh_lifetime_seed();

void print_compact_surface_sample_mismatch(
    const luisa_backend::detail::SurfaceSampleTraceCall &actual,
    const luisa_backend::detail::SurfaceSampleTraceCall &expected,
    std::string_view backend, std::size_t topology,
    std::size_t scenario);

[[nodiscard]] bool equal(
    luisa::float2 actual,
    luisa::float2 expected,
    float tolerance) noexcept;
[[nodiscard]] bool equal(
    luisa::float3 actual,
    luisa::float3 expected,
    float tolerance) noexcept;
[[nodiscard]] bool finite_compact_value(
    luisa::float3 value) noexcept;
void print_compact_value(luisa::float3 value);
[[nodiscard]] bool equal(
    const luisa_backend::detail::SurfacePreparationCall &actual,
    const luisa_backend::detail::SurfacePreparationCall &expected,
    float tolerance) noexcept;
[[nodiscard]] bool equal(
    const luisa_backend::detail::SurfaceClosureTraceCall &actual,
    const luisa_backend::detail::SurfaceClosureTraceCall &expected,
    float tolerance) noexcept;
[[nodiscard]] bool equal(
    const luisa_backend::detail::SurfaceEvaluationCall &actual,
    const luisa_backend::detail::SurfaceEvaluationCall &expected,
    float tolerance) noexcept;
[[nodiscard]] bool equal(
    const luisa_backend::detail::SurfaceSampleTraceCall &actual,
    const luisa_backend::detail::SurfaceSampleTraceCall &expected,
    float tolerance) noexcept;

void report_compact_surface_preparation_mismatch(
    std::string_view backend,
    std::size_t topology,
    std::size_t scenario,
    const luisa_backend::detail::SurfacePreparationCall &actual,
    const luisa_backend::detail::SurfacePreparationCall &expected);

// The shifted form places a live Add node before the Color Ramp, giving the
// runtime table a different ParameterId without changing the ramp evaluator.
[[nodiscard]] contract::ShaderGraph
make_sampled_color_ramp_graph(std::string table, bool shifted_parameter);

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

// Inspect the unified host bytecode image rather than pixels. The controlled
// fixture has two nested Bump records. Refinement must retain both
// configurations as bump_samples, eliminate every recursive Bump operation and
// hidden height program, and keep the instruction/evaluator relation total.
[[nodiscard]] CompactSurfaceProgramEvidence inspect_compact_surface_program(
    const luisa_backend::detail::SurfaceValueRuntime &runtime) noexcept;

} // namespace psycles::test_support
