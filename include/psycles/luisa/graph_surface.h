#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/graph_surface.h> through the Psycles::luisa target."
#endif

#include <memory>

#include <psycles/luisa/surface.h>

namespace psycles::compiler {
class SurfaceProgram;
class SurfaceClosurePlan;
}// namespace psycles::compiler

namespace psycles::luisa_backend {

namespace detail {
class GraphSurfaceImplementation;
}// namespace detail

// A compiled Cycles surface graph. The implementation deliberately lives
// behind a private host-side object model: its polymorphic nodes are invoked
// while Luisa records the device AST, so the emitted kernel remains fused and
// contains no device-side C++ virtual dispatch.
class GraphSurface final : public Surface {

private:
    std::unique_ptr<detail::GraphSurfaceImplementation> _implementation;

public:
    explicit GraphSurface(
        std::shared_ptr<const compiler::SurfaceProgram> program) noexcept;
    GraphSurface(
        std::shared_ptr<const compiler::SurfaceProgram> program,
        compiler::SurfaceClosurePlan closure_plan) noexcept;
    ~GraphSurface() noexcept override;

    GraphSurface(const GraphSurface &) = delete;
    GraphSurface(GraphSurface &&) = delete;
    GraphSurface &operator=(const GraphSurface &) = delete;
    GraphSurface &operator=(GraphSurface &&) = delete;

    [[nodiscard]] SurfaceCapabilities capabilities()
        const noexcept override;

    [[nodiscard]] SurfaceClosureCollection collect_closures(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept override;

    [[nodiscard]] SurfacePopulation populate(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept override;

    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept override;

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept override;

    [[nodiscard]] SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept override;

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics) const noexcept override;

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept override;

    [[nodiscard]] SurfacePreparation prepare(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePreparationQuery &query) const noexcept override;

    [[nodiscard]] Float3 emission(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        Expr<bool> reflective_caustics) const noexcept override;

    [[nodiscard]] Float3 constant_emission(
        const SurfaceParameterServices &services,
        Expr<std::uint32_t> parameter_block) const noexcept override;

    [[nodiscard]] Float3 transparent_extinction(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override;

    [[nodiscard]] VolumeCoefficients evaluate_volume(
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector *collector) const noexcept override;

    [[nodiscard]] Float3 displacement(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override;

    [[nodiscard]] Float3 shading_normal(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override;

    // Host/JIT integration boundary for scene-wide value scheduling. The
    // implementation pointer is never exposed to device code.
    [[nodiscard]] const detail::GraphSurfaceImplementation *
    internal_implementation() const noexcept;

};

}// namespace psycles::luisa_backend
