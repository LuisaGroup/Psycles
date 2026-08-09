#include <psycles/luisa/graph_surface.h>

#include <utility>

#include "graph_surface_internal.h"

namespace psycles::luisa_backend {

GraphSurface::GraphSurface(
    std::shared_ptr<const compiler::SurfaceProgram> program) noexcept
    : _implementation{
          std::make_unique<detail::GraphSurfaceImplementation>(
              std::move(program))} {}

GraphSurface::~GraphSurface() noexcept = default;

SurfaceCapabilities GraphSurface::capabilities() const noexcept {
    return _implementation->capabilities();
}

SurfaceClosureCollection GraphSurface::collect_closures(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    SurfaceClosureCollector &collector) const noexcept {
    return _implementation->collect_closures(
        services,
        point,
        reflective_caustics,
        refractive_caustics,
        collector);
}

UInt GraphSurface::runtime_flags(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics) const noexcept {
    return _implementation->runtime_flags(
        services,
        point,
        glossy_filter_roughness,
        reflective_caustics,
        refractive_caustics);
}

SurfaceEvaluation GraphSurface::evaluate(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query) const noexcept {
    return _implementation->evaluate(
        services, point, outgoing, query);
}

SurfaceEvaluation GraphSurface::evaluate_light(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing,
    const SurfaceLightQuery &query) const noexcept {
    return _implementation->evaluate_light(
        services, point, outgoing, query);
}

SurfaceSample GraphSurface::sample(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return _implementation->sample(
        services, point, u_lobe, u_direction, query);
}

SurfaceClosureTrace GraphSurface::closure_trace(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<std::uint32_t> requested_index,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics) const noexcept {
    return _implementation->closure_trace(
        services,
        point,
        requested_index,
        reflective_caustics,
        refractive_caustics);
}

SurfaceSampleTrace GraphSurface::sample_trace(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return _implementation->sample_trace(
        services, point, u_lobe, u_direction, query);
}

SurfacePreparation GraphSurface::prepare(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing,
    Expr<float> glossy_filter_roughness,
    Expr<bool> emission_reflective_caustics,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    Expr<bool> include_runtime_flags,
    Expr<bool> include_aov) const noexcept {
    return _implementation->prepare(
        services,
        point,
        outgoing,
        glossy_filter_roughness,
        emission_reflective_caustics,
        reflective_caustics,
        refractive_caustics,
        include_runtime_flags,
        include_aov);
}

Float3 GraphSurface::emission(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing,
    Expr<bool> reflective_caustics) const noexcept {
    return _implementation->emission(
        services,
        point,
        outgoing,
        reflective_caustics);
}

Float3 GraphSurface::constant_emission(
    const SurfaceParameterServices &services,
    Expr<std::uint32_t> parameter_block) const noexcept {
    return _implementation->constant_emission(
        services, parameter_block);
}

Float3 GraphSurface::transparent_extinction(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    return _implementation->transparent_extinction(
        services, point);
}

VolumeCoefficients GraphSurface::evaluate_volume(
    const ShaderServices &services,
    const SurfacePoint &point,
    const VolumeQuery &query,
    VolumePhaseCollector *collector) const noexcept {
    return _implementation->evaluate_volume(
        services,
        point,
        query,
        collector);
}

Float3 GraphSurface::displacement(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    return _implementation->displacement(services, point);
}

Float3 GraphSurface::shading_normal(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    return _implementation->shading_normal(
        services, point);
}

SurfaceAov GraphSurface::aov(
    const ShaderServices &services,
    const SurfacePoint &point) const noexcept {
    return _implementation->aov(services, point);
}

}// namespace psycles::luisa_backend
