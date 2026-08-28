#include "luisa_surface_closure_collection_test_support.h"

#include <cstdint>

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::test_support {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

RequestedClosureCollector::RequestedClosureCollector(
    UInt requested) noexcept
    : _requested{requested} {}

void RequestedClosureCollector::begin(
    Expr<luisa::float3> shading_normal) noexcept {
    // Each runtime material branch records its own source-order sequence.
    // Clearing the host vector is required to prevent cross-branch leakage.
    _closures.clear();
    _shading_normal = shading_normal;
}

void RequestedClosureCollector::add(
    const SurfaceClosureRecord &closure) noexcept {
    _closures.emplace_back(closure);
}

void RequestedClosureCollector::finish() noexcept {
    for (const auto &closure : _closures) {
        const auto allocated =
            closure.allocation_weight >=
            cycles_closure::closure_weight_cutoff;
        const auto match = allocated & (_count == _requested);
        _selected.closure_type = select(
            _selected.closure_type, closure.closure_type, match);
        _selected.microfacet_fresnel = select(
            _selected.microfacet_fresnel,
            closure.microfacet_fresnel,
            match);
        _selected.weight = select(_selected.weight, closure.weight, match);
        _selected.allocation_weight = select(
            _selected.allocation_weight,
            closure.allocation_weight,
            match);
        _selected.sample_weight = select(
            _selected.sample_weight,
            closure.sample_weight,
            match);
        _selected.setup_valid = select(
            _selected.setup_valid,
            closure.setup_valid,
            match);
        _selected.albedo = select(_selected.albedo, closure.albedo, match);
        _selected.reflection_albedo = select(
            _selected.reflection_albedo,
            closure.reflection_albedo,
            match);
        _selected.transmission_albedo = select(
            _selected.transmission_albedo,
            closure.transmission_albedo,
            match);
        _selected.color = select(_selected.color, closure.color, match);
        _selected.normal = select(_selected.normal, closure.normal, match);
        _selected.roughness = select(
            _selected.roughness, closure.roughness, match);
        _selected.microfacet_tangent = select(
            _selected.microfacet_tangent,
            closure.microfacet_tangent,
            match);
        _selected.microfacet_alpha_x = select(
            _selected.microfacet_alpha_x,
            closure.microfacet_alpha_x,
            match);
        _selected.microfacet_alpha_y = select(
            _selected.microfacet_alpha_y,
            closure.microfacet_alpha_y,
            match);
        _selected.diffuse_roughness = select(
            _selected.diffuse_roughness,
            closure.diffuse_roughness,
            match);
        _selected.metallic = select(
            _selected.metallic, closure.metallic, match);
        _selected.ior = select(_selected.ior, closure.ior, match);
        _selected.thin_film_thickness = select(
            _selected.thin_film_thickness,
            closure.thin_film_thickness,
            match);
        _selected.thin_film_ior = select(
            _selected.thin_film_ior,
            closure.thin_film_ior,
            match);
        _selected.specular_ior_level = select(
            _selected.specular_ior_level,
            closure.specular_ior_level,
            match);
        _selected.specular_tint = select(
            _selected.specular_tint,
            closure.specular_tint,
            match);
        _selected.sheen_transform_a = select(
            _selected.sheen_transform_a,
            closure.sheen_transform_a,
            match);
        _selected.sheen_transform_b = select(
            _selected.sheen_transform_b,
            closure.sheen_transform_b,
            match);
        _selected.evaluation_scale = select(
            _selected.evaluation_scale,
            closure.evaluation_scale,
            match);
        _selected.fresnel_f0 = select(
            _selected.fresnel_f0, closure.fresnel_f0, match);
        _selected.fresnel_f90 = select(
            _selected.fresnel_f90, closure.fresnel_f90, match);
        _selected.reflection_tint = select(
            _selected.reflection_tint,
            closure.reflection_tint,
            match);
        _selected.transmission_tint = select(
            _selected.transmission_tint,
            closure.transmission_tint,
            match);
        _selected.preserve_ggx_energy = select(
            _selected.preserve_ggx_energy,
            closure.preserve_ggx_energy,
            match);
        _selected.beckmann = select(
            _selected.beckmann, closure.beckmann, match);
        _selected.bssrdf_method = select(
            _selected.bssrdf_method, closure.bssrdf_method, match);
        _selected.bssrdf_radius = select(
            _selected.bssrdf_radius, closure.bssrdf_radius, match);
        _selected.bssrdf_albedo = select(
            _selected.bssrdf_albedo, closure.bssrdf_albedo, match);
        _selected.bssrdf_ior = select(
            _selected.bssrdf_ior, closure.bssrdf_ior, match);
        _selected.bssrdf_roughness = select(
            _selected.bssrdf_roughness,
            closure.bssrdf_roughness,
            match);
        _selected.bssrdf_anisotropy = select(
            _selected.bssrdf_anisotropy,
            closure.bssrdf_anisotropy,
            match);
        _valid |= match;
        _count += select(0u, 1u, allocated);
    }
}

CollectedClosureTrace RequestedClosureCollector::result() const noexcept {
    return {
        .count = _count,
        .closure = _selected,
        .valid = _valid,
        .shading_normal = _shading_normal};
}

} // namespace psycles::test_support
