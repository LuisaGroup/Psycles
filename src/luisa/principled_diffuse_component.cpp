#include "principled_diffuse_component.h"

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

PrincipledDiffuseSetupResult setup_principled_diffuse(
    const PrincipledDiffuseSetupInput &input) noexcept {
    const auto base_color = max(
        input.color, make_float3(0.0f));
    const auto diffuse_weight = max(
        input.lower_weight * base_color,
        make_float3(0.0f));
    const auto subsurface_weight = clamp(
        input.subsurface_weight, 0.0f, 1.0f);
    const auto pre_weight =
        diffuse_weight * (1.0f - subsurface_weight);
    const auto allocation_weight = sample_weight(
        max(pre_weight, make_float3(0.0f)));
    const auto allocated =
        allocation_weight >=
        cycles_closure::closure_weight_cutoff;
    const auto weight = select(
        make_float3(0.0f), pre_weight, allocated);
    return {
        .weight = weight,
        .allocation_weight = allocation_weight,
        .sample_weight = select(
            0.0f, allocation_weight, allocated)};
}

PrincipledDiffuseComponent::PrincipledDiffuseComponent(
    const ShaderServices &services) noexcept
    : _services{services} {}

PrincipledDiffuseSetupResult
PrincipledDiffuseComponent::setup(
    const PrincipledDiffuseSetupInput &input) const noexcept {
    const auto *provider =
        _services.surface_closure_setup_provider();
    return provider != nullptr
               ? provider->principled_diffuse(input)
               : setup_principled_diffuse(input);
}

}// namespace psycles::luisa_backend::detail
