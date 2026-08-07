#include "thin_subsurface_component.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] TracedClosure setup_diffuse_family(
    const TracedClosure &prototype,
    compiler::ClosureOperation operation,
    SurfaceClosureKind physical_kind,
    Float3 normal,
    Float3 weight,
    Float roughness,
    Bool enabled) noexcept {
    auto closure = prototype;
    closure.operation = operation;
    closure.physical_kind = physical_kind;
    closure.principled_lobe = PrincipledLobe::none;
    closure.normal = normal;
    closure.roughness = roughness;
    closure.diffuse_roughness = roughness;
    closure.evaluation_scale = make_float3(1.0f);
    closure.preserve_ggx_energy = false;
    closure.beckmann = false;
    closure.setup_valid = true;

    const auto allocated_weight = max(weight, make_float3(0.0f));
    const auto allocation_weight = sample_weight(allocated_weight);
    const auto allocated =
        enabled &
        (allocation_weight >= cycles_closure::closure_weight_cutoff);
    closure.weight = select(
        make_float3(0.0f), allocated_weight, allocated);
    closure.allocation_weight = select(
        0.0f, allocation_weight, allocated);
    closure.sample_weight = closure.allocation_weight;
    closure.albedo = closure.weight;
    closure.reflection_albedo = make_float3(0.0f);
    closure.transmission_albedo = make_float3(0.0f);
    return closure;
}

}// namespace

ThinSubsurfaceResult ThinSubsurfaceComponent::setup(
    const TracedClosure &prototype,
    Float3 weight,
    Bool enabled) const noexcept {
    const auto anisotropy = prototype.subsurface_anisotropy;
    const auto reflection_fraction = clamp(
        0.5f * (1.0f - anisotropy), 0.0f, 1.0f);
    const auto transmission_fraction = clamp(
        0.5f * (1.0f + anisotropy), 0.0f, 1.0f);
    const auto roughness = clamp(
        prototype.diffuse_roughness, 0.0f, 1.0f);
    const auto smooth = roughness < 1.0e-5f;

    auto reflection = setup_diffuse_family(
        prototype,
        compiler::ClosureOperation::diffuse,
        SurfaceClosureKind::none,
        prototype.normal,
        weight * reflection_fraction,
        roughness,
        enabled);
    auto smooth_transmission = setup_diffuse_family(
        prototype,
        compiler::ClosureOperation::translucent,
        SurfaceClosureKind::none,
        prototype.normal,
        weight * transmission_fraction,
        0.0f,
        enabled & smooth);
    auto rough_transmission = setup_diffuse_family(
        prototype,
        compiler::ClosureOperation::translucent,
        SurfaceClosureKind::rough_translucent,
        -prototype.normal,
        weight * transmission_fraction,
        roughness,
        enabled & !smooth);
    return {
        .reflection = reflection,
        .smooth_transmission = smooth_transmission,
        .rough_transmission = rough_transmission};
}

}// namespace psycles::luisa_backend::detail
