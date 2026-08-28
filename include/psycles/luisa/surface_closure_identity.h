#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_identity.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>
#include <psycles/luisa/surface_closure_reachability.h>

namespace psycles::luisa_backend::detail {

// Minimal expression projection of the state observed by Cycles closure
// allocation, setup, and post-setup identity classification. Keeping this
// public to the Luisa renderer modules lets storage record ClosureType once at
// the allocation boundary instead of rebuilding it in every hot consumer.
struct SurfaceClosureIdentityExpression {
    Expr<std::uint32_t> kind;
    Expr<std::uint32_t> lobe;
    Expr<std::uint32_t> bssrdf_method;
    Expr<float> allocation_weight;
    Expr<bool> setup_valid;
    Expr<float> roughness;
    Expr<bool> preserve_ggx_energy;
    Expr<bool> beckmann;
};

[[nodiscard]] inline SurfaceClosureIdentityExpression
surface_closure_identity(
    const SurfaceClosurePhysicalRecord &closure) noexcept {
    return {
        .kind = Expr<std::uint32_t>{closure.kind.expression()},
        .lobe = Expr<std::uint32_t>{closure.lobe.expression()},
        .bssrdf_method = Expr<std::uint32_t>{
            closure.bssrdf_method.expression()},
        .allocation_weight =
            Expr<float>{closure.allocation_weight.expression()},
        .setup_valid = Expr<bool>{closure.setup_valid.expression()},
        .roughness = Expr<float>{closure.roughness.expression()},
        .preserve_ggx_energy = Expr<bool>{
            closure.preserve_ggx_energy.expression()},
        .beckmann = Expr<bool>{closure.beckmann.expression()}};
}

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosurePhysicalRecord &closure) noexcept;
[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureIdentityExpression &closure) noexcept;

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosurePhysicalRecord &closure,
    Float glossy_filter_roughness = 0.0f,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;
[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureIdentityExpression &closure,
    Float glossy_filter_roughness,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

// Exact post-setup Cycles ClosureType. For a retained slot, type_none is the
// sole setup-failure state; allocation failure is represented by absence from
// the retained prefix rather than by another field in this identity.
[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosurePhysicalRecord &closure,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;
[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosureIdentityExpression &closure,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

} // namespace psycles::luisa_backend::detail
