#pragma once

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

// The exact mutable subset touched by integrate_surface_ray_portal. Fields
// absent here (dD, primitive identity, forward MIS and pass weights) are not
// rewritten by this operation. These are DSL values, never a device heap or
// an additional persistent coroutine-frame allocation.
struct RayPortalState {
  luisa::compute::Float3 P;
  luisa::compute::Float3 D;
  luisa::compute::Float tmin;
  luisa::compute::Float tmax;
  luisa::compute::Float dP;
  luisa::compute::Float3 throughput;
  luisa::compute::UInt isect_object;
  cycles_path_state::State path;
};

// object_inverse_transform is the effective object_get_inverse_transform:
// static KernelObject::itfm or the time-resolved motion inverse. It is not
// unconditionally ShaderData::ob_itfm_motion (uninitialized for static objects
// in the original kernel). The geometry provider resolves this at the hit.
[[nodiscard]] luisa::compute::Float3 integrate_surface_ray_offset(
    const KernelGlobals &kg, const ShaderData &sd,
    luisa::compute::Expr<luisa::float4x4> object_inverse_transform,
    luisa::compute::Expr<luisa::float3> P,
    luisa::compute::Expr<luisa::float3> D) noexcept;

[[nodiscard]] luisa::compute::UInt integrate_surface_ray_portal(
    const KernelGlobals &kg, const ShaderData &sd,
    luisa::compute::Expr<std::uint32_t> closure_index,
    luisa::compute::Expr<luisa::float4x4> object_inverse_transform,
    RayPortalState &state, const cycles_path_state::Limits &limits) noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
