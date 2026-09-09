/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_surface_integrator.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_transform.h>
#include <psycles/luisa/surface_ray.h>

#include <limits>

namespace psycles::luisa_backend::cycles_svm::detail {
using namespace luisa::compute;
namespace abi = compiler::cycles_svm;

Float3 integrate_surface_ray_offset(const KernelGlobals &kg, const ShaderData &sd,
                                   Expr<luisa::float4x4> object_inverse_transform,
                                   Expr<luisa::float3> P,
                                   Expr<luisa::float3> D) noexcept {
  Float3 result = P;
  $if((sd.type & unsigned(abi::PRIMITIVE_TRIANGLE)) != 0u) {
    TriangleVertices vertices{};
    $if(sd.type == unsigned(abi::PRIMITIVE_TRIANGLE)) {
      vertices = kg.triangle_vertices(sd.object, sd.prim);
    }
    $else { vertices = kg.motion_triangle_vertices(sd.object, sd.prim, sd.time); };
    Float3 local_P = P;
    Float3 local_D = D;
    $if((sd.object_flag & unsigned(abi::SD_OBJECT_TRANSFORM_APPLIED)) == 0u) {
      local_P = cycles_transform::point(object_inverse_transform, local_P);
      local_D = cycles_transform::direction(object_inverse_transform, local_D);
    };
    result = surface_ray::origin_with_explicit_self_exclusion(
        P, sd.Ng, local_P, local_D, vertices.v0, vertices.v1, vertices.v2);
  };
  return result;
}

UInt integrate_surface_ray_portal(const KernelGlobals &kg, const ShaderData &sd,
                                 Expr<std::uint32_t> closure_index,
                                 Expr<luisa::float4x4> object_inverse_transform,
                                 RayPortalState &state,
                                 const cycles_path_state::Limits &limits) noexcept {
  UInt label = cycles_closure::label_none;
  const auto &pool = *sd.closure;
  const auto pc = pool.ray_portal(closure_index);
  Float sum_sample_weight = 0.0f;
  UInt i = 0u;
  $while(i < pool.count()) {
    const auto sc = pool.common(i);
    $if(cycles_closure::is_bsdf_or_bssrdf(sc.type)) {
      sum_sample_weight += sc.sample_weight;
    };
    i += 1u;
  };
  // Keep the native comparison direction; this is not a positive-finite
  // weight guard or an epsilon test on the selected portal's signed weight.
  $if(!(sum_sample_weight <= 0.0f)) {
    $if(dot(sd.P - pc.param.P, sd.P - pc.param.P) > 1e-9f) {
      state.isect_object = ~0u;
      state.P = pc.param.P;
    }
    $else {
      state.P = integrate_surface_ray_offset(
          kg, sd, object_inverse_transform, pc.param.P, pc.param.D);
    };
    state.D = pc.param.D;
    state.tmin = 0.0f;
    state.tmax = std::numeric_limits<float>::max();
    state.dP = sd.dP;
    const auto pick_pdf = pc.common.sample_weight / sum_sample_weight;
    state.throughput *= pc.common.weight / pick_pdf;
    label = cycles_closure::label_transmit | cycles_closure::label_ray_portal;
    state.path = cycles_path_state::next_surface(
        state.path, label,
        (sd.flag & unsigned(abi::SD_BSDF_HAS_TRANSMISSION)) != 0u,
        (sd.flag & unsigned(abi::SD_RAY_PORTAL)) != 0u, limits);
  };
  return label;
}

} // namespace psycles::luisa_backend::cycles_svm::detail
