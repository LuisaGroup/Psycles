#pragma once

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/surface_ray.h>

namespace psycles::luisa_backend::cycles_volume_boundary {

struct Continuation {
  luisa::compute::Float3 throughput;
  luisa::compute::UInt bounds_bounce;
  luisa::compute::UInt rng_offset;
  luisa::compute::Float ray_tmin;
  luisa::compute::Bool valid;
};

// Cycles integrate_surface_terminate + integrate_surface_volume_only_bounce.
// No surface/transparent bounce counters, visibility, MIS state, differential,
// ray origin/direction, or self IDs change at a pure volume boundary.
[[nodiscard]] inline Continuation advance(
    luisa::compute::Expr<luisa::float3> throughput,
    luisa::compute::Expr<std::uint32_t> bounds_bounce,
    luisa::compute::Expr<std::uint32_t> rng_offset,
    luisa::compute::Expr<float> ray_tmin,
    luisa::compute::Expr<float> ray_length,
    luisa::compute::Expr<float> continuation_probability,
    luisa::compute::Expr<std::uint32_t> path_flag) noexcept {
  using namespace luisa::compute;
  const Float probability = select(continuation_probability, 0.0f,
      (path_flag & cycles_path_state::flag_terminate_on_next_surface) != 0u);
  Continuation result{throughput, bounds_bounce, rng_offset, ray_tmin, probability != 0.0f};
  $if(result.valid) {
    $if(probability != 1.0f) { result.throughput /= probability; };
    result.bounds_bounce += 1u;
    result.valid = result.bounds_bounce <= 1024u; // VOLUME_BOUNDS_MAX
    $if(result.valid) {
      result.rng_offset += cycles_path_state::bounce_dimension_count;
      result.ray_tmin = surface_ray::intersection_t_offset(ray_length);
    };
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_volume_boundary
