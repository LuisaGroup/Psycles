#pragma once

#include <cstdint>

namespace psycles::luisa_backend::detail {

// Host/JIT policy for placing coroutine transitions in the one authoritative
// path program. This is deliberately not a device enum: `none` records the
// megakernel, while the coroutine policy inserts `$suspend` statements around
// the same stage objects and closure implementations.
enum class PathCoroutineCutPolicy : std::uint8_t {
  none,
  // Materialize the same main-path scheduling boundaries as Cycles' GPU
  // integrator. The policy changes only the host-recorded stage graph.
  cycles_wavefront,
};

// Host-side placement of the pure per-bounce sampler expression. The three
// sites are related as follows:
//
//   megakernel + volume: F(sample, rng_offset) dominates both consumers;
//   megakernel - volume: F is emitted at the surface consumer;
//   wavefront + volume:  F is independently emitted in the disjoint volume
//                        and surface continuations;
//   wavefront - volume:  F is emitted at the surface continuation.
//
// Re-emitting F in disjoint continuations is value-preserving: Cycles' Sobol
// sampler is a pure function of immutable sample identity, RNG hash, offset,
// and dimension. It also prevents NEE/light-distribution expressions from
// becoming part of the intersect_closest continuation or its live frame.
struct PathBounceRandomPlan {
  bool before_event_resolution;
  bool shade_volume;
  bool shade_surface;
};

[[nodiscard]] constexpr PathBounceRandomPlan
make_path_bounce_random_plan(PathCoroutineCutPolicy cut_policy,
                             bool has_volume) noexcept {
  if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
    return {.before_event_resolution = false,
            .shade_volume = has_volume,
            .shade_surface = true};
  }
  return {.before_event_resolution = has_volume,
          .shade_volume = false,
          .shade_surface = !has_volume};
}

} // namespace psycles::luisa_backend::detail
