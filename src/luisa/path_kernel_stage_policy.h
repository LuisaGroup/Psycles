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

} // namespace psycles::luisa_backend::detail
