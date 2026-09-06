#pragma once

#include <array>

namespace psycles::test_support {

// Prepared traversal inputs, not an alternate traversal or shading oracle.
// The HIP probe feeds these into Cycles' original integrator_intersect_shadow
// and observes the real shadow queue transition and its atomic counters.
struct ShadowQueueInput {
  unsigned blocked;
  unsigned count;
};
inline constexpr std::array shadow_queue_inputs{
    ShadowQueueInput{1u, 0u}, ShadowQueueInput{0u, 0u},
    ShadowQueueInput{1u, 2u}, ShadowQueueInput{0u, 2u}};

} // namespace psycles::test_support
