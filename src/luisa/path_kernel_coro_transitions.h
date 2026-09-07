#pragma once

#include "path_kernel_stage_policy.h"
#include "path_kernel_transitions.h"

#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

inline void suspend_before_closest_intersection(
    PathCoroutineCutPolicy policy, bool has_subsurface,
    luisa::compute::Expr<bool> pending_subsurface_exit) noexcept {
  if (policy == PathCoroutineCutPolicy::cycles_wavefront) {
    if (has_subsurface) {
      // Cycles subsurface_scatter already committed the next intersection
      // and schedules SHADE_SURFACE directly. The normal bounce setup below
      // consumes that stored hit without traversal. Do not insert a queue
      // visit (and frame round trip) merely to reach that bypass branch.
      $if(!pending_subsurface_exit) {
        $suspend(path_transition::intersect_closest);
      };
    } else {
      // A scene without BSSRDF has no pending-hit state or dynamic cut guard.
      $suspend(path_transition::intersect_closest);
    }
  }
}

} // namespace psycles::luisa_backend::detail
