#include "path_kernel_stage_policy.h"

#include <psycles/luisa/path_tracer.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>

int main() {
  using psycles::luisa_backend::luisa_path_scheduler_name;
  using psycles::luisa_backend::
      luisa_path_scheduler_uses_cycles_stage_partition;
  using psycles::luisa_backend::luisa_path_scheduler_uses_per_sample_dispatch;
  using psycles::luisa_backend::LuisaPathScheduler;
  using psycles::luisa_backend::LuisaPathTracerOptions;
  using psycles::luisa_backend::parse_luisa_path_scheduler;
  using psycles::luisa_backend::valid_luisa_persistent_scheduler_shape;
  using psycles::luisa_backend::valid_luisa_wavefront_execution_block_size;
  using psycles::luisa_backend::detail::make_path_bounce_random_plan;
  using psycles::luisa_backend::detail::PathCoroutineCutPolicy;

  constexpr auto serial_surface =
      make_path_bounce_random_plan(PathCoroutineCutPolicy::none, false);
  static_assert(!serial_surface.before_event_resolution &&
                !serial_surface.shade_volume && serial_surface.shade_surface);
  constexpr auto serial_volume =
      make_path_bounce_random_plan(PathCoroutineCutPolicy::none, true);
  static_assert(serial_volume.before_event_resolution &&
                !serial_volume.shade_volume && !serial_volume.shade_surface);
  constexpr auto staged_surface = make_path_bounce_random_plan(
      PathCoroutineCutPolicy::cycles_wavefront, false);
  static_assert(!staged_surface.before_event_resolution &&
                !staged_surface.shade_volume && staged_surface.shade_surface);
  constexpr auto staged_volume = make_path_bounce_random_plan(
      PathCoroutineCutPolicy::cycles_wavefront, true);
  static_assert(!staged_volume.before_event_resolution &&
                staged_volume.shade_volume && staged_volume.shade_surface);

  constexpr std::array schedulers{LuisaPathScheduler::megakernel,
                                  LuisaPathScheduler::megakernel_per_sample,
                                  LuisaPathScheduler::wavefront,
                                  LuisaPathScheduler::wavefront_graph,
                                  LuisaPathScheduler::wavefront_staged,
                                  LuisaPathScheduler::persistent};
  for (const auto scheduler : schedulers) {
    const auto name = luisa_path_scheduler_name(scheduler);
    if (name.empty() || parse_luisa_path_scheduler(name) != scheduler) {
      return EXIT_FAILURE;
    }
  }
  if (parse_luisa_path_scheduler("state-machine") ||
      parse_luisa_path_scheduler("") ||
      !luisa_path_scheduler_name(static_cast<LuisaPathScheduler>(255u))
           .empty()) {
    return EXIT_FAILURE;
  }
  static_assert(!luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::megakernel));
  static_assert(luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::megakernel_per_sample));
  static_assert(luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::wavefront));
  static_assert(luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::wavefront_graph));
  static_assert(luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::wavefront_staged));
  static_assert(luisa_path_scheduler_uses_per_sample_dispatch(
      LuisaPathScheduler::persistent));
  static_assert(!luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::megakernel));
  static_assert(!luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::megakernel_per_sample));
  static_assert(luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::wavefront));
  static_assert(luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::wavefront_graph));
  static_assert(luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::wavefront_staged));
  static_assert(luisa_path_scheduler_uses_cycles_stage_partition(
      LuisaPathScheduler::persistent));
  static_assert(!luisa_path_scheduler_uses_cycles_stage_partition(
      static_cast<LuisaPathScheduler>(255u)));

  // The production wavefront capacity matches Cycles' 2^20 maximum path
  // states. Persistent defaults retain the scheduler experiment's 2^15
  // workers with SoA and global-memory extension enabled.
  constexpr LuisaPathTracerOptions options;
  static_assert(options.enable_fast_math);
  static_assert(options.scheduler == LuisaPathScheduler::megakernel);
  static_assert(options.wavefront_frame_capacity == (1u << 20u));
  static_assert(options.wavefront_execution_block_size == 32u);
  static_assert(options.wavefront_graph_worker_count == 0u);
  static_assert(!options.wavefront_graph_selective_scheduling);
  static_assert(options.wavefront_graph_refill_threshold == 0u);
  static_assert(options.wavefront_counter_readback_batch_size == 4u);
  static_assert(options.wavefront_counter_readback_pipeline_depth == 2u);
  static_assert(options.wavefront_tail_megakernel_threshold == 4096u);
  static_assert(options.staged_surface_sorting);
  static_assert(!options.staged_direct_light_queue);
  static_assert(!valid_luisa_wavefront_execution_block_size(0u));
  static_assert(!valid_luisa_wavefront_execution_block_size(16u));
  static_assert(valid_luisa_wavefront_execution_block_size(32u));
  static_assert(valid_luisa_wavefront_execution_block_size(64u));
  static_assert(valid_luisa_wavefront_execution_block_size(1024u));
  static_assert(!valid_luisa_wavefront_execution_block_size(1056u));
  static_assert(options.persistent_worker_count == (1u << 15u));
  static_assert(options.persistent_block_size == 32u);
  static_assert(options.persistent_fetch_size == 1u);
  static_assert(valid_luisa_persistent_scheduler_shape(1u, 32u, 1u));
  static_assert(!valid_luisa_persistent_scheduler_shape(0u, 32u, 1u));
  static_assert(!valid_luisa_persistent_scheduler_shape(1u, 16u, 1u));
  static_assert(!valid_luisa_persistent_scheduler_shape(1u, 32u, 0u));
  static_assert(!valid_luisa_persistent_scheduler_shape(
      1u, 1024u, std::numeric_limits<std::uint32_t>::max()));
  static_assert(options.persistent_shared_memory_soa);
  static_assert(options.persistent_global_memory_extension);
  static_assert(options.max_samples_per_dispatch == 64u);
  return EXIT_SUCCESS;
}
