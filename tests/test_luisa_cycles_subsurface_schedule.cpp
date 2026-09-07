#include "path_kernel_coro_transitions.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/luisa-compute.h>

#include <array>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace psycles::luisa_backend::detail;

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr auto paths = 43u;
  auto output = device.create_buffer<unsigned>(paths);
  bool passed = true;
  for (const bool has_subsurface : {false, true}) {
    Coroutine<void(unsigned, Buffer<unsigned>)> coroutine{
        [has_subsurface](UInt scenario, BufferUInt out) {
          Bool pending_exit = false;
          UInt shaded = 0u;
          $for(step, 3u) {
            // This is the production pipeline's cut policy, not a copied
            // test-only suspend graph. Actual queue executions are counted.
            suspend_before_closest_intersection(
                PathCoroutineCutPolicy::cycles_wavefront, has_subsurface, pending_exit);
            pending_exit = false;
            $suspend(path_transition::shade_surface);
            shaded += 1u;
            if (has_subsurface) {
              $if((step < 2u) & ((scenario == 1u) | ((step == 0u) & (scenario >= 2u)))) {
                $suspend(path_transition::intersect_subsurface);
                $if(scenario == 3u) { $break; };
                pending_exit = true;
              };
            }
          };
          out.write(dispatch_x(), shaded);
        }};
    for (const bool soa : {true, false}) {
      WavefrontCoroSchedulerConfig config;
      config.thread_count = 19u; // Force capacity refills and frame reuse.
      config.global_memory_soa = soa;
      config.report_stats = true;
      config.execution_block_size = 32u;
      config.largest_continuation_first = true;
      config.incremental_continuation_counts = true;
      config.shader_option.enable_cache = false;
      config.shader_option.enable_fast_math = true;
      WavefrontCoroScheduler<unsigned, Buffer<unsigned>> scheduler{device, coroutine, config};
      for (auto scenario = 0u; scenario < 4u; ++scenario) {
        // Original Cycles integrator/subsurface.h: successful scatter queues
        // SHADE_SURFACE directly (excluding MNEE/raytrace, not admitted here).
        // Failed scatter terminates in INTERSECT_SUBSURFACE. No second ray is
        // traced and no INTERSECT_CLOSEST queue is visited for a stored exit.
        const auto surfaces = has_subsurface && scenario == 3u ? 1u : 3u;
        const auto subsurface = !has_subsurface || scenario == 0u ? 0u : (scenario == 1u ? 2u : 1u);
        const auto closest = surfaces - (scenario == 3u ? 0u : subsurface);
        for (auto repeat = 0u; repeat < 2u; ++repeat) {
          stream << scheduler(scenario, output).dispatch(paths);
          std::array<unsigned, paths> actual{};
          stream << output.copy_to(actual.data()) << synchronize();
          const auto &stats = scheduler.last_dispatch_stats();
          passed &= stats.collected && stats.generated_count == paths;
          for (const auto &[name, expected] :
               {std::pair{path_transition::intersect_closest, closest},
                std::pair{path_transition::shade_surface, surfaces},
                std::pair{path_transition::intersect_subsurface, subsurface}}) {
            const auto *node = coroutine.graph().node_by_name(name);
            const auto visits = node ? stats.continuations[node->index].executed_count : 0ull;
            if (visits != std::uint64_t(expected) * paths ||
                (!has_subsurface && name == path_transition::intersect_subsurface && node)) {
              std::cerr << "SSS=" << has_subsurface << " soa=" << soa << " scenario=" << scenario
                        << " queue=" << name << " got=" << visits << " expected=" << expected * paths << '\n';
              passed = false;
            }
          }
          for (const auto value : actual) { passed &= value == surfaces; }
        }
      }
    }
  }
  return passed;
}
} // namespace
int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "fallback") ? 0 : 1;
}
