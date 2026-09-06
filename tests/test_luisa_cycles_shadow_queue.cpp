#include "cycles_shadow_queue_fixture.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_transitions.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/luisa-compute.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;

template <typename T> T unbound() {
  return T{luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

bool read_oracle(std::array<unsigned, 4u> &shade_counts) {
  std::ifstream input{PSYCLES_SHADOW_QUEUE_ORACLE};
  std::string tag;
  unsigned intersect{}, shade{};
  if (!(input >> tag >> intersect >> shade) || tag != "kernels" ||
      intersect == 0u || shade == 0u || intersect == shade) {
    return false;
  }
  for (auto transparent = 0u; transparent < 2u; ++transparent) {
    for (auto i = 0u; i < shadow_queue_inputs.size(); ++i) {
      unsigned mode{}, row{}, queued{}, isect_count{}, shade_count{},
          terminal{};
      if (!(input >> tag >> mode >> row >> queued >> isect_count >>
            shade_count >> terminal) ||
          tag != "Q" || mode != transparent || row != i || isect_count != 0u ||
          shade_count > 1u || terminal > 1u || terminal + shade_count != 1u ||
          queued != shade_count * shade) {
        return false;
      }
      if (transparent == 0u) {
        shade_counts[i] = shade_count;
      } else if (shade_counts[i] != shade_count) {
        return false;
      }
    }
  }
  return !(input >> tag) && input.eof();
}

bool run(const char *program, const char *backend, bool no_cache) {
  std::array<unsigned, 4u> oracle{};
  if (!read_oracle(oracle)) {
    std::cerr << "Invalid Cycles HIP shadow queue oracle\n";
    return false;
  }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr auto cases = 12u;
  constexpr auto paths = 67u;
  std::array<ShadowIntersectionBatchCall, cases * 3u> inputs{};
  for (auto &batch : inputs) {
    for (auto &hit : batch.hits) {
      hit.distance = 100.0f; // Production collector's inactive-lane sentinel.
    }
  }
  auto set_batch = [&](unsigned scenario, unsigned batch, unsigned count,
                       bool blocked = false) {
    auto &b = inputs[scenario * 3u + batch];
    b.count = b.total = count;
    b.blocked = blocked;
    for (auto j = 0u; j < count; ++j) {
      b.hits[j].distance = float(batch * 4u + j + 1u);
    }
  };
  for (auto i = 0u; i < shadow_queue_inputs.size(); ++i) {
    const auto c = shadow_queue_inputs[i];
    set_batch(i, 0u, c.count, c.blocked != 0u);
  }
  set_batch(4u, 0u, 1u); // Shading itself becomes opaque: it must run once.
  inputs[4u * 3u].hits[0].primitive = 1u;
  for (auto c = 5u; c <= 8u; ++c) {
    set_batch(c, 0u, 4u);
  }
  set_batch(6u, 1u, 0u, true); // Opaque after a full transparent batch.
  set_batch(7u, 1u, 1u);
  set_batch(8u, 1u, 4u);
  set_batch(9u, 0u, 2u);
  inputs[9u * 3u].hits[1].primitive = 1u;
  set_batch(10u, 0u, 4u);
  inputs[10u * 3u].hits[3].primitive = 1u;

  auto batches =
      device.create_buffer<ShadowIntersectionBatchCall>(inputs.size());
  stream << batches.copy_from(inputs.data()) << synchronize();
  LocalShadowIntersectionCallable collect =
      [batches = batches.view()](Var<luisa::compute::Ray> ray, UInt source,
                                 UInt, UInt, UInt, UInt) {
        UInt batch = 0u;
        $if(ray->t_min() > 8.0f) { batch = 2u; }
        $elif(ray->t_min() > 0.0f) { batch = 1u; };
        return batches->read(source * 3u + batch);
      };
  // Prepared attenuation isolates queue routing, not SVM/geometry correctness
  // (the separate native-shadow oracle tests those real implementations).
  EvaluateShadowSurfaceCallable surface =
      [](Var<luisa::compute::Ray>, Var<ShadowIntersectionCall> hit, Float,
         Float, Var<ShadowShaderContextCall>, Var<RenderKernelParameters>) {
        Var<ShadowSurfaceEvaluationCall> result;
        result.transmittance = make_float3(0.5f);
        $if(hit.primitive != 0u) { result.transmittance = make_float3(0.0f); };
        return result;
      };
  const DirectLightTaskEvaluator evaluator{
      .intersect_shadow =
          std::make_shared<ShadowIntersectionComponent>(std::move(collect)),
      .shade_shadow_surface = surface,
      .trace_shadow = unbound<TraceShadowCallable>(),
      .light_sample_roulette = unbound<LightSampleRouletteCallable>(),
      .clamp_contribution = unbound<ClampLightContributionCallable>(),
      .split_scattered_light = unbound<SplitScatteredLightCallable>()};
  Coroutine<void(unsigned, Buffer<luisa::float4>, Buffer<luisa::uint4>)>
      coroutine{
          [evaluator](UInt scenario, BufferFloat4 output, BufferUInt4 meta) {
            const auto id = dispatch_x();
            Var<DirectLightTaskCall> task;
            task.source_object = scenario;
            task.ray_direction = make_float3(0.0f, 0.0f, 1.0f);
            task.ray_maximum = 100.0f;
            task.shadow_throughput = make_float3(2.0f, 4.0f, 8.0f);
            task.transparent_depth = 3u;
            task.rng_offset = 72u;
            Var<RenderKernelParameters> params;
            params.transparent_max_bounces = 32u;
            Bool active = scenario != 11u; // No NEE proposal: no shadow work.
            Bool visible = false;
            evaluator.trace_staged(task, params, active, visible);
            output.write(
                id, make_float4(task.shadow_throughput, visible.cast<float>()));
            meta.write(id, make_uint4(task.transparent_depth, task.rng_offset,
                                      task.volume_bounds_bounce, id));
          }};
  auto output = device.create_buffer<luisa::float4>(paths);
  auto meta = device.create_buffer<luisa::uint4>(paths);
  std::array<luisa::float4, paths> actual{};
  std::array<luisa::uint4, paths> state{};
  // Full transparent batches continue even when the next traversal is empty.
  // These counts follow Cycles integrate_transparent_shadow's exact four-hit
  // continuation rule, independently covered by the native-shadow oracle.
  const std::array<unsigned, cases> isect_count{1, 1, 1, 1, 1, 2,
                                                2, 2, 3, 1, 1, 0};
  const std::array<unsigned, cases> shade_count{
      oracle[0], oracle[1], oracle[2], oracle[3], 1, 2, 1, 2, 3, 1, 1, 0};
  const std::array<unsigned, cases> surviving_hits{0, 0, 0, 2, 0, 4,
                                                   4, 5, 8, 1, 3, 0};
  const std::array<unsigned, cases> visible{0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0};
  bool passed = true;
  for (const bool soa : {true, false}) {
    WavefrontCoroSchedulerConfig config;
    config.thread_count = 19u; // Force refills and preserve logical path IDs.
    config.global_memory_soa = soa;
    config.report_stats = true;
    config.execution_block_size = 32u;
    config.largest_continuation_first = true;
    config.incremental_continuation_counts = true;
    config.shader_option.enable_cache = !no_cache;
    WavefrontCoroScheduler<unsigned, Buffer<luisa::float4>,
                           Buffer<luisa::uint4>>
        scheduler{device, coroutine, config};
    for (auto c = 0u; c < cases; ++c) {
      scheduler(c, output, meta).dispatch(paths)(stream);
      stream << output.copy_to(actual.data()) << meta.copy_to(state.data())
             << synchronize();
      const auto &stats = scheduler.last_dispatch_stats();
      passed &= stats.collected && stats.generated_count == paths;
      for (const auto &[name, expected] :
           {std::pair{path_transition::intersect_shadow, isect_count[c]},
            std::pair{path_transition::shade_shadow, shade_count[c]}}) {
        const auto *node = coroutine.graph().node_by_name(name);
        if (!node || stats.continuations[node->index].executed_count !=
                         std::uint64_t(expected) * paths) {
          std::cerr << "soa=" << soa << " case=" << c
                    << " continuation=" << name << " actual="
                    << (node ? stats.continuations[node->index].executed_count
                             : ~0ull)
                    << " expected=" << expected * paths << '\n';
          passed = false;
        }
      }
      for (auto i = 0u; i < paths; ++i) {
        const auto factor = std::ldexp(1.0f, -int(surviving_hits[c]));
        const auto v = actual[i];
        const auto s = state[i];
        if (v.x != 2.0f * factor || v.y != 4.0f * factor ||
            v.z != 8.0f * factor || v.w != float(visible[c]) ||
            s.x != 3u + surviving_hits[c] ||
            s.y != 72u + 16u * surviving_hits[c] || s.z != 0u || s.w != i) {
          std::cerr << "soa=" << soa << " case=" << c << " path=" << i
                    << " shadow state/visibility mismatch\n";
          passed = false;
          break;
        }
      }
    }
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  const bool passed =
      run(argv[0], argc > 1 ? argv[1] : "fallback",
          argc > 2 && std::string_view{argv[2]} == "--no-cache");
  if (passed) {
    std::cout << "Cycles shadow queue transitions passed\n";
  }
  return passed ? 0 : 1;
}
