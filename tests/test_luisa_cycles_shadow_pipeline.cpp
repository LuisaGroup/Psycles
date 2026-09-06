#include "cycles_shadow_queue_fixture.h"
#include "path_kernel_direct_light_queue.h"
#include "path_kernel_transitions.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/luisa-compute.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;

Var<DirectLightTaskCall> fixture_task(UInt id) {
  auto scenario = id % 8u;
  Var<DirectLightTaskCall> task;
  task.ray_direction = make_float3(0.0f, 0.0f, 1.0f);
  task.ray_maximum = 100.0f;
  task.source_object = scenario;
  task.pixel = id;
  task.sample_index = id;
  task.transparent_depth = 3u;
  task.rng_offset = 72u;
  task.unshadowed_contribution = make_float3(2.0f, 4.0f, 8.0f);
  task.shadow_throughput = task.unshadowed_contribution;
  task.nee_path_throughput = make_float3(2.0f);
  task.light_shader = make_float3(0.25f);
  task.constant_light_shader = ((scenario != 5u) & (scenario != 6u)).cast<unsigned>();
  task.light_terminate_sample = select(0.0f, 1.0f, scenario == 5u);
  return task;
}

bool run(const char *program, const char *backend, bool no_cache) {
  // This is the independently produced Cycles HIP queue oracle used by the
  // coroutine shadow regression, now applied to the actual side-work pool.
  std::array<unsigned, 4u> oracle_shade{};
  std::ifstream oracle{PSYCLES_SHADOW_QUEUE_ORACLE};
  std::string tag;
  unsigned isect_kernel{}, shade_kernel{};
  if (!(oracle >> tag >> isect_kernel >> shade_kernel) || tag != "kernels") {
    return false;
  }
  for (auto mode = 0u; mode < 2u; ++mode) {
    for (auto row = 0u; row < 4u; ++row) {
      unsigned m{}, r{}, queued{}, isect{}, shade{}, terminal{};
      if (!(oracle >> tag >> m >> r >> queued >> isect >> shade >> terminal) ||
          tag != "Q" || m != mode || r != row || isect != 0u || shade > 1u ||
          queued != shade * shade_kernel || terminal + shade != 1u) {
        return false;
      }
      if (mode == 0u) {
        oracle_shade[row] = shade;
      } else if (oracle_shade[row] != shade) {
        return false;
      }
    }
  }
  if (oracle >> tag) {
    return false;
  }

  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr auto paths = 67u;
  constexpr auto cases = 8u;
  std::array<ShadowIntersectionBatchCall, cases * 2u> inputs{};
  for (auto &b : inputs) {
    for (auto &hit : b.hits) {
      hit.distance = 100.0f;
    }
  }
  auto set = [&](unsigned c, unsigned count, unsigned blocked) {
    auto &b = inputs[c * 2u];
    b.count = b.total = count;
    b.blocked = blocked;
    for (auto i = 0u; i < count; ++i) {
      b.hits[i].distance = float(i + 1u);
    }
  };
  for (auto c = 0u; c < 4u; ++c) {
    set(c, shadow_queue_inputs[c].count, shadow_queue_inputs[c].blocked);
  }
  set(4u, 4u, 0u);
  set(6u, 4u, 0u);
  set(7u, 1u, 0u);
  inputs[14u].hits[0].primitive = 1u;
  auto batches = device.create_buffer<ShadowIntersectionBatchCall>(inputs.size());
  stream << batches.copy_from(inputs.data()) << synchronize();
  LocalShadowIntersectionCallable collect =
      [batches = batches.view()](Var<luisa::compute::Ray> ray, UInt source, UInt, UInt,
                                 UInt, UInt) {
        return batches->read(source * 2u + (ray->t_min() > 0.0f).cast<unsigned>());
      };
  EvaluateShadowSurfaceCallable surface =
      [](Var<luisa::compute::Ray>, Var<ShadowIntersectionCall> hit, Float, Float,
         Var<ShadowShaderContextCall>, Var<RenderKernelParameters>) {
        Var<ShadowSurfaceEvaluationCall> result;
        result.transmittance = make_float3(select(0.5f, 0.0f, hit.primitive != 0u));
        return result;
      };
  auto intersection = std::make_shared<ShadowIntersectionComponent>(std::move(collect));
  DirectLightTaskEvaluator evaluator{
      .intersect_shadow = intersection,
      .shade_shadow_surface = surface,
      .trace_shadow = make_fused_shadow_trace_callable(intersection, surface),
      .light_sample_roulette = [](Float3, Float sample,
                                  Float) { return select(1.0f, 0.0f, sample == 1.0f); },
      .clamp_contribution = [](Float3 value, UInt, Float, Float) { return value; },
      .split_scattered_light =
          [](Float3 value, Float3, Float3, Bool) {
            Var<LightPassContributionCall> result;
            result.diffuse_direct = value;
            return result;
          }};
  ShaderOption options;
  options.enable_cache = !no_cache;
  bool passed = true;
  for (auto capacity : {19u, 32u}) {
    auto binding =
        make_direct_light_task_queue(device, evaluator, capacity, 32u, options);
    if (binding.work->stage_count() != 3u) {
      std::cerr << "Shadow pool must expose three independently scheduled stages\n";
      return false;
    }
    RenderCoroutine coroutine{
        [sink = binding.sink](BufferFloat4, BufferFloat4, BufferFloat4, BufferFloat4,
                              BufferUInt main_visits, BufferFloat4, BufferUInt,
                              BufferFloat4, UInt, UInt, BufferFloat4, BufferFloat,
                              Var<RenderKernelParameters> parameters) {
          auto id = dispatch_x();
          UInt iteration = 0u;
          $while(iteration < 2u) {
            $suspend(path_transition::shade_surface);
            sink->emit(fixture_task(id), parameters.wavefront_frame_capacity);
            iteration += 1u;
            $suspend(path_transition::intersect_closest);
            main_visits.atomic(id).fetch_add(1u);
          };
        }};
    using Scheduler = WavefrontCoroScheduler<
        Buffer<luisa::float4>, Buffer<luisa::float4>, Buffer<luisa::float4>,
        Buffer<luisa::float4>, Buffer<unsigned>, Buffer<luisa::float4>, Buffer<unsigned>,
        Buffer<luisa::float4>, unsigned, unsigned, Buffer<luisa::float4>, Buffer<float>,
        RenderKernelParameters>;
    Scheduler scheduler{device,
                        coroutine,
                        {.thread_count = capacity,
                         .gather_by_sorting = false,
                         .frame_buffer_compaction = true,
                         .report_stats = true,
                         .shader_option = options,
                         .execution_block_size = 32u,
                         .largest_continuation_first = true,
                         .incremental_continuation_counts = true}};
    scheduler.register_auxiliary_work(binding.work);
    auto combined = device.create_buffer<luisa::float4>(paths);
    auto passes = device.create_buffer<luisa::float4>(paths * light_pass_buffer_count);
    auto main_visits = device.create_buffer<unsigned>(paths);
    auto dummy = device.create_buffer<float>(1u);
    RenderKernelParameters parameters{};
    parameters.wavefront_frame_capacity = capacity;
    parameters.transparent_max_bounces = 32u;
    for (auto repeat = 0u; repeat < 2u; ++repeat) {
      std::vector<luisa::float4> colors(paths),
          light_passes(paths * light_pass_buffer_count);
      std::vector<unsigned> visits(paths);
      stream << combined.copy_from(colors.data()) << passes.copy_from(light_passes.data())
             << main_visits.copy_from(visits.data());
      scheduler(combined, combined, combined, passes, main_visits, combined, main_visits,
                combined, 0u, 0u, combined, dummy, parameters)
          .dispatch(paths)(stream);
      stream << combined.copy_to(colors.data()) << passes.copy_to(light_passes.data())
             << main_visits.copy_to(visits.data()) << synchronize();
      std::array<uint64_t, 3u> expected_counts{};
      const std::array<unsigned, cases> isects{1, 1, 1, 1, 2, 0, 2, 1};
      const std::array<unsigned, cases> shades{
          oracle_shade[0], oracle_shade[1], oracle_shade[2], oracle_shade[3], 2, 0, 2, 1};
      const std::array<float, cases> factors{0, 1, 0, .25f, .0625f, 0, .03125f, 0};
      for (auto i = 0u; i < paths; ++i) {
        auto c = i % cases;
        expected_counts[0] += 2u * unsigned(c == 5u || c == 6u);
        expected_counts[1] += 2u * isects[c];
        expected_counts[2] += 2u * shades[c];
        auto expected = luisa::make_float3(4.0f, 8.0f, 16.0f) * factors[c];
        auto pass = light_passes[i * light_pass_buffer_count +
                                 light_pass_index(LightPassBuffer::diffuse_direct)];
        for (auto lane = 0u; lane < 3u; ++lane) {
          passed &= std::isfinite(colors[i][lane]) && colors[i][lane] == expected[lane];
          passed &= pass[lane] == expected[lane];
        }
        passed &= visits[i] == 2u;
      }
      auto &&stats = scheduler.last_dispatch_stats();
      for (auto stage = 0u; stage < 3u; ++stage) {
        if (stats.auxiliary_work[stage].executed_count != expected_counts[stage]) {
          std::cerr << "stage=" << stage
                    << " actual=" << stats.auxiliary_work[stage].executed_count
                    << " expected=" << expected_counts[stage] << '\n';
          passed = false;
        }
      }
      passed &= binding.work->host_count() == 0u;
    }
    if (capacity == 19u) {
      // Force in-place relocation while BOTH NEE payloads and shade-owned
      // traversal batches remain live: 19 allocated, six opaque terminals,
      // 13 live, then six new publications. Sources and destination holes
      // straddle the live-prefix boundary; restarting the whole pool or
      // copying only the invariant task would lose contributions here.
      Kernel1D publish = [sink = binding.sink](Var<RenderKernelParameters> params,
                                               UInt first, UInt count) {
        $if(dispatch_x() < count) {
          sink->emit(fixture_task(first + dispatch_x()), params.wavefront_frame_capacity);
        };
      };
      auto publisher = device.compile(publish, options);
      std::vector<luisa::float4> colors(paths),
          light_passes(paths * light_pass_buffer_count);
      stream << combined.copy_from(colors.data())
             << passes.copy_from(light_passes.data());
      binding.work->reset(stream);
      auto read_counts = [&] {
        binding.work->enqueue_count_readback(stream);
        stream << synchronize();
      };
      auto dispatch_stage = [&](unsigned stage) {
        binding.work->dispatch_stage(stage, stream, combined, combined, combined, passes,
                                     main_visits, combined, main_visits, combined, 0u, 0u,
                                     combined, dummy, parameters);
        read_counts();
      };
      stream << publisher(parameters, 0u, 19u).dispatch(19u);
      read_counts();
      dispatch_stage(1u);
      passed &= binding.work->host_count() == 13u;
      passed &= binding.work->stage_host_count(0u) == 4u;
      passed &= binding.work->stage_host_count(2u) == 9u;
      binding.work->prepare_for_producer(stream, 6u);
      stream << publisher(parameters, 19u, 6u).dispatch(6u);
      read_counts();
      passed &= binding.work->host_count() == 19u;
      auto steps = 0u;
      while (binding.work->host_count() != 0u && steps++ < 12u) {
        dispatch_stage(binding.work->admission_stage());
      }
      passed &= binding.work->host_count() == 0u;
      stream << combined.copy_to(colors.data()) << passes.copy_to(light_passes.data())
             << synchronize();
      const std::array<float, cases> factors{0, 1, 0, .25f, .0625f, 0, .03125f, 0};
      for (auto i = 0u; i < paths; ++i) {
        auto expected =
            luisa::make_float3(2.0f, 4.0f, 8.0f) * (i < 25u ? factors[i % cases] : 0.0f);
        auto pass = light_passes[i * light_pass_buffer_count +
                                 light_pass_index(LightPassBuffer::diffuse_direct)];
        for (auto lane = 0u; lane < 3u; ++lane) {
          passed &= colors[i][lane] == expected[lane] && pass[lane] == expected[lane];
        }
      }
    }
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  auto passed = run(argv[0], argc > 1 ? argv[1] : "fallback",
                    argc > 2 && std::string_view{argv[2]} == "--no-cache");
  std::cout << "Independent Cycles shadow pipeline " << (passed ? "passed" : "FAILED")
            << '\n';
  return passed ? 0 : 1;
}
