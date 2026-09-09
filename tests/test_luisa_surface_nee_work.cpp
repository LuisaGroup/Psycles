#include "path_kernel_builder.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <luisa/luisa-compute.h>

#include <array>
#include <iostream>
#include <vector>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace sobol = psycles::sampling::tabulated_sobol;

// Unused production stages are unbound, never replaced by a host evaluator.
template<typename Function> Function unbound() {
  return Function{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr auto paths = 37u;
  auto selection_counts = device.create_buffer<unsigned>(paths);
  auto body_counts = device.create_buffer<unsigned>(paths);
  auto unused = device.create_buffer<float>(1u);
  const auto generated = sobol::generate_table(256u);
  std::vector<luisa::float4> table;
  for (const auto value : generated) {
    table.emplace_back(luisa::make_float4(value.x, value.y, value.z, value.w));
  }
  auto samples = device.create_buffer<luisa::float4>(table.size());
  stream << samples.copy_from(table.data());
  bool passed = true;
  for (const bool use_light_tree : {false, true}) {
    PathKernelConfig config{
        .use_light_tree = use_light_tree,
        .light_transport = {unbound<SafeNormalizeCallable>(),
                            unbound<ForwardLightWeightCallable>(),
                            unbound<NeeLightWeightCallable>(),
                            unbound<ClampLightContributionCallable>(),
                            unbound<LightSampleRouletteCallable>(),
                            unbound<LightComponentRatioCallable>()},
        .light_distribution_sample = unbound<LightDistributionSampleCallable>(),
        .light_tree = {unbound<LightTreeSampleCallable>(),
                       unbound<LightTreeSampleCallable>(),
                       unbound<LightTreePdfCallable>(),
                       unbound<LightTreePdfCallable>(),
                       unbound<LightTreeForwardPdfCallable>(),
                       unbound<LightTreeTriangleEmitterCallable>()},
        .surfaces = {nullptr, unbound<SurfacePreparationCallable>(),
                     unbound<SurfaceEvaluateLightCallable>(),
                     unbound<SurfaceConstantEmissionCallable>(),
                     unbound<SurfaceEmissionCallable>(),
                     unbound<SurfaceSampleCallable>(),
                     unbound<SurfaceClosureTraceCallable>(),
                     unbound<SurfaceSampleTraceCallable>(),
                     unbound<SurfaceBssrdfNormalCallable>()},
        .shade_shadow_surface = unbound<EvaluateShadowSurfaceCallable>(),
        .trace_shadow = unbound<TraceShadowCallable>()};
    // Only count discrete-selection invocations; do not implement another
    // sampler, shader or renderer as the expected-value oracle.
    config.light_distribution_sample = [&](Float) {
      selection_counts->atomic(dispatch_x()).fetch_add(1u);
      Var<LightDistributionGpu> result;
      return result;
    };
    const auto random = make_path_bounce_random_stage();
    Kernel1D kernel = [&](BufferFloat4 sobol_table, BufferUInt output,
                          BufferFloat dummy, Bool use_direct_light,
                          Var<RenderKernelParameters> parameters) {
      UInt first = 0u;
      PathKernelInvocation invocation{
          .config = config,
          .film_accumulation = PathFilmAccumulation::serial,
          .combined = sobol_table, .normal = sobol_table, .albedo = sobol_table,
          .light_passes = sobol_table, .sample_count = output,
          .volume_guiding_raw = sobol_table, .volume_guiding_denoised = output,
          .path_trace = sobol_table, .sample_first = first,
          .sobol_table = sobol_table, .filter_table = dummy,
          .parameters = parameters};
      PathSampleContext sample{.invocation = invocation};
      sample.sample_index = dispatch_x();
      sample.rng_hash = 123u;
      sample.cycles_rng_offset = 16u;
      const auto flags = (dispatch_x() & 1u) * closure::runtime_bsdf_has_eval |
                         ((dispatch_x() >> 1u) & 1u) *
                             closure::runtime_bsdf_has_transmission;
      // Original Cycles 5.2.1 integrate_surface_direct_light returns before
      // PRNG_LIGHT or emitter selection unless both conditions hold.
      const auto active = use_direct_light &
                          ((flags & closure::runtime_bsdf_has_eval) != 0u);
      random->with_active_state(sample, active, [&](PathBounceRandomState &) {
        output.atomic(dispatch_x()).fetch_add(1u);
      });
    };
    auto shader = device.compile(kernel, ShaderOption{
        .enable_cache = false, .enable_fast_math = true});
    RenderKernelParameters parameters{};
    parameters.sobol_sequence_size = 256u;
    for (const bool use_direct_light : {false, true}) {
      std::array<unsigned, 2u * paths> actual{};
      stream << selection_counts.copy_from(actual.data())
             << body_counts.copy_from(actual.data() + paths)
             << shader(samples, body_counts, unused, use_direct_light, parameters)
                    .dispatch(paths)
             << selection_counts.copy_to(actual.data())
             << body_counts.copy_to(actual.data() + paths) << synchronize();
      for (auto path = 0u; path < paths; ++path) {
        const auto expected_body = unsigned(use_direct_light && (path & 1u));
        const auto expected_flat = use_light_tree ? 0u : expected_body;
        if (actual[path] != expected_flat ||
            actual[paths + path] != expected_body) {
          std::cerr << "tree=" << use_light_tree << " direct=" << use_direct_light
                    << " path=" << path << " selection=" << actual[path]
                    << '/' << expected_flat << " body=" << actual[paths + path]
                    << '/' << expected_body << '\n';
          passed = false;
        }
      }
    }
  }
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
