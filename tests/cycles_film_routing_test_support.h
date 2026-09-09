#pragma once

#include "cycles_film_routing_fixture.h"
#include "cycles_surface_emission_test_support.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_film.h"

namespace psycles::test_support::film_routing {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
namespace f = film_routing_fixture;

inline unsigned path_flags(unsigned route) {
  return route == 0   ? 0u
         : route == 1 ? cycles_path_state::flag_surface_pass
         : route == 2 ? cycles_path_state::flag_volume_pass
         : route == 3 ? cycles_path_state::flag_any_pass
                      : cycles_path_state::flag_transparent |
                            cycles_path_state::flag_volume_primary_transmit;
}
inline luisa::float3 rgb(f::RGB v) { return {v[0], v[1], v[2]}; }

// Call the production state preparation and film sinks, including the
// explicitly captured shadow event after a later main-path mutation.
inline void record(const PathKernelConfig &config,
                   const DirectLightTaskEvaluator &evaluator, unsigned operation,
                   PathFilmAccumulation mode, bool detached,
                   const BufferFloat4 &combined, const BufferFloat4 &film,
                   const BufferFloat4 &weights, const BufferUInt &state,
                   const BufferFloat &unused, UInt flags, UInt depth,
                   Float3 contribution, Float3 diffuse, Float3 glossy,
                   Float3 bsdf_diffuse, Float3 bsdf_glossy, Float3 bsdf_sum) {
  UInt first = 0;
  Var<RenderKernelParameters> parameters;
  parameters.sample_clamp_direct = 2.0f;
  parameters.sample_clamp_indirect = 1.5f;
  PathKernelInvocation invocation{.config = config,
                                  .film_accumulation = mode,
                                  .combined = combined,
                                  .normal = film,
                                  .albedo = film,
                                  .light_passes = film,
                                  .sample_count = state,
                                  .volume_guiding_raw = film,
                                  .volume_guiding_denoised = state,
                                  .path_trace = weights,
                                  .sample_first = first,
                                  .sobol_table = film,
                                  .filter_table = unused,
                                  .parameters = parameters};
  invocation.pixel = 0;
  invocation.light_pass_base = 1;
  PathSampleContext sample{.invocation = invocation};
  sample.path_flags = flags;
  sample.path_depth = depth;
  sample.cycles_path_visibility = cycles_path_state::visibility_camera;
  sample.path_diffuse_weight = diffuse;
  sample.path_glossy_weight = glossy;
  if (operation == f::surface_nee) {
    auto transport = DirectLightTransportState::empty();
    transport.diffuse_bsdf = bsdf_diffuse;
    transport.glossy_bsdf = bsdf_glossy;
    transport.bsdf = bsdf_sum;
    Var<DirectLightTaskCall> task;
    prepare_surface_shadow_pass(task, sample, transport);
    task.path_depth = depth;
    task.path_visibility = sample.cycles_path_visibility;
    state.write(0u, task.path_flags);
    state.write(1u, task.path_depth);
    state.write(2u, task.path_visibility);
    weights.write(0u, make_float4(task.diffuse_weight, 0.0f));
    weights.write(1u, make_float4(task.glossy_weight, 0.0f));
    // Deferred shadow film must not observe these advanced main-path fields.
    sample.path_depth += 7u;
    sample.path_flags ^= cycles_path_state::flag_any_pass;
    sample.path_diffuse_weight = make_float3(0.75f);
    sample.path_glossy_weight = make_float3(0.125f);
    const auto value = evaluator.contribution(task, contribution, parameters);
    sample.accumulate_radiance_at_state(value, task.path_flags,
                                        task.path_visibility, task.path_depth);
    if (detached) {
      evaluator.accumulate_passes_atomic(film, UInt{1u}, task, value);
    } else {
      evaluator.accumulate_passes(sample, task, value);
    }
  } else if (operation == f::volume_nee) {
    accumulate_volume_nee_film(sample, invocation.clamp_contribution(contribution, depth));
  } else {
    const auto value = invocation.clamp_emission_contribution(contribution, depth);
    sample.accumulate_radiance(value);
    sample.accumulate_emission_or_background(
        value, operation == f::emission ? LightPassBuffer::emission
                                       : LightPassBuffer::environment);
  }
  if (mode == PathFilmAccumulation::serial) {
    combined.write(0u, make_float4(sample.radiance, 0.0f));
    film.write(1u, make_float4(sample.sample_diffuse_direct, 0.0f));
    film.write(2u, make_float4(sample.sample_diffuse_indirect, 0.0f));
    film.write(3u, make_float4(sample.sample_glossy_direct, 0.0f));
    film.write(4u, make_float4(sample.sample_glossy_indirect, 0.0f));
    film.write(5u, make_float4(sample.sample_transmission_direct, 0.0f));
    film.write(6u, make_float4(sample.sample_transmission_indirect, 0.0f));
    film.write(7u, make_float4(sample.sample_volume_direct, 0.0f));
    film.write(8u, make_float4(sample.sample_volume_indirect, 0.0f));
    film.write(9u, make_float4(sample.sample_emission, 0.0f));
    film.write(10u, make_float4(sample.sample_environment, 0.0f));
  }
}
} // namespace psycles::test_support::film_routing
