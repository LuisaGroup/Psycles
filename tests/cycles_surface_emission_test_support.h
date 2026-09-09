#pragma once

#include "path_kernel_surface_emission.h"

namespace psycles::test_support::surface_emission {
using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;

template <typename Function> Function unbound() {
  return Function{
      luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{}};
}

inline PathKernelConfig config(bool nee, bool trace) {
  auto scene = std::make_shared<LuisaSceneData>();
  scene->emissive_triangle_count = 1;
  scene->triangle_area_pdf = 0.25f;
  return {.scene = scene,
          .next_event_estimation = nee,
          .path_trace_enabled = trace,
          .light_transport = make_light_transport_callables(
              contract::DirectLightSampling::multiple_importance_sampling),
          .light_distribution_sample =
              unbound<LightDistributionSampleCallable>(),
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
}

// Exercise the actual surface-stage emission operation and its real film
// methods. No shader, sampler or light evaluator is replaced by host math.
inline void record(const PathKernelConfig &config, PathFilmAccumulation mode,
                   const BufferFloat4 &film, const BufferFloat4 &trace,
                   const BufferUInt &counts, const BufferFloat &dummy,
                   UInt shader_flags, Bool subsurface_exit, UInt path_flags,
                   UInt depth, Float3 emission, UInt sampling,
                   Float direct_limit, Float indirect_limit) {
  UInt first = 0;
  Var<RenderKernelParameters> parameters;
  parameters.sample_clamp_direct = direct_limit;
  parameters.sample_clamp_indirect = indirect_limit;
  PathKernelInvocation invocation{.config = config,
                                  .film_accumulation = mode,
                                  .combined = film,
                                  .normal = film,
                                  .albedo = film,
                                  .light_passes = film,
                                  .sample_count = counts,
                                  .volume_guiding_raw = film,
                                  .volume_guiding_denoised = counts,
                                  .path_trace = trace,
                                  .sample_first = first,
                                  .sobol_table = film,
                                  .filter_table = dummy,
                                  .parameters = parameters};
  invocation.pixel = 0;
  invocation.light_pass_base = 1;
  PathSampleContext sample{.invocation = invocation};
  sample.ray =
      make_ray(make_float3(0, 0, 1), make_float3(0, 0, -1), 0.0f, 10.0f);
  sample.throughput = make_float3(0.5f, 0.75f, 1.25f);
  sample.path_diffuse_weight = make_float3(0.25f, 0.125f, 0.5f);
  sample.path_glossy_weight = make_float3(0.125f, 0.25f, 0.25f);
  sample.path_flags = path_flags;
  sample.path_depth = depth;
  sample.previous_bsdf_pdf = 0.375f;
  sample.path_trace_active = true;
  UInt step = 0;
  PathBounceContext bounce{.sample = sample,
                           .path_step = step,
                           .random_state = nullptr,
                           .subsurface_exit = subsurface_exit};
  SurfaceGeometryContext surface{.bounce = bounce};
  surface.emission_sampling = sampling;
  surface.wp0 = make_float3(0);
  surface.wp1 = make_float3(1, 0, 0);
  surface.wp2 = make_float3(0, 1, 0);
  surface.point.geometric_normal = make_float3(0, 0, 1);
  auto preparation = SurfacePreparation::zero(surface.point);
  preparation.runtime_flags = shader_flags;
  preparation.emission = emission;
  emit_surface_emission(surface, preparation,
                        *make_emissive_triangle_component());
  if (mode == PathFilmAccumulation::serial) {
    film.write(0u, make_float4(sample.radiance, 0.0f));
    film.write(1u, make_float4(sample.sample_diffuse_direct, 0.0f));
    film.write(2u, make_float4(sample.sample_diffuse_indirect, 0.0f));
    film.write(3u, make_float4(sample.sample_glossy_direct, 0.0f));
    film.write(4u, make_float4(sample.sample_glossy_indirect, 0.0f));
    film.write(5u, make_float4(sample.sample_transmission_direct, 0.0f));
    film.write(6u, make_float4(sample.sample_transmission_indirect, 0.0f));
    film.write(7u, make_float4(sample.sample_volume_direct, 0.0f));
    film.write(8u, make_float4(sample.sample_volume_indirect, 0.0f));
    film.write(9u, make_float4(sample.sample_emission, 0.0f));
  }
}
} // namespace psycles::test_support::surface_emission
