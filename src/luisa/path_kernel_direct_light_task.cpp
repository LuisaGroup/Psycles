#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"

namespace psycles::luisa_backend::detail {
Var<ShadowTraceResultCall> DirectLightTaskEvaluator::trace(
    const Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Var<luisa::compute::Ray> shadow_ray = make_ray(
      task.ray_origin, task.ray_direction, task.ray_minimum, task.ray_maximum);
  return trace_shadow(
      shadow_ray, task.ray_dP, task.ray_dD, task.source_object,
      task.source_primitive, task.light_object, task.light_primitive,
      parameters.transparent_max_bounces,
      pack_shader_evaluation_state(cycles_path_state::shadow_shader_state(
          task.path_depth, task.diffuse_depth, task.glossy_depth,
          task.transparent_depth, task.transmission_depth)));
}

Float3 DirectLightTaskEvaluator::contribution(
    const Var<DirectLightTaskCall> &task, Float3 transmittance,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  return clamp_contribution(task.unshadowed_contribution * transmittance,
                            task.path_depth, parameters.sample_clamp_direct,
                            parameters.sample_clamp_indirect);
}

Var<LightPassContributionCall>
DirectLightTaskEvaluator::split(const Var<DirectLightTaskCall> &task,
                                Float3 contribution) const noexcept {
  return split_scattered_light(contribution, task.diffuse_weight,
                               task.glossy_weight, task.path_depth == 0u);
}

void DirectLightTaskEvaluator::emit_atomic(
    const Var<DirectLightTaskCall> &task, const DirectLightTaskFilm &film,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  const auto shadow = trace(task, parameters);
  $if(any(shadow->transmittance > 0.0f)) {
    const auto value = contribution(task, shadow->transmittance, parameters);
    atomic_accumulate_radiance(
        film.combined, film.volume_guiding_raw, task.pixel,
        task.pixel * volume_guiding::raw_pixel_stride, volume_guiding,
        task.path_flags, task.path_visibility, task.path_depth, value);
    atomic_accumulate_light_passes(film.light_passes,
                                   task.pixel * light_pass_buffer_count,
                                   split(task, value));
  };
}

DirectLightTaskEvaluator
make_direct_light_task_evaluator(const PathKernelConfig &config) noexcept {
  return {.trace_shadow = config.trace_shadow,
          .clamp_contribution = config.light_transport.clamp_light_contribution,
          .split_scattered_light = config.light_transport.split_scattered_light,
          .volume_guiding = config.volume_state != nullptr};
}

} // namespace psycles::luisa_backend::detail
