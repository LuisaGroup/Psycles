#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {

Float3 finalize_direct_light_sample(
    const LightSampleRouletteCallable &light_sample_roulette,
    Float3 local_unshadowed, Float3 path_throughput,
    Float light_terminate_sample, Float inverse_threshold) noexcept {
  const auto roulette_weight = light_sample_roulette(
      local_unshadowed, light_terminate_sample, inverse_threshold);
  return path_throughput * local_unshadowed * roulette_weight;
}

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

Bool DirectLightTaskEvaluator::shade_light_nee(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  const auto local_unshadowed =
      task.unshadowed_contribution * task.light_shader;
  task.unshadowed_contribution = finalize_direct_light_sample(
      light_sample_roulette, local_unshadowed, task.nee_path_throughput,
      task.light_terminate_sample, parameters.light_inv_rr_threshold);
  return any(task.unshadowed_contribution != 0.0f);
}

void DirectLightTaskEvaluator::intersect(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  const auto ray = make_ray(task.ray_origin, task.ray_direction,
                            task.ray_minimum, task.ray_maximum);
  const auto remaining =
      parameters.transparent_max_bounces -
      min(task.transparent_depth, parameters.transparent_max_bounces);
  task.shadow_batch =
      intersect_shadow(ray, task.source_object, task.source_primitive,
                       task.light_object, task.light_primitive, remaining);
}

DirectLightShadowStep DirectLightTaskEvaluator::shade_shadow(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Bool continue_shadow = false;
  Bool visible = false;

  $if(task.shadow_batch.blocked == 0u) {
    Bool carries_light = true;
    Float last_distance = task.ray_minimum;
    const auto ray = make_ray(task.ray_origin, task.ray_direction,
                              task.ray_minimum, task.ray_maximum);
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      const auto shade =
          static_cast<std::uint32_t>(index) < task.shadow_batch.count;
      $if(shade & carries_light) {
        const auto &hit =
            task.shadow_batch.hits[static_cast<luisa::uint>(index)];
        const auto surface = shade_shadow_surface(
            ray, hit, task.ray_dP, task.ray_dD,
            pack_shader_evaluation_state(cycles_path_state::shadow_shader_state(
                task.path_depth, task.diffuse_depth, task.glossy_depth,
                task.transparent_depth, task.transmission_depth)));
        const auto transparent = surface->transmittance;
        carries_light =
            max(transparent.x, max(transparent.y, transparent.z)) > 0.0f;
        $if(carries_light) {
          task.shadow_transmittance *= transparent;
          task.transparent_depth += 1u;
          last_distance = hit->distance;
        };
      };
    }
    $if(carries_light) {
      const auto has_remaining =
          task.shadow_batch.total > task.shadow_batch.count;
      $if(has_remaining) {
        task.ray_minimum = surface_ray::intersection_t_offset(last_distance);
        continue_shadow = true;
      }
      $else { visible = true; };
    };
  };
  return {.continue_shadow = std::move(continue_shadow),
          .visible = std::move(visible)};
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
    const Var<DirectLightTaskCall> &input, const DirectLightTaskFilm &film,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  auto task = def(input);
  Bool active = true;
  $if(task.constant_light_shader == 0u) {
    active = shade_light_nee(task, parameters);
  };
  $if(active) {
    const auto shadow = trace(task, parameters);
    $if(any(shadow->transmittance > 0.0f)) {
      const auto value =
          contribution(task, shadow->transmittance, parameters);
      atomic_accumulate_radiance(
          film.combined, film.volume_guiding_raw, task.pixel,
          task.pixel * volume_guiding::raw_pixel_stride, volume_guiding,
          task.path_flags, task.path_visibility, task.path_depth, value);
      atomic_accumulate_light_passes(
          film.light_passes, task.pixel * light_pass_buffer_count,
          split(task, value));
    };
  };
}

DirectLightTaskEvaluator
make_direct_light_task_evaluator(const PathKernelConfig &config) noexcept {
  return {.intersect_shadow = config.intersect_shadow,
          .shade_shadow_surface = config.shade_shadow_surface,
          .trace_shadow = config.trace_shadow,
          .light_sample_roulette =
              config.light_transport.light_sample_roulette_weight,
          .clamp_contribution = config.light_transport.clamp_light_contribution,
          .split_scattered_light = config.light_transport.split_scattered_light,
          .volume_guiding = config.volume_state != nullptr};
}

} // namespace psycles::luisa_backend::detail
