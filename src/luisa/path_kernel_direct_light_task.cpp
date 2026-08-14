#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

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

Bool DirectLightTaskEvaluator::shade_light_nee(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  const auto evaluated =
      task.unshadowed_contribution * task.light_shader;
  const auto roulette_weight = light_sample_roulette(
      evaluated, task.light_terminate_sample,
      parameters.light_inv_rr_threshold);
  task.unshadowed_contribution = evaluated * roulette_weight;
  return any(task.unshadowed_contribution != 0.0f);
}

void DirectLightTaskEvaluator::intersect(
    Var<DirectLightTaskCall> &task) const noexcept {
  const auto ray = make_ray(task.ray_origin, task.ray_direction,
                            task.ray_minimum, task.ray_maximum);
  const auto hit = intersect_shadow(
      ray, task.source_object, task.source_primitive,
      task.light_object, task.light_primitive);
  task.shadow_hit_instance = hit->instance;
  task.shadow_hit_primitive = hit->primitive;
  task.shadow_hit_type = hit->hit_type;
  task.shadow_hit_distance = hit->distance;
  task.shadow_hit_barycentric = hit->barycentric;
}

DirectLightShadowStep DirectLightTaskEvaluator::shade_shadow(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Bool continue_shadow = false;
  Bool visible = false;
  const auto missed =
      task.shadow_hit_type == static_cast<std::uint32_t>(
                                  luisa::compute::HitType::Miss);
  $if(missed) {
    visible = true;
  }
  $else {
    // Cycles' INTERSECT_SHADOW makes the next hit opaque once the transparent
    // bounce budget is exhausted. Material evaluation is therefore reachable
    // iff this strict precondition holds.
    $if(task.transparent_depth < parameters.transparent_max_bounces) {
      Var<ShadowIntersectionCall> hit;
      hit->instance = task.shadow_hit_instance;
      hit->primitive = task.shadow_hit_primitive;
      hit->hit_type = task.shadow_hit_type;
      hit->distance = task.shadow_hit_distance;
      hit->barycentric = task.shadow_hit_barycentric;
      const auto ray = make_ray(task.ray_origin, task.ray_direction,
                                task.ray_minimum, task.ray_maximum);
      const auto surface = shade_shadow_surface(
          ray, hit, task.ray_dP, task.ray_dD,
          pack_shader_evaluation_state(
              cycles_path_state::shadow_shader_state(
                  task.path_depth, task.diffuse_depth,
                  task.glossy_depth, task.transparent_depth,
                  task.transmission_depth)));
      const auto transparent = surface->transmittance;
      const auto carries_light =
          max(transparent.x, max(transparent.y, transparent.z)) > 0.0f;
      $if(carries_light) {
        task.shadow_transmittance *= transparent;
        task.transparent_depth += 1u;
        task.ray_minimum = surface_ray::intersection_t_offset(
            task.shadow_hit_distance);
        continue_shadow = true;
      };
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
