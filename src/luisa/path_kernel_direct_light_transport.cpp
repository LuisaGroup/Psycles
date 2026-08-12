#include "path_kernel_builder.h"

#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_direct_light_trace.h"

#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {

DirectLightTransportState DirectLightTransportState::empty() noexcept {
  return {.weighted_bsdf = make_float3(0.0f),
          .light_shader = make_float3(0.0f),
          .direction = make_float3(0.0f, 0.0f, 1.0f),
          .target_position = make_float3(0.0f),
          .bsdf = make_float3(0.0f),
          .diffuse_bsdf = make_float3(0.0f),
          .glossy_bsdf = make_float3(0.0f),
          .light_object = surface_ray::invalid_primitive,
          .light_primitive = surface_ray::invalid_primitive,
          .shader_flags = 0u,
          .average_roughness_squared = 0.0f,
          .distant = false,
          .valid = false};
}

void DirectLightTransportState::accept(
    const SurfaceEvaluation &evaluation, Float3 proposal_weighted_bsdf,
    Float3 proposal_light_shader, Float3 proposal_direction,
    Float3 proposal_target_position, Bool proposal_distant,
    UInt proposal_light_object, UInt proposal_light_primitive,
    UInt proposal_shader_flags) noexcept {
  weighted_bsdf = proposal_weighted_bsdf;
  light_shader = proposal_light_shader;
  direction = proposal_direction;
  target_position = proposal_target_position;
  bsdf = evaluation.f;
  diffuse_bsdf = evaluation.diffuse_f;
  glossy_bsdf = evaluation.glossy_f;
  light_object = proposal_light_object;
  light_primitive = proposal_light_primitive;
  shader_flags = proposal_shader_flags;
  average_roughness_squared = evaluation.average_roughness_squared;
  distant = proposal_distant;
  valid = true;
}

namespace {

class CommonDirectLightTransportStage final : public DirectLightTransportStage {

private:
  std::shared_ptr<const DirectLightTraceRecorder> _trace;
  DirectLightTaskEvaluator _evaluator;
  std::shared_ptr<const DirectLightTaskSink> _task_sink;

public:
  explicit CommonDirectLightTransportStage(
      std::shared_ptr<const DirectLightTraceRecorder> trace,
      DirectLightTaskEvaluator evaluator,
      std::shared_ptr<const DirectLightTaskSink> task_sink)
      : _trace{std::move(trace)}, _evaluator{std::move(evaluator)},
        _task_sink{std::move(task_sink)} {}

  void
  emit(DirectLightingContext &context,
       const DirectLightTransportState &transport) const noexcept override {
    auto &bounce = context.bounce;
    auto &sample = bounce.sample;
    auto &invocation = sample.invocation;
    const auto &config = invocation.config;
    auto &surface = context.surface;

    $if(transport.valid) {
      const auto unshadowed = transport.weighted_bsdf * transport.light_shader;
      const auto roulette_weight = invocation.sample_light_roulette(
          unshadowed, bounce.random().light_terminate_sample);
      const auto surviving_unshadowed = unshadowed * roulette_weight;
      _trace->record_transport(
          bounce, {.light_shader = transport.light_shader,
                   .unshadowed = sample.throughput * surviving_unshadowed});
      $if(any(surviving_unshadowed != 0.0f)) {
        const auto shadow = surface.make_shadow_origin(transport.direction);
        Float3 shadow_direction = transport.direction;
        Float shadow_maximum = ray_maximum;
        $if(!transport.distant) {
          const auto shadow_offset =
              transport.target_position - shadow.position;
          const auto shadow_distance =
              sqrt(max(length_squared(shadow_offset), 1.0e-20f));
          shadow_direction = shadow_offset / shadow_distance;
          shadow_maximum = shadow_distance;
        };
        const auto source_object =
            select(surface_ray::invalid_primitive, surface.cycles_object_index,
                   shadow.skip_self);
        const auto source_primitive =
            select(surface_ray::invalid_primitive,
                   surface.cycles_primitive_index, shadow.skip_self);
        const auto cast_shadow = (transport.shader_flags &
                                  cycles_shader_identity::cast_shadow) != 0u;
        const auto shadow_differential =
            cycles_ray_differential::for_surface_shadow(
                sample.ray_dD, surface.differential_radius,
                transport.average_roughness_squared);
        const auto direct = sample.path_depth == 0u;
        Var<DirectLightTaskCall> task;
        task.ray_origin = shadow.position;
        task.ray_direction = shadow_direction;
        task.unshadowed_contribution = sample.throughput * surviving_unshadowed;
        task.diffuse_weight =
            select(sample.path_diffuse_weight,
                   config.light_transport.light_component_ratio(
                       transport.diffuse_bsdf, transport.bsdf),
                   direct);
        task.glossy_weight =
            select(sample.path_glossy_weight,
                   config.light_transport.light_component_ratio(
                       transport.glossy_bsdf, transport.bsdf),
                   direct);
        task.ray_minimum = 0.0f;
        task.ray_maximum = shadow_maximum;
        task.ray_dP = shadow_differential.position;
        task.ray_dD = shadow_differential.direction;
        task.source_object = source_object;
        task.source_primitive = source_primitive;
        task.light_object = transport.light_object;
        task.light_primitive = transport.light_primitive;
        task.pixel = invocation.pixel;
        task.path_depth = sample.path_depth;
        task.path_flags = sample.path_flags;
        task.path_visibility = sample.cycles_path_visibility;
        task.diffuse_depth = sample.diffuse_depth;
        task.glossy_depth = sample.glossy_depth;
        task.transparent_depth = sample.transparent_depth;
        task.transmission_depth = sample.transmission_depth;

        if (_task_sink) {
          _task_sink->emit(task);
        } else {
          const auto shadow_result =
              _evaluator.trace(task, invocation.parameters);
          const auto transmittance = shadow_result->transmittance;
          _trace->record_shadow(
              bounce, {.origin = task.ray_origin,
                       .direction = task.ray_direction,
                       .minimum = task.ray_minimum,
                       .maximum = task.ray_maximum,
                       .cast_shadow = cast_shadow,
                       .source_object = source_object,
                       .source_primitive = source_primitive,
                       .skip_self = shadow.skip_self,
                       .light_object = transport.light_object,
                       .light_primitive = transport.light_primitive,
                       .first_hit = shadow_result->first_hit != 0u,
                       .first_object = shadow_result->first_object,
                       .first_primitive = shadow_result->first_primitive,
                       .first_kind = shadow_result->first_kind,
                       .first_distance = shadow_result->first_distance,
                       .first_barycentric = shadow_result->first_barycentric,
                       .transmittance = transmittance});
          $if(any(transmittance > 0.0f)) {
            const auto unclamped_contribution =
                task.unshadowed_contribution * transmittance;
            _trace->record_contribution(bounce, unclamped_contribution);
            const auto contribution = _evaluator.contribution(
                task, transmittance, invocation.parameters);
            sample.accumulate_radiance(contribution);
            sample.accumulate_light_pass(_evaluator.split(task, contribution));
          };
        }
      };
    };
  }
};

} // namespace

std::unique_ptr<DirectLightTransportStage> make_direct_light_transport_stage(
    std::shared_ptr<const DirectLightTraceRecorder> trace,
    DirectLightTaskEvaluator evaluator,
    std::shared_ptr<const DirectLightTaskSink> task_sink) {
  return std::make_unique<CommonDirectLightTransportStage>(
      std::move(trace), std::move(evaluator), std::move(task_sink));
}

} // namespace psycles::luisa_backend::detail
