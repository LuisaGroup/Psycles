#include "path_kernel_builder.h"

#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_transitions.h"

#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

#include <luisa/dsl/coro_func.h>

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
            .constant_light_shader = false,
            .distant = false,
            .valid = false};
}

void DirectLightTransportState::accept(
    const SurfaceEvaluation &evaluation, Float3 proposal_weighted_bsdf,
    Float3 proposal_light_shader, Float3 proposal_direction,
    Float3 proposal_target_position, Bool proposal_distant,
    UInt proposal_light_object, UInt proposal_light_primitive,
    UInt proposal_shader_flags,
    Bool proposal_constant_light_shader) noexcept {
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
    constant_light_shader = proposal_constant_light_shader;
    distant = proposal_distant;
    valid = true;
}

namespace {

class CommonDirectLightTransportStage final : public DirectLightTransportStage {

  private:
    std::shared_ptr<const DirectLightTraceRecorder> _trace;
    DirectLightTaskEvaluator _evaluator;
    std::shared_ptr<const DirectLightTaskSink> _task_sink;
    bool _path_trace_enabled{};

  public:
    explicit CommonDirectLightTransportStage(
        std::shared_ptr<const DirectLightTraceRecorder> trace,
        DirectLightTaskEvaluator evaluator,
        std::shared_ptr<const DirectLightTaskSink> task_sink,
        bool path_trace_enabled)
        : _trace{std::move(trace)}, _evaluator{std::move(evaluator)},
          _task_sink{std::move(task_sink)},
          _path_trace_enabled{path_trace_enabled} {}

    DirectLightTransportPreparation
    prepare(DirectLightingContext &context,
            const DirectLightTransportState &transport) const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        auto &surface = context.surface;
        DirectLightTransportPreparation preparation;
        auto &task = preparation.task;

        $if(transport.valid) {
            const auto local_unshadowed =
                transport.weighted_bsdf * transport.light_shader;
            const auto constant_contribution = finalize_direct_light_sample(
                config.light_transport.light_sample_roulette_weight,
                local_unshadowed, sample.throughput,
                bounce.random().light_terminate_sample,
                invocation.parameters.light_inv_rr_threshold);
            const auto initial_contribution = select(
                transport.weighted_bsdf, constant_contribution,
                transport.constant_light_shader);
            const auto publish =
                (!transport.constant_light_shader) |
                any(initial_contribution != 0.0f);
            $if(publish) {
                const auto shadow =
                    surface.make_shadow_origin(transport.direction);
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
                    select(surface_ray::invalid_primitive,
                           surface.cycles_object_index, shadow.skip_self);
                const auto source_primitive =
                    select(surface_ray::invalid_primitive,
                           surface.cycles_primitive_index, shadow.skip_self);
                const auto shadow_differential =
                    cycles_ray_differential::for_surface_shadow(
                        sample.ray_dD, surface.differential_radius,
                        transport.average_roughness_squared);
                const auto direct = sample.path_depth == 0u;
                task.ray_origin = shadow.position;
                task.ray_direction = shadow_direction;
                task.unshadowed_contribution = initial_contribution;
                task.nee_path_throughput = sample.throughput;
                task.light_shader = transport.light_shader;
                task.shadow_transmittance = make_float3(1.0f);
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
                task.light_terminate_sample =
                    bounce.random().light_terminate_sample;
                task.source_object = source_object;
                task.source_primitive = source_primitive;
                task.light_object = transport.light_object;
                task.light_primitive = transport.light_primitive;
                task.constant_light_shader =
                    cast<std::uint32_t>(transport.constant_light_shader);
                task.shader_flags = transport.shader_flags;
                task.pixel = invocation.pixel;
                task.path_depth = sample.path_depth;
                task.path_flags = sample.path_flags;
                task.path_visibility = sample.cycles_path_visibility;
                task.diffuse_depth = sample.diffuse_depth;
                task.glossy_depth = sample.glossy_depth;
                task.transparent_depth = sample.transparent_depth;
                task.transmission_depth = sample.transmission_depth;
                preparation.valid = true;
            };
        };
        return preparation;
    }

    bool defer_until_after_surface_scatter(
        PathCoroutineCutPolicy cut_policy) const noexcept override {
        return cut_policy == PathCoroutineCutPolicy::cycles_wavefront &&
               !_path_trace_enabled && _task_sink == nullptr;
    }

    void emit(PathBounceContext &bounce,
              DirectLightTransportPreparation preparation,
              PathCoroutineCutPolicy cut_policy) const noexcept override {
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        auto &task = preparation.task;

        $if(preparation.valid) {
            if (_task_sink) {
                _task_sink->emit(
                    task, invocation.parameters.wavefront_frame_capacity);
            } else if (defer_until_after_surface_scatter(cut_policy)) {
                Bool active = true;
                Bool visible = false;

                // Non-constant light shaders take SHADE_LIGHT_NEE. The
                // constant-light edge bypasses directly to INTERSECT_SHADOW.
                // These are disjoint graph edges, so their branch-local frame
                // values do not interfere.
                $if(task.constant_light_shader == 0u) {
                    $suspend(path_transition::shade_light_nee);
                    active = _evaluator.shade_light_nee(
                        task, invocation.parameters);
                }
                $else {
                    active = any(task.unshadowed_contribution != 0.0f);
                };
                _trace->record_transport(
                    bounce, {.light_shader = task.light_shader,
                             .unshadowed = task.unshadowed_contribution});

                $while(active) {
                    $suspend(path_transition::intersect_shadow);
                    // The four-hit batch is produced after INTERSECT_SHADOW
                    // and consumed completely by SHADE_SHADOW. It is not part
                    // of the invariant task state and therefore cannot leak
                    // onto either adjacent coroutine edge.
                    const auto shadow_batch =
                        _evaluator.intersect(task, invocation.parameters);

                    $suspend(path_transition::shade_shadow);
                    const auto step =
                        _evaluator.shade_shadow(
                            task, shadow_batch, invocation.parameters);
                    active = step.continue_shadow;
                    visible = visible | step.visible;
                };

                $if(visible) {
                    const auto transmittance = task.shadow_transmittance;
                    const auto unclamped_contribution =
                        task.unshadowed_contribution * transmittance;
                    _trace->record_contribution(
                        bounce, unclamped_contribution);
                    const auto contribution = _evaluator.contribution(
                        task, transmittance, invocation.parameters);
                    sample.accumulate_radiance_at_state(
                        contribution, task.path_flags, task.path_visibility,
                        task.path_depth);
                    sample.accumulate_light_pass(
                        _evaluator.split(task, contribution));
                };
            } else {
                Bool active = true;
                $if(task.constant_light_shader == 0u) {
                    active = _evaluator.shade_light_nee(
                        task, invocation.parameters);
                };
                _trace->record_transport(
                    bounce, {.light_shader = task.light_shader,
                             .unshadowed = task.unshadowed_contribution});
                $if(active) {
                    const auto shadow_result =
                        _evaluator.trace(task, invocation.parameters);
                    const auto transmittance = shadow_result->transmittance;
                    _trace->record_shadow(
                        bounce,
                        {.origin = task.ray_origin,
                         .direction = task.ray_direction,
                         .minimum = task.ray_minimum,
                         .maximum = task.ray_maximum,
                         .cast_shadow =
                             (task.shader_flags &
                              cycles_shader_identity::cast_shadow) != 0u,
                         .source_object = task.source_object,
                         .source_primitive = task.source_primitive,
                         .skip_self =
                             task.source_object !=
                             surface_ray::invalid_primitive,
                         .light_object = task.light_object,
                         .light_primitive = task.light_primitive,
                         .first_hit = shadow_result->first_hit != 0u,
                         .first_object = shadow_result->first_object,
                         .first_primitive = shadow_result->first_primitive,
                         .first_kind = shadow_result->first_kind,
                         .first_distance = shadow_result->first_distance,
                         .first_barycentric =
                             shadow_result->first_barycentric,
                         .transmittance = transmittance});
                    $if(any(transmittance > 0.0f)) {
                        const auto unclamped_contribution =
                            task.unshadowed_contribution * transmittance;
                        _trace->record_contribution(
                            bounce, unclamped_contribution);
                        const auto contribution = _evaluator.contribution(
                            task, transmittance, invocation.parameters);
                        sample.accumulate_radiance_at_state(
                            contribution, task.path_flags,
                            task.path_visibility, task.path_depth);
                        sample.accumulate_light_pass(
                            _evaluator.split(task, contribution));
                    };
                };
            }
        };
    }
};

}// namespace

std::unique_ptr<DirectLightTransportStage> make_direct_light_transport_stage(
    std::shared_ptr<const DirectLightTraceRecorder> trace,
    DirectLightTaskEvaluator evaluator,
    std::shared_ptr<const DirectLightTaskSink> task_sink,
    bool path_trace_enabled) {
    return std::make_unique<CommonDirectLightTransportStage>(
        std::move(trace), std::move(evaluator), std::move(task_sink),
        path_trace_enabled);
}

}// namespace psycles::luisa_backend::detail
