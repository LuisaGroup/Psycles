#include "path_kernel_builder.h"

#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_direct_light_trace.h"
#include "virtual_texture_cache.h"
#include "path_kernel_transitions.h"

#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

#include <luisa/dsl/coro_func.h>

namespace psycles::luisa_backend::detail {

DirectLightSampleState DirectLightSampleState::empty(
    contract::SpectralMode spectral_mode) noexcept {
    DirectLightSampleState state{
        .direction = make_float3(0.0f, 0.0f, 1.0f),
        .target_position = make_float3(0.0f),
        .light_normal = make_float3(0.0f, 0.0f, 1.0f),
        .light_uv = make_float2(0.0f),
        .barycentric = make_float2(0.0f),
        .radiometric_weight = make_float3(0.0f),
        .light_shader = make_float3(0.0f),
        .pdf = 0.0f,
        .normalization_pdf = 1.0f,
        .distance = ray_maximum,
        .emitter_kind = static_cast<std::uint32_t>(
            sampling::LightDistributionEmitterKind::sentinel),
        .emitter_index = surface_ray::invalid_primitive,
        .light_object = surface_ray::invalid_primitive,
        .light_primitive = surface_ray::invalid_primitive,
        .shader_flags = 0u,
        .apply_mis = false,
        .constant_light_shader = false,
        .distant = false,
        .valid = false};
    if (spectral_mode == contract::SpectralMode::hero4) {
        state.spectral_radiometric_weight.emplace(make_float4(0.0f));
        state.spectral_light_shader.emplace(make_float4(0.0f));
    }
    return state;
}

void DirectLightSampleState::accept(
    const DirectLightSampleState &proposal) noexcept {
    // Every caller is guarded by equality with a distinct
    // LightDistributionEmitterKind value. Consequently accepted writes are
    // pairwise disjoint and this is a phi-like merge, not last-writer-wins
    // conflict resolution.
    direction = proposal.direction;
    target_position = proposal.target_position;
    light_normal = proposal.light_normal;
    light_uv = proposal.light_uv;
    barycentric = proposal.barycentric;
    radiometric_weight = proposal.radiometric_weight;
    light_shader = proposal.light_shader;
    pdf = proposal.pdf;
    normalization_pdf = proposal.normalization_pdf;
    distance = proposal.distance;
    emitter_kind = proposal.emitter_kind;
    emitter_index = proposal.emitter_index;
    light_object = proposal.light_object;
    light_primitive = proposal.light_primitive;
    shader_flags = proposal.shader_flags;
    apply_mis = proposal.apply_mis;
    constant_light_shader = proposal.constant_light_shader;
    distant = proposal.distant;
    valid = proposal.valid;
    if (spectral_radiometric_weight.has_value() &&
        proposal.spectral_radiometric_weight.has_value()) {
        *spectral_radiometric_weight = *proposal.spectral_radiometric_weight;
    }
    if (spectral_light_shader.has_value() &&
        proposal.spectral_light_shader.has_value()) {
        *spectral_light_shader = *proposal.spectral_light_shader;
    }
}

DirectLightTransportState DirectLightTransportState::empty(
    contract::SpectralMode spectral_mode) noexcept {
    DirectLightTransportState state{
        .weighted_bsdf = make_float3(0.0f),
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
        .light_terminate_sample = 0.0f,
        .constant_light_shader = false,
        .distant = false,
        .valid = false};
    if (spectral_mode == contract::SpectralMode::hero4) {
        state.spectral_weighted_bsdf.emplace(make_float4(0.0f));
        state.spectral_light_shader.emplace(make_float4(0.0f));
        state.spectral_bsdf.emplace(make_float4(0.0f));
        state.spectral_diffuse_bsdf.emplace(make_float4(0.0f));
        state.spectral_glossy_bsdf.emplace(make_float4(0.0f));
    }
    return state;
}

void DirectLightTransportState::accept(
    const DirectLightTransportState &other) noexcept {
    weighted_bsdf = other.weighted_bsdf;
    light_shader = other.light_shader;
    direction = other.direction;
    target_position = other.target_position;
    bsdf = other.bsdf;
    diffuse_bsdf = other.diffuse_bsdf;
    glossy_bsdf = other.glossy_bsdf;
    light_object = other.light_object;
    light_primitive = other.light_primitive;
    shader_flags = other.shader_flags;
    average_roughness_squared = other.average_roughness_squared;
    light_terminate_sample = other.light_terminate_sample;
    constant_light_shader = other.constant_light_shader;
    distant = other.distant;
    valid = other.valid;
    const auto copy_opt = [](std::optional<Float4> &dst,
                             const std::optional<Float4> &src) noexcept {
        if (dst.has_value() && src.has_value()) {
            *dst = *src;
        }
    };
    copy_opt(spectral_weighted_bsdf, other.spectral_weighted_bsdf);
    copy_opt(spectral_light_shader, other.spectral_light_shader);
    copy_opt(spectral_bsdf, other.spectral_bsdf);
    copy_opt(spectral_diffuse_bsdf, other.spectral_diffuse_bsdf);
    copy_opt(spectral_glossy_bsdf, other.spectral_glossy_bsdf);
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

void DirectLightTransportState::accept(
    const SurfaceEvaluation &evaluation,
    const cycles_spectrum::Spectrum &proposal_weighted_bsdf,
    const cycles_spectrum::Spectrum &proposal_light_shader,
    Float3 proposal_direction, Float3 proposal_target_position,
    Bool proposal_distant, UInt proposal_light_object,
    UInt proposal_light_primitive, UInt proposal_shader_flags,
    Bool proposal_constant_light_shader,
    const cycles_spectrum::SpectralContext *spectral) noexcept {
    if (proposal_weighted_bsdf.is_xyz3()) {
        accept(evaluation, proposal_weighted_bsdf.xyz3(),
               proposal_light_shader.vec3(), proposal_direction,
               proposal_target_position, proposal_distant,
               proposal_light_object, proposal_light_primitive,
               proposal_shader_flags, proposal_constant_light_shader);
        if (spectral != nullptr && evaluation.spectral_f.has_value()) {
            bsdf = cycles_spectrum::Spectrum::from_eval4(
                       spectral, *evaluation.spectral_f)
                       .xyz3();
        }
        if (spectral != nullptr && evaluation.spectral_diffuse_f.has_value()) {
            diffuse_bsdf = cycles_spectrum::Spectrum::from_eval4(
                               spectral, *evaluation.spectral_diffuse_f)
                               .xyz3();
        }
        if (spectral != nullptr && evaluation.spectral_glossy_f.has_value()) {
            glossy_bsdf = cycles_spectrum::Spectrum::from_eval4(
                              spectral, *evaluation.spectral_glossy_f)
                              .xyz3();
        }
        return;
    }
    accept(evaluation,
           proposal_weighted_bsdf.is_hero4()
               ? proposal_weighted_bsdf.lanes().xyz()
               : proposal_weighted_bsdf.rgb(),
           proposal_light_shader.is_hero4()
               ? proposal_light_shader.lanes().xyz()
               : proposal_light_shader.rgb(),
           proposal_direction, proposal_target_position, proposal_distant,
           proposal_light_object, proposal_light_primitive,
           proposal_shader_flags, proposal_constant_light_shader);
    if (spectral_weighted_bsdf.has_value()) {
        *spectral_weighted_bsdf = proposal_weighted_bsdf.lanes();
    }
    if (spectral_light_shader.has_value()) {
        *spectral_light_shader = proposal_light_shader.lanes();
    }
    if (spectral_bsdf.has_value()) {
        *spectral_bsdf = evaluation.spectral_f.value_or(
            make_float4(evaluation.f, 0.0f));
    }
    if (spectral_diffuse_bsdf.has_value()) {
        *spectral_diffuse_bsdf = evaluation.spectral_diffuse_f.value_or(
            make_float4(evaluation.diffuse_f, 0.0f));
    }
    if (spectral_glossy_bsdf.has_value()) {
        *spectral_glossy_bsdf = evaluation.spectral_glossy_f.value_or(
            make_float4(evaluation.glossy_f, 0.0f));
    }
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
        const bool spectral = sample.is_spectral();
        const bool hero4 = sample.is_hero4();
        if (spectral) {
            preparation.spectral.emplace(DirectLightSpectralExtension{
                .u_lambda = *sample.spectral_u_lambda});
        }

        $if(transport.valid) {
            const Float light_terminate_sample =
                bounce.random_state != nullptr
                    ? bounce.random().light_terminate_sample
                    : transport.light_terminate_sample;
            Float3 initial_rgb = make_float3(0.0f);
            Float4 initial_spec = make_float4(0.0f);
            Float4 light_shader_spec = make_float4(0.0f);
            Bool publish = false;
            if (hero4) {
                const auto weighted_spec =
                    transport.spectral_weighted_bsdf.value_or(
                        make_float4(transport.weighted_bsdf, 0.0f));
                light_shader_spec = transport.spectral_light_shader.value_or(
                    make_float4(transport.light_shader, 0.0f));
                const auto local_unshadowed_spec =
                    weighted_spec * light_shader_spec;
                const auto constant_spec = finalize_direct_light_sample_spectral(
                    config.light_transport.light_sample_roulette_weight,
                    local_unshadowed_spec, sample.throughput.lanes(),
                    sample.spectral_secondary_terminated(),
                    light_terminate_sample,
                    invocation.parameters.light_inv_rr_threshold);
                initial_spec = select(weighted_spec, constant_spec,
                                      transport.constant_light_shader);
                publish = (!transport.constant_light_shader) |
                          any(initial_spec != 0.0f);
            } else {
                const auto local_unshadowed =
                    transport.weighted_bsdf * transport.light_shader;
                const auto constant_contribution = finalize_direct_light_sample(
                    config.light_transport.light_sample_roulette_weight,
                    local_unshadowed, sample.throughput.vec3(),
                    light_terminate_sample,
                    invocation.parameters.light_inv_rr_threshold);
                initial_rgb = select(
                    transport.weighted_bsdf, constant_contribution,
                    transport.constant_light_shader);
                publish =
                    (!transport.constant_light_shader) |
                    any(initial_rgb != 0.0f);
            }
            $if(publish) {
                auto shadow =
                    surface.make_shadow_terminator_origin(transport.direction);
                Float3 shadow_direction = transport.direction;
                Float shadow_maximum = ray_maximum;
                $if(!transport.distant) {
                    const auto shadow_offset =
                        transport.target_position - shadow.position;
                    shadow_direction = surface_ray::safe_normalize_length(
                        shadow_offset, shadow_maximum);
                };
                // Cycles' integrate_surface_ray_offset runs after finite
                // setup and observes the final setup direction. It may move
                // P for an ambiguous source triangle, but never re-aims D/t.
                shadow.position = surface.apply_shadow_certificate(
                    shadow.position, shadow_direction, shadow.skip_self);
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
                task.ray_origin = shadow.position;
                task.ray_direction = shadow_direction;
                if (hero4) {
                    const auto path_t = sample.throughput.lanes();
                    task.unshadowed_contribution = initial_spec.xyz();
                    task.nee_path_throughput = path_t.xyz();
                    task.light_shader = light_shader_spec.xyz();
                    task.shadow_throughput = initial_spec.xyz();
                    preparation.spectral->unshadowed_contribution_w =
                        initial_spec.w;
                    preparation.spectral->nee_path_throughput_w = path_t.w;
                    preparation.spectral->light_shader_w = light_shader_spec.w;
                    preparation.spectral->shadow_throughput_w = initial_spec.w;
                } else {
                    task.unshadowed_contribution = initial_rgb;
                    task.nee_path_throughput = sample.throughput.vec3();
                    task.light_shader = transport.light_shader;
                    task.shadow_throughput = initial_rgb;
                }
                prepare_surface_shadow_pass(task, sample, transport);
                task.ray_minimum = 0.0f;
                task.ray_maximum = shadow_maximum;
                task.ray_dP = shadow_differential.position;
                task.ray_dD = shadow_differential.direction;
                task.ray_time = surface.point.time;
                task.light_terminate_sample = light_terminate_sample;
                task.sample_index = sample.sample_index;
                task.rng_hash = sample.rng_hash;
                task.rng_offset = sample.cycles_rng_offset;
                task.source_object = source_object;
                task.source_primitive = source_primitive;
                task.light_object = transport.light_object;
                task.light_primitive = transport.light_primitive;
                task.constant_light_shader =
                    cast<std::uint32_t>(transport.constant_light_shader);
                task.shader_flags = transport.shader_flags;
                task.pixel = invocation.pixel;
                task.path_depth = sample.path_depth;
                task.path_visibility = sample.cycles_path_visibility;
                task.diffuse_depth = sample.diffuse_depth;
                task.glossy_depth = sample.glossy_depth;
                task.transparent_depth = sample.transparent_depth.cast<std::uint16_t>();
                task.transmission_depth = sample.transmission_depth;
                task.portal_depth = sample.portal_depth.cast<std::uint16_t>();
                task.volume_bounds_bounce = sample.volume_bounds_bounce;
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

        // The traced NEE contribution is a total function of a surface
        // event.  Initialize it to the additive identity before any of the
        // early-exit predicates below; a shadow path that reaches a visible
        // terminal state overwrites this value with its final unshadowed
        // contribution times transmittance.  This makes light-sampling
        // failure, roulette rejection, opaque traversal, and transparent
        // shading that becomes opaque observationally equivalent to the
        // zero contribution they have in the film, without duplicating a
        // diagnostic write in every control-flow exit.
        _trace->record_contribution(bounce, make_float3(0.0f));

        $if(preparation.valid) {
            DirectLightSpectralExtension *spectral_ptr =
                preparation.spectral.has_value() ? &*preparation.spectral
                                                 : nullptr;
            if (_task_sink) {
                _task_sink->emit(
                    task, invocation.parameters.wavefront_frame_capacity,
                    spectral_ptr);
            } else if (defer_until_after_surface_scatter(cut_policy)) {
                Bool active = true;
                Bool visible = false;

                // Non-constant light shaders take SHADE_LIGHT_NEE. The
                // constant-light edge bypasses directly to INTERSECT_SHADOW.
                // These are disjoint graph edges, so their branch-local frame
                // values do not interfere.
                $if(task.constant_light_shader == 0u) {
                    if (sample.vt_wait_resident) {
                        $loop {
                            $suspend(path_transition::shade_light_nee);
                            UInt vt_miss = vt_invalid_page;
                            active = _evaluator.shade_light_nee(
                                task, invocation.parameters, &vt_miss,
                                spectral_ptr);
                            $if(vt_miss != vt_invalid_page) {
                                sample.vt_wait_resident(vt_miss);
                                $continue;
                            };
                            $break;
                        };
                    } else {
                        $suspend(path_transition::shade_light_nee);
                        active = _evaluator.shade_light_nee(
                            task, invocation.parameters, nullptr, spectral_ptr);
                    }
                }
                $else {
                    active = spectral_ptr != nullptr
                                 ? any(make_float4(
                                           task.unshadowed_contribution,
                                           spectral_ptr->unshadowed_contribution_w) !=
                                       0.0f)
                                 : any(task.unshadowed_contribution != 0.0f);
                };
                _trace->record_transport(
                    bounce, {.light_shader = task.light_shader,
                             .unshadowed = task.unshadowed_contribution});

                _evaluator.trace_staged(
                    task, invocation.parameters, active, visible,
                    sample.vt_wait_resident, spectral_ptr);

                $if(visible) {
                    const auto unclamped_contribution = task.shadow_throughput;
                    _trace->record_contribution(
                        bounce, unclamped_contribution);
                    const auto contribution = _evaluator.contribution(
                        task, unclamped_contribution, invocation.parameters,
                        spectral_ptr);
                    sample.accumulate_radiance_at_state(
                        contribution, task.path_flags, task.path_visibility,
                        task.path_depth);
                    _evaluator.accumulate_passes(sample, task, contribution);
                };
            } else if (spectral_ptr != nullptr) {
                Bool active = true;
                if (sample.vt_wait_resident) {
                    $loop {
                        UInt vt_miss = vt_invalid_page;
                        $if(task.constant_light_shader == 0u) {
                            active = _evaluator.shade_light_nee(
                                task, invocation.parameters, &vt_miss,
                                spectral_ptr);
                        }
                        $else {
                            active = any(make_float4(
                                             task.unshadowed_contribution,
                                             spectral_ptr->unshadowed_contribution_w) !=
                                         0.0f);
                        };
                        $if(vt_miss != vt_invalid_page) {
                            sample.vt_wait_resident(vt_miss);
                            $continue;
                        };
                        $break;
                    };
                } else {
                    $if(task.constant_light_shader == 0u) {
                        active = _evaluator.shade_light_nee(
                            task, invocation.parameters, nullptr, spectral_ptr);
                    }
                    $else {
                        active = any(make_float4(
                                         task.unshadowed_contribution,
                                         spectral_ptr->unshadowed_contribution_w) !=
                                     0.0f);
                    };
                }
                _trace->record_transport(
                    bounce, {.light_shader = task.light_shader,
                             .unshadowed = task.unshadowed_contribution});
                $if(active) {
                    if (cut_policy == PathCoroutineCutPolicy::none &&
                        invocation.stage_profile != nullptr) {
                        invocation.stage_profile->enter(
                            MegakernelStageProfileState::stage_shadow_trace);
                    }
                    Bool visible = false;
                    _evaluator.trace_spectral(
                        task, *spectral_ptr, invocation.parameters, visible,
                        sample.vt_wait_resident);
                    $if(visible) {
                        const auto contribution = _evaluator.contribution(
                            task, task.shadow_throughput, invocation.parameters,
                            spectral_ptr);
                        _trace->record_contribution(bounce, contribution);
                        sample.accumulate_radiance_at_state(
                            contribution, task.path_flags,
                            task.path_visibility, task.path_depth);
                        _evaluator.accumulate_passes(sample, task, contribution);
                    };
                };
            } else {
                Bool active = true;
                if (sample.vt_wait_resident) {
                    $loop {
                        UInt vt_miss = vt_invalid_page;
                        $if(task.constant_light_shader == 0u) {
                            active = _evaluator.shade_light_nee(
                                task, invocation.parameters, &vt_miss);
                        };
                        $if(vt_miss != vt_invalid_page) {
                            sample.vt_wait_resident(vt_miss);
                            $continue;
                        };
                        $break;
                    };
                } else {
                    $if(task.constant_light_shader == 0u) {
                        active = _evaluator.shade_light_nee(
                            task, invocation.parameters);
                    };
                }
                _trace->record_transport(
                    bounce, {.light_shader = task.light_shader,
                             .unshadowed = task.unshadowed_contribution});
                $if(active) {
                    if (cut_policy == PathCoroutineCutPolicy::none &&
                        invocation.stage_profile != nullptr) {
                        invocation.stage_profile->enter(
                            MegakernelStageProfileState::stage_shadow_trace);
                    }
                    Var<ShadowTraceResultCall> shadow_result;
                    if (sample.vt_wait_resident) {
                        $loop {
                            shadow_result =
                                _evaluator.trace(task, invocation.parameters);
                            $if(shadow_result->vt_miss_page != vt_invalid_page) {
                                sample.vt_wait_resident(
                                    shadow_result->vt_miss_page);
                                $continue;
                            };
                            $break;
                        };
                    } else {
                        shadow_result =
                            _evaluator.trace(task, invocation.parameters);
                    }
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
                    $if(any(shadow_result->throughput != 0.0f)) {
                        const auto unclamped_contribution =
                            shadow_result->throughput;
                        _trace->record_contribution(
                            bounce, unclamped_contribution);
                        const auto contribution = _evaluator.contribution(
                            task, unclamped_contribution, invocation.parameters);
                        sample.accumulate_radiance_at_state(
                            contribution, task.path_flags,
                            task.path_visibility, task.path_depth);
                        _evaluator.accumulate_passes(sample, task, contribution);
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
