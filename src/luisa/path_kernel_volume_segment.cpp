#include "path_kernel_builder.h"
#include "path_kernel_heterogeneous_volume.h"
#include "path_kernel_volume_direct_light.h"
#include "path_tracer_cycles_svm_volume.h"

#include <psycles/luisa/cycles_noise.h>

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/homogeneous_volume_segment.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathVolumeSegmentStageImpl final
    : public PathVolumeSegmentStage {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::unique_ptr<
        HomogeneousVolumeSegmentComponent>
        _homogeneous;
    std::unique_ptr<
        PathHeterogeneousVolumeComponent>
        _heterogeneous;
    std::unique_ptr<
        VolumeDirectLightingComponent>
        _direct_lighting;

    [[nodiscard]] static bool
    _has_majorant_resources(
        const std::shared_ptr<const VolumeMajorantRuntime> &majorants) noexcept {
        return majorants &&
            majorants->node_count >
                0u &&
            majorants->root_count >
                0u &&
            majorants->range_count >
                0u &&
            majorants->world_range <
                majorants->range_count;
    }

  public:
    explicit PathVolumeSegmentStageImpl(
        const PathKernelConfig &config)
        : _scene{config.scene},
          _homogeneous{
              make_homogeneous_volume_segment_component()},
          _heterogeneous{
              _has_majorant_resources(config.volume_majorants)
                  ? make_path_heterogeneous_volume_component(_scene, config.volume_majorants)
                  : nullptr},
          _direct_lighting{
              config.next_event_estimation
                  ? make_volume_direct_lighting_component(
                        config)
                  : nullptr} {}

    VolumeSegmentEvent emit(
        ClosestPathEvent &event)
        const noexcept override {
        auto &bounce = event.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &parameters =
            invocation.parameters;
        const auto &sobol_table =
            invocation.sobol_table;
        auto &stack = *sample.volume.stack;
        auto &ray = sample.ray;
        auto &throughput = sample.throughput;
        auto &path_flags = sample.path_flags;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth =
            sample.diffuse_depth;
        auto &glossy_depth =
            sample.glossy_depth;
        auto &transparent_depth =
            sample.transparent_depth;
        auto &transmission_depth =
            sample.transmission_depth;
        auto &cycles_path_visibility =
            sample.cycles_path_visibility;
        auto &cycles_rng_offset =
            sample.cycles_rng_offset;
        auto &ray_events = sample.ray_events;
        auto &volume_bounce =
            sample.volume_bounce;
        auto &optical_depth =
            sample.optical_depth;
        auto &continuation_probability =
            sample.continuation_probability;
        auto &ray_source_object =
            sample.ray_source_object;
        auto &ray_source_primitive =
            sample.ray_source_primitive;
        auto &ray_dP = sample.ray_dP;
        auto &ray_dD = sample.ray_dD;
        auto &previous_bsdf_pdf =
            sample.previous_bsdf_pdf;
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
        auto &previous_light_tree_dt =
            sample.previous_light_tree_dt;
        auto &minimum_bsdf_pdf =
            sample.minimum_bsdf_pdf;
        auto &path_diffuse_weight =
            sample.path_diffuse_weight;
        auto &path_glossy_weight =
            sample.path_glossy_weight;

        $if(event.background) {
            stack.clean_for_background(
                _scene->volume_metadata
                    .has_world_volume);
        };
        const auto inside_volume =
            !stack.empty();
        path_flags |=
            select(
                0u,
                cycles_path_state::
                    flag_volume_primary_transmit,
                inside_volume &
                    (path_depth == 0u));
        const auto segment_start =
            ray->t_min();
        const auto segment_length =
            max(
                event.distance -
                    segment_start,
                0.0f);
        const auto segment_position =
            ray->origin() +
            ray->direction() *
                segment_start;
        const auto transport_terminate =
            (path_flags &
             cycles_path_state::
                 flag_terminate) != 0u;
        VolumeDirectLightProposal
            direct_proposal{
                .emitter_kind =
                    ~0u,
                .emitter_index = 0u,
                .requested_method =
                    volume_sample_none,
                .light_position =
                    make_float3(0.0f),
                .interval =
                    {.minimum = 0.0f,
                     .maximum =
                         segment_length},
                .valid = false};
        VolumeDirectLightSample
            direct_light{
            .direction = make_float3(0.0f),
            .position = make_float3(0.0f),
            .radiance = make_float3(0.0f),
            .pdf = 0.0f,
            .maximum_distance =
                ray_maximum,
            .light_object =
                surface_ray::invalid_primitive,
            .light_primitive =
                surface_ray::invalid_primitive,
            .light_instance =
                surface_ray::invalid_primitive,
            .light_accel_primitive =
                surface_ray::invalid_primitive,
            .emission_is_constant = false,
            .use_mis = false,
            .valid = false};
        if (sample.is_spectral()) {
            direct_light.spectral_radiance.emplace(make_float4(0.0f));
        }
        const auto outline_volume =
            _scene && static_cast<bool>(_scene->device) &&
            _scene->device.backend_name() == "simd" &&
            _scene->device.compute_warp_size() > 1u &&
            ([]() noexcept {
                const char *raw = std::getenv("PSYCLES_SIMD_OUTLINE_VOLUME");
                return raw == nullptr || std::string_view{raw} != "0";
            })();
        std::unique_ptr<
            VolumeDirectLightProvider>
            direct_provider;
        if (_direct_lighting) {
            $outline_when_with_name(outline_volume, "simd_volume_direct_propose") {
                const auto proposed =
                    _direct_lighting->propose(
                        event,
                        stack,
                        segment_position,
                        ray->direction(),
                        segment_length);
                direct_proposal.emitter_kind = proposed.emitter_kind;
                direct_proposal.emitter_index = proposed.emitter_index;
                direct_proposal.requested_method = proposed.requested_method;
                direct_proposal.light_position = proposed.light_position;
                direct_proposal.interval.minimum = proposed.interval.minimum;
                direct_proposal.interval.maximum = proposed.interval.maximum;
                direct_proposal.valid = proposed.valid;
            };
            direct_provider =
                _direct_lighting
                    ->make_light_provider(
                        event,
                        direct_proposal,
                        segment_position,
                        ray->direction(),
                        direct_light);
        }

        const cycles_svm::PathState volume_state{
            cycles_path_visibility, path_flags, path_depth, transparent_depth,
            diffuse_depth, glossy_depth, transmission_depth, sample.portal_depth};
        const PathCyclesSvmVolumeShader volume_shader{
            _scene, parameters, ray->origin(), ray->direction(), segment_start,
            0.5f, stack.entry(0u).object, volume_state,
            cycles_noise::hash_uint3(
                sample.rng_hash ^ 0x15b4f88du, cycles_rng_offset, sample.sample_index),
            false, false, sample.spectral_context()};
        const auto phase_random =
            cycles_sampler::sample_2d(
                sobol_table,
                parameters
                    .sobol_sequence_size,
                sample.sample_index,
                sample.rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        sampling::
                            tabulated_sobol::
                                volume_phase_dimension));
        const auto channel_random =
            cycles_sampler::sample_1d(
                sobol_table,
                parameters
                    .sobol_sequence_size,
                sample.sample_index,
                sample.rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        sampling::
                            tabulated_sobol::
                                volume_reservoir_dimension));
        const auto scatter_random =
            cycles_sampler::sample_1d(
                sobol_table,
                parameters
                    .sobol_sequence_size,
                sample.sample_index,
                sample.rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        sampling::
                            tabulated_sobol::
                                volume_scatter_distance_dimension));
        const auto heterogeneous =
            _heterogeneous
                ? _heterogeneous->stack_is_heterogeneous(stack)
                : def(false);
        const bool spectral = sample.is_spectral();
        Float3 result_throughput =
            spectral ? make_float3(1.0f) : throughput.rgb();
        Float3 result_emission = make_float3(0.0f);
        Float4 result_throughput_4 =
            spectral ? throughput.lanes() : make_float4(0.0f);
        Float4 result_emission_4 = make_float4(0.0f);
        Float result_distance = segment_length;
        Float result_optical_depth = 0.0f;
        VolumePhaseSetSample result_phase{
            .direction = ray->direction(),
            .pdf = 0.0f,
            .sampled_roughness = 1.0f,
            .selection_rescaled = phase_random.x,
            .closure_index = 0u,
            .closure_type = 0u,
            .valid = false};
        Bool result_scattered = false;
        Bool result_phase_failed = false;
        Float3 direct_transport_throughput = make_float3(1.0f);
        Float4 direct_transport_throughput_4 = make_float4(1.0f);
        Float direct_transport_distance = 0.0f;
        VolumePhaseSetEvaluation direct_phase{
            .value = 0.0f,
            .pdf = 0.0f,
            .sample_weight = 0.0f,
            .valid = false};
        Bool direct_scattered = false;

        if (spectral) {
            const auto sec_term = sample.spectral_secondary_terminated();
            const auto emit_homogeneous_spec = [&]() noexcept {
                $outline_when_with_name(outline_volume, "simd_volume_homogeneous_spec") {
                    const auto result = _homogeneous->emit_spectral(
                        volume_shader,
                        stack,
                        segment_position,
                        -ray->direction(),
                        segment_length,
                        throughput.lanes(),
                        scatter_random,
                        channel_random,
                        phase_random,
                        transport_terminate,
                        sec_term,
                        {.requested_method = direct_proposal.requested_method,
                         .light_position = direct_proposal.light_position,
                         .interval = direct_proposal.interval,
                         .enabled = inside_volume & direct_proposal.valid},
                        direct_provider.get());
                    if (_direct_lighting) {
                        direct_transport_throughput_4 = result.direct_transport.throughput;
                        direct_transport_distance = result.direct_transport.distance;
                        direct_phase.value = result.direct_phase.value;
                        direct_phase.pdf = result.direct_phase.pdf;
                        direct_phase.sample_weight = result.direct_phase.sample_weight;
                        direct_phase.valid = result.direct_phase.valid;
                        direct_scattered = result.direct_transport.scattered;
                    }
                    result_throughput_4 = result.transport.throughput;
                    result_emission_4 = result.transport.emission;
                    result_distance = result.transport.distance;
                    const auto st4 = result.coefficients.spectral_sigma_t.value_or(
                        make_float4(result.coefficients.sigma_t, 0.0f));
                    result_optical_depth =
                        cycles_spectrum::max_active(st4, sec_term) * segment_length;
                    result_phase.direction = result.phase.direction;
                    result_phase.pdf = result.phase.pdf;
                    result_phase.sampled_roughness = result.phase.sampled_roughness;
                    result_phase.selection_rescaled = result.phase.selection_rescaled;
                    result_phase.closure_index = result.phase.closure_index;
                    result_phase.closure_type = result.phase.closure_type;
                    result_phase.valid = result.phase.valid;
                    result_scattered = result.scattered;
                    result_phase_failed = result.phase_failed;
                };
            };
            if (_heterogeneous) {
                $if(heterogeneous) {
                    $outline_when_with_name(outline_volume, "simd_volume_heterogeneous_spec") {
                        const auto result = _heterogeneous->emit_spectral(
                            {.stack = stack,
                             .shader = volume_shader,
                             .sobol_table = sobol_table,
                             .sobol_sequence_size = parameters.sobol_sequence_size,
                             .sample_index = sample.sample_index,
                             .rng_hash = sample.rng_hash,
                             .path_rng_offset = cycles_rng_offset,
                             .ray_origin = ray->origin(),
                             .ray_direction = ray->direction(),
                             .ray_minimum = segment_start,
                             .ray_maximum = event.distance,
                             .throughput = throughput.lanes(),
                             .reservoir_random = channel_random,
                             .phase_random = phase_random,
                             .direct =
                                 {.requested_method = direct_proposal.requested_method,
                                  .light_position = direct_proposal.light_position,
                                  .interval = direct_proposal.interval,
                                  .enabled = inside_volume & direct_proposal.valid},
                             .direct_light = direct_provider.get(),
                             .terminate = transport_terminate,
                             .secondary_terminated = sec_term});
                        if (_direct_lighting) {
                            direct_transport_throughput_4 =
                                result.direct_transport.throughput;
                            direct_transport_distance =
                                result.direct_transport.distance - segment_start;
                            direct_phase.value = result.direct_phase.value;
                            direct_phase.pdf = result.direct_phase.pdf;
                            direct_phase.sample_weight = result.direct_phase.sample_weight;
                            direct_phase.valid = result.direct_phase.valid;
                            direct_scattered = result.direct_transport.scattered;
                        }
                        result_throughput_4 = result.transport.throughput;
                        result_emission_4 = result.transport.emission;
                        result_distance = result.transport.distance - segment_start;
                        result_optical_depth = result.transport.optical_depth;
                        result_phase.direction = result.phase.direction;
                        result_phase.pdf = result.phase.pdf;
                        result_phase.sampled_roughness = result.phase.sampled_roughness;
                        result_phase.selection_rescaled = result.phase.selection_rescaled;
                        result_phase.closure_index = result.phase.closure_index;
                        result_phase.closure_type = result.phase.closure_type;
                        result_phase.valid = result.phase.valid;
                        result_scattered = result.scattered;
                        result_phase_failed = result.phase_failed;
                    };
                }
                $else {
                    emit_homogeneous_spec();
                };
            } else {
                emit_homogeneous_spec();
            }
            if (_direct_lighting) {
                $outline_when_with_name(outline_volume, "simd_volume_direct_accumulate_spec") {
                    _direct_lighting->accumulate(
                        event,
                        direct_light,
                        {.throughput = direct_transport_throughput_4.xyz(),
                         .distance = direct_transport_distance,
                         .phase = direct_phase,
                         .scattered = direct_scattered,
                         .spectral_throughput = direct_transport_throughput_4},
                        stack,
                        segment_position,
                        inside_volume);
                };
            }

            const Float4 spec_emission =
                select(make_float4(0.0f), result_emission_4, inside_volume);
            const Float3 raw_emission = sample.spectrum_to_rgb(
                cycles_spectrum::Spectrum::from_lanes(spec_emission));
            const auto emission = invocation.clamp_emission_contribution(
                raw_emission, path_depth);
            sample.accumulate_radiance(emission);
            sample.accumulate_emission_or_background(
                emission, LightPassBuffer::emission);
            optical_depth +=
                select(0.0f, result_optical_depth,
                       inside_volume & (path_depth == 0u));
            throughput = cycles_spectrum::Spectrum::from_lanes(
                select(throughput.lanes(), result_throughput_4, inside_volume));
        } else {
            const Float3 volume_input_throughput = throughput.rgb();
            const auto emit_homogeneous =
                [&]() noexcept {
                    $outline_when_with_name(outline_volume, "simd_volume_homogeneous") {
                        const auto result =
                            _homogeneous->emit(
                                volume_shader,
                                stack,
                                segment_position,
                                -ray->direction(),
                                segment_length,
                                volume_input_throughput,
                                scatter_random,
                                channel_random,
                                phase_random,
                                transport_terminate,
                                {.scattered_radiance =
                                     invocation
                                         .volume_guiding_scattered_radiance,
                                 .transmitted_radiance =
                                     invocation
                                         .volume_guiding_transmitted_radiance,
                                 .majorant_optical_depth =
                                     invocation
                                         .volume_guiding_majorant_optical_depth(),
                                 .enabled =
                                     inside_volume &
                                     (path_depth == 0u)},
                                {.requested_method =
                                     direct_proposal
                                         .requested_method,
                                 .light_position =
                                     direct_proposal
                                         .light_position,
                                 .interval =
                                     direct_proposal
                                         .interval,
                                 .enabled =
                                     inside_volume &
                                     direct_proposal.valid},
                                direct_provider.get());
                        if (_direct_lighting) {
                            direct_transport_throughput =
                                result.direct_transport.throughput;
                            direct_transport_distance =
                                result.direct_transport.distance;
                            direct_phase.value = result.direct_phase.value;
                            direct_phase.pdf = result.direct_phase.pdf;
                            direct_phase.sample_weight =
                                result.direct_phase.sample_weight;
                            direct_phase.valid = result.direct_phase.valid;
                            direct_scattered =
                                result.direct_transport.scattered;
                        }
                        result_throughput =
                            result.transport
                                .throughput;
                        result_emission =
                            result.transport.emission;
                        result_distance =
                            result.transport.distance;
                        result_optical_depth =
                            max(
                                result.coefficients
                                    .sigma_t.x,
                                max(
                                    result.coefficients
                                        .sigma_t.y,
                                    result.coefficients
                                        .sigma_t.z)) *
                            segment_length;
                        result_phase.direction = result.phase.direction;
                        result_phase.pdf = result.phase.pdf;
                        result_phase.sampled_roughness =
                            result.phase.sampled_roughness;
                        result_phase.selection_rescaled =
                            result.phase.selection_rescaled;
                        result_phase.closure_index = result.phase.closure_index;
                        result_phase.closure_type = result.phase.closure_type;
                        result_phase.valid = result.phase.valid;
                        result_scattered =
                            result.scattered;
                        result_phase_failed =
                            result.phase_failed;
                    };
                };
            if (_heterogeneous) {
                $if(heterogeneous) {
                    $outline_when_with_name(outline_volume, "simd_volume_heterogeneous") {
                        const auto result =
                            _heterogeneous->emit(
                                {.stack = stack,
                                 .shader = volume_shader,
                                 .sobol_table =
                                     sobol_table,
                                 .sobol_sequence_size =
                                     parameters
                                         .sobol_sequence_size,
                                 .sample_index =
                                     sample.sample_index,
                                 .rng_hash =
                                     sample.rng_hash,
                                 .path_rng_offset =
                                     cycles_rng_offset,
                                 .ray_origin =
                                     ray->origin(),
                                 .ray_direction =
                                     ray->direction(),
                                 .ray_minimum =
                                     segment_start,
                                 .ray_maximum =
                                     event.distance,
                                 .throughput =
                                     volume_input_throughput,
                                 .reservoir_random =
                                     channel_random,
                                 .phase_random =
                                     phase_random,
                                 .guiding =
                                     {.scattered_radiance =
                                          invocation
                                              .volume_guiding_scattered_radiance,
                                      .transmitted_radiance =
                                          invocation
                                              .volume_guiding_transmitted_radiance,
                                      .majorant_optical_depth =
                                          invocation
                                              .volume_guiding_majorant_optical_depth(),
                                      .enabled =
                                          inside_volume &
                                          (path_depth ==
                                           0u)},
                                 .direct =
                                     {.requested_method =
                                          direct_proposal
                                              .requested_method,
                                      .light_position =
                                          direct_proposal
                                              .light_position,
                                      .interval =
                                          direct_proposal
                                              .interval,
                                      .enabled =
                                          inside_volume &
                                          direct_proposal
                                              .valid},
                                 .direct_light =
                                     direct_provider.get(),
                                 .terminate =
                                     transport_terminate});
                        if (_direct_lighting) {
                            direct_transport_throughput =
                                result.direct_transport.throughput;
                            direct_transport_distance =
                                result.direct_transport.distance - segment_start;
                            direct_phase.value = result.direct_phase.value;
                            direct_phase.pdf = result.direct_phase.pdf;
                            direct_phase.sample_weight =
                                result.direct_phase.sample_weight;
                            direct_phase.valid = result.direct_phase.valid;
                            direct_scattered =
                                result.direct_transport.scattered;
                        }
                        result_throughput =
                            result.transport
                                .throughput;
                        result_emission =
                            result.transport.emission;
                        result_distance =
                            result.transport.distance -
                            segment_start;
                        result_optical_depth =
                            result.transport
                                .optical_depth;
                        result_phase.direction = result.phase.direction;
                        result_phase.pdf = result.phase.pdf;
                        result_phase.sampled_roughness =
                            result.phase.sampled_roughness;
                        result_phase.selection_rescaled =
                            result.phase.selection_rescaled;
                        result_phase.closure_index = result.phase.closure_index;
                        result_phase.closure_type = result.phase.closure_type;
                        result_phase.valid = result.phase.valid;
                        result_scattered =
                            result.scattered;
                        result_phase_failed =
                            result.phase_failed;
                    };
                }
                $else {
                    emit_homogeneous();
                };
            } else {
                emit_homogeneous();
            }
            if (_direct_lighting) {
                $outline_when_with_name(outline_volume, "simd_volume_direct_accumulate") {
                    _direct_lighting->accumulate(
                        event,
                        direct_light,
                        {.throughput = direct_transport_throughput,
                         .distance = direct_transport_distance,
                         .phase = direct_phase,
                         .scattered = direct_scattered},
                        stack,
                        segment_position,
                        inside_volume);
                };
            }

            Float3 raw_emission =
                select(make_float3(0.0f), result_emission, inside_volume);
            const auto emission =
                invocation.clamp_emission_contribution(raw_emission, path_depth);
            sample.accumulate_radiance(
                emission);
            sample.accumulate_emission_or_background(emission, LightPassBuffer::emission);
            optical_depth +=
                select(
                    0.0f,
                    result_optical_depth,
                    inside_volume &
                        (path_depth == 0u));
            throughput = cycles_spectrum::Spectrum::from_rgb_raw(
                select(throughput.rgb(), result_throughput, inside_volume));
        }

        const auto terminates_in_volume =
            inside_volume &
            cycles_path_state::volume_segment_terminates(
                path_flags,
                result_phase_failed);
        const auto scattered =
            inside_volume &
            result_scattered &
            !terminates_in_volume;
        $if(scattered) {
            const auto collision_position =
                segment_position +
                ray->direction() *
                    result_distance;
            $if(continuation_probability !=
                1.0f) {
                throughput /=
                    continuation_probability;
            };
            ray = make_ray(
                collision_position,
                normalize(
                    result_phase.direction),
                0.0f,
                ray_maximum);
            ray_dP = 0.0f;
            ray_dD = max(
                ray_dD,
                result_phase
                    .sampled_roughness);
            ray_source_object =
                surface_ray::invalid_primitive;
            ray_source_primitive =
                surface_ray::invalid_primitive;
            previous_bsdf_pdf =
                result_phase.pdf;
            previous_mis_origin_normal =
                collision_position -
                segment_position;
            previous_light_tree_dt = segment_length;
            if (invocation.config.has_filter_glossy ||
                invocation.config.path_trace_enabled) {
                minimum_bsdf_pdf =
                    min(
                        minimum_bsdf_pdf,
                        result_phase.pdf);
            }
            ray_events = 0u;
            path_diffuse_weight =
                select(
                    path_diffuse_weight,
                    make_float3(1.0f),
                    path_depth == 0u);
            path_glossy_weight =
                select(
                    path_glossy_weight,
                    make_float3(0.0f),
                    path_depth == 0u);

            const auto next =
                cycles_path_state::
                    next_volume(
                        {.flag =
                             path_flags,
                         .visibility =
                             cycles_path_visibility,
                         .bounce =
                             path_depth,
                         .diffuse_bounce =
                             diffuse_depth,
                         .glossy_bounce =
                             glossy_depth,
                         .transmission_bounce =
                             transmission_depth,
                         .transparent_bounce =
                             transparent_depth,
                         .rng_offset =
                             cycles_rng_offset},
                        volume_bounce,
                        parameters
                            .max_volume_bounces,
                        parameters
                            .max_bounces);
            path_flags =
                next.state.flag;
            cycles_path_visibility =
                next.state.visibility;
            path_depth =
                next.state.bounce;
            diffuse_depth =
                next.state.diffuse_bounce;
            glossy_depth =
                next.state.glossy_bounce;
            transmission_depth =
                next.state
                    .transmission_bounce;
            transparent_depth =
                next.state
                    .transparent_bounce;
            cycles_rng_offset =
                next.state.rng_offset;
            volume_bounce =
                next.volume_bounce;
        };
        return {
            .scattered =
                scattered,
            .terminated =
                terminates_in_volume};
    }
};

}// namespace

std::unique_ptr<PathVolumeSegmentStage>
make_path_volume_segment_stage(
    const PathKernelConfig &config) {
    return std::make_unique<
        PathVolumeSegmentStageImpl>(
        config);
}

}// namespace psycles::luisa_backend::detail
