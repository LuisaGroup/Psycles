#include "path_kernel_builder.h"
#include "path_kernel_heterogeneous_volume.h"
#include "path_kernel_volume_direct_light.h"
#include "path_kernel_volume_point.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/homogeneous_volume_segment.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathVolumeSegmentStageImpl final
    : public PathVolumeSegmentStage {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
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
        const LuisaSceneData &scene) noexcept {
        return
            scene.volume_majorant_node_count >
                0u &&
            scene.volume_majorant_root_count >
                0u &&
            scene.volume_majorant_range_count >
                0u &&
            scene.volume_majorant_world_range <
                scene.volume_majorant_range_count;
    }

  public:
    explicit PathVolumeSegmentStageImpl(
        const PathKernelConfig &config)
        : _scene{config.scene},
          _points{
              make_scene_volume_stack_entry_point_provider(
                  _scene)},
          _homogeneous{
              make_homogeneous_volume_segment_component(
                  _scene->surfaces,
                  _points,
                  _scene->volume_metadata
                      .closure_allocation_budget)},
          _heterogeneous{
              _has_majorant_resources(*_scene)
                  ? make_path_heterogeneous_volume_component(
                        _scene,
                        _points,
                        _scene->volume_metadata
                            .closure_allocation_budget)
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
        auto &sample_emission =
            sample.sample_emission;
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
        auto &ray_visibility =
            sample.ray_visibility;
        auto &ray_events = sample.ray_events;
        auto &volume_bounce =
            sample.volume_bounce;
        auto &optical_depth =
            sample.optical_depth;
        auto &continuation_probability =
            sample.continuation_probability;
        auto &continuation_decided =
            sample.continuation_decided_in_volume;
        auto &terminate_on_next_surface =
            sample.terminate_on_next_surface;
        auto &terminate_after_transparent =
            sample.terminate_after_transparent;
        auto &ray_source_instance =
            sample.ray_source_instance;
        auto &ray_source_primitive =
            sample.ray_source_primitive;
        auto &ray_dP = sample.ray_dP;
        auto &ray_dD = sample.ray_dD;
        auto &previous_bsdf_pdf =
            sample.previous_bsdf_pdf;
        auto &previous_delta =
            sample.previous_delta;
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
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
        continuation_decided =
            inside_volume;

        continuation_probability =
            select(
                continuation_probability,
                cycles_path_state::
                    continuation_probability(
                        path_flags,
                        path_depth,
                        transparent_depth,
                        parameters.min_bounces,
                        parameters
                            .transparent_min_bounces,
                        throughput),
                inside_volume);
        const auto roulette_failed =
            inside_volume &
            (continuation_probability != 1.0f) &
            ((continuation_probability <= 0.0f) |
             (bounce.terminate_sample >=
              continuation_probability));
        path_flags |= select(
            0u,
            cycles_path_state::
                flag_terminate_on_next_surface,
            roulette_failed &
                event.surface_may_emit);
        path_flags |= select(
            0u,
            cycles_path_state::
                flag_terminate_in_next_volume,
            roulette_failed &
                !event.surface_may_emit);
        terminate_on_next_surface =
            (path_flags &
             cycles_path_state::
                 flag_terminate_on_next_surface) !=
            0u;

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
            .radiance = make_float3(0.0f),
            .pdf = 0.0f,
            .maximum_distance =
                ray_maximum,
            .light_instance =
                surface_ray::invalid_primitive,
            .light_primitive =
                surface_ray::invalid_primitive,
            .use_mis = false,
            .valid = false};
        std::unique_ptr<
            VolumeDirectDirectionProvider>
            direct_direction;
        if (_direct_lighting) {
            direct_proposal =
                _direct_lighting->propose(
                    event,
                    stack,
                    segment_position,
                    ray->direction(),
                    segment_length);
            direct_direction =
                _direct_lighting
                    ->make_direction_provider(
                        event,
                        direct_proposal,
                        segment_position,
                        ray->direction(),
                        direct_light);
        }

        BufferShaderServices services{
            _scene->parameter_buffer,
            _scene->cycles_bsdf_table_buffer,
            _scene->texture_heap,
            _scene->heap,
            _scene->attribute_binding_slot,
            _scene->attribute_range_slot,
            _scene->nishita_texture_bindings,
            _scene->shader_color_space};
        const VolumeShadingState state{
            .position = segment_position,
            .incoming =
                -ray->direction(),
            .ray_visibility =
                ray_visibility,
            .ray_events = ray_events,
            .ray_depth = path_depth,
            .diffuse_depth =
                diffuse_depth,
            .glossy_depth =
                glossy_depth,
            .transparent_depth =
                transparent_depth,
            .transmission_depth =
                transmission_depth,
            .ray_length = 0.0f,
            .time = 0.0f};
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
                ? _heterogeneous
                      ->stack_is_heterogeneous(
                          stack)
                : def(false);
        Float3 result_throughput =
            throughput;
        Float3 result_emission =
            make_float3(0.0f);
        Float result_distance =
            segment_length;
        Float result_optical_depth = 0.0f;
        VolumePhaseSetSample result_phase{
            .direction =
                ray->direction(),
            .pdf = 0.0f,
            .sampled_roughness = 1.0f,
            .selection_rescaled =
                phase_random.x,
            .closure_index = 0u,
            .closure_type = 0u,
            .valid = false};
        Bool result_scattered = false;
        Bool result_phase_failed = false;

        const auto emit_homogeneous =
            [&]() noexcept {
                const auto result =
                    _homogeneous->emit(
                        stack,
                        services,
                        state,
                        segment_length,
                        throughput,
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
                        direct_direction.get());
                if (_direct_lighting) {
                    _direct_lighting->accumulate(
                        event,
                        direct_light,
                        {.throughput =
                             result
                                 .direct_transport
                                 .throughput,
                         .distance =
                             result
                                 .direct_transport
                                 .distance,
                         .phase =
                             result.direct_phase,
                         .scattered =
                             result
                                 .direct_transport
                                 .scattered},
                        stack,
                        segment_position,
                        inside_volume);
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
                result_phase = result.phase;
                result_scattered =
                    result.scattered;
                result_phase_failed =
                    result.phase_failed;
            };
        if (_heterogeneous) {
            $if(heterogeneous) {
                const auto result =
                    _heterogeneous->emit(
                        {.stack = stack,
                         .services = services,
                         .state = state,
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
                             throughput,
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
                         .direct_direction =
                             direct_direction.get(),
                         .terminate =
                             transport_terminate});
                if (_direct_lighting) {
                    _direct_lighting->accumulate(
                        event,
                        direct_light,
                        {.throughput =
                             result
                                 .direct_transport
                                 .throughput,
                         .distance =
                             result
                                 .direct_transport
                                 .distance -
                             segment_start,
                         .phase =
                             result.direct_phase,
                         .scattered =
                             result
                                 .direct_transport
                                 .scattered},
                        stack,
                        segment_position,
                        inside_volume);
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
                result_phase = result.phase;
                result_scattered =
                    result.scattered;
                result_phase_failed =
                    result.phase_failed;
            }
            $else {
                emit_homogeneous();
            };
        } else {
            emit_homogeneous();
        }

        const auto emission =
            invocation
                .clamp_emission_contribution(
                select(
                    make_float3(0.0f),
                    result_emission,
                    inside_volume),
                path_depth);
        sample.accumulate_radiance(
            emission);
        const auto directly_visible =
            (path_flags &
             cycles_path_state::
                 flag_any_pass) == 0u;
        sample_emission +=
            select(
                make_float3(0.0f),
                emission,
                directly_visible);
        sample.accumulate_scattered_light(
            select(
                emission,
                make_float3(0.0f),
                directly_visible));
        optical_depth +=
            select(
                0.0f,
                result_optical_depth,
                inside_volume &
                    (path_depth == 0u));
        throughput =
            select(
                throughput,
                result_throughput,
                inside_volume);

        const auto terminates_in_volume =
            inside_volume &
            (((path_flags &
               cycles_path_state::
                   flag_terminate_in_next_volume) !=
              0u) |
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
            ray_source_instance =
                surface_ray::invalid_primitive;
            ray_source_primitive =
                surface_ray::invalid_primitive;
            previous_bsdf_pdf =
                result_phase.pdf;
            previous_delta = false;
            previous_mis_origin_normal =
                collision_position -
                segment_position;
            minimum_bsdf_pdf =
                min(
                    minimum_bsdf_pdf,
                    result_phase.pdf);
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
            ray_visibility =
                cycles_path_state::
                    contract_visibility(
                        cycles_path_visibility);
            terminate_on_next_surface =
                (path_flags &
                 cycles_path_state::
                     flag_terminate_on_next_surface) !=
                0u;
            terminate_after_transparent =
                (path_flags &
                 cycles_path_state::
                     flag_terminate_after_transparent) !=
                0u;
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
