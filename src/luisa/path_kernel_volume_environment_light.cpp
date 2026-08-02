#include "path_kernel_volume_environment_light.h"

#include "path_kernel_environment_light.h"

#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/light_distribution.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr auto environment_emitter_kind =
    static_cast<std::uint32_t>(
        sampling::
            LightDistributionEmitterKind::
                environment);

class EnvironmentVolumeLightProvider final
    : public VolumeDirectLightProvider {

  private:
    ClosestPathEvent &_event;
    VolumeDirectLightProposal
        _proposal;
    Float3 _segment_position;
    Float3 _segment_direction;
    VolumeDirectLightSample
        &_result;
    std::shared_ptr<
        const EnvironmentLightComponent>
        _environment_light;
    mutable Bool _sample_valid{false};

  public:
    EnvironmentVolumeLightProvider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal
            &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample
            &result,
        std::shared_ptr<
            const EnvironmentLightComponent>
            environment_light) noexcept
        : _event{event},
          _proposal{proposal},
          _segment_position{
              std::move(
                  segment_position)},
          _segment_direction{
              std::move(
                  segment_direction)},
          _result{result},
          _environment_light{
              std::move(
                  environment_light)} {}

    VolumeDirectDirectionSample sample_direction(
        Float distance)
        const noexcept override {
        const auto active =
            _proposal.valid &
            (_proposal.emitter_kind ==
             environment_emitter_kind);
        $if(active) {
            const auto position =
                _segment_position +
                _segment_direction *
                    distance;
            const auto light =
                _environment_light
                    ->from_position(
                        _event.bounce
                            .sample
                            .invocation
                            .config.scene,
                        position,
                        _event.bounce
                            .light_sample
                            .xy(),
                        _event.bounce
                            .selected_light
                            .selection_pdf);
            _sample_valid = light.valid;
            $if(light.valid) {
                _result.direction =
                    light.direction;
                _result.radiance =
                    make_float3(1.0f);
                _result.pdf = light.pdf;
                _result.maximum_distance =
                    ray_maximum;
                _result.light_instance =
                    surface_ray::
                        invalid_primitive;
                _result.light_primitive =
                    surface_ray::
                        invalid_primitive;
                _result.use_mis = true;
                _result.valid = true;
            };
        };
        return {
            .direction =
                _result.direction,
            .valid =
                _result.valid};
    }

    void evaluate_constant_emission()
        const noexcept override {
        if (_event.bounce.sample
                .invocation.config.scene
                ->environment_emission_is_constant) {
            $if(_sample_valid) {
                _result.radiance *=
                    _environment_light
                        ->evaluate_constant_emission(
                            _event.bounce
                                .sample);
            };
        }
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero)
        const noexcept override {
        if (!_event.bounce.sample
                 .invocation.config.scene
                 ->environment_emission_is_constant) {
            $if(_sample_valid &
                receiving_nonzero) {
                _result.radiance *=
                    _environment_light
                        ->evaluate_emission(
                            _event.bounce
                                .sample,
                            _result.direction,
                            cycles_path_state::
                                light_emission_shader_state(
                                    _event.bounce
                                        .sample.path_depth,
                                    _event.bounce
                                        .sample.diffuse_depth,
                                    _event.bounce
                                        .sample.glossy_depth,
                                    _event.bounce
                                        .sample.transparent_depth,
                                    _event.bounce
                                        .sample.transmission_depth));
            };
        }
    }
};

class PathVolumeEnvironmentLightComponent final
    : public VolumeEnvironmentLightComponent {

  private:
    PathKernelConfig _config;
    std::shared_ptr<
        const EnvironmentLightComponent>
        _environment_light{
            make_environment_light_component()};

  public:
    explicit PathVolumeEnvironmentLightComponent(
        const PathKernelConfig &config)
        : _config{config} {}

    void propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
        Float segment_length,
        VolumeDirectLightProposal
            &result)
        const noexcept override {
        static_cast<void>(path_stack);
        const auto selected_environment =
            event.bounce
                .selected_light.kind ==
            environment_emitter_kind;
        const auto visible_to_volume =
            (_config.scene
                 ->world_visibility_mask &
             volume_scatter_visibility) !=
            0u;
        $if(selected_environment &
            visible_to_volume &
            (segment_length > 0.0f)) {
            result.emitter_kind =
                environment_emitter_kind;
            result.emitter_index =
                event.bounce
                    .selected_light.index;
            result.requested_method =
                volume_sample_distance;
            result.light_position =
                make_float3(0.0f);
            result.interval = {
                .minimum = 0.0f,
                .maximum =
                    segment_length};
            result.valid = true;
        };
    }

    std::unique_ptr<
        VolumeDirectLightProvider>
    make_light_provider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal
            &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample
            &result)
        const override {
        return std::make_unique<
            EnvironmentVolumeLightProvider>(
                event,
                proposal,
                std::move(
                    segment_position),
                std::move(
                    segment_direction),
                result,
                _environment_light);
    }
};

}// namespace

std::unique_ptr<
    VolumeEnvironmentLightComponent>
make_volume_environment_light_component(
    const PathKernelConfig &config) {
    return std::make_unique<
        PathVolumeEnvironmentLightComponent>(
        config);
}

}// namespace psycles::luisa_backend::detail
