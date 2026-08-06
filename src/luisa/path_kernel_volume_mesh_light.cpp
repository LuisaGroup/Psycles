#include "path_kernel_volume_mesh_light.h"

#include "path_kernel_emissive_triangle.h"

#include <psycles/luisa/surface_ray.h>
#include <psycles/luisa/volume_light_interval.h>

#include <optional>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr auto mesh_emitter_kind =
    static_cast<std::uint32_t>(
        sampling::
            LightDistributionEmitterKind::
                emissive_triangle);

class MeshVolumeLightProvider final
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
        const EmissiveTriangleComponent>
        _emissive_triangle;
    mutable std::optional<
        EmissiveTriangleLightProposal>
        _sample;
    mutable Bool _sample_valid{false};
    mutable Bool _emission_is_constant{false};

  public:
    MeshVolumeLightProvider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal
            &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample
            &result,
        std::shared_ptr<
            const EmissiveTriangleComponent>
            emissive_triangle) noexcept
        : _event{event},
          _proposal{proposal},
          _segment_position{
              std::move(
                  segment_position)},
          _segment_direction{
              std::move(
                  segment_direction)},
          _result{result},
          _emissive_triangle{
              std::move(
                  emissive_triangle)} {}

    VolumeDirectDirectionSample sample_direction(
        Float distance)
        const noexcept override {
        const auto active =
            _proposal.valid &
            (_proposal.emitter_kind ==
             mesh_emitter_kind);
        $if(active) {
            const auto position =
                _segment_position +
                _segment_direction *
                    distance;
            const auto light =
                _emissive_triangle
                    ->from_position(
                        _event.bounce
                            .sample
                            .invocation
                            .config.scene,
                        _proposal
                            .emitter_index,
                        position,
                        _event.bounce
                            .light_sample
                            .xy(),
                        _event.bounce
                            .selected_light
                            .selection_pdf);
            const auto visible =
                (light.geometry
                     .emitter
                     .visibility_mask &
                 volume_scatter_visibility) !=
                0u;
            const auto valid =
                light.valid &
                visible;
            _sample = light;
            _sample_valid = valid;
            _emission_is_constant =
                light.geometry
                    .emitter
                    .emission_is_constant !=
                0u;
            $if(valid) {
                _result.direction =
                    light.light.direction;
                _result.radiance =
                    make_float3(1.0f);
                _result.pdf = light.pdf;
                _result.maximum_distance =
                    light.light.distance;
                _result.light_object =
                    light.geometry
                        .emitter
                        .cycles_object_index;
                _result.light_primitive =
                    light.geometry
                        .emitter
                        .cycles_primitive_index;
                _result.light_instance =
                    light.geometry
                        .emitter
                        .instance_index;
                _result.light_accel_primitive =
                    light.geometry
                        .emitter
                        .primitive_index;
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
        if (_sample) {
            $if(_sample_valid &
                _emission_is_constant) {
                _result.radiance *=
                    _emissive_triangle
                        ->evaluate_constant_emission(
                            _event.bounce
                                .sample,
                            *_sample);
            };
        }
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero)
        const noexcept override {
        if (_sample) {
            $if(_sample_valid &
                !_emission_is_constant &
                receiving_nonzero) {
                _result.radiance *=
                    _emissive_triangle
                        ->evaluate_emission(
                            _event.bounce
                                .sample,
                            *_sample);
            };
        }
    }
};

class PathVolumeMeshLightComponent final
    : public VolumeMeshLightComponent {

  private:
    PathKernelConfig _config;
    std::shared_ptr<
        const EmissiveTriangleComponent>
        _emissive_triangle{
            make_emissive_triangle_component()};
    VolumeLightInterval
        _light_interval;

  public:
    explicit PathVolumeMeshLightComponent(
        const PathKernelConfig &config)
        : _config{config} {}

    void propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float segment_length,
        VolumeDirectLightProposal
            &result)
        const noexcept override {
        const auto &selected =
            event.bounce
                .selected_light;
        const auto selected_mesh =
            selected.kind ==
            mesh_emitter_kind;
        $if(selected_mesh &
            (segment_length > 0.0f)) {
            const auto proposal =
                _emissive_triangle
                    ->from_segment(
                        _config.scene,
                        selected.index,
                        segment_position,
                        event.bounce
                            .light_sample
                            .xy());
            const auto visible =
                (proposal.geometry
                     .emitter
                     .visibility_mask &
                 volume_scatter_visibility) !=
                0u;
            const auto interval =
                _light_interval
                    .triangle(
                        {.ray_origin =
                             segment_position,
                         .ray_direction =
                             segment_direction,
                         .interval =
                             {.minimum =
                                  0.0f,
                              .maximum =
                                  segment_length},
                         .plane_point =
                             proposal.light
                                 .position,
                         .normal =
                             proposal.geometry
                                 .geometric_normal,
                         .sample_front =
                             proposal.geometry
                                 .sample_front,
                         .sample_back =
                             proposal.geometry
                                 .sample_back});
            const auto valid =
                proposal.valid &
                visible &
                interval.valid;
            $if(valid) {
                result.emitter_kind =
                    mesh_emitter_kind;
                result.emitter_index =
                    selected.index;
                result.requested_method =
                    path_stack
                        .sample_method();
                result.light_position =
                    proposal.light
                        .position;
                result.interval =
                    interval.interval;
                result.valid = true;
            };
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
            MeshVolumeLightProvider>(
                event,
                proposal,
                std::move(
                    segment_position),
                std::move(
                    segment_direction),
                result,
                _emissive_triangle);
    }
};

}// namespace

std::unique_ptr<
    VolumeMeshLightComponent>
make_volume_mesh_light_component(
    const PathKernelConfig &config) {
    return std::make_unique<
        PathVolumeMeshLightComponent>(
        config);
}

}// namespace psycles::luisa_backend::detail
